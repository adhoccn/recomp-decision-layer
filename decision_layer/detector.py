"""Top-level orchestration for the GCRR verifier.

Product note: the published pipeline re-encodes the reference with ffmpeg to
manufacture positive pairs. A deployed verifier does NOT need that step --
the candidate video under investigation already exists. So the product path is
reference + candidate -> L0 gate -> features -> scores -> verdict, with no
transcoding.
"""

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .constants import (ALGO_VERSION, DEFAULT_CLIP_MODE, DEFAULT_WINDOW_CAP,
                        N_FRAMES, THRESHOLD, UNIFIED_SAMPLE_INTERVAL)
from .features import FeatureExtractor
from .gate import PreconditionGate
from .media import frame_count, read_frames_at_indices
from .scoring import (compare_features, compute_regional_score, fuse,
                      regional_maps, sample_indices)

VERDICT_POSITIVE = 'RECOMPRESSED'
VERDICT_NEGATIVE = 'NOT_RECOMPRESSED'
VERDICT_INAPPLICABLE = 'INAPPLICABLE'

STRENGTH_NORMAL = 'NORMAL'
STRENGTH_WEAK = 'WEAK'
STRENGTH_NONE = 'NONE'


class GCRRVerifier:
    """Non-blind recompression verifier.

    Answers: is `candidate` a recompressed derivative of `reference`?
    It does NOT answer: has this video been recompressed by someone, somewhere?

    One window for both channels
    ----------------------------
    The published pipeline scored its two channels on candidates of different
    lengths (15 s for S_global, 10 s for reg). A deployed verifier receives a
    single candidate and cannot ask the caller to supply another length, so it
    picks one common window W = min(window_cap, N_reference, N_candidate) and
    drives both channels inside it. Measured cost of that choice is small:
    with sample_interval=120 over the full 450-frame window, S_global lands
    +2.6e-5 from the published row, inside the 5e-5 storage precision of the
    published CSV. See constants.py for the full comparison table.
    """

    def __init__(self,
                 extractor: Optional[FeatureExtractor] = None,
                 sample_interval: int = UNIFIED_SAMPLE_INTERVAL,
                 use_wavelet: bool = True,
                 threshold: float = THRESHOLD,
                 window_cap: Optional[int] = DEFAULT_WINDOW_CAP,
                 clip_mode: str = DEFAULT_CLIP_MODE,
                 gate: Optional[PreconditionGate] = None,
                 enforce_gate: bool = True):
        self.extractor = extractor or FeatureExtractor(
            sample_interval=sample_interval, use_wavelet=use_wavelet)
        self.threshold = threshold
        self.window_cap = window_cap
        self.clip_mode = clip_mode
        self.enforce_gate = enforce_gate
        self.gate = gate

    # ── window resolution ─────────────────────────────────
    def resolve_window(self, reference_path: str,
                       candidate_path: str) -> int:
        """Common frame window for both channels."""
        n_ref = frame_count(reference_path)
        n_cand = frame_count(candidate_path)
        limits = [n for n in (n_ref, n_cand) if n > 0]
        if not limits:
            return 0
        w = min(limits)
        if self.window_cap:
            w = min(w, self.window_cap)
        return max(int(w), 0)

    # ── global channel ─────────────────────────────────────
    def score_global(self, reference_path: str, candidate_path: str,
                     window: Optional[int] = None) -> float:
        if window is None:
            window = self.resolve_window(reference_path, candidate_path)
        feat_ref = self.extractor.extract_video_features(reference_path, window)
        feat_cand = self.extractor.extract_video_features(candidate_path, window)
        return compare_features(feat_ref, feat_cand)

    # ── regional channel ───────────────────────────────────
    def _score_regional(self, reference_path: str, candidate_path: str,
                        window: int, with_maps: bool = False):
        """One decode for the scalars and (optionally) the per-cell maps.

        The maps are produced by the same call that produces the score, not by
        a second pass over the frames: a teaching view that re-derives the
        score by a different route can silently disagree with the reported one,
        and a duplicate decode doubles the most expensive stage.
        """
        indices = sample_indices(window, N_FRAMES)
        if not indices:
            return None, None, None, None
        orig_frames = read_frames_at_indices(reference_path, indices)
        test_frames = read_frames_at_indices(candidate_path, indices)
        if with_maps:
            m = regional_maps(orig_frames, test_frames, self.extractor)
            return m['reg'], m['rd'], m['rb'], m
        reg, rd, rb = compute_regional_score(orig_frames, test_frames,
                                             self.extractor)
        return reg, rd, rb, None

    def score_regional(self, reference_path: str, candidate_path: str,
                       window: Optional[int] = None
                       ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if window is None:
            window = self.resolve_window(reference_path, candidate_path)
        reg, rd, rb, _ = self._score_regional(reference_path, candidate_path,
                                              window)
        return reg, rd, rb

    # ── full verdict ───────────────────────────────────────
    def verify(self, reference_path: str, candidate_path: str,
               with_maps: bool = False) -> Dict:
        """Full verdict. `with_maps` additionally returns the per-cell regional
        maps the teaching shell draws; the scored numbers are identical either
        way, and the default call site sees no new keys.
        """
        for p in (reference_path, candidate_path):
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        window = self.resolve_window(reference_path, candidate_path)

        gate_result = None
        if self.enforce_gate:
            gate = self.gate or PreconditionGate()
            gate_result = gate.check(reference_path, candidate_path)

        if self.enforce_gate and gate_result is not None \
                and not gate_result.should_score:
            return {
                'algo_version': ALGO_VERSION,
                'reference': reference_path,
                'candidate': candidate_path,
                'verdict': VERDICT_INAPPLICABLE,
                'evidence_strength': STRENGTH_NONE,
                'reasons': gate_result.blocking_codes,
                'warnings': [f.code for f in gate_result.warnings],
                'gate': gate_result.to_dict(),
                'S_global': None,
                'reg': None,
                'rd': None,
                'rb': None,
                'S_fused': None,
                'threshold': self.threshold,
                'window_frames': window,
            }

        s_global = self.score_global(reference_path, candidate_path, window)
        reg, rd, rb, maps = self._score_regional(reference_path, candidate_path,
                                                 window, with_maps=with_maps)
        if reg is None:
            reg, rd, rb = 0.0, 0.0, 0.0

        s_fused = fuse(s_global, reg)
        verdict = VERDICT_POSITIVE if s_fused >= self.threshold else VERDICT_NEGATIVE

        # Risk R2 guard: W_REGIONAL (0.2) > T (0.195), so a saturated regional
        # term can clear the threshold on its own. Flag that as weak evidence
        # instead of reporting a confident positive. Measured margin on
        # 0920.mp4 is 0.005 -- one thousandth of nothing.
        if verdict == VERDICT_POSITIVE and rd is not None and rd >= 1.0 \
                and s_global < self.threshold:
            strength = STRENGTH_WEAK
        else:
            strength = STRENGTH_NORMAL

        out = {
            'algo_version': ALGO_VERSION,
            'reference': reference_path,
            'candidate': candidate_path,
            'S_global': s_global,
            'reg': reg,
            'rd': rd,
            'rb': rb,
            'S_fused': s_fused,
            'threshold': self.threshold,
            'verdict': verdict,
            'evidence_strength': strength,
            'window_frames': window,
            'reasons': [],
            'warnings': ([] if gate_result is None
                         else [f.code for f in gate_result.warnings]),
            'gate': None if gate_result is None else gate_result.to_dict(),
        }
        if with_maps:
            out['regional_maps'] = maps
        return out
