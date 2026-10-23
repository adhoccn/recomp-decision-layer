"""L0 precondition gate.

Why this layer exists at all
----------------------------
Every headline number of the paper was measured under a *paired identity*
protocol: reference and candidate are the same content, with equal frame
counts, equal resolution, and pixel alignment. The paper states this itself
(main text L514). Outside that protocol the false-positive rate is not low:

    cross-content pairs      45/45 -> 100% false positive
    rescaled candidates      0.75 / 0.5 / 0.333 -> 100% false positive
    frame-count drift        302 vs 300 frames -> S_global +0.33 for free
    identity self-compare    1 px shift -> 65-74% false positive

Those are not defects in the score; they are violations of a precondition the
score assumes. L0 checks the preconditions *before* any feature is extracted
and refuses to answer when they do not hold. `INAPPLICABLE` plus a reason code
is strictly better than a confident wrong verdict, and it is also the only
defensible posture in a forensic deliverable.

Cost envelope
-------------
This layer must stay cheap or nobody will call it. It reads container
metadata and decodes ~5 frames, then measures both a content fingerprint
(64x64 greyscale) and a geometric offset (256x256 centre crop, phase
correlation) from that single decode pass. Measured at roughly 95 ms per pair,
about 0.4% of the ~22 s a full scoring run costs. The 41-D feature bank is
never invoked, so a rejected pair never pays for features.

Coverage, and one thing that changed
------------------------------------
G6 (geometric alignment) was added after the shift case was measured directly:
a 1 px offset on a losslessly re-encoded pair moves S_global from 0.000 to
0.660, and S_fused to 0.728 -- 3.7x the threshold -- while the content
fingerprint still reports corr 0.993 / MAE 0.017, comfortably inside the pass
band. Content metrics cannot see a shift because the content genuinely is the
same. G6 therefore measures the offset itself.

G6 deliberately abstains when it cannot trust its own estimate: the phase-
correlation response for one cross-content pair reached 0.963, inside the true
positive range, so response alone proves nothing. Response only gates whether
the measured displacement may be acted on.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import (
    BLOCK,
    CODEC_LIMITED,
    CONTENT_LOW_CONFIDENCE_CORR,
    CONTENT_MAX_MEDIAN_MAE,
    CONTENT_MIN_MEDIAN_CORR,
    CONTENT_PROBE_FRAMES,
    CONTENT_PROBE_SIZE,
    E_CONTENT_MISMATCH,
    E_FRAME_COUNT_MISMATCH,
    E_GEOMETRY_SHIFT,
    E_GEOMETRY_AFFINE,
    E_INPUT_MISSING,
    E_INPUT_UNREADABLE,
    E_NO_VIDEO_STREAM,
    E_PIXFMT_MISMATCH,
    E_RESOLUTION_MISMATCH,
    E_WINDOW_TOO_SHORT,
    GEOMETRY_AFFINE_EPS,
    GEOMETRY_AFFINE_MAX_ITERS,
    GEOMETRY_MAX_ANGLE_DEG,
    GEOMETRY_MAX_ANISOTROPY,
    GEOMETRY_MAX_LOG_SCALE,
    GEOMETRY_MAX_SHIFT,
    GEOMETRY_MIN_RESPONSE,
    GEOMETRY_NOISE_ANGLE_DEG,
    GEOMETRY_NOISE_ANISOTROPY,
    GEOMETRY_NOISE_LOG_SCALE,
    GEOMETRY_PROBE_SIZE,
    GRID,
    MIN_REGION_EDGE,
    MIN_WINDOW_EDGE,
    MIN_WINDOW_FRAMES,
    N_FRAMES,
    PIXFMT_BLOCK_LEVELS,
    PIXFMT_WARN_LEVELS,
    WARN,
    W_CODEC_LIMITED,
    W_CODEC_UNKNOWN,
    W_CONTENT_LOW_CONFIDENCE,
    W_DURATION_DRIFT,
    W_GEOMETRY_MARGINAL,
    W_GEOMETRY_UNVERIFIABLE,
    W_PIXFMT_MISMATCH,
    W_PIXFMT_UNKNOWN,
    W_RESOLUTION_TOO_SMALL,
)
from .media import (probe_codec_cached, probe_container_cached,
                    read_frames_at_indices)
from .pixfmt import probe_pix_fmt_pair

# Facade: the reason codes and calibration above live in constants.py (public
# audit vocabulary) but are used here; the package __init__ re-exports them.

# Frames may differ by at most this much by default. Anything larger risks the
# measured +0.33 artefact, and the published 302-vs-300 case differs by only 2.
DEFAULT_MAX_FRAME_DELTA = 0


@dataclass
class Finding:
    """One gate outcome. Codes are stable across versions; messages are not."""

    code: str
    severity: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {'code': self.code, 'severity': self.severity,
                'message': self.message, 'detail': self.detail}

    def __str__(self) -> str:
        return f'[{self.severity}] {self.code}: {self.message}'


@dataclass
class GateResult:
    passed: bool
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == BLOCK]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    @property
    def codes(self) -> List[str]:
        return [f.code for f in self.findings]

    @property
    def blocking_codes(self) -> List[str]:
        return [f.code for f in self.blocking]

    @property
    def should_score(self) -> bool:
        """True when the pair may be handed to L1 for scoring."""
        return self.passed

    def to_dict(self) -> Dict[str, Any]:
        return {'passed': self.passed,
                'findings': [f.to_dict() for f in self.findings],
                'metrics': self.metrics}

    def __str__(self) -> str:
        if not self.findings:
            return 'gate: clean'
        return 'gate: ' + '; '.join(str(f) for f in self.findings)


# ── cheap probes ──────────────────────────────────────────
# probe_container / probe_codec live in media.py so that L0 and L3 can share
# them without importing each other.


def _grey_probe(frame: np.ndarray, size: int = CONTENT_PROBE_SIZE) -> Optional[np.ndarray]:
    """Downsampled greyscale in [0,1]; None on flat or unreadable frames."""
    if frame is None:
        return None
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    f = g.astype(np.float32) / 255.0
    # Flat frames carry no structure; correlating them is meaningless and a
    # cross-content pair can agree on a black frame purely by accident.
    if float(np.std(f)) < 1e-3:
        return None
    return f


def probe_indices(span: int, n_probes: int = CONTENT_PROBE_FRAMES) -> List[int]:
    """Frame indices spread across the FULL common timeline.

    `span` MUST be the whole common timeline, not the scoring window. Measured
    reason: this corpus shares a station intro of roughly 15-20 s across many
    files (0001 / 0006 / 0010 agree at corr 0.9999 up to t=15 s and diverge
    from t=30 s). Sampling only the first 450 frames -- which is exactly the
    scoring window -- makes unrelated videos look identical, so the gate would
    wave through precisely the pairs it exists to stop. Probe 0 lands on the
    intro and the remainder expose the divergence.
    """
    if span <= 0:
        return []
    if span <= n_probes:
        return list(range(span))
    return [min(int(span * i / n_probes), span - 1) for i in range(n_probes)]


def centre_crop_grey(frame: np.ndarray,
                     size: int = GEOMETRY_PROBE_SIZE) -> Optional[np.ndarray]:
    """Centre crop at full resolution.

    Centre CROP, not resize: downscaling would shrink a 1 px offset to a
    fraction of a pixel and the estimate would lose the very thing it is
    looking for. Only clips smaller than the probe are resized, and callers
    scale the result back (see estimate_shift).
    """
    if frame is None:
        return None
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = g.shape[:2]
    if h < size or w < size:
        return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return g[y0:y0 + size, x0:x0 + size].astype(np.float32)


def estimate_shift(a: np.ndarray, b: np.ndarray,
                   size: int = GEOMETRY_PROBE_SIZE
                   ) -> Optional[Tuple[float, float, float]]:
    """Sub-pixel offset between two frames, in ORIGINAL-image pixels.

    Returns (dx, dy, response), or None when a frame is too flat to correlate.

    `response` is the peak height of the phase-correlation surface. It does not
    by itself separate shifted pairs from unrelated ones -- one cross-content
    pair measured 0.963, inside the true-positive range. Its only job is to say
    whether the estimate is trustworthy enough to act on; below
    GEOMETRY_MIN_RESPONSE the gate abstains instead of blocking.
    """
    ca = centre_crop_grey(a, size)
    cb = centre_crop_grey(b, size)
    if ca is None or cb is None:
        return None
    if float(np.std(ca)) < 1.0 or float(np.std(cb)) < 1.0:
        return None
    win = cv2.createHanningWindow((size, size), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(ca, cb, win)
    except Exception:
        return None
    # A clip smaller than the probe was upscaled to `size`; convert the offset
    # back to image pixels so the threshold means the same thing everywhere.
    scale = 1.0
    src_h = a.shape[0] if a is not None else 0
    if src_h and src_h < size:
        scale = src_h / float(size)
    return float(dx) * scale, float(dy) * scale, float(response)


def decompose_affine(M: np.ndarray) -> Dict[str, float]:
    """Pull inversion-invariant quantities out of a 2x3 affine warp.

    G6 measures translation; this covers everything translation cannot
    express -- rotation, zoom and shear. The split is a polar decomposition
    A = R * P with R a rotation and P symmetric positive definite, taken from
    the SVD A = U S V^T as R = U V^T, which is what keeps the three
    quantities well defined for a general warp.

    The three are NOT independent components. A shear is the case that shows
    it: [[1, k], [0, 1]] decomposes into a rotation of k/2 rad plus a stretch,
    so a shear of k = 0.005 reads 0.143 deg of rotation as well as 0.005 of
    anisotropy -- and 0.143 deg is already past the block threshold. Do not
    read `angle` as "rotation and nothing else"; read it as "the rotation part
    of the warp", which shear legitimately has.

    All three are reported as magnitudes because the sign is a convention:
    whether an estimator returns the warp or its inverse depends on the
    estimator, and both are equally valid answers to "how far apart are these
    two frames". Inverting the warp negates angle, log_scale and anisotropy
    alike, so every magnitude below is inversion-invariant.

        angle        degrees of the rotation part R.
        log_scale    mean(log S) over the singular values; equals
                     log(sqrt|det A|). A pure zoom of 1.01 gives 0.00995.
        anisotropy   |log(S_max / S_min)|. Zero for a rotation or a uniform
                     zoom; non-zero for shear and non-uniform stretch.
    """
    A = np.asarray(M, dtype=np.float64)[:2, :2]
    try:
        U, S, Vt = np.linalg.svd(A)
    except Exception:
        return {'angle_deg': 0.0, 'log_scale': 0.0, 'anisotropy': 0.0}
    R = U @ Vt
    angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    S = np.maximum(np.abs(S), 1e-12)
    log_scale = float(np.mean(np.log(S)))
    anisotropy = abs(float(np.log(S[0] / S[-1])))
    return {'angle_deg': float(angle), 'log_scale': float(log_scale),
            'anisotropy': float(anisotropy)}


def estimate_affine(a: np.ndarray, b: np.ndarray,
                    size: int = GEOMETRY_PROBE_SIZE
                    ) -> Optional[Dict[str, float]]:
    """Full affine offset between two frames (ECC alignment).

    Companion to estimate_shift(). Phase correlation answers translation only
    and silently produces a meaningless peak when the frames are rotated; ECC
    alignment recovers rotation, zoom and shear as well.

    Returns {'angle_deg', 'log_scale', 'anisotropy', 'tx', 'ty', 'cc'} where
    `cc` is the ECC correlation coefficient -- the confidence signal, playing
    the same role that `response` plays for phase correlation. Returns None
    when the frames are too flat or alignment did not converge.

    All geometric quantities are magnitudes; see decompose_affine().
    """
    ca = centre_crop_grey(a, size)
    cb = centre_crop_grey(b, size)
    if ca is None or cb is None:
        return None
    if float(np.std(ca)) < 1.0 or float(np.std(cb)) < 1.0:
        return None
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                GEOMETRY_AFFINE_MAX_ITERS, GEOMETRY_AFFINE_EPS)
    try:
        cc, warp = cv2.findTransformECC(ca, cb, warp,
                                        cv2.MOTION_AFFINE, criteria)
    except Exception:
        # Raises rather than returning a flag when it fails to converge.
        return None
    if not np.isfinite(warp).all():
        return None
    out = decompose_affine(warp)
    out['tx'] = float(warp[0, 2])
    out['ty'] = float(warp[1, 2])
    out['cc'] = float(cc)
    return out


def probe_pair(reference_path: str, candidate_path: str,
               span: int,
               n_probes: int = CONTENT_PROBE_FRAMES,
               indices=None,
               with_affine: bool = False) -> Dict[str, Any]:
    """One decode pass, both L0 metrics.

    Returns {'corrs', 'maes', 'shifts', 'indices'}, where shifts is a list of
    (dx, dy, response); with_affine=True adds 'affines', a list of dicts from
    estimate_affine(). Reading the frames is the expensive part -- a seek into
    a long video -- so the content fingerprint and the geometry probes must
    share it rather than each paying for their own pass.
    """
    out: Dict[str, Any] = {'corrs': [], 'maes': [], 'shifts': [],
                           'indices': [], 'affines': []}
    if span <= 0:
        return out
    idx = indices if indices is not None else probe_indices(span, n_probes)
    idx = [min(int(i), span - 1) for i in idx]
    out['indices'] = list(idx)

    ref_frames = read_frames_at_indices(reference_path, idx)
    cand_frames = read_frames_at_indices(candidate_path, idx)

    for a, b in zip(ref_frames, cand_frames):
        fa = _grey_probe(a)
        fb = _grey_probe(b)
        if fa is not None and fb is not None \
                and not np.isnan(fa).any() and not np.isnan(fb).any():
            try:
                r = float(np.corrcoef(fa.ravel(), fb.ravel())[0, 1])
            except Exception:
                r = float('nan')
            if not math.isnan(r):
                out['corrs'].append(r)
                out['maes'].append(float(np.mean(np.abs(fa - fb))))
        shift = estimate_shift(a, b)
        if shift is not None:
            out['shifts'].append(shift)
        if with_affine:
            aff = estimate_affine(a, b)
            if aff is not None:
                out['affines'].append(aff)
    return out


def content_agreement(reference_path: str, candidate_path: str,
                      span: int,
                      n_probes: int = CONTENT_PROBE_FRAMES,
                      indices=None
                      ) -> Tuple[List[float], List[float], List[int]]:
    """Per-frame correlation and MAE between the two videos.

    Cheap content fingerprint: a handful of frames downsampled to 64x64.
    Kept as a thin wrapper over probe_pair() so existing callers and tests
    keep working; new code should call probe_pair() and get the geometry for
    free.
    """
    if span <= 0:
        return [], [], []
    res = probe_pair(reference_path, candidate_path, span,
                     n_probes=n_probes, indices=indices)
    return res['corrs'], res['maes'], res['indices']


class PreconditionGate:
    """Checks that the paired-identity protocol holds for a (ref, cand) pair.

    Typical use::

        gate = PreconditionGate()
        res = gate.check(ref, cand)
        if not res.should_score:
            return {'verdict': 'INAPPLICABLE', 'reasons': res.blocking_codes}
    """

    def __init__(self,
                 max_frame_delta: int = DEFAULT_MAX_FRAME_DELTA,
                 min_median_corr: float = CONTENT_MIN_MEDIAN_CORR,
                 max_median_mae: float = CONTENT_MAX_MEDIAN_MAE,
                 check_content: bool = True,
                 check_geometry: bool = True,
                 min_geometry_response: float = GEOMETRY_MIN_RESPONSE,
                 max_shift: float = GEOMETRY_MAX_SHIFT,
                 max_angle: float = GEOMETRY_MAX_ANGLE_DEG,
                 max_log_scale: float = GEOMETRY_MAX_LOG_SCALE,
                 max_anisotropy: float = GEOMETRY_MAX_ANISOTROPY,
                 noise_angle: float = GEOMETRY_NOISE_ANGLE_DEG,
                 noise_log_scale: float = GEOMETRY_NOISE_LOG_SCALE,
                 noise_anisotropy: float = GEOMETRY_NOISE_ANISOTROPY,
                 require_resolution_match: bool = True,
                 check_pixfmt: bool = True,
                 pixfmt_block_levels=PIXFMT_BLOCK_LEVELS,
                 pixfmt_warn_levels=PIXFMT_WARN_LEVELS):
        self.max_frame_delta = max_frame_delta
        self.min_median_corr = min_median_corr
        self.max_median_mae = max_median_mae
        self.check_content = check_content
        self.check_geometry = check_geometry
        self.min_geometry_response = min_geometry_response
        self.max_shift = max_shift
        self.max_angle = max_angle
        self.max_log_scale = max_log_scale
        self.max_anisotropy = max_anisotropy
        self.noise_angle = noise_angle
        self.noise_log_scale = noise_log_scale
        self.noise_anisotropy = noise_anisotropy
        self.require_resolution_match = require_resolution_match
        self.check_pixfmt = check_pixfmt
        self.pixfmt_block_levels = tuple(pixfmt_block_levels)
        self.pixfmt_warn_levels = tuple(pixfmt_warn_levels)

    # ── main entry ────────────────────────────────────────
    def check(self, reference_path: str, candidate_path: str) -> GateResult:
        findings: List[Finding] = []
        metrics: Dict[str, Any] = {}
        pair: Optional[Dict[str, Any]] = None

        for label, path in (('reference', reference_path),
                            ('candidate', candidate_path)):
            if not path or not os.path.exists(path):
                findings.append(Finding(
                    E_INPUT_MISSING, BLOCK,
                    f'{label} does not exist',
                    {'label': label, 'path': str(path)}))

        if any(f.code == E_INPUT_MISSING for f in findings):
            return GateResult(passed=False, findings=findings, metrics=metrics)

        ref = probe_container_cached(reference_path)
        cand = probe_container_cached(candidate_path)
        metrics['reference'] = ref
        metrics['candidate'] = cand

        if not ref['opened']:
            findings.append(Finding(
                E_INPUT_UNREADABLE, BLOCK,
                'reference could not be decoded', {'path': reference_path}))
        if not cand['opened']:
            findings.append(Finding(
                E_INPUT_UNREADABLE, BLOCK,
                'candidate could not be decoded', {'path': candidate_path}))
        if not (ref['opened'] and cand['opened']):
            return GateResult(passed=False, findings=findings, metrics=metrics)

        if ref['frames'] <= 0:
            findings.append(Finding(E_NO_VIDEO_STREAM, BLOCK,
                                    'reference has no decodable video stream'))
        if cand['frames'] <= 0:
            findings.append(Finding(E_NO_VIDEO_STREAM, BLOCK,
                                    'candidate has no decodable video stream'))
        if ref['frames'] <= 0 or cand['frames'] <= 0:
            return GateResult(passed=False, findings=findings, metrics=metrics)

        # ── G1 frame count ────────────────────────────────
        # 302 vs 300 buys S_global +0.33 for free. Refuse anything outside
        # tolerance rather than silently inheriting the artefact.
        n_delta = abs(ref['frames'] - cand['frames'])
        metrics['frame_delta'] = n_delta
        if n_delta > self.max_frame_delta:
            findings.append(Finding(
                E_FRAME_COUNT_MISMATCH, BLOCK,
                f'frame counts differ by {n_delta} '
                f'({ref["frames"]} vs {cand["frames"]}); the paired-identity '
                f'protocol requires one-to-one frames',
                {'reference_frames': ref['frames'],
                 'candidate_frames': cand['frames'],
                 'delta': n_delta}))

        # ── G2 resolution ─────────────────────────────────
        # Rescaled candidates false-positive at 100% on every tested factor.
        same_res = (ref['width'] == cand['width']
                    and ref['height'] == cand['height'])
        metrics['resolution_match'] = same_res
        if not same_res and self.require_resolution_match:
            findings.append(Finding(
                E_RESOLUTION_MISMATCH, BLOCK,
                f'resolution differs ({ref["width"]}x{ref["height"]} vs '
                f'{cand["width"]}x{cand["height"]}); resampling breaks the '
                f'block-grid alignment the score relies on',
                {'reference': [ref['width'], ref['height']],
                 'candidate': [cand['width'], cand['height']]}))

        # ── G8 colour pipeline ────────────────────────────
        # Runs before the content and geometry probes because it costs two
        # subprocess calls and no decode: a pair refused here never pays for
        # the frames the later checks would read.
        #
        # A pixel-format difference is not cosmetic. Measured losslessly:
        # yuv444p costs S_global 0.0925 and crosses the threshold on its own;
        # gray costs 0.2117. See the calibration table in constants.py.
        #
        # ⚠ What this check CANNOT see: a colour round-trip that ends at the
        # same declared format. Decoding to raw RGB and re-encoding back to
        # yuv420p scores S_global 0.2896 -- past the threshold -- and reports
        # yuv420p, indistinguishable from the reference by name. That gap is
        # recorded as risk R5b and stated on every forensic report.
        if self.check_pixfmt:
            pf = probe_pix_fmt_pair(reference_path, candidate_path)
            metrics['pix_fmt'] = pf
            level = pf['level']
            if level in self.pixfmt_block_levels:
                findings.append(Finding(
                    E_PIXFMT_MISMATCH, BLOCK,
                    f'pixel formats differ in {level} '
                    f'({pf["reference_name"]} vs {pf["candidate_name"]}); the '
                    f'colour pipeline resampled the chroma, which on its own '
                    f'moves the score past the threshold',
                    {'level': level, 'fields': pf['fields'],
                     'reference': pf['reference_name'],
                     'candidate': pf['candidate_name']}))
            elif level in self.pixfmt_warn_levels:
                code = (W_PIXFMT_UNKNOWN if level == 'unknown'
                        else W_PIXFMT_MISMATCH)
                what = ('could not be established' if level == 'unknown'
                        else f'differs in {level}')
                findings.append(Finding(
                    code, WARN,
                    f'colour pipeline {what} '
                    f'({pf["reference_name"]} vs {pf["candidate_name"]}); a '
                    f'colour-pipeline difference adds to the score and cannot '
                    f'be separated from recompression here',
                    {'level': level, 'fields': pf['fields'],
                     'reference': pf['reference_name'],
                     'candidate': pf['candidate_name']}))

        # ── G3 window length ──────────────────────────────
        window = min(ref['frames'], cand['frames'])
        metrics['window_frames'] = window
        if window < MIN_WINDOW_FRAMES:
            findings.append(Finding(
                E_WINDOW_TOO_SHORT, BLOCK,
                f'common window is {window} frames, fewer than the '
                f'{MIN_WINDOW_FRAMES} the regional stage needs',
                {'window_frames': window, 'required': MIN_WINDOW_FRAMES}))

        min_edge = min(ref['width'], ref['height'])
        metrics['min_edge'] = min_edge
        if min_edge < MIN_WINDOW_EDGE:
            findings.append(Finding(
                W_RESOLUTION_TOO_SMALL, WARN,
                f'smaller edge is {min_edge}px; regional lattice needs '
                f'{MIN_WINDOW_EDGE}px, so the rescue channel may be empty',
                {'min_edge': min_edge, 'required': MIN_WINDOW_EDGE}))

        # ── G4 content identity ───────────────────────────
        # Cross-content pairs scored above threshold 45/45 in the audit.
        # Probes span the FULL common timeline, never just the scoring window:
        # this corpus shares a 15-20 s station intro, so a window-only probe
        # reports unrelated videos as identical. See content_agreement().
        if self.check_content and window >= MIN_WINDOW_FRAMES:
            span = min(ref['frames'], cand['frames'])
            pair = probe_pair(reference_path, candidate_path, span,
                              with_affine=self.check_geometry)
            corrs, maes, probes = pair['corrs'], pair['maes'], pair['indices']
            if not corrs:
                findings.append(Finding(
                    E_CONTENT_MISMATCH, BLOCK,
                    'no comparable frame could be extracted; the pair cannot '
                    'be assumed to be the same content'))
            else:
                med_corr = float(np.median(corrs))
                med_mae = float(np.median(maes))
                metrics['content_median_corr'] = med_corr
                metrics['content_median_mae'] = med_mae
                metrics['content_frame_count'] = len(corrs)
                metrics['content_corr_min'] = float(np.min(corrs))
                metrics['content_corr'] = [round(c, 4) for c in corrs]
                metrics['content_mae'] = [round(m, 4) for m in maes]
                metrics['content_probe_indices'] = probes

                # Intro-only agreement: the opening matches but the body does
                # not. Worth naming explicitly, because the raw score will
                # still look confident and the operator needs to know why the
                # pair was refused.
                intro_only = (len(corrs) > 1 and corrs[0] > 0.98
                              and min(corrs[1:]) < 0.80)
                if intro_only:
                    metrics['content_intro_only'] = True

                if med_corr < self.min_median_corr or med_mae > self.max_median_mae:
                    msg = (f'content fingerprints disagree (median corr '
                           f'{med_corr:.4f} < {self.min_median_corr}, median '
                           f'MAE {med_mae:.4f} > {self.max_median_mae}); the '
                           f'pair is not frame-aligned copies of one video, '
                           f'and scoring it would return a false positive')
                    if intro_only:
                        msg += ('; the opening frames do agree, so this is '
                                'most likely a shared intro/outro rather than '
                                'a genuine recompression')
                    findings.append(Finding(
                        E_CONTENT_MISMATCH, BLOCK, msg,
                        {'median_corr': med_corr, 'median_mae': med_mae,
                         'corr_min': float(np.min(corrs)),
                         'corr_per_probe': [round(c, 4) for c in corrs],
                         'n_probes': len(corrs),
                         'intro_only': intro_only}))
                elif med_corr < CONTENT_LOW_CONFIDENCE_CORR:
                    findings.append(Finding(
                        W_CONTENT_LOW_CONFIDENCE, WARN,
                        f'content agreement is only moderate (median corr '
                        f'{med_corr:.4f}); the pair may be trimmed, edited, '
                        f'or shifted',
                        {'median_corr': med_corr, 'median_mae': med_mae}))

        # ── G6 geometric alignment ────────────────────────
        # Only meaningful once the content check has passed: if the two clips
        # are not the same content, a measured offset is meaningless (cross-
        # content pairs reported offsets of 30-70 px). Runs on the frames
        # already decoded for G4, so it costs an FFT, not another seek.
        #
        # This is the gap the content fingerprint cannot close. A 1 px offset
        # moves S_global from 0.000 to 0.660 while corr stays at 0.993, so
        # content metrics pass it clean and the score answers "recompressed"
        # for a pair that was merely cropped.
        if self.check_geometry and self.check_content \
                and window >= MIN_WINDOW_FRAMES \
                and not any(f.severity == BLOCK for f in findings):
            shifts = pair['shifts'] if pair else []
            if shifts:
                dx = float(np.median([s[0] for s in shifts]))
                dy = float(np.median([s[1] for s in shifts]))
                resp = float(np.median([s[2] for s in shifts]))
                resp_min = float(np.min([s[2] for s in shifts]))
                disp = max(abs(dx), abs(dy))
                metrics['geometry_dx'] = round(dx, 4)
                metrics['geometry_dy'] = round(dy, 4)
                metrics['geometry_response'] = round(resp, 4)
                metrics['geometry_response_min'] = round(resp_min, 4)
                metrics['geometry_displacement'] = round(disp, 4)
                metrics['geometry_frame_count'] = len(shifts)

                if resp < self.min_geometry_response:
                    # Do not block on a number we do not trust.
                    findings.append(Finding(
                        W_GEOMETRY_UNVERIFIABLE, WARN,
                        f'geometric alignment could not be established '
                        f'(phase-correlation response {resp:.4f} < '
                        f'{self.min_geometry_response}); a sub-pixel offset '
                        f'cannot be ruled out',
                        {'response': resp, 'response_min': resp_min,
                         'dx': dx, 'dy': dy}))
                elif disp > self.max_shift:
                    findings.append(Finding(
                        E_GEOMETRY_SHIFT, BLOCK,
                        f'frames are offset by approximately {disp:.2f} px '
                        f'(dx {dx:.2f}, dy {dy:.2f}); the block grid no longer '
                        f'corresponds and the score would read the offset as '
                        f'compression evidence',
                        {'dx': dx, 'dy': dy, 'displacement': disp,
                         'response': resp, 'response_min': resp_min,
                         'n_probes': len(shifts)}))

        # ── G7 rotation / zoom / shear ─────────────────────
        # G6 cannot see these: a 0.25 deg turn still phase-correlates to
        # response 1.000 and displacement 0.00 px, so it passes clean and
        # scores S_fused 0.692. ECC alignment recovers the full warp.
        #
        # Coverage is partial and that is measured, not assumed. Genuine
        # recompression perturbs apparent geometry by up to 0.05 deg / 0.15%
        # / 0.27% -- deblocking shifts edges -- which is as much as a small
        # real warp, so the two populations overlap. Warps clear of that
        # envelope are blocked; warps inside it are flagged MARGINAL rather
        # than refused, because refusing would reject real evidence.
        if self.check_geometry and self.check_content \
                and window >= MIN_WINDOW_FRAMES \
                and not any(f.severity == BLOCK for f in findings):
            affines = pair['affines'] if pair else []
            if affines:
                amed = lambda k: float(np.median([a[k] for a in affines]))
                angle = abs(amed('angle_deg'))
                lscale = abs(amed('log_scale'))
                aniso = amed('anisotropy')
                cc = float(np.median([a['cc'] for a in affines]))
                metrics['affine_angle_deg'] = round(angle, 5)
                metrics['affine_log_scale'] = round(lscale, 6)
                metrics['affine_anisotropy'] = round(aniso, 6)
                metrics['affine_cc'] = round(cc, 4)
                metrics['affine_frame_count'] = len(affines)

                over = []
                if angle > self.max_angle:
                    over.append(f'rotation {angle:.3f} deg')
                if lscale > self.max_log_scale:
                    over.append(f'zoom {math.exp(lscale):.4f}x')
                if aniso > self.max_anisotropy:
                    over.append(f'anisotropy {aniso:.5f}')
                marginal = (angle > self.noise_angle
                            or lscale > self.noise_log_scale
                            or aniso > self.noise_anisotropy)

                if cc < self.min_geometry_response:
                    findings.append(Finding(
                        W_GEOMETRY_UNVERIFIABLE, WARN,
                        f'affine alignment did not converge (ECC correlation '
                        f'{cc:.4f} < {self.min_geometry_response}); rotation, '
                        f'zoom and shear cannot be ruled out',
                        {'cc': cc, 'angle_deg': angle, 'log_scale': lscale,
                         'anisotropy': aniso}))
                elif over:
                    findings.append(Finding(
                        E_GEOMETRY_AFFINE, BLOCK,
                        f'frames differ by a geometric transform beyond the '
                        f'alignment noise envelope ({", ".join(over)}); the '
                        f'pair was resampled, so the block grid no longer '
                        f'corresponds and the score would read the warp as '
                        f'compression evidence',
                        {'angle_deg': angle, 'log_scale': lscale,
                         'anisotropy': aniso, 'cc': cc,
                         'n_probes': len(affines)}))
                elif marginal:
                    findings.append(Finding(
                        W_GEOMETRY_MARGINAL, WARN,
                        f'a small geometric transform was measured (rotation '
                        f'{angle:.3f} deg, zoom {math.exp(lscale):.5f}x, '
                        f'anisotropy {aniso:.5f}) but it sits inside the '
                        f'envelope that heavy recompression itself produces; '
                        f'a sub-0.1-deg warp cannot be ruled out and would '
                        f'raise the score on its own',
                        {'angle_deg': angle, 'log_scale': lscale,
                         'anisotropy': aniso, 'cc': cc}))

        # ── G5 advisory: duration / codec ─────────────────
        if ref['fps'] > 0 and cand['fps'] > 0:
            rel = abs(ref['fps'] - cand['fps']) / max(ref['fps'], 1e-9)
            metrics['fps_relative_delta'] = rel
            if rel > 0.01:
                findings.append(Finding(
                    W_DURATION_DRIFT, WARN,
                    f'frame rates differ by {rel:.2%} '
                    f'({ref["fps"]:.3f} vs {cand["fps"]:.3f})',
                    {'reference_fps': ref['fps'], 'candidate_fps': cand['fps']}))

        codec = probe_codec_cached(candidate_path)
        metrics['candidate_codec'] = codec
        if codec is None:
            findings.append(Finding(
                W_CODEC_UNKNOWN, WARN,
                'candidate codec could not be probed; no codec-specific '
                'caveat was applied'))
        elif codec in CODEC_LIMITED:
            findings.append(Finding(
                W_CODEC_LIMITED, WARN,
                f'candidate codec {codec} is outside the validated domain; '
                f'the 8 px block imprint does not transfer and the score may '
                f'degrade',
                {'codec': codec}))

        passed = not any(f.severity == BLOCK for f in findings)
        return GateResult(passed=passed, findings=findings, metrics=metrics)
