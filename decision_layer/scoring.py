"""Score computation: global distance, regional rescue, and GCRR fusion.

Sources of truth:
    global   : lightweight-final.py  L382-L426  (compare_features)
    regional : real_codec_full_recompute.py L173-L207 (compute_regional_score)
    fusion   : real_codec_full_recompute.py L391
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import euclidean

from .constants import (
    COLOR_MOMENT_DIM,
    COLOR_MOMENT_GROUP_NORMS,
    GRID,
    MIN_REGION_EDGE,
    N_BLOCK,
    N_DCT,
    N_FRAMES,
    NORM_FACTORS,
    W_BLOCK,
    W_DCT,
    W_GLOBAL,
    W_REGIONAL,
    WEIGHTS,
)


def compare_features(feat_orig: Dict[str, np.ndarray],
                     feat_test: Dict[str, np.ndarray]) -> float:
    """Weighted normalised distance between two feature dictionaries -> [0, 1].

    This is the S_global of the paper (also called v3_score in the experiment
    scripts). Weights are the offline Fisher weights; no training happens here.
    """
    total = 0.0
    used_weight = 0.0
    for key, w in WEIGHTS.items():
        if key not in feat_orig or key not in feat_test:
            continue
        if len(feat_orig[key]) == 0 or len(feat_test[key]) == 0:
            continue
        orig_mean = np.atleast_1d(np.mean(feat_orig[key], axis=0))
        test_mean = np.atleast_1d(np.mean(feat_test[key], axis=0))

        if key == 'color_moments' and len(orig_mean) == COLOR_MOMENT_DIM:
            # mean / std / skew carry different units -> normalise per group
            diff_norm = 0.0
            for g in range(3):
                base = g * 3
                for i, gnorm in enumerate(COLOR_MOMENT_GROUP_NORMS):
                    d = abs(float(orig_mean[base + i]) - float(test_mean[base + i]))
                    diff_norm += min(d / gnorm, 1.0)
            diff_norm /= 9.0
        else:
            diff = euclidean(orig_mean, test_mean)
            diff_norm = min(diff / NORM_FACTORS[key], 1.0)

        total += w * diff_norm
        used_weight += w

    if used_weight > 0:
        total /= used_weight
    return total


def _regional_arrays(orig_frames: Sequence,
                     test_frames: Sequence,
                     extractor,
                     grid: int = GRID) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell raw distances over a grid x grid lattice, max-pooled over frames.

    Returns two (grid, grid) float arrays of *unnormalised* distances; a cell
    that could not be scored (too small, or the extractor raised) is NaN.

    This is the single collection path for both the published scalar score and
    the teaching heat map. They must never be two implementations: a heat map
    that disagrees with the score it is supposed to explain is worse than no
    heat map, because it looks like evidence.

    Max over the per-cell arrays equals max over the old flat list, because
    the cell -> frame grouping is a partition: max over (frame, cell) pairs is
    the same number whether you group by cell first or not.
    """
    rd_map = np.full((grid, grid), np.nan, dtype=float)
    rb_map = np.full((grid, grid), np.nan, dtype=float)
    nf = min(len(orig_frames), len(test_frames), N_FRAMES)
    d = extractor.extract_dct_histogram_features
    b = extractor.extract_block_artifact_features

    for fi in range(nf):
        of, tf = orig_frames[fi], test_frames[fi]
        if of is None or tf is None:
            continue
        og = cv2.cvtColor(of, cv2.COLOR_BGR2GRAY) if len(of.shape) == 3 else of
        tg = cv2.cvtColor(tf, cv2.COLOR_BGR2GRAY) if len(tf.shape) == 3 else tf
        if og.shape != tg.shape:
            tg = cv2.resize(tg, (og.shape[1], og.shape[0]))
        h, w = og.shape
        rh, rw = h // grid, w // grid
        for ri in range(grid):
            for rj in range(grid):
                ro = og[ri * rh:(ri + 1) * rh, rj * rw:(rj + 1) * rw]
                rc = tg[ri * rh:(ri + 1) * rh, rj * rw:(rj + 1) * rw]
                if ro.shape[0] < MIN_REGION_EDGE or ro.shape[1] < MIN_REGION_EDGE:
                    continue
                try:
                    v = float(np.linalg.norm(d(ro) - d(rc)))
                    if np.isnan(rd_map[ri, rj]) or v > rd_map[ri, rj]:
                        rd_map[ri, rj] = v
                except Exception:
                    pass
                try:
                    v = float(np.linalg.norm(b(ro) - b(rc)))
                    if np.isnan(rb_map[ri, rj]) or v > rb_map[ri, rj]:
                        rb_map[ri, rj] = v
                except Exception:
                    pass

    return rd_map, rb_map


def _clip_max(arr: np.ndarray, norm: float) -> Optional[float]:
    """max / norm clipped to 1.0, or None when no cell was scored."""
    if not np.any(~np.isnan(arr)):
        return None
    return float(min(np.nanmax(arr) / norm, 1.0))


def compute_regional_score(orig_frames: Sequence,
                           test_frames: Sequence,
                           extractor,
                           grid: int = GRID) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Max-pooled regional DCT / block distance over a grid x grid lattice.

    Returns (reg, rd, rb), or (None, None, None) when no region could be scored.

    `extractor` is injected rather than taken from a module global, so the
    function is testable and usable inside a long-lived service process.
    """
    rd_map, rb_map = _regional_arrays(orig_frames, test_frames, extractor, grid)
    rd = _clip_max(rd_map, N_DCT)
    rb = _clip_max(rb_map, N_BLOCK)
    if rd is None and rb is None:
        return None, None, None
    rd = 0.0 if rd is None else rd
    rb = 0.0 if rb is None else rb
    return W_DCT * rd + W_BLOCK * rb, rd, rb


def regional_maps(orig_frames: Sequence,
                  test_frames: Sequence,
                  extractor,
                  grid: int = GRID) -> Dict[str, Any]:
    """Same computation as `compute_regional_score`, with the cells kept.

    Returns the published scalars (reg / rd / rb, identical to the tuple form)
    plus the per-cell normalised maps the teaching shell draws as a heat map.

    Two things a reader of the heat map needs to be told, and this function is
    where they are made explicit rather than left in a comment somewhere:

    * `rd_cell` / `rb_cell` are normalised per channel and per cell, so the
      published `reg` is `W_DCT * max(rd_cell) + W_BLOCK * max(rb_cell)` --
      **not** the maximum of any per-cell combined score. The two maximising
      cells can be different, and then `max(reg_cell) < reg`. The inequality
      `max(reg_cell) <= reg` always holds; the gap is reported as
      `pooling_gap` so the shell can show it instead of hiding it.
    * `reg_mean` is a *counterfactual*: what the score would be under mean
      pooling instead of max pooling. It is not the published number and must
      be labelled as such wherever it is displayed.
    """
    rd_map, rb_map = _regional_arrays(orig_frames, test_frames, extractor, grid)

    def cellmap(arr, norm):
        out = []
        for ri in range(grid):
            row = []
            for rj in range(grid):
                v = arr[ri, rj]
                row.append(None if np.isnan(v) else float(min(v / norm, 1.0)))
            out.append(row)
        return out

    rd_cell = cellmap(rd_map, N_DCT)
    rb_cell = cellmap(rb_map, N_BLOCK)

    def argmax_cell(arr):
        if not np.any(~np.isnan(arr)):
            return None
        idx = int(np.nanargmax(arr))
        return [idx // grid, idx % grid]

    def nanmean(arr):
        if not np.any(~np.isnan(arr)):
            return None
        return float(np.nanmean(arr))

    rd = _clip_max(rd_map, N_DCT)
    rb = _clip_max(rb_map, N_BLOCK)
    if rd is None and rb is None:
        reg = None
    else:
        reg = W_DCT * (rd or 0.0) + W_BLOCK * (rb or 0.0)

    # Mean pooling counterfactual: the mean of the *per-cell normalised*
    # values, not the mean of the raw distances. The two differ as soon as any
    # cell saturates, and the shell shows the per-cell values, so this is the
    # definition that matches what a student sees when they flip the toggle.
    def mean_cell(cells):
        vals = [v for row in cells for v in row if v is not None]
        return None if not vals else float(sum(vals) / len(vals))

    rd_m = mean_cell(rd_cell)
    rb_m = mean_cell(rb_cell)
    reg_mean = None
    if rd_m is not None or rb_m is not None:
        reg_mean = W_DCT * (rd_m or 0.0) + W_BLOCK * (rb_m or 0.0)

    # max over cells of the combined per-cell score, for the inequality above
    per_cell_max = None
    if reg is not None:
        vals = []
        for ri in range(grid):
            for rj in range(grid):
                if rd_cell[ri][rj] is None and rb_cell[ri][rj] is None:
                    continue
                vals.append(W_DCT * (rd_cell[ri][rj] or 0.0)
                            + W_BLOCK * (rb_cell[ri][rj] or 0.0))
        per_cell_max = max(vals) if vals else None

    return {
        'grid': int(grid),
        'rd': rd, 'rb': rb, 'reg': reg,
        'rd_cell': rd_cell,
        'rb_cell': rb_cell,
        'argmax_rd': argmax_cell(rd_map),
        'argmax_rb': argmax_cell(rb_map),
        'n_scored': int(np.count_nonzero(~np.isnan(rd_map))),
        'reg_mean': reg_mean,
        'per_cell_max': per_cell_max,
        'pooling_gap': (None if (reg is None or per_cell_max is None)
                        else float(reg - per_cell_max)),
    }


def fuse(s_global: float, reg: float) -> float:
    """GCRR fusion: S_fused = 0.8 * S_global + 0.2 * reg."""
    return W_GLOBAL * s_global + W_REGIONAL * reg


def sample_indices(total: int, n: int = N_FRAMES) -> list:
    """Uniform frame indices, matching the reference sampling formula."""
    if total <= 0:
        return []
    return [int(total * i / n) for i in range(n)] if total > n else list(range(total))
