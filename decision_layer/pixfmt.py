"""Pixel-format classification for the L0 colour-pipeline gate (G8).

Why a whole module for a string comparison
------------------------------------------
Risk R5, measured in M2: pushing a losslessly re-encoded yuv420p clip through
a BGR intermediate and back to yuv420p -- no compression, no geometry, no
editing of any kind -- moves S_global from 0.000 to 0.104 and S_fused to
0.283. That is past the 0.195 threshold on its own. The only thing that
changed was the colour pipeline the pixels travelled through.

So "the two files have different pixel formats" is not cosmetic. It predicts
a score difference that has nothing to do with recompression. But not every
difference is equally dangerous: yuv420p vs yuvj420p differ only in whether
the range is limited or full, while yuv420p vs yuv444p differ in how much
chroma detail survives. This module names the difference so the gate can react
proportionally instead of treating every string mismatch as the same event.

Classification is deliberately textual. ffprobe is not always installed and
OpenCV does not expose pix_fmt at all, so all we ever have is the name.
"""

import re
from typing import Any, Dict, Optional

from .media import probe_pix_fmt, probe_pix_fmt_cached

_SUBSAMPLING = ('444', '422', '440', '420', '411', '410')

# Non-planar and semi-planar names that carry an implicit subsampling the
# 4xx notation does not spell out.
_IMPLICIT_SUBSAMPLING = {
    'nv12': '420', 'nv21': '420', 'p010': '420',
    'nv16': '422', 'nv61': '422',
    'nv24': '444', 'nv42': '444',
    'yuyv422': '422', 'uyvy422': '422', 'yvyu422': '422',
    'yuv422p': '422', 'yuv420p': '420', 'yuv444p': '444',
}

_DEPTH_RE = re.compile(r'(\d{1,2})(?:le|be)$')
_SUB_RE = re.compile(r'(4\d\d)')

# Ordered most to least dangerous. A pair is reported at its worst level.
LEVEL_IDENTICAL = 'identical'
LEVEL_RANGE = 'range'              # same layout, different luma/chroma range
LEVEL_DEPTH = 'depth'              # same layout, different bit depth
LEVEL_SUBSAMPLING = 'subsampling'  # different chroma resolution
LEVEL_FAMILY = 'family'            # yuv vs rgb vs gray
LEVEL_UNKNOWN = 'unknown'          # one or both sides could not be identified

LEVEL_ORDER = (LEVEL_IDENTICAL, LEVEL_RANGE, LEVEL_DEPTH,
               LEVEL_SUBSAMPLING, LEVEL_FAMILY, LEVEL_UNKNOWN)

# Levels at which the gate refuses to score rather than warn. Calibrated by
# tests/probe_pixfmt.py -- see the M3 report for the measurement.
BLOCKING_LEVELS = (LEVEL_SUBSAMPLING, LEVEL_FAMILY)


def pix_fmt_class(name: Optional[str]) -> Dict[str, Any]:
    """Break a pix_fmt name into family / subsampling / depth / range.

    Unknown inputs return a well-formed dict with `None` fields rather than
    raising: a report must never fail because a container used a format this
    code has never heard of.
    """
    out: Dict[str, Any] = {'name': name, 'family': None, 'subsampling': None,
                           'depth': None, 'range': None}
    if not name:
        return out
    low = name.strip().lower()

    if low.startswith(('gray', 'mono', 'ya')):
        out['family'] = 'gray'
    elif low.startswith(('rgb', 'bgr', 'gbr')):
        out['family'] = 'rgb'
    elif low.startswith(('yuv', 'nv', 'uyvy', 'yuyv', 'yvyu', 'vuyx')):
        out['family'] = 'yuv'
    else:
        out['family'] = None

    sub = None
    m = _SUB_RE.search(low)
    if m and m.group(1) in _SUBSAMPLING:
        sub = m.group(1)
    elif low in _IMPLICIT_SUBSAMPLING:
        sub = _IMPLICIT_SUBSAMPLING[low]
    else:
        for key, val in _IMPLICIT_SUBSAMPLING.items():
            if low.startswith(key):
                sub = val
                break
    out['subsampling'] = sub

    d = _DEPTH_RE.search(low)
    out['depth'] = int(d.group(1)) if d else 8

    # The 'j' in yuvj420p marks full-range JPEG video. Everything else in
    # this corpus is limited range, which is the x264/x265 default.
    if low.startswith('yuvj'):
        out['range'] = 'full'
    elif out['family'] == 'yuv':
        out['range'] = 'limited'
    else:
        out['range'] = None
    return out


def pix_fmt_difference(a: Optional[str], b: Optional[str]) -> Dict[str, Any]:
    """Classify how far apart two pixel formats are.

    Returns {'level', 'identical', 'fields', 'reference', 'candidate'}, where
    `fields` lists which attributes differ. `level` is the most severe of the
    differing attributes, so a pair that differs in both depth and
    subsampling is reported at LEVEL_SUBSAMPLING.
    """
    ca, cb = pix_fmt_class(a), pix_fmt_class(b)
    fields = []
    if a is None or b is None:
        return {'level': LEVEL_UNKNOWN, 'identical': False,
                'fields': ['unknown'], 'reference': ca, 'candidate': cb}
    if a != b:
        fields.append('name')
    for key in ('family', 'subsampling', 'depth', 'range'):
        va, vb = ca[key], cb[key]
        if va is None or vb is None:
            continue
        if va != vb:
            fields.append(key)

    level = LEVEL_IDENTICAL
    for key in ('range', 'depth', 'subsampling', 'family'):
        if key in fields:
            cand = {LEVEL_RANGE: 'range', LEVEL_DEPTH: 'depth',
                    LEVEL_SUBSAMPLING: 'subsampling',
                    LEVEL_FAMILY: 'family'}[key]
            if LEVEL_ORDER.index(cand) > LEVEL_ORDER.index(level):
                level = cand
    return {'level': level, 'identical': level == LEVEL_IDENTICAL,
            'fields': fields, 'reference': ca, 'candidate': cb}


def probe_pix_fmt_pair(reference_path: str,
                       candidate_path: str) -> Dict[str, Any]:
    """Probe both sides and classify the difference in one call.

    The two probes run concurrently. Each is a process spawn -- measured at
    350-480 ms on Windows, where process creation dominates -- so doing them
    back to back costs 620-850 ms and puts G8 ahead of every other gate check
    combined. They are independent subprocesses, so two threads halve it.

    Results go through the process-local probe cache (M4): an audit that runs
    one reference against fifty candidates would otherwise spawn a hundred
    processes to learn fifty times that both sides are yuv420p.
    """
    a = b = None
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(probe_pix_fmt_cached, reference_path)
            fb = pool.submit(probe_pix_fmt_cached, candidate_path)
            a, b = fa.result(), fb.result()
    except Exception:
        a = probe_pix_fmt_cached(reference_path)
        b = probe_pix_fmt_cached(candidate_path)
    out = pix_fmt_difference(a, b)
    out['reference_name'] = a
    out['candidate_name'] = b
    return out
