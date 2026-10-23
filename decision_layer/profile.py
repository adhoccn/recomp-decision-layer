"""Frozen parameter profile: what a verdict is answerable to.

A score is only evidence if the reader knows which numbers produced it. "This
video is a recompression, S_fused 0.31" says nothing about the threshold that
0.31 was compared against, the fusion weights that built it, or the sampling
window it was measured over -- and every one of those moves the answer. Two
years from now, in front of someone who did not run the tool, the record has
to stand on its own.

So the published configuration is pinned here as data, hashed, and shipped
with every record. The hash is the interesting part: it turns "was this
produced by the published method?" from a question about trust into a
question about arithmetic. A verifier on the other side recomputes the digest
from the constants module and compares.

Deviations are allowed but never silent. A caller may legitimately want
threshold 0.35 for a triage sweep; what they may not have is a forensic
report that looks standard while being anything but. Every deviation from the
profile is listed by name, and the forensic shell prints them on the report's
face.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    ALGO_VERSION,
    COLOR_MOMENT_GROUP_NORMS,
    DEFAULT_CLIP_MODE,
    GRID,
    N_BLOCK,
    N_DCT,
    N_FRAMES,
    NORM_FACTORS,
    THRESHOLD,
    UNIFIED_CLIP_FRAMES,
    UNIFIED_SAMPLE_INTERVAL,
    W_BLOCK,
    W_DCT,
    W_GLOBAL,
    W_REGIONAL,
    WEIGHTS,
)

PROFILE_ID = 'gcrr-forensic-v1'

# Everything downstream of this dict is derived, never hand-written, so the
# profile cannot drift from the constants module it claims to describe.
FROZEN_PARAMS: Dict[str, Any] = {
    'algo_version': ALGO_VERSION,
    'threshold': THRESHOLD,
    'w_global': W_GLOBAL,
    'w_regional': W_REGIONAL,
    'n_frames': N_FRAMES,
    'grid': GRID,
    'n_dct': N_DCT,
    'n_block': N_BLOCK,
    'w_dct': W_DCT,
    'w_block': W_BLOCK,
    'weights': dict(WEIGHTS),
    'norm_factors': dict(NORM_FACTORS),
    'color_moment_group_norms': list(COLOR_MOMENT_GROUP_NORMS),
    'sample_interval': UNIFIED_SAMPLE_INTERVAL,
    'window_cap': UNIFIED_CLIP_FRAMES,
    'clip_mode': DEFAULT_CLIP_MODE,
}

# Parameters the caller may override per request. Anything else in the profile
# is structural and cannot be overridden at all -- if it could, the digest
# would stop identifying the method.
OVERRIDABLE = ('threshold', 'window_cap', 'sample_interval')


def canonical_json(obj: Any) -> str:
    """Deterministic serialisation: sorted keys, no incidental whitespace.

    Float repr is exact round-trip in Python 3, so 0.195 and 0.195000000001
    hash differently -- which is the point.
    """
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def params_digest(params: Optional[Dict[str, Any]] = None) -> str:
    """SHA-256 of the canonical form of a parameter set."""
    return hashlib.sha256(
        canonical_json(params if params is not None else FROZEN_PARAMS)
        .encode('utf-8')).hexdigest()


#: Digest of the published parameter set. Changes iff a constant changes.
PROFILE_DIGEST = params_digest()


def runtime_params(threshold: float,
                   window_cap: Optional[int],
                   sample_interval: Optional[int] = None,
                   clip_mode: str = DEFAULT_CLIP_MODE) -> Dict[str, Any]:
    """The parameter set a request actually used.

    Built by copying the profile and overwriting only what the caller set, so
    an unspecified option cannot quietly become a deviation.
    """
    params = dict(FROZEN_PARAMS)
    for key, value in (('threshold', threshold), ('window_cap', window_cap),
                       ('sample_interval', sample_interval),
                       ('clip_mode', clip_mode)):
        if value is None:
            continue
        params[key] = value
    if params['sample_interval'] is None:
        params['sample_interval'] = UNIFIED_SAMPLE_INTERVAL
    return params


def deviations(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every way `params` differs from the frozen profile, as records.

    Only OVERRIDABLE keys can appear: a non-overridable difference means the
    code was changed underneath the profile, which is a build problem and not
    something a caller can cause.
    """
    out = []
    for key in OVERRIDABLE:
        expected = FROZEN_PARAMS.get(key)
        actual = params.get(key)
        if actual == expected:
            continue
        out.append({'param': key, 'profile': expected, 'used': actual})
    # algo_version tracks the constants module; a mismatch here means the
    # caller is running code the profile was not written for.
    if params.get('algo_version') not in (None, FROZEN_PARAMS['algo_version']):
        out.append({'param': 'algo_version',
                    'profile': FROZEN_PARAMS['algo_version'],
                    'used': params.get('algo_version')})
    return out


def is_standard(params: Dict[str, Any]) -> bool:
    return not deviations(params)


def describe(params: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
    """(digest, standard) for a parameter set; defaults to the profile."""
    p = params if params is not None else FROZEN_PARAMS
    return params_digest(p), is_standard(p)
