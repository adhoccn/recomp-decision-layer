"""Public SDK surface: one call in, one auditable record out.

The verifier in detector.py answers the algorithmic question. This module
answers the product question: given two files on disk, produce a record that
can be logged, filed, and independently re-derived months later.

Three properties matter more here than convenience:

1. Self-describing. Every record carries `algo_version` and the exact
   parameters used, so a verdict never floats free of the code that produced
   it. A forensic deliverable that says "recompressed" without saying which
   threshold and which window is not evidence.
2. Complete on failure. A refused pair returns the same record shape with
   `verdict: INAPPLICABLE` and reason codes, never an exception for a
   precondition the caller could have checked.
3. Serializable. Everything is plain JSON types; no numpy scalars leak out.

Batch mode is separate from single-pair mode because the costs differ: a
single pair wants a readable record, a thousand pairs want a stream and a
worker pool.
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .constants import ALGO_VERSION, DEFAULT_WINDOW_CAP, THRESHOLD
from .detector import (STRENGTH_NONE, VERDICT_INAPPLICABLE, VERDICT_NEGATIVE,
                       VERDICT_POSITIVE, GCRRVerifier)
from .gate import PreconditionGate
from .profile import (PROFILE_DIGEST, PROFILE_ID, deviations, params_digest,
                      runtime_params)

SCHEMA_VERSION = '1.1'   # 1.1 adds params.digest / params.deviations (M3)

# Exit codes for the CLI. Deliberately not "0 = success": in a batch audit the
# interesting outcome is the verdict, and a shell pipeline should be able to
# select on it directly.
EXIT_NEGATIVE = 0        # verified NOT recompressed
EXIT_POSITIVE = 1        # verified RECOMPRESSED
EXIT_INAPPLICABLE = 2    # preconditions violated, no verdict given
EXIT_ERROR = 3           # bad usage, unreadable input, or internal failure

EXIT_CODES = {
    VERDICT_NEGATIVE: EXIT_NEGATIVE,
    VERDICT_POSITIVE: EXIT_POSITIVE,
    VERDICT_INAPPLICABLE: EXIT_INAPPLICABLE,
}


def sha256_file(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """Content hash, or None when the file cannot be read.

    Optional because it costs a full read of every input. Turn it on when the
    record has to stand up in court; leave it off for batch triage.
    """
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for block in iter(lambda: fh.read(chunk), b''):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def _plain(obj: Any) -> Any:
    """Coerce numpy / Path / anything else into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


class VerificationRequest:
    """Everything needed to re-derive one verdict.

    Held as an explicit object (rather than **kwargs) so it can be shipped to
    a worker process: on Windows the default start method is spawn, so any
    configuration a worker needs has to travel with the task.
    """

    def __init__(self,
                 reference: str,
                 candidate: str,
                 threshold: float = THRESHOLD,
                 window_cap: Optional[int] = DEFAULT_WINDOW_CAP,
                 enforce_gate: bool = True,
                 max_frame_delta: int = 0,
                 sample_interval: Optional[int] = None,
                 hash_inputs: bool = False,
                 pair_id: Optional[str] = None):
        self.reference = reference
        self.candidate = candidate
        self.threshold = threshold
        self.window_cap = window_cap
        self.enforce_gate = enforce_gate
        self.max_frame_delta = max_frame_delta
        self.sample_interval = sample_interval
        self.hash_inputs = hash_inputs
        self.pair_id = pair_id or os.path.basename(candidate)

    def build_verifier(self) -> GCRRVerifier:
        kw = {}
        if self.sample_interval is not None:
            kw['sample_interval'] = self.sample_interval
        gate = PreconditionGate(max_frame_delta=self.max_frame_delta) \
            if self.enforce_gate else None
        return GCRRVerifier(threshold=self.threshold,
                            window_cap=self.window_cap,
                            enforce_gate=self.enforce_gate,
                            gate=gate,
                            **kw)


def verify_pair(reference: str, candidate: str, **kwargs) -> Dict[str, Any]:
    """Verify one pair. Returns a JSON-safe record; raises only on bad input.

    A precondition violation is NOT an error: it returns
    `verdict: INAPPLICABLE` with `reasons`. Callers who want the exception
    back can pass `strict=True`.
    """
    strict = kwargs.pop('strict', False)
    req = kwargs.pop('request', None) or VerificationRequest(
        reference, candidate, **kwargs)

    started = time.time()
    try:
        verifier = req.build_verifier()
        result = verifier.verify(req.reference, req.candidate)
    except FileNotFoundError:
        if strict:
            raise
        result = {'verdict': VERDICT_INAPPLICABLE,
                  'evidence_strength': STRENGTH_NONE,
                  'reasons': ['E_INPUT_MISSING'],
                  'warnings': [], 'gate': None,
                  'S_global': None, 'reg': None, 'rd': None, 'rb': None,
                  'S_fused': None, 'window_frames': None}
    except Exception as exc:                      # never lose the record
        if strict:
            raise
        result = {'verdict': VERDICT_INAPPLICABLE,
                  'evidence_strength': STRENGTH_NONE,
                  'reasons': ['E_INTERNAL_ERROR'],
                  'warnings': [], 'gate': None,
                  'S_global': None, 'reg': None, 'rd': None, 'rb': None,
                  'S_fused': None, 'window_frames': None,
                  'error': f'{type(exc).__name__}: {exc}'}

    elapsed_ms = int(round((time.time() - started) * 1000))
    used = runtime_params(req.threshold, req.window_cap, req.sample_interval)
    dev = deviations(used)
    record = {
        'schema_version': SCHEMA_VERSION,
        'algo_version': ALGO_VERSION,
        'pair_id': req.pair_id,
        'reference': req.reference,
        'candidate': req.candidate,
        'verdict': result.get('verdict'),
        'evidence_strength': result.get('evidence_strength'),
        'reasons': result.get('reasons', []),
        'warnings': result.get('warnings', []),
        'scores': {k: result.get(k) for k in
                   ('S_global', 'reg', 'rd', 'rb', 'S_fused')},
        # The digest is the point of this block: it lets a reader check the
        # verdict against the published parameter set without re-running the
        # tool. Deviations are listed rather than merged away, because a
        # report that looks standard while being something else is worse than
        # no report.
        'params': {'threshold': req.threshold,
                   'window_cap': req.window_cap,
                   'sample_interval': (req.sample_interval
                                       or verifier_default_interval()),
                   'enforce_gate': req.enforce_gate,
                   'max_frame_delta': req.max_frame_delta,
                   'profile_id': PROFILE_ID,
                   'profile_digest': PROFILE_DIGEST,
                   'digest': params_digest(used),
                   'standard': not dev,
                   'deviations': dev},
        'window_frames': result.get('window_frames'),
        'gate': result.get('gate'),
        'elapsed_ms': elapsed_ms,
    }
    if 'error' in result:
        record['error'] = result['error']
    if req.hash_inputs:
        record['hashes'] = {'reference': sha256_file(req.reference),
                            'candidate': sha256_file(req.candidate)}
    return _plain(record)


def verifier_default_interval():
    from .constants import UNIFIED_SAMPLE_INTERVAL
    return UNIFIED_SAMPLE_INTERVAL


def _worker(req: VerificationRequest) -> Dict[str, Any]:
    return verify_pair(req.reference, req.candidate, request=req)


def verify_batch(requests: Iterable[VerificationRequest],
                 jobs: int = 1) -> Iterator[Dict[str, Any]]:
    """Verify many pairs, yielding records as they complete.

    jobs=1 runs in-process, which is what tests and small jobs want: spawning
    a pool for three pairs costs more than the work. jobs>1 uses a process
    pool; each task carries its own configuration because Windows defaults to
    spawn and a worker inherits nothing.
    """
    reqs = list(requests)
    if jobs <= 1 or len(reqs) == 1:
        for r in reqs:
            yield _worker(r)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for record in pool.map(_worker, reqs):
            yield record


# ── manifests ─────────────────────────────────────────────
def load_manifest(path: str, **defaults) -> List[VerificationRequest]:
    """Read a list of pairs from CSV, JSON or JSONL.

    CSV needs a header with `reference` and `candidate` columns; `pair_id` is
    optional. JSON may be either a list of objects or an object with a
    `pairs` key. Every row inherits the CLI defaults, and an explicit column
    wins over the default.

    Relative paths are resolved against the directory holding the manifest, not
    against the current directory. A manifest is meant to travel with the data
    it describes; one that only works from a particular working directory is a
    trap, and the failure mode is a thousand `E_INPUT_MISSING` rows.
    """
    ext = os.path.splitext(path)[1].lower()
    rows: List[Dict[str, Any]] = []

    if ext in ('.jsonl', '.ndjson'):
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext == '.json':
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        rows = data['pairs'] if isinstance(data, dict) else data
    else:
        import csv
        with open(path, 'r', encoding='utf-8-sig', newline='') as fh:
            for row in csv.DictReader(fh):
                rows.append({(k or '').strip(): (v or '').strip()
                             for k, v in row.items()})

    base = os.path.dirname(os.path.abspath(path))

    def _here(p: str) -> str:
        # normpath, not just join: a manifest written on Windows carries
        # `videos\a.mp4`, and the same file read on Linux would produce a
        # path whose backslash is part of the filename.
        p = p.replace('\\', '/') if os.sep == '/' else p
        return p if os.path.isabs(p) else os.path.normpath(
            os.path.join(base, p))

    out = []
    for i, row in enumerate(rows):
        ref = row.get('reference') or row.get('ref')
        cand = row.get('candidate') or row.get('cand')
        if not ref or not cand:
            continue
        ref, cand = _here(ref), _here(cand)
        kw = dict(defaults)
        for key in ('threshold', 'window_cap', 'max_frame_delta',
                    'sample_interval'):
            if row.get(key) not in (None, ''):
                kw[key] = type(_DEFAULT_TYPES[key])(row[key])
        for key in ('enforce_gate', 'hash_inputs'):
            if row.get(key) not in (None, ''):
                kw[key] = str(row[key]).lower() in ('1', 'true', 'yes', 'on')
        out.append(VerificationRequest(ref, cand,
                                       pair_id=row.get('pair_id')
                                       or f'pair{i:04d}', **kw))
    return out


_DEFAULT_TYPES = {'threshold': 1.0, 'window_cap': 1,
                  'max_frame_delta': 1, 'sample_interval': 1}


def to_json(record: Dict[str, Any], indent: Optional[int] = 2) -> str:
    return json.dumps(record, indent=indent, ensure_ascii=False, sort_keys=False)
