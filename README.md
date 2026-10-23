# recomp-decision-layer

Reference implementation of the **decision layer** described in *Signature
Existence and Optimal Multi-Stage Fusion for Blind Video Recompression
Detection* (Part I): the cascade–fusion layer that turns per-stage detector
scores into one verdict.

This is a trimmed extract of the [GCRR verifier](https://github.com/adhoccn/gcrr),
kept to the layers that participate in a verdict:

| Layer | Module | Role |
|---|---|---|
| L0 | `gate.py` | `PreconditionGate` — refuse a pair violating the paired-identity protocol the score assumes (frame count, resolution, window, content identity, codec, geometry, affine, colour pipeline) |
| L1 | `features.py` | `FeatureExtractor` — the feature bank |
| L2 | `scoring.py` | `S_global`, regional score, fusion rule |
| L3 | `detector.py` | `GCRRVerifier` — orchestration and verdict |

The deployment shells of the full product (resumable audit jobs, sandbox
worker, web front end) are **not** included, so this package imports nothing
beyond the layers above.

## The numbers the manuscript quotes

Frozen constants, not tunable defaults:

```python
>>> from decision_layer.scoring import fuse
>>> from decision_layer.constants import THRESHOLD, W_GLOBAL, W_REGIONAL
>>> THRESHOLD, W_GLOBAL, W_REGIONAL
(0.195, 0.8, 0.2)
>>> fuse(1.0, 0.0), fuse(0.0, 1.0)
(0.8, 0.20000000000000004)
```

`fuse` is exactly `S = alpha * S_global + (1 - alpha) * S_regional` with
`alpha = 0.8`, matching Part I Sec. VII. `ALGO_VERSION` is written into every
result payload so a verdict can be reproduced years later; changing any frozen
constant changes the headline numbers and therefore the version.

## Scope, stated plainly

The engine is **non-blind by construction**: it answers *is this candidate a
recompression of this reference?* and requires the reference video. It does not
answer *has this video been compressed twice?* That is not a limitation of the
implementation but the product's identity — Part I proves blind detection has
undetectable points.

Consequently this code does **not** implement Part I's blind-detection chain
(the Stage 1/2/3 scores `S_1`, `S_2`, `S_3`, their blind frequency selection, or
the `epsilon^{-2}` sample-complexity bound). Those belong to the blind setting;
see `adhoccn/bitstream-recompression-artifact` for the Part I theory scripts.

## Requirements

Python 3.13 with `numpy`, `scipy`, `opencv-python`, `PyWavelets`; `ffmpeg` on
`PATH` for decoding. See `requirements.txt`.

## License and provenance

This repository is an extract of [`adhoccn/gcrr`](https://github.com/adhoccn/gcrr)
(`gcrr_core/`), published here so the availability statement in Part I is
verifiable. Licence terms follow that repository; a copy is not duplicated here
to avoid drift — consult the LICENSE in `adhoccn/gcrr`.
