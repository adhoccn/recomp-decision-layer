"""Recompression decision layer.

This package is the verdict layer referenced by Part I of "Signature Existence
and Optimal Multi-Stage Fusion for Blind Video Recompression Detection": the
cascade--fusion layer that turns per-stage scores into one decision.

It is a trimmed extract of the GCRR verifier (ALGO_VERSION below), keeping the
layers that participate in a verdict and dropping the deployment shells
(audit jobs, sandbox worker, web front end):

    L0  gate.PreconditionGate      refuse pairs violating the paired-identity
                                   protocol the score assumes
    L1  features.FeatureExtractor  the feature bank
    L2  scoring                    S_global, reg, fusion, thresholds
    L3  detector.GCRRVerifier      orchestration and verdict

The constants below are frozen per ALGO_VERSION; changing any of them changes
the headline numbers, so they must never be edited silently.

This extract deliberately does not import the L4 layers, so importing this
package pulls in no deployment dependencies. SCHEMA_VERSION lives in api.py,
which *does* import the audit shells; read it from decision_layer.api directly
if those are present.
"""

from .constants import ALGO_VERSION, THRESHOLD, W_GLOBAL, W_REGIONAL

__version__ = ALGO_VERSION

__all__ = [
    "ALGO_VERSION",
    "THRESHOLD",
    "W_GLOBAL",
    "W_REGIONAL",
]
