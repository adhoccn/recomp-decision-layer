"""Paper-mandated constants for the GCRR verifier.

Every value here is bound to the published pipeline. Changing any of them
changes the headline numbers, so they are frozen per ALGO_VERSION and must
never be edited silently.

Provenance:
    C:\\baidunetdiskdownload\\lightweight\\real_codec_full_recompute.py L55-L61
    C:\\baidunetdiskdownload\\lightweight\\lightweight-final.py L389-L397
"""

# ── Algorithm identity ────────────────────────────────────
# Bump this whenever a numeric semantic changes. Written into every result
# payload so a verdict can be reproduced years later.
#
#   0.1.0-m0  engine split out of the research scripts; two-window sampling
#   0.2.0-m1  L0 precondition gate wired in; channels unified onto one
#             450-frame window at sample_interval=120, and results may come
#             back INAPPLICABLE with no score at all
#   0.3.0-m2  G6 geometric alignment added. A pair may now be refused for a
#             sub-pixel offset, which the content fingerprint cannot see.
#             No scoring behaviour changed: every number a passing pair
#             produces is identical to 0.2.0-m1.
#   0.4.0-m2  G7 affine alignment added (rotation / zoom / shear), plus the
#             SDK and CLI surface. Again no scoring behaviour changed: a pair
#             that passed 0.3.0-m2 produces identical numbers here. What did
#             change is which pairs are refused, and the introduction of
#             W_GEOMETRY_MARGINAL for warps inside the alignment noise
#             envelope -- see the calibration note below.
#   0.5.0-m3  G8 colour-pipeline gate added, plus the forensic shell
#             (frozen parameter profile, deterministic PDF report, evidence
#             bundle). Scoring is again untouched; what changed is (a) which
#             pairs are refused or warned and (b) the addition of a
#             params_digest to every record, so a verdict can be checked
#             against the published parameter set without re-running it.
#   0.6.0-m4  Audit shell: jobs that survive interruption (content-keyed
#             tasks, crash-safe appends, refusals to continue a job whose
#             parameters changed). Scoring untouched a fifth time. The
#             profile digest moves because algo_version is part of the
#             profile -- that is the digest doing its job, not a regression.
#   0.7.0-m5  Teaching shell: the regional channel now also yields its
#             per-cell grids, from the same decode that yields the score, so a
#             heat map can be drawn without a second implementation. Scoring
#             is untouched a sixth time -- verify with tests.test_baseline_
#             parity, which still compares against the reference module. The
#             grids are additive; no scalar changes.
#   0.8.0     Hardening (R10/R11): per-task subprocess with a real timeout and
#             three distinct failure codes, plus a cost model in seconds that
#             follows pixels instead of file count. Scoring untouched a seventh
#             time. The isolation cost is ~3.7 s per task on the dev box.
#   0.8.1     Scheduler correctness (found deploying 0.8.0 to Linux). Two
#             manifest rows with the same pair_id *and* the same paths share a
#             task_key; indexing completed work by that key made the second
#             collapse into the first, and the run then hung with nothing
#             running and nothing left to start. Now indexed by position, the
#             record file is named by position too, and a scheduler that still
#             stops making progress fails the remainder with E_TASK_STALL
#             instead of spinning. Scoring is untouched; this only governs
#             whether a batch finishes.
#   0.8.2     Web shell only: every POST created an upload staging directory
#             before looking at the body, so a request that names server
#             paths -- which writes no file at all -- left an empty directory
#             that no task referred to and nothing removed before the TTL
#             sweep. One leaked directory per request, and the task count
#             stopped matching the disk. The staging directory is now created
#             on the first part that carries a file, and is removed again if
#             the request writes nothing or fails. No scoring, batch or CLI
#             behaviour changes; 0.8.1 and 0.8.2 give identical verdicts.
#   0.8.3     Web shell only, two things a live run exposed. (a) The page
#             never showed the video: the shell had no media route and no
#             player, so a verdict could be read but not looked at. There is
#             now GET /media/<id>/<ref|cand>, which answers byte ranges
#             (without 206 and a Content-Range, browsers will not seek and
#             Safari will not play at all), and two players beside the
#             numbers. (b) In server-path mode the two path boxes carried
#             only placeholders, so submitting the form empty produced a raw
#             400 -- the same 400 whether nothing had been sent or one of
#             the two boxes was blank. The boxes are now required, the error
#             names the empty one, and GET /api/server-videos lists what the
#             server root actually holds so a path never has to be guessed.
#             Scoring untouched an eighth time: the media route reads files,
#             the picker lists them, and neither reaches the pipeline.
#   0.8.4     Web shell reliability, no scoring change. The daily TTL sweep
#             timer was failing with 203/EXEC because the bundle (built on
#             Windows) shipped the sweep script without its execute bit, so a
#             fresh deploy left it non-runnable. The systemd unit now invokes
#             `/bin/bash` explicitly and make_bundle.py forces 0o755 on every
#             .sh member, so the cleanup cannot silently stop again. The
#             picker endpoint (/api/server-videos) is now cached for 60 s, and
#             /api/sweep accepts ttl_hours / dry_run. Verdicts unchanged: 0.8.3
#             and 0.8.4 are byte-for-byte identical on the pipeline.
ALGO_VERSION = "0.8.4"

# ── Decision ──────────────────────────────────────────────
THRESHOLD = 0.195          # T. WARNING: W_REGIONAL (0.2) > T (0.195).
                           # If reg saturates to 1.0, the regional term alone
                           # clears T for any S_global >= 0. See risk R2.

# ── Fusion: S_fused = W_GLOBAL * S_global + W_REGIONAL * reg ──
W_GLOBAL = 0.8
W_REGIONAL = 0.2

# ── Regional channel ──────────────────────────────────────
N_FRAMES = 5               # uniform frame sampling for the regional stage
GRID = 4                   # 4x4 = 16 regions
N_DCT = 12.0               # normalisation constant for rd
N_BLOCK = 4.0              # normalisation constant for rb
W_DCT = 0.6                # reg = W_DCT * rd + W_BLOCK * rb
W_BLOCK = 0.4
MIN_REGION_EDGE = 16       # regions smaller than this are skipped

# ── Global channel: 41-D weighted distance ────────────────
# Fisher weights, set offline on 70 disjoint videos. No training at runtime.
# Insertion order is part of the numeric contract. Float accumulation is not
# associative, so iterating these keys in a different order can move the last
# bit. Keep it identical to the reference implementation.
WEIGHTS = {
    'dct': 0.28,
    'block': 0.20,
    'pixel_contrast': 0.12,
    'color_moments': 0.14,
    'motion': 0.08,
    'residual': 0.10,
    'texture': 0.08,
}

NORM_FACTORS = {
    'dct': 5.0,
    'block': 3.0,
    'pixel_contrast': 0.5,
    'motion': 5.0,
    'residual': 10.0,
    'texture': 5.0,
}

# Colour moments are grouped by moment order: mean / std / skew each carry a
# different unit, so they get their own divisor.
COLOR_MOMENT_GROUP_NORMS = (255.0, 128.0, 3.0)
COLOR_MOMENT_DIM = 9

FEATURE_KEYS = (
    'pixel_contrast', 'color_moments', 'motion',
    'dct', 'block', 'residual', 'texture',
)

# ── Sampling configuration of the reference pipeline ──────
# Resolved by direct measurement in M0 (tests/smoke_e2e.py, 0010.mp4 / h264):
#
#   sample_interval=120 + 15 s clip  ->  S_global 0.585426 vs published
#                                        0.585400, delta +2.6e-5. The CSV
#                                        stores 4 decimals, so anything under
#                                        5e-5 is exact within storage precision.
#   sample_interval=60  + 10 s clip  ->  S_global 0.539734, delta -0.046.
#
# The regional channel does NOT depend on sample_interval: it picks N_FRAMES
# frames by index and calls the extractors directly, bypassing the sampling
# loop. The `sample_interval=60` in the reference worker is therefore inert
# for reg -- confirmed by reg delta = 0.000000.
SAMPLE_INTERVAL_GLOBAL = 120
CLIP_SECONDS_GLOBAL = 15.0
CLIP_SECONDS_REGIONAL = 10.0
DEFAULT_SAMPLE_INTERVAL = SAMPLE_INTERVAL_GLOBAL
DEFAULT_CLIP_SECONDS = CLIP_SECONDS_REGIONAL

# ── Clip policy ───────────────────────────────────────────
# Measured in M1 by tests/probe_clip_policy.py (0010.mp4 / h264, published row
# S_global 0.585400, reg 1.000000, S_fused 0.668320). Costs are standalone --
# caches cleared per row, so each figure is what that config costs alone:
#
#   configuration            dSG       dreg      cost
#   faithful g15/r10 si120   +2.6e-5   0.000000  20.5 s
#   unified  15s     si120   +2.6e-5   0.000000  21.7 s   <-- PINNED
#   unified  15s     si60    -0.0333   0.000000  36.7 s
#   unified  10s     si120   -0.0016   0.000000  17.0 s
#   unified  10s     si60    -0.0457   0.000000  24.7 s
#
# CORRECTION to an earlier claim: unified is NOT "twice as fast" as faithful.
# Both cost about the same (21.7 s vs 20.5 s) because faithful's two windows
# nest -- one 450-frame decode serves both the 15 s global window and the 10 s
# regional window. The real cost lever is the sample interval: si=120 extracts
# 4 frames per video where si=60 extracts 8, and that is where the ~40% saving
# comes from. Unified was adopted for PRODUCT SEMANTICS (a caller supplies one
# candidate and cannot be asked for a second length), not for speed.
#
# dSG budget is 5e-5, the storage precision of the published CSV. Only si=120
# over the full 450-frame window clears it.
#
# `dreg == 0` across this table is NOT evidence that reg is window-invariant.
# This pair is saturated (rd = rb = 1.0) and 99.8875% of the corpus is too. An
# unsaturated pair can still move with the window; see risk R2.
UNIFIED_CLIP_SECONDS = 15.0
UNIFIED_SAMPLE_INTERVAL = 120
UNIFIED_CLIP_FRAMES = 450            # 15 s @ 30 fps

# Shortest window we will still score, below which the sampling grid starves.
MIN_WINDOW_FRAMES_FALLBACK = N_FRAMES + 1

CLIP_MODE_FAITHFUL = 'faithful'      # keep the two-window behaviour (legacy)
CLIP_MODE_UNIFIED = 'unified'
DEFAULT_CLIP_MODE = CLIP_MODE_UNIFIED
# ── task isolation outcome codes (M4 hardening, risk R10) ──────────────
# Defined here rather than in taskrunner.py so job.py can consult them without
# importing the runner, which would make the two modules import each other.
#
# Each is a distinct claim, and collapsing them is how an audit file stops
# being readable:
#   TIMEOUT   -- we stopped it; the file may be fine, we simply ran out of patience
#   CRASHED   -- the worker died without leaving anything behind
#   UNKILLABLE-- we could not stop it; that worker slot is gone
#   STALL     -- the scheduler itself stopped making progress and gave up
#                instead of spinning. Unlike the others this is never a
#                property of the input; it is a bug in the runner, which is
#                precisely why it must be loud rather than quiet.
E_TASK_TIMEOUT = 'E_TASK_TIMEOUT'
E_TASK_CRASHED = 'E_TASK_CRASHED'
E_TASK_UNKILLABLE = 'E_TASK_UNKILLABLE'
E_TASK_STALL = 'E_TASK_STALL'

#: How long the scheduler may go without emitting anything, starting anything,
#: or collecting anything, before it declares itself stuck and fails the rest.
#: Nothing legitimate looks like this: a running child is progress, and a child
#: that is running but over budget is caught by DEFAULT_TASK_TIMEOUT instead.
DEFAULT_STALL_TIMEOUT = 300.0

# ── cost model (risk R11: the tail is not flat) ────────────────────────
# Fitted from tests/probe_cost_tiers.py on the real corpus, September 2026.
#
#   tier    n   measured pair cost      seconds per megapixel
#   720p    2    16.5 - 18.3 s          19.86
#   1080p   2    34.2 - 39.0 s          18.79
#   4K      2   153.3 - 159.6 s         18.04
#
# The ratio being flat across a 9.6x span of pixels is the finding: cost is a
# linear function of pixels, so one coefficient serves every tier and there is
# no need for a piecewise table. Frame count barely enters it -- 0216.mp4 has
# 3133 frames and costs slightly MORE than 0215.mp4's 15000, which is the same
# result M4 found about the gate (cost follows resolution, not length).
#
# Fitting the six points gives seconds = 17.54 * MPx + 1.25, which lands within
# 3 % of every measurement.
COST_S_PER_MEGAPIXEL = 17.54
COST_S_FIXED = 1.25

#: Identity pairs (a file compared with itself) finish sooner than real ones:
#: the optical-flow residual is zero, so it converges faster. Measured on
#: 1080p: 36.6 s identity versus 58 s for a real pair. Anything that quotes an
#: absolute duration from the model above has to undo this.
COST_REAL_PAIR_FACTOR = 1.585

#: Default seconds any single task may consume before it is terminated.
#: Sized from measurement, not comfort: M4's slowest measured 1080p pair was
#: 86 s, and a 4K file is 4.3x the pixels, so 900 s is roughly 10x headroom on
#: the worst case we have actually seen.
DEFAULT_TASK_TIMEOUT = 900.0

DEFAULT_WINDOW_CAP = UNIFIED_CLIP_FRAMES

# ── L0 tunables ───────────────────────────────────────────
# Product-layer policy, NOT paper constants: these govern how strict the gate
# is and changing them moves no score.
#
# Calibrated in M1 by tests/probe_gate_thresholds.py over three populations:
#
#   POS        same content recompressed   n=10  corr 0.9997..1.0000
#                                                mae  0.0006..0.0038
#   NEG-FAR    unrelated, whole timeline   n=45  corr -0.1192..0.4041
#                                                mae  0.1730..0.3901
#   NEG-INTRO  unrelated, first 15 s       n=45  corr -0.1855..0.9999
#                                                mae  0.0028..0.4548
#
# NEG-FAR and POS are separated perfectly by any corr threshold in
# [0.50, 0.98] and any mae threshold in [0.02, 0.15]; the pinned values sit
# mid-gap. NEG-INTRO overlaps POS completely, which is the empirical reason
# the probe must span the WHOLE timeline (see gate.content_agreement) -- those
# pairs share a 15-20 s station intro and are indistinguishable inside the
# scoring window.
CONTENT_PROBE_FRAMES = 5     # frames sampled for the content fingerprint
CONTENT_PROBE_SIZE = 64      # downsample target
CONTENT_MIN_MEDIAN_CORR = 0.85   # below this -> different content
CONTENT_MAX_MEDIAN_MAE = 0.12    # above this -> different content (normalised)
CONTENT_LOW_CONFIDENCE_CORR = 0.95   # below this -> WARN, still scored

# ── L0 geometry probe (G6) ────────────────────────────────
# A 1 px offset between content-identical frames drives S_global from 0.000 to
# 0.660 (S_fused 0.728, 3.7x the threshold) while the content fingerprint still
# reports corr 0.993 / MAE 0.017 -- comfortably inside the pass band. Content
# metrics cannot see a shift because the content genuinely is the same, so L0
# needs a geometric check. Calibrated by tests/probe_geometry_calibration.py
# over 36 true recompressions, 6 shifted pairs and 4 cross-content pairs:
#
#   population          resp_median range     |shift| range
#   POS  recompressed   0.5151 .. 1.0032      0.00 .. 0.18
#   SHIFT 1/2/4 px      0.9804 .. 0.9993      1.00 .. 4.00
#   XCON cross-content  0.0046 .. 0.9630      30   .. 70
#
# Two conclusions shaped these numbers. First, the DISPLACEMENT is the
# discriminating quantity, not the response: POS never exceeds 0.18 px and
# SHIFT never falls below 1.00 px, so 0.5 px sits mid-gap (margin 0.32 below,
# 0.50 above). Second, the response cannot separate cross-content pairs on its
# own -- one such pair reaches 0.963, overlapping POS -- so its only job is to
# decide whether the estimate is trustworthy enough to act on. Below
# GEOMETRY_MIN_RESPONSE the probe abstains rather than blocking.
GEOMETRY_PROBE_SIZE = 256     # centre crop; keeps the pixel grid intact
GEOMETRY_PROBE_FRAMES = 5
GEOMETRY_MIN_RESPONSE = 0.4   # below this the estimate is noise -> abstain
GEOMETRY_MAX_SHIFT = 0.5      # px, in original-image scale

# ── Affine branch (G7): rotation, zoom, shear ───────────────
# Phase correlation sees none of these: a 0.25 degree turn still yields a
# response of 1.000 and a displacement of 0.00 px, so G6 waves it straight
# through and the pair scores S_fused 0.692 (3.5x the threshold). Alignment is
# therefore done by ECC, which recovers the full 2x3 warp.
#
# Calibration (tests/probe_affine_calibration.py, 15 true recompressions over
# 3 videos x {h264 23/28/38, h265 30/38}, 9 lossless warps, 1 control):
#
#   quantity     CTRL   POS min..max        WARP seen at
#   angle_deg    0.000  0.00098 .. 0.05045  0.05 deg -> 0.0492
#                                           0.10 deg -> 0.0995
#                                           0.25 deg -> 0.2519
#   log_scale    0.000  0.00001 .. 0.00146  1.001 -> 0.00100
#                                           1.002 -> 0.00201
#                                           1.005 -> 0.00504
#   anisotropy   0.000  0.00005 .. 0.00273  0.001 -> 0.00098
#                                           0.002 -> 0.00198
#                                           0.005 -> 0.00501
#
# THE POPULATIONS OVERLAP, and that is the finding. A genuine h265 crf38
# recompression measures 0.0504 deg, which is MORE than an actual 0.05 deg
# rotation. Compression damage -- deblocking moves edges by a fraction of a
# pixel -- perturbs apparent geometry by as much as a small warp does, so no
# threshold can separate the two at that magnitude.
#
# What this means physically: a 0.05 deg turn displaces pixels 128 px from the
# centre by only 0.11 px, yet S_fused goes from 0.000 to 0.569. The score is
# sensitive at the 0.1-px level; the alignment probe cannot resolve 0.1 px on
# a compressed candidate. That gap is irreducible, not a tuning problem.
#
# G7 therefore covers only warps well clear of the noise envelope, at 1.8x the
# worst true positive measured:
#
#   rotation  >= 0.09 deg      misses 0.05 deg   (S_fused 0.569)
#   zoom      >= 1.003         misses 1.001/1.002 (0.547 / 0.622)
#   shear     >= 0.005         misses 0.001/0.002 (0.412 / 0.516)
#
# Anything inside the noise envelope is reported as W_GEOMETRY_MARGINAL: not
# blocked, because refusing would reject real evidence, but not silent either.
# Sub-threshold warps remain a documented false-positive source (risk R6).
GEOMETRY_AFFINE_MAX_ITERS = 50
GEOMETRY_AFFINE_EPS = 1e-4
GEOMETRY_MIN_CC = 0.4         # ECC correlation below this -> abstain

GEOMETRY_NOISE_ANGLE_DEG = 0.05    # >= POS max 0.05045, rounded up
GEOMETRY_NOISE_LOG_SCALE = 0.0015  # >= POS max 0.00146
GEOMETRY_NOISE_ANISOTROPY = 0.0028  # >= POS max 0.00273

GEOMETRY_MAX_ANGLE_DEG = 0.09      # 1.8x POS max
GEOMETRY_MAX_LOG_SCALE = 0.0026
GEOMETRY_MAX_ANISOTROPY = 0.0049

# ── Colour pipeline (G8): pix_fmt ───────────────────────────
# Risk R5. Every variant below is a LOSSLESS re-encode of one base clip that
# changes nothing except `-pix_fmt`, measured by tests/probe_pixfmt.py:
#
#   variant        level         S_global   S_fused   crosses T=0.195
#   420 (control)  identical       0.0000    0.0000   no
#   420 via FFV1   identical       0.0000    0.0000   no
#   yuvj420p       range           0.0474    0.0902   no
#   yuv422p        subsampling     0.0252    0.0331   no
#   yuv444p        subsampling     0.0925    0.2080   YES
#   gray           family          0.2117    0.3022   YES
#   RGB round-trip identical       0.2896    0.3155   YES  <-- see below
#
# Two controls, both exactly zero, which is what licenses the rest of the
# table: the lossless claim holds for both encoders, so every number above is
# attributable to the colour pipeline and nothing else.
#
# BLOCK is set at `subsampling` and above, not at the level where the measured
# effect crosses T. That is deliberate. yuv422p costs only 0.0331 on this
# clip, but separating 422 from 444 would mean drawing a line from one clip,
# one codec and one resolution -- and a threshold calibrated on n=1 is not a
# threshold. The principled line is "the chroma resolution differs", which is
# also what an operator can act on: transcode to match, then re-run.
#
# `range` warns instead of blocking. Full-vs-limited conversion is common
# enough in the wild (phone and platform uploads) that refusing would reject
# far more usable pairs than it saves, and 0.0902 cannot reach T on its own.
#
# ⚠ THE BLIND SPOT, measured not assumed: the RGB round-trip -- decode to raw
# RGB, re-encode losslessly back to yuv420p -- scores S_global 0.2896 and
# crosses the threshold, yet the finished file REPORTS yuv420p, identical to
# the reference. No comparison of pixel-format names can see it. G8 therefore
# closes only the DECLARED half of R5 (R5a); an undeclared colour round-trip
# (R5b) stays a live false-positive source and every forensic report carries
# that caveat on its face.
PIXFMT_BLOCK_LEVELS = ('subsampling', 'family')
PIXFMT_WARN_LEVELS = ('range', 'depth', 'unknown')

MIN_WINDOW_FRAMES = N_FRAMES + 1     # regional stage needs N_FRAMES samples
MIN_WINDOW_EDGE = GRID * MIN_REGION_EDGE   # smallest scorable region lattice

# Frames may differ by at most this much. Anything larger risks the measured
# +0.33 artefact, and the published 302-vs-300 case is only 2 frames apart.
DEFAULT_MAX_FRAME_DELTA = 0

# ── L0 reason codes ───────────────────────────────────────
# Stable identifiers. Safe to persist in an audit log or a court report, and
# safe to branch on in code. Messages are NOT stable; never match on those.
BLOCK = 'BLOCK'    # precondition violated -> refuse to score
WARN = 'WARN'      # usable, but the verdict carries a caveat

E_INPUT_MISSING = 'E_INPUT_MISSING'
E_INPUT_UNREADABLE = 'E_INPUT_UNREADABLE'
E_NO_VIDEO_STREAM = 'E_NO_VIDEO_STREAM'
E_FRAME_COUNT_MISMATCH = 'E_FRAME_COUNT_MISMATCH'
E_RESOLUTION_MISMATCH = 'E_RESOLUTION_MISMATCH'
E_CONTENT_MISMATCH = 'E_CONTENT_MISMATCH'
E_WINDOW_TOO_SHORT = 'E_WINDOW_TOO_SHORT'
E_GEOMETRY_SHIFT = 'E_GEOMETRY_SHIFT'
E_GEOMETRY_AFFINE = 'E_GEOMETRY_AFFINE'
E_PIXFMT_MISMATCH = 'E_PIXFMT_MISMATCH'

W_RESOLUTION_TOO_SMALL = 'W_RESOLUTION_TOO_SMALL'
W_DURATION_DRIFT = 'W_DURATION_DRIFT'
W_CODEC_LIMITED = 'W_CODEC_LIMITED'
W_CODEC_UNKNOWN = 'W_CODEC_UNKNOWN'
W_CONTENT_LOW_CONFIDENCE = 'W_CONTENT_LOW_CONFIDENCE'
W_GEOMETRY_UNVERIFIABLE = 'W_GEOMETRY_UNVERIFIABLE'
W_GEOMETRY_MARGINAL = 'W_GEOMETRY_MARGINAL'
W_PIXFMT_MISMATCH = 'W_PIXFMT_MISMATCH'
W_PIXFMT_UNKNOWN = 'W_PIXFMT_UNKNOWN'

ALL_REASON_CODES = (
    E_INPUT_MISSING, E_INPUT_UNREADABLE, E_NO_VIDEO_STREAM,
    E_FRAME_COUNT_MISMATCH, E_RESOLUTION_MISMATCH, E_CONTENT_MISMATCH,
    E_WINDOW_TOO_SHORT, E_GEOMETRY_SHIFT, E_GEOMETRY_AFFINE,
    W_RESOLUTION_TOO_SMALL, W_DURATION_DRIFT, W_CODEC_LIMITED,
    W_CODEC_UNKNOWN, W_CONTENT_LOW_CONFIDENCE, W_GEOMETRY_UNVERIFIABLE,
    W_GEOMETRY_MARGINAL, W_PIXFMT_MISMATCH, W_PIXFMT_UNKNOWN,
)

# ── Codecs ────────────────────────────────────────────────
FORMATS = {
    'h264': {'codec': 'libx264', 'crf': '28', 'preset': 'ultrafast'},
    'h265': {'codec': 'libx265', 'crf': '30', 'preset': 'ultrafast'},
    'vp9': {'codec': 'libvpx-vp9', 'crf': '32', 'preset': 'ultrafast'},
    'xvid': {'codec': 'mpeg4', 'crf': None, 'preset': 'ultrafast'},
}
XVID_BITRATE = ('-b:v', '500k', '-maxrate', '750k', '-bufsize', '1000k')

SUPPORTED_CODECS = tuple(FORMATS)

# Codecs whose forensic imprint sits on a different block grid. The 8 px
# grid that carries H.264 does not transfer; tilting to 32 px recovers H.265
# (AUC 0.609 -> 0.947) but NOT VP9 (0.728 / 0.504).
CODEC_LIMITED = ('vp9',)
