"""Media primitives shared by every layer.

Kept separate because both L0 (gate) and L3 (detector) need frame access, and
the dependency direction must stay one-way: orchestration depends on the gate,
never the reverse. Putting these in detector.py would make the two modules
import each other.
"""

import os
import re
import subprocess
from typing import Optional

import cv2

# ffprobe ships with neither imageio_ffmpeg nor this machine's PATH, so codec
# identification falls back to parsing `ffmpeg -i` and then to the container
# FourCC. All three are best-effort: a forensic report must never fail just
# because a codec could not be named.
_FFMPEG_CANDIDATES = (
    r'C:\Users\adhoc\AppData\Local\Programs\Python\Python313\Lib'
    r'\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe',
)
_FFPROBE_CANDIDATES = (
    r'C:\Users\adhoc\AppData\Local\Programs\Python\Python313\Lib'
    r'\site-packages\imageio_ffmpeg\binaries\ffprobe-win-x86_64-v7.1.exe',
)

_VIDEO_CODEC_RE = re.compile(r'Video:\s*([A-Za-z0-9_\-]+)')

# `ffmpeg -i` prints the pixel format as the bare token immediately before the
# resolution, optionally followed by a parenthesised modifier list:
#
#   Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive),
#          1920x1080 [...]
#
# The modifier list can be absent entirely on some builds, so it is optional.
# Anchoring on the WxH that follows is what keeps this from latching onto the
# codec name or the FourCC.
_PIX_FMT_RE = re.compile(
    r'Video:.*?,\s*([A-Za-z0-9_]+)\s*(?:\([^)]*\))?\s*,\s*\d{2,5}x\d{2,5}')

_FFMPEG_VERSION_RE = re.compile(r'ffmpeg version\s+(\S+)')

# Normalise the many names one codec travels under.
_CODEC_ALIASES = {
    'libx264': 'h264', 'avc1': 'h264', 'x264': 'h264',
    'libx265': 'h265', 'hev1': 'h265', 'hvc1': 'h265', 'x265': 'h265',
    'hevc': 'h265',
    'libvpx-vp9': 'vp9', 'vp09': 'vp9', 'libvpx': 'vp8',
    'mpeg4': 'mpeg4', 'mp4v': 'mpeg4', 'fmp4': 'mpeg4', 'xvid': 'mpeg4',
    'divx': 'mpeg4', 'avc3': 'h264',
}


def _first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def ffmpeg_path():
    p = _first_existing(_FFMPEG_CANDIDATES)
    if p:
        return p
    for name in ('ffmpeg.exe', 'ffmpeg'):
        from shutil import which
        w = which(name)
        if w:
            return w
    return None


def ffprobe_path():
    p = _first_existing(_FFPROBE_CANDIDATES)
    if p:
        return p
    from shutil import which
    return which('ffprobe') or which('ffprobe.exe')


def _normalise_codec(name):
    if not name:
        return None
    low = name.strip().lower()
    return _CODEC_ALIASES.get(low, low)


def _codec_via_ffprobe(path, ffprobe):
    try:
        r = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=codec_name',
                            '-of', 'default=noprint_wrappers=1:nokey=1', path],
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=30)
        return r.stdout.decode('utf-8', 'ignore').strip() or None
    except Exception:
        return None


def _codec_via_ffmpeg(path, ffmpeg):
    """`ffmpeg -i` prints the stream list on stderr and exits 1 for want of an
    output file; the stream line is still what we want."""
    try:
        r = subprocess.run([ffmpeg, '-hide_banner', '-i', path],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=60)
        text = r.stderr.decode('utf-8', 'ignore')
        m = _VIDEO_CODEC_RE.search(text)
        return m.group(1) if m else None
    except Exception:
        return None


def _codec_via_fourcc(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    if fourcc <= 0:
        return None
    name = ''.join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    name = name.strip('\x00 ')
    return name or None


def probe_codec(path):
    """Best-effort codec name, normalised; None when nothing could identify it.

    Tries ffprobe, then `ffmpeg -i` stderr, then the container FourCC.
    """
    ffprobe = ffprobe_path()
    if ffprobe:
        name = _codec_via_ffprobe(path, ffprobe)
        if name:
            return _normalise_codec(name)

    ffmpeg = ffmpeg_path()
    if ffmpeg:
        name = _codec_via_ffmpeg(path, ffmpeg)
        if name:
            return _normalise_codec(name)

    return _normalise_codec(_codec_via_fourcc(path))


def ffmpeg_version() -> Optional[str]:
    """`ffmpeg version X.Y` string, or None.

    Goes into every forensic report: a verdict is only reproducible if the
    decoder that produced it is named. The banner is suppressed by
    -hide_banner, so this needs its own call.
    """
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return None
    try:
        r = subprocess.run([ffmpeg, '-version'], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=30)
        text = r.stdout.decode('utf-8', 'ignore')
        m = _FFMPEG_VERSION_RE.search(text)
        return m.group(1) if m else None
    except Exception:
        return None


def _pix_fmt_via_ffprobe(path, ffprobe):
    try:
        r = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=pix_fmt',
                            '-of', 'default=noprint_wrappers=1:nokey=1', path],
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=30)
        return r.stdout.decode('utf-8', 'ignore').strip() or None
    except Exception:
        return None


def _pix_fmt_via_ffmpeg(path, ffmpeg):
    try:
        r = subprocess.run([ffmpeg, '-hide_banner', '-i', path],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=60)
        text = r.stderr.decode('utf-8', 'ignore')
        m = _PIX_FMT_RE.search(text)
        return m.group(1) if m else None
    except Exception:
        return None


def probe_pix_fmt(path: str) -> Optional[str]:
    """Pixel format of the first video stream, or None.

    Why this matters (risk R5): converting a losslessly re-encoded yuv420p
    clip through a BGR intermediate and back to yuv420p -- zero geometric
    change, zero compression change -- moves S_global from 0.000 to 0.104 and
    S_fused to 0.283, which is past the threshold on its own. A pair whose two
    sides sit in different colour pipelines carries that artefact for free, so
    L0 has to name the difference rather than let the score absorb it.
    """
    ffprobe = ffprobe_path()
    if ffprobe:
        v = _pix_fmt_via_ffprobe(path, ffprobe)
        if v:
            return v.lower()
    ffmpeg = ffmpeg_path()
    if ffmpeg:
        v = _pix_fmt_via_ffmpeg(path, ffmpeg)
        if v:
            return v.lower()
    return None


def frame_count(path: str) -> int:
    """Total decodable frames reported by the container; 0 when unopenable."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return n


def read_frames_at_indices(video_path: str, indices) -> list:
    """Seek-and-read the requested frame indices.

    Missing or undecodable positions yield None in place, callers must filter.
    Seeking is preferred over sequential scanning because the regional stage
    touches only ~5 frames of a possibly hour-long video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        frames.append(frame if ret else None)
    cap.release()
    return frames


def probe_container(path: str) -> dict:
    """Width, height, fps, frame count. Never raises."""
    out = {'width': 0, 'height': 0, 'fps': 0.0, 'frames': 0, 'opened': False}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return out
    out['opened'] = True
    out['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out['fps'] = float(cap.get(cv2.CAP_PROP_FPS))
    out['frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return out


# ── probe cache ───────────────────────────────────────────
# Probing is cheap per call and expensive per thousand calls. An audit that
# compares one reference against fifty candidates asks the same question about
# the same file fifty times, and each `ffmpeg -i` is a process creation --
# measured at 350-480 ms on Windows, where that dominates the work.
#
# The cache is keyed on identity AND content stamp (size + nanosecond mtime),
# so a file replaced on disk is re-probed rather than remembered wrongly. It
# is process-local on purpose: no cross-process state to go stale, and a
# worker that dies takes only its own cache with it.
_PROBE_CACHE: dict = {}
_PROBE_CACHE_LIMIT = 512
_PROBE_CACHE_ENABLED = os.environ.get('GCRR_NO_PROBE_CACHE', '') == ''


def _stat_key(path: str):
    try:
        st = os.stat(path)
    except Exception:
        return None            # unreadable path: never cache the answer
    return (os.path.normcase(os.path.abspath(path)),
            st.st_size, st.st_mtime_ns)


def _cached(kind: str, path: str, fn):
    if not _PROBE_CACHE_ENABLED:
        return fn(path)
    key = _stat_key(path)
    if key is None:
        return fn(path)
    ck = (kind,) + key
    hit = _PROBE_CACHE.get(ck)
    if hit is not None:
        return dict(hit) if isinstance(hit, dict) else hit
    val = fn(path)
    if len(_PROBE_CACHE) >= _PROBE_CACHE_LIMIT:
        _PROBE_CACHE.clear()
    _PROBE_CACHE[ck] = val
    return dict(val) if isinstance(val, dict) else val


def probe_container_cached(path: str) -> dict:
    """probe_container, remembered for the lifetime of this process."""
    return _cached('container', path, probe_container)


def probe_pix_fmt_cached(path: str):
    """probe_pix_fmt, remembered for the lifetime of this process."""
    return _cached('pixfmt', path, probe_pix_fmt)


def probe_codec_cached(path: str):
    """probe_codec, remembered for the lifetime of this process."""
    return _cached('codec', path, probe_codec)


def clear_probe_cache() -> None:
    """Drop every remembered probe. Tests use this to force a fresh read."""
    _PROBE_CACHE.clear()
