"""41-dimensional feature extraction, extracted verbatim (numerically) from the
reference implementation.

Source of truth:
    C:\\baidunetdiskdownload\\lightweight\\lightweight-final.py
    class VideoRecompressionDetector, L39-L380

This module reorganises that code into an injectable, side-effect-free
extractor. Every arithmetic path is preserved bit-for-bit; the parity suite in
tests/test_baseline_parity.py asserts that against the original module.
"""

import cv2
import numpy as np
import pywt
from scipy.fft import dctn
from scipy.spatial.distance import euclidean

from .constants import FEATURE_KEYS


class FeatureExtractor:
    """Extracts the seven feature groups that make up the 41-D global vector.

    Feature groups and dimensions:
        pixel_contrast 1, color_moments 9, motion 6, dct 6,
        block 8, residual 5, texture 6                -> 41
    """

    def __init__(self, sample_interval: int = 60, use_wavelet: bool = True):
        self.sample_interval = sample_interval
        self.use_wavelet = use_wavelet

    # ── 1. pixel contrast (1-D) ────────────────────────────
    def extract_pixel_contrast(self, frame_gray: np.ndarray) -> float:
        if self.use_wavelet:
            coeffs = pywt.dwt2(frame_gray, 'db1')
            cA, (cH, cV, cD) = coeffs
            high_freq_mean = (np.mean(np.abs(cH)) + np.mean(np.abs(cV))
                              + np.mean(np.abs(cD))) / 3
            low_freq_mean = np.mean(np.abs(cA))
            delta = high_freq_mean / (low_freq_mean + 1e-6)
            R = 0.05
            return float(delta) if delta > R else 0.0
        return cv2.Laplacian(frame_gray, cv2.CV_64F).var()

    # ── 2. colour moments (9-D) ────────────────────────────
    def extract_color_moments(self, frame_bgr: np.ndarray) -> np.ndarray:
        moments = []
        for channel in cv2.split(frame_bgr):
            mean = np.mean(channel)
            std = np.std(channel)
            skew = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-6)
            moments.extend([mean, std, skew])
        return np.array(moments, dtype=np.float32)

    # ── 3. motion (6-D) ────────────────────────────────────
    def extract_motion_vectors(self, prev_gray: np.ndarray,
                               curr_gray: np.ndarray) -> np.ndarray:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        fx, fy = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(fx ** 2 + fy ** 2).flatten()

        mag_mean = float(np.mean(magnitude))
        mag_var = float(np.var(magnitude))
        zero_ratio = float(np.sum(magnitude < 0.5) / max(len(magnitude), 1))

        mag_hist, _ = np.histogram(magnitude, bins=32, range=(0, 20), density=False)
        mag_hist = mag_hist.astype(np.float64) + 1e-10
        mag_hist /= mag_hist.sum()
        mag_entropy = float(-np.sum(mag_hist * np.log2(mag_hist)))

        angle = np.arctan2(fy.flatten(), fx.flatten())
        angle_hist, _ = np.histogram(angle, bins=36, range=(-np.pi, np.pi),
                                     density=False)
        angle_hist = angle_hist.astype(np.float64) + 1e-10
        angle_hist /= angle_hist.sum()
        angle_entropy = float(-np.sum(angle_hist * np.log2(angle_hist)))

        mag_p95 = float(np.percentile(magnitude, 95))

        return np.array([mag_mean, mag_var, zero_ratio, mag_entropy,
                         angle_entropy, mag_p95], dtype=np.float32)

    # ── 4. DCT histogram (6-D) ─────────────────────────────
    def extract_dct_histogram_features(self, frame_gray: np.ndarray) -> np.ndarray:
        h, w = frame_gray.shape
        bh, bw = h // 8, w // 8
        if bh < 1 or bw < 1:
            return np.zeros(6, dtype=np.float32)

        img = frame_gray[:bh * 8, :bw * 8].astype(np.float32)
        blocks = img.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8).copy()
        dct_blocks = dctn(blocks, type=2, norm='ortho', axes=(-2, -1))

        ac_blocks = dct_blocks.copy()
        ac_blocks[:, 0, 0] = 0
        ac_flat = ac_blocks.flatten()

        zero_ratio = np.sum(np.abs(ac_flat) < 0.5) / max(len(ac_flat), 1)

        hf_energy = np.sum(dct_blocks[:, 4:, 4:] ** 2)
        total_energy = np.sum(dct_blocks ** 2) + 1e-6
        hf_ratio = hf_energy / total_energy

        hist, _ = np.histogram(ac_flat, bins=256, range=(-128, 128), density=True)
        hist = hist + 1e-10
        entropy = -np.sum(hist * np.log2(hist))

        if len(ac_flat) > 3 and np.std(ac_flat) > 1e-6:
            mean_val = np.mean(ac_flat)
            std_val = np.std(ac_flat)
            kurtosis = np.mean(((ac_flat - mean_val) / std_val) ** 4) - 3
        else:
            kurtosis = 0.0

        block_ac_means = np.mean(np.abs(ac_blocks.reshape(-1, 64)[:, 1:]), axis=1)
        inter_block_var = np.var(block_ac_means)

        mean_ac = np.mean(np.abs(ac_flat))

        return np.array([zero_ratio, hf_ratio, entropy, kurtosis,
                         inter_block_var, mean_ac], dtype=np.float32)

    # ── 5. block artifacts (8-D) ───────────────────────────
    def extract_block_artifact_features(self, frame_gray: np.ndarray) -> np.ndarray:
        h, w = frame_gray.shape
        img = frame_gray.astype(np.float32)
        features = []

        for block_size in [8, 16]:
            bh, bw = h // block_size, w // block_size
            if bh < 2 or bw < 2:
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue

            cropped = img[:bh * block_size, :bw * block_size]

            h_diff = np.abs(np.diff(cropped, axis=1))
            v_diff = np.abs(np.diff(cropped, axis=0))

            h_bnd_cols = np.array([k * block_size - 1 for k in range(1, bw)])
            h_mask = np.zeros(h_diff.shape[1], dtype=bool)
            h_mask[h_bnd_cols] = True

            h_bnd = h_diff[:, h_mask].flatten()
            h_nonbnd = h_diff[:, ~h_mask].flatten()

            h_bnd_mean = np.mean(h_bnd) if len(h_bnd) > 0 else 0.0
            h_nonbnd_mean = np.mean(h_nonbnd) if len(h_nonbnd) > 0 else 1.0
            h_ratio = h_bnd_mean / (h_nonbnd_mean + 1e-6)

            v_bnd_rows = np.array([k * block_size - 1 for k in range(1, bh)])
            v_mask = np.zeros(v_diff.shape[0], dtype=bool)
            v_mask[v_bnd_rows] = True

            v_bnd = v_diff[v_mask, :].flatten()
            v_nonbnd = v_diff[~v_mask, :].flatten()

            v_bnd_mean = np.mean(v_bnd) if len(v_bnd) > 0 else 0.0
            v_nonbnd_mean = np.mean(v_nonbnd) if len(v_nonbnd) > 0 else 1.0
            v_ratio = v_bnd_mean / (v_nonbnd_mean + 1e-6)

            comb_bnd = np.concatenate([h_bnd, v_bnd])
            comb_nonbnd = np.concatenate([h_nonbnd, v_nonbnd])
            comb_ratio = np.mean(comb_bnd) / (np.mean(comb_nonbnd) + 1e-6)
            comb_abs = np.mean(comb_bnd)

            features.extend([h_ratio, v_ratio, comb_ratio, comb_abs])

        return np.array(features, dtype=np.float32)

    # ── 6. residual energy (5-D) ───────────────────────────
    def extract_residual_energy_features(self, prev_gray: np.ndarray,
                                         curr_gray: np.ndarray) -> np.ndarray:
        h, w = curr_gray.shape
        prev_f = prev_gray.astype(np.float32)
        curr_f = curr_gray.astype(np.float32)

        raw_diff = curr_f - prev_f
        raw_l2 = np.sqrt(np.mean(raw_diff ** 2))

        if raw_l2 > 1e-6:
            raw_h, raw_w = raw_diff.shape
            min_dim = min(raw_h, raw_w)
            if min_dim >= 4:
                coeffs = pywt.dwt2(raw_diff, 'db1')
                _, (cH, cV, cD) = coeffs
                raw_hf = (np.mean(cH ** 2) + np.mean(cV ** 2)
                          + np.mean(cD ** 2)) / (raw_l2 ** 2 + 1e-6)
            else:
                raw_hf = 0.0
        else:
            raw_hf = 0.0

        if h >= 2 and w >= 2:
            corr = cv2.phaseCorrelate(prev_f, curr_f)
            dx, dy = corr[0] if corr[0] is not None else (0.0, 0.0)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned = cv2.warpAffine(prev_f, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mc_diff = curr_f - aligned
            mc_l2 = np.sqrt(np.mean(mc_diff ** 2))

            if mc_l2 > 1e-6:
                min_dim = min(h, w)
                if min_dim >= 4:
                    coeffs = pywt.dwt2(mc_diff, 'db1')
                    _, (cH, cV, cD) = coeffs
                    mc_hf = (np.mean(cH ** 2) + np.mean(cV ** 2)
                             + np.mean(cD ** 2)) / (mc_l2 ** 2 + 1e-6)
                else:
                    mc_hf = 0.0
            else:
                mc_hf = 0.0
        else:
            mc_l2 = 0.0
            mc_hf = 0.0

        decay_idx = 1.0 - (mc_l2 / (raw_l2 + 1e-6)) if raw_l2 > 1e-6 else 0.0
        decay_idx = max(0.0, min(decay_idx, 1.0))

        return np.array([raw_l2, raw_hf, mc_l2, mc_hf, decay_idx], dtype=np.float32)

    # ── 7. texture consistency (6-D) ───────────────────────
    def extract_texture_consistency_features(self, frame_gray: np.ndarray) -> np.ndarray:
        img = frame_gray.astype(np.float32)
        h, w = img.shape
        if h < 32 or w < 32:
            return np.zeros(6, dtype=np.float32)

        g0 = img
        g1 = cv2.pyrDown(g0)
        g2 = cv2.pyrDown(g1)
        g3 = cv2.pyrDown(g2)

        up1 = cv2.pyrUp(g1, dstsize=(w, h))
        up2 = cv2.pyrUp(g2, dstsize=(g1.shape[1], g1.shape[0]))
        up3 = cv2.pyrUp(g3, dstsize=(g2.shape[1], g2.shape[0]))

        l0 = g0 - up1
        l1 = g1 - up2
        l2 = g2 - up3

        e0 = float(np.mean(l0 ** 2))
        e1 = float(np.mean(l1 ** 2))
        e2 = float(np.mean(l2 ** 2))

        l0_small = cv2.resize(l0, (g1.shape[1], g1.shape[0]))
        l0_flat = l0_small.flatten()
        l1_flat = l1.flatten()
        corr_01 = float(np.corrcoef(l0_flat, l1_flat)[0, 1]) \
            if np.std(l0_flat) > 1e-6 and np.std(l1_flat) > 1e-6 else 0.0

        l1_small = cv2.resize(l1, (g2.shape[1], g2.shape[0]))
        l1_flat2 = l1_small.flatten()
        l2_flat = l2.flatten()
        corr_12 = float(np.corrcoef(l1_flat2, l2_flat)[0, 1]) \
            if np.std(l1_flat2) > 1e-6 and np.std(l2_flat) > 1e-6 else 0.0

        decay = e0 / (e1 + 1e-6) if e1 > 1e-6 else 0.0

        return np.array([e0, e1, e2, corr_01, corr_12, decay], dtype=np.float32)

    # ── whole-video driver ─────────────────────────────────
    def extract_video_features(self, video_path: str,
                               max_frames: int = None) -> dict:
        """Extract features over the video.

        max_frames caps how many frames are decoded. The reference pipeline
        always ran on a 10-15 s clip; decoding a full 10-minute 1080p source
        costs 10-30 s per video, so callers should bound it. max_frames=None
        reproduces the reference behaviour exactly.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        features = {k: [] for k in FEATURE_KEYS}
        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % self.sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                features['pixel_contrast'].append(self.extract_pixel_contrast(gray))
                features['color_moments'].append(self.extract_color_moments(frame))
                features['dct'].append(self.extract_dct_histogram_features(gray))
                features['block'].append(self.extract_block_artifact_features(gray))
                features['texture'].append(self.extract_texture_consistency_features(gray))
                if prev_gray is not None:
                    features['motion'].append(
                        self.extract_motion_vectors(prev_gray, gray))
                    features['residual'].append(
                        self.extract_residual_energy_features(prev_gray, gray))
                prev_gray = gray.copy()
            frame_idx += 1

        cap.release()
        for k in features:
            features[k] = np.array(features[k]) if len(features[k]) > 0 else np.array([])
        return features
