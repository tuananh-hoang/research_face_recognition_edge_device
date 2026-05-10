"""
Module B — Image Quality Assessment (IQA)
Tính context vector C = [L, N] theo phương pháp trong báo cáo:
  L = Luminance (độ sáng trung bình, [0,1])
  N = Noise level (std của residual sau Gaussian blur, [0,1])
  bin_id = 'dark' | 'medium' | 'bright'

Kết hợp feature_norm từ Module A làm quality proxy tổng hợp q.
"""

import numpy as np
import cv2
from typing import Tuple, Dict


class IQAModule:
    def __init__(self,
                 t_dark: float = 0.30,
                 t_bright: float = 0.60,
                 blur_kernel: int = 5,
                 noise_norm_factor: float = 0.08,
                 norm_min: float = 5.0,
                 norm_max: float = 30.0):
        """
        t_dark   : L < t_dark  → 'dark'
        t_bright : L > t_bright → 'bright'
        blur_kernel : kernel size cho Gaussian blur khi tính noise residual
        noise_norm_factor : chia std(residual) để normalize về [0,1]
        norm_min/max : range của feature norm để normalize quality q
        """
        self.t_dark = t_dark
        self.t_bright = t_bright
        self.blur_kernel = blur_kernel
        self.noise_norm_factor = noise_norm_factor
        self.norm_min = norm_min
        self.norm_max = norm_max

    def compute_context(self, image: np.ndarray,
                        feature_norm: float = None) -> Dict:
        """
        Args:
            image: BGR image
            feature_norm: raw feature norm từ Module A (optional)
        Returns:
            dict với keys: L, N, bin_id, q, q_optical, q_norm
        """
        if image is None or image.size == 0:
            return self._empty_context()

        # ── Bước 1: Luminance L ──────────────────────────────
        # Chuyển sang grayscale qua kênh Y của YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)
        L = float(Y.mean() / 255.0)

        # ── Bước 2: Noise N (residual method) ────────────────
        k = self.blur_kernel
        blurred = cv2.GaussianBlur(Y, (k, k), 0)
        residual = Y - blurred
        raw_noise_std = float(residual.std())
        N = float(np.clip(raw_noise_std / (self.noise_norm_factor * 255.0), 0.0, 1.0))

        # ── Bước 3: Bin phân loại ─────────────────────────────
        if L < self.t_dark:
            bin_id = 'dark'
        elif L > self.t_bright:
            bin_id = 'bright'
        else:
            bin_id = 'medium'

        # ── Bước 4: Quality score q ───────────────────────────
        # q_optical: từ tín hiệu quang học (L và N)
        # Ánh sáng yếu VÀ nhiễu cao → quality thấp nhất (interaction term)
        q_optical = float(np.clip(L * (1.0 - N), 0.01, 1.0))

        # q_norm: từ feature norm (proxy theo AdaFace)
        if feature_norm is not None:
            q_norm = float(np.clip(
                (feature_norm - self.norm_min) / (self.norm_max - self.norm_min),
                0.01, 1.0
            ))
        else:
            q_norm = q_optical  # fallback nếu không có feature_norm

        # q tổng hợp: kết hợp cả hai signal
        alpha = 0.5  # trọng số giữa optical và norm signal
        q = float(alpha * q_optical + (1 - alpha) * q_norm)

        return {
            'L': L,
            'N': N,
            'bin_id': bin_id,
            'q': q,
            'q_optical': q_optical,
            'q_norm': q_norm,
            'raw_noise_std': raw_noise_std,
        }

    def _empty_context(self) -> Dict:
        return {
            'L': 0.0, 'N': 1.0, 'bin_id': 'dark',
            'q': 0.1, 'q_optical': 0.1, 'q_norm': 0.1,
            'raw_noise_std': 0.0
        }

    def calibrate_thresholds(self, bright_images, dark_images,
                              percentile_dark=75, percentile_bright=25):
        """
        Tự động calibrate T_dark và T_bright từ tập ảnh thật.
        Chạy sau khi có data để tránh hard-code.
        """
        bright_L = [self.compute_context(img)['L'] for img in bright_images if img is not None]
        dark_L = [self.compute_context(img)['L'] for img in dark_images if img is not None]

        if bright_L and dark_L:
            self.t_bright = float(np.percentile(bright_L, percentile_bright))
            self.t_dark = float(np.percentile(dark_L, percentile_dark))
            print(f"Calibrated: T_dark={self.t_dark:.3f}, T_bright={self.t_bright:.3f}")

        return self.t_dark, self.t_bright


# ─── Sanity test ────────────────────────────────────────────────
if __name__ == "__main__":
    iqa = IQAModule()

    # Ảnh sáng (uniform bright)
    bright_img = np.full((112, 112, 3), 200, dtype=np.uint8)
    ctx_bright = iqa.compute_context(bright_img, feature_norm=25.0)
    print(f"BRIGHT → L={ctx_bright['L']:.3f}, N={ctx_bright['N']:.3f}, "
          f"bin={ctx_bright['bin_id']}, q={ctx_bright['q']:.3f}")

    # Ảnh tối (uniform dark)
    dark_img = np.full((112, 112, 3), 30, dtype=np.uint8)
    ctx_dark = iqa.compute_context(dark_img, feature_norm=8.0)
    print(f"DARK   → L={ctx_dark['L']:.3f}, N={ctx_dark['N']:.3f}, "
          f"bin={ctx_dark['bin_id']}, q={ctx_dark['q']:.3f}")

    # Ảnh tối + nhiễu (worst case)
    dark_noisy = np.random.randint(10, 60, (112, 112, 3), dtype=np.uint8)
    ctx_noisy = iqa.compute_context(dark_noisy, feature_norm=6.0)
    print(f"DARK+N → L={ctx_noisy['L']:.3f}, N={ctx_noisy['N']:.3f}, "
          f"bin={ctx_noisy['bin_id']}, q={ctx_noisy['q']:.3f}")

    print("Module B OK")