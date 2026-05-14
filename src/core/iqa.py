"""
IQAModule — Image Quality Assessment.

Trả về:
  L (luminance), N (noise), bin_id, q (quality)
  hoặc dict context cho AdaptiveThreshold.

Usage:
    # Tuple form
    L, N, bin_id, q = IQAModule.compute(image)

    # Dict form (dùng với AdaptiveThreshold)
    ctx = IQAModule.compute_context(image, feature_norm=None)
    tau = AdaptiveThreshold().get_tau(ctx, 'interaction')
"""
import numpy as np
import cv2


class IQAModule:
    """
    Image Quality Assessment module.

    Metrics:
      L  : Luminance (0=đen, 1=trắng) — mean của Y channel
      N  : Noise (0=không nhiễu, 1=rất nhiễu) — std của residual
      q  : Quality = max(0, 1-N)
      bin_id : 'bright' | 'medium' | 'dark'

    Bin thresholds:
      dark   : L < 0.30
      medium : 0.30 <= L <= 0.60
      bright : L > 0.60
    """

    # Luminance bin thresholds
    L_DARK_THRESHOLD = 0.30
    L_BRIGHT_THRESHOLD = 0.60

    @staticmethod
    def compute(img: np.ndarray) -> tuple[float, float, str, float]:
        """
        Compute IQA metrics as tuple.

        Returns
        -------
        L      : float  — luminance [0, 1]
        N      : float  — noise [0, 1]
        bin_id : str    — 'bright' | 'medium' | 'dark'
        q      : float  — quality = max(0, 1-N)
        """
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        L = float(np.mean(ycrcb[:, :, 0])) / 255.0

        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        diff = img.astype(np.float32) - blurred.astype(np.float32)
        N = float(np.std(diff)) / 255.0

        if L < IQAModule.L_DARK_THRESHOLD:
            bin_id = 'dark'
        elif L > IQAModule.L_BRIGHT_THRESHOLD:
            bin_id = 'bright'
        else:
            bin_id = 'medium'

        q = float(max(0.0, min(1.0, 1.0 - N)))
        return L, N, bin_id, q

    @staticmethod
    def compute_context(
        img: np.ndarray,
        feature_norm: float | None = None,
    ) -> dict:
        """
        Compute IQA metrics as dict (for AdaptiveThreshold).

        Parameters
        ----------
        img          : np.ndarray — BGR image
        feature_norm : float | None — optional embedding norm (used as q override)

        Returns
        -------
        dict with keys: L, N, q, bin_id
        """
        L, N, bin_id, q = IQAModule.compute(img)

        # Cho phép override q bằng feature_norm nếu có
        if feature_norm is not None:
            # Normalize feature_norm: buffalo_sc embeddings có norm ~20-30
            q = float(np.clip(feature_norm / 30.0, 0.0, 1.0))

        return {
            'L': L,
            'N': N,
            'q': q,
            'bin_id': bin_id,
        }

    @staticmethod
    def bin_from_L(L: float) -> str:
        """Xác định bin_id từ luminance."""
        if L < IQAModule.L_DARK_THRESHOLD:
            return 'dark'
        elif L > IQAModule.L_BRIGHT_THRESHOLD:
            return 'bright'
        else:
            return 'medium'