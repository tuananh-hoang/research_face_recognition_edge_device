"""Quality/context helpers for conditional pipeline experiments."""

from __future__ import annotations

import cv2
import numpy as np


def compute_brightness(image: np.ndarray) -> float:
    """Return normalized luminance L in [0, 1]."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return float(np.mean(ycrcb[:, :, 0])) / 255.0


def compute_noise(image: np.ndarray) -> float:
    """Estimate high-frequency residual noise N in [0, 1]."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    blurred = cv2.GaussianBlur(y, (5, 5), 0)
    return float(np.std(y - blurred)) / 255.0


def assign_condition_bin(
    L: float,
    dark_threshold: float = 0.30,
    bright_threshold: float = 0.60,
) -> str:
    """Map luminance to dark/medium/bright bins."""
    if L < dark_threshold:
        return "dark"
    if L > bright_threshold:
        return "bright"
    return "medium"


def compute_context(
    image: np.ndarray,
    det_score: float | None = None,
    dark_threshold: float = 0.30,
    bright_threshold: float = 0.60,
) -> dict:
    """Return the context vector C(x) used by policy and thresholds."""
    L = compute_brightness(image)
    N = compute_noise(image)
    q = float(det_score) if det_score is not None else float(max(0.0, min(1.0, 1.0 - N)))
    return {
        "L": L,
        "N": N,
        "q": q,
        "bin_id": assign_condition_bin(L, dark_threshold, bright_threshold),
    }
