"""Processing paths for the conditional edge pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


FAST = "fast"
ROBUST = "robust"
DEFER = "defer"


def apply_gamma(image: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    """Apply gamma correction; gamma < 1 brightens low-light images."""
    gamma = max(float(gamma), 1e-3)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE on the luminance channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


@dataclass(frozen=True)
class FastPath:
    name: str = FAST
    enhancement_type: str = "none"

    def process(self, image: np.ndarray) -> np.ndarray:
        return image


@dataclass(frozen=True)
class RobustPath:
    enhancement_type: str = "clahe"
    gamma: float = 0.65
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    name: str = ROBUST

    def process(self, image: np.ndarray) -> np.ndarray:
        if self.enhancement_type == "gamma":
            return apply_gamma(image, self.gamma)
        if self.enhancement_type == "clahe":
            return apply_clahe(image, self.clahe_clip_limit, self.clahe_tile_grid_size)
        if self.enhancement_type in {"gamma+clahe", "clahe+gamma"}:
            return apply_clahe(
                apply_gamma(image, self.gamma),
                self.clahe_clip_limit,
                self.clahe_tile_grid_size,
            )
        return image


@dataclass(frozen=True)
class DeferPath:
    name: str = DEFER
    enhancement_type: str = "none"

    def process(self, image: np.ndarray) -> np.ndarray:
        return image


def path_for_name(path: str, robust_enhancement: str = "clahe"):
    if path == ROBUST:
        return RobustPath(enhancement_type=robust_enhancement)
    if path == DEFER:
        return DeferPath()
    return FastPath()
