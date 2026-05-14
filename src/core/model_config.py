"""Face model configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass


MODEL_ALIASES = {
    "mobile": "buffalo_sc",
    "mobilefacenet": "buffalo_sc",
    "mobile_face_net": "buffalo_sc",
    "mobile-face-net": "buffalo_sc",
    "mbf": "buffalo_sc",
    "edge": "buffalo_sc",
    "buffalo_sc": "buffalo_sc",
    "buffalo_s": "buffalo_s",
    "buffalo_m": "buffalo_m",
    "buffalo_l": "buffalo_l",
}

MODEL_DESCRIPTIONS = {
    "buffalo_sc": "MobileFaceNet-style MBF@WebFace600K recognition pack for edge/mobile use",
    "buffalo_s": "small InsightFace pack for edge/mobile use",
    "buffalo_m": "medium InsightFace pack",
    "buffalo_l": "large/server-oriented InsightFace pack",
}


@dataclass(frozen=True)
class FaceModelConfig:
    requested_name: str
    model_name: str
    det_size: tuple[int, int]
    description: str


def resolve_model_name(name: str | None = None) -> str:
    requested = (name or os.getenv("FACE_MODEL_NAME") or "mobilefacenet").strip()
    key = requested.lower()
    return MODEL_ALIASES.get(key, requested)


def parse_det_size(value: str | None = None) -> tuple[int, int]:
    raw = (value or os.getenv("FACE_DET_SIZE") or "320,320").lower().replace("x", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        return (320, 320)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return (320, 320)


def get_face_model_config(name: str | None = None, det_size: str | tuple[int, int] | None = None) -> FaceModelConfig:
    requested = (name or os.getenv("FACE_MODEL_NAME") or "mobilefacenet").strip()
    model_name = resolve_model_name(requested)
    if isinstance(det_size, tuple):
        resolved_det_size = det_size
    else:
        resolved_det_size = parse_det_size(det_size)
    return FaceModelConfig(
        requested_name=requested,
        model_name=model_name,
        det_size=resolved_det_size,
        description=MODEL_DESCRIPTIONS.get(model_name, "custom InsightFace model pack"),
    )
