"""Evaluation loop for conditional pipeline experiments."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

import cv2
import numpy as np

from .paths import DEFER, ROBUST, path_for_name
from .quality import compute_context

try:
    import psutil
except ImportError:
    psutil = None


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    label: int
    person_id: str = ""
    image1_path: str | None = None
    image2_path: str | None = None
    sim: float | None = None
    L: float | None = None
    N: float | None = None
    q: float | None = None
    bin_id: str | None = None


@dataclass(frozen=True)
class MethodConfig:
    name: str
    policy: object
    threshold_policy: object


class ConditionalEvaluator:
    """Runs method x pair evaluation and returns per-sample log rows."""

    def __init__(
        self,
        embedder=None,
        robust_enhancement: str = "clahe",
        max_image_dim: int = 640,
        synthetic_latency_ms: dict[str, float] | None = None,
        synthetic_robust_delta: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.robust_enhancement = robust_enhancement
        self.max_image_dim = max_image_dim
        self.synthetic_latency_ms = synthetic_latency_ms or {
            "fast": 2.0,
            "robust": 8.0,
            "defer": 0.3,
        }
        self.synthetic_robust_delta = float(synthetic_robust_delta)
        self._process = psutil.Process() if psutil is not None else None

    def evaluate(self, records: list[PairRecord], methods: list[MethodConfig]) -> list[dict]:
        rows: list[dict] = []
        for method in methods:
            for record in records:
                rows.append(self.evaluate_one(record, method))
        return rows

    def evaluate_one(self, record: PairRecord, method: MethodConfig) -> dict:
        start = time.perf_counter()
        context, img1, img2, context_error = self._context_and_images(record)
        selected_path = method.policy.select_path(context)
        enhancement_type = "none"

        base_row = {
            "image_id": record.pair_id,
            "person_id": record.person_id,
            "method_name": method.name,
            "selected_path": selected_path,
            "enhancement_type": enhancement_type,
            "is_genuine": int(record.label),
            "brightness_L": float(context.get("L", 0.0)),
            "noise_N": float(context.get("N", 0.0)),
            "det_score_q": float(context.get("q", 0.0)),
            "condition_bin": str(context.get("bin_id", "medium")),
            "ram_mb": self._rss_mb(),
        }

        if context_error:
            return self._deferred_row(
                base_row,
                start,
                reason=context_error,
                threshold_policy=method.threshold_policy,
            )

        if selected_path == DEFER:
            return self._deferred_row(
                base_row,
                start,
                reason="policy_defer",
                threshold_policy=method.threshold_policy,
            )

        if selected_path == ROBUST:
            enhancement_type = self.robust_enhancement

        if record.sim is not None:
            sim = self._synthetic_similarity(record, selected_path)
            latency_ms = self.synthetic_latency_ms.get(selected_path, 2.0)
        else:
            if self.embedder is None:
                return self._deferred_row(
                    base_row,
                    start,
                    reason="missing_embedder",
                    threshold_policy=method.threshold_policy,
                )
            if img1 is None or img2 is None:
                return self._deferred_row(
                    base_row,
                    start,
                    reason="missing_image",
                    threshold_policy=method.threshold_policy,
                )

            processor = path_for_name(selected_path, self.robust_enhancement)
            enhancement_type = processor.enhancement_type
            img1_proc = processor.process(img1)
            img2_proc = processor.process(img2)

            emb1, _ = self.embedder.get_embedding(img1_proc)
            emb2, _ = self.embedder.get_embedding(img2_proc)
            if emb1 is None or emb2 is None:
                row = dict(base_row)
                row["enhancement_type"] = enhancement_type
                return self._deferred_row(
                    row,
                    start,
                    reason="face_not_detected",
                    threshold_policy=method.threshold_policy,
                )

            sim = float(np.dot(emb1, emb2))
            latency_ms = (time.perf_counter() - start) * 1000.0

        decision_info = self._decision_info(method.threshold_policy, context, selected_path, sim)
        row = dict(base_row)
        row.update(
            {
                "enhancement_type": enhancement_type,
                "similarity_score": float(sim),
                **decision_info,
                "latency_ms": float(latency_ms),
                "ram_mb": self._rss_mb(),
            }
        )
        return row

    def _context_and_images(self, record: PairRecord):
        if record.sim is not None:
            context = {
                "L": float(record.L if record.L is not None else 0.5),
                "N": float(record.N if record.N is not None else 0.0),
                "q": float(record.q if record.q is not None else 1.0),
                "bin_id": str(record.bin_id or "medium"),
            }
            return context, None, None, ""

        img1 = self._read_image(record.image1_path)
        img2 = self._read_image(record.image2_path)
        if img1 is None or img2 is None:
            context = {
                "L": float(record.L if record.L is not None else 0.0),
                "N": float(record.N if record.N is not None else 1.0),
                "q": float(record.q if record.q is not None else 0.0),
                "bin_id": str(record.bin_id or "dark"),
            }
            return context, img1, img2, "missing_image"

        context = compute_context(img2)
        if record.bin_id is not None:
            # Keep the dataset condition label stable for condition-specific summaries.
            context["bin_id"] = record.bin_id
        return context, img1, img2, ""

    def _read_image(self, raw_path: str | None):
        if raw_path is None:
            return None
        path = Path(raw_path)
        image = cv2.imread(str(path))
        if image is None:
            return None
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if self.max_image_dim and max_dim > self.max_image_dim:
            scale = self.max_image_dim / max_dim
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def _synthetic_similarity(self, record: PairRecord, selected_path: str) -> float:
        sim = float(record.sim if record.sim is not None else 0.0)
        if selected_path == ROBUST and self.synthetic_robust_delta:
            condition = str(record.bin_id or "medium")
            if condition == "dark":
                sim += self.synthetic_robust_delta if record.label == 1 else self.synthetic_robust_delta * 0.25
            elif condition == "medium":
                sim += self.synthetic_robust_delta * 0.25 if record.label == 1 else 0.0
        return float(np.clip(sim, -1.0, 1.0))

    def _decision_info(self, threshold_policy: object, context: dict, selected_path: str, sim: float) -> dict:
        if hasattr(threshold_policy, "decide"):
            info = dict(threshold_policy.decide(context, selected_path, sim))
            info.setdefault("threshold", info.get("threshold_accept", ""))
            info.setdefault("threshold_accept", info.get("threshold", ""))
            info.setdefault("threshold_reject", "")
            info.setdefault("far_budget", getattr(threshold_policy, "far_budget", ""))
            info.setdefault("defer_margin", getattr(threshold_policy, "defer_margin", ""))
            info.setdefault("deferred", info.get("decision") == "defer")
            info.setdefault("defer_reason", "uncertain_score" if info.get("deferred") else "")
            return info

        threshold = float(threshold_policy.get_threshold(context, selected_path))
        decision = "accept" if sim >= threshold else "reject"
        return {
            "threshold": threshold,
            "threshold_accept": threshold,
            "threshold_reject": "",
            "far_budget": "",
            "defer_margin": "",
            "decision": decision,
            "deferred": False,
            "defer_reason": "",
        }

    def _deferred_row(
        self,
        base_row: dict,
        start: float,
        reason: str,
        threshold_policy: object | None = None,
    ) -> dict:
        row = dict(base_row)
        path = row.get("selected_path", DEFER)
        latency_ms = self.synthetic_latency_ms.get(path, 0.3)
        if row.get("similarity_score") is None and row.get("image_id", "").startswith("syn_"):
            measured = latency_ms
        else:
            measured = (time.perf_counter() - start) * 1000.0
        row.update(
            {
                "similarity_score": "",
                "threshold": "",
                "threshold_accept": "",
                "threshold_reject": "",
                "far_budget": getattr(threshold_policy, "far_budget", ""),
                "defer_margin": getattr(threshold_policy, "defer_margin", ""),
                "decision": "defer",
                "latency_ms": float(measured),
                "ram_mb": self._rss_mb(),
                "deferred": True,
                "defer_reason": reason,
            }
        )
        return row

    def _rss_mb(self) -> float:
        if self._process is None:
            return 0.0
        try:
            return float(self._process.memory_info().rss / 1024 / 1024)
        except Exception:
            return 0.0
