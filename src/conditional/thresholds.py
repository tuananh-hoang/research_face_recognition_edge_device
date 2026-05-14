"""Threshold policies tau_r(C) for conditional pipeline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.threshold import (
    formula_bin_specific,
    formula_fixed,
    formula_interaction,
    formula_linear,
)


def _ctx_values(context: dict) -> tuple[str, float, float, float]:
    return (
        str(context.get("bin_id", "medium")),
        float(context.get("L", 0.5)),
        float(context.get("N", 0.0)),
        float(context.get("q", 1.0)),
    )


def _safe_float(value) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_deferred(row: dict) -> bool:
    value = row.get("deferred", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "defer"}


def _candidate_thresholds(scores: Iterable[float]) -> list[float]:
    unique_scores = sorted({float(score) for score in scores})
    if not unique_scores:
        return [1.0]
    epsilon = 1e-6
    return unique_scores + [unique_scores[-1] + epsilon]


def _risk_stats_for_threshold(
    rows: list[dict],
    threshold: float,
    defer_margin: float = 0.0,
    enable_defer: bool = False,
) -> dict:
    genuine_total = 0
    impostor_total = 0
    genuine_active = 0
    impostor_active = 0
    genuine_rejected = 0
    impostor_accepted = 0

    threshold_reject = threshold - float(defer_margin)
    for row in rows:
        score = _safe_float(row.get("similarity_score"))
        if score is None or _is_deferred(row):
            continue

        is_genuine = int(row.get("is_genuine", 0)) == 1
        if is_genuine:
            genuine_total += 1
        else:
            impostor_total += 1

        if enable_defer and threshold_reject <= score < threshold:
            continue

        if is_genuine:
            genuine_active += 1
            if score < threshold:
                genuine_rejected += 1
        else:
            impostor_active += 1
            if score >= threshold:
                impostor_accepted += 1

    far = impostor_accepted / max(1, impostor_active)
    frr = genuine_rejected / max(1, genuine_active)
    return {
        "n_calibration_genuine": genuine_total,
        "n_calibration_impostor": impostor_total,
        "n_calibration_active_genuine": genuine_active,
        "n_calibration_active_impostor": impostor_active,
        "calibration_FAR": float(far),
        "calibration_FRR": float(frr),
    }


def _select_risk_threshold(
    rows: list[dict],
    far_budget: float,
    defer_margin: float = 0.0,
    enable_defer: bool = False,
    default_threshold: float = 1.0,
) -> tuple[float, dict]:
    active_scores = [
        score
        for row in rows
        for score in [_safe_float(row.get("similarity_score"))]
        if score is not None and not _is_deferred(row)
    ]
    if not active_scores:
        stats = _risk_stats_for_threshold(rows, default_threshold, defer_margin, enable_defer)
        return float(default_threshold), stats

    for threshold in _candidate_thresholds(active_scores):
        stats = _risk_stats_for_threshold(rows, threshold, defer_margin, enable_defer)
        if stats["calibration_FAR"] <= far_budget:
            return float(threshold), stats

    threshold = float(max(active_scores) + 1e-6)
    stats = _risk_stats_for_threshold(rows, threshold, defer_margin, enable_defer)
    return threshold, stats


@dataclass(frozen=True)
class FixedThreshold:
    tau: float = 0.44
    name: str = "fixed"

    def get_threshold(self, context: dict, path: str) -> float:
        bin_id, L, N, q = _ctx_values(context)
        return float(formula_fixed(bin_id, L, N, q) if self.tau == 0.44 else self.tau)


@dataclass(frozen=True)
class BinSpecificThreshold:
    thresholds: dict[str, float] = field(
        default_factory=lambda: {"bright": 0.48, "medium": 0.42, "dark": 0.35}
    )
    name: str = "bin"

    def get_threshold(self, context: dict, path: str) -> float:
        bin_id, L, N, q = _ctx_values(context)
        if self.thresholds == {"bright": 0.48, "medium": 0.42, "dark": 0.35}:
            return float(formula_bin_specific(bin_id, L, N, q))
        return float(self.thresholds.get(bin_id, self.thresholds.get("medium", 0.42)))


@dataclass(frozen=True)
class PathSpecificBinThreshold:
    """Path-conditioned bin thresholds.

    Defaults keep robust dark threshold higher than fast dark because enhancement
    should recover some quality instead of just lowering tau.
    """

    thresholds: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "fast": {"bright": 0.48, "medium": 0.42, "dark": 0.35},
            "robust": {"bright": 0.50, "medium": 0.44, "dark": 0.38},
        }
    )
    fallback: dict[str, float] = field(
        default_factory=lambda: {"bright": 0.48, "medium": 0.42, "dark": 0.35}
    )
    name: str = "path_bin"

    def get_threshold(self, context: dict, path: str) -> float:
        bin_id = str(context.get("bin_id", "medium"))
        by_path = self.thresholds.get(path, self.fallback)
        return float(by_path.get(bin_id, by_path.get("medium", 0.42)))


@dataclass(frozen=True)
class LinearThreshold:
    name: str = "linear"

    def get_threshold(self, context: dict, path: str) -> float:
        bin_id, L, N, q = _ctx_values(context)
        return float(formula_linear(bin_id, L, N, q))


@dataclass(frozen=True)
class InteractionThreshold:
    name: str = "interaction"

    def get_threshold(self, context: dict, path: str) -> float:
        bin_id, L, N, q = _ctx_values(context)
        return float(formula_interaction(bin_id, L, N, q))


@dataclass(frozen=True)
class RiskConstrainedBinThreshold:
    """Thresholds calibrated to satisfy a FAR budget on calibration rows."""

    thresholds: dict[str, float]
    calibration_stats: dict[str, dict]
    far_budget: float
    fallback_threshold: float
    defer_margin: float = 0.0
    enable_defer: bool = False
    min_impostors: int = 1
    name: str = "risk_bin"

    @classmethod
    def from_rows(
        cls,
        rows: list[dict],
        far_budget: float,
        defer_margin: float = 0.0,
        enable_defer: bool = False,
        condition_bins: tuple[str, ...] = ("bright", "medium", "dark"),
        min_impostors: int = 1,
    ) -> "RiskConstrainedBinThreshold":
        active_rows = [row for row in rows if not _is_deferred(row)]
        fallback_threshold, fallback_stats = _select_risk_threshold(
            active_rows,
            far_budget=float(far_budget),
            defer_margin=defer_margin,
            enable_defer=enable_defer,
        )

        thresholds: dict[str, float] = {}
        stats_by_bin: dict[str, dict] = {
            "global": {
                **fallback_stats,
                "threshold_accept": fallback_threshold,
                "threshold_reject": (
                    fallback_threshold - float(defer_margin) if enable_defer else ""
                ),
                "used_global_fallback": False,
            }
        }

        for bin_id in condition_bins:
            bin_rows = [
                row
                for row in active_rows
                if str(row.get("condition_bin", "medium")) == bin_id
            ]
            impostors = [
                row
                for row in bin_rows
                if int(row.get("is_genuine", 0)) == 0
                and _safe_float(row.get("similarity_score")) is not None
            ]

            if len(impostors) < min_impostors:
                threshold = fallback_threshold
                stats = _risk_stats_for_threshold(
                    bin_rows,
                    threshold,
                    defer_margin=defer_margin,
                    enable_defer=enable_defer,
                )
                used_global = True
            else:
                threshold, stats = _select_risk_threshold(
                    bin_rows,
                    far_budget=float(far_budget),
                    defer_margin=defer_margin,
                    enable_defer=enable_defer,
                    default_threshold=fallback_threshold,
                )
                used_global = False

            thresholds[bin_id] = threshold
            stats_by_bin[bin_id] = {
                **stats,
                "threshold_accept": threshold,
                "threshold_reject": threshold - float(defer_margin) if enable_defer else "",
                "used_global_fallback": used_global,
            }

        return cls(
            thresholds=thresholds,
            calibration_stats=stats_by_bin,
            far_budget=float(far_budget),
            fallback_threshold=fallback_threshold,
            defer_margin=float(defer_margin),
            enable_defer=bool(enable_defer),
            min_impostors=int(min_impostors),
        )

    def get_threshold(self, context: dict, path: str) -> float:
        bin_id = str(context.get("bin_id", "medium"))
        return float(self.thresholds.get(bin_id, self.fallback_threshold))

    def get_threshold_reject(self, context: dict, path: str) -> float | str:
        if not self.enable_defer:
            return ""
        return float(self.get_threshold(context, path) - self.defer_margin)

    def decide(self, context: dict, path: str, sim: float) -> dict:
        threshold_accept = self.get_threshold(context, path)
        threshold_reject = self.get_threshold_reject(context, path)

        if sim >= threshold_accept:
            decision = "accept"
            deferred = False
            reason = ""
        elif self.enable_defer and float(threshold_reject) <= sim < threshold_accept:
            decision = "defer"
            deferred = True
            reason = "uncertain_score"
        else:
            decision = "reject"
            deferred = False
            reason = ""

        return {
            "decision": decision,
            "deferred": deferred,
            "defer_reason": reason,
            "threshold": threshold_accept,
            "threshold_accept": threshold_accept,
            "threshold_reject": threshold_reject,
            "far_budget": self.far_budget,
            "defer_margin": self.defer_margin if self.enable_defer else "",
        }

    def calibration_rows(self, method_name: str) -> list[dict]:
        rows: list[dict] = []
        for condition_bin in sorted(self.calibration_stats):
            if condition_bin == "global":
                continue
            stats = self.calibration_stats[condition_bin]
            rows.append(
                {
                    "method_name": method_name,
                    "far_budget": self.far_budget,
                    "condition_bin": condition_bin,
                    "threshold_accept": stats.get("threshold_accept", ""),
                    "threshold_reject": stats.get("threshold_reject", ""),
                    "n_calibration_genuine": stats.get("n_calibration_genuine", 0),
                    "n_calibration_impostor": stats.get("n_calibration_impostor", 0),
                    "calibration_FAR": stats.get("calibration_FAR", 0.0),
                    "calibration_FRR": stats.get("calibration_FRR", 0.0),
                    "used_global_fallback": stats.get("used_global_fallback", False),
                }
            )
        return rows
