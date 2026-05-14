"""Threshold policies tau_r(C) for conditional pipeline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

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
