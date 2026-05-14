"""Path-selection policies pi(C)."""

from __future__ import annotations

from dataclasses import dataclass

from .paths import DEFER, FAST, ROBUST


@dataclass(frozen=True)
class AlwaysFastPolicy:
    name: str = "always_fast"

    def select_path(self, context: dict) -> str:
        return FAST


@dataclass(frozen=True)
class AlwaysRobustPolicy:
    name: str = "always_robust"

    def select_path(self, context: dict) -> str:
        return ROBUST


@dataclass(frozen=True)
class ConditionalPolicy:
    """Rule-based policy for the first conditional experiment."""

    l_defer: float = 0.12
    l_robust: float = 0.30
    n_robust: float | None = None
    q_defer: float | None = None
    name: str = "conditional"

    def select_path(self, context: dict) -> str:
        L = float(context.get("L", 0.5))
        N = float(context.get("N", 0.0))
        q = float(context.get("q", 1.0))

        if L < self.l_defer:
            return DEFER
        if self.q_defer is not None and q < self.q_defer:
            return DEFER
        if L < self.l_robust:
            return ROBUST
        if self.n_robust is not None and N > self.n_robust:
            return ROBUST
        return FAST
