"""
AdaptiveThreshold — Unified threshold wrapper.

Cung cấp giao diện class cho 4 công thức ngưỡng, dùng bởi run_all.py
và các scripts cần gọi `thresh.get_tau(ctx, formula_name)`.

Thứ tự ưu tiên (fallback chain):
    interaction → linear → bin_specific → fixed

Usage:
    thresh = AdaptiveThreshold()
    tau = thresh.get_tau(ctx, 'interaction')

    ctx = {
        'L': float,          # Luminance [0, 1]
        'N': float,          # Noise [0, 1]
        'q': float,          # Quality [0, 1]
        'bin_id': str,       # 'bright' | 'medium' | 'dark'
    }
"""
from __future__ import annotations

from .fixed import formula_fixed
from .bin_specific import formula_bin_specific
from .linear import formula_linear
from .interaction import formula_interaction

__all__ = ['AdaptiveThreshold']

_FORMULA_NAMES = {
    'fixed',
    'bin',
    'bin_specific',
    'linear',
    'interaction',
}


class AdaptiveThreshold:
    """
    Unified adaptive threshold interface.

    Hỗ trợ 4 formula:
      - 'fixed'         → constant 0.44
      - 'bin' / 'bin_specific' → 3-bin fixed
      - 'linear'        → τ(L, N) = 0.48 - 0.10*(1-L) - 0.05*N
      - 'interaction'    → τ(C) với interaction term [DEFAULT]

    Parameters
    ----------
    default_formula : str
        Formula mặc định khi gọi get_tau mà không chỉ định tên.
        Mặc định: 'interaction'.
    """

    def __init__(self, default_formula: str = 'interaction'):
        if default_formula not in _FORMULA_NAMES:
            raise ValueError(
                f"Unknown formula '{default_formula}'. "
                f"Available: {_FORMULA_NAMES}"
            )
        self._default = default_formula

        # Map alias 'bin' → 'bin_specific'
        self._funcs = {
            'fixed': formula_fixed,
            'bin': formula_bin_specific,
            'bin_specific': formula_bin_specific,
            'linear': formula_linear,
            'interaction': formula_interaction,
        }

    def get_tau(
        self,
        ctx: dict,
        formula_name: str | None = None,
    ) -> float:
        """
        Tính ngưỡng τ từ context dict.

        Parameters
        ----------
        ctx : dict
            Context chứa các trường:
              - L      : float  (0–1)
              - N      : float  (0–1)
              - q      : float  (0–1)
              - bin_id : str    ('bright' | 'medium' | 'dark')
            Nếu thiếu trường nào → dùng giá trị mặc định hợp lý:
              - bin_id mặc định = 'medium'
              - N mặc định = 0.0 (giả định ít noise)
        """
        name = formula_name or self._default

        if name not in self._funcs:
            raise ValueError(
                f"Unknown formula '{name}'. Available: {sorted(_FORMULA_NAMES)}"
            )

        # Extract với defaults
        L      = float(ctx.get('L', 0.5))
        N      = float(ctx.get('N', 0.0))
        q      = float(ctx.get('q', 1.0))
        bin_id = str(ctx.get('bin_id', 'medium'))

        func = self._funcs[name]
        return func(bin_id, L, N, q)

    def get_tau_fixed(self, bin_id: str = 'medium') -> float:
        """Tính τ_fixed cho nhanh (không cần context)."""
        return formula_fixed(bin_id, 0.5, 0.0, 1.0)

    def get_tau_interaction(self, ctx: dict) -> float:
        """Tính τ_interaction cho nhanh."""
        return self.get_tau(ctx, 'interaction')

    def describe_formula(self, name: str) -> str:
        """Trả về mô tả bằng text cho một formula."""
        descriptions = {
            'fixed': (
                "Fixed threshold: τ = 0.44 (constant). "
                "Baseline, không dùng context."
            ),
            'bin': (
                "Bin-specific threshold: "
                "τ_bright=0.48, τ_medium=0.42, τ_dark=0.35. "
                "Chỉ dùng bin_id."
            ),
            'linear': (
                "Linear threshold: τ(L, N) = 0.48 - 0.10*(1-L) - 0.05*N. "
                "Giảm tuyến tính khi L thấp hoặc N cao."
            ),
            'interaction': (
                "Interaction threshold: "
                "τ(C) = 0.48 * (1 - 0.25*(1-L)*N) * q + 0.30*(1-q). "
                "[OURS] Multiplicative interaction giữa (1-L) và N, "
                "điều chỉnh bởi image quality q."
            ),
        }
        return descriptions.get(name, f"Unknown formula: {name}")

    def table_summary(self, ctx: dict) -> dict:
        """
        Trả về dict chứa τ của tất cả formulas cho một context.
        Tiện cho so sánh trong báo cáo.
        """
        result = {'context': ctx}
        for name in ['fixed', 'bin', 'linear', 'interaction']:
            result[name] = self.get_tau(ctx, name)
        return result

    def __repr__(self) -> str:
        return f"AdaptiveThreshold(default='{self._default}')"
