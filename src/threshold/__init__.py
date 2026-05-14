"""
src.threshold — Threshold formula modules.

Exports:
    formula_fixed       : Constant τ = 0.44
    formula_bin_specific: 3-bin fixed τ
    formula_linear     : Linear τ(L, N)
    formula_interaction : Interaction τ(C) [OURS]
    AdaptiveThreshold   : Unified class wrapper (recommended)
"""
from .fixed import formula_fixed
from .bin_specific import formula_bin_specific
from .linear import formula_linear
from .interaction import formula_interaction
from .adaptive_threshold import AdaptiveThreshold

__all__ = [
    'formula_fixed',
    'formula_bin_specific',
    'formula_linear',
    'formula_interaction',
    'AdaptiveThreshold',
]
