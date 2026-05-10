"""
Module C — Adaptive Threshold τ(C)
Công thức từ báo cáo (Section 3.3):

  τ(C) = τ_base · (1 - γ·(1-L)·N) · q + τ_floor · (1-q) - β·N·[N > N_penalty]

Trong đó:
  L        = Luminance [0,1]
  N        = Noise level [0,1]
  q        = quality score [0,1]
  γ        = interaction weight (dark × noise)
  τ_base   = ngưỡng baseline (điều kiện tốt nhất)
  τ_floor  = ngưỡng sàn tối thiểu (không bao giờ về 0)
  β        = noise penalty (chống FAR tăng khi nhiễu cao)
  N_penalty= ngưỡng kích hoạt noise penalty

So sánh 4 phương pháp:
  A: Fixed τ = 0.44
  B: Bin-specific τ (per condition)
  C: Linear τ = a + b(1-L) + cN
  D: Interaction τ(C) với (1-L)·N [ours]
"""

import numpy as np
from typing import Dict, Tuple
import json


class AdaptiveThreshold:
    def __init__(self,
                 tau_base: float = 0.48,
                 tau_floor: float = 0.30,
                 gamma: float = 0.25,
                 beta: float = 0.04,
                 n_penalty_threshold: float = 0.6,
                 # Linear model params (fallback)
                 linear_a: float = 0.48,
                 linear_b: float = 0.12,
                 linear_c: float = 0.05,
                 # Bin-specific thresholds
                 tau_bright: float = 0.48,
                 tau_medium: float = 0.42,
                 tau_dark: float = 0.35):

        self.tau_base = tau_base
        self.tau_floor = tau_floor
        self.gamma = gamma
        self.beta = beta
        self.n_penalty_threshold = n_penalty_threshold

        # Linear model
        self.a = linear_a
        self.b = linear_b
        self.c = linear_c

        # Bin-specific
        self.bin_tau = {
            'bright': tau_bright,
            'medium': tau_medium,
            'dark': tau_dark,
        }

    # ─── 4 phương pháp để compare ────────────────────────────────

    def fixed(self) -> float:
        """Method A: Fixed threshold (baseline)"""
        return 0.44

    def bin_specific(self, bin_id: str) -> float:
        """Method B: Bin-specific threshold"""
        return self.bin_tau.get(bin_id, 0.44)

    def linear(self, L: float, N: float) -> float:
        """Method C: Linear model τ = a + b(1-L) + cN"""
        tau = self.a - self.b * (1 - L) - self.c * N
        return float(np.clip(tau, self.tau_floor, self.tau_base))

    def interaction(self, L: float, N: float, q: float) -> float:
        """
        Method D: Interaction model với term (1-L)·N [ours — novelty chính]

        τ(C) = τ_base · (1 - γ·(1-L)·N) · q + τ_floor · (1-q) - β·N·[N > threshold]

        Tại sao interaction term tốt hơn linear:
        - Linear: effect của L và N độc lập (cộng)
        - Interaction: tối + nhiễu cùng lúc → NHÂN → worst-case được xử lý đặc biệt
        - Analogy: lái xe tối + mưa không phải giảm 50km/h mà giảm 80km/h
        """
        # Core interaction: (1-L)·N capture worst-case synergy
        interaction_term = (1.0 - L) * N

        # Threshold chính với quality-gating
        tau_main = (self.tau_base * (1 - self.gamma * interaction_term) * q
                    + self.tau_floor * (1 - q))

        # Noise penalty: chống FAR tăng khi nhiễu quá cao
        noise_penalty = self.beta * N if N > self.n_penalty_threshold else 0.0

        tau = tau_main - noise_penalty
        return float(np.clip(tau, self.tau_floor, self.tau_base))

    def get_tau(self, context: Dict, method: str = 'interaction') -> float:
        """
        Unified interface để lấy threshold theo method chỉ định.

        Args:
            context: dict từ IQAModule.compute_context()
            method: 'fixed' | 'bin' | 'linear' | 'interaction'
        """
        L = context.get('L', 0.5)
        N = context.get('N', 0.1)
        q = context.get('q', 0.7)
        bin_id = context.get('bin_id', 'medium')

        if method == 'fixed':
            return self.fixed()
        elif method == 'bin':
            return self.bin_specific(bin_id)
        elif method == 'linear':
            return self.linear(L, N)
        else:  # 'interaction' — default
            return self.interaction(L, N, q)

    # ─── Calibration ────────────────────────────────────────────

    def calibrate_bin_thresholds(self,
                                  pairs: list,
                                  target_far: float = 0.01) -> Dict:
        """
        Tự động calibrate τ_bin cho mỗi condition sao cho FAR ≈ target_far.

        Args:
            pairs: list of (sim, label, bin_id) tuples
                   label=1 (same person), label=0 (different person)
            target_far: False Acceptance Rate mục tiêu
        Returns:
            dict của tau per bin
        """
        from scipy.optimize import brentq

        bins = ['bright', 'medium', 'dark']
        calibrated = {}

        for bin_id in bins:
            bin_pairs = [(sim, lab) for sim, lab, b in pairs if b == bin_id]
            if not bin_pairs:
                calibrated[bin_id] = self.bin_tau.get(bin_id, 0.44)
                continue

            neg_sims = [sim for sim, lab in bin_pairs if lab == 0]
            if not neg_sims:
                calibrated[bin_id] = self.bin_tau.get(bin_id, 0.44)
                continue

            neg_sims = sorted(neg_sims, reverse=True)

            def far_at_tau(tau):
                accepted = sum(1 for s in neg_sims if s >= tau)
                return accepted / len(neg_sims) - target_far

            try:
                tau_opt = brentq(far_at_tau,
                                 a=self.tau_floor,
                                 b=self.tau_base,
                                 maxiter=100)
                calibrated[bin_id] = float(tau_opt)
            except ValueError:
                calibrated[bin_id] = self.bin_tau.get(bin_id, 0.44)

        self.bin_tau.update(calibrated)
        print(f"Calibrated bin thresholds (FAR={target_far:.2%}): {calibrated}")
        return calibrated

    def calibrate_interaction_params(self,
                                      pairs: list,
                                      contexts: list) -> None:
        """
        Calibrate γ và β bằng grid search trên dev set.

        Args:
            pairs: list of (sim, label) tuples
            contexts: list of context dicts tương ứng
        """
        best_score = -np.inf
        best_gamma, best_beta = self.gamma, self.beta

        for gamma in np.arange(0.1, 0.5, 0.05):
            for beta in np.arange(0.0, 0.1, 0.02):
                self.gamma = gamma
                self.beta = beta

                # Đo EER trên tập này
                scores = [
                    (self.interaction(ctx['L'], ctx['N'], ctx['q']), sim, label)
                    for (sim, label), ctx in zip(pairs, contexts)
                ]

                # EER proxy: FAR - FRR tại một ngưỡng
                frr = sum(1 for tau, sim, lab in scores
                          if lab == 1 and sim < tau) / max(1, sum(lab for _, _, lab in scores if lab == 1))
                far = sum(1 for tau, sim, lab in scores
                          if lab == 0 and sim >= tau) / max(1, sum(1 for _, _, lab in scores if lab == 0))

                # Tối ưu: minimize |FRR - FAR| (EER condition)
                score = -abs(frr - far)
                if score > best_score:
                    best_score = score
                    best_gamma, best_beta = gamma, beta

        self.gamma = best_gamma
        self.beta = best_beta
        print(f"Calibrated: gamma={self.gamma:.3f}, beta={self.beta:.3f}")

    def save(self, path: str) -> None:
        params = {
            'tau_base': self.tau_base,
            'tau_floor': self.tau_floor,
            'gamma': self.gamma,
            'beta': self.beta,
            'n_penalty_threshold': self.n_penalty_threshold,
            'linear_a': self.a,
            'linear_b': self.b,
            'linear_c': self.c,
            'bin_tau': self.bin_tau,
        }
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Threshold params saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'AdaptiveThreshold':
        with open(path) as f:
            params = json.load(f)
        obj = cls(
            tau_base=params['tau_base'],
            tau_floor=params['tau_floor'],
            gamma=params['gamma'],
            beta=params['beta'],
            n_penalty_threshold=params['n_penalty_threshold'],
            linear_a=params['linear_a'],
            linear_b=params['linear_b'],
            linear_c=params['linear_c'],
            tau_bright=params['bin_tau'].get('bright', 0.48),
            tau_medium=params['bin_tau'].get('medium', 0.42),
            tau_dark=params['bin_tau'].get('dark', 0.35),
        )
        return obj


# ─── Sanity test ────────────────────────────────────────────────
if __name__ == "__main__":
    thresh = AdaptiveThreshold()

    test_cases = [
        {'name': 'BRIGHT (good)',  'L': 0.75, 'N': 0.05, 'q': 0.90, 'bin_id': 'bright'},
        {'name': 'MEDIUM',         'L': 0.45, 'N': 0.15, 'q': 0.65, 'bin_id': 'medium'},
        {'name': 'DARK (low N)',   'L': 0.15, 'N': 0.10, 'q': 0.40, 'bin_id': 'dark'},
        {'name': 'DARK+NOISE (worst)', 'L': 0.10, 'N': 0.70, 'q': 0.15, 'bin_id': 'dark'},
    ]

    print(f"\n{'Condition':<25} {'Fixed':>7} {'Bin':>7} {'Linear':>7} {'Interact':>9}")
    print("-" * 60)
    for tc in test_cases:
        ctx = tc
        t_fixed = thresh.get_tau(ctx, 'fixed')
        t_bin   = thresh.get_tau(ctx, 'bin')
        t_lin   = thresh.get_tau(ctx, 'linear')
        t_int   = thresh.get_tau(ctx, 'interaction')
        print(f"{tc['name']:<25} {t_fixed:>7.3f} {t_bin:>7.3f} {t_lin:>7.3f} {t_int:>9.3f}")

    print("\nKey insight: DARK+NOISE (worst case) → interaction < linear (lenient hơn)")
    print("Module C OK")