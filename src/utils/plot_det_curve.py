"""
plot_det_curve.py — Task 4.1
DET Curve (Detection Error Tradeoff) cho 4 formulas trên dark condition.

DET curve khác ROC:
  - X axis: FAR (log scale)
  - Y axis: FRR (log scale)
  - EER = điểm giao FRR = FAR
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.experiments.experiment_formulas import load_synthetic, evaluate_formula
from src.threshold import formula_fixed, formula_bin_specific, formula_linear, formula_interaction


def compute_det_points(pairs, formula_func, n_steps=1000):
    """
    Sweep threshold từ min_sim đến max_sim.
    Trả về danh sách (FAR, FRR) points.
    """
    sims_same = [p['sim'] for p in pairs if p['label'] == 1]
    sims_diff = [p['sim'] for p in pairs if p['label'] == 0]

    if not sims_same or not sims_diff:
        return [], []

    all_sims = sims_same + sims_diff
    min_t = float(min(all_sims))
    max_t = float(max(all_sims))
    thresholds = np.linspace(min_t, max_t, n_steps)

    far_points, frr_points = [], []
    for tau in thresholds:
        frr = sum(1 for s in sims_same if s < tau) / max(1, len(sims_same))
        far = sum(1 for s in sims_diff if s >= tau) / max(1, len(sims_diff))
        far_points.append(far)
        frr_points.append(frr)

    return np.array(far_points), np.array(frr_points)


def find_eer(far_points, frr_points):
    """Tìm điểm EER = argmin |FRR - FAR|."""
    diffs = np.abs(frr_points - far_points)
    idx = np.argmin(diffs)
    return (far_points[idx] + frr_points[idx]) / 2


def plot_det_curve(pairs, out_path):
    formulas = {
        'fixed': ('Fixed \u03c4', '#d62728', formula_fixed),
        'bin': ('Bin-specific', '#ff7f0e', formula_bin_specific),
        'linear': ('Linear \u03c4(L,N)', '#2ca02c', formula_linear),
        'interaction': ('Interaction \u03c4(C)', '#1f77b4', formula_interaction),
    }

    fig, ax = plt.subplots(figsize=(9, 7))

    eer_values = {}

    for f_key, (label, color, func) in formulas.items():
        far_points, frr_points = compute_det_points(pairs, func)

        if len(far_points) == 0:
            continue

        ax.plot(
            far_points, frr_points,
            color=color, linewidth=2.5,
            label=label,
            zorder=3,
        )

        eer = find_eer(far_points, frr_points)
        eer_values[f_key] = eer

        eers = [(f, v) for f, v in eer_values.items()]
        idx_eer = np.argmin(np.abs(far_points - eer))
        ax.plot(
            far_points[idx_eer], frr_points[idx_eer],
            marker='X', color=color, markersize=12,
            zorder=5,
        )
        ax.annotate(
            f'EER={eer:.2%}',
            (far_points[idx_eer], frr_points[idx_eer]),
            xytext=(10, -15), textcoords='offset points',
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
        )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(1e-4, 1.0)

    ax.set_xlabel('False Acceptance Rate (FAR)', fontsize=12)
    ax.set_ylabel('False Rejection Rate (FRR)', fontsize=12)
    ax.set_title(
        'DET Curve — 4 Threshold Formulas on Dark Condition',
        fontsize=13, fontweight='bold',
    )

    ax.grid(True, which='major', linestyle='-', alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)

    ax.plot([1e-4, 1.0], [1e-4, 1.0],
            color='gray', linestyle='--', linewidth=1.5,
            label='EER = FAR = FRR', zorder=2)

    operating_zone = plt.Rectangle(
        (0.001, 0.01), 0.019, 0.09,
        linewidth=1.5, edgecolor='#28a745',
        facecolor='#28a745', alpha=0.08,
        linestyle='--', zorder=1,
        label='Operating zone\n(FAR<2%, FRR<10%)',
    )
    ax.add_patch(operating_zone)

    ax.legend(
        loc='upper right', fontsize=9,
        framealpha=0.9, edgecolor='#dee2e6',
    )

    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"DET curve saved: {out_path}")

    return eer_values


def main():
    out_dir = _ROOT / 'outputs' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'Figure_06_DET_curve.png'

    print("Loading data for DET curve...")
    pairs, _ = load_synthetic()
    dark_pairs = [p for p in pairs if p['bin_id'] == 'dark']

    print(f"Dark condition: {len(dark_pairs)} pairs "
          f"({sum(1 for p in dark_pairs if p['label']==1)} same, "
          f"{sum(1 for p in dark_pairs if p['label']==0)} diff)")

    eer_values = plot_det_curve(dark_pairs, out_path)

    print("\nEER Summary (Dark Condition):")
    for f_key, eer in eer_values.items():
        print(f"  {f_key:<15}: EER = {eer:.2%}")


if __name__ == '__main__':
    main()
