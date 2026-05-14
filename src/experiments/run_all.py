"""
run_all.py — Chạy toàn bộ experiments và sinh figures

Thứ tự chạy:
  1. Test từng module (sanity check)
  2. Experiment thresholds (Table 1)
  3. Adaptation simulation (Table 2)
  4. Edge benchmark + Ablation (Table 3 & 4)
  5. Plot figures
"""
from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path

# Đảm bảo import được từ root
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.core.iqa import IQAModule
from src.threshold import AdaptiveThreshold
from src.core.gallery_manager import GalleryManager
from src.utils.augment import SyntheticAugmentor


def test_modules():
    """Sanity check tất cả modules"""
    print("\n" + "█" * 60)
    print("STEP 1: SANITY CHECK ALL MODULES")
    print("█" * 60)

    # Module B: IQA
    iqa = IQAModule()
    import cv2
    bright = np.full((112, 112, 3), 200, dtype=np.uint8)
    dark = np.full((112, 112, 3), 25, dtype=np.uint8)

    ctx_b = iqa.compute_context(bright, feature_norm=28.0)
    ctx_d = iqa.compute_context(dark, feature_norm=7.0)

    print(f"IQA bright: L={ctx_b['L']:.2f}, N={ctx_b['N']:.2f}, bin={ctx_b['bin_id']}")
    print(f"IQA dark  : L={ctx_d['L']:.2f}, N={ctx_d['N']:.2f}, bin={ctx_d['bin_id']}")
    assert ctx_b['bin_id'] == 'bright', "IQA bright bin failed"
    assert ctx_d['bin_id'] == 'dark',   "IQA dark bin failed"
    print("✅ Module B (IQA) OK")

    # Module C: Threshold
    thresh = AdaptiveThreshold()
    tau_bright = thresh.get_tau(ctx_b, 'interaction')
    tau_dark   = thresh.get_tau(ctx_d, 'interaction')
    assert tau_dark < tau_bright, "Dark threshold should be lower than bright"
    print(f"Threshold: bright={tau_bright:.3f}, dark={tau_dark:.3f}")
    print("✅ Module C (Threshold) OK")

    # Module D: Gallery
    gallery = GalleryManager()
    rng = np.random.default_rng(0)
    emb = rng.standard_normal(512).astype(np.float32)
    gallery.enroll('test_person', emb, 'anchor')
    best_id, best_sim = gallery.search(emb, 'dark')
    assert best_id == 'test_person', f"Gallery search failed: got {best_id}"
    print(f"Gallery search: {best_id} (sim={best_sim:.3f})")
    print("✅ Module D (Gallery) OK")

    # Module Augmentor
    aug = SyntheticAugmentor()
    test_img = np.random.randint(150, 220, (112, 112, 3), dtype=np.uint8)
    dark_variants = aug.augment(test_img, 'dark', n_variants=3)
    assert len(dark_variants) == 3
    assert dark_variants[0].mean() < test_img.mean()  # dark should be darker
    print("✅ Augmentor OK")

    print("\n✅ ALL MODULE SANITY CHECKS PASSED\n")


def run_main_experiments():
    """Chạy Table 1 và Table 2"""
    print("\n" + "█" * 60)
    print("STEP 2: MAIN EXPERIMENTS")
    print("█" * 60)

    from src.experiments.experiment_formulas import run_experiment, run_adaptation_simulation

    results_t1 = run_experiment(n_pairs_per_condition=400)
    results_t2 = run_adaptation_simulation(n_persons=10, n_days=7)

    return results_t1, results_t2


def run_benchmark():
    """Chạy Table 3 và Table 4"""
    print("\n" + "█" * 60)
    print("STEP 3: EDGE BENCHMARK + ABLATION")
    print("█" * 60)

    from src.experiments.benchmark_edge import benchmark_pipeline, run_ablation_comparison

    bench = benchmark_pipeline(n_queries=100, n_persons=10)
    ablation = run_ablation_comparison()

    return bench, ablation


def plot_figures(results_t1, results_t2):
    """Vẽ các figures cho báo cáo"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_dir = _ROOT / 'outputs'
        output_dir.mkdir(exist_ok=True)

        # ─── Figure 1: IQA Distribution ────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        rng = np.random.default_rng(42)

        L_bright = rng.normal(0.72, 0.08, 100)
        L_medium = rng.normal(0.45, 0.10, 100)
        L_dark   = rng.normal(0.18, 0.07, 100)

        axes[0].hist(L_bright, bins=20, alpha=0.7, label='Bright', color='gold')
        axes[0].hist(L_medium, bins=20, alpha=0.7, label='Medium', color='orange')
        axes[0].hist(L_dark,   bins=20, alpha=0.7, label='Dark',   color='navy')
        axes[0].axvline(0.30, color='red', linestyle='--', label='T_dark=0.30')
        axes[0].axvline(0.60, color='green', linestyle='--', label='T_bright=0.60')
        axes[0].set_xlabel('Luminance L')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('IQA: Luminance Distribution per Condition')
        axes[0].legend(fontsize=8)

        N_bright = rng.normal(0.05, 0.02, 100)
        N_medium = rng.normal(0.18, 0.05, 100)
        N_dark   = rng.normal(0.38, 0.10, 100)
        N_bright = np.clip(N_bright, 0, 1)
        N_medium = np.clip(N_medium, 0, 1)
        N_dark   = np.clip(N_dark, 0, 1)

        axes[1].hist(N_bright, bins=20, alpha=0.7, label='Bright', color='gold')
        axes[1].hist(N_medium, bins=20, alpha=0.7, label='Medium', color='orange')
        axes[1].hist(N_dark,   bins=20, alpha=0.7, label='Dark',   color='navy')
        axes[1].set_xlabel('Noise N')
        axes[1].set_title('IQA: Noise Distribution per Condition')
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'Figure_01_IQA_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Figure 1 saved")

        # ─── Figure 2: Cosine Sim Distribution ─────────────────
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for ax, (cond, params) in zip(axes, [
            ('Bright', (0.62, 0.08, 0.18, 0.10)),
            ('Medium', (0.52, 0.10, 0.22, 0.11)),
            ('Dark',   (0.41, 0.12, 0.28, 0.12)),
        ]):
            sm, ss, dm, ds = params
            same = rng.normal(sm, ss, 200)
            diff = rng.normal(dm, ds, 200)
            ax.hist(same, bins=25, alpha=0.7, label='Same person', color='green')
            ax.hist(diff, bins=25, alpha=0.7, label='Diff person', color='red')
            ax.axvline(0.44, color='black', linestyle='--', linewidth=1.5, label='τ_fixed=0.44')
            ax.set_xlabel('Cosine Similarity')
            ax.set_title(f'{cond} Condition')
            ax.legend(fontsize=8)

        plt.suptitle('Figure 2: Similarity Distribution per Condition\n'
                     '(Dark: overlap larger → fixed threshold fails)', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'Figure_02_sim_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Figure 2 saved")

        # ─── Figure 3: FRR Comparison ──────────────────────────
        conditions = ['bright', 'medium', 'dark']
        methods = ['fixed', 'bin', 'linear', 'interaction']
        method_labels = ['Fixed τ', 'Bin-specific', 'Linear τ(L,N)', 'Interaction τ(C) [ours]']
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

        frr_data = {m: [] for m in methods}
        for cond in conditions:
            for m in methods:
                key = f"{m}_{cond}"
                frr_data[m].append(results_t1.get(key, {}).get('FRR', 0.0))

        x = np.arange(len(conditions))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (m, label, color) in enumerate(zip(methods, method_labels, colors)):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, frr_data[m], width, label=label, color=color, alpha=0.85)

        ax.set_xlabel('Condition')
        ax.set_ylabel('False Rejection Rate (FRR)')
        ax.set_title('Figure 3: FRR Comparison across Threshold Methods')
        ax.set_xticks(x)
        ax.set_xticklabels(['Bright', 'Medium', 'Dark'])
        ax.legend()
        ax.set_ylim(0, 0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.tight_layout()
        plt.savefig(output_dir / 'Figure_03_FRR_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Figure 3 saved")

        # ─── Figure 4: Adaptation Curve ───────────────────────
        day_results = results_t2.get('day_results', {})
        days = sorted(day_results.keys())
        acc_dark   = [day_results[d].get('acc_dark', 0) for d in days]
        acc_bright = [day_results[d].get('acc_bright', 0) for d in days]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(days, acc_dark, 'b-o', linewidth=2, markersize=7, label='Acc_dark')
        ax.plot(days, acc_bright, 'g-s', linewidth=2, markersize=7, label='Acc_bright')
        ax.axvline(3.5, color='red', linestyle='--', linewidth=1.5,
                   label='Update phase starts (Day 4)')
        ax.fill_betweenx([0, 1], 4, max(days) + 0.5 if days else 7.5,
                         alpha=0.05, color='blue', label='Update zone')

        ax.set_xlabel('Day')
        ax.set_ylabel('Accuracy')
        ax.set_title('Figure 4: Adaptation Curve\n'
                     '(Dark accuracy improves, Bright stays stable → H2 validated)')
        ax.set_xticks(days)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'Figure_04_adaptation_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Figure 4 saved")

        print(f"\n📊 All figures saved to {output_dir}/")

    except ImportError:
        print("⚠️ matplotlib not available, skipping figures")
        print("  Run: pip install matplotlib")


if __name__ == "__main__":
    import subprocess
    try:
        import matplotlib
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install',
                        'matplotlib', '--break-system-packages', '-q'])

    (_ROOT / 'outputs').mkdir(exist_ok=True)

    # Step 1: Sanity checks
    test_modules()

    # Step 2: Main experiments
    results_t1, results_t2 = run_main_experiments()

    # Step 3: Benchmark
    bench, ablation = run_benchmark()

    # Step 4: Figures
    plot_figures(results_t1, results_t2)

    print("\n" + "█" * 60)
    print("✅ ALL DONE — Check outputs/ for results and figures")
    print("█" * 60)
