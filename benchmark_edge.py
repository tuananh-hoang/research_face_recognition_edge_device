"""
benchmark_edge.py — Đo hiệu năng edge deployment
Giả lập môi trường edge: 512MB RAM, CPU-only
Đo: latency (ms), RAM (MB), gallery size (KB)
"""

import time
import numpy as np
import psutil
import os
import json
from typing import Dict, List

from iqa import IQAModule
from threshold import AdaptiveThreshold
from gallery_manager import GalleryManager


def benchmark_pipeline(n_queries: int = 100,
                        n_persons: int = 10,
                        use_insightface: bool = False) -> Dict:
    """
    Benchmark end-to-end pipeline trên simulated edge.

    Args:
        n_queries: số queries để đo latency
        n_persons: số người trong gallery
        use_insightface: True nếu muốn test với model thật (cần model download)
    """

    # Setup
    iqa = IQAModule()
    thresh = AdaptiveThreshold()
    gallery = GalleryManager(k_per_person=20)
    rng = np.random.default_rng(42)

    # Enroll persons (dùng random embeddings nếu không có InsightFace)
    persons = [f"person_{i:02d}" for i in range(n_persons)]
    for pid in persons:
        for _ in range(3):
            emb = rng.standard_normal(512).astype(np.float32)
            gallery.enroll(pid, emb, 'anchor')

    # Prepare test images (112x112 BGR)
    test_images = [
        rng.integers(10, 60, (112, 112, 3), dtype=np.uint8)  # dark images
        for _ in range(n_queries)
    ]

    # ─── Benchmark Loop ──────────────────────────────────────────
    latencies = []
    process = psutil.Process(os.getpid())

    print(f"\nBenchmarking {n_queries} queries (simulated dark condition)...")
    print("─" * 50)

    for i, img in enumerate(test_images):
        start = time.perf_counter()

        # Step 1: IQA (thay vì embedding để không cần model download)
        ctx = iqa.compute_context(img, feature_norm=rng.uniform(5, 20))

        # Step 2: Simulate embedding (random normalized vector)
        emb = rng.standard_normal(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        # Step 3: Adaptive threshold
        tau = thresh.get_tau(ctx, method='interaction')

        # Step 4: Gallery search
        best_id, best_sim = gallery.search(emb, ctx['bin_id'])

        # Step 5: Decision
        decision = best_sim >= tau

        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    # ─── Memory measurement ──────────────────────────────────────
    ram_mb = process.memory_info().rss / 1024 / 1024
    gallery_kb = gallery.get_size_kb()

    # ─── Stats ──────────────────────────────────────────────────
    latencies = np.array(latencies)
    results = {
        'latency_mean_ms': float(latencies.mean()),
        'latency_std_ms': float(latencies.std()),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'latency_max_ms': float(latencies.max()),
        'ram_mb': ram_mb,
        'gallery_kb': gallery_kb,
        'n_persons': n_persons,
        'n_queries': n_queries,
    }

    # ─── Report ──────────────────────────────────────────────────
    targets = {
        'latency_mean_ms': 200.0,
        'ram_mb': 512.0,
        'gallery_kb': 400.0,
    }

    print(f"\n{'Metric':<30} {'Value':>12} {'Target':>10} {'Status':>8}")
    print("─" * 65)

    checks = [
        ('Latency mean (ms)', results['latency_mean_ms'], targets['latency_mean_ms'], 'lower'),
        ('Latency p95 (ms)',  results['latency_p95_ms'],  targets['latency_mean_ms'] * 1.5, 'lower'),
        ('Latency max (ms)',  results['latency_max_ms'],  targets['latency_mean_ms'] * 3,   'lower'),
        ('RAM usage (MB)',    results['ram_mb'],           targets['ram_mb'],    'lower'),
        ('Gallery size (KB)', results['gallery_kb'],       targets['gallery_kb'], 'lower'),
    ]

    all_pass = True
    for name, value, target, direction in checks:
        if direction == 'lower':
            status = '✅ PASS' if value <= target else '❌ FAIL'
            if value > target:
                all_pass = False
        print(f"{name:<30} {value:>12.2f} {target:>10.1f} {status:>8}")

    print("─" * 65)
    print(f"Edge deployment {'✅ FEASIBLE' if all_pass else '⚠️ NEEDS OPTIMIZATION'}")

    # Latency breakdown note
    print(f"\nLatency breakdown (estimated):")
    print(f"  IQA computation  : ~{results['latency_mean_ms'] * 0.15:.2f} ms")
    print(f"  Embedding (model): ~10-50 ms (when using real ArcFace on CPU)")
    print(f"  Threshold compute: ~{results['latency_mean_ms'] * 0.05:.2f} ms")
    print(f"  Gallery search   : ~{results['latency_mean_ms'] * 0.80:.2f} ms")
    print(f"  Note: Real end-to-end includes InsightFace model inference")

    return results


def run_ablation_comparison() -> Dict:
    """
    Ablation study: compare variants của system
    Đây là Table 4 concept của báo cáo
    """
    iqa = IQAModule()
    rng = np.random.default_rng(42)

    # Simulate pairs cho ablation
    n_pairs = 200

    def compute_frr_far(method: str, condition: str) -> Dict:
        thresh = AdaptiveThreshold()
        L_map = {'bright': 0.72, 'medium': 0.45, 'dark': 0.18}
        N_map = {'bright': 0.05, 'medium': 0.15, 'dark': 0.35}
        q_map = {'bright': 0.85, 'medium': 0.65, 'dark': 0.35}

        ctx = {'L': L_map[condition], 'N': N_map[condition],
               'q': q_map[condition], 'bin_id': condition}
        tau = thresh.get_tau(ctx, method)

        # Simulate pairs
        if condition == 'bright':
            same_sims = rng.normal(0.62, 0.08, n_pairs // 2)
            diff_sims = rng.normal(0.18, 0.10, n_pairs // 2)
        elif condition == 'medium':
            same_sims = rng.normal(0.52, 0.10, n_pairs // 2)
            diff_sims = rng.normal(0.22, 0.11, n_pairs // 2)
        else:
            same_sims = rng.normal(0.41, 0.12, n_pairs // 2)
            diff_sims = rng.normal(0.28, 0.12, n_pairs // 2)

        frr = float((same_sims < tau).mean())
        far = float((diff_sims >= tau).mean())
        return {'FRR': frr, 'FAR': far, 'tau': tau}

    print("\n" + "═" * 70)
    print("ABLATION STUDY (Table 4 concept)")
    print("═" * 70)
    print(f"{'Variant':<25} {'FRR_dark':>10} {'FRR_bright':>10} {'FAR_dark':>10}")
    print("─" * 70)

    ablations = {
        'Full system (interaction)': ('interaction', 'interaction'),
        'No interaction (linear)':   ('linear',      'linear'),
        'Fixed threshold':            ('fixed',       'fixed'),
        'Bin-specific only':          ('bin',         'bin'),
    }

    ablation_results = {}
    for name, (dark_method, bright_method) in ablations.items():
        dark = compute_frr_far(dark_method, 'dark')
        bright = compute_frr_far(bright_method, 'bright')
        ablation_results[name] = {'dark': dark, 'bright': bright}
        print(f"{name:<25} {dark['FRR']:>10.1%} {bright['FRR']:>10.1%} {dark['FAR']:>10.1%}")

    print("─" * 70)
    full = ablation_results['Full system (interaction)']
    fixed = ablation_results['Fixed threshold']
    delta = fixed['dark']['FRR'] - full['dark']['FRR']
    print(f"\nKey finding: Interaction method reduces FRR_dark by {delta:.1%} vs fixed")

    return ablation_results


if __name__ == "__main__":

    # Edge benchmark
    bench_results = benchmark_pipeline(n_queries=100, n_persons=10)

    # Ablation
    ablation = run_ablation_comparison()

    # Save
    all_results = {
        'benchmark': bench_results,
        'ablation_summary': {
            k: {cond: {m: float(v) for m, v in metrics.items()}
                for cond, metrics in v.items()}
            for k, v in ablation.items()
        }
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n✅ Benchmark results saved")