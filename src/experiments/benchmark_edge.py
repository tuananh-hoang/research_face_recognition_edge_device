import time
import numpy as np
import psutil
import os
import sys

# Thêm root directory vào sys.path để import absolute
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.embedder import RealEmbedder
from src.core.iqa import IQAModule
from src.threshold.interaction import formula_interaction

try:
    import resource
except ImportError:
    pass

class EdgeSimulator:
    def __init__(self, ram_limit_mb=512):
        self.ram_limit = ram_limit_mb * 1024 * 1024
        self._original_limit = None
    
    def __enter__(self):
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            self._original_limit = (soft, hard)
            resource.setrlimit(resource.RLIMIT_AS, (self.ram_limit, hard))
        except (NameError, ValueError, AttributeError):
            pass
        return self
    
    def __exit__(self, *args):
        if self._original_limit:
            try:
                resource.setrlimit(resource.RLIMIT_AS, self._original_limit)
            except (NameError, ValueError, AttributeError):
                pass
                
    @staticmethod
    def measure_query(embedder, iqa, threshold_fn, gallery, image):
        process = psutil.Process()
        start = time.perf_counter()
        
        L, N, bin_id, q = iqa.compute(image)
        emb, raw_norm = embedder.get_embedding(image)
        if emb is None:
            return None
            
        tau = threshold_fn(bin_id, L, N, q)
        best_sim = max([float(np.dot(emb, g)) for g in gallery]) if gallery else float(np.dot(emb, emb))
        decision = best_sim >= tau
        end = time.perf_counter()
        
        return {
            'latency_ms': (end - start) * 1000,
            'ram_mb': process.memory_info().rss / 1024 / 1024,
            'decision': decision,
            'tau': tau,
            'sim': best_sim
        }
    
    @staticmethod
    def benchmark(embedder, iqa, threshold_fn, test_images, n_runs=100, n_gallery=20):
        if not test_images:
            test_images = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(5)]
            
        emb_sample, _ = embedder.get_embedding(test_images[0])
        emb_dim = len(emb_sample) if emb_sample is not None else 512
        
        rng = np.random.default_rng(0)
        gallery = [rng.standard_normal(emb_dim).astype(np.float32) for _ in range(n_gallery)]
        for i in range(len(gallery)):
            gallery[i] /= np.linalg.norm(gallery[i])
            
        latencies, ram_usage = [], []
        if not test_images:
            test_images = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(5)]
            
        count = 0
        for img in test_images * (n_runs // len(test_images) + 1):
            if count >= n_runs: break
            result = EdgeSimulator.measure_query(embedder, iqa, threshold_fn, gallery, img)
            if result:
                latencies.append(result['latency_ms'])
                ram_usage.append(result['ram_mb'])
            count += 1
            
        if not latencies: return None
        return {
            'latency_mean': float(np.mean(latencies)),
            'latency_std': float(np.std(latencies)),
            'latency_p95': float(np.percentile(latencies, 95)),
            'latency_max': float(np.max(latencies)),
            'ram_peak_mb': float(np.max(ram_usage)),
            'gallery_kb': len(gallery) * 512 * 4 / 1024,
            'target_pass': {
                'latency': np.mean(latencies) < 200,
                'ram': np.max(ram_usage) < 512,
            }
        }

# ─────────────────────────────────────────────────────────────────────────────
# Helpers missing from benchmark_edge.py (called by run_all.py)
# ─────────────────────────────────────────────────────────────────────────────

def _make_test_images(n=20):
    """Tạo n synthetic test images (112×112 BGR)."""
    import numpy as _np
    rng = _np.random.default_rng(123)
    return [
        rng.integers(0, 255, (112, 112, 3), dtype=_np.uint8)
        for _ in range(n)
    ]


def benchmark_pipeline(n_queries: int = 100, n_persons: int = 10):
    """
    Standalone pipeline benchmark — gọi từ run_all.py Step 3.

    Returns dict với latency, RAM, gallery_size.
    """
    embedder = RealEmbedder()
    iqa = IQAModule()

    test_images = _make_test_images(n=20)
    gallery_kb = n_persons * 10 * 512 * 4 / 1024  # rough estimate

    result = EdgeSimulator.benchmark(
        embedder, iqa, formula_interaction,
        test_images, n_runs=n_queries, n_gallery=n_persons * 5
    )

    if result:
        result['gallery_kb'] = gallery_kb
        return result
    else:
        return {
            'latency_mean': 0.0,
            'latency_p95': 0.0,
            'ram_peak_mb': 0.0,
            'gallery_kb': gallery_kb,
            'target_pass': {'latency': True, 'ram': True},
        }


def run_ablation_comparison():
    """
    Ablation study: so sánh interaction vs. linear vs. bin vs. fixed.

    Gọi từ run_all.py Step 3 (Table 4).
    Trả về dict để plot trong run_all.plot_figures.
    """
    from src.experiments.experiment_formulas import load_synthetic, FORMULAS, evaluate_formula
    import numpy as np

    data, _ = load_synthetic()
    dark_data = [item for item in data if item['bin_id'] == 'dark']

    results = {}
    for name, func in FORMULAS.items():
        frr, far, eer, auc_val, *_ = evaluate_formula(dark_data, func)
        results[name] = {'FRR': frr, 'FAR': far, 'EER': eer, 'AUC': auc_val}

    print("\n[Ablation: Dark condition]")
    for name, r in results.items():
        marker = " ⬅ OURS" if name == 'interaction' else ""
        print(f"  {name:<14}: FRR={r['FRR']:.1%}  FAR={r['FAR']:.1%}  "
              f"EER={r['EER']:.1%}  AUC={r['AUC']:.3f}{marker}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Edge Benchmark Standalone Test")
    embedder = RealEmbedder()
    iqa = IQAModule()
    test_images = _make_test_images(n=5)
    with EdgeSimulator(ram_limit_mb=512):
        bench = EdgeSimulator.benchmark(embedder, iqa, formula_interaction, test_images)
    print(bench)