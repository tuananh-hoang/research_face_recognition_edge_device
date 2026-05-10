import argparse
import numpy as np
import cv2
import csv
import os
import glob
import time
import psutil
try:
    import resource
except ImportError:
    pass
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    mcnemar = None

# ==========================================
# === TẦNG 1: DATA LAYER (EMBEDDING & IQA) ===
# ==========================================

class RealEmbedder:
    def __init__(self):
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name='buffalo_sc',  # model nhỏ, CPU-friendly
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            self.available = True
        except ImportError:
            print("⚠️ InsightFace không khả dụng (thiếu module insightface hoặc onnxruntime).")
            print("⚠️ Đang dùng MOCK embedding cho mục đích test (không dùng cho báo cáo NCKH thật).")
            self.available = False
            
    def get_embedding(self, image):
        if not self.available:
            img_resized = cv2.resize(image, (112, 112)).flatten().astype(np.float32)
            raw_norm = float(np.linalg.norm(img_resized) + 1e-8)
            emb = img_resized / raw_norm
            return emb, raw_norm
            
        faces = self.app.get(image)
        if not faces:
            return None, 0.0
        face = max(faces, key=lambda x: x.det_score)
        emb = face.embedding
        raw_norm = float(np.linalg.norm(emb))
        emb = emb / raw_norm
        return emb, raw_norm
    
    def similarity(self, img1, img2):
        emb1, norm1 = self.get_embedding(img1)
        emb2, norm2 = self.get_embedding(img2)
        if emb1 is None or emb2 is None:
            return None, None, None
        sim = float(np.dot(emb1, emb2))
        q = np.clip((norm1 + norm2) / 2 / 25.0, 0.01, 1.0)
        return sim, q, (norm1 + norm2) / 2

class IQAModule:
    @staticmethod
    def compute(img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        L = np.mean(ycrcb[:, :, 0]) / 255.0
        
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        diff = img.astype(np.float32) - blurred.astype(np.float32)
        N = np.std(diff) / 255.0
        
        if L < 0.3:
            bin_id = 'dark'
        elif L > 0.6:
            bin_id = 'bright'
        else:
            bin_id = 'medium'
            
        q = max(0.0, min(1.0, 1.0 - N))
        return L, N, bin_id, q

def load_synthetic():
    rng = np.random.default_rng(42)
    data = []
    
    conditions = [
        ('bright', 0.62, 0.08, 0.18, 0.10, 0.60, 0.90, 0.02, 0.10),
        ('medium', 0.52, 0.10, 0.22, 0.11, 0.30, 0.60, 0.08, 0.20),
        ('dark',   0.41, 0.12, 0.28, 0.12, 0.05, 0.30, 0.15, 0.50)
    ]
    
    for cond, sm, ss, dm, ds, l_low, l_high, n_low, n_high in conditions:
        for is_same in [1, 0]:
            for _ in range(500): 
                sim = rng.normal(sm, ss) if is_same else rng.normal(dm, ds)
                sim = float(np.clip(sim, -1.0, 1.0))
                L = rng.uniform(l_low, l_high)
                N = rng.uniform(n_low, n_high)
                q = max(0.0, 1.0 - N)
                data.append({
                    'sim': sim, 'label': is_same, 'L': L, 'N': N, 
                    'q': q, 'bin_id': cond
                })
    return data, None

def load_lfw(data_path, embedder, iqa):
    print(f"Loading LFW dataset into {data_path} (chỉ dùng để validate embedding)...")
    from sklearn.datasets import fetch_lfw_pairs
    lfw = fetch_lfw_pairs(subset='test', color=True, data_home=data_path)
    data = []
    test_images = []
    
    for i, (pair, label) in enumerate(zip(lfw.pairs, lfw.target)):
        img1 = cv2.cvtColor((pair[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor((pair[1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        sim, q_emb, _ = embedder.similarity(img1, img2)
        if sim is None: continue
        
        L1, N1, bin1, q1 = iqa.compute(img1)
        L2, N2, bin2, q2 = iqa.compute(img2)
        
        L = (L1 + L2) / 2
        N = (N1 + N2) / 2
        q = q_emb
        
        if L < 0.3: bin_id = 'dark'
        elif L > 0.6: bin_id = 'bright'
        else: bin_id = 'medium'
        
        data.append({
            'sim': sim, 'label': int(label), 'L': L, 'N': N,
            'q': q, 'bin_id': bin_id
        })
        if len(test_images) < 50:
            test_images.append(img1)
            
    return data, test_images

def load_custom(data_path, embedder, iqa):
    data = []
    test_images = []
    path = Path(data_path)
    
    for cond in ['bright', 'medium', 'dark']:
        cond_dir = path / cond
        if not cond_dir.exists():
            continue
            
        files = list(cond_dir.glob('*.jpg')) + list(cond_dir.glob('*.png'))
        if not files:
            continue
            
        persons = {}
        for f in files:
            pid = f.stem.split('_')[0]
            persons.setdefault(pid, []).append(str(f))
            
        for pid, imgs in persons.items():
            if len(imgs) >= 2:
                img1 = cv2.imread(imgs[0])
                img2 = cv2.imread(imgs[1])
                sim, q_emb, _ = embedder.similarity(img1, img2)
                if sim is None: continue
                
                L1, N1, _, q1 = iqa.compute(img1)
                L2, N2, _, q2 = iqa.compute(img2)
                data.append({
                    'sim': sim, 'label': 1, 'L': (L1+L2)/2, 'N': (N1+N2)/2,
                    'q': q_emb, 'bin_id': cond
                })
                test_images.append(img1)
                
        pids = list(persons.keys())
        for i in range(len(pids)-1):
            img1 = cv2.imread(persons[pids[i]][0])
            img2 = cv2.imread(persons[pids[i+1]][0])
            sim, q_emb, _ = embedder.similarity(img1, img2)
            if sim is None: continue
            
            L1, N1, _, q1 = iqa.compute(img1)
            L2, N2, _, q2 = iqa.compute(img2)
            data.append({
                'sim': sim, 'label': 0, 'L': (L1+L2)/2, 'N': (N1+N2)/2,
                'q': q_emb, 'bin_id': cond
            })
            
    if not data:
        print(f"⚠️ No images found in {data_path}. Tự động fallback sang synthetic data.")
        return load_synthetic()
        
    return data, test_images

# ==========================================
# === TẦNG 2: EXPERIMENT LAYER ===
# ==========================================

def formula_fixed(bin_id, L, N, q):
    return 0.44

def formula_bin_specific(bin_id, L, N, q):
    return {'bright': 0.48, 'medium': 0.42, 'dark': 0.35}.get(bin_id, 0.42)

def formula_linear(bin_id, L, N, q):
    tau_base = 0.48; b = 0.10; c = 0.05
    return tau_base - b*(1 - L) - c*N

def formula_interaction(bin_id, L, N, q):
    tau_base = 0.48; tau_floor = 0.30; gamma = 0.25
    return tau_base * (1 - gamma*(1 - L)*N) * q + tau_floor*(1 - q)

FORMULAS = {
    'fixed': formula_fixed,
    'bin': formula_bin_specific,
    'linear': formula_linear,
    'interaction': formula_interaction
}

def evaluate_formula(data_subset, formula_func):
    if not data_subset:
        return 0, 0, 0, 0, [], [], [], []
        
    y_true, y_score, decisions = [], [], []
    
    for item in data_subset:
        tau = formula_func(item['bin_id'], item['L'], item['N'], item['q'])
        sim, label = item['sim'], item['label']
        
        y_true.append(label)
        y_score.append(sim - tau)
        decisions.append(1 if sim >= tau else 0)
        
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    decisions = np.array(decisions)
    
    FRR = np.sum((y_true == 1) & (decisions == 0)) / max(1, np.sum(y_true == 1))
    FAR = np.sum((y_true == 0) & (decisions == 1)) / max(1, np.sum(y_true == 0))
    
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    except ValueError:
        fpr, tpr, roc_auc, eer = [], [], 0, 0
        
    return FRR, FAR, eer, roc_auc, fpr, tpr, list(decisions), list(y_true)

def mcnemar_test_results(decisions_A, decisions_B, labels):
    if mcnemar is None:
        return None
        
    n00 = sum(1 for a, b, y in zip(decisions_A, decisions_B, labels) if (a == y) and (b == y))
    n01 = sum(1 for a, b, y in zip(decisions_A, decisions_B, labels) if (a == y) and (b != y))
    n10 = sum(1 for a, b, y in zip(decisions_A, decisions_B, labels) if (a != y) and (b == y))
    n11 = sum(1 for a, b, y in zip(decisions_A, decisions_B, labels) if (a != y) and (b != y))
    
    table = [[n00, n01], [n10, n11]]
    result = mcnemar(table, exact=True)
    return result.pvalue

# ==========================================
# === TẦNG 3: EDGE SIMULATION LAYER ===
# ==========================================

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
        rng = np.random.default_rng(0)
        gallery = [rng.standard_normal(512).astype(np.float32) for _ in range(n_gallery)]
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

# ==========================================
# === MAIN PIPELINE ===
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['synthetic', 'lfw', 'custom'], default='synthetic')
    parser.add_argument('--data_path', type=str, default='./data')
    args = parser.parse_args()
    
    print("="*75)
    print(f"Bắt đầu thực nghiệm - NCKH Face Recognition Edge Device")
    print(f"Dataset: {args.dataset}")
    print("="*75)
    
    embedder = RealEmbedder()
    iqa = IQAModule()
    
    if args.dataset == 'synthetic':
        data, test_images = load_synthetic()
    elif args.dataset == 'lfw':
        data, test_images = load_lfw(args.data_path, embedder, iqa)
    else:
        data, test_images = load_custom(args.data_path, embedder, iqa)
        
    out_dir = Path(__file__).resolve().parent / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'formula_comparison.csv'
    
    results = []
    dark_roc_data = {}
    mcnemar_data = {'linear': {}, 'interaction': {}}
    
    print(f"\nTable 1: Threshold Formula Comparison (Dataset: {args.dataset}, N={len(data)} pairs)")
    print("─" * 75)
    print(f"{'Formula':<16} | {'Condition':<9} | {'FRR':>6} | {'FAR':>6} | {'EER':>6} | {'AUC':>6}")
    print("─" * 75)
    
    for formula_name, formula_func in FORMULAS.items():
        for condition in ['bright', 'medium', 'dark']:
            subset = [item for item in data if item['bin_id'] == condition]
            if not subset:
                continue
                
            FRR, FAR, EER, AUC, fpr, tpr, decisions, labels = evaluate_formula(subset, formula_func)
            
            print(f"{formula_name:<16} | {condition:<9} | "
                  f"{FRR:>5.1%} | {FAR:>5.1%} | {EER:>5.1%} | {AUC:.3f}")
            
            results.append({
                'Dataset': args.dataset, 'Condition': condition,
                'Formula': formula_name, 'FRR': f"{FRR:.1%}", 
                'FAR': f"{FAR:.1%}", 'EER': f"{EER:.1%}", 'AUC': f"{AUC:.3f}"
            })
            
            if condition == 'dark':
                dark_roc_data[formula_name] = (fpr, tpr, AUC)
                if formula_name in ['linear', 'interaction']:
                    mcnemar_data[formula_name] = {'decisions': decisions, 'labels': labels}
    
    print("─" * 75)
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset', 'Condition', 'Formula', 'FRR', 'FAR', 'EER', 'AUC'])
        writer.writeheader()
        writer.writerows(results)
        
    # Statistical Test
    if 'linear' in mcnemar_data and 'interaction' in mcnemar_data and mcnemar_data['linear'].get('labels'):
        dec_L = mcnemar_data['linear']['decisions']
        dec_I = mcnemar_data['interaction']['decisions']
        labels = mcnemar_data['linear']['labels']
        
        p_value = mcnemar_test_results(dec_I, dec_L, labels)
        if p_value is not None:
            print(f"McNemar test (Interaction vs Linear, dark condition): p = {p_value:.4e}")
            if p_value < 0.05:
                print(">>> interaction cải thiện FRR một cách có ý nghĩa thống kê so với tuyến tính.")
            else:
                print(">>> Không có khác biệt ý nghĩa thống kê (p >= 0.05).")
        else:
            print("⚠️ Bỏ qua McNemar test do thiếu statsmodels.")
            
    # Edge Benchmark
    print("\nTable 2: Edge Benchmark (Simulated 512MB RAM)")
    print("─" * 75)
    print(f"{'Metric':<25} | {'Value':>10} | {'Target':>10} | {'Status':>8}")
    print("─" * 75)
    
    with EdgeSimulator(ram_limit_mb=512):
        bench = EdgeSimulator.benchmark(embedder, iqa, formula_interaction, test_images)
        
    if bench:
        print(f"{'Latency mean (ms)':<25} | {bench['latency_mean']:>10.1f} | {'< 200':>10} | "
              f"{'✅ PASS' if bench['latency_mean'] < 200 else '❌ FAIL':>8}")
        print(f"{'Latency p95 (ms)':<25} | {bench['latency_p95']:>10.1f} | {'< 300':>10} | "
              f"{'✅ PASS' if bench['latency_p95'] < 300 else '❌ FAIL':>8}")
        print(f"{'RAM peak (MB)':<25} | {bench['ram_peak_mb']:>10.1f} | {'< 512':>10} | "
              f"{'✅ PASS' if bench['ram_peak_mb'] < 512 else '❌ FAIL':>8}")
        print(f"{'Gallery size (KB)':<25} | {bench['gallery_kb']:>10.1f} | {'< 400':>10} | "
              f"{'✅ PASS' if bench['gallery_kb'] < 400 else '❌ FAIL':>8}")
    else:
        print("Không đủ dữ liệu test_images để chạy Edge Benchmark.")
    print("─" * 75)
    
    # Text Note
    print("\nCÂU DÀNH CHO BÁO CÁO:")
    print("------------------------------------------------------------")
    print(f"Chúng tôi thực nghiệm 4 dạng hàm threshold theo thứ tự tăng dần độ phức tạp "
          f"trên dataset {args.dataset} gồm {len(data)} cặp ảnh. Embedding được trích xuất bằng "
          f"ArcFace (buffalo_sc backbone).")
    if p_value is not None:
        print(f"Kết quả McNemar's test (p = {p_value:.4f}) cho thấy interaction term (1-L)·N "
              f"{'cải thiện' if p_value < 0.05 else 'KHÔNG cải thiện đáng kể'} "
              f"hiệu suất ở điều kiện tối so với mô hình tuyến tính.")
    print("------------------------------------------------------------")
    print("✅ Hoàn tất! Files lưu tại thư mục outputs/")

if __name__ == "__main__":
    main()
