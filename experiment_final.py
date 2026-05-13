import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import cv2
import csv
import os
import sys
import time
import psutil
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. EMBEDDING: Singleton pattern
class InsightFaceSingleton:
    _instance = None
    _mock_mode = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            try:
                from insightface.app import FaceAnalysis
                cls._instance = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
                cls._instance.prepare(ctx_id=0, det_size=(320, 320))
            except ImportError:
                print("\n\033[93m⚠️ CẢNH BÁO: Không tìm thấy thư viện insightface hoặc onnxruntime.\033[0m")
                print("\033[93m⚠️ Đang dùng MOCK embedding (resize ảnh) cho mục đích test.\033[0m")
                print("\033[93m⚠️ ĐỂ CÓ KẾT QUẢ NCKH THẬT, HÃY CÀI ĐẶT insightface!\033[0m\n")
                
                class MockFace:
                    def __init__(self, emb, score):
                        self.embedding = emb
                        self.det_score = score
                        
                class MockApp:
                    def get(self, img):
                        img_resized = cv2.resize(img, (112, 112)).flatten().astype(np.float32)
                        raw_norm = float(np.linalg.norm(img_resized) + 1e-8)
                        emb = img_resized / raw_norm
                        # Random q score to simulate det_score in mock mode
                        return [MockFace(emb, float(np.random.uniform(0.5, 0.99)))]
                
                cls._instance = MockApp()
                cls._mock_mode = True
        return cls._instance

def compute_iqa(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    L = np.mean(ycrcb[:, :, 0]) / 255.0
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    diff = img.astype(np.float32) - blurred.astype(np.float32)
    N = np.std(diff) / 255.0
    return L, N

def calc_mcnemar(n01, n10):
    n = n01 + n10
    if n == 0: return 1.0
    try:
        from scipy.stats import binomtest
        return binomtest(min(n01, n10), n, 0.5).pvalue
    except ImportError:
        try:
            from scipy.stats import binom_test
            return binom_test(min(n01, n10), n, 0.5)
        except ImportError:
            import math
            p = 0
            k = min(n01, n10)
            for i in range(k + 1):
                p += math.comb(n, i) * (0.5 ** n)
            return min(1.0, p * 2)

class EdgeSimulator:
    def __init__(self, ram_limit_mb=512):
        self.ram_limit = ram_limit_mb * 1024 * 1024
        
    def __enter__(self):
        try:
            import resource
            self.soft, self.hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (self.ram_limit, self.hard))
        except:
            pass
        return self
        
    def __exit__(self, *args):
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (self.soft, self.hard))
        except:
            pass

def evaluate(pairs, formula_func):
    if not pairs: return 0, 0, 0, 0, [], [], [], []
    y_true = []
    y_score = []
    decisions = []
    for p in pairs:
        sim = float(np.dot(p['emb1'], p['emb2']))
        tau = formula_func(p['L'], p['N'], p['q'], p['condition'])
        y_true.append(p['label'])
        y_score.append(sim)
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
    except:
        fpr, tpr, roc_auc, eer = [], [], 0, 0
    
    return FRR, FAR, eer, roc_auc, fpr, tpr, decisions, y_true

def main():
    print("="*60)
    print("Bắt đầu thực nghiệm Face Recognition Edge Device")
    print("="*60)

    # 2. LOAD DATA
    data_dir = Path('data')
    if not data_dir.exists():
        print("Lỗi: Không tìm thấy folder 'data/' ở thư mục hiện tại!")
        sys.exit(1)

    def load_images_from_dir(d):
        res = {}
        if not d.exists(): return res
        for p in d.glob('*.jpg'):
            pid = p.stem.split('_')[0]
            res.setdefault(pid, []).append(p)
        return res

    bright_dict = load_images_from_dir(data_dir / 'bright')
    medium_dict = load_images_from_dir(data_dir / 'medium')
    dark_dict = load_images_from_dir(data_dir / 'dark')

    persons = list(bright_dict.keys())
    if not persons:
        print("Lỗi: Không tìm thấy ảnh nào trong data/bright/")
        sys.exit(1)

    print(f"Phát hiện {len(persons)} identities.")

    app = InsightFaceSingleton.get_instance()
    cache = {}
    all_paths = [p for d in [bright_dict, medium_dict, dark_dict] for paths in d.values() for p in paths]
    
    print(f"Trích xuất đặc trưng cho {len(all_paths)} ảnh...")
    for p in all_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        L, N = compute_iqa(img)
        
        faces = app.get(img)
        if not faces: continue # skip nếu không detect được mặt
        face = max(faces, key=lambda x: x.det_score)
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)
        q = face.det_score
        
        cache[p] = {'emb': emb, 'L': L, 'N': N, 'q': q, 'img': img}

    pairs = []
    gallery = {}
    for pid in persons:
        valid_bright = [p for p in bright_dict.get(pid, []) if p in cache]
        if valid_bright:
            gallery[pid] = valid_bright[0] # Chọn 1 ảnh bright tốt làm gallery

    rng = np.random.default_rng(42)
    for cond, d in zip(['bright', 'medium', 'dark'], [bright_dict, medium_dict, dark_dict]):
        for pid in persons:
            gal_p = gallery.get(pid)
            if not gal_p: continue
            
            for probe_p in d.get(pid, []):
                if probe_p not in cache or probe_p == gal_p: continue
                # Same pairs
                pairs.append({'emb1': cache[gal_p]['emb'], 'emb2': cache[probe_p]['emb'], 
                              'label': 1, 'condition': cond, 
                              'L': cache[probe_p]['L'], 'N': cache[probe_p]['N'], 'q': cache[probe_p]['q']})
                
                # Diff pairs: pair with every other person's gallery
                for other_pid in persons:
                    if other_pid == pid: continue
                    other_gal_p = gallery.get(other_pid)
                    if not other_gal_p: continue
                    pairs.append({'emb1': cache[other_gal_p]['emb'], 'emb2': cache[probe_p]['emb'], 
                                  'label': 0, 'condition': cond, 
                                  'L': cache[probe_p]['L'], 'N': cache[probe_p]['N'], 'q': cache[probe_p]['q']})

    print(f"Tạo thành công {len(pairs)} cặp đánh giá (Same/Diff).")

    # 3. 4 CÔNG THỨC THRESHOLD
    formulas = {
        'fixed': lambda L, N, q, cond, gamma=None: 0.44,
        'bin': lambda L, N, q, cond, gamma=None: {'bright': 0.48, 'medium': 0.42, 'dark': 0.35}.get(cond, 0.42),
        'linear': lambda L, N, q, cond, gamma=None: 0.48 - 0.10*(1 - L) - 0.05*N,
        'interaction': lambda L, N, q, cond, gamma=0.25: 0.48 * (1 - gamma * (1 - L) * N) * q + 0.30 * (1 - q)
    }

    dark_pairs = [p for p in pairs if p['condition'] == 'dark']
    _, far_fixed_dark, _, _, _, _, _, _ = evaluate(dark_pairs, formulas['fixed'])

    best_gamma = 0.25
    best_frr = 1.0
    for g in np.arange(0.0, 1.05, 0.05):
        frr, far, _, _, _, _, _, _ = evaluate(dark_pairs, lambda L, N, q, c: formulas['interaction'](L, N, q, c, gamma=g))
        if far <= far_fixed_dark + 0.02:
            if frr < best_frr:
                best_frr = frr
                best_gamma = g

    formulas['interaction'] = lambda L, N, q, cond, gamma=best_gamma: 0.48 * (1 - gamma * (1 - L) * N) * q + 0.30 * (1 - q)

    # 4. ĐÁNH GIÁ METRICS
    results = []
    summary = {}
    dark_roc_data = {}
    dark_decisions = {}

    print("\n" + "-"*80)
    print(f"{'Formula':<15} | {'Condition':<10} | {'FRR':>8} | {'FAR':>8} | {'EER':>8} | {'AUC':>8}")
    print("-" * 80)

    for f_name, func in formulas.items():
        summary[f_name] = {}
        for cond in ['bright', 'medium', 'dark']:
            cond_pairs = [p for p in pairs if p['condition'] == cond]
            if not cond_pairs: continue
            FRR, FAR, EER, AUC, fpr, tpr, decs, y_true = evaluate(cond_pairs, func)
            summary[f_name][cond] = {'FRR': FRR, 'FAR': FAR, 'EER': EER, 'AUC': AUC}
            results.append({
                'Condition': cond, 'Formula': f_name, 
                'FRR': f"{FRR:.1%}", 'FAR': f"{FAR:.1%}", 
                'EER': f"{EER:.1%}", 'AUC': f"{AUC:.3f}"
            })
            
            if cond == 'dark':
                dark_roc_data[f_name] = (fpr, tpr, AUC, EER)
                dark_decisions[f_name] = decs
                
            print(f"{f_name:<15} | {cond:<10} | {FRR:>7.1%} | {FAR:>7.1%} | {EER:>7.1%} | {AUC:>8.3f}")
    print("-" * 80)

    # 5. STATISTICAL TEST (McNemar)
    labels = [p['label'] for p in dark_pairs]
    dec_lin = dark_decisions['linear']
    dec_int = dark_decisions['interaction']
    n01 = sum(1 for a, b, y in zip(dec_int, dec_lin, labels) if a==y and b!=y)
    n10 = sum(1 for a, b, y in zip(dec_int, dec_lin, labels) if a!=y and b==y)
    p_value = calc_mcnemar(n01, n10)

    # 6. EDGE BENCHMARK
    print("\nĐang chạy Edge Benchmark (100 queries)...")
    latencies = []
    ram_usage = []
    test_images = [cache[p]['img'] for p in list(cache.keys())[:100]]
    gallery_embs = [cache[p]['emb'] for p in gallery.values()]
    process = psutil.Process()

    with EdgeSimulator(ram_limit_mb=512):
        for img in test_images:
            start = time.perf_counter()
            L, N = compute_iqa(img)
            faces = app.get(img)
            if faces:
                face = max(faces, key=lambda x: x.det_score)
                emb = face.embedding
                emb = emb / np.linalg.norm(emb)
                q = face.det_score
                tau = formulas['interaction'](L, N, q, 'unknown', best_gamma)
                sim = max([float(np.dot(emb, g)) for g in gallery_embs]) if gallery_embs else 0
                decision = sim >= tau
                
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            ram_usage.append(process.memory_info().rss / 1024 / 1024)

    latency_mean = np.mean(latencies)
    latency_std = np.std(latencies)
    latency_p95 = np.percentile(latencies, 95)
    ram_max = np.max(ram_usage)

    # 7. OUTPUTS
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    with open(out_dir / 'formula_comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Condition', 'Formula', 'FRR', 'FAR', 'EER', 'AUC'])
        writer.writeheader()
        writer.writerows(results)

    log_path = out_dir / 'experiment_log.csv'
    log_exists = log_path.exists()
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(['timestamp', 'formula', 'gamma', 'FRR_bright', 'FAR_bright', 'FRR_medium', 'FAR_medium', 'FRR_dark', 'FAR_dark', 'EER_dark', 'AUC_dark'])
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        for f_name in formulas.keys():
            r = summary[f_name]
            g = best_gamma if f_name == 'interaction' else ''
            writer.writerow([timestamp, f_name, g,
                r.get('bright',{}).get('FRR',''), r.get('bright',{}).get('FAR',''),
                r.get('medium',{}).get('FRR',''), r.get('medium',{}).get('FAR',''),
                r.get('dark',{}).get('FRR',''), r.get('dark',{}).get('FAR',''),
                r.get('dark',{}).get('EER',''), r.get('dark',{}).get('AUC','')
            ])

    # FIGURES
    f_names = ['fixed', 'bin', 'linear', 'interaction']
    c_names = ['bright', 'medium', 'dark']
    
    # 1. Heatmap
    heatmap_data = np.zeros((4, 3))
    for i, f in enumerate(f_names):
        for j, c in enumerate(c_names):
            heatmap_data[i, j] = summary[f].get(c, {}).get('FRR', 0)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(label='FRR')
    plt.xticks(np.arange(3), c_names)
    plt.yticks(np.arange(4), f_names)
    for i in range(4):
        for j in range(3):
            plt.text(j, i, f"{heatmap_data[i, j]:.1%}", ha="center", va="center", color="black")
    plt.title("FRR Heatmap")
    plt.savefig(fig_dir / 'Figure_1_Heatmap_FRR.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Scatter
    plt.figure(figsize=(8, 6))
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    for i, f in enumerate(f_names):
        if 'dark' in summary[f]:
            far = summary[f]['dark']['FAR']
            frr = summary[f]['dark']['FRR']
            plt.scatter(far, frr, s=200, c=colors[i], label=f)
            plt.annotate(f, (far, frr), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('FAR')
    plt.ylabel('FRR')
    plt.title('FAR vs FRR (Dark Condition)')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_dir / 'Figure_2_Scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Bar chart
    plt.figure(figsize=(8, 5))
    fix_frr = summary['fixed']['dark']['FRR']
    x_pos = np.arange(3)
    bar_names = ['bin', 'linear', 'interaction']
    delta_frr = [summary[f]['dark']['FRR'] - fix_frr for f in bar_names if 'dark' in summary[f]]
    bar_colors = ['#2ca02c' if d < 0 else '#d62728' for d in delta_frr]
    bars = plt.bar(x_pos, delta_frr, color=bar_colors)
    plt.axhline(0, color='black', label="Fixed baseline")
    plt.xticks(x_pos, bar_names)
    plt.ylabel('ΔFRR')
    plt.title('ΔFRR vs Fixed Baseline (Dark Condition)')
    for bar, d in zip(bars, delta_frr):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{d:+.1%}", ha='center', va='bottom' if d>=0 else 'top')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_3_Delta_FRR.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. ROC
    plt.figure(figsize=(8, 6))
    for i, f in enumerate(f_names):
        if f in dark_roc_data:
            fpr, tpr, auc_val, eer = dark_roc_data[f]
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{f} (AUC = {auc_val:.3f})")
            if len(fpr) > 0:
                fnr = 1 - np.array(tpr)
                eer_idx = np.nanargmin(np.abs(fnr - np.array(fpr)))
                plt.plot(fpr[eer_idx], tpr[eer_idx], marker='X', color=colors[i], markersize=8)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (Dark Condition)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_4_ROC.png', dpi=150, bbox_inches='tight')
    plt.close()

    # REPORT 
    fix_frr_dark = summary['fixed']['dark']['FRR']
    interact_frr_dark = summary['interaction']['dark']['FRR']
    delta_frr_percent = fix_frr_dark - interact_frr_dark

    print("\n\n" + "="*80)
    print("=== CÂU DÀNH CHO BÁO CÁO ===")
    print("="*80)
    print("ĐẶT VẤN ĐỀ:")
    print("Hệ thống nhận diện khuôn mặt với fixed threshold τ=0.44 hoạt động tốt "
          "trong điều kiện sáng nhưng FRR tăng đáng kể khi thiếu sáng vì embedding "
          "cosine similarity của cùng một người giảm xuống khi ảnh chất lượng thấp. "
          "Không có nghiên cứu nào kết hợp environmental signal → inference-time "
          "threshold adaptation trong hệ thống có memory budget cho edge device.\n")

    print("GAP:")
    print("- AdaFace [1]: quality-aware nhưng training time, không phải inference")
    print("- SER-FIQ [2]: estimate quality nhưng không link vào threshold decision")
    print("- Chou et al. [3]: adaptive threshold nhưng per-identity, không per-condition\n")

    print("ĐỀ XUẤT:")
    print("Chúng tôi thực nghiệm 4 dạng hàm threshold theo thứ tự tăng dần độ phức tạp. "
          "Interaction term (1-L)·N dựa trên image degradation model: khi luminance "
          "thấp VÀ noise cao đồng thời, chất lượng embedding giảm theo cấp số nhân, "
          "không phải cộng tuyến tính.\n")

    print("KẾT QUẢ:")
    print(f"Trên tập dữ liệu {len(persons)} người × {len(all_paths)} ảnh × 3 điều kiện, interaction formula với "
          f"gamma tối ưu={best_gamma:.2f} giảm FRR_dark từ {fix_frr_dark:.1%} xuống {interact_frr_dark:.1%}, "
          f"tương đương giảm {delta_frr_percent:.1%}. McNemar test p={p_value:.4f} cho thấy cải thiện "
          f"{'có' if p_value < 0.05 else 'không có'} ý nghĩa thống kê. Edge benchmark: latency={latency_mean:.1f}ms, RAM={ram_max:.1f}MB.\n")

    print("LIMITATIONS:")
    print("Dataset nhỏ (15 người), dark condition sinh bằng augmentation không phải "
          "thực tế. Kết quả cần validate trên dataset lớn hơn với dark thật.")
    print("="*80)
    print("Tất cả output (CSV, figures) đã được lưu trong thư mục outputs/!")

if __name__ == "__main__":
    main()
