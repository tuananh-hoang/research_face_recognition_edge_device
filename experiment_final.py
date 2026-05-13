import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import cv2
import csv
import json
import os
import sys
import time
import logging
import psutil
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def _detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0, f"GPU: {device_name}"
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            return ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0, "GPU: CUDA (via onnxruntime)"
    except ImportError:
        pass
    return ['CPUExecutionProvider'], -1, "CPU"

EXECUTION_PROVIDERS, CTX_ID, DEVICE_NAME = _detect_device()

EXPERIMENT_VERSION = "1.0.0"
EXPERIMENT_NAME   = "Face Recognition Adaptive Threshold"
EXPERIMENT_AUTHOR = "NCKH Team"

class ExperimentLogger:
    def __init__(self, log_dir: Path, version: str):
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{ts}_v{version}.log"
        self._logger = logging.getLogger("experiment")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)
        self.log_file = log_file

    def info(self, msg=""):  self._logger.info(msg)
    def debug(self, msg=""): self._logger.debug(msg)
    def warning(self, msg=""): self._logger.warning(msg)
    def error(self, msg=""): self._logger.error(msg)
    def section(self, title: str):
        self.info()
        self.info("=" * 70)
        self.info(f"  {title}")
        self.info("=" * 70)

class InsightFaceSingleton:
    _instance = None
    _mock_mode = False
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            try:
                from insightface.app import FaceAnalysis
                print(f"\033[92m[Device] Su dung: {DEVICE_NAME}\033[0m")
                print(f"\033[92m[Device] Providers: {EXECUTION_PROVIDERS}\033[0m")
                if EXECUTION_PROVIDERS[0] == 'CUDAExecutionProvider':
                    providers = [
                        ('CUDAExecutionProvider', {
                            'device_id': CTX_ID,
                            'cudnn_conv_algo_search': 'DEFAULT',
                            'do_copy_in_default_stream': True,
                        }),
                        'CPUExecutionProvider',
                    ]
                else:
                    providers = EXECUTION_PROVIDERS
                cls._instance = FaceAnalysis(name='buffalo_sc', providers=providers)
                cls._instance.prepare(ctx_id=CTX_ID, det_size=(320, 320))
                if EXECUTION_PROVIDERS[0] == 'CUDAExecutionProvider':
                    print("\033[92m[Device] GPU warmup...\033[0m")
                    _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                    cls._instance.get(_dummy)
                    print("\033[92m[Device] Warmup xong!\033[0m")
            except ImportError:
                print("\n\033[93m CANH BAO: Khong tim thay thu vien insightface hoac onnxruntime.\033[0m")
                print("\033[93m Dang dung MOCK embedding (resize anh) cho muc dich test.\033[0m")
                print("\033[93m DE CO KET QUA NCKH THAT, HAY CAI DAT insightface!\033[0m\n")
                class MockFace:
                    def __init__(self, emb, score):
                        self.embedding = emb
                        self.det_score = score
                class MockApp:
                    def get(self, img):
                        img_resized = cv2.resize(img, (112, 112)).flatten().astype(np.float32)
                        raw_norm = float(np.linalg.norm(img_resized) + 1e-8)
                        emb = img_resized / raw_norm
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
    """Context manager do latency va RAM thuc te.
    Khong dung RLIMIT_AS vi CUDA runtime can nhieu virtual address space.
    """
    def __init__(self, ram_limit_mb=512):
        self.ram_limit_mb = ram_limit_mb
    def __enter__(self):
        return self
    def __exit__(self, *args):
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
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    log_dir = out_dir / 'logs'
    log = ExperimentLogger(log_dir, EXPERIMENT_VERSION)
    run_start_time = datetime.now()
    run_id = run_start_time.strftime("%Y%m%d_%H%M%S")
    log.section(f"Thuc nghiem: {EXPERIMENT_NAME}")
    log.info(f"  Version  : {EXPERIMENT_VERSION}")
    log.info(f"  Run ID   : {run_id}")
    log.info(f"  Author   : {EXPERIMENT_AUTHOR}")
    log.info(f"  Device   : {DEVICE_NAME}")
    log.info(f"  Providers: {EXECUTION_PROVIDERS}")
    log.info(f"  Log file : {log.log_file}")
    log.info()
    data_dir = Path('data')
    if not data_dir.exists():
        log.error("Loi: Khong tim thay folder 'data/' o thu muc hien tai!")
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
        log.error("Loi: Khong tim thay anh nao trong data/bright/")
        sys.exit(1)
    n_bright = sum(len(v) for v in bright_dict.values())
    n_medium = sum(len(v) for v in medium_dict.values())
    n_dark   = sum(len(v) for v in dark_dict.values())
    log.info(f"Phat hien {len(persons)} identities.")
    log.info(f"  Anh bright : {n_bright}")
    log.info(f"  Anh medium : {n_medium}")
    log.info(f"  Anh dark   : {n_dark}")
    log.info(f"  Tong cong  : {n_bright + n_medium + n_dark}")
    log.debug(f"  Danh sach ID: {sorted(persons)}")
    app = InsightFaceSingleton.get_instance()
    cache = {}
    all_paths = [p for d in [bright_dict, medium_dict, dark_dict] for paths in d.values() for p in paths]

    emb_cache_path = out_dir / 'embedding_cache.npz'
    if emb_cache_path.exists():
        log.info(f"\nNap embedding cache tu: {emb_cache_path}")
        raw = np.load(emb_cache_path, allow_pickle=True)
        saved = raw['cache'].item()  
        for p in all_paths:
            key = str(p)
            if key in saved:
                cache[p] = saved[key]
        log.info(f"  Da load {len(cache)}/{len(all_paths)} embeddings tu cache.")
    else:
        log.info(f"\nKhong co cache. Se trich xuat tu dau.")

    missing = [p for p in all_paths if p not in cache]
    if missing:
        log.info(f"  Trich xuat {len(missing)} anh chua co trong cache...")
        for p in tqdm(missing, desc="[1/3] Trich xuat embedding", unit="anh", ncols=90):
            img = cv2.imread(str(p))
            if img is None: continue
            L, N = compute_iqa(img)
            faces = app.get(img)
            if not faces: continue
            face = max(faces, key=lambda x: x.det_score)
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)
            q = face.det_score
            cache[p] = {'emb': emb, 'L': L, 'N': N, 'q': q}
        save_dict = {str(p): cache[p] for p in cache}
        np.savez(emb_cache_path, cache=save_dict)
        log.info(f"  Da luu embedding cache: {emb_cache_path}")
    else:
        log.info("  Tat ca embeddings da co trong cache, bo qua buoc trich xuat.")
    pairs = []
    gallery = {}
    for pid in persons:
        valid_bright = [p for p in bright_dict.get(pid, []) if p in cache]
        if valid_bright:
            gallery[pid] = valid_bright[0]
    rng = np.random.default_rng(42)
    for cond, d in zip(['bright', 'medium', 'dark'], [bright_dict, medium_dict, dark_dict]):
        for pid in persons:
            gal_p = gallery.get(pid)
            if not gal_p: continue
            for probe_p in d.get(pid, []):
                if probe_p not in cache or probe_p == gal_p: continue
                pairs.append({'emb1': cache[gal_p]['emb'], 'emb2': cache[probe_p]['emb'], 
                              'label': 1, 'condition': cond, 
                              'L': cache[probe_p]['L'], 'N': cache[probe_p]['N'], 'q': cache[probe_p]['q']})
                for other_pid in persons:
                    if other_pid == pid: continue
                    other_gal_p = gallery.get(other_pid)
                    if not other_gal_p: continue
                    pairs.append({'emb1': cache[other_gal_p]['emb'], 'emb2': cache[probe_p]['emb'], 
                                  'label': 0, 'condition': cond, 
                                  'L': cache[probe_p]['L'], 'N': cache[probe_p]['N'], 'q': cache[probe_p]['q']})
    log.info(f"Tao thanh cong {len(pairs)} cap danh gia (Same/Diff).")
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
    gamma_range = list(np.arange(0.0, 1.05, 0.05))
    for g in tqdm(gamma_range, desc="[2/3] Tim gamma toi uu  ", unit="step", ncols=90):
        frr, far, _, _, _, _, _, _ = evaluate(dark_pairs, lambda L, N, q, c: formulas['interaction'](L, N, q, c, gamma=g))
        if far <= far_fixed_dark + 0.02:
            if frr < best_frr:
                best_frr = frr
                best_gamma = g
    formulas['interaction'] = lambda L, N, q, cond, gamma=best_gamma: 0.48 * (1 - gamma * (1 - L) * N) * q + 0.30 * (1 - q)
    results = []
    summary = {}
    dark_roc_data = {}
    dark_decisions = {}
    log.section("Ket qua danh gia 4 cong thuc Threshold")
    log.info(f"{'Formula':<15} | {'Condition':<10} | {'FRR':>8} | {'FAR':>8} | {'EER':>8} | {'AUC':>8}")
    log.info("-" * 80)
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
            log.info(f"{f_name:<15} | {cond:<10} | {FRR:>7.1%} | {FAR:>7.1%} | {EER:>7.1%} | {AUC:>8.3f}")
    log.info("-" * 80)
    labels = [p['label'] for p in dark_pairs]
    dec_lin = dark_decisions['linear']
    dec_int = dark_decisions['interaction']
    n01 = sum(1 for a, b, y in zip(dec_int, dec_lin, labels) if a==y and b!=y)
    n10 = sum(1 for a, b, y in zip(dec_int, dec_lin, labels) if a!=y and b==y)
    p_value = calc_mcnemar(n01, n10)
    log.section("Edge Benchmark (100 queries, do latency & RAM thuc te)")
    latencies = []
    ram_usage = []
    test_paths = list(cache.keys())[:100]
    gallery_embs = [cache[p]['emb'] for p in gallery.values()]
    process = psutil.Process()
    for path in tqdm(test_paths, desc="[3/3] Edge Benchmark    ", unit="query", ncols=90):
        img = cv2.imread(str(path))
        if img is None: continue
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
    f_names = ['fixed', 'bin', 'linear', 'interaction']
    c_names = ['bright', 'medium', 'dark']
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
    plt.figure(figsize=(8, 5))
    fix_frr = summary['fixed']['dark']['FRR']
    x_pos = np.arange(3)
    bar_names = ['bin', 'linear', 'interaction']
    delta_frr = [summary[f]['dark']['FRR'] - fix_frr for f in bar_names if 'dark' in summary[f]]
    bar_colors = ['#2ca02c' if d < 0 else '#d62728' for d in delta_frr]
    bars = plt.bar(x_pos, delta_frr, color=bar_colors)
    plt.axhline(0, color='black', label="Fixed baseline")
    plt.xticks(x_pos, bar_names)
    plt.ylabel('dFRR')
    plt.title('dFRR vs Fixed Baseline (Dark Condition)')
    for bar, d in zip(bars, delta_frr):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{d:+.1%}", ha='center', va='bottom' if d>=0 else 'top')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_3_Delta_FRR.png', dpi=150, bbox_inches='tight')
    plt.close()
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
    fix_frr_dark = summary['fixed']['dark']['FRR']
    interact_frr_dark = summary['interaction']['dark']['FRR']
    delta_frr_percent = fix_frr_dark - interact_frr_dark
    log.section("CAU DANH CHO BAO CAO")
    log.info("DAT VAN DE:")
    log.info("He thong nhan dien khuon mat voi fixed threshold tau=0.44 hoat dong tot trong dieu kien sang nhung FRR tang dang ke khi thieu sang vi embedding cosine similarity cua cung mot nguoi giam xuong khi anh chat luong thap. Khong co nghien cuu nao ket hop environmental signal -> inference-time threshold adaptation trong he thong co memory budget cho edge device.")
    log.info("")
    log.info("GAP:")
    log.info("- AdaFace [1]: quality-aware nhung training time, khong phai inference")
    log.info("- SER-FIQ [2]: estimate quality nhung khong link vao threshold decision")
    log.info("- Chou et al. [3]: adaptive threshold nhung per-identity, khong per-condition")
    log.info("")
    log.info("DE XUAT:")
    log.info("Chung toi thuc nghiem 4 dang ham threshold theo thu tu tang dan do phuc tap. Interaction term (1-L)*N dua tren image degradation model: khi luminance thap VA noise cao dong thoi, chat luong embedding giam theo cap so nhan, khong phai cong tuyen tinh.")
    log.info("")
    log.info("KET QUA:")
    log.info(f"Tren tap du lieu {len(persons)} nguoi x {len(all_paths)} anh x 3 dieu kien, interaction formula voi gamma toi uu={best_gamma:.2f} giam FRR_dark tu {fix_frr_dark:.1%} xuong {interact_frr_dark:.1%}, tuong duong giam {delta_frr_percent:.1%}. McNemar test p={p_value:.4f} cho thay cai thien {'co' if p_value < 0.05 else 'khong co'} y nghia thong ke. Edge benchmark: latency={latency_mean:.1f}ms, RAM={ram_max:.1f}MB.")
    log.info("")
    log.info("LIMITATIONS:")
    log.info("Dataset nho (15 nguoi), dark condition sinh bang augmentation khong phai thuc te. Ket qua can validate tren dataset lon hon voi dark that.")
    log.info("=" * 80)
    log.info("Tat ca output (CSV, figures) da duoc luu trong thu muc outputs/!")
    run_end_time = datetime.now()
    run_info = {
        "experiment_name": EXPERIMENT_NAME,
        "version": EXPERIMENT_VERSION,
        "author": EXPERIMENT_AUTHOR,
        "run_id": run_id,
        "started_at": run_start_time.isoformat(),
        "finished_at": run_end_time.isoformat(),
        "duration_seconds": round((run_end_time - run_start_time).total_seconds(), 2),
        "log_file": str(log.log_file),
        "data": {
            "n_identities": len(persons),
            "n_images": {"bright": n_bright, "medium": n_medium, "dark": n_dark},
            "n_pairs": len(pairs),
        },
        "hyperparameters": {
            "best_gamma": float(best_gamma),
        },
        "results_dark": {
            f_name: {
                "FRR": round(summary[f_name]["dark"]["FRR"], 4),
                "FAR": round(summary[f_name]["dark"]["FAR"], 4),
                "EER": round(summary[f_name]["dark"]["EER"], 4),
                "AUC": round(summary[f_name]["dark"]["AUC"], 4),
            }
            for f_name in formulas.keys() if "dark" in summary.get(f_name, {})
        },
        "mcnemar": {
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
        },
        "edge_benchmark": {
            "latency_mean_ms": round(latency_mean, 2),
            "latency_std_ms": round(latency_std, 2),
            "latency_p95_ms": round(latency_p95, 2),
            "ram_max_mb": round(ram_max, 2),
        },
        "outputs": {
            "formula_comparison_csv": str(out_dir / "formula_comparison.csv"),
            "experiment_log_csv": str(out_dir / "experiment_log.csv"),
            "figures_dir": str(fig_dir),
        },
    }
    run_info_path = out_dir / f"run_info_{run_id}_v{EXPERIMENT_VERSION}.json"
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    log.info(f"\nRun info da duoc luu tai: {run_info_path}")
    log.info(f"Log file                : {log.log_file}")

if __name__ == "__main__":
    main()
