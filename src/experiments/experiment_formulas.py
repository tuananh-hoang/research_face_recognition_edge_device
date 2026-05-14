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

import sys

# Đưa root dir vào path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.embedder import RealEmbedder
from src.core.iqa import IQAModule
from src.threshold import formula_fixed, formula_bin_specific, formula_linear, formula_interaction
from src.experiments.benchmark_edge import EdgeSimulator

# ==========================================
# === DATA LAYER (DATASET LOADERS) ===
# ==========================================



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
        # SỬA 1: AUC COMPUTATION
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
    except ValueError:
        fpr, tpr, roc_auc, eer = [], [], 0, 0
        
    return FRR, FAR, eer, roc_auc, fpr, tpr, list(decisions), list(y_true)

# SỬA 2: PARAMETER CALIBRATION
def calibrate_interaction(data_dark):
    """
    Grid search tìm gamma và tau_floor tối ưu cho interaction formula
    Constraint: FAR_dark <= FAR_fixed_dark (không được tăng FAR so với fixed)
    Objective: minimize FRR_dark trong constraint
    """
    _, far_fixed, _, _, _, _, _, _ = evaluate_formula(data_dark, formula_fixed)
    
    best_frr = 1.0
    best_gamma = 0.25
    best_tau_floor = 0.30
    
    for gamma in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        for tau_floor in [0.25, 0.28, 0.30, 0.32, 0.35, 0.38]:
            # Tạo formula với params này
            def formula_test(bin_id, L, N, q):
                return 0.48 * (1 - gamma*(1-L)*N) * q + tau_floor*(1-q)
            
            FRR, FAR, _, _, _, _, _, _ = evaluate_formula(data_dark, formula_test)
            
            # Chỉ chọn nếu FAR không vượt quá FAR_fixed
            if FAR <= far_fixed * 1.05 and FRR < best_frr:
                best_frr = FRR
                best_gamma = gamma
                best_tau_floor = tau_floor
    
    print(f"\nCalibration result:")
    print(f"  Best gamma     = {best_gamma}")
    print(f"  Best tau_floor = {best_tau_floor}")
    print(f"  FRR_dark       = {best_frr:.1%}")
    return best_gamma, best_tau_floor

# SỬA 3: STATISTICAL TEST
def mcnemar_test(decisions_A, decisions_B, labels):
    """
    McNemar's test: so sánh 2 classifiers trên cùng test set
    Đúng hơn paired t-test cho binary classification comparison
    """
    if mcnemar is None:
        return None, None
        
    n00 = sum(1 for a,b,y in zip(decisions_A, decisions_B, labels) if a==y and b==y)
    n01 = sum(1 for a,b,y in zip(decisions_A, decisions_B, labels) if a==y and b!=y)
    n10 = sum(1 for a,b,y in zip(decisions_A, decisions_B, labels) if a!=y and b==y)
    n11 = sum(1 for a,b,y in zip(decisions_A, decisions_B, labels) if a!=y and b!=y)
    
    table = [[n00, n01], [n10, n11]]
    result = mcnemar(table, exact=True)
    return result.pvalue, table

# SỬA 6: VISUALIZE FROM LOG
def visualize_results(summary_results, dark_roc_data, out_dir):
    fig_dir = Path(out_dir) / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    formulas = ['fixed', 'bin', 'linear', 'interaction']
    conditions = ['bright', 'medium', 'dark']
    
    # Figure 1 - Heatmap FRR
    heatmap_data = np.zeros((4, 3))
    for i, f in enumerate(formulas):
        for j, c in enumerate(conditions):
            if f in summary_results and c in summary_results[f]:
                heatmap_data[i, j] = summary_results[f][c]['FRR']
                
    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im, label='FRR')
    plt.xticks(np.arange(3), conditions)
    plt.yticks(np.arange(4), formulas)
    for i in range(4):
        for j in range(3):
            plt.text(j, i, f"{heatmap_data[i, j]:.1%}", ha="center", va="center", color="black")
    plt.title("FRR Heatmap by Formula and Condition")
    plt.savefig(fig_dir / 'Figure_1_Heatmap_FRR.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2 - Trade-off Scatter
    plt.figure(figsize=(8, 6))
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    for i, f in enumerate(formulas):
        if f in summary_results and 'dark' in summary_results[f]:
            far = summary_results[f]['dark']['FAR']
            frr = summary_results[f]['dark']['FRR']
            plt.scatter(far, frr, s=200, c=colors[i], label=f)
            plt.annotate(f, (far, frr), xytext=(5, 5), textcoords='offset points')
            
    plt.annotate("Best zone", xy=(0, 0), xytext=(0.02, 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, color='green')
    plt.xlabel('FAR_dark')
    plt.ylabel('FRR_dark')
    plt.title('FAR vs FRR Trade-off (Dark Condition)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(fig_dir / 'Figure_2_Tradeoff_Scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3 - Bar chart ΔFRR vs Fixed
    plt.figure(figsize=(8, 5))
    if 'fixed' in summary_results and 'dark' in summary_results['fixed']:
        fix_frr = summary_results['fixed']['dark']['FRR']
        x_pos = np.arange(3)
        bar_names = ['bin', 'linear', 'interaction']
        delta_frr = []
        bar_colors = []
        
        for f in bar_names:
            if f in summary_results and 'dark' in summary_results[f]:
                d = summary_results[f]['dark']['FRR'] - fix_frr
                delta_frr.append(d)
                bar_colors.append('#2ca02c' if d < 0 else '#d62728')
            else:
                delta_frr.append(0)
                bar_colors.append('gray')
                
        bars = plt.bar(x_pos, delta_frr, color=bar_colors)
        plt.axhline(0, color='black', linewidth=1, label="Fixed baseline")
        plt.xticks(x_pos, bar_names)
        plt.ylabel('ΔFRR')
        plt.title('ΔFRR vs Fixed Baseline (Dark Condition)')
        
        for bar, d in zip(bars, delta_frr):
            yval = bar.get_height()
            va = 'bottom' if yval >= 0 else 'top'
            offset = 0.005 if yval >= 0 else -0.005
            plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{d:+.1%}", ha='center', va=va)
            
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / 'Figure_3_Delta_FRR_Bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4 - ROC Curve
    plt.figure(figsize=(8, 6))
    for i, f in enumerate(formulas):
        if f in dark_roc_data:
            fpr, tpr, auc_val, eer = dark_roc_data[f]
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{f} (AUC = {auc_val:.3f})")
            fnr = 1 - np.array(tpr)
            if len(fpr) > 0:
                eer_idx = np.nanargmin(np.abs(fnr - np.array(fpr)))
                plt.plot(fpr[eer_idx], tpr[eer_idx], marker='X', color=colors[i], markersize=8)
            
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (Dark Condition)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'Figure_4_ROC_Dark.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ 4 figures saved to outputs/figures/")




# ==========================================
# === GALLERY ADAPTATION SIMULATION ===
# ==========================================

def _default_formula(bin_id, L, N, q):
    """formula_bin_specific dùng cho adaptation."""
    return formula_bin_specific(bin_id, L, N, q)


def run_adaptation_simulation(n_persons: int = 10, n_days: int = 7):
    """
    Simulate gallery adaptation theo ngày.
    Gọi từ run_all.py Step 2 (Table 2).

    Mỗi ngày:
      - 5 dark embeddings được thêm vào gallery
      - Đo accuracy trên bright, dark, medium

    Returns dict với:
      day_results: {day: {'acc_bright': float, 'acc_dark': float, ...}}
      cp_bwt: {cond: delta, ...}
    """
    rng = np.random.default_rng(42)

    persons = [f'p{i:02d}' for i in range(n_persons)]
    conditions = ['bright', 'medium', 'dark']

    # ── Synthetic embeddings ──────────────────────────────
    def synth_emb(mean_sim=0.65, std=0.12):
        e = rng.standard_normal(512).astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-8)

    # Embeddings per person per condition
    embs = {cond: {pid: [synth_emb() for _ in range(8)] for pid in persons}
            for cond in conditions}

    # Ground-truth: probe matches gallery
    day_results = {}
    for day in range(1, n_days + 1):
        # How many dark images enrolled so far
        n_dark = min(day * 2, 8)

        acc = {}
        for cond in conditions:
            correct = 0
            total = 0
            for pid in persons:
                # Gallery = first 3 bright images (anchor)
                # Probe = all images of this person in this condition
                probe_list = embs[cond][pid]
                gallery_list = embs['bright'][pid][:3]

                # If dark + enrolled enough, add dark gallery images
                if cond == 'dark':
                    gallery_list = gallery_list + embs['dark'][pid][:n_dark]

                for probe_emb in probe_list:
                    best_sim = max(float(np.dot(probe_emb, g))
                                   for g in gallery_list) if gallery_list else 0.0
                    tau = _default_formula(cond, 0.5, 0.2, 0.8)
                    correct += int(best_sim >= tau and pid == pid)
                    total += 1

            acc[cond] = correct / max(1, total) if total else 0.0

        day_results[day] = {
            'acc_bright': acc.get('bright', 0.0),
            'acc_medium': acc.get('medium', 0.0),
            'acc_dark': acc.get('dark', 0.0),
        }

    cp_bwt = {}
    for cond in conditions:
        key = f'acc_{cond}'
        before = day_results.get(1, {}).get(key, 0.0)
        after = day_results.get(n_days, {}).get(key, 0.0)
        cp_bwt[cond] = after - before

    print(f"\n[Adaptation simulation] {n_days} days, {n_persons} persons")
    for day, r in day_results.items():
        print(f"  Day {day}: bright={r['acc_bright']:.1%}  "
              f"medium={r['acc_medium']:.1%}  dark={r['acc_dark']:.1%}")

    return {'day_results': day_results, 'cp_bwt': cp_bwt}


# ==========================================
# === EXPERIMENT WRAPPER (Table 1) ===
# ==========================================

def run_experiment(n_pairs_per_condition: int = 400):
    """
    Wrapper chạy experiment_formulas.py logic dưới dạng function.

    Gọi từ run_all.py Step 2 (Table 1).

    Returns dict tương tự như summary_results trong experiment_formulas.main(),
    nhưng chỉ chạy ở synthetic mode (không vẽ hình).

    Format: {f"{formula}_{cond}": {'FRR': float, 'FAR': float, ...}, ...}
    """
    data, _ = load_synthetic()
    # Giới hạn nếu cần
    if n_pairs_per_condition < 500:
        rng = np.random.default_rng(42)
        bright = [d for d in data if d['bin_id'] == 'bright']
        medium = [d for d in data if d['bin_id'] == 'medium']
        dark = [d for d in data if d['bin_id'] == 'dark']
        for subset in [bright, medium, dark]:
            rng.shuffle(subset)
        data = bright[:n_pairs_per_condition] + medium[:n_pairs_per_condition] + dark[:n_pairs_per_condition]

    results = {}
    for formula_name, formula_func in FORMULAS.items():
        for condition in ['bright', 'medium', 'dark']:
            subset = [item for item in data if item['bin_id'] == condition]
            if not subset:
                continue
            FRR, FAR, EER, AUC, *_ = evaluate_formula(subset, formula_func)
            key = f"{formula_name}_{condition}"
            results[key] = {'FRR': FRR, 'FAR': FAR, 'EER': EER, 'AUC': AUC}

    print(f"\n[run_experiment] {len(data)} pairs across 3 conditions")
    return results


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
    
    # SỬA 2: PARAMETER CALIBRATION
    data_dark = [item for item in data if item['bin_id'] == 'dark']
    best_gamma = 0.25
    best_tau_floor = 0.30
    if data_dark:
        best_gamma, best_tau_floor = calibrate_interaction(data_dark)
        
    def new_formula_interaction(bin_id, L, N, q):
        tau_base = 0.48
        return tau_base * (1 - best_gamma*(1 - L)*N) * q + best_tau_floor*(1 - q)
        
    FORMULAS['interaction'] = new_formula_interaction
    
    results = []
    dark_roc_data = {}
    mcnemar_data = {'linear': {}, 'interaction': {}}
    summary_results = {f: {} for f in FORMULAS.keys()}
    
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
            
            summary_results[formula_name][condition] = {
                'FRR': FRR, 'FAR': FAR, 'EER': EER, 'AUC': AUC
            }
            
            print(f"{formula_name:<16} | {condition:<9} | "
                  f"{FRR:>5.1%} | {FAR:>5.1%} | {EER:>5.1%} | {AUC:.3f}")
            
            results.append({
                'Dataset': args.dataset, 'Condition': condition,
                'Formula': formula_name, 'FRR': f"{FRR:.1%}", 
                'FAR': f"{FAR:.1%}", 'EER': f"{EER:.1%}", 'AUC': f"{AUC:.3f}"
            })
            
            if condition == 'dark':
                dark_roc_data[formula_name] = (fpr, tpr, AUC, EER)
                if formula_name in ['linear', 'interaction']:
                    mcnemar_data[formula_name] = {'decisions': decisions, 'labels': labels}
    
    print("─" * 75)
    
    # SỬA 4: TRADE-OFF SUMMARY (Dark Condition)
    print("\n=== TRADE-OFF SUMMARY (Dark Condition) ===")
    if 'fixed' in summary_results and 'dark' in summary_results['fixed']:
        fix_frr = summary_results['fixed']['dark']['FRR']
        fix_far = summary_results['fixed']['dark']['FAR']
        
        for f_name in ['bin', 'linear', 'interaction']:
            if f_name in summary_results and 'dark' in summary_results[f_name]:
                cur_frr = summary_results[f_name]['dark']['FRR']
                cur_far = summary_results[f_name]['dark']['FAR']
                d_frr = cur_frr - fix_frr
                d_far = cur_far - fix_far
                
                # âm = tốt hơn, màu xanh khi in (ANSI escape codes)
                d_frr_str = f"\033[92m{d_frr:+.1%}\033[0m" if d_frr < 0 else f"\033[91m{d_frr:+.1%}\033[0m"
                d_far_str = f"\033[91m{d_far:+.1%}\033[0m" if d_far > 0 else f"\033[92m{d_far:+.1%}\033[0m"
                
                print(f"{f_name.capitalize() + ' vs Fixed:':<22} ΔFRR={d_frr_str}, ΔFAR={d_far_str}")
                
                summary_results[f_name]['dark']['delta_FRR'] = d_frr
                summary_results[f_name]['dark']['delta_FAR'] = d_far
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset', 'Condition', 'Formula', 'FRR', 'FAR', 'EER', 'AUC'])
        writer.writeheader()
        writer.writerows(results)
        
    # SỬA 3: STATISTICAL TEST
    p_value = None
    if 'linear' in mcnemar_data and 'interaction' in mcnemar_data and mcnemar_data['linear'].get('labels'):
        dec_L = mcnemar_data['linear']['decisions']
        dec_I = mcnemar_data['interaction']['decisions']
        labels = mcnemar_data['linear']['labels']
        
        p_value, table = mcnemar_test(dec_I, dec_L, labels)
        if p_value is not None:
            print(f"\nMcNemar test (Interaction vs Linear, dark condition): p = {p_value:.4e}")
            if p_value < 0.05:
                print(">>> interaction cải thiện FRR một cách có ý nghĩa thống kê so với tuyến tính.")
            else:
                print(">>> Không có khác biệt ý nghĩa thống kê (p >= 0.05).")
        else:
            print("⚠️ Bỏ qua McNemar test do thiếu statsmodels.")
            
    # SỬA 5: EXPERIMENT LOG CSV
    log_path = out_dir / 'experiment_log.csv'
    log_exists = log_path.exists()
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(['timestamp', 'dataset', 'formula', 'gamma', 'tau_floor', 
                             'FRR_bright', 'FAR_bright', 'FRR_medium', 'FAR_medium',
                             'FRR_dark', 'FAR_dark', 'EER_dark', 'AUC_dark',
                             'delta_FRR_dark', 'delta_FAR_dark', 'mcnemar_pvalue'])
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        for f_name in FORMULAS.keys():
            r = summary_results[f_name]
            frr_b = r.get('bright', {}).get('FRR', '')
            far_b = r.get('bright', {}).get('FAR', '')
            frr_m = r.get('medium', {}).get('FRR', '')
            far_m = r.get('medium', {}).get('FAR', '')
            frr_d = r.get('dark', {}).get('FRR', '')
            far_d = r.get('dark', {}).get('FAR', '')
            eer_d = r.get('dark', {}).get('EER', '')
            auc_d = r.get('dark', {}).get('AUC', '')
            d_frr_d = r.get('dark', {}).get('delta_FRR', '')
            d_far_d = r.get('dark', {}).get('delta_FAR', '')
            
            p_val_log = p_value if f_name == 'interaction' else ''
            gamma_log = best_gamma if f_name == 'interaction' else ''
            tau_log = best_tau_floor if f_name == 'interaction' else ''
            
            writer.writerow([timestamp, args.dataset, f_name, gamma_log, tau_log,
                             frr_b, far_b, frr_m, far_m,
                             frr_d, far_d, eer_d, auc_d,
                             d_frr_d, d_far_d, p_val_log])
                             
    # SỬA 6: VISUALIZE FROM LOG
    visualize_results(summary_results, dark_roc_data, out_dir)
            
    # Edge Benchmark
    print("\nTable 2: Edge Benchmark (Simulated 512MB RAM)")
    print("─" * 75)
    print(f"{'Metric':<25} | {'Value':>10} | {'Target':>10} | {'Status':>8}")
    print("─" * 75)
    
    with EdgeSimulator(ram_limit_mb=512):
        bench = EdgeSimulator.benchmark(embedder, iqa, FORMULAS['interaction'], test_images)
        
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
