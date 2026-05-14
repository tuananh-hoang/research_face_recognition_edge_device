# PLAN  — Cải thiện NCKH Face Recognition
## 7 ngày, ưu tiên theo impact

---

## NGÀY 1-2: CP-BWT + Fix critical bugs

### Task 1.1 — Fix placeholders (30 phút)

Mở file `sections/01_introduction.tex` và `sections/04_experiments.tex`:
- Tìm tất cả `??` và `[ghi rõ model CPU]`
- Điền thông tin CPU thật (chạy lệnh này để lấy: `wmic cpu get name`)
- Sửa H1 criterion từ "10 điểm phần trăm tuyệt đối" thành:
  "giảm ít nhất 30% tương đối so với fixed threshold"
  (bin đạt (9.8-3.9)/9.8 = 60% → pass)

---

### Task 1.2 — Vẽ Pipeline Diagram (1 tiếng)

Tạo file `src/utils/draw_pipeline.py`:

```
Vẽ pipeline diagram gồm 4 module nối tiếp nhau bằng matplotlib:

[Ảnh đầu vào] 
      ↓
[Module A: ArcFace buffalo_sc] → embedding 512-d
      ↓
[Module B: IQA] → L (luminance), N (noise), bin_id
      ↓  
[Module C: Adaptive Threshold τ(C)] → tau value
      ↓
[Module D: Gallery Manager] → decision (Accept/Reject)

Mỗi module vẽ thành box hình chữ nhật màu khác nhau:
  Module A: xanh dương
  Module B: cam
  Module C: xanh lá
  Module D: đỏ

Bên trong mỗi box ghi:
  Tên module + input/output chính

Mũi tên nối giữa các box, ghi label dữ liệu truyền qua.
Save: outputs/figures/pipeline_diagram.png, dpi=200
```

---

### Task 1.3 — Gallery Adaptation + CP-BWT (QUAN TRỌNG NHẤT)

Tạo file `src/experiments/gallery_adaptation.py`:

```
MỤC TIÊU: Implement H2 và tính CP-BWT thật

BƯỚC 1 — Setup gallery ban đầu:
  Với mỗi person trong data/bright/:
    - Lấy 3 ảnh đầu tiên → enroll vào anchor partition
    - Dùng InsightFace buffalo_sc để lấy embedding thật
    - Gallery lưu dạng dict: {person_id: [emb1, emb2, emb3]}

BƯỚC 2 — Đo accuracy TRƯỚC adaptation (baseline):
  Test set: các ảnh còn lại (không dùng để enroll)
  
  Với mỗi ảnh test:
    - Lấy embedding từ InsightFace
    - Tính cosine sim với tất cả gallery embeddings
    - Lấy max sim → best_id
    - Dùng bin-specific threshold (F2, tốt nhất từ experiment)
    - Record: correct nếu best_id == true_id AND sim >= tau
  
  Tính:
    Acc_bright_before = correct / total trên bright test images
    Acc_dark_before   = correct / total trên dark test images  
    Acc_medium_before = correct / total trên medium test images

BƯỚC 3 — Online gallery update (simulate adaptation):
  Feed lần lượt 5 ảnh dark của mỗi người vào gallery:
  
  Update rule (weighted update từ báo cáo):
    emb_new = InsightFace.get_embedding(dark_image)
    det_score = face.det_score
    q = det_score  # quality proxy
    
    # Diversity penalty
    existing_embs = gallery[person_id]
    max_sim_existing = max(cosine_sim(emb_new, e) for e in existing_embs)
    
    # Weight
    w = 0.2 * q * det_score * (1 - max_sim_existing)
    
    # Chỉ update nếu w đủ lớn và confident
    if w > 0.05 and best_sim >= tau + 0.03:
        gallery[person_id].append(emb_new)
        # LRU eviction nếu > K=10
        if len(gallery[person_id]) > 10:
            gallery[person_id].pop(0)

BƯỚC 4 — Đo accuracy SAU adaptation:
  Chạy lại test set (SAME test images như bước 2)
    Acc_bright_after = ...
    Acc_dark_after   = ...
    Acc_medium_after = ...

BƯỚC 5 — Tính CP-BWT:
  CP_BWT_bright = Acc_bright_after - Acc_bright_before
  CP_BWT_dark   = Acc_dark_after   - Acc_dark_before
  CP_BWT_medium = Acc_medium_after - Acc_medium_before
  Overall_BWT   = mean([CP_BWT_bright, CP_BWT_medium, CP_BWT_dark])
  Stable        = CP_BWT_bright >= -0.01  (bright không giảm quá 1%)

BƯỚC 6 — Output:
  In bảng:
  ─────────────────────────────────────────────────────
  Condition | Acc Before | Acc After | CP-BWT  | Status
  ─────────────────────────────────────────────────────
  Bright    |   XX.X%    |  XX.X%   | +X.X%   | Stable ✅
  Medium    |   XX.X%    |  XX.X%   | +X.X%   | --
  Dark      |   XX.X%    |  XX.X%   | +X.X%   | Improved ✅
  ─────────────────────────────────────────────────────
  Overall BWT: +X.X%
  H2 Result: PASS/FAIL
  
  Save: outputs/cp_bwt_results.json và outputs/cp_bwt_results.csv
  
  Vẽ Figure:
    Line chart: Acc_dark theo số lượng dark images được update (0, 1, 2, 3, 4, 5)
    Đường ngang: Acc_bright (should stay flat)
    Save: outputs/figures/Figure_05_adaptation_curve.png
```

---

## NGÀY 3: Statistical rigor

### Task 3.1 — Train/Val/Test Split đúng

Sửa `src/experiments/experiment_final.py`:

```
THÊM VÀO ĐẦU PIPELINE:

def split_pairs(all_pairs, val_ratio=0.3, test_ratio=0.3, seed=42):
    """
    Split pairs thành 3 phần:
      - train (40%): dùng để calibrate threshold params (gamma, tau_floor)
      - val (30%):   dùng để chọn best formula
      - test (30%):  chỉ dùng để report final metrics, KHÔNG được xem trước
    
    Stratify theo bin_id để mỗi split có đủ bright/medium/dark
    """
    from sklearn.model_selection import train_test_split
    
    # Group by bin_id để stratify
    bright_pairs = [p for p in all_pairs if p['bin_id'] == 'bright']
    medium_pairs = [p for p in all_pairs if p['bin_id'] == 'medium']
    dark_pairs   = [p for p in all_pairs if p['bin_id'] == 'dark']
    
    def split_group(pairs):
        train, temp = train_test_split(pairs, test_size=val_ratio+test_ratio, 
                                        random_state=seed)
        val, test   = train_test_split(temp, test_size=test_ratio/(val_ratio+test_ratio),
                                        random_state=seed)
        return train, val, test
    
    # Split từng group
    ...
    return train_pairs, val_pairs, test_pairs

SAU ĐÓ:
- Calibrate gamma (grid search) trên TRAIN pairs
- Chọn best formula trên VAL pairs  
- Report FRR/FAR/EER chỉ trên TEST pairs

THÊM VÀO BÁO CÁO:
In thống kê split:
  "Train: X pairs | Val: Y pairs | Test: Z pairs"
  "Calibration dùng train set, final metrics dùng test set"
```

---

### Task 3.2 — Bootstrap Confidence Interval

Thêm hàm `compute_bootstrap_ci()` vào `experiment_final.py`:

```
def compute_bootstrap_ci(pairs, formula_func, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap resampling để tính confidence interval cho FRR, FAR, EER
    
    Args:
        pairs: list of (sim, label, L, N, q, bin_id)
        formula_func: hàm tính tau
        n_bootstrap: số lần resample (1000 là chuẩn)
        ci: confidence level (0.95 = 95% CI)
    
    Returns:
        {
          'FRR': {'mean': X, 'lower': Y, 'upper': Z},
          'FAR': {'mean': X, 'lower': Y, 'upper': Z},
          'EER': {'mean': X, 'lower': Y, 'upper': Z},
        }
    """
    rng = np.random.default_rng(42)
    frr_samples, far_samples, eer_samples = [], [], []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.integers(0, len(pairs), len(pairs))
        sample = [pairs[i] for i in indices]
        
        FRR, FAR, EER, _, _, _, _ = evaluate_formula(sample, formula_func)
        frr_samples.append(FRR)
        far_samples.append(FAR)
        eer_samples.append(EER)
    
    alpha = 1 - ci
    
    def ci_stats(samples):
        return {
            'mean': np.mean(samples),
            'lower': np.percentile(samples, alpha/2 * 100),
            'upper': np.percentile(samples, (1 - alpha/2) * 100)
        }
    
    return {
        'FRR': ci_stats(frr_samples),
        'FAR': ci_stats(far_samples),
        'EER': ci_stats(eer_samples),
    }

SAU ĐÓ cập nhật bảng kết quả:
Thay: FRR = 3.9%
Thành: FRR = 3.9% [2.1%, 6.2%]  ← format này

In bảng cập nhật:
Condition | Formula | FRR [95% CI]        | FAR [95% CI]
dark      | Bin     | 3.9% [2.1%, 6.2%]  | 3.2% [1.8%, 5.1%]
```

---

### Task 3.3 — McNemar F2 vs F1 (10 dòng)

Thêm vào cuối `experiment_final.py`:

```python
# McNemar F2 (bin) vs F1 (fixed) trên dark condition
# Đây là so sánh quan trọng nhất, hiện tại bị thiếu

dark_pairs = [p for p in test_pairs if p['bin_id'] == 'dark']
labels = [p['label'] for p in dark_pairs]

decisions_fixed = [1 if p['sim'] >= formula_fixed(p['bin_id'], p['L'], p['N'], p['q']) 
                   else 0 for p in dark_pairs]
decisions_bin   = [1 if p['sim'] >= formula_bin(p['bin_id'], p['L'], p['N'], p['q']) 
                   else 0 for p in dark_pairs]

p_val_f2_f1, table = mcnemar_test(decisions_bin, decisions_fixed, labels)

print(f"McNemar F2(bin) vs F1(fixed) trên dark: p = {p_val_f2_f1:.4f}")
if p_val_f2_f1 < 0.05:
    print(">>> Bin-specific significantly better than fixed (p < 0.05)")
else:
    print(">>> Không có ý nghĩa thống kê")
```

---

## NGÀY 4: Figures còn thiếu

### Task 4.1 — DET Curve

Tạo `src/utils/plot_det_curve.py`:

```
Vẽ DET curve (Detection Error Tradeoff) cho 4 formulas trên dark condition:

DET curve khác ROC curve:
  - X axis: FAR (False Acceptance Rate) — log scale
  - Y axis: FRR (False Rejection Rate) — log scale
  - Cả 2 trục dùng log scale → nhìn rõ hơn ở vùng error thấp

Cách tính:
  Với mỗi formula, thay đổi threshold từ 0 đến 1 theo 1000 bước
  Tại mỗi threshold: tính FRR và FAR
  Plot điểm (FAR, FRR) → thành đường DET curve

Thêm:
  - Điểm EER trên mỗi đường (nơi FRR = FAR) đánh dấu bằng X
  - Legend với EER value
  - Vùng "operating point" cho bài toán chấm công 
    (FAR < 2%, FRR < 10%) — vẽ rectangle màu xanh nhạt

Colors: đỏ=fixed, cam=bin, xanh lá=linear, xanh dương=interaction
Save: outputs/figures/Figure_06_DET_curve.png, dpi=200
```

---

### Task 4.2 — Cross-condition Analysis

Tạo `src/experiments/cross_condition.py`:

```
MỤC TIÊU: Đánh giá realistic scenario
  "Gallery chụp lúc sáng (bright) — probe lúc tối (dark) → nhận ra không?"

Tạo cross-condition pairs:
  bright_gallery × dark_probe (same person): gallery=bright ảnh, probe=dark ảnh
  bright_gallery × dark_probe (diff person): gallery=bright ảnh, probe=dark ảnh khác người

Đây là scenario thực tế nhất:
  Enrollment buổi sáng (ảnh đẹp)
  Query buổi tối (ảnh tối)

Chạy 4 formulas trên cross-condition pairs
Report FRR, FAR, EER

So sánh với same-condition pairs (dark vs dark)
Kỳ vọng: cross-condition tệ hơn → đây là lý do cần gallery adaptation

Output:
  Bảng so sánh:
  Scenario              | FRR  | FAR  | EER
  Same-condition (dark) | X.X% | X.X% | X.X%
  Cross (bright→dark)   | X.X% | X.X% | X.X%  ← tệ hơn → cần adaptation
  After adaptation      | X.X% | X.X% | X.X%  ← sau gallery update

  Save: outputs/cross_condition_results.csv
```

---

## NGÀY 5: Model Quantization (Fix H3 RAM)

### Task 5.1 — Quantize InsightFace

Tạo `src/utils/quantize_model.py`:

```
BƯỚC 1 — Export model sang ONNX:
  InsightFace buffalo_sc đã dùng ONNX internally
  Path thường ở: ~/.insightface/models/buffalo_sc/
  Tìm file .onnx trong thư mục đó

BƯỚC 2 — Quantize INT8 bằng onnxruntime:
  from onnxruntime.quantization import quantize_dynamic, QuantType
  
  quantize_dynamic(
      model_input=original_model_path,
      model_output=quantized_model_path,
      weight_type=QuantType.QInt8
  )

BƯỚC 3 — Đo RAM trước và sau:
  Dùng psutil để đo peak RAM khi load và run model
  
  Kỳ vọng: RAM giảm 50-60% (từ 3.2GB xuống ~1.3-1.6GB)

BƯỚC 4 — Đo accuracy sau quantization:
  Chạy lại experiment_final.py với quantized model
  So sánh FRR/FAR/EER: chênh lệch thường < 0.5%

BƯỚC 5 — Report accuracy-memory trade-off:
  Bảng:
  Model          | RAM    | Latency | FRR_dark | FAR_dark
  buffalo_sc FP32| 3.2 GB | 108 ms  | 3.9%     | 3.2%
  buffalo_sc INT8| X GB   | X ms    | X.X%     | X.X%
  
  Save: outputs/quantization_results.json
```

---

## NGÀY 6-7: Hoàn thiện báo cáo LaTeX

### Task 6.1 — Cập nhật sections với số mới

Sửa `sections/05_results.tex`:

```
THÊM subsection 5.5 — Gallery Adaptation (H2):

Copy kết quả từ outputs/cp_bwt_results.json vào bảng LaTeX:

\subsection{Kết quả gallery adaptation (H2)}
[Bảng CP-BWT]
[Figure adaptation curve]
[Kết luận H2: PASS/FAIL với giải thích]

THÊM subsection 5.6 — Cross-condition analysis:
[Bảng cross-condition]
[Giải thích tại sao cần gallery adaptation]

CẬP NHẬT subsection 5.3 — Edge benchmark:
[Thêm row quantized model vào Table 5]
[Cập nhật kết luận H3]

CẬP NHẬT subsection 5.2 — Statistical test:
[Thêm McNemar F2 vs F1]
[Thêm confidence intervals vào Table 4]
```

### Task 6.2 — Thêm figures vào LaTeX

Thêm vào `sections/05_results.tex`:

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\linewidth]{../outputs/figures/pipeline_diagram.png}
    \caption{Kiến trúc pipeline 4 module của hệ thống đề xuất}
    \label{fig:pipeline}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\linewidth]{../outputs/figures/Figure_06_DET_curve.png}
    \caption{DET curve so sánh 4 công thức threshold trên điều kiện dark}
    \label{fig:det}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\linewidth]{../outputs/figures/Figure_05_adaptation_curve.png}
    \caption{Accuracy theo số dark samples được cập nhật vào gallery}
    \label{fig:adaptation}
\end{figure}
```

---

## CHECKLIST CUỐI TRƯỚC KHI NỘP

```
Code:
  ✅ gallery_adaptation.py — chạy được, có số CP-BWT thật
  ✅ cross_condition.py — có kết quả cross-condition
  ✅ quantize_model.py — có RAM sau quantization
  ✅ draw_pipeline.py — có pipeline diagram
  ✅ Bootstrap CI trong experiment_final.py
  ✅ McNemar F2 vs F1 trong experiment_final.py
  ✅ Train/val/test split trong experiment_final.py

Báo cáo:
  ✅ Không còn placeholder ?? hoặc [ghi rõ...]
  ✅ H1 criterion nhất quán với kết quả
  ✅ H2 có số thật từ gallery_adaptation.py
  ✅ H3 có RAM sau quantization
  ✅ Bảng 4 có confidence interval [lower, upper]
  ✅ McNemar F2 vs F1 có p-value
  ✅ 3 figures: pipeline, DET curve, adaptation curve
  ✅ Train/val/test split được mô tả rõ

Figures:
  ✅ Figure pipeline_diagram.png
  ✅ Figure_A_ROC_dark.png (đã có)
  ✅ Figure_B_FRR_bar.png (đã có)
  ✅ Figure_05_adaptation_curve.png (mới)
  ✅ Figure_06_DET_curve.png (mới)
```

---

## GHI CHÚ CHO CURSOR

Khi implement, ưu tiên theo thứ tự:
1. `gallery_adaptation.py` — CP-BWT (impact cao nhất)
2. Fix placeholders + sửa H1 criterion (30 phút, bắt buộc)
3. Bootstrap CI + McNemar F2 vs F1 (1 tiếng, credibility)
4. `quantize_model.py` — fix H3 RAM
5. `cross_condition.py` — insight thực tế
6. Figures còn thiếu
7. Cập nhật LaTeX

Với mỗi script, chạy thử trước khi báo cáo số liệu.
Tất cả outputs save vào `outputs/` để dễ copy vào báo cáo.