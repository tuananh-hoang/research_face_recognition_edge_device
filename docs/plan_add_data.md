# PLAN MULTI-DATASET TESTING
## Mục tiêu: Test trên nhiều dataset để chứng minh tính tổng quát

---

## TẠI SAO CẦN MULTI-DATASET

Reviewer và hội đồng sẽ hỏi:
> "Kết quả này chỉ đúng trên 14 người của bạn hay tổng quát hơn?"

Câu trả lời cần là:
> "Chúng tôi test trên 3 dataset với đặc điểm khác nhau và 
> thấy bin-specific threshold nhất quán tốt hơn fixed threshold 
> trong điều kiện quality thấp."

---

## 3 DATASET PHÙ HỢP VỚI DOMAIN — Lý do chọn

```
Dataset         | Lý do phù hợp                    | Availability
────────────────|───────────────────────────────────|─────────────
Custom (của ta) | Vietnamese context, indoor        | Đã có
XQLFW           | Cross-quality pairs, có sẵn pairs | Free, GitHub
DARK FACE       | Real nighttime, outdoor           | Free, Drive
```

**KHÔNG dùng LFW làm main benchmark** vì LFW là ảnh sáng, controlled —
bin 'dark' gần như trống, không test được gì có ý nghĩa.

---

## DATASET 1: Custom (Đã có — Primary)

```
Đường dẫn: data/bright/, data/medium/, data/dark/
Số lượng: 14 người, 1144 ảnh, 2954 pairs
Đặc điểm: Indoor hallway, Vietnamese faces, augmented dark
Vai trò: PRIMARY dataset — kết quả chính trong paper
```

Không cần làm gì thêm — đã chạy xong.

---

## DATASET 2: XQLFW — Cross-Quality Pairs

### Tại sao XQLFW phù hợp nhất:
XQLFW tập trung vào sự khác biệt chất lượng ảnh lớn giữa 
các cặp — chứa ảnh bị degraded thực tế hơn so với LFW thông thường.

Đây là bộ dataset có same-person pairs với **chất lượng khác nhau** 
(1 ảnh sáng/rõ + 1 ảnh blur/low-quality) — gần nhất với scenario 
gallery sáng / probe tối của bài toán chấm công.

### Download:
```bash
# Clone repo
git clone https://github.com/Martlgap/xqlfw.git

# Download dataset từ link trong README của repo:
# https://martlgap.github.io/xqlfw/
# File: xqlfw_aligned_112.tar.gz (~1.5GB)
# Giải nén vào: data/xqlfw/
```

### Tạo file `src/experiments/test_xqlfw.py`:

```
MỤC TIÊU: Test 4 công thức threshold trên XQLFW

BƯỚC 1 — Load XQLFW pairs:
  XQLFW có file pairs.csv với columns: img1_path, img2_path, label
  Với mỗi pair:
    img1 = ảnh chất lượng cao (gallery)
    img2 = ảnh chất lượng thấp (probe — bị blur/noise)
    label = 1 (same person) hoặc 0 (different)

BƯỚC 2 — Tính embedding bằng InsightFace:
  emb1, det1 = embedder.get_embedding(img1)
  emb2, det2 = embedder.get_embedding(img2)
  sim = cosine_similarity(emb1, emb2)

BƯỚC 3 — Tính IQA cho img2 (probe image):
  L, N, bin_id, q = iqa.compute_context(img2)
  # img2 là ảnh bị degraded → sẽ rơi vào medium hoặc dark bin
  # img1 là ảnh tốt → bright bin

BƯỚC 4 — Chạy 4 công thức threshold:
  tau_fixed       = 0.44
  tau_bin         = bin_tau[bin_id]
  tau_linear      = 0.48 - 0.10*(1-L) - 0.05*N
  tau_interaction = formula_interaction(L, N, q)
  
  Record: decision và ground truth cho mỗi formula

BƯỚC 5 — Tính metrics per bin:
  Phân loại pairs theo IQA của img2:
    high_quality_pairs  = pairs có L > 0.6 và N < 0.1
    medium_quality_pairs = pairs có 0.3 <= L <= 0.6
    low_quality_pairs   = pairs có L < 0.3 hoặc N > 0.3
  
  Tính FRR, FAR, EER cho mỗi (formula × quality_bin)

BƯỚC 6 — Output:
  Bảng kết quả XQLFW:
  Quality Bin    | Formula     | FRR   | FAR   | EER
  High quality   | Fixed       | X.X%  | X.X%  | X.X%
  High quality   | Bin         | X.X%  | X.X%  | X.X%
  Low quality    | Fixed       | X.X%  | X.X%  | X.X%  ← FRR cao → gap lớn
  Low quality    | Bin         | X.X%  | X.X%  | X.X%  ← FRR thấp hơn
  
  Save: outputs/xqlfw_results.csv
  
  In nhận xét:
  "Trên XQLFW low-quality pairs: bin-specific giảm FRR X% so với fixed"
  "Kết quả nhất quán với Custom dataset"  ← đây là câu quan trọng nhất
```

---

## DATASET 3: DARK FACE — Real Nighttime Images

### Tại sao DARK FACE phù hợp:
DARK FACE cung cấp 6000 ảnh thực tế chụp ban đêm tại 
các tòa nhà, đường phố, công viên — có bounding box annotations 
cho khuôn mặt.

**Lưu ý:** DARK FACE là detection dataset (không có recognition pairs) 
→ KHÔNG dùng để test FRR/FAR. Dùng để:
1. Validate IQA pipeline: ảnh dark thật có L thấp đúng không?
2. Test embedding quality: buffalo_sc có detect được face ban đêm thật không?
3. Tạo pairs giả lập: match face trong dark với ảnh identity từ dataset khác

### Download:
```
Google Drive link từ: https://flyywh.github.io/CVPRW2019LowLight/
File: DARK_FACE_train.zip
Giải nén vào: data/dark_face/
```

### Tạo file `src/experiments/test_dark_face_iqa.py`:

```
MỤC TIÊU: Validate IQA pipeline trên ảnh dark thật

BƯỚC 1 — Load DARK FACE images:
  Đọc tất cả ảnh trong data/dark_face/
  Crop face region từ bounding box annotations (file .txt đi kèm)

BƯỚC 2 — Chạy IQA trên từng face crop:
  L, N, bin_id, q = iqa.compute_context(face_crop)
  
  Record: L, N, bin_id cho mỗi face

BƯỚC 3 — Kiểm tra phân phối:
  Plot histogram của L và N trên DARK FACE images
  
  Kỳ vọng: 
    L phân phối thấp (< 0.3) → hầu hết rơi vào dark bin
    N phân phối cao (> 0.2) → nhiều nhiễu
  
  So sánh với Custom dataset:
    Custom dark (augmented): L ~ [0.05, 0.15]
    DARK FACE (real night):  L ~ ???  ← điền sau khi chạy

BƯỚC 4 — Test embedding detection rate:
  Đếm: bao nhiêu % face crops được InsightFace detect thành công?
  
  Nếu detection rate thấp (< 50%) → note trong báo cáo:
    "buffalo_sc gặp khó khăn với dark images thật"
    "Đây là hướng cải thiện: cần backbone tốt hơn cho dark condition"

BƯỚC 5 — Output:
  Bảng IQA validation:
  Dataset         | Mean L | Mean N | % in dark bin | Detection rate
  Custom bright   | 0.72   | 0.05   | 0%            | ~95%
  Custom dark (aug)| 0.15  | 0.35   | 90%           | ~80%
  DARK FACE (real)| ???    | ???    | ???%          | ???%
  
  Figure: Histogram L và N so sánh 3 datasets
  Save: outputs/iqa_validation_darkface.csv
  Save: outputs/figures/Figure_07_IQA_validation.png
  
  Ý nghĩa: Chứng minh IQA pipeline hoạt động đúng trên dark thật
```

---

## TỔNG HỢP: Multi-Dataset Comparison Table

Sau khi chạy cả 3 dataset, tạo file `src/experiments/summarize_all_datasets.py`:

```
Tạo master comparison table:

Dataset         | N_persons | N_pairs | FRR_fixed_dark | FRR_bin_dark | Delta  | Consistent?
Custom (ours)   | 14        | 2,954   | 9.8%           | 3.9%         | -5.9pp | ✅
XQLFW (low-q)   | ~1000+    | ~6,000  | ???%           | ???%         | ???pp  | ✅/❌
DARK FACE (IQA) | N/A       | N/A     | IQA validation only                        | ✅/❌

Kết luận:
"Trên cả Custom dataset (indoor, Vietnamese context) và XQLFW 
(international, cross-quality benchmark), bin-specific threshold 
nhất quán cải thiện FRR trong điều kiện chất lượng thấp so với 
fixed threshold. IQA pipeline được validate trên DARK FACE real 
nighttime images."

Save: outputs/multi_dataset_summary.csv
```

---

## VIẾT VÀO BÁO CÁO

Thêm vào `sections/04_experiments.tex`:

```latex
\subsection{Tập dữ liệu bổ sung}

Để đánh giá tính tổng quát của phương pháp, chúng tôi 
bổ sung hai tập dữ liệu công khai:

\textbf{XQLFW}~\cite{knoche2021xqlfw}: Cross-Quality LFW 
chứa các cặp same/different với chênh lệch chất lượng lớn — 
phù hợp để test adaptive threshold trên cross-quality scenario.

\textbf{DARK FACE}~\cite{yang2020darkface}: 6,000 ảnh 
chụp ban đêm thực tế — dùng để validate IQA pipeline 
trên dark images thật, không phải augmentation.
```

Thêm vào `references.bib`:

```bibtex
@inproceedings{knoche2021xqlfw,
  title={Cross-Quality LFW: A Database for Analyzing 
         Cross-Resolution Image Face Recognition in 
         Unconstrained Environments},
  author={Knoche, Martin and H{\"o}rmann, Stefan and Rigoll, Gerhard},
  booktitle={2021 16th IEEE International Conference on 
             Automatic Face and Gesture Recognition (FG 2021)},
  year={2021},
  organization={IEEE}
}

@article{yang2020darkface,
  title={Advancing Image Understanding in Poor Visibility 
         Environments: A Collective Benchmark Study},
  author={Yang, Wenhan and Yuan, Ye and Ren, Wenqi and Liu, 
          Jiaying and Scheirer, Walter J and Wang, Zhangyang 
          and Zhang, Taiheng and Zhong, Qiaosong and Xie, Di 
          and Pu, Shiliang and others},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={5168--5182},
  year={2021}
}
```

---

## THỨ TỰ THỰC HIỆN

```
Ngày 1 (song song với CP-BWT):
  □ Download XQLFW (~1.5GB) — để chạy background
  □ Download DARK FACE sample (100 test images trước)

Ngày 2:
  □ Chạy test_xqlfw.py → có kết quả XQLFW
  □ Chạy test_dark_face_iqa.py → có IQA validation

Ngày 3:
  □ Chạy summarize_all_datasets.py → có master table
  □ Cập nhật LaTeX với kết quả mới

Điều kiện dừng:
  Nếu XQLFW cho kết quả nhất quán với Custom (bin > fixed) → claim generalizability
  Nếu không nhất quán → note là limitation, cần investigate thêm
```

---

## LƯU Ý QUAN TRỌNG 

```
1. XQLFW pairs không có ánh sáng label → dùng IQA để tự phân bin
   Đây là đúng methodology — IQA tự động detect quality

2. DARK FACE không có identity pairs → CHỈ dùng cho IQA validation
   Không claim FRR/FAR trên DARK FACE

3. Nếu download chậm → test trước với 200 pairs của XQLFW
   Đủ để có kết quả sơ bộ trong 30 phút

4. Nếu InsightFace không detect được nhiều face trong DARK FACE
   → Đó là kết quả có giá trị: ghi vào limitation
   → "buffalo_sc cần enhance trước khi nhận diện trên dark thật"
```