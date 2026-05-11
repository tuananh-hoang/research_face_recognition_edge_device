# Face Recognition Threshold Optimization on Edge Devices

Đây là mã nguồn phục vụ Nghiên cứu Khoa học (NCKH) về việc tối ưu hóa ngưỡng (Adaptive Threshold) cho hệ thống nhận diện khuôn mặt (Face Recognition) khi triển khai thực tế trên các thiết bị Edge (VD: Raspberry Pi).

## 🎯 Mục tiêu Nghiên cứu
Hệ thống sử dụng các công thức tính toán ngưỡng động (Adaptive Threshold) phụ thuộc vào 2 yếu tố môi trường:
- **Độ sáng (Luminance - L)**
- **Độ nhiễu (Noise - N)**

Mục tiêu là tìm ra công thức tốt nhất giúp cân bằng giữa tỉ lệ nhận diện sai (FAR) và từ chối sai (FRR) trong điều kiện ánh sáng yếu, mà không làm tiêu tốn quá nhiều tài nguyên của thiết bị Edge.

---

## 📂 Cấu trúc Thư mục (Modular Design)

Project được tổ chức theo kiến trúc phân tầng để dễ dàng thay đổi và thử nghiệm:

```text
face_recognition/
├── src/
│   ├── core/                   ← Chứa logic chính (Backbone)
│   │   ├── embedder.py         # Trích xuất đặc trưng khuôn mặt dùng InsightFace (ArcFace)
│   │   ├── iqa.py              # Đánh giá chất lượng ảnh (Tính L và N)
│   │   └── gallery_manager.py  # Quản lý Gallery lưu trữ Embeddings
│   │
│   ├── threshold/              ← Chứa các công thức toán học Threshold
│   │   ├── fixed.py            # Baseline: Ngưỡng cố định
│   │   ├── bin_specific.py     # Ngưỡng phân rã theo cụm sáng/tối
│   │   ├── linear.py           # Tuyến tính theo L và N
│   │   └── interaction.py      # [Ours] Thuật toán đề xuất: Interaction term (1-L)*N
│   │
│   ├── experiments/            ← Chạy đánh giá và thực nghiệm NCKH
│   │   ├── experiment_formulas.py  # Đánh giá 4 công thức, chạy McNemar Test, ghi log CSV
│   │   ├── benchmark_edge.py       # Giả lập môi trường 512MB RAM, benchmark Latency/Memory
│   │   └── run_all.py              # Chạy tổng thể và xuất Figures
│   │
│   └── utils/                  ← Tiện ích hỗ trợ
│       └── augment.py          # Sinh dữ liệu Synthetic mô phỏng độ sáng/nhiễu
│
├── data/                       ← Nơi chứa custom data phân theo [bright/medium/dark]
├── outputs/                    ← Chứa kết quả CSV, Logs và Figures (Heatmap, ROC, v.v.)
├── docs/                       ← Tài liệu nghiên cứu (plan.md)
└── requirements.txt            ← Danh sách thư viện Python
```

---

## ⚙️ Cài đặt Môi trường (Installation)

Sử dụng `conda` để tạo môi trường ảo độc lập:

```bash
# 1. Tạo môi trường ảo tên face_env dùng Python 3.10
conda create -n face_env python=3.10 -y

# 2. Kích hoạt môi trường
conda activate face_env

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

*(Lưu ý: Thư viện `insightface` cần bộ biên dịch C++. Nếu bạn dùng Windows, hãy chắc chắn đã cài đặt "Desktop development with C++" qua Microsoft C++ Build Tools).*

---

## 🚀 Cách chạy Thực nghiệm

Để chạy kiểm thử 4 công thức và xuất kết quả báo cáo, sử dụng script `experiment_formulas.py` nằm trong mục experiments. Chạy lệnh tại thư mục gốc của project:

```bash
# Chạy với dữ liệu giả lập (Synthetic Data)
python src/experiments/experiment_formulas.py --dataset synthetic

# Chạy với tập dữ liệu thực LFW (Labeled Faces in the Wild)
python src/experiments/experiment_formulas.py --dataset lfw
```

**Kết quả nhận được (trong folder `outputs/`):**
1. File `formula_comparison.csv` (Chi tiết FRR, FAR, EER, AUC).
2. File `experiment_log.csv` (Lưu vết các lần chạy thực nghiệm, bao gồm siêu tham số tối ưu).
3. Thư mục `figures/` chứa 4 biểu đồ báo cáo:
   - Heatmap FRR
   - FAR-FRR Trade-off Scatter Plot
   - Delta FRR Bar Chart so với Baseline
   - ROC Curve

---
*Developed for Academic Research & Edge Deployment Analysis.*
