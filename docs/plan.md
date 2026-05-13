# BLUEPRINT: Từ Nghiên Cứu Đến Sản Phẩm
## Context-Aware Adaptive Face Recognition cho Chấm Công Thực Tế

---

# MỤC LỤC

1. [Tổng quan kiến trúc hệ thống](#1-tổng-quan-kiến-trúc-hệ-thống)
2. [Phase 1 — Research Prototype](#2-phase-1--research-prototype-tuần-1-8)
3. [Phase 2 — Engineering MVP](#3-phase-2--engineering-mvp-tuần-9-16)
4. [Phase 3 — Production System](#4-phase-3--production-system-tuần-17-24)
5. [Lựa chọn Model chi tiết](#5-lựa-chọn-model-chi-tiết)
6. [Phần cứng và Edge Deployment](#6-phần-cứng-và-edge-deployment)
7. [Data Pipeline và Dataset](#7-data-pipeline-và-dataset)
8. [Testing, Monitoring và Vận hành](#8-testing-monitoring-và-vận-hành)
9. [Ước tính chi phí và Timeline](#9-ước-tính-chi-phí-và-timeline)
10. [Rủi ro và Mitigation](#10-rủi-ro-và-mitigation)

---

# 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG

## 1.1 Kiến trúc 4 tầng

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TẦNG 4: APPLICATION                              │
│  Dashboard quản lý · API chấm công · Báo cáo · Alerts                  │
├─────────────────────────────────────────────────────────────────────────┤
│                        TẦNG 3: DECISION ENGINE                          │
│  Adaptive Threshold τ(C) · Accept/Reject Logic · Confidence Scoring    │
├─────────────────────────────────────────────────────────────────────────┤
│                        TẦNG 2: INTELLIGENCE                             │
│  Face Embedding · IQA (L, N estimation) · Gallery Management           │
│  Quality Gate · Online Prototype Update                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                        TẦNG 1: PERCEPTION                               │
│  Camera Input · Face Detection · Alignment · Preprocessing             │
│  Anti-spoofing · Liveness Detection                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1.2 Luồng xử lý chính (Inference Pipeline)

```
Camera Frame
    │
    ▼
[1] Face Detection (SCRFD/YOLOv8-face)
    │── không có mặt → skip
    │── có mặt → crop + align (5-point landmark)
    ▼
[2] Anti-spoofing Check (MiniFASNet/Silent-FAS)
    │── spoofing detected → REJECT + log
    │── live face → continue
    ▼
[3] Image Quality Assessment
    │── estimate L (luminance), N (noise), blur score
    │── compute context vector C = [L, N, sin(2πh/24), cos(2πh/24)]
    ▼
[4] Face Embedding (ArcFace/MobileFaceNet)
    │── extract 512-d embedding
    │── compute quality proxy q = σ(α·‖emb‖₂ - β)
    ▼
[5] Gallery Matching
    │── cosine similarity với tất cả prototypes
    │── top-1 match: (identity_id, score)
    ▼
[6] Adaptive Threshold Decision
    │── compute τ(C) = τ_base·(1 - γ·(1-L)·N)·q + τ_floor·(1-q)
    │── score ≥ τ(C) → ACCEPT → [7]
    │── score < τ(C) → REJECT → log + maybe prompt retry
    ▼
[7] Gallery Update (conditional)
    │── if score > τ_high AND q > q_min → trigger update
    │── weighted moving average prototype update
    │── LRU eviction nếu vượt memory budget
    ▼
[8] Output
    └── attendance record + confidence + condition metadata
```

## 1.3 Deployment Topology

```
                    ┌──────────────────┐
                    │   CLOUD SERVER    │
                    │  ─────────────── │
                    │  Admin Dashboard  │
                    │  Enrollment API   │
                    │  Analytics DB     │
                    │  Model Registry   │
                    │  Gallery Sync     │
                    └────────┬─────────┘
                             │ HTTPS/gRPC
                ┌────────────┼────────────┐
                ▼            ▼            ▼
         ┌────────────┐ ┌────────────┐ ┌────────────┐
         │  EDGE BOX   │ │  EDGE BOX   │ │  EDGE BOX   │
         │  Site A     │ │  Site B     │ │  Site C     │
         │ ──────────  │ │ ──────────  │ │ ──────────  │
         │ Inference   │ │ Inference   │ │ Inference   │
         │ Local Gallery│ │ Local Gallery│ │ Local Gallery│
         │ Offline mode│ │ Offline mode│ │ Offline mode│
         └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                │               │               │
           ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
           │ Camera 1│    │ Camera 1│    │ Camera 1│
           │ Camera 2│    │ Camera 2│    │ Camera 2│
           └─────────┘    └─────────┘    └─────────┘
```

---

# 2. PHASE 1 — RESEARCH PROTOTYPE (Tuần 1-8)

Mục tiêu: validate 3 hypotheses trên controlled dataset.

## 2.1 Tuần 1-2: Data Collection & Baseline Setup

### Thiết lập thu thập dữ liệu

Camera setup cho 3 điều kiện:

| Điều kiện | Luminance (lux) | Setup | Ghi chú |
|-----------|-----------------|-------|---------|
| Bright | > 300 lux | Văn phòng, đèn đủ sáng | Control condition |
| Medium | 50-300 lux | Hành lang, đèn mờ | Transition condition |
| Dark | < 50 lux | Mô phỏng ca đêm, tắt bớt đèn | Target condition |

Yêu cầu tối thiểu:
- N_person ≥ 30 (tối thiểu cho statistical significance, lý tưởng ≥ 50)
- 10 ảnh/người/điều kiện = 900 ảnh minimum
- Log metadata: timestamp, camera exposure, ISO, white balance

Camera khuyên dùng cho thu thập:
- Logitech C920/C922 (USB, 1080p, autofocus) — phổ biến, giá rẻ, ~800K VND
- Hoặc Hikvision DS-2CD2T47G2 (IP camera, có IR) — gần với deployment thực

Quy trình thu thập:
```
Mỗi người thực hiện:
  1. Enrollment: 5 ảnh frontal, ánh sáng tốt, nền sạch
  2. Bright set: 10 ảnh, thay đổi nhẹ góc/biểu cảm
  3. Medium set: 10 ảnh, cùng protocol
  4. Dark set: 10 ảnh, cùng protocol
  5. Impostor set: 10 ảnh random pairs (cho FAR calculation)

Metadata log (CSV):
  person_id, condition, timestamp, camera_exposure, iso, lux_meter_reading, filename
```

### Baseline Model Setup

```bash
# Cài đặt InsightFace (ArcFace pretrained)
pip install insightface onnxruntime-gpu  # hoặc onnxruntime cho CPU

# Download pretrained models
# buffalo_l: ResNet100 + ArcFace, 512-d embedding
# buffalo_sc: MobileFaceNet, nhẹ hơn cho edge
```

Code baseline evaluation:
```python
import insightface
from insightface.app import FaceAnalysis

# Load model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Extract embedding
faces = app.get(img)
if faces:
    embedding = faces[0].normed_embedding  # 512-d, L2-normalized
    det_score = faces[0].det_score
    bbox = faces[0].bbox
    landmark = faces[0].kps  # 5 keypoints
```

## 2.2 Tuần 3-4: Module A+B — IQA & Baseline Metrics

### Image Quality Assessment Module

Không cần model phức tạp. Dùng signal processing:

```python
import cv2
import numpy as np

def estimate_luminance(face_crop: np.ndarray) -> float:
    """Estimate luminance L ∈ [0, 1] từ face crop."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    L = gray.mean() / 255.0
    return L

def estimate_noise(face_crop: np.ndarray) -> float:
    """Estimate noise level N ∈ [0, 1] bằng Laplacian variance."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    # Noise estimation via high-frequency energy
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_raw = laplacian.var()
    # Normalize: empirically, laplacian var ∈ [0, 5000]
    # Cần calibrate trên data thực
    N = min(noise_raw / 5000.0, 1.0)
    return N

def estimate_blur(face_crop: np.ndarray) -> float:
    """Estimate blur score. Thấp = mờ."""
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_context_vector(face_crop: np.ndarray, hour: float) -> np.ndarray:
    """Context vector C = [L, N, sin(2πh/24), cos(2πh/24)]."""
    L = estimate_luminance(face_crop)
    N = estimate_noise(face_crop)
    time_sin = np.sin(2 * np.pi * hour / 24.0)
    time_cos = np.cos(2 * np.pi * hour / 24.0)
    return np.array([L, N, time_sin, time_cos])
```

LƯU Ý QUAN TRỌNG:
- Laplacian variance đo cả texture lẫn noise — face crop có texture tự nhiên cao.
  Cần calibrate ngưỡng trên data thực, KHÔNG dùng giá trị mặc định.
- Luminance mean có thể bị skew bởi backlight. Cân nhắc dùng percentile (p25)
  thay vì mean để robust hơn.

### Baseline Evaluation Script

```python
def evaluate_fixed_threshold(embeddings, labels, conditions, tau_fixed=0.44):
    """Evaluate FRR, FAR với fixed threshold, chia theo condition bin."""
    results = {}
    for cond in ['bright', 'medium', 'dark']:
        mask = conditions == cond
        genuine_scores = [...]   # cosine sim giữa genuine pairs
        impostor_scores = [...]  # cosine sim giữa impostor pairs
        
        FRR = (genuine_scores < tau_fixed).sum() / len(genuine_scores)
        FAR = (impostor_scores >= tau_fixed).sum() / len(impostor_scores)
        results[cond] = {'FRR': FRR, 'FAR': FAR}
    return results
```

Deliverable tuần 3-4:
- iqa_stats.csv: L, N, blur score cho mọi ảnh
- baseline_results.csv: FRR, FAR per condition bin với τ = 0.40, 0.42, 0.44, 0.46, 0.48
- Histogram plots: genuine vs impostor score distribution per bin

## 2.3 Tuần 5-6: Module C — Adaptive Threshold

### Calibration Protocol

```python
from scipy.optimize import minimize_scalar

def calibrate_threshold_per_bin(genuine_scores, impostor_scores, target_far=0.01):
    """Tìm τ sao cho FAR ≤ target_far, minimize FRR."""
    # Sort impostor scores descending
    sorted_imp = np.sort(impostor_scores)[::-1]
    # Tìm τ tại FAR = target_far
    idx = int(len(sorted_imp) * target_far)
    tau = sorted_imp[idx] if idx < len(sorted_imp) else sorted_imp[-1]
    
    FRR = (genuine_scores < tau).sum() / len(genuine_scores)
    return tau, FRR

# Calibrate per bin
for cond in ['bright', 'medium', 'dark']:
    tau_opt, frr = calibrate_threshold_per_bin(
        genuine_scores[cond], impostor_scores[cond], target_far=0.01
    )
    print(f"{cond}: τ_opt={tau_opt:.4f}, FRR={frr:.4f}")
```

### Fit Adaptive Threshold Function

Chiến lược từ đơn giản đến phức tạp:

```
Step 1: Linear model (baseline)
  τ(C) = a + b·(1-L) + c·N
  Fit a, b, c từ τ_opt per bin

Step 2: + Interaction term
  τ(C) = a + b·(1-L) + c·N + d·(1-L)·N
  
Step 3: + Quality gate (nếu Step 2 tốt hơn Step 1 significantly)
  τ(C) = [a + b·(1-L) + c·N + d·(1-L)·N] · q + τ_floor · (1-q)

Step 4: + Time feature (nếu collect data đủ nhiều giờ)
  Thêm sin/cos vào linear model

So sánh bằng FRR@FAR=1% trên held-out set.
Dùng cross-validation (leave-3-persons-out) để tránh overfit.
```

NGUYÊN TẮC: Không nhảy thẳng vào công thức phức tạp. Validate từng bước.
Nếu linear model đã đạt H1 (giảm FRR ≥ 10%), không cần phức tạp hóa.

## 2.4 Tuần 7-8: Module D — Gallery Update + Ablation

### Weighted Prototype Update

```python
class AdaptiveGallery:
    def __init__(self, max_per_person=50, lambda_lr=0.15):
        self.prototypes = {}      # person_id → np.array (512-d)
        self.anchors = {}         # person_id → np.array (enrollment prototype)
        self.update_counts = {}   # person_id → int
        self.lambda_lr = lambda_lr
        self.max_per_person = max_per_person
        
    def enroll(self, person_id: str, embeddings: list[np.ndarray]):
        """Enrollment từ ảnh sáng."""
        proto = np.mean(embeddings, axis=0)
        proto = proto / np.linalg.norm(proto)
        self.prototypes[person_id] = proto.copy()
        self.anchors[person_id] = proto.copy()  # anchor KHÔNG BAO GIỜ thay đổi
        self.update_counts[person_id] = 0
        
    def try_update(self, person_id: str, new_emb: np.ndarray, 
                   quality: float, det_score: float,
                   match_score: float) -> bool:
        """Update prototype nếu đủ điều kiện an toàn."""
        
        # Safety gate 1: confidence phải cao
        if match_score < 0.50 or det_score < 0.7:
            return False
            
        # Safety gate 2: quality tối thiểu
        if quality < 0.2:
            return False
            
        # Safety gate 3: không drift quá xa từ anchor
        anchor_sim = np.dot(new_emb, self.anchors[person_id])
        if anchor_sim < 0.65:  # quá khác với enrollment → nghi ngờ
            return False
        
        # Compute adaptive learning rate
        existing_sim = np.dot(new_emb, self.prototypes[person_id])
        diversity_bonus = max(0, 1 - existing_sim)  # xa prototype hiện tại → informative hơn
        
        w = self.lambda_lr * quality * det_score * diversity_bonus
        w = min(w, 0.3)  # cap learning rate
        
        # Update
        new_proto = (1 - w) * self.prototypes[person_id] + w * new_emb
        new_proto = new_proto / np.linalg.norm(new_proto)
        
        # Safety gate 4: verify updated proto vẫn gần anchor
        if np.dot(new_proto, self.anchors[person_id]) < 0.60:
            return False  # update sẽ gây drift → reject
            
        self.prototypes[person_id] = new_proto
        self.update_counts[person_id] += 1
        return True
```

### Ablation Study Design

```
Experiment matrix (6 configurations):

Config 1: ArcFace + τ_fixed                    (baseline)
Config 2: ArcFace + τ_linear(L, N)             (+ adaptive threshold basic)
Config 3: ArcFace + τ_interaction(L, N)         (+ interaction term)
Config 4: ArcFace + τ_quality_gated(L, N, q)   (+ quality gate)
Config 5: ArcFace + τ_full(L, N, q, T)         (+ time feature)
Config 6: Config 4 + gallery update             (full system)

Report for each:
  - FRR@FAR=1% per condition bin
  - EER overall và per bin
  - BWT (cho Config 6)
  - Inference latency (ms)
  - Memory footprint (KB)
```

---

# 3. PHASE 2 — ENGINEERING MVP (Tuần 9-16)

Mục tiêu: chuyển prototype thành hệ thống chạy được trên edge device.

## 3.1 Kiến trúc phần mềm

### Tech Stack

```
Edge Device:
  OS:        Ubuntu 22.04 LTS (hoặc Debian 12)
  Runtime:   Python 3.10+ với ONNX Runtime
  API:       FastAPI (lightweight, async)
  DB local:  SQLite (attendance log) + numpy .npz (gallery)
  Queue:     Redis (nếu multi-camera) hoặc in-process queue

Cloud Server:
  Backend:   FastAPI hoặc Django REST
  DB:        PostgreSQL
  Storage:   MinIO / S3 (ảnh enrollment)
  Queue:     Redis / RabbitMQ (sync tasks)
  Dashboard: React + Tailwind (admin panel)
```

### Project Structure

```
face-attendance/
├── edge/                          # Chạy trên edge device
│   ├── config/
│   │   ├── config.yaml            # Thresholds, camera config, model paths
│   │   └── calibration.json       # τ_base, τ_floor, γ, α, β (từ Phase 1)
│   ├── models/                    # ONNX models
│   │   ├── det_scrfd_2.5g.onnx   # Face detection (~3MB)
│   │   ├── rec_mobilefacenet.onnx # Face recognition (~4.5MB)  
│   │   └── fas_minifasnet.onnx   # Anti-spoofing (~1MB)
│   ├── core/
│   │   ├── detector.py            # Face detection + alignment
│   │   ├── recognizer.py          # Embedding extraction
│   │   ├── iqa.py                 # Image quality assessment
│   │   ├── anti_spoof.py          # Liveness detection
│   │   ├── threshold.py           # Adaptive threshold τ(C)
│   │   ├── gallery.py             # Gallery management + update
│   │   └── pipeline.py            # Orchestrate full pipeline
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   ├── routes.py              # Endpoints: /recognize, /enroll, /health
│   │   └── schemas.py             # Pydantic models
│   ├── storage/
│   │   ├── gallery.npz            # Prototype vectors
│   │   ├── attendance.db          # SQLite log
│   │   └── sync_queue.json        # Pending cloud sync
│   └── scripts/
│       ├── enroll.py              # Batch enrollment script
│       ├── calibrate.py           # Threshold calibration từ data
│       └── benchmark.py           # Latency/memory profiling
│
├── cloud/                         # Cloud server
│   ├── api/                       # REST API
│   ├── dashboard/                 # React admin panel
│   ├── services/
│   │   ├── enrollment.py          # Enrollment management
│   │   ├── sync.py                # Gallery sync với edge devices
│   │   ├── analytics.py           # Attendance analytics
│   │   └── alerts.py              # Anomaly alerts
│   └── models/                    # DB models
│
├── shared/                        # Shared utilities
│   ├── proto/                     # Protobuf definitions (nếu dùng gRPC)
│   └── constants.py
│
├── research/                      # Notebooks & experiments từ Phase 1
│   ├── notebooks/
│   ├── data/
│   └── results/
│
├── tests/
├── docker/
│   ├── Dockerfile.edge
│   ├── Dockerfile.cloud
│   └── docker-compose.yml
│
└── docs/
    ├── api.md
    ├── deployment.md
    └── calibration_guide.md
```

## 3.2 Model Optimization cho Edge

### ONNX Export & Quantization

```python
# Export InsightFace model sang ONNX (đã có sẵn)
# Focus: quantize để giảm size và tăng speed

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization: INT8 weights, giảm ~60% size, tăng ~30% speed trên CPU
quantize_dynamic(
    model_input="rec_r100_arcface.onnx",
    model_output="rec_r100_arcface_int8.onnx",
    weight_type=QuantType.QUInt8
)

# Verify accuracy không drop đáng kể
# Chạy eval trên test set, so sánh FP32 vs INT8
```

Model size sau quantization (ước tính):

| Model | FP32 | INT8 | Accuracy drop |
|-------|------|------|---------------|
| SCRFD-2.5G (detection) | 3.2 MB | 1.1 MB | < 0.5% mAP |
| MobileFaceNet (recognition) | 4.5 MB | 1.8 MB | < 0.3% accuracy |
| ResNet50-ArcFace (recognition) | 166 MB | 44 MB | < 0.5% accuracy |
| MiniFASNet (anti-spoof) | 1.2 MB | 0.5 MB | < 1% accuracy |

### Inference Optimization

```python
# ONNX Runtime session với optimization
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4     # tune theo CPU cores
sess_options.inter_op_num_threads = 1
sess_options.enable_mem_pattern = True
sess_options.enable_cpu_mem_arena = True

# Nếu edge device có GPU (Jetson)
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'gpu_mem_limit': 512 * 1024 * 1024,  # 512MB limit
    }),
    'CPUExecutionProvider'  # fallback
]

session = ort.InferenceSession("model.onnx", sess_options, providers=providers)
```

## 3.3 API Design

### Edge API Endpoints

```yaml
POST /api/v1/recognize
  Input:  image (base64 hoặc multipart)
  Output: 
    person_id: str | null
    confidence: float
    condition: {luminance, noise, quality_score}
    decision: "ACCEPT" | "REJECT" | "UNKNOWN"
    latency_ms: float

POST /api/v1/enroll
  Input:  person_id, images[] (3-5 ảnh)
  Output: 
    success: bool
    gallery_size: int
    quality_scores: float[]

GET /api/v1/health
  Output:
    status: "ok"
    model_loaded: bool
    gallery_size: int
    memory_usage_mb: float
    uptime_seconds: int

GET /api/v1/gallery/stats
  Output:
    total_persons: int
    avg_prototype_age: float
    memory_kb: float
    partitions: {bright: int, dark: int, ...}

POST /api/v1/calibrate
  Input:  calibration_data (genuine/impostor scores per condition)
  Output: updated thresholds
  Note:   Admin-only endpoint, trigger recalibration
```

## 3.4 Anti-Spoofing Module

Bài toán chấm công BẮT BUỘC có anti-spoofing. Không có nó, hệ thống vô nghĩa.

Các phương pháp từ nhẹ đến nặng:

| Phương pháp | Model | Size | Latency | Độ tin cậy |
|-------------|-------|------|---------|------------|
| Texture-based (LBP) | Không cần DL | 0 MB | < 5ms | Thấp, dễ bypass |
| MiniFASNet (Silent-FAS) | CNN nhẹ | ~1 MB | ~15ms | Trung bình |
| Multi-modal (RGB + IR) | Dual-stream CNN | ~10 MB | ~30ms | Cao |
| Challenge-response (chớp mắt) | Blink detection | ~2 MB | ~200ms | Trung bình-Cao |

Khuyến nghị cho MVP:
- Dùng MiniFASNet + depth estimation (nếu camera hỗ trợ IR/depth)
- Nếu camera chỉ RGB: MiniFASNet + blink detection (2 frames)
- Print attack và replay attack là phổ biến nhất trong chấm công → ưu tiên chống 2 loại này

---

# 4. PHASE 3 — PRODUCTION SYSTEM (Tuần 17-24)

## 4.1 Tính năng production cần bổ sung

### Offline Mode
Edge device phải hoạt động khi mất kết nối cloud:
```
Online:   recognize → log → sync to cloud realtime
Offline:  recognize → log locally → queue sync → auto-sync khi có mạng lại

SQLite attendance log format:
  id | person_id | timestamp | confidence | condition_json | synced | sync_timestamp
```

### Gallery Synchronization
```
Cloud → Edge: enrollment mới, xóa người nghỉ việc
Edge → Cloud: attendance logs, gallery update metadata
Conflict resolution: cloud enrollment ALWAYS wins (authoritative source)

Sync protocol:
  1. Edge gửi last_sync_timestamp
  2. Cloud trả về delta (enrollments mới/xóa sau timestamp đó)
  3. Edge apply delta, gửi ACK
  4. Edge gửi queued attendance logs
```

### Multi-camera Support
```
Nếu 1 site có 2+ camera:
  - Shared gallery (load vào RAM 1 lần)
  - Per-camera IQA calibration (mỗi camera có L, N distribution khác nhau)
  - Dedup: nếu cùng person_id recognize trong < 30s → chỉ log 1 record
```

### Audit Trail
Mỗi decision phải log đầy đủ:
```json
{
  "timestamp": "2025-06-15T22:14:33Z",
  "camera_id": "cam-factory-01",
  "decision": "ACCEPT",
  "person_id": "EMP-0042",
  "confidence": 0.52,
  "threshold_used": 0.48,
  "context": {"L": 0.15, "N": 0.42, "q": 0.61, "hour": 22.24},
  "face_bbox": [102, 55, 230, 210],
  "det_score": 0.92,
  "anti_spoof_score": 0.97,
  "gallery_updated": false,
  "latency_ms": 67
}
```
Lưu ý: KHÔNG lưu ảnh gốc (privacy). Chỉ lưu metadata. Ảnh chỉ lưu tạm trong buffer
để xử lý, rồi xóa ngay.

## 4.2 Dashboard Requirements

```
Admin Dashboard (Cloud):
  ├── Attendance Overview
  │   ├── Today's records (realtime)
  │   ├── Attendance rate by department
  │   └── Late/absent alerts
  ├── System Health
  │   ├── Edge device status (online/offline)
  │   ├── Recognition rate per device
  │   ├── Condition distribution (% dark, medium, bright)
  │   └── Threshold adaptation history
  ├── Enrollment Management
  │   ├── Add/remove employees
  │   ├── Re-enrollment (khi quality thấp)
  │   └── Bulk import
  ├── Analytics
  │   ├── FRR/FAR trends over time
  │   ├── Gallery drift monitoring (anchor distance)
  │   └── Condition-specific accuracy
  └── Settings
      ├── Threshold parameters (τ_base, τ_floor, γ)
      ├── Gallery update policy (on/off, safety thresholds)
      └── Camera configuration
```

---

# 5. LỰA CHỌN MODEL CHI TIẾT

## 5.1 Face Detection

| Model | Params | Size | Speed (CPU) | mAP (WIDER) | Khuyến nghị |
|-------|--------|------|-------------|-------------|-------------|
| SCRFD-0.5G | 0.6M | 0.8 MB | ~8ms | 88.7% | Edge cực nhẹ |
| SCRFD-2.5G | 0.8M | 3.2 MB | ~15ms | 93.8% | **Khuyên dùng cho MVP** |
| SCRFD-10G | 3.9M | 15 MB | ~35ms | 95.2% | Khi cần accuracy cao |
| YOLOv8n-face | 3.2M | 6.5 MB | ~20ms | ~94% | Alternative |
| RetinaFace-MNet | 0.6M | 1.7 MB | ~12ms | 89.4% | Lightweight backup |

Khuyên dùng: **SCRFD-2.5G** — balance tốt nhất giữa accuracy và speed cho bài toán chấm công
(mặt thường lớn trong frame, không cần detect mặt siêu nhỏ).

## 5.2 Face Recognition (Embedding)

| Model | Backbone | Params | Size | Speed (CPU) | LFW | Khuyến nghị |
|-------|----------|--------|------|-------------|-----|-------------|
| MobileFaceNet | MobileNet | 1.2M | 4.5 MB | ~12ms | 99.5% | **Edge deployment** |
| ResNet34-ArcFace | R34 | 21M | 83 MB | ~45ms | 99.7% | Balanced |
| ResNet50-ArcFace | R50 | 25M | 166 MB | ~80ms | 99.78% | Server-side |
| ResNet100-ArcFace | R100 | 65M | 249 MB | ~150ms | 99.82% | Research baseline |
| AdaFace-R50 | R50 | 25M | 166 MB | ~80ms | 99.82% | **Nếu low-light là priority** |

Chiến lược hai lớp:
1. Research/validation: dùng ResNet100-ArcFace (best accuracy, không quan tâm speed)
2. Production edge: dùng MobileFaceNet (fast + small)
3. Nếu MobileFaceNet không đủ accuracy ở dark: chuyển ResNet34

LƯU Ý VỀ AdaFace: AdaFace pretrained trên WebFace4M với quality-aware margin.
Feature norm ‖z‖₂ CÓ THỂ informative hơn cho quality proxy q.
Nhưng cần verify: download AdaFace pretrained, check histogram ‖z‖₂ theo condition
trên data thực của bạn TRƯỚC khi commit.

## 5.3 Anti-Spoofing

| Model | Type | Size | Speed | Protocol | Khuyến nghị |
|-------|------|------|-------|----------|-------------|
| MiniFASNet | Binary CNN | 1.2 MB | ~15ms | Single frame | MVP |
| CDCN | Depth map | 8.5 MB | ~40ms | Single frame | Better accuracy |
| FAS-SGTD | Multi-frame | 12 MB | ~80ms | 3 frames | Best accuracy |
| Passive Liveness APIs | Cloud API | 0 MB | ~200ms | Single frame | Nếu có budget |

Khuyên dùng: **MiniFASNet cho MVP**, upgrade lên CDCN nếu cần accuracy cao hơn.

## 5.4 Tổng budget model trên edge

```
Minimal (CPU-only, MobileFaceNet):
  Detection:   SCRFD-2.5G         3.2 MB
  Recognition: MobileFaceNet      4.5 MB
  Anti-spoof:  MiniFASNet         1.2 MB
  Gallery:     50 persons × 512d  ~100 KB
  ────────────────────────────────────────
  Total:                          ~9 MB models + <1 MB gallery

Balanced (CPU, ResNet34):
  Detection:   SCRFD-2.5G         3.2 MB
  Recognition: R34-ArcFace        83 MB (INT8: ~22 MB)
  Anti-spoof:  MiniFASNet         1.2 MB
  Gallery:     200 persons × 512d ~400 KB
  ────────────────────────────────────────
  Total:                          ~27 MB models + <1 MB gallery

Full inference latency (CPU i5/Ryzen 5):
  Detection:    15ms
  Anti-spoof:   15ms
  Recognition:  12-45ms (tuỳ backbone)
  IQA:          2ms
  Matching:     <1ms (50 persons) / ~5ms (500 persons)
  ────────────────────────────────────────
  Total:        45-80ms → OK cho real-time (< 100ms target)
```

---

# 6. PHẦN CỨNG VÀ EDGE DEPLOYMENT

## 6.1 Edge Device Options

### Option A: Mini PC (Khuyên dùng cho MVP)

| Thiết bị | CPU | RAM | Giá (~VND) | Ưu điểm | Nhược điểm |
|----------|-----|-----|------------|---------|-------------|
| Intel NUC 12 (i5) | i5-1240P | 16GB | 8-12M | Mạnh, x86 ecosystem | Đắt, cần adapter |
| Beelink SER5 | Ryzen 5 5600H | 16GB | 5-7M | Giá tốt, đủ mạnh | Fan noise |
| **Beelink EQ12** | N100 | 16GB | 3-4M | **Rẻ, fanless, đủ dùng** | Yếu hơn |
| MinisForum UM590 | Ryzen 9 5900HX | 32GB | 10-14M | Rất mạnh | Quá đắt cho use case |

Khuyên dùng: **Beelink EQ12 (Intel N100)** cho deployment đại trà.
Intel N100 4 cores, TDP 6W, đủ chạy MobileFaceNet ONNX ở ~50ms.
Nếu cần mạnh hơn (ResNet34, hoặc multi-camera): Beelink SER5.

### Option B: ARM SBC (Single Board Computer)

| Thiết bị | SoC | RAM | Giá (~VND) | NPU | Khuyên |
|----------|-----|-----|------------|-----|--------|
| Raspberry Pi 5 | BCM2712 | 8GB | 2-3M | Không | Quá yếu cho FR |
| **Orange Pi 5** | RK3588S | 8GB | 2.5-3M | **6 TOPS NPU** | Tiềm năng |
| Radxa Rock 5B | RK3588 | 16GB | 4-5M | 6 TOPS NPU | Mạnh hơn |
| Khadas VIM4 | A311D2 | 8GB | 4-5M | 5 TOPS NPU | Alternative |

Nếu muốn dùng ARM + NPU: **Orange Pi 5 với RKNN toolkit** để deploy ONNX models
lên Rockchip NPU. Latency có thể giảm 3-5x so với CPU-only.
CẢNH BÁO: RKNN toolkit documentation kém, cần effort đáng kể để port model.

### Option C: NVIDIA Jetson (nếu cần GPU edge)

| Thiết bị | GPU | RAM | Giá (~VND) | TOPS | Khuyên |
|----------|-----|-----|------------|------|--------|
| Jetson Orin Nano | Ampere 1024 CUDA | 8GB | 5-7M | 40 TOPS | Overkill cho FR |
| Jetson Nano (cũ) | Maxwell 128 CUDA | 4GB | 3-4M | 0.5 TOPS | Đủ cho FR |

Jetson chỉ cần nếu: multi-camera (3+), hoặc thêm các task khác (object detection, people counting).
Cho bài toán chấm công đơn camera, Jetson là overkill.

### Khuyến nghị tổng hợp

```
Budget thấp (< 5M/site):     Orange Pi 5 + USB camera
Budget trung bình (5-8M):    Beelink EQ12 + IP camera  ← SWEET SPOT
Budget cao (> 10M):          Intel NUC i5 + Hikvision IP camera
Đã có hạ tầng IT:            Deploy trên PC hiện có (Docker)
```

## 6.2 Camera Selection

### Yêu cầu camera cho chấm công

```
Bắt buộc:
  - Resolution: ≥ 1080p (mặt cần ≥ 100x100 pixels trong frame)
  - FPS: ≥ 15fps
  - Focus: autofocus hoặc fixed focus phù hợp khoảng cách
  - Kết nối: USB hoặc RTSP (IP camera)

Khuyên dùng cho low-light:
  - Sensor size: ≥ 1/2.8" (lớn hơn = thu sáng tốt hơn)
  - IR illuminator: built-in hoặc external (quan trọng cho ca đêm)
  - WDR (Wide Dynamic Range): chống backlight

Không cần:
  - 4K (tốn bandwidth, face crop không cần >200x200)
  - PTZ (vị trí cố định)
  - Audio
```

### Camera options

| Camera | Loại | Giá (~VND) | IR | WDR | Ghi chú |
|--------|------|------------|-----|-----|---------|
| Logitech C920 | USB | 800K-1.2M | ❌ | ❌ | Chỉ cho sáng, dev/test |
| **Logitech BRIO 300** | USB | 1.5-2M | ❌ | ✅ | **Tốt cho văn phòng** |
| Hikvision DS-2CD2121 | IP | 1.5-2.5M | ✅ 30m | ✅ | Budget IP camera |
| **Hikvision DS-2CD2T47** | IP | 3-5M | ✅ 80m | ✅ | **Tốt cho nhà máy** |
| Dahua IPC-HFW2441T | IP | 2.5-4M | ✅ 60m | ✅ | Alternative |

Khuyên dùng:
- Văn phòng (sáng đều): Logitech BRIO 300 (USB, plug-and-play)
- Nhà máy ca đêm: **Hikvision DS-2CD2T47** (IR built-in, critical cho dark condition)

QUAN TRỌNG: Nếu camera có IR illuminator, ảnh IR sẽ là grayscale.
Face recognition model train trên RGB → cần validate accuracy trên IR images.
Có thể cần: (a) fine-tune model trên IR data, hoặc (b) dùng external visible-light
illuminator thay vì IR.

## 6.3 Sơ đồ lắp đặt vật lý

```
Bài toán: Cổng vào nhà máy, 1 làn

    ┌──────────────────────────────────────────────┐
    │                                              │
    │   [IR LED panel]    [Camera]    [IR LED]     │  ← Trần, cao 2.2m
    │         ↓              ↓           ↓         │
    │                                              │
    │              Khoảng cách: 0.8-1.5m           │
    │                                              │
    │         ┌──────────────────┐                 │
    │         │   Standing zone   │                 │  ← Vạch đứng trên sàn
    │         │  (dấu chân/vạch)  │                 │
    │         └──────────────────┘                 │
    │                                              │
    │   [Màn hình hiển thị]  [Edge Box]            │  ← Tường, ngang tầm mắt
    │   - Tên nhân viên      - Beelink EQ12        │
    │   - Trạng thái         - Ethernet cable      │
    │   - "Không nhận diện"  - Power               │
    │                                              │
    │   [Barrier/cổng từ]                          │  ← Mở khi ACCEPT
    │                                              │
    └──────────────────────────────────────────────┘

Lưu ý lắp đặt:
  - Camera góc 15-20° nhìn xuống (tránh trần → frontal face)
  - Standing zone cách camera 1m (mặt chiếm ~30% frame width → đủ resolution)
  - IR LED panel hướng vào standing zone nếu dùng visible light supplemental
  - Edge box đặt trong tủ kín, thông gió, có UPS nhỏ (chống mất điện)
```

---

# 7. DATA PIPELINE VÀ DATASET

## 7.1 Enrollment Pipeline

```
Enrollment workflow (admin thực hiện):
  
  1. Chụp 5 ảnh enrollment (ánh sáng tốt, frontal)
     ├── Tự động check quality: L > 0.5, blur > 100, det_score > 0.9
     ├── Reject ảnh kém quality, yêu cầu chụp lại
     └── Accept khi đủ 5 ảnh tốt
  
  2. Extract embeddings → compute mean prototype
     ├── Check inter-image consistency: pairwise cosine > 0.7
     ├── Loại outlier nếu có
     └── Store: anchor prototype (512-d vector)
  
  3. Check duplicate
     ├── Compare với tất cả prototypes hiện có
     ├── Nếu similarity > 0.6 với ai đó → cảnh báo "có thể trùng"
     └── Admin xác nhận
  
  4. Assign to gallery
     ├── Lưu prototype vào gallery.npz
     ├── Lưu enrollment metadata vào DB
     └── Sync to cloud + tất cả edge devices
```

## 7.2 Synthetic Data Augmentation

Vì dark dataset nhỏ, cần augment:

```python
import albumentations as A

# Realistic low-light augmentation pipeline
dark_augment = A.Compose([
    # Giảm brightness (mô phỏng thiếu sáng)
    A.RandomBrightnessContrast(
        brightness_limit=(-0.5, -0.2),  # chỉ giảm, không tăng
        contrast_limit=(-0.3, 0.1),
        p=0.8
    ),
    # Thêm Gaussian noise (mô phỏng ISO cao)
    A.GaussNoise(var_limit=(30, 100), p=0.7),
    # Giảm color saturation (camera low-light thường desaturate)
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=(-40, -10),
        val_shift_limit=(-30, 0),
        p=0.5
    ),
    # Motion blur nhẹ (người đi qua)
    A.MotionBlur(blur_limit=(3, 7), p=0.3),
    # JPEG compression (camera quality thấp)
    A.ImageCompression(quality_lower=50, quality_upper=80, p=0.3),
])

# Apply lên bright images để tạo synthetic dark pairs
for img_path in bright_images:
    img = cv2.imread(img_path)
    augmented = dark_augment(image=img)['image']
    # Save với label condition='synthetic_dark'
```

CẨN THẬN: Synthetic dark ≠ real dark. Camera low-light có color shift,
banding noise, và auto-exposure artifacts mà augmentation không capture.
LUÔN validate trên real dark data. Synthetic chỉ dùng để tăng diversity,
KHÔNG thay thế real data.

## 7.3 Continuous Data Collection

```
Sau khi deploy, hệ thống tự thu thập data cho improvement:

High-confidence accepts (score > τ_high):
  → Dùng cho gallery update (đã có trong pipeline)
  
Reject cases (score < τ nhưng gần ngưỡng):
  → Queue cho human review (admin xác nhận identity)
  → Nếu confirmed → trở thành labeled data cho recalibration
  
Edge cases (score ≈ τ, condition extreme):
  → Log metadata chi tiết
  → Dùng cho threshold recalibration định kỳ (monthly)
```

---

# 8. TESTING, MONITORING VÀ VẬN HÀNH

## 8.1 Testing Strategy

### Unit Tests
```python
# test_threshold.py
def test_adaptive_threshold_dark_lower_than_bright():
    """Threshold ở dark phải thấp hơn bright."""
    C_bright = [0.8, 0.1, 0, 1]  # L=0.8, N=0.1, midnight
    C_dark = [0.1, 0.5, 0, 1]    # L=0.1, N=0.5, midnight
    assert compute_threshold(C_dark) < compute_threshold(C_bright)

def test_threshold_never_below_floor():
    """Threshold không bao giờ < τ_floor."""
    C_worst = [0.0, 1.0, 0, 1]   # worst case
    assert compute_threshold(C_worst) >= TAU_FLOOR

def test_gallery_update_respects_anchor():
    """Update không drift quá xa anchor."""
    gallery = AdaptiveGallery()
    gallery.enroll("person1", enrollment_embeddings)
    
    # Fake 100 updates với embedding xa anchor
    for _ in range(100):
        fake_emb = np.random.randn(512)
        fake_emb = fake_emb / np.linalg.norm(fake_emb)
        gallery.try_update("person1", fake_emb, 0.8, 0.9, 0.7)
    
    # Prototype vẫn gần anchor
    sim = np.dot(gallery.prototypes["person1"], gallery.anchors["person1"])
    assert sim >= 0.60, f"Prototype drifted too far: sim={sim}"
```

### Integration Tests
```python
def test_full_pipeline_bright():
    """Full pipeline trên ảnh sáng phải recognize đúng."""
    result = pipeline.recognize(bright_image)
    assert result.decision == "ACCEPT"
    assert result.person_id == expected_person_id
    assert result.latency_ms < 100

def test_full_pipeline_dark_with_adaptation():
    """Sau adaptation, dark accuracy phải tăng."""
    # Phase 1: baseline dark accuracy
    acc_before = evaluate_dark(pipeline)
    
    # Phase 2: simulate 20 high-confidence updates
    for img in high_conf_dark_images:
        pipeline.process_and_update(img)
    
    # Phase 3: re-evaluate
    acc_after = evaluate_dark(pipeline)
    assert acc_after >= acc_before  # không yêu cầu +5% trong unit test
```

### Stress Tests
```
- 10 requests/second liên tục 1 giờ → latency p99 < 200ms?
- Gallery 500 persons → matching time < 10ms?
- 24h chạy liên tục → memory leak? (RSS không tăng > 10%)
- Camera disconnect rồi reconnect → recovery < 5s?
- 100 enrollment + 100 delete liên tục → gallery consistent?
```

## 8.2 Production Monitoring

### Metrics cần track realtime

```yaml
# System metrics (every 10s)
system.cpu_percent
system.memory_rss_mb
system.disk_usage_percent
system.temperature_celsius       # important cho edge box

# Pipeline metrics (every request)
pipeline.latency_total_ms
pipeline.latency_detection_ms
pipeline.latency_recognition_ms
pipeline.latency_matching_ms
pipeline.face_detected            # bool
pipeline.anti_spoof_score
pipeline.decision                 # ACCEPT/REJECT/UNKNOWN

# Quality metrics (aggregated hourly)
quality.avg_luminance
quality.avg_noise
quality.avg_quality_score
quality.pct_dark_conditions       # % requests ở dark bin
quality.threshold_mean            # τ(C) trung bình giờ đó

# Accuracy proxy metrics (aggregated daily)
accuracy.accept_rate              # % ACCEPT / total
accuracy.reject_rate              # % REJECT / total  
accuracy.unknown_rate             # % UNKNOWN / total
accuracy.gallery_update_count     # bao nhiêu lần update gallery
accuracy.avg_anchor_drift         # trung bình cosine(proto, anchor)
```

### Alert Rules

```yaml
alerts:
  - name: high_reject_rate
    condition: accuracy.reject_rate > 30% trong 1 giờ
    severity: WARNING
    action: Check camera, lighting condition
    
  - name: gallery_drift
    condition: accuracy.avg_anchor_drift < 0.70
    severity: CRITICAL  
    action: Gallery đang bị corrupt, cần review + possibly re-enroll
    
  - name: anti_spoof_anomaly
    condition: >10 spoof attempts trong 1 giờ
    severity: CRITICAL
    action: Someone đang cố bypass, cần kiểm tra physical
    
  - name: latency_spike
    condition: pipeline.latency_total_ms p95 > 200ms trong 5 phút
    severity: WARNING
    action: Check CPU load, có process khác đang chạy?
    
  - name: condition_shift
    condition: quality.pct_dark_conditions > 80% ban ngày (6-18h)
    severity: WARNING
    action: Camera có vấn đề? Lens bị che? Đèn hỏng?
```

## 8.3 Maintenance Schedule

```
Hàng ngày (tự động):
  - Sync attendance logs to cloud
  - Rotate local logs (giữ 7 ngày)
  - Health check report

Hàng tuần (tự động + review):
  - Gallery drift report (avg anchor distance per person)
  - Condition distribution report (% bright/medium/dark)
  - Reject case review queue (admin review borderline cases)

Hàng tháng (manual):
  - Threshold recalibration (nếu có đủ labeled reject cases)
  - Gallery cleanup (remove persons đã nghỉ việc)
  - Camera cleaning (physical lens cleaning!)
  - Model update check (có pretrained mới tốt hơn?)

Hàng quý:
  - Full accuracy audit (chọn 1 ngày, manual verify tất cả decisions)
  - Hardware health check (temperature trends, disk health)
  - Security audit (access logs, enrollment audit)
```

---

# 9. ƯỚC TÍNH CHI PHÍ VÀ TIMELINE

## 9.1 Chi phí phần cứng (per site)

### Setup tối thiểu (1 camera, 1 edge box)

```
                                    Budget      Standard     Premium
                                    ─────────   ──────────   ──────────
Edge device                         3-4M        5-7M         8-12M
  (Orange Pi 5)                     (EQ12)      (NUC i5)
Camera                              800K-1.2M   1.5-2M       3-5M  
  (C920)                            (BRIO 300)  (DS-2CD2T47)
Màn hình hiển thị                   1-2M        2-3M         3-5M
  (7" LCD)                          (10" touch)  (15" touch)
Phụ kiện (dây, mount, tủ, UPS)     500K-1M     1-2M         2-3M
IR LED panel (nếu cần)             0           500K-1M      1-2M
                                    ─────────   ──────────   ──────────
TOTAL per site:                     ~6-8M VND   ~10-15M VND  ~17-27M VND
                                    (~$240-320) (~$400-600)  (~$680-1080)
```

### Setup multi-camera (3 cameras, 1 site)

```
Edge device (mạnh hơn):                8-12M
Cameras × 3:                           9-15M
Managed switch (PoE):                  2-3M
Phụ kiện:                              3-5M
──────────────────────────────────────────────
TOTAL:                                 ~22-35M VND (~$880-1400)
```

## 9.2 Chi phí phần mềm và vận hành

```
Cloud server (cho 10 sites):
  VPS 4 vCPU, 8GB RAM:             ~1.5-3M/tháng
  Domain + SSL:                     ~500K/năm
  
Development (1 engineer, 6 tháng):  
  Phase 1 (research):              2 tháng
  Phase 2 (engineering):           2 tháng  
  Phase 3 (production):            2 tháng
  
Maintenance (ongoing):
  ~0.5 ngày/tuần/10 sites
```

## 9.3 Timeline tổng hợp

```
Tháng 1-2:   Phase 1 — Research Prototype
              ├── Data collection (2 tuần)
              ├── Baseline + IQA (2 tuần)
              ├── Adaptive threshold (2 tuần)
              └── Gallery update + ablation (2 tuần)
              
Tháng 3-4:   Phase 2 — Engineering MVP
              ├── ONNX optimization + edge setup (2 tuần)
              ├── API + pipeline integration (2 tuần)
              ├── Anti-spoofing integration (1 tuần)
              └── Testing + bug fixing (3 tuần)
              
Tháng 5-6:   Phase 3 — Production
              ├── Dashboard + cloud backend (2 tuần)
              ├── Offline mode + sync (1 tuần)
              ├── Pilot deployment 1 site (2 tuần)
              ├── Monitoring + alerting (1 tuần)
              └── Documentation + handover (2 tuần)

Tháng 7+:    Vận hành + mở rộng
              ├── Threshold recalibration từ production data
              ├── Scale to more sites
              └── Feature additions (multi-face, visitor mode, etc.)
```

---

# 10. RỦI RO VÀ MITIGATION

## 10.1 Technical Risks

| Rủi ro | Xác suất | Impact | Mitigation |
|--------|----------|--------|------------|
| Feature norm ‖z‖₂ không informative cho quality (ArcFace pretrained) | Trung bình | Cao — quality gate vô dụng | Validate sớm (Tuần 3). Backup: dùng SER-FIQ hoặc simple IQA thay thế |
| Gallery update gây prototype corruption | Trung bình | Rất cao | 4 safety gates + anchor constraint + manual reset option |
| Camera IR mode cho ảnh grayscale → accuracy drop | Cao | Trung bình | Test trên IR images sớm. Fallback: visible-light supplemental |
| Synthetic dark ≠ real dark → threshold calibration sai | Cao | Cao | Thu real dark data đủ lớn. Synthetic chỉ supplement |
| Model quá lớn cho edge budget | Thấp | Trung bình | Luôn có MobileFaceNet fallback |
| Anti-spoofing bị bypass (3D printed mask) | Thấp | Rất cao | Depth camera (tốn tiền) hoặc accept risk + physical security |

## 10.2 Operational Risks

| Rủi ro | Mitigation |
|--------|------------|
| Camera bị che/dơ lens | Alert khi dark conditions bất thường ban ngày |
| Mất điện | UPS nhỏ (giữ 15-30 phút) + auto-resume |
| Mất mạng | Offline mode, queue sync |
| Nhân viên thay đổi ngoại hình (cắt tóc, đeo kính, khẩu trang) | Gallery update tự động handle dần; re-enrollment nếu reject rate tăng |
| Privacy concern (GDPR/PDPA) | Không lưu ảnh, chỉ lưu embedding + metadata |
| Scale > 500 người/site | Hierarchical gallery (cluster prototypes), hoặc move matching to GPU |

## 10.3 Research Risks (cho thesis)

| Rủi ro | Mitigation |
|--------|------------|
| H1 (adaptive threshold) không đạt 10% FRR reduction | Nới lỏng target xuống 5%. Nếu vẫn không đạt → contribution là negative result + analysis tại sao |
| H2 (online adaptation) gây forgetting | Đây chính là finding thú vị. Report BWT, analyze conditions gây forgetting |
| Dataset quá nhỏ cho statistical significance | Augment + public dataset. Report confidence intervals |
| Reviewer cho rằng đây là engineering, không phải research | Strengthen bằng Bayesian theoretical justification + novel metric (CP-BWT) |

---

# PHỤ LỤC A: QUICK START COMMANDS

```bash
# === SETUP DEVELOPMENT ENVIRONMENT ===

# 1. Clone và setup
git clone <repo>
cd face-attendance
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install insightface onnxruntime opencv-python-headless \
    fastapi uvicorn sqlalchemy albumentations scikit-learn \
    numpy scipy

# 3. Download pretrained models
mkdir -p edge/models
# InsightFace models auto-download, hoặc manual:
# https://github.com/deepinsight/insightface/tree/master/model_zoo

# 4. Run baseline evaluation
python research/scripts/evaluate_baseline.py \
    --data-dir research/data/collected \
    --model buffalo_l \
    --output research/results/baseline.csv

# 5. Start edge API (development)
cd edge && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Run benchmark
python edge/scripts/benchmark.py --model mobilefacenet --iterations 100
```

# PHỤ LỤC B: CONFIG FILE TEMPLATE

```yaml
# edge/config/config.yaml

system:
  device_id: "edge-factory-01"
  log_level: INFO
  max_memory_mb: 512

camera:
  source: "rtsp://192.168.1.100:554/stream1"  # hoặc 0 cho USB
  width: 1920
  height: 1080
  fps: 15

models:
  detection:
    path: "models/det_scrfd_2.5g.onnx"
    input_size: [640, 640]
    conf_threshold: 0.5
    nms_threshold: 0.4
  recognition:
    path: "models/rec_mobilefacenet.onnx"
    embedding_dim: 512
  anti_spoof:
    path: "models/fas_minifasnet.onnx"
    threshold: 0.5

threshold:
  tau_base: 0.48
  tau_floor: 0.30
  gamma: 0.3       # interaction weight
  alpha: 1.0       # sigmoid scale for quality
  beta: 15.0       # sigmoid shift for quality

gallery:
  path: "storage/gallery.npz"
  max_persons: 200
  update:
    enabled: true
    min_match_score: 0.50
    min_quality: 0.2
    min_det_score: 0.7
    min_anchor_sim: 0.65
    learning_rate: 0.15
    max_learning_rate: 0.30

api:
  host: "0.0.0.0"
  port: 8000
  workers: 2

sync:
  cloud_url: "https://api.attendance.company.com"
  interval_seconds: 300
  retry_max
