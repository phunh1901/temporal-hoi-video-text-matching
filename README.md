# Temporal HOI Video–Text Matching

> Nghiên cứu Zero-shot / Open-vocabulary Human-Object Interaction (HOI) Detection và học căn chỉnh Video–Văn bản trong không gian ngữ nghĩa chung.

**Sinh viên thực hiện:** Ngô Hoàng Phú  
**Tài liệu kỹ thuật:** [PRD](docs/prd.md) · [Giao thức thực nghiệm (Protocols)](docs/protocols/)

---

## 1. Phạm vi nghiên cứu

Hệ thống được thiết kế thành hai mô-đun độc lập nhưng có khả năng tương hỗ biểu diễn:

| Mô-đun | Đầu vào & Đầu ra | Mục tiêu & Đánh giá |
|---|---|---|
| **M1 — Zero-shot / Open-vocabulary HOI Detection** | Video và danh sách tương tác bằng văn bản $\to$ Hộp người, hộp vật, nhãn tương tác, điểm số (score) theo từng mốc thời gian. | Nhận diện chi tiết hành vi người–vật trong video; đánh giá bằng **frame-level HOI mAP** trên tập **VidHOI**. |
| **M2 — Video–Text Cross-modal Retrieval** | Video và câu mô tả tự nhiên $\to$ Biểu diễn trong **Shared Semantic Embedding Space**, ma trận tương đồng và xếp hạng. | Tìm kiếm video theo văn bản và ngược lại; đánh giá bằng **Recall@1/5/10, MedR, MRR** trên tập **MSR-VTT**. |

- **Mô-đun 1 (Video HOI):** Sử dụng tập nền tảng **VidHOI** (78 lớp vật thể, 50 lớp hành động). Nghiên cứu tập trung vào hướng tiếp cận tập mở (open-set/zero-shot) như ACoLP hoặc chuyển giao từ mô hình ảnh (SL-HOI), phân tách rõ giữa đánh giá với hộp chuẩn (Oracle) và hộp do mô hình tự dự đoán.
- **Mô-đun 2 (Video–Text Retrieval):** Sử dụng tập chuẩn **MSR-VTT** theo phân chia chuẩn JSFusion (9.000 train / 1.000 test). Thiết kế tập validation (8.100 train / 900 validation) gom nhóm theo URL video gốc nhằm loại bỏ hoàn toàn nguy cơ rò rỉ dữ liệu (data leakage).

---

## 2. Kiến trúc Hệ thống

Hệ thống xử lý luồng dữ liệu đa phương thức qua các tầng:

```mermaid
flowchart LR
    subgraph Input [Đầu vào]
        V[Video thô]
        T[Câu mô tả / Nhãn tương tác]
    end

    subgraph M1 [Mô-đun 1: Temporal HOI]
        VR[VideoReader / Dataloader]
        HD[Phát hiện Bounding Box Người/Vật]
        HA[Nhận diện tương tác theo thời gian]
        VR --> HD --> HA
    end

    subgraph M2 [Mô-đun 2: Video-Text Retrieval]
        VE[Visual Encoder]
        TE[Text Encoder]
        SS[Shared Semantic Embedding Space]
        SIM[Tính ma trận tương đồng & Xếp hạng]
        VE --> SS
        TE --> SS
        SS --> SIM
    end

    subgraph Evaluation [Đánh giá chuẩn mực]
        MAP[HOI frame-level mAP]
        RET[Recall@K / MedR / MRR hai chiều]
    end

    V --> VR
    V --> VE
    T --> TE
    HA --> MAP
    SIM --> RET
```

---

## 3. Cấu trúc Thư mục Kho lưu trữ

Cấu trúc các thư mục và tệp tin chính thức trên Git:

```text
configs/                                Cấu hình tham số thực nghiệm (JSON)
  ├── w02_lightweight.json              Cấu hình trích xuất khung hình và pilot nhẹ
  ├── w03_msrvtt_protocol.json          Cấu hình benchmark và chia tập MSR-VTT
  └── w03_vidhoi_protocol.json          Cấu hình đánh giá mAP trên VidHOI
docs/                                   Tài liệu phân tích và đặc tả kỹ thuật
  ├── prd.md                            Tài liệu yêu cầu sản phẩm và nghiên cứu (PRD)
  └── protocols/                        Đặc tả giao thức thực nghiệm
      ├── metric_conformance_w03.md     Đối chuẩn độ đo với CLIP4Clip
      ├── msrvtt_w03.md                 Giao thức chia tập MSR-VTT chống rò rỉ dữ liệu
      └── vidhoi_w03.md                 Giao thức đánh giá frame mAP trên VidHOI
scripts/                                Kịch bản tiện ích thực thi
  ├── check_environment.py              Kiểm tra môi trường máy (PyTorch, CPU/GPU, OS)
  ├── download_samples.py               Tải video mẫu về máy cục bộ
  ├── prepare_lightweight_pilot.py       Khởi tạo dữ liệu nhãn mẫu
  ├── smoke_test.py                     Chạy thử nghiệm luồng trích xuất video–text với OpenCLIP
  ├── lock_msrvtt_protocol.py           Thuật toán phân chia dữ liệu MSR-VTT
  └── compare_benchmark_metrics.py      Đối chiếu các hàm đo lường với CLIP4Clip
src/temporal_hoi/                       Mã nguồn thuật toán cốt lõi
  ├── data/                             Bộ nạp video và xử lý tập dữ liệu
  │   ├── dataset.py                    Lớp TemporalHOIDataset
  │   └── video_reader.py               Lớp VideoReader giải mã khung hình
  └── evaluation/                       Bộ công cụ tính toán độ đo
      └── metrics.py                    Recall@K, MedR, MRR hai chiều, tIoU, ghép sự kiện
tests/                                  Bộ kiểm thử tự động (Unit tests chạy với pytest)
  ├── test_benchmark_metrics.py         Kiểm thử đối chuẩn bộ đo với mã nguồn tác giả
  ├── test_data.py                      Kiểm thử VideoReader và TemporalHOIDataset
  ├── test_metrics.py                   Kiểm thử các hàm đo lường toán học
  ├── test_msrvtt_protocol.py           Kiểm thử logic phân chia dữ liệu MSR-VTT
  └── test_retrieval.py                 Kiểm thử tìm kiếm video–văn bản hai chiều
.gitignore                              Quy tắc loại trừ dữ liệu, kết quả chạy và kế hoạch cá nhân
pyproject.toml, uv.lock                 Cấu hình môi trường và quản lý gói phụ thuộc (uv)
README.md                               Tài liệu giới thiệu dự án (Project Landing Page)
```

---

## 4. Cài đặt và Bắt đầu nhanh (Quickstart)

Dự án sử dụng Python 3.12 và công cụ quản lý gói `uv`.

### 4.1. Thiết lập môi trường
```powershell
# Đồng bộ môi trường và các gói phát triển
uv sync --locked --group dev

# Kiểm tra tương thích phần cứng và môi trường
uv run --frozen python scripts/check_environment.py
```

### 4.2. Chạy bộ kiểm thử tự động (Unit Tests)
```powershell
# Chạy toàn bộ 60 bài kiểm thử tự động
uv run --frozen pytest
```

### 4.3. Tải dữ liệu mẫu và chạy Smoke Test
```powershell
# Tải video mẫu nhẹ
uv run --frozen python scripts/download_samples.py

# Khởi tạo nhãn mẫu nếu bắt đầu từ đầu
uv run --frozen python scripts/prepare_lightweight_pilot.py

# Chạy thử nghiệm luồng trích xuất đặc trưng video–text mẫu với OpenCLIP
uv run --frozen python scripts/smoke_test.py
```

---

## 5. Ví dụ sử dụng API

### Đọc dữ liệu video
```python
from temporal_hoi.data import TemporalHOIDataset

dataset = TemporalHOIDataset(
    split="dev",
    clips_csv="data/manifests/clips.csv",
    texts_csv="data/manifests/texts.csv",
    splits_csv="data/manifests/splits.csv",
    relevance_csv="data/manifests/relevance.csv",
    num_frames=8,
    max_frame_side=320,
)
sample = dataset[0]
print("Clip ID:", sample["clip_id"], "| Frames:", len(sample["pil_frames"]))
```

### Đánh giá tìm kiếm hai chiều (Bidirectional Retrieval)
```python
import numpy as np
from temporal_hoi.evaluation import compute_bidirectional_retrieval

# Ma trận điểm số tương đồng giữa video và text [num_videos, num_texts]
similarity_scores = np.random.rand(10, 10)
# Danh sách positive index cho từng query
video_to_text_positives = [[i] for i in range(10)]
text_to_video_positives = [[i] for i in range(10)]

results = compute_bidirectional_retrieval(
    scores=similarity_scores,
    video_to_text_positives=video_to_text_positives,
    text_to_video_positives=text_to_video_positives,
)
print("Text-to-Video R@1:", results["text_to_video"]["R@1"])
print("Video-to-Text R@1:", results["video_to_text"]["R@1"])
```
