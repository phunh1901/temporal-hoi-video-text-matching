# Temporal HOI Video–Text Matching

> Nghiên cứu Zero-shot / Open-vocabulary HOI Detection và học căn chỉnh video–văn bản trong không gian ngữ nghĩa chung.

**Người thực hiện:** Ngô Hoàng Phú
**Đối soát phạm vi:** 30/09/2026
**Tài liệu:** [PRD](docs/prd.md) · [Plan](plan.md) · [W01](reports/weekly/week_01.md) · [W02](reports/weekly/week_02.md)

## 1. Phạm vi nghiên cứu

| Module | Đầu vào và đầu ra mục tiêu | Đánh giá chính |
|---|---|---|
| **M1 — Zero-shot / Open-vocabulary HOI Detection** | Video và danh sách tương tác bằng văn bản → hộp người, hộp vật, nhãn tương tác, score và mốc khung hình; quỹ đạo là đầu vào/trung gian nếu phương pháp cần | HOI mAP theo evaluator và giao thức seen/unseen đã khóa |
| **M2 — Video–Text Cross-modal Alignment / Retrieval** | Video và câu mô tả → embedding được học căn chỉnh trong **Shared Semantic Embedding Space**, ma trận similarity và thứ hạng | R@1/5/10 và MedR; báo cáo riêng text→video và video→text |

M1 dùng **một tập nền tảng: VidHOI** để giữ định hướng tương tác trong video. M2 dùng **MSR-VTT** làm benchmark retrieval riêng. Không thu thập bộ dữ liệu hoặc huấn luyện mô hình riêng cho từng hành vi. Pilot hiện có chỉ dùng kiểm tra kỹ thuật.

Zero-shot cho phép học mô hình chung từ nhãn seen; nhãn unseen không được dùng để huấn luyện hoặc chọn cấu hình cho phép đánh giá đó. Open-vocabulary phải ghi rõ phần mở là tương tác, động từ hay vật thể. Không khẳng định mô hình nền tảng chưa từng gặp khái niệm trong pretraining.

**Giới hạn cần giải quyết ở W03:** ST-HOI là baseline lịch sử trên VidHOI, chưa chứng minh giao thức hiện dùng là zero-shot. Cần xác minh giao thức công bố và baseline tương thích. Nếu phải tạo split zero-shot riêng, ghi rõ đây là giao thức nội bộ; không coi đó là tái lập SOTA công bố. Chi tiết và điều kiện tiếp tục ở PRD mục 3 và Plan.

Phát hiện vi phạm theo ROI/thời lượng là **ứng dụng mở rộng có điều kiện**, sau hai module. Nhận biết hành vi chưa đủ để kết luận vi phạm. Event-F1 không thay thế HOI mAP; demo quy tắc mới không tự chứng minh zero-shot HOI.

## 2. Lộ trình và trạng thái thực tế

1. **Giai đoạn 1 — Baseline & Benchmark Enhancement:** khảo sát phương pháp hiện đại cho cả hai module; tái lập đánh giá bằng checkpoint trước, huấn luyện lại khi đủ tài nguyên; phân tích bottleneck/ablation và thử cải tiến nhỏ.
2. **Giai đoạn 2 — Novel Architecture & Publication:** từ bằng chứng giai đoạn 1, chọn một cơ chế mới có giả thuyết rõ; kiểm tra độ chính xác, chi phí hoặc khả năng tổng quát hóa; chuẩn bị bản thảo. Chưa cam kết tăng điểm hay được nhận bài.

**Đã có:** bộ đọc/dataset pilot, R@K/MRR/MedR và retrieval hai chiều, metric sự kiện, smoke test OpenCLIP, kiểm toán dữ liệu. Báo cáo W01/W02 đã hoàn thiện nội dung và bằng chứng kỹ thuật để nộp; xem [kiểm chứng mới](outputs/w02/completion_verification.json).
**Chưa có bằng chứng hoàn thành:** evaluator HOI mAP, bộ huấn luyện alignment, benchmark chính thức hai module, tái lập SOTA hoặc kiến trúc mới.

W02 giữ **8 đoạn từ 5 video, 22,63 MiB**, gồm 6 dev và 2 extension đã xem. Có 10 câu và **80 cặp clip–text, không phải 80 đoạn**. Không có test độc lập. Nguồn: [audit hiện có](outputs/w02/data_audit.json). Đây là ảnh chụp trạng thái; chạy lại audit khi dữ liệu thay đổi.

Cấu hình [w02_lightweight.json](configs/w02_lightweight.json): tối đa 50 MiB dữ liệu thô, 8 khung hình/đoạn, cạnh dài tối đa 320 px, batch 1, num_workers=0. Không áp hạn mức pilot này lên benchmark chính thức rồi so điểm như cùng giao thức. RAM đã đo cho loader không bao gồm model/huấn luyện.

## 3. Thiết kế thực nghiệm

- **M1:** VidHOI; ST-HOI là mốc lịch sử; chọn thêm baseline open-vocabulary phù hợp cùng giao thức ở W03. Nghiên cứu HOI ảnh chỉ dùng tham khảo, không thay ngầm benchmark video.
- **M2:** MSR-VTT, cấu hình mục tiêu 9k-train/1k-test của CLIP4Clip; cần xác minh và khóa danh sách ID ở W03. Mean pooling là mốc lịch sử. Chọn thêm ứng viên hiện đại theo mã nguồn, checkpoint và tài nguyên.
- **B0/B1 nội bộ:** global frame/union crop, cùng encoder, pooling, head và nguồn hộp. B1 phụ thuộc detector hoặc nhãn hộp đã xác minh.
- **Ablation:** ma trận temporal bật/tắt × geometry bật/tắt dùng cùng alignment head; kiểm tra head riêng. Giữ cùng split, ngân sách học và evaluator.
- **Huấn luyện M2 dự kiến:** đóng băng backbone ở cấu hình nhẹ, học head dùng contrastive loss đối xứng có xử lý nhiều positive; chọn checkpoint bằng validation. Cấu hình khởi đầu và điều kiện tài nguyên ở PRD.
- Tách kết quả dùng hộp/quỹ đạo chuẩn (oracle) khỏi kết quả hộp/quỹ đạo dự đoán. Không dùng test để sửa prompt, threshold hoặc kiến trúc.

R@K là tỷ lệ truy vấn có ít nhất một đáp án đúng trong top K theo giao thức; MedR là trung vị thứ hạng đáp án đúng đầu tiên; MRR là trung bình nghịch đảo thứ hạng đó. R@3 và Pairwise Accuracy chỉ là chỉ số phụ của pilot. Luôn ghi chiều retrieval, số ứng viên và quy tắc nhiều positive.

## 4. Dữ liệu và kiến trúc

Pilot dùng các manifest thực tế:

| File | Trường chính |
|---|---|
| clips.csv | clip_id, video_id, video_path, session_id, start_s, end_s, behavior_id, rule_id, has_violation |
| texts.csv | text_id, behavior_id, prompt_text, prompt_type, description |
| splits.csv | clip_id, split, session_id; hiện có dev/extension |
| relevance.csv | clip_id, text_id, is_match (0/1), reviewer, reviewed_at |
| sources.csv | video_id, video_path, source_url, attribution, license_status, license_url, sha256 |

Nhãn vi phạm trống được đọc thành None. Cặp clip–text chưa biết được bỏ khỏi bảng relevance, không gán âm tính và không ghi is_match trống. Giấy phép kho code không tự xác nhận quyền dùng video. Benchmark chính thức cần adapter/schema riêng cho hộp HOI, lớp seen/unseen và caption; không coi CSV pilot đã chứa các nhãn đó.

Kiến trúc mục tiêu gồm data → models → training/inference → evaluation. M1 và M2 có đầu ra/evaluator độc lập; có thể chia sẻ biểu diễn khi kiểm chứng được lợi ích. Module data chỉ xử lý dữ liệu; evaluation chỉ chấm điểm. Những phần models/training/inference chưa được triển khai không xuất hiện như tính năng đã chạy.

```text
configs/                       Cấu hình pilot
data/manifests/                 Metadata và nhãn pilot
docs/prd.md                    Yêu cầu nghiên cứu và giao thức mục tiêu
plan.md                        Công việc W01–W10, phụ thuộc và điều kiện nghiệm thu
reports/weekly/                Một báo cáo mỗi tuần
scripts/                       Smoke test, đọc mẫu và kiểm toán
src/temporal_hoi/data/          VideoReader và TemporalHOIDataset
src/temporal_hoi/evaluation/    R@K/MRR/MedR hai chiều, pairwise, tIoU, ghép sự kiện
tests/                         Kiểm thử dữ liệu và metric
pyproject.toml, uv.lock         Môi trường dự án chính
```

## 5. Cài đặt và ví dụ đã đối chiếu API

Chạy từ thư mục gốc với Python 3.12 và uv. Môi trường chính hiện ưu tiên CPU. Baseline bên ngoài có thể cần môi trường riêng; khóa dependency không bảo đảm kết quả giống nhau trên mọi phần cứng.

```powershell
uv sync --locked --group dev
uv run --frozen python scripts/check_environment.py
uv run --frozen pytest
uv run --frozen python scripts/verify_weekly.py
```

Đọc pilot đã có trên máy; ví dụ cần video cục bộ tại đường dẫn trong manifest:

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
print(sample["clip_id"], len(sample["pil_frames"]), sample["has_violation"])
```

Chấm sự kiện trên dữ liệu giả để minh họa API, không phải kết quả nghiên cứu:

```python
from temporal_hoi.evaluation import compute_event_metrics

ground_truth = [
    {"video_id": "clip_01", "rule_id": "R01", "start_s": 2.0, "end_s": 8.0}
]
predictions = [
    {"video_id": "clip_01", "rule_id": "R01", "start_s": 2.2, "end_s": 7.9, "score": 0.88}
]
results = compute_event_metrics(
    predictions=predictions,
    ground_truths=ground_truth,
    iou_threshold=0.5,
)
for key in ("precision", "recall", "f1_score"):
    value = results[key]
    print(f"{key}: {value:.3f}" if value is not None else f"{key}: undefined")
```

Retrieval hai chiều dùng `compute_bidirectional_retrieval(scores, video_to_text_positives, text_to_video_positives)`, với scores có shape [video,text]. API mới mặc định R@1/5/10 và trả MRR/MedR, rank 1-based cùng số query được chấm/bị loại. Truy vấn thiếu annotation dùng None; ties giữ thứ tự candidate. Ví dụ tính tay và giới hạn ở [báo cáo W02](reports/weekly/week_02.md).

## 6. Nguồn tham khảo

- [ST-HOI/VidHOI — Chiou et al., ACM ICMR Workshop 2021](https://github.com/coldmanck/VidHOI): baseline HOI trong video và evaluator tham chiếu.
- [CLIP4Clip — mã nguồn tác giả](https://github.com/ArrowLuo/CLIP4Clip): baseline retrieval lịch sử, không tự coi là SOTA hiện tại.
- [SL-HOI — CVPR 2026](https://github.com/MPI-Lab/SL-HOI): ứng viên khảo sát open-vocabulary HOI trên ảnh; chưa phải baseline video đã tái lập.
- [OpenCLIP](https://github.com/mlfoundations/open_clip): thư viện triển khai và checkpoint; tên thư viện không phải tên một kiến trúc đối chứng riêng.
- [Thư mục báo cáo của dự án](https://drive.google.com/drive/folders/1gzjTKl1uKl0Vh60P39wmgpQdl2FZudV1?hl=vi).
