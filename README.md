# Temporal HOI Video–Text Matching

Nghiên cứu hai hướng: **NC1** phát hiện sự kiện vi phạm bằng hành vi + vùng + thời gian; **NC2** so khớp video–văn bản. Dùng encoder ảnh/chữ tiền huấn luyện cùng hệ, đóng băng trọng số; không train/fine-tune adapter, projection hoặc classifier.

Phạm vi hiện hành: [kế hoạch 10 tuần](plan.md). Đỗ xe và đổ rác là hai hành vi chính; đi xe đạp là hành vi mở rộng. Các thiết kế có huấn luyện trong [PRD cũ](docs/prd.md) chỉ là bối cảnh lịch sử.

## Trạng thái đã kiểm chứng ngày 30/09/2026

- **W01:** smoke test OpenCLIP chạy lại thành công với 8 hình và 4 cosine hữu hạn. Đây là kiểm tra kỹ thuật trên video mẫu, chưa đo chất lượng phát hiện hành vi.
- **W02 theo yêu cầu máy cá nhân:** dùng 8 đoạn/5 video hiện có (22,63 MiB), 6 dev +2 demo mở rộng. Không tải thêm video; bỏ chỉ tiêu 80/120. Bộ đọc và nhãn mô tả mẫu đã kiểm tra; chưa có dữ liệu đủ bằng chứng để chấm vi phạm hai hành vi.
- Bộ chấm và bộ đọc đã sửa lỗi, có kiểm thử hồi quy. Đã sửa nhãn theo ảnh rà, thêm nguồn/quyền dùng và bảng relevance; ghi rõ AI rà ảnh mẫu, không là người chấm độc lập. Nhãn vi phạm chưa đủ bằng chứng giữ null.
- Báo cáo hiện hành: [tuần 01](reports/weekly/week_01.md), [tuần 02](reports/weekly/week_02.md). Chưa có chứng cứ nộp báo cáo; các bản trong docs/reports được giữ để truy vết.

## Cài đặt và kiểm tra

Python 3.12, môi trường CPU theo uv.lock:

```powershell
uv sync --locked --group dev
uv run --frozen python scripts/check_environment.py
uv run --frozen python -m pytest -q
uv run --frozen ruff check .
```

Chạy từ thư mục gốc dự án:

```powershell
uv run --frozen python scripts/smoke_test.py --output-dir outputs/w01/review_run
uv run --frozen python scripts/test_dataloader.py
uv run --frozen python scripts/audit_data.py
```

Smoke test cần data/sample/smoke_sample.mp4 và checkpoint ViT-B-32 / laion2b_s34b_b79k (lần đầu có thể cần tải). Video và trọng số không được commit. Script download_samples.py tải các video mẫu khác phục vụ bộ đọc; không cung cấp tập benchmark đã nghiệm thu hoặc nhãn đã duyệt.

Bộ benchmark đọc 6 clip dev lần lượt, cạnh hình tối đa 320 px, không import PyTorch khi chỉ đọc ảnh. Cấu hình configs/w02_lightweight.json giới hạn dữ liệu thô 50 MiB. Audit trả PASS_TECHNICAL_PILOT khi bộ mẫu đạt kiểm tra, còn research_status báo riêng khả năng đánh giá vi phạm thật.

## Quy ước dữ liệu và phép đo

- Dataset kiểm tra khóa clip duy nhất, đầy đủ split và không trùng session/video giữa các tập.
- texts.csv là mẫu prompt theo hành vi, **không phải nhãn đúng cho từng clip**. Bảng data/manifests/relevance.csv hiện chứa 80 cặp nhãn cho 8 clip/10 câu. Truyền relevance_csv để nạp; thiếu nhãn giữ unknown.
- VideoReader lấy 8 hình khác nhau trong [start,end), kiểm tra timestamp decoder. Video lỗi/thiếu hình/ngoài khoảng báo lỗi; không chèn ảnh đen. Hỗ trợ CFR, cần chuyển VFR sang CFR trước dùng.
- Sự kiện bắt buộc có video_id, rule_id, start_s, end_s; score tùy chọn. Ghép một–một cùng video/luật, tIoU ≥0,5; dự đoán trùng tính FP.
- Mẫu số 0 trả None/null. Pairwise dùng đúng > sai (hòa không thắng). FA/giờ cần số cảnh báo riêng trên video bình thường và thời lượng thật.
- Test chính khóa đến tuần 8; mẫu test đã xem trong quá trình dựng bộ đọc không được coi là test chính chưa từng mở.

## Tổ chức

| Thư mục | Nội dung |
|---|---|
| src/temporal_hoi/data/ | Đọc video và danh mục |
| src/temporal_hoi/evaluation/ | Chấm matching và sự kiện |
| tests/ | Ví dụ tính tay, lỗi biên, chống rò rỉ |
| data/manifests/ | Clip, prompt mẫu, split và trạng thái nhãn |
| scripts/ | Chạy mẫu, benchmark, audit |
| outputs/w01/, outputs/w02/ | Bằng chứng máy |
| reports/weekly/ | Một báo cáo hiện hành mỗi tuần |

[CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) là nguồn khảo sát; báo cáo tuần 01 giải thích lựa chọn và giới hạn.

## Dùng bộ mẫu nhỏ

```python
from temporal_hoi.data.dataset import TemporalHOIDataset

dataset = TemporalHOIDataset(
    split="dev",
    relevance_csv="data/manifests/relevance.csv",
    max_frame_side=320,
)
for item in dataset:
    # Chỉ giữ một item/lần; chưa có nhãn vi phạm thì has_violation là None.
    print(item["clip_id"], len(item["pil_frames"]), item["has_violation"])
```

Bản manifest trước rà giữ trong data/manifests/archive/pre_lightweight/. Ảnh rà và hash nguồn tại outputs/w02/review/. Bộ hiện tại đã xem nên không gọi là test mù; không sử dụng clip đường cao tốc để giả làm người đổ rác.
