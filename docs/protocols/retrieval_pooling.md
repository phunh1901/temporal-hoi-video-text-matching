# Đối chứng B0 B1 B2 cho truy vấn video và văn bản

Luồng này chạy M2 từ đặc trưng khung hình của encoder CLIP4Clip đóng băng. Mã
thuật toán nằm trong `src/temporal_hoi/inference/retrieval_pooling.py`; bộ điều
phối nằm trong `pooling_workflow.py` cùng thư mục. Có thể tính lại từ cache đã
xác minh hoặc tạo cache mới từ video và checkpoint cục bộ. Không cần mã trong
thư mục kết quả thử nghiệm để import hoặc chạy chương trình.

## Ba phương pháp và điều kiện giữ cố định

- **B0:** chuẩn hóa L2 từng frame hợp lệ, lấy trung bình, chuẩn hóa vector video;
  đối chiếu với vector văn bản đã chuẩn hóa. Giữ phép tính CPU/NumPy FP32 của
  baseline để tránh thay đổi hạng khi các ứng viên hòa điểm.
- **B1:** tính độ tương đồng câu truy vấn với từng frame, chia cho `tau`, dùng
  softmax trên các frame hợp lệ rồi gom đặc trưng theo trọng số; chuẩn hóa và
  đối chiếu với câu truy vấn ban đầu. Mặc định `tau=0.05`.
- **B2:** kết hợp **điểm số** `(1-alpha) × B0 + alpha × B1`, với `alpha=0.25`.
  Đây không phải phép trộn trực tiếp hai vector video.

Ba bản dùng cùng cache, checkpoint, frame mask, caption, thứ tự ID và tập ứng
viên. B1/B2 ở đây không học tham số mới, không bổ sung hộp người–vật và không
được gọi là đối chứng temporal × geometry. Chúng khác nhánh global–union crop
dùng tên B0/B1 trong `configs/global_local_comparison.json` và `configs/hoi_union_crop.json`.
Head có học B3 là thí nghiệm riêng, không được đưa vào lệnh chạy này.

Điểm có trục `[số câu, số video]`. Điểm cosine được nhân `logit_scale` dương,
giữ mặc định 100 cho cấu hình tham chiếu. Giá trị này phải phù hợp cache và
checkpoint; điểm không phải xác suất. Bộ chấm nội bộ báo R@1/5/10 theo **phần
trăm**, MedR, MeanR và MRR theo hai chiều; mỗi truy vấn có đúng ít nhất một
positive theo ID. Khi bằng điểm, ưu tiên thứ tự ứng viên đã khóa. Bộ chấm gốc
CLIP4Clip có cách tính hòa điểm khác và được báo riêng khi áp dụng được, không
trộn hai cách tính trong một bảng.

Phần tương thích evaluator gốc ghi nguồn ArrowLuo/CLIP4Clip và giữ
[thông báo MIT](../../src/temporal_hoi/inference/CLIP4Clip_LICENSE.txt) cạnh mã.

## Đầu vào và ranh giới Git

Chuẩn bị metadata archive và protocol lock như [giao thức MSR-VTT](msrvtt_protocol.md).
Không phân chia lại train/test hoặc đổi caption khi chạy B1/B2. Cấu hình mẫu
[msrvtt_pooling.json](../../configs/msrvtt_pooling.json) chứa đường dẫn tương đối
và các trường hash còn trống; điền bằng đầu vào thật trong một bản cấu hình local.

Cache cần các trường:

| Trường | Shape hoặc ý nghĩa |
|---|---|
| `text` | FP32 `[T,D]`, đặc trưng câu truy vấn |
| `frames` | FP32 `[V,F,D]`, đặc trưng từng frame; giữ frame padding cùng mask |
| `mask` | Boolean `[V,F]`; mỗi video có ít nhất một frame hợp lệ |
| `text_ids` | `T` ID câu duy nhất, đúng thứ tự pool |
| `text_video_ids` | `T` ID video positive tương ứng với các câu |
| `video_ids` | `V` ID video duy nhất, đúng thứ tự tập ứng viên |
| `metadata` | Thông tin nguồn encoder, preprocessing, split và cache |

Tệp `.pt` chỉ được đọc khi SHA256 trùng và `torch.load(weights_only=True)` thành
công; không có cách đọc pickle không an toàn khi thất bại. Tệp `.npz` được đọc
với `allow_pickle=False`. Những ID/cache không khớp pool, frame lỗi hoặc vector
không hữu hạn/không hợp lệ đều làm dừng chạy, không bị loại âm thầm.

Giữ dữ liệu, cache, trọng số và kết quả trong `data/`, `weights/`, `outputs/` đã
được ignore. Source CLIP4Clip tham chiếu đặt tại `data/reference/CLIP4Clip`, gồm
license và BPE, theo đúng commit/hash trong [giao thức native](baseline_inference.md).
Mã công khai không phân phối video, checkpoint, cache hay bản ghi kết quả cá nhân.

## Chạy từ cache đã có

Xem đầy đủ tham số:

```powershell
uv run --frozen python scripts/run_msrvtt_pooling.py --help
```

Điền `metadata_archive`, `metadata_lock`, cache/hash và reference/hash vào config
local. Reference NPZ cần điểm baseline B0, `text_ids` và `video_ids` đúng thứ tự;
`reference_is_logits` xác định điểm reference đã nhân scale hay chưa. Giữ
`require_external_reference=true` để chặn B1/B2 khi B0 chưa đối chiếu đạt.

```powershell
# Kiểm đầu vào; mỗi lần dùng thư mục output mới
uv run --frozen python scripts/run_msrvtt_pooling.py --config <run.json> --output-dir outputs/pooling_preflight

# Chạy đúng cấu hình đã chốt trên tập test
uv run --frozen python scripts/run_msrvtt_pooling.py --config <run.json> --execute --allow-final-evaluation --output-dir outputs/pooling_test
```

Đối với test, config còn cần `selection_fixed_before_scores=true`. Đây là khai
báo trách nhiệm của người chạy, không biến test đã xem thành tập chưa từng truy
cập. Không chọn `tau`, `alpha`, checkpoint hoặc prompt từ kết quả test. Chỉ dùng
train/validation phù hợp nguồn supervision cho phát triển phương pháp.

`--feature-cache`, `--cache-sha256`, `--baseline-reference` và `--reference-sha256`
có thể ghi đè những trường đường dẫn/hash tương ứng. Các đường dẫn là input
tường minh; chương trình không tự dò file trong thư mục kết quả cũ.

Nếu chỉ kiểm phần mềm hoặc dữ liệu riêng chưa có reference, có thể dùng
`--allow-implementation-only`. Kết quả phải được đọc đúng là kiểm công thức
nội bộ, chưa xác nhận tái lập baseline bên ngoài. Không dùng tùy chọn này để
bỏ qua reference đã không khớp.

## Tạo cache mới từ video và checkpoint

Điền `source_root`, `media_root`, `checkpoint`, `checkpoint_sha256` và
`checkpoint_provenance`; giữ cùng archive/lock, frame sampling và precision.

```powershell
uv run --frozen python scripts/run_msrvtt_pooling.py --config <run.json> --build-cache --output-dir outputs/pooling_media_preflight
uv run --frozen python scripts/run_msrvtt_pooling.py --config <run.json> --build-cache --execute --allow-final-evaluation --output-dir outputs/pooling_media_test
```

Native loader xác minh mã nguồn trước import, dùng BPE/caption gốc, chọn tối đa
12 frame theo quy tắc 1 Hz rồi lấy đều, RGB/CLIP preprocessing và FP32. Encoder
được đóng băng. Lỗi thiếu media/checkpoint/CUDA hoặc giải mã làm dừng chạy;
không tải thay mô hình hoặc tạo dữ liệu giả. Thiết bị mặc định là CPU; chỉ đổi
`device` sang CUDA trong môi trường và ngân sách được phép. Đo mẫu nhỏ trước
khi encode toàn bộ; cache replay không đo lại chi phí giải mã/encoder.

## Đầu ra và kiểm chứng

Mỗi lần chạy dùng thư mục mới, ghi cấu hình, hash/provenance, tình trạng parity,
ma trận điểm, thứ hạng và metric. B0 phải đạt kiểm công thức và thứ hạng hai chiều;
chế độ mặc định còn yêu cầu reference bên ngoài đạt trước khi chạy B1/B2. Không
đánh đồng khớp số thực trong dung sai với giống tuyệt đối từng bit.

So sánh B1−B0/B2−B0 giữ tập ứng viên cố định; bootstrap lấy mẫu theo ID video,
không thay cho nhiều seed huấn luyện hay chứng minh độc lập nguồn. Khi encoder
đã học train9k, validation900 nằm trong nguồn supervision của encoder; không
gọi đây là validation độc lập. Không suy M1, chất lượng zero-shot hoặc hoàn
thành toàn đồ án từ các metric retrieval.

Kiểm mã cục bộ:

```powershell
uv run --frozen ruff check src/temporal_hoi/inference scripts/run_msrvtt_pooling.py tests/test_retrieval_pooling.py tests/test_msrvtt_pooling_cli.py
uv run --frozen pytest tests/test_retrieval_pooling.py tests/test_msrvtt_pooling_cli.py
```

Các test nhỏ kiểm thuật toán, input và đường chạy; chúng không phải một lượt
inference checkpoint trên benchmark. Không cần encoder, mạng hoặc GPU để chạy
các test này.
