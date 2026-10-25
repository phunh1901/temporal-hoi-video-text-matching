# Học head video–text từ đặc trưng đóng băng

Đường học này thực hiện PRD mục 5.2–6.2: mặc định hai head tuyến tính không bias,
chuẩn hóa L2, AdamW và loss contrastive nhiều positive hai chiều. Đây là head
mới trên encoder OpenCLIP đã xác minh ở B0; không gọi nó là tái lập huấn luyện
CLIP4Clip hoặc thay thế checkpoint fine-tuned MSR-VTT của tác giả.

Đối chứng `head=mlp` thêm hai lớp GELU với `hidden_dim` khai báo trong config,
giữ cùng feature/loss/split. Cấu hình, probe nhiều seed và audit được mô tả trong
[development_controls.md](development_controls.md). Chưa có kết quả benchmark
linear/MLP từ việc triển khai hoặc probe tổng hợp.

## Cấu hình và điều kiện dữ liệu

`configs/frozen_feature_alignment.json` chứa toàn bộ tham số bắt đầu: 8 frame/video,
decode batch 1, feature batch 16 video khác nhau, lr `1e-4`, weight decay `0.01`,
temperature cố định `0.07`, tối đa 10 epoch, patience 3, seed 42, CPU float32.
Số chiều lấy từ encoder/cache; head vuông khởi tạo identity. Encoder hoàn toàn
đóng băng; input feature bị detach tại ranh giới head.

Chỉ `train` 8.100 video / 162.000 caption và `validation` 900 video / 18.000
caption source-group v2 được dùng. Adapter xác minh archive/lock/caption/pool
hash và không chấp nhận source URL trùng giữa train và validation. Split
`train9k`, video-ID v1 và `test` không có trong giao diện cache hoặc training.
Overlap 341 URL giữa official train9k/test là giới hạn của protocol gốc, không
được giải quyết bằng việc dùng test để chọn head.

```powershell
.venv/Scripts/python.exe scripts/train_alignment.py --config configs/frozen_feature_alignment.json --readiness-only
.venv/Scripts/python.exe scripts/cache_msrvtt_features.py --help
.venv/Scripts/python.exe scripts/train_alignment.py --help
```

`outputs/experiments/pilot/alignment_and_profiling/alignment/readiness.json` ghi missing IDs từng split, cache presence,
`status=blocked`, `benchmark_training_status=not_run`, metric/checkpoint `null`
khi thiếu đầu vào. `inputs_present` chỉ xác nhận có file; integrity còn phải qua
các kiểm tra hash khi nạp cache. Không tạo `seed42/best.pt` bằng dữ liệu giả.

## Cache thật, có kiểm tra provenance

Chạy sau khi có media đúng tên `data/raw/msrvtt/<video_id>.mp4` và đủ ngân sách:

```powershell
.venv/Scripts/python.exe scripts/cache_msrvtt_features.py --config configs/frozen_feature_alignment.json --split train
.venv/Scripts/python.exe scripts/cache_msrvtt_features.py --config configs/frozen_feature_alignment.json --split validation
```

Encoder đọc từ `encoder_run_json=outputs/experiments/pilot/baseline/b0/baseline_run.json`. Đường dẫn
checkpoint cục bộ và SHA256 phải khớp run này trước khi nạp; script không tải
weights. Image transform, số frame, decode downsampling phải giống baseline
đã khóa. Tokenizer lấy đúng kiến trúc cùng encoder. Cache mới dựa vào caption
thô của metadata benchmark, không dựa vào 10 câu pilot.

Cache có `cache_manifest.json` và shard tối đa 500 video. Mỗi NPZ mở bằng
`allow_pickle=False` gồm:

| Trường | Kiểu và shape | Nội dung |
|---|---|---|
| video_ids | Unicode `[Nv]` | ID video trong split đã khóa |
| frame_embeddings | float32 `[Nv,8,D]` | Vector frame được L2 normalize |
| valid_frame_mask | bool `[Nv,8]` | Frame hợp lệ; hiện decoder strict không chèn padding |
| timestamps_s | float64 `[Nv,8]` | Timestamp thực, hữu hạn và tăng trên frame valid |
| text_ids | Unicode `[Nt]` | Mọi sentence row ID của các video trong shard |
| text_video_ids | Unicode `[Nt]` | Video sở hữu caption, mapping chính xác |
| text_embeddings | float32 `[Nt,D]` | Caption feature được L2 normalize |

Chuỗi caption trùng vẫn giữ đủ sentence IDs. Manifest khóa file hash từng
shard, source media hashes, encoder/checkpoint/preprocess/sampling hash,
source-group hash, raw caption hash và ordered query/candidate/positive hashes.
Train chỉ nhận manifest `status=ready` có cả hai split. Thay encoder,
preprocess, checkpoint, split/cohort hoặc sửa caption khiến cache bị từ chối.
Loader đọc một shard rồi mean-pool `frame L2 → masked mean → video L2`, bỏ
frame features trước shard tiếp theo và đặt caption về đúng thứ tự toàn pool.

Lưu ý tài nguyên: một shard 500 video × 8 × D=512 và 10.000 caption cần khoảng
28,7 MB dữ liệu float32 chưa nén; train caption features toàn split khoảng
331,8 MB. Đó chưa tính encoder, PyTorch, optimizer và bộ nhớ tạm. Trần RSS
`1.073.741.824` byte và thời gian 2 giờ được kiểm tra ở các bước; vượt trần phải
dừng hoặc hạ batch/review ngân sách. Preflight tính cả output feature arrays,
NPZ uncompressed sizes và các temporary float64 của mean pooling, rồi kiểm
RSS/thời gian ở mỗi shard. Không suy ra khả năng full training từ
probe nhỏ.

## B0/B1 và cohort chung

Giao diện cache hỗ trợ `--crop-mode global|union`. B1 cần `--boxes` theo schema
`person_object_union/v1` và provenance annotation/detector/evidence có hash,
không thay bằng crop toàn frame. `VerifiedUnionBoxes` kiểm timestamp, frame
index, source video hash, kích thước gốc và đủ mọi frame của cohort; không có
fallback khi thiếu hộp.

So subset phải tạo cohort lock cùng dùng cho B0/B1:

```json
{"splits":{"train":{"video_ids":["ID từ train đã khóa"]},"validation":{"video_ids":["ID từ validation đã khóa"]}}}
```

`--cohort-lock <path> --allow-subset` phải được truyền cho cả hai cache.
Config training tương ứng đặt `allow_subset=true`, `cohort_lock=<path>`,
`cache_dir` và `crop_mode`; B1 thêm `boxes={"train":<path>,"validation":<path>}`.
Manifest lưu cohort file hash.
Box manifests phải tách theo split, chứa đủ và chỉ frame của split/cohort đó;
không đưa cả train/validation vào manifest nạp cho một lượt cache split.
Adapter giữ upstream order và mọi caption của cohort; metric ghi `scope=subset`.
Không so subset với full benchmark hoặc so frozen B0 với learned B1.

## Loss và training

Logits = cosine/temperature. Với mỗi anchor hợp lệ,
`loss = logsumexp(candidate hợp lệ) − logsumexp(positive đã biết)`; loss cuối
là trung bình mean text→video và mean video→text. Nhãn `1=positive`,
`0=negative đã biết`, `-1=unknown`. Unknown không nằm trong mẫu số; caption
cùng video và mọi positive đã biết đều nằm trong tử số, không thành false
negative. Anchor thiếu positive hoặc negative bị loại và số lượng được báo;
cả hai chiều phải có anchor hữu ích.

Benchmark MSR-VTT dùng relevance hoàn chỉnh theo video identity trong lock;
không suy diễn semantic positives. API loss hỗ trợ thêm positive/unknown cho
giao thức có nhãn tường minh; validation của CLI hiện yêu cầu relevance
benchmark hoàn chỉnh. Batch lấy đủ caption của các video được chọn, thường
16 video/320 caption. Tail một video được gộp vào batch trước; ít hơn 2 video
bị từ chối. Gradient accumulation không được dùng để giả tăng negatives.

```powershell
.venv/Scripts/python.exe scripts/train_alignment.py --config configs/frozen_feature_alignment.json
.venv/Scripts/python.exe scripts/train_alignment.py --config configs/frozen_feature_alignment.json --resume outputs/experiments/pilot/alignment_and_profiling/alignment/seed42/last.pt
```

Chọn checkpoint bằng validation text→video R@1, cải thiện phải `>`; hòa điểm
giữ checkpoint sớm hơn. Kiểm patience sau mỗi epoch. Validation project/rank
theo chunk, giữ score trên CPU, không giữ GPU score hoặc mở test. Chunk giới hạn
working intermediates; projected vectors và score matrix CPU vẫn được lưu đủ.
Preflight tính peak dự kiến trước cấp phát; kiểm RSS và cùng deadline training
ở mỗi projection/scoring/ranking chunk và mỗi vòng loss. `val_scores.npz`
có axes `[text,video]` cùng exact IDs; `val_metrics.json` báo R@1/5/10, MedR,
MRR hai chiều, counts, pool hashes và stable candidate-order ties.

`train_log.csv` có epoch, train/val loss, val_t2v_r1, val_v2t_r1, lr,
train_seconds, selection_split, gradient norm và step count. `best.pt`/`last.pt`
lưu model/optimizer state, epoch, seed, config/config hash, encoder SHA256,
split/input provenance hashes, batch/torch RNG, history, patience counter và
initial weights. Nạp bằng `weights_only=True`; resume chỉ tiếp tục original
output với cùng config/cache/split, không khởi tạo lại optimizer. Unit test
kiểm resume cho ra chính xác weights/loss của chạy liền mạch.

## Probe kỹ thuật khi thiếu benchmark

```powershell
.venv/Scripts/python.exe -m pytest tests/test_alignment_training.py -q
.venv/Scripts/python.exe scripts/train_alignment.py --config configs/frozen_feature_alignment.json --synthetic-probe --probe-steps 3 --output-dir outputs/experiments/pilot/alignment_and_profiling/alignment/probe
```

Phải chỉ định `--synthetic-probe`; script không tự chuyển benchmark thiếu media
thành dữ liệu giả. Probe sinh và lưu 6 video tensor/12 caption mỗi split,
source groups riêng, dimensions 12, seed 42; không dùng 8 clip pilot. Chạy thật
3 optimizer steps rồi nạp checkpoint/optimizer để chạy thêm 1 step. Các đầu ra
đều nằm trong `alignment/probe`, được đánh dấu `synthetic_technical_probe`:
features, log, checkpoint, diagnostic validation scores và `probe.json`.
Khi có cache thật, `--probe-steps 3` chạy chính xác 3 optimizer steps trên train,
không phải 3 epoch, rồi kiểm nạp optimizer thêm một step; không chọn `best.pt`
hoặc đo benchmark validation trong chế độ `frozen_feature_code_probe` này.
Probe ghi loss hữu hạn, head gradient khác 0, norm cập nhật weights >0,
frozen-input grad None và optimizer step tăng khi resume.

Không có backbone thật được nạp trong probe; `encoder_grad=None` là ranh giới
feature được kiểm, không phải bằng chứng inference encoder trên MSR-VTT.
Score synthetic chỉ đo hoạt động code; `benchmark_training_run=false` và
`benchmark_metrics=null`. Tiêu chí train thực/benchmark của kế hoạch vẫn còn
blocked cho đến khi có media/cache và ngân sách đã đo.
