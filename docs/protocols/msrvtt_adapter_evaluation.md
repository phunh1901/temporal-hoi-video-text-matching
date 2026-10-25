# MSR-VTT: adapter metadata, candidate pool và đánh giá artifact

`src/temporal_hoi/data/msrvtt.py` đọc metadata zip CLIP4Clip và lock `msrvtt_w03_v2`.
Adapter không tải video/weights, không tạo caption mới và không chọn model từ test.
Nó kiểm tra SHA256 toàn zip, từng member, thứ tự ID, mapping caption và query/positive.
Nguồn metadata và các hash tham chiếu được mô tả trong [giao thức khóa](msrvtt_protocol.md).

## Các split và nhãn

| Tên API | Nguồn ID | Caption/query | Vai trò |
|---|---|---|---|
| `train` | 8.100 ID source-group v2 | 162.000 dòng gốc | Học head/đối chứng |
| `validation` | 900 ID source-group v2 | 18.000 dòng gốc | Chọn model từ held-out train |
| `train9k` | 9.000 ID train upstream | 180.000 dòng gốc | Ghi điều kiện supervision tham chiếu |
| `test` | 1.000 ID JSFusion CSV | 1.000 câu CSV | Đánh giá tham chiếu cố định/final |
| `test_all_captions` | Cùng 1.000 ID | 20.000 dòng JSON | Extension; bảng riêng |

Caption có ID `msrvtt_sen_<sen_id>` và `annotation_row`; chuỗi trùng vẫn là dòng riêng.
Test dùng ID `msrvtt_jsfusion_0000…0999` và giữ nguyên câu CSV, kể cả 35 câu không
khớp nguyên văn JSON annotation. Positive dựa trên `video_id`, không dựa trên chuỗi
caption hoặc URL nguồn. Train/validation v2 phải không chung ID/URL; 341 URL chung
official train/test là giới hạn kế thừa được giữ và báo riêng. Độc lập exact URL
không chứng minh độc lập toàn bộ nội dung/pretraining.

`RetrievalPool.positive_indices()` trả mapping cho cả hai chiều. Ma trận có hàng
text, cột video. Với test chuẩn, diagonal là positive vì hai trục cùng thứ tự CSV.
Train/validation và extension có nhiều text positive/video. Subset chỉ chọn ID có
trong split, giữ thứ tự gốc và giữ mọi query có positive trong subset. R@K của subset
không so trực tiếp với full pool.

## Audit không inference

Chạy từ gốc repo với môi trường đã cài package:

```powershell
.venv/Scripts/python.exe scripts/evaluate_msrvtt.py
```

Mặc định đọc `data/metadata/msrvtt_data.zip` và
`data/metadata/msrvtt_protocol_lock.json`; media root `data/raw/msrvtt`.
Có thể chỉ định `--archive`, `--lock`, `--media-root`, `--media-suffix`, `--output`.
Audit lưu số ID/caption, hash pool, độ phủ file, hash source và môi trường.
Không có score đầu vào thì `benchmark_metrics=null`, `benchmark_inference_run=false`.
File rỗng bị coi là missing; file hiện diện chỉ là **file presence**, chưa là decode
coverage. Video phải có đúng tên `<video_id>.mp4`; adapter không tự đoán tên tương đương.

JSON `m2_internvideo_jsfusion_test.json` xuất cạnh audit: 1.000 đối tượng có
`video: "video<ID>.mp4"` và `caption` nguyên văn. Adapter kiểm tra round-trip ID,
caption, positive mapping và hash file. Schema này khớp `load_anno(media_type='video')`
và nhánh `VidTxtRetEvalDataset.build_data_multi_txt_gt` của
[loader InternVideo2 tại commit khóa](https://github.com/OpenGVLab/InternVideo/blob/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo2/multi_modality/dataset/ret_dataset.py).
[Utility upstream](https://github.com/OpenGVLab/InternVideo/blob/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo2/multi_modality/dataset/utils.py)
đọc trường `video` rồi dùng `pre_text` để lowercase/bỏ dấu câu. JSON giữ query raw;
inference phải ghi hash cả raw và text thực sau preprocessing. Không khẳng định
byte identity với annotation VINDLU gốc hoặc tương đương score của model chỉ từ schema.

Nguồn loader được tìm trong `--internvideo-source-dir` (mặc định
`outputs/experiments/pilot/baseline/m2_sources`), độc lập với thư mục `--output`. Khi ghi lần rà soát
vào thư mục mới, provenance vẫn giữ đường dẫn và hash của snapshot đã khóa.

## Đánh giá score/embedding có provenance

```powershell
.venv/Scripts/python.exe scripts/evaluate_msrvtt.py --scores path/to/scores.npz --provenance path/to/provenance.json --split test --output outputs/evaluation/msrvtt.json
```

NPZ dùng `allow_pickle=False`, có hai mảng Unicode `text_ids`, `video_ids` đúng thứ
tự pool và một trong các dạng sau:

- `scores`: ma trận hữu hạn `[text, video]` đã chấm bởi model.
- `text_embeddings`, `video_embeddings`: hai mảng `[N,D]` cùng chiều vector;
  evaluator L2 normalize rồi cosine nhân `--score-scale` dương (mặc định 1).
- `text_embeddings`, `frame_embeddings [video,frame,D]`, `valid_frame_mask [video,frame]`:
  L2 từng frame, masked mean frame valid, L2 video/text rồi cosine scale. Đây là
  hợp đồng meanP trên feature đầu vào; frame sampling/encoder vẫn phải xác minh riêng.

ID/shape sai bị từ chối, không âm thầm reorder. Zero vector hoặc video không có
frame valid bị từ chối; padding mask không được trở thành query video. Muốn chấm
subset phải truyền `--allow-subset`, lưu coverage và hash subset. Khi toàn bộ media
missing, không có subset rỗng hợp lệ để tạo metric.

Provenance JSON yêu cầu các trường không rỗng: `model_name`, `checkpoint_sha256`
(SHA256 lowercase 64 ký tự), `preprocess`, `pooling`, `score_type`, `training_regime`,
`artifact_kind`. `--checkpoint` cho phép kiểm tra hash file weights cục bộ có khớp
provenance hay không. Hash được khai báo mà không có file kiểm tra sẽ được ghi là
`local_checkpoint_hash_verified=false`. Script chỉ đánh giá artifact ngoài;
không tự xác nhận đã inference/training hoặc tương thích kiến trúc từ tên/hash.

R@1/5/10 và MedR hai chiều là chính; MRR là phụ. API nội bộ dùng fraction, tie theo
candidate order. Test singleton còn chấm evaluator CLIP4Clip nguyên bản sau kiểm
tra hash source; xuất bảng riêng có thang fraction và quy tắc tie upstream. Cần
snapshot `src/temporal_hoi/evaluation/reference/clip4clip_metrics.py.txt` với SHA256
`103e93090de14f55d1db61e40ecf1fbc814670975bd9b9ab1fe341d0c95f0a6f`.
Train/validation/extension chấm multi-positive nội bộ và không gọi nhánh diagonal
singleton của upstream. Các lần đọc/chấm official test được ghi mục đích trong
artifact; không sử dụng điểm test để chọn model.

## Kiểm tra tái lập

```powershell
.venv/Scripts/python.exe -m pytest tests/test_msrvtt_adapter.py -q
```

Fixtures nhỏ kiểm tra tamper/hash, split nguồn, row trùng, mismatch CSV, subset,
axis order, điểm tính tay hai chiều và meanP mask/per-frame normalization. Điểm
fixture là phép kiểm tra evaluator, không là chất lượng retrieval benchmark.
