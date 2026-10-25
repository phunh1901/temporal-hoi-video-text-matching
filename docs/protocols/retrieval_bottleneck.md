# Profiling retrieval và đối chứng global/union

Giao thức này đo chi phí của pipeline đã khóa và chuẩn bị đối chứng crop. Một
run trên pilot tiện lợi chỉ xác nhận đường chạy; không thay thế MSR-VTT hoặc
VidHOI benchmark, không chứng minh cơ chế HOI và không xác nhận cải thiện SOTA.

## Đo lại pipeline

Chạy từ repository bằng môi trường đã cài đặt:

```powershell
.venv/Scripts/python.exe scripts/profile_pipeline.py --run-json outputs/experiments/pilot/baseline/b0/baseline_run.json --warmup 1 --repeat 3 --batch-size 1 --output outputs/experiments/pilot/alignment_and_profiling/profile.csv
```

`--run-json` là bản chạy thành công với checkpoint **cục bộ** và SHA256, lock
manifest, cấu hình, frame sampling, transform và candidate order. Profiler
chạy lại video decoder và encoder, không chia tổng thời gian lịch sử thành các
ước lượng. Mỗi lần đo chạy trong một process mới; warmup làm ấm cache của hệ
điều hành, không giữ model trong bộ nhớ cho các lần đo sau. Mỗi repeat do đó
vẫn có chi phí load model. Import/checkpoint hashing và serialization được
tính trong end-to-end, nhưng nằm ngoài các stage thuật toán.

Batch ảnh/text bằng 1, 2 CPU threads theo run tham chiếu, fp32, encoder đóng
băng, seed 42. Chưa hỗ trợ profiler CUDA hoặc batch thay đổi vì điều đó làm
thay cấu hình đã khóa. Không tải thêm weights/video. Mỗi replay phải giữ
checkpoint, source/manifest hashes, ID/query pool, transform, pooling và
normalization; sai khác bị từ chối trước báo cáo tổng hợp.

| Cột | Ranh giới đo |
|---|---|
| `model_load_ms` | Tạo model và load checkpoint local; không gồm import Torch |
| `decode_ms` | Mở/seek/decode video, resize max-side và đổi RGB/PIL |
| `preprocess_ms` | Transform CLIP và tokenizer; không gồm decode resize |
| `image_encoder_ms`, `text_encoder_ms` | Forward encoder CPU đồng bộ, chuyển feature sang NumPy |
| `pooling_normalization_ms` | L2 từng frame/text, mean và L2 video |
| `head_ms` | 0 đối với frozen identity; không có learned head |
| `similarity_ms` | Phép nhân ma trận cosine; báo riêng để không giấu vào head |
| `evaluation_ms` | Retrieval hai chiều theo scope và pairwise trên nhãn đã khóa |
| `end_to_end_ms` | Toàn thời gian `run()`: cả import/hash/I/O và memory checks |
| `subprocess_wall_ms` | Thời gian caller quan sát, thêm khởi động interpreter/process |
| `peak_working_set_bytes` | Peak RAM resident theo bộ đếm process của Windows |
| `peak_commit_bytes` | Peak committed memory của process, khác resident RAM |
| `peak_cuda_allocated_bytes` | `null`/ô CSV trống khi chạy CPU; không giả số đo GPU |

Không cộng stage để thay end-to-end vì còn import/hash/I/O. Bản JSON từng
repeat lưu số calls và tỷ trọng trên tổng **stage đã đo**, không phải trên toàn
wall time. Bộ đếm peak process của OS bắt được peak resident/commit; các cực
tiểu memory hệ thống được sample mỗi 0,5 giây có thể bỏ qua dip ngắn hơn.

Trước load, giữ bound available commit `2 × bytes checkpoint + 512 MiB` đã
có trong runner; giữa clip giữ tối thiểu 256 MiB. Bound là chính sách an toàn
cục bộ, không là yêu cầu phần cứng đã công bố của model. Thiếu memory sinh
blocked record và metric/stage chưa đo để trống, không đổi bound để ép chạy.

Đầu ra `profile.csv` có mỗi repeat, `bottleneck_summary.json` có mean và sample
standard deviation; các JSON/embeddings/predictions từng run trong
`bottleneck/`. Ba lần lặp đo biến thiên chi phí của cấu hình hiện hành, không
là ba seed training hoặc bằng chứng độc lập chất lượng.

## Ca lỗi quan sát

`error_cases.csv` chứa các cột `module,split,video_id,text_id,error_kind,true_rank,
predicted_id,box_source,evidence_path` và thêm query ID, candidate group, score
gap. Split pilot được gọi `pilot_dev`/`pilot_extension`; không gọi là test.

Lỗi top-1 được lấy từ predictions thật khi candidate đầu là negative rõ ràng.
`true_rank` là rank của positive đầu tiên. Nhóm candidate dùng nhãn cảnh/hành
động của manifest; chỉ là phân nhóm quan sát, không tự kết luận nguyên nhân do
temporal order, geometry hoặc box localization. Báo cả số cặp positive/negative
và số positive không đứng trên negative theo score; tie không là pairwise win.
Query không có positive đã biết bị loại với số lượng rõ, không biến thành lỗi.

M1 chỉ phân tích missing/duplicate/wrong_object/IoU khi có predictions thật và
GT cùng frame mask. Fixture evaluator phải ghi `synthetic_evaluator_case`.
Pilot hiện thiếu nhãn tương tác và boxes GT nên HOI localization là `not_run`;
không điền count=0 hoặc suy lỗi HOI từ ca retrieval.

## Nguồn hộp và crop

`src/temporal_hoi/data/pair_crops.py` triển khai union và chọn pair xác định.
Tọa độ XYXY phải thuộc kích thước **frame gốc**, không phải tensor encoder.
Adapter detector phải inverse resize trước xuất. Box nguồn cần finite và có
positive area; clip vào biên frame rồi loại box không còn area. Union là min
hai góc trái/trên và max hai góc phải/dưới. Khi reader downsample, scale union
theo từng chiều, floor biên trái/trên và ceil biên phải/dưới để giữ vùng hộp.
Sau crop giữ đúng transform/preprocess encoder của B0.

Pair được chọn bằng `pair_score` giảm dần, rồi `person_track_id`,
`object_track_id`, class và tọa độ tăng dần khi hòa score. Không dùng caption,
relevance hay điểm retrieval để chọn box. Thiếu pair trả `missing_pair`;
không fallback full frame. Một **pair thật đã xác minh** có union chạm tất cả
biên vẫn được chấp nhận và đếm `full_frame_union_count`; ca đó không có local
contrast, không tạo thành oracle giả.

Manifest union có cấu trúc:

```json
{
  "schema_version": "person_object_union/v1",
  "coordinates": "original_frame_pixels_xyxy",
  "provenance": {
    "source_type": "detector",
    "artifact_path": "path/to/upstream_predictions",
    "artifact_sha256": "SHA256",
    "detector_name": "verified detector identity",
    "detector_checkpoint_path": "path/to/checkpoint",
    "detector_checkpoint_sha256": "SHA256",
    "detector_config_path": "path/to/detector_config",
    "detector_config_sha256": "SHA256",
    "verification": {
      "status": "verified",
      "verifier": "identified reviewer",
      "evidence_path": "path/to/review_evidence",
      "evidence_sha256": "SHA256"
    }
  },
  "records": []
}
```

`oracle` thay `detector` chỉ khi nguồn annotation và evidence review là thật;
không yêu cầu detector metadata ở nhánh oracle. Mỗi record cần `clip_id`
(hoặc video ID khi dùng adapter MSR-VTT), `frame_index`, `timestamp_s`,
`source_video_sha256`, `width,height`, `person_class=person`, `object_class`,
`person_box_xyxy`, `object_box_xyxy`. Runner đối chiếu video hash, timestamp,
kích thước và độ phủ toàn bộ frame đã sample. Thừa/thiếu record hoặc nguồn
file/hash không khớp là lỗi. Hash xác nhận artifact không đổi; nội dung box
và bằng chứng xác minh vẫn phải được kiểm tra theo nguồn, không thể được
chứng nhận chỉ bằng việc tự đặt `status=verified`.

`configs/hoi_union_crop.json` hiện giữ detector/weights/predictions ở `null` vì chưa
có nguồn runnable được xác minh. `extract_pair_boxes.py --config ...` ghi
readiness `blocked` và trả code 2; không tạo `boxes.jsonl` giả. Khi đủ artifact,
CLI có thể chuyển **dự đoán detector đã xác minh** thành pairs/manifest, nhưng
không thực hiện inference detector chưa chọn. Schema nguồn cho nhánh đó có
`coordinates`, `provenance`, `frames`; mỗi frame có video ID/frame index/time,
kích thước, video hash và danh sách `pairs` đã inverse transform. Đây là
adapter chuẩn bị, chưa phải triển khai một detector cụ thể.

## So sánh công bằng

B0/B1 phải dùng cùng encoder/checkpoint, head (frozen với frozen; learned với
learned cùng initialization/budget), frame indices, pooling, normalization,
data source, split, cohort và query pool. Chọn cohort theo quy tắc có đủ pair
trước nhìn score retrieval, loại cùng các video ở cả B0 và B1, báo ID bị loại
và coverage. Không gọi development subset là full benchmark.

`cohort_lock.json` cho feature cache dùng schema:

```json
{"splits": {"train": {"video_ids": []}, "validation": {"video_ids": []}}}
```

Cache global và union phải nhận cùng `--cohort-lock`/hash và `--allow-subset`
khi cohort nhỏ hơn split chính thức. `--crop-mode union --boxes ...` cần nguồn
boxes xác minh. Đầu ra train/validation không có media hoặc cohort/boxes đầy
đủ vẫn là `not_run`/blocked; không thay bằng 8 clip tiện lợi.

Pilot B1 chạy bằng runner với `--union-boxes` sau khi có manifest thật.
`compare_global_local.py --b0-run-json ... --b1-run-json ...` từ chối thay
encoder, head, environment, query pool, transform, sampling hoặc coverage và
kiểm tra text embeddings giữ nguyên. Delta chỉ được báo sau khi contract đạt.
Hiện chưa có B1 thật, nên `global_local/comparison.json` ghi blocker,
`delta=null`. `global_local.csv` dành riêng benchmark development; mọi hàng
chưa chạy giữ `status=not_run` và metrics trống. B0 pilot được liên kết riêng
trong JSON để không đưa điểm pilot vào hàng MSR-VTT.
