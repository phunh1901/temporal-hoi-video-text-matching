# VidHOI adapter và evaluator ST-HOI

Evaluator nhận JSON kết quả ST-HOI và chấm HOI theo từng frame. Đối chiếu tương thích dùng notebook tác giả tại [commit `c906d7e`](https://github.com/coldmanck/VidHOI/blob/c906d7ece7c6db423742b35859e0126373422595/vidor_eval.ipynb). Việc chấm prediction có sẵn chỉ xác minh evaluator; script này không tải hoặc chạy model.

## 1. Hợp đồng dữ liệu

- `adapt_sthoi_result` trong `src/temporal_hoi/data/vidhoi.py` đọc GT và proposal box `[0,x1,y1,x2,y2]`, class/score `[0,value]`, chỉ số cặp subject/object và vector predicate. GT phải là nhãn binary; mọi score/box phải hữu hạn; chỉ số cặp và class phải hợp lệ. Subject GT là person ID 0.
- Result JSON của notebook có thể thiếu ID frame. Manifest bắt buộc xác định **đúng thứ tự hàng**, `video_id`, `frame_id`, `box_transform` và nguồn tọa độ. Không dùng số thứ tự hàng làm frame ID hoặc suy track từ index hộp. Nếu `orig_video_idx` hiện diện, CLI đối chiếu với `result_frame_key` từng hàng; nếu không, output ghi `caller_provided_ordered_manifest_only`.
- Upstream result boxes thuộc model frame đã resize. `BoxTransform` biểu diễn `model_x=original_x*scale_x+offset_x` và tương tự cho y; phép đảo áp dụng giống nhau cho GT và proposal. CLI yêu cầu transform explicit; identity transform chỉ đúng nếu đã xác minh result thật sự dùng pixel gốc. Không tự giả định kích thước short-side 224 hoặc hệ frame gốc.
- `timestamp_s` có thể chưa biết và giữ null; không suy từ tên frame khi thiếu FPS/index convention. API thư viện báo `xyxy_pixels_model_frame` nếu chưa có transform.
- `adapt_vidhoi_annotations` đọc schema annotation upstream gồm `video_folder/video_id`, `person_id/object_id`, `object_class/action_class`, `person_box/object_box`. Annotation frame key `frame_id` được join bằng `source_frame_id` của manifest, còn media index lấy từ manifest `frame_id`; tránh nhầm timestamp và frame index. Nhãn trùng cặp được gộp multi-label, track ID được giữ từ annotation, hộp/lớp cùng track phải nhất quán. Manifest phải bao phủ toàn input; subset phải được chọn explicit trước gọi.
- `validate_video_partitions` phát hiện video trùng train/research-validation/final. Split `official_val` của VidHOI là final evaluation công khai, không là validation chọn model.

Mapping numeric xác minh từ `idx_to_obj.pkl` / `idx_to_pred.pkl` tác giả: **78 objects, 50 predicates, person=0**. Pickle nhỏ được đọc bằng unpickler chỉ primitive, không cho phép global objects. File JSON local khóa có SHA-256 `a74c9c6c0e1efa116eb368935aded8b5a124c3d758376391924db8e2a58368d7`. CLI actual ST-HOI check yêu cầu đúng hash và số lớp; API thư viện hỗ trợ mapping nhỏ để kiểm thử. Hash numeric mapping không chứng minh đã có annotation hoặc tập HOI triplet hỗ trợ đầy đủ.

## 2. Quy tắc metric giữ nguyên upstream

`src/temporal_hoi/evaluation/hoi.py` triển khai:

1. Tập lớp macro chỉ gồm triplet có GT trong những frame đã chọn. Lớp có GT nhưng không prediction nhận AP=0; lớp không GT không tham gia macro.
2. Score triplet là `log(HOI)+log(subject_proposal)+log(object_proposal)`; value không dương thay `1e-300`, không sigmoid-threshold.
3. Chọn **top 100 triplets mỗi frame trước khi lọc lớp**. Ties trong frame theo tuple giảm dần `(score,pair_index,predicate_index)`. Sort score toàn lớp dùng `numpy.argsort` mặc định như notebook; output ghi version NumPy để đối chiếu ties.
4. Cùng subject/object class; IoU từng hộp ≥0,5; chọn cặp GT có min IoU lớn nhất. Match set dùng **chung cho các predicates trên cùng frame**, đúng code tác giả. Vì vậy cùng một GT pair multi-label có thể chỉ một predicate được TP. Không âm thầm sửa thành matching độc lập theo class khi gọi reproduction.
5. AP VOC 11-point, macro mAP thang **0–1**. Rare là GT count **<25**, nonrare **≥25**, đếm trên selected results; không đếm train frequency. Full/rare/nonrare là các run riêng.
6. Nếu subset không có lớp được GT hỗ trợ, trả `map=null` và `no_GT_supported_classes`; input frame rỗng, duplicate ID hoặc trộn box source/split/hệ tọa độ bị từ chối.

## 3. Mask, nguồn hộp và file CLI

Mask JSON bắt buộc có `protocol_id` không rỗng và `frame_ids` unique, nonempty. Mọi ID mask phải có trong input; giữ thứ tự result để không đổi tie order. `expected_frame_count`, nếu có, phải là integer ≥số ID mask. Output lưu hash file và hash ordered IDs, số input/selected/excluded frames, denominator coverage explicit. Coverage trên một mask chưa được audit không chứng minh full benchmark.

Oracle/predicted chạy riêng; `--box-source` phải trùng `metadata.box_source`. Không so ngang hai run khác mask/split/proposal source. Metadata còn cần `results_sha256`, `mapping_sha256`, `split`, `selection_configuration_fixed_before_scores` và ordered `frames`. Với `official_val`, fixed-selection phải khai explicit true trước score; output ghi đây là **assertion trong metadata, chưa được xác minh độc lập** và ghi lần truy cập final split. Không dùng score này chọn/tune model.

Ví dụ cấu trúc manifest (số minh họa, không là transform đã xác minh của dữ liệu thật):

```json
{
  "results_sha256": "<hash JSON predictions>",
  "mapping_sha256": "<hash canonical mapping>",
  "split": "official_val",
  "box_source": "oracle",
  "selection_configuration_fixed_before_scores": true,
  "frames": [{
    "video_id": "0001/video_id", "frame_id": 106,
    "result_frame_key": "0001/video_id_000106",
    "timestamp_s": null,
    "box_transform": {"scale_x": 1.0, "scale_y": 1.0, "offset_x": 0.0, "offset_y": 0.0},
    "gt_track_ids": {}, "proposal_track_ids": {}
  }]
}
```

CLI cho predictions được cung cấp, sau khi xác minh manifest/transform/mask:

```powershell
uv run python scripts/evaluate_vidhoi.py --results <results.json> --metadata <manifest.json> --mapping <class_mapping.json> --mask <mask.json> --box-source oracle --subset full --output <metrics.json>
```

Output luôn ghi `provided_prediction_evaluator_check`, `model_inference_run=false` và không tự cấp `benchmark_comparison_eligible`. Muốn xác nhận inference cần pipeline model riêng, provenance checkpoint, cấu hình và predictions mới. Muốn so benchmark cần audit đầy đủ annotation/split/mask/track/preprocess và regime trước đọc score.

## 4. Đối chiếu nguồn và điều kiện chạy

Lệnh đối chiếu metric bằng notebook đã kiểm hash:

```powershell
uv run python scripts/evaluate_vidhoi.py --verify-upstream-notebook <vidor_eval.ipynb> --output <conformance.json>
```

Notebook SHA-256 `211fd893d3e36159395e6dba3535ed0f39dff46dfd3c9705adb592c6498b2e0b` được kiểm trước khi thực thi đúng ba cell IoU/matching/AP. Chỉ bỏ magic, imports và print; matching/AP vẫn giữ thứ tự nguồn. Các fixtures có kỳ vọng tính tay, không dùng video/nhãn benchmark. Bộ kiểm thử clone chạy độc lập bằng fixtures, không cần notebook/Drive/dataset. Notebook chỉ cần cho lệnh đối chiếu tác giả.

Ngày rà 03/10/2026: 11/11 fixtures tương thích gồm perfect/wrong box, IoU=0,5, duplicate, shared multi-label pair, tuple tie, score floor, GT không prediction, top100 trước class filter, rare24/nonrare25. **Không có benchmark mAP hoặc inference checkpoint mới từ các fixtures này.**

[README tác giả](https://github.com/coldmanck/VidHOI/blob/c906d7ece7c6db423742b35859e0126373422595/README.md) vẫn ghi weights bị gỡ ngày 21/08/2022. Drive listing lần rà này xác minh oracle baseline prediction JSON có 290.387.853 bytes, train annotation 493.659.675 bytes; binary chưa tải/hash. Numeric mapping đã khóa; annotation/split IDs, triplet-support, transform/media frame join thực và mask pose 21.758 frames còn thiếu. Metadata folder đọc được không có nghĩa checkpoint/prediction đã tải hoặc quyền footage đã approved. Chi tiết máy và các mức `not_run/blocked` nằm trong artifact readiness local.

## 5. Khóa input local trước chấm benchmark

`scripts/lock_vidhoi_protocol.py` đọc một specification JSON có `protocol_id`,
`upstream_commit`, `expected_mask_frame_count` và dictionary `inputs`. Mỗi input có
`path`, SHA-256 dự kiến nếu đã xác minh, và mô tả `source`. Bảy input bắt buộc là
`mapping`, `train_annotations`, `val_annotations`, `train_frame_manifest`,
`val_frame_manifest`, `video_partitions` và `frame_mask`. Đường dẫn tương đối được
giải theo `--base-dir` (mặc định thư mục đang chạy). Không có thao tác tải mạng.

Manifest train/val là array các frame có `video_id`, `source_frame_id`, `frame_id`,
`result_frame_key`, `timestamp_s` nếu biết, `box_transform` và
`box_transform_provenance`. Transform **bắt buộc có đủ** `scale_x`, `scale_y`,
`offset_x`, `offset_y`; provenance phải mô tả nguồn xác minh preprocessing thực.
Không điền identity transform để vượt gate. `result_frame_key` là ID mà result
tác giả sử dụng; nó không mặc nhiên bằng media frame ID. Partition JSON có đúng
ba list video không rỗng `research_train`, `research_validation`, `official_val`;
hai list đầu bao phủ toàn bộ upstream train và list cuối bao phủ official val.

```powershell
uv run --frozen python scripts/lock_vidhoi_protocol.py --inputs <input_spec.json> --output <input_lock.json> --require-ready
```

Nếu thiếu file, CLI vẫn lưu danh sách file thiếu và hash file hiện có; với
`--require-ready` trả exit code 2. Khi có đủ file, khóa kiểm tra mỗi join frame và
track/box/class qua adapter; split video disjoint và bao phủ đúng annotation;
mask kiểm **ID thực** và số lượng kỳ vọng, không chỉ đếm 21.758. Annotation array
được đọc từng phần, các dòng cùng frame phải liên tiếp như loader upstream.
Output lưu hash file/ordered IDs, video IDs và triplet support theo số GT pair
có nhãn sau gộp multi-label. Hàng annotation trùng không đếm thêm positive.

`input_lock_ready=true` chỉ xác nhận tính nhất quán local. Nguồn xuất mask,
transform, tính đầy đủ upstream và quyền footage vẫn cần đối soát độc lập;
`benchmark_comparison_eligible` giữ false. Triplet support không tự tạo hoặc
chứng nhận split zero-shot. Công cụ này không chạy checkpoint hay chấm score,
không thay gate evaluator/prediction của mục 3. Phiên bản specification input
lock mới không sửa metric hoặc khóa lịch sử `vidhoi_w03_v1`.
