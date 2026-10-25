# Đối chứng head và audit trước khóa mô hình

Đây là công cụ phát triển trên split train/validation đã khóa. Chưa có kết quả
benchmark từ việc chuẩn bị cấu hình hoặc chạy probe. Giữ riêng đường tái lập
CLIP4Clip lịch sử 9k/1k và đường nghiên cứu nội bộ 8.100/900; weights đã học cả
train9k không cung cấp validation độc lập trên validation900.

## Linear và MLP

`configs/development_head_study.json` khai báo hai head, seeds 42/43/44,
fixture seed 42, metric chọn checkpoint `text_to_video.R@1` và phép bác bỏ.
CLI tạo sáu cấu hình từ `configs/frozen_feature_alignment.json`, giữ nguyên cache, split,
optimizer, lr, số epoch tối đa, patience và giới hạn tài nguyên. Mỗi cấu hình
chạy qua `scripts/train_alignment.py` khi cache thật đã sẵn sàng.

- Linear: hai phép chiếu không bias; ma trận vuông khởi tạo identity.
- MLP: hai nhánh độc lập D → 128 → D, GELU, bias ở hai lớp, không dropout;
  khởi tạo mặc định PyTorch dưới seed đã ghi. Đây là đối chứng dung lượng,
  không phải cơ chế mới hoặc bổ sung thông tin geometry/temporal.
- Cả hai detach feature, chuẩn hóa L2 đầu vào/đầu ra và giữ nguyên loss
  multi-positive hai chiều. Không đổi positive mapping hay biến unknown thành negative.
- Cùng số chiều đầu ra và cùng trần ngân sách, nhưng số tham số khác nhau:
  linear = 2D²; MLP = 2(2DH + H + D). Báo cả số tham số, thời gian và RAM thực đo.
  Early stopping có thể làm số epoch đã chạy khác nhau; phải công bố log thực tế.
- Checkpoint chứa config/head/seed/RNG/optimizer/input hashes; resume từ chối
  thay cấu hình hoặc cache. Kiểm thử đối chiếu resume với chạy liên tục cho cả hai head.

```powershell
.venv/Scripts/python.exe scripts/prepare_development_study.py --output outputs/development_review
.venv/Scripts/python.exe scripts/prepare_development_study.py --output outputs/development_probe --synthetic-probes
```

Output phải là thư mục mới. Lệnh thứ hai chạy tensor probe CPU, không encode video
hay tải dữ liệu. Sáu lượt dùng cùng 6 video train/6 video validation tổng hợp,
12 caption mỗi split, D=12. Mỗi lượt có 3 bước rồi resume thêm 1 bước. Fixture
seed khác optimization seed; SHA256 feature phải giống giữa các head/seed.
Linear identity với một full batch gần như không khác theo seed; std gần 0 ở
probe này không chứng minh độ ổn định benchmark. MLP có khởi tạo ngẫu nhiên.

`study_summary.json` giữ thông số probe riêng khỏi `benchmark_metrics=null`.
Mean/std của `head_update_norm` là kiểm tra kỹ thuật, không là R@K, mAP hoặc
bằng chứng head tốt hơn. `elapsed_seconds` chỉ là thời gian trong trainer, không
bao gồm import, chuẩn bị dữ liệu hoặc toàn bộ tiến trình. RSS là giá trị quan sát
tại các điểm kiểm, không bảo đảm bắt được đỉnh tuyệt đối. CPU không có số đo VRAM.

`summarize_seeds` chỉ gộp các run có cùng input/config (trừ seed/output), metric,
đơn vị và mức thực thi. Seed trùng, NaN, đổi đơn vị hoặc trộn probe/benchmark bị
từ chối. Std dùng mẫu số n−1; một seed ghi `one-seed exploratory`, std=null.
Đây không phải khoảng tin cậy theo video hay kiểm định ý nghĩa thống kê.

## Temporal × geometry

`ablation_matrix.json` là ma trận đăng ký trước C00/C10/C01/C11; mọi hàng giữ
union crop, head linear được học (kể cả C00), split và ngân sách giống nhau.
Trường `implementation_blockers` ghi rõ chưa có sequence trainer/geometry
encoder. Không coi probe global head là chạy ma trận này; không tạo feature hộp
bằng cách dùng full-frame làm oracle.

Trước triển khai các nhánh cần nguồn hộp/track, shared cohort, timestamp giây,
mask và geometry chuẩn hóa theo kích thước frame. Vận tốc phải chia delta-time,
không nối qua track gap hoặc delta-time ≤ 0. Đây là yêu cầu đầu vào còn chờ,
không là tính năng đã thực hiện. Chỉ mở cơ chế mới sau gate nghiên cứu.

## Audit zero-shot bằng usage manifest

`audit_zero_shot_usage` nhận `schema_version=zero_shot_usage_v1` với:

| Trường | Hợp đồng |
|---|---|
| seen_classes, pseudo_unseen_classes, final_unseen_classes | Ba danh sách ID lớp không rỗng, duy nhất, rời nhau; đây là vai trò phát triển, pseudo-unseen giữ riêng khỏi seen-train |
| samples | sample_id, video_id, split (`train`, `selection`, `final`); ID mẫu duy nhất, video không vượt split |
| Nhãn mỗi sample | positive_classes, negative_classes, unknown_classes, context_positive_classes: danh sách ID, kể cả danh sách rỗng; nhãn phải thuộc universe |
| uses | sample_id và purpose: supervised_training, cache_training, validation_selection, prompt_selection hoặc checkpoint_selection |

Positive/context/negative của final-unseen không được dùng trong phát triển;
pseudo-unseen không vào supervised training hoặc cache dùng cho training. Mẫu
final không được dùng chọn prompt/checkpoint. Unknown không được đồng thời là
positive/negative. Mẫu window phải bao phủ nhãn của cả ngữ cảnh, không chỉ frame giữa.

Có thể thêm `--zero-shot-manifest <file>` vào lệnh chuẩn bị. Audit thất bại lưu
vi phạm và trả exit code 2. Không có manifest thật thì chưa audit được dữ liệu
M1; kiểm thử manifest tổng hợp chỉ chứng minh các điều kiện từ chối. Kết quả
`manifest_consistent` chỉ nói nội dung khai báo nhất quán, chưa chứng minh manifest
bao phủ đủ annotation/cache/prompt/checkpoint. Cần đối chiếu file/hash nguồn độc lập,
train-only class support, nguồn supervision và giới hạn pretraining trước nghiệm thu.

## Điều kiện khóa model

Không chọn model bằng probe hoặc score final đã xem. Khóa cuối phải có checkpoint
thật, config, seed, split/pool/candidate/positive hashes, encoder/tokenizer/prompt,
preprocess và evaluator hashes cùng log lựa chọn trên validation hợp lệ. Khi
chưa có ứng viên đạt yêu cầu baseline, hồ sơ khóa ghi `blocked` và checkpoint=null.
Không tự chuyển một checkpoint bên thứ ba thành baseline được chấp nhận chỉ vì
đã chạy inference hoặc có metric. Hash tài liệu/cấu hình không thay hash weights.
