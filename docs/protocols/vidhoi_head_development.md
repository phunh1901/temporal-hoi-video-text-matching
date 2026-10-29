# M1: phát triển head nhẹ từ đặc trưng CLIP đóng băng

Đường chạy này dùng một cohort lớn hơn probe tĩnh, tách **train** và
**validation/development** theo protocol đã khóa, đều lấy từ **upstream train
VidHOI**. Hộp và lớp vật là oracle. Original CLIP chỉ tạo đặc trưng; ba head
residual nhỏ được học để đối chiếu object, union và fusion. Không dùng official
final-val hoặc final-unseen để chọn epoch/config, tạo ảnh hay cache học.

Đây là **internal oracle head development**. Kết quả development dùng lựa chọn
không phải ước lượng final độc lập. Không suy đã hoàn tất benchmark VidHOI,
zero-shot cuối, detector predicted, temporal hoặc cải thiện M2 từ tác vụ này.

## Dữ liệu và chống leakage

Hai manifest `vidhoi_head_frames_v1` có `upstream_split=train`, role `train`
hoặc `validation`, hộp/lớp vật oracle, SHA256 protocol/mapping và frame fields
như static probe: video ID, source/media frame ID, timestamp, ảnh/hash, kích
thước gốc, entities/hộp/track/lớp và ground truth positive predicates.

Train và development phải rời video theo protocol; nếu dữ liệu cho phép, audit
cả nguồn video/camera. Danh sách seen/pseudo-unseen/final composition candidates
và quy tắc lọc lấy từ train/protocol; không chọn bằng development hoặc test score.
Giữ sample gốc và log lọc. Nhãn thiếu không tự thành negative. Cả ảnh/ngữ cảnh
và text features liên quan final composition bị giữ lại phải tuân đúng phạm vi
supervision đã khóa, không lọt vào feature cache học.

Cohort chọn theo độ phủ annotation trước model score phục vụ development;
không coi là mẫu đại diện toàn benchmark. Các composition được giữ lại có
thiết kế và các composition tình cờ thiếu trong frame train cần báo riêng.
Audit tách AP của lớp đã có positive trong frame train, pseudo-unseen đã khóa,
composition mới với thành phần đã thấy và thành phần chưa có supervision.
Phân nhóm chỉ tổng hợp AP đã tính với cùng candidate pool/top-k; không chạy
lại evaluator sau khi lọc prediction theo nhóm để làm thay đổi bài toán.

Builder chỉ kiểm role/hash, đường dẫn, ảnh và video disjointness cần thiết;
không tự chứng nhận protocol nguồn hoặc quyền media bằng hash. Preflight/cache
phải audit schema, coverage và mục đích dùng nhãn trước model inference.

## Cơ chế và đối chứng

Encoder ảnh/text cùng original CLIP checkpoint đóng băng. Cache giữ tách object
crop và person–object union crop theo cùng frame/cặp. Fusion lấy trung bình
cố định 0,5 hai biểu diễn đã chuẩn hóa theo thiết kế được lưu, không tăng gấp đôi
chiều input một cách ngầm định. Ba head cùng dạng residual D → 128 → D và cùng
ngân sách; số tham số thực được ghi.

Loss theo protocol là **positive-only cosine alignment + feature anchor λ=1**.
Không dùng BCE với mọi nhãn chưa gán làm negative, không tự tạo hard negative
hoặc pseudo-label để tăng điểm. Mức loss/projection/normalization và những hàng
không có positive hữu ích phải được ghi rõ ở config/cache/trainer.

Ba mode `object`, `union`, `fusion` chạy đủ seeds 42/43/44, tối đa 20 epoch,
patience 5 theo config đã review. Frozen baselines và epoch 0 được giữ. Có thể
chọn epoch 0 nếu học không cải thiện development; đây vẫn là kết quả âm hợp lệ,
không được che bằng đổi seed/epoch/metric theo score cuối. Báo đủ mean/std,
không chọn seed tốt nhất rồi xem như kết quả đại diện.

Học thực phải có optimizer steps, gradient hữu hạn và thay đổi tham số/đặc
trưng ở **head cuối đã học**. Selected checkpoint có thể bằng mốc epoch 0;
không yêu cầu nó khác mốc chỉ để tuyên bố có học. Không gọi head training là
fine-tune toàn encoder.

## Cache, checkpoint và đầu ra

Cache numeric NPZ được đọc với `allow_pickle=False`, gắn role/IDs/thứ tự,
protocol/manifests/model/mapping/preprocess hashes, feature shape và masks nhãn.
Không sử dụng cache cùng tên nhưng khác provenance. Đầu vào lỗi phải dừng, không
bỏ frame/cặp hoặc thay ảnh bằng zero tensor để tiếp tục.

Head checkpoint dùng safetensors hoặc tensor state dict được đọc tường minh
`torch.load(weights_only=True)`; JSON riêng ghi config/seed/epoch/cache hashes.
Không dùng pickle fallback. Giữ cả initial, selected, last checkpoint, history,
train log và metric cho từng mode/seed.

Bộ chạy theo stage `preflight`, `cache`, `train`, `evaluate`; stage evaluate
kiểm lại checkpoint/predictions/metric và tạo receipt tổng, không mở final-val.
Replay khôi phục device và số luồng CPU đã ghi khi học trước khi tính score;
khác phép cộng dồn do số luồng có thể đổi thứ tự các score gần bằng nhau và AP.
Sai khác AP/TP/rank vẫn bị từ chối dù sai khác score nằm trong tolerance.
Đầu ra gồm cache, ba frozen baselines, 9 lượt head, aggregate mean/std, replay
kiểm chứng và file inventory. Khác biệt evaluator/tie-breaking giữ rõ theo
protocol; known-positive ranks là chẩn đoán, không thay frame-level HOI mAP.

## Gói Kaggle tự chứa

Builder `scripts/prepare_vidhoi_head_kaggle.py` nhận đủ input local và không
download/upload/khởi chạy GPU. Hai manifest và protocol được đóng cùng mapping,
frame train/development, original CLIP weights/provenance/license, source tối
thiểu và wheelhouse offline CPython 3.11/3.12/3.13 Linux x86_64. Không bao gồm
AGENTS/rules/rubric, kế hoạch, báo cáo, nhật ký tiến độ, secrets hoặc dataset
archive đầy đủ. Mọi output người dùng yêu cầu được giữ.

```powershell
uv run --frozen --offline python scripts/prepare_vidhoi_head_kaggle.py `
  --train-manifest <train_manifest.json> --dev-manifest <dev_manifest.json> `
  --protocol <protocol.json> --mapping <class_mapping.json> `
  --clip-checkpoint <original_clip.safetensors> --clip-sha256 <sha256> `
  --checkpoint-provenance <model_provenance.json> --study-config <study_config.json> `
  --model-license <CLIP_LICENSE> --data-provenance <data_receipt.json> `
  --wheelhouse <offline_wheelhouse> --output <new_package_folder>
```

Gói gồm `vidhoi_m1_head_input.zip`, `vidhoi_m1_head_kaggle.ipynb`, hướng dẫn và
build receipt. Dùng Dataset riêng tư **VidHOI M1 Head Development Input** và
notebook **VidHOI M1 Head Development**; Import → Add Input → Save Version /
Save & Run All. Internet Off; GPU nếu có. Notebook 7 cell tự chạy authenticated
setup, dependency/environment, preflight, cache, train, development evaluate,
export. Torch/torchvision/NumPy/Pillow Kaggle được giữ; dependency riêng dùng
`--no-deps --no-index --target`.

Input/dependency roots gắn manifest prefix; ZIP đổi tên hoặc Dataset giải nén
vẫn nhận theo hash. Path traversal/symlink/bytes bị sửa/gói sai phiên bản bị
chặn. Giữ giới hạn file/byte hiện có, đo mẫu nhỏ trước workload lớn; không tăng
CUDA/ngân sách tự động. Mỗi Run All tạo run riêng và log heartbeat. Lỗi thông
thường lưu ZIP partial; kernel OOM/session mất có thể làm mất log chưa ghi.

Export chỉ hoàn tất khi receipt stage, cache, đủ 3 mode × 3 seed, frozen
baselines, state/history và hash/cấu hình/cohort khớp. Không thay receipt thiếu
bằng cờ hoàn tất. Fixture API dùng level riêng và không được production
notebook nhận làm training/inference thật.

## Kiểm chứng

Test packaging kiểm allowlist, train/development role/hash, overlap, cache /
checkpoint / receipt không đầy đủ, source hash thay đổi, notebook stage order và
điều phối với fixture tường minh. Chạy fixture không phải kết quả benchmark.
Kết quả thật cần ZIP chạy Kaggle cùng logs/cache/head/checkpoint đã kiểm độc lập.

```powershell
uv run --frozen --offline ruff check scripts/prepare_vidhoi_head_kaggle.py tests/test_vidhoi_head_kaggle.py
uv run --frozen --offline pytest tests/test_vidhoi_head_kaggle.py -q
```
