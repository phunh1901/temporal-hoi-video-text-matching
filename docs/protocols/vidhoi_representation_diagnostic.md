# Chẩn đoán biểu diễn M1 với ba vùng ảnh

Đường chạy này so **full_frame**, **object_only** và **union** trên cùng mẫu
VidHOI train đã khóa, cùng checkpoint original CLIP, mapping, danh sách cặp,
prompt và evaluator. Mục đích là xác định mô hình đang dựa vào ngữ cảnh toàn
ảnh, hình dáng vật thể hay vùng người–vật trước khi bổ sung geometry/temporal
hoặc chuyển thông tin tương tác sang retrieval M2.

Đây là **representation diagnostic train oracle**. Hộp và lớp vật thể đều từ
annotation, không phải output detector. Không train, chọn model cuối, đánh giá
zero-shot hoặc chấm benchmark VidHOI đầy đủ. Kết quả không tự chứng minh tăng
benchmark MSR-VTT hay giải quyết đúng chủ thể tương tác.

## Biến đổi và điều kiện so sánh

| Variant | Ảnh đưa vào encoder | Thông tin được giữ |
|---|---|---|
| `full_frame` | Toàn frame gốc | Ngữ cảnh, mọi người/vật trong ảnh |
| `object_only` | Hộp vật thể của cặp | Hình dáng, texture và một phần ngữ cảnh trong hộp vật |
| `union` | Vùng bao hộp người và hộp vật | Cả hai thực thể cùng ngữ cảnh giữa chúng |

Cả ba cách đều chấm cùng toàn bộ danh sách predicate cho mọi cặp person →
entity khác người đó. Không chỉ sinh cặp GT-positive. GT relation chỉ phục vụ
đánh giá; không dùng để chọn crop, loại cặp hoặc rút ngắn danh sách predicate.
Dùng cùng thứ tự frame/cặp/predicate, score transform, evaluator và quy tắc tie.
Không bỏ frame/cặp thiếu nhãn chỉ để cải thiện kết quả. Nhãn chưa biết không
được tự coi là negative chắc chắn.

Ba nhánh **không có chi phí encode ảnh bằng nhau**: full-frame có thể tái dùng
một embedding mỗi frame, object-only tái dùng theo object track trong frame,
union theo cặp có vùng union riêng. Receipt ghi số ảnh đầu vào, số lần gọi
encoder và thời gian thực. Đối chứng cùng model/dữ liệu không đồng nghĩa cùng
compute; không quy mọi lợi ích cho cơ chế HOI nếu chỉ tăng lượng ảnh được xem.

## Đọc đầu ra đúng mức

- `metrics.json`: frame-level HOI mAP theo evaluator hiện có, trên mẫu train
  nhỏ và hộp/lớp vật oracle. Không gọi là tube mAP hoặc số benchmark đầy đủ.
- `known_positive_ranks.json`: hạng của predicate **đã được gán positive** trong
  cùng cặp. Đây là chẩn đoán thứ hạng, không thay mAP. Các predicate chưa được
  gán nhãn có thể đúng; hạng thấp không tự chứng minh mô hình nhầm với negative.
- `comparison.json`: đối chiếu ba cách biểu diễn, gồm kết quả âm; không tự chọn
  variant thắng để làm model cuối.
- `role_symmetry.json`: kiểm độ bất biến/đồng điểm khi đổi vai trò/cặp theo cấu
  trúc biểu diễn. Full-frame bỏ mất ràng buộc cặp; object-only dùng chung crop
  cho nhiều người cùng hướng vào một vật nên không phân biệt người nào đang
  tương tác. Union của hai hộp cũng đối xứng về hình học khi đổi hai đầu cặp.
  Đây là giới hạn biểu diễn; đồng/khác score không tự chứng minh actor đúng/sai.

Object-only tốt trên mẫu nhỏ có thể phản ánh vật thể dự báo mạnh nhãn tương tác;
không đủ kết luận mô hình đã hiểu HOI. Union tốt hơn chỉ là tín hiệu để thiết
kế đối chứng phát triển hợp lệ; không cam kết temporal/geometry sẽ tăng điểm.

## Gói Kaggle và thực thi

Builder `scripts/prepare_vidhoi_diagnostics_kaggle.py` tái dùng bộ đóng gói và
bootstrap offline đã kiểm của static probe; chỉ bổ sung module diagnostic và
CLI tương ứng. Không gom toàn workspace. Không tải/upload/chạy GPU tự động.
Dùng manifest/frame/mapping/checkpoint/provenance đã có local và thư mục output
mới. Các cờ giống static builder:

```powershell
uv run --frozen --offline python scripts/prepare_vidhoi_diagnostics_kaggle.py `
  --probe-manifest <train_manifest.json> --mapping <mapping.json> `
  --clip-checkpoint <original_clip.safetensors> --clip-sha256 <sha256> `
  --checkpoint-provenance <model_provenance.json> --model-license <CLIP_LICENSE> `
  --data-provenance <data_receipt.json> --wheelhouse <offline_wheelhouse> `
  --output <new_package_folder>
```

Đầu ra gồm `vidhoi_m1_diagnostics_input.zip`,
`vidhoi_m1_diagnostics_kaggle.ipynb`, `HUONG_DAN.md`, `build_receipt.json`.
Giữ Dataset **VidHOI M1 Representation Diagnostics Input** và notebook
**VidHOI M1 Representation Diagnostics** riêng tư. Import notebook, Add Input
ZIP đúng phiên bản, đợi xử lý xong rồi Save Version → Save & Run All.
Notebook không hỏi tương tác, chạy tự động đủ ba nhánh qua sáu cell.

Gói giữ đủ dependency offline cho CPython 3.11/3.12/3.13 Linux x86_64. Pip cài
vào working với `--no-deps --no-index`; không thay Torch/torchvision/NumPy/Pillow
của Kaggle. Không tải checkpoint/video thay thế. ZIP đổi tên hoặc Dataset đã
giải nén vẫn được nhận bằng manifest/SHA256; symlink, traversal, source bị sửa
hoặc gói sai phiên bản bị từ chối. Input và dependency target mang prefix hash
riêng, tránh dùng code/các wheel từ lượt khác.

Mỗi lượt tạo `diagnostics_train_<timestamp>_<id>`. Console có log sớm và
heartbeat; lỗi thông thường vẫn xuất ZIP gồm logs và output đã có. Cell 6 chỉ
báo hoàn tất sau khi root receipt và cả ba variant khớp schema, chế độ, số
frame/cặp, không drop dữ liệu và toàn bộ output hashes. Receipt thiếu/sai không
được thay bằng trạng thái hoàn tất. Kernel OOM hoặc session bị ngắt có thể làm
mất phần log chưa ghi; không cam kết khôi phục những trường hợp này.

## Kiểm chứng local

Unit tests kiểm source allowlist/hashes, static compatibility, notebook sáu
cell, missing/wrong output và chống false completion. Test điều phối có encoder
fixture **được truyền tường minh**, ghi level riêng và model inference false;
production notebook không chấp nhận fixture để hoàn tất lượt inference thật.
Probe thật trên Kaggle vẫn cần artifact nhận về và kiểm độc lập.

```powershell
uv run --frozen --offline ruff check scripts/prepare_vidhoi_diagnostics_kaggle.py tests/test_vidhoi_diagnostics_kaggle.py
uv run --frozen --offline pytest tests/test_vidhoi_diagnostics_kaggle.py -q
```
