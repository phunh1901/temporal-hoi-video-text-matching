# Probe VidHOI tĩnh với CLIP và hộp chuẩn trên train

Đường chạy này kiểm tra inference HOI cơ bản trước khi nghiên cứu temporal,
detector predicted hoặc cầu nối M1 → M2. Một mẫu nhỏ lấy từ **upstream train**,
hộp và lớp vật thể đều là ground truth. Kết quả là **train oracle probe**;
không phải benchmark VidHOI đầy đủ, đánh giá zero-shot, tái lập ST-HOI/ACoLP,
hoặc bằng chứng cải thiện retrieval MSR-VTT. Giữ benchmark M1 và M2 riêng.

## Đầu vào và phạm vi

`scripts/run_vidhoi_static.py` nhận cấu hình `vidhoi_static_config_v1`. Cấu hình
ghi đường dẫn và SHA256 manifest, mapping, checkpoint; provenance checkpoint;
device, batch size, số frame/pair tối đa, threads, preview và logit scale.
Đường dẫn được giải tương đối với cấu hình. Checkpoint phải là state dict
tensor của **original OpenAI CLIP ViT-B/32**; ưu tiên safetensors. Checkpoint
TorchScript không được nạp bằng cơ chế pickle/JIT thiếu kiểm soát; nếu cần
chuyển định dạng, giữ nguồn, SHA256 hai bản và bằng chứng tensor khớp.

Manifest `vidhoi_static_oracle_probe_v1` bắt buộc:

- `split=train`, `purpose=train_probe`, `box_source=oracle`,
  `object_label_source=oracle`, `mapping_sha256`.
- Mỗi frame có `video_id`, `source_frame_id`, `frame_id`, `timestamp_s`,
  `image_path` tương đối, `image_sha256`, kích thước frame gốc.
- `entities`: track ID, numeric class ID và hộp `bbox_xyxy` pixel gốc.
- `ground_truth`: subject/object track ID và danh sách predicate ID;
  không dùng GT relation để chỉ sinh những cặp vốn đã biết là positive.

Frame/media lấy từ train, có hash và provenance. Không đưa official-val vào
probe để chọn prompt, config hoặc threshold. Numeric mapping thật phải khớp
78 objects/50 predicates đã khóa; fixture nhỏ chỉ có trong API test tường minh.
Mỗi score là độ tương đồng/logit, không phải xác suất hành vi hoặc vi phạm.
Oracle boxes/object labels phải được nhắc cạnh metric.

## Chuẩn bị gói Kaggle local

`scripts/prepare_vidhoi_kaggle.py` nhận dữ liệu và trọng số **đã có local**,
không tải mạng, upload, tạo Dataset hoặc khởi chạy GPU. Dùng thư mục output
mới để giữ bằng chứng cũ:

```powershell
uv run --frozen --offline python scripts/prepare_vidhoi_kaggle.py `
  --probe-manifest <train_probe/manifest.json> `
  --mapping <class_mapping.json> `
  --clip-checkpoint <ViT-B-32.safetensors> --clip-sha256 <sha256> `
  --checkpoint-provenance <provenance.json> `
  --model-license <CLIP_LICENSE.txt> `
  --data-provenance <data_source_receipt.json> `
  --wheelhouse <wheelhouse> --output <new_output_folder>
```

Gói ZIP chỉ có source inference được allowlist, các frame/nhãn train được
chọn, mapping, checkpoint, provenance/giấy phép được chỉ định và dependency
wheels tùy chọn. Không gom cả workspace. Không bao gồm AGENTS/rules/rubric,
kế hoạch, báo cáo, nhật ký tiến độ, secrets hoặc dataset archive đầy đủ.
Các namespace `data/__init__.py` và `evaluation/__init__.py` trong gói được tạo
tối thiểu để tránh import module không liên quan; manifest ghi rõ hai file này.
Source còn lại giữ hash byte của file đã kiểm, không sửa repository initializers.

Wheelhouse offline chứa `requirements-offline.txt` với mỗi dòng `name==version`
và các wheel phù hợp Python/Linux Kaggle. Pin `open-clip-torch==3.3.0`;
bao gồm đủ dependency của nó và safetensors. Không đưa wheel Torch,
torchvision, NumPy hoặc Pillow vào gói. Notebook cài dependency bằng pip
`--no-deps --no-index --find-links ... -r ... --target ...` vào thư mục riêng,
để giữ Torch/CUDA của Kaggle. Pip tự chọn wheel hợp phiên bản Python; không
cài mọi file wheel cùng lúc. Nếu thiếu dependency và không có wheelhouse,
notebook dừng; Internet chỉ dùng cài dependency khi người chạy bật rõ cell 2.
Không có cơ chế tải thay checkpoint hoặc frame.

Wheelhouse cần chứa wheel native cho từng phiên bản Python dự định chạy;
wheel `cp312-cp312` không dùng cho Python 3.13. Giữ phiên bản dependency đã khóa,
bổ sung wheel phù hợp rồi dựng lại cặp ZIP/notebook với manifest mới. Kiểm pip
offline theo platform/ABI đích chỉ xác nhận chọn được wheel, chưa xác nhận import
hoặc chạy GPU trên Kaggle.

Đầu ra builder:

| Tệp | Vai trò |
|---|---|
| `vidhoi_oracle_train_input.zip` | Frame/nhãn/source/weights đã khóa manifest |
| `vidhoi_oracle_train_kaggle.ipynb` | Sáu cell: input, dependency, môi trường, preflight, inference, export |
| `HUONG_DAN.md` | Import Notebook → Add Input → Run All → tải kết quả |
| `build_receipt.json` | Hash/số byte/file inventory; ghi chưa inference/upload |

Notebook nhận ZIP đã đổi tên hoặc Dataset đã giải nén bằng SHA256 manifest.
Chỉ import notebook chưa gắn data: trong editor dùng **Add Input**, chọn Dataset
riêng tư chứa ZIP cùng phiên bản, đợi xử lý hoàn tất rồi chạy cell 1. Log
`Visible files: []` cho biết chưa thấy file input đọc được; log quét riêng ghi
đường dẫn, symlink bị bỏ qua và lỗi đọc. Không sửa hash để nhận gói khác phiên bản.
Mỗi member phải có đường dẫn POSIX tương đối chuẩn và hash/size đúng. Từ chối
traversal, symlink, encrypted/duplicate/extra ZIP entries và source bị sửa.
Không import source Dataset trước kiểm hash. Hash chỉ xác nhận byte, không
chứng minh quyền media/weights hoặc tính đầy đủ annotation.

## Chạy và giữ bằng chứng

Trên Kaggle, giữ Dataset/notebook riêng tư và gắn input ZIP; chỉ người dùng
thực hiện upload trong phạm vi quyền đã có. Notebook không tự publish gì.
Device mặc định `auto`: dùng GPU nếu đã bật, nếu không dùng CPU và ghi rõ.
Đặt `cuda` nếu muốn thiếu GPU phải dừng. Không thay đổi driver/Torch/CUDA.

Mỗi Run All tạo thư mục kết quả mới. Console có log ngay đầu mỗi cell và
heartbeat khi subprocess im lặng. Snapshot ZIP được giữ cả khi có lỗi thông
thường. OOM toàn process/kernel mất hoặc Kaggle ngắt session có thể cắt log;
không cam kết tự khôi phục những trường hợp đó.

Giữ config, input receipt, môi trường, logs, predictions/prompts/metrics/
preview do runner xuất và ZIP kết quả. Chạy preflight chỉ kiểm input, không
chứng minh inference. Có metric oracle trên vài frame train cũng không đủ
để chọn cấu hình từ final-val hoặc kết luận M1 đã hoàn tất.

## Kiểm thử

Kiểm builder/runtime bằng fixture nhỏ, input bị sửa và path traversal; xác
minh notebook có sáu cell hợp cú pháp và không nhúng tài liệu riêng. Test
điều phối notebook dùng encoder fixture tường minh, không báo benchmark.
Probe model/checkpoint thật cần lượt chạy riêng với dữ liệu và provenance thật.

```powershell
uv run --frozen --offline ruff check scripts/prepare_vidhoi_kaggle.py src/temporal_hoi/inference/hoi_kaggle.py tests/test_vidhoi_kaggle.py
uv run --frozen --offline pytest tests/test_vidhoi_kaggle.py -q
```
