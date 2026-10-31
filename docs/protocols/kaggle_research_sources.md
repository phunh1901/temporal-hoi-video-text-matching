# Mã thực nghiệm MSR VTT trên Kaggle

Các module tại [scripts/kaggle/msrvtt](../../scripts/kaggle/msrvtt/) là bản mã
được đối chiếu với nguồn đi kèm các lượt thực nghiệm B0–B3. Mục đích là cho phép
review thuật toán, kiểm tra nguồn và chuẩn bị chạy lại. Không dùng việc có mã
hoặc một hash để thay cho bằng chứng chất lượng mô hình.

## Thành phần

| Luồng | Notebook công khai | Điểm vào thuật toán |
|---|---|---|
| B0 mean pooling, B1 query pooling, B2 trộn điểm | `pooling_b0_b1_b2.ipynb` | `workflow.py`, `retrieval_core.py` |
| Học head B3 trên train8100, encoder đóng băng | `train_b3.ipynb` | `fulltrain_runtime.py`, `b3_probe_runtime.py` |
| Đánh giá head đã khóa với B0–B2 từ cache test1k | `evaluate_b0_b1_b2_b3.ipynb` | `b3_eval_runtime.py` |

`data_pipeline.py` kiểm input, metadata và trích đặc trưng; `resources.py` đo tài
nguyên; `test_core.py` và `test_pipeline.py` là self-test tổng hợp của bộ mã này.
Các module Python gốc giữ nguyên byte. `prepare_sources.py` là phần bổ sung để
khôi phục bố cục chạy từ mã công khai, không thay thuật toán.
`.gitattributes` ngăn Git đổi xuống dòng trong các source snapshot có hash,
để giữ cùng SHA256 khi checkout trên Windows hoặc Unix.
Ba snapshot `retrieval_core.py`, `test_core.py` và `workflow.py` gốc dùng CRLF;
Git được khai báo nhận đúng CRLF và vẫn kiểm khoảng trắng dư, dòng trống cuối
file và dấu cách trước tab. Không chuẩn hóa lại byte các snapshot này.

## Nguồn và mức kiểm chứng

[provenance.json](../../scripts/kaggle/msrvtt/provenance.json) ghi SHA256 từng
file, tên/hash ZIP kết quả dùng để đối chiếu, và danh mục nguồn từng giai đoạn.
Nguồn B3 được lấy từ `runtime_sources.zip` bên trong ZIP kết quả training và
evaluation. Nguồn B0–B2 được kiểm với payload notebook; các module dùng chung
khớp bản có trong hai ZIP B3. ZIP kết quả B0–B2 không có một source archive riêng.
Việc file nằm trong source capture không khẳng định mọi hàm trong file đã chạy.

Ba notebook công khai là **bản thích nghi để chia sẻ**, chưa phải notebook có
output thực thi. Đã bỏ output, metadata phiên và payload nhúng. Phần bootstrap
được thay bằng khôi phục mã theo hash; cấu hình lấy từ preregistration của lượt
chạy, với đường dẫn input được để trống. Đặc biệt, B0–B2 dùng `RUN_TRAIN=False`
như lượt đã lưu; không lấy giá trị `True` trong notebook chuẩn bị làm bằng chứng
đã huấn luyện. Cấu hình đọc riêng tại [configs/kaggle](../../configs/kaggle/).

Không phân phối video, caption CSV/JSON, manifest chứa danh sách media, feature
cache, checkpoint, log hoặc bảng kết quả cá nhân. Bản gốc và hồ sơ thực nghiệm
giữ local ngoài Git. Các giá trị hash trong mã/cấu hình là danh tính đầu vào,
không phải nội dung dữ liệu và không tạo quyền phân phối dữ liệu.

## Chuẩn bị chạy lại

1. Đưa mã repository vào Kaggle và chỉnh `PUBLIC_CODE` trong cell bootstrap tới
   thư mục `scripts/kaggle/msrvtt`. Chọn môi trường phù hợp trước khi chạy.
2. Gắn các input có quyền sử dụng và điền đường dẫn trong cell cấu hình. Pooling
   cần media test, metadata, checkpoint encoder và baseline reference. Training
   cần train media/metadata cùng `msrvtt_train8100_v2.manifest.json` đúng hash.
   Evaluation cần cache test, feature/provenance manifest, head epoch5 đã khóa,
   training report, baseline reference và `evaluation_input_manifest.json`.
3. Bootstrap khôi phục source, kiểm hash rồi tạo source capture trong thư mục
   làm việc. Manifest dữ liệu thiếu hoặc sai hash làm dừng; không dựng lại bằng
   dữ liệu thay thế. Chọn thư mục kết quả mới; file khác nội dung không bị ghi đè.
4. Pooling/training dùng tokenizer được gắn riêng hoặc tải từ URL công khai đã
   khóa, kiểm size/hash. Evaluation từ cache không cần encoder/tokenizer.
   Các notebook này chưa được chạy lại trên GPU sau khi tách phần công khai.

Giữ nguyên seed, checkpoint, frame sampling, preprocessing, candidate order,
temperature, trọng số trộn và evaluator khi đối chiếu. Test1k đã được xem và
development900 nằm trong supervision train9k của encoder; không dùng chúng để
tiếp tục chọn cấu hình. Self-test tổng hợp không thay cho benchmark thật.

## Kiểm mã mà không chạy GPU

```powershell
uv run --frozen python scripts/kaggle/msrvtt/prepare_sources.py --verify-only
uv run --frozen python scripts/kaggle/msrvtt/prepare_sources.py --destination .cache/kaggle_msrvtt
uv run --frozen python -m unittest discover -s .cache/kaggle_msrvtt -p "test_core.py"
uv run --frozen pytest tests/test_kaggle_research_sources.py
```

`test_pipeline.py` cần source vendor đã khôi phục và tokenizer đúng hash trong
thư mục vendor của bản làm việc; không chạy trực tiếp trên thư mục snapshot
`.py.txt`. Các test này dùng encoder nhỏ khởi tạo và video tổng hợp, không chứng
minh chất lượng MSR-VTT. Trên Windows, dùng thư mục tạm ghi được nếu môi trường
hiện tại không cho phép xóa file trong thư mục tạm mặc định.

## Mã kế thừa

Các file tham chiếu trong `vendor/` được giữ dạng `.py.txt` để bảo toàn byte mã
upstream và phân biệt với mã dự án. Bootstrap khôi phục đúng tên `.py` khi chạy.
Giấy phép gốc đi kèm: `LICENSE.CLIP4Clip` và `LICENSE.OpenAI-CLIP`. Nguồn tham chiếu
là [CLIP4Clip tại commit khóa](https://github.com/ArrowLuo/CLIP4Clip/tree/508ffa3de39ba0563a03199c440ab602a72e9b6f);
tokenizer được khóa riêng trong `bpe_bootstrap.py`. Không áp giấy phép mã cho
dataset, checkpoint bên thứ ba hoặc video.
