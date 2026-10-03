# W03-04 — Đối chiếu metric với evaluator CLIP4Clip

Ngày rà và thực hiện: **03/10/2026**. Mức công việc: **kiểm tra evaluator**.
Đầu ra máy: [metric_conformance.json](../../outputs/w03/metric_conformance.json).

## 1. Phần đã có và phần mới của tuần 3

W02 đã có R@K, MRR, MedR, truy hồi hai chiều, nhiều positive, loại query chưa
gán nhãn và giữ thứ tự candidate khi hòa điểm. W03 **không tính lại triển khai
này như việc mới**, không sửa `src/temporal_hoi/evaluation/metrics.py`.

Phần mới là khóa mã evaluator của tác giả, chạy cùng các đầu vào nhỏ có thứ
hạng tính tay, xác định điều kiện tương đương và ghi rõ trường hợp hai giao
thức cho kết quả khác nhau. Những số bên dưới là kết quả kiểm chứng định nghĩa
metric, không đo chất lượng mô hình và không dùng 8 đoạn pilot.

## 2. Nguồn chính thức và khóa phiên bản

- Repo tác giả: [ArrowLuo/CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).
- Commit: `508ffa3de39ba0563a03199c440ab602a72e9b6f`.
- Evaluator: [metrics.py tại commit đã khóa](https://github.com/ArrowLuo/CLIP4Clip/blob/508ffa3de39ba0563a03199c440ab602a72e9b6f/metrics.py).
- Điểm gọi evaluator: [main_task_retrieval.py](https://github.com/ArrowLuo/CLIP4Clip/blob/508ffa3de39ba0563a03199c440ab602a72e9b6f/main_task_retrieval.py), hàm `eval_epoch`.
- Mapping MSR-VTT: [dataloader_msrvtt_retrieval.py](https://github.com/ArrowLuo/CLIP4Clip/blob/508ffa3de39ba0563a03199c440ab602a72e9b6f/dataloaders/dataloader_msrvtt_retrieval.py), lớp `MSRVTT_DataLoader` lấy trực tiếp `video_id` và `sentence` mỗi dòng CSV.

Bản mã evaluator 3.029 byte được giữ nguyên tại
[clip4clip_metrics.py](../../outputs/w03/sources/clip4clip_metrics.py).
SHA256: `103e93090de14f55d1db61e40ecf1fbc814670975bd9b9ab1fe341d0c95f0a6f`.
Script từ chối chạy nếu byte mã nguồn không khớp hash; kiểm thử có ca xác minh
việc từ chối này. Mã nguồn tải chỉ là evaluator nhỏ, không chứa weights/video.

## 3. Định nghĩa và khác biệt cần khóa

| Hạng mục | CLIP4Clip chính thức | Bộ đo nội bộ W02 | Quyết định W03 |
|---|---|---|---|
| Thang đo Recall | R1/R5/R10 theo phần trăm | R@1/5/10 theo tỷ lệ 0–1 | Chia giá trị official cho 100 khi đối chiếu; báo thang đo rõ |
| Rank và cutoff | `compute_metrics` dùng rank 0-based, so `<K`, báo median `+1` | Rank 1-based, so `<=K` | Tương đương nếu cùng pool/mapping và không tie tại score positive |
| Positive, nhánh một caption | Positive ở đường chéo; caption/video cùng thứ tự CSV | Mapping positive được truyền tường minh | Adapter phải giữ đúng thứ tự ID, cùng pool; không lấy đường chéo từ ma trận bị đổi thứ tự |
| Hòa điểm, nhánh thường | Lấy mọi vị trí có score bằng score đường chéo; có thể tạo nhiều rank/query | Chọn một rank của positive đầu tiên sau stable sort theo thứ tự candidate | Giữ evaluator official để reproduction; ghi khác biệt, không đổi hợp đồng W02 âm thầm |
| MedR, nhánh thường | NumPy median: trung bình hai rank giữa nếu chẵn | NumPy median cùng quy ước | Tương đương |
| MedR, nhánh tensor | `torch.median`: chọn rank giữa thấp nếu chẵn | NumPy median | Ghi riêng nếu dùng nhánh multi-sentence |
| Hòa điểm, nhánh tensor | `torch.argsort` không yêu cầu `stable=True` | Stable sort theo candidate order | Không coi rank tensor khi tie là bảo đảm portable qua thiết bị/phiên bản |
| T→V, nhiều caption | Tensor `[nhóm video, caption, video candidate]`; mỗi caption có một video nguồn positive | Mỗi caption query với positive video nguồn tương ứng | Tương đương sau ánh xạ và loại padding, trừ các khác biệt median/tie đã nêu |
| V→T, nhiều caption | Max score các caption theo video nguồn, rồi xếp hạng **nhóm video** | Có thể xếp hạng tất cả caption, lấy positive đầu tiên | Hai pool khác nhau; chạy internal trên pool đã gom thì tương đương official, pool caption phẳng báo riêng |
| Padding / unknown | Tensor loại diagonal NaN/Inf là padding; nhánh thường giả định diagonal đều có nhãn | Từ chối score không hữu hạn; `None`/positive rỗng loại query và báo coverage | Adapter xử lý padding trước phép đo; benchmark đủ nhãn không dùng unknown để đổi mẫu số |
| MRR | Không có trong evaluator tác giả | Metric phụ có sẵn | Không gọi MRR nội bộ là metric reproduction chính thức |

MSR-VTT 1k với lớp `MSRVTT_DataLoader` mặc định dùng **nhánh thường**, một
sentence mỗi dòng CSV. Tensor multi-sentence dưới đây được kiểm toán để ngăn
áp sai định nghĩa cho thí nghiệm mở rộng; không mặc nhiên thay protocol 1k
đã khóa bằng toàn bộ caption của mỗi video.

## 4. Kết quả thực chạy trên fixtures

Chạy bằng Python môi trường `.venv` của dự án, CPU. Các phiên bản NumPy/Torch
được lưu trong JSON kết quả. Tất cả fixture đều tự tạo số hữu hạn nhỏ; không
dùng encoder, annotation chất lượng pilot hoặc dự đoán mô hình.

| Fixture | Kết quả và ý nghĩa |
|---|---|
| Singleton T→V không tie | Rank `[2,3,2]`, official và internal khớp R@1/5/10 và MedR |
| Singleton V→T không tie | Rank `[2,2,2]`, hai evaluator khớp |
| 12 candidates và các biên K | Có đủ rank 1–12: R@1=`1/12`, R@5=`5/12`, R@10=`10/12`, MedR=`6,5`; hai evaluator khớp |
| Hòa score positive | Hai query cho official rank `[1,2,2]`, R@1=`1/3`, MedR=`2`; internal `[1,2]`, R@1=`0,5`, MedR=`1,5` |
| Tensor, số query chẵn | Rank `[1,2]`: MedR official tensor=`1`, internal=`1,5`; R@1 đều `0,5` |
| Nhiều caption T→V | 8 caption query/4 video candidate; mapping cùng video nguồn, hai evaluator khớp |
| Nhiều caption V→T | Official pool 4 nhóm rank `[4,1,1,1]`, R@5=`1`; internal pool 8 caption rank `[7,1,1,1]`, R@5=`0,75`. Internal trên **cùng pool 4 nhóm** khớp official |
| Padding tensor | 4 vị trí gồm 1 padding; official và internal sau xử lý adapter cùng 3 query, rank nội bộ `[1,1,2,null]`, kết quả khớp |
| Query finite chưa biết nhãn | Official vẫn chấm 2 query, R@1=`0,5`; internal loại `None`, chấm 1 query, R@1=`1` |

Tổng: **9 fixture, 5 tương đương, 4 sai khác giao thức dự kiến, 0 sai khác ngoài
dự kiến**. Sai khác không phải cải thiện chất lượng; ví dụ R@5 giảm từ 1 xuống
0,75 vì đổi candidate pool, không phải vì mô hình kém đi.

Kiểm thử mới kết hợp bộ retrieval W02: **28 passed, 3,36 giây**. Trong đó 9
test mới dùng kỳ vọng tính tay để kiểm tra checksum, hai chiều, cutoff, tie,
median, pool nhiều caption, coverage/padding và JSON không có NaN. Tổng kết
kiểm tra cả tương đương trên cùng pool trong ca có sai khác pool dự kiến;
test giả lập lỗi xác nhận ca này vẫn làm tăng số sai khác ngoài dự kiến.

## 5. Tái chạy và giới hạn

```powershell
& '.\.venv\Scripts\python.exe' scripts/compare_benchmark_metrics.py
& '.\.venv\Scripts\python.exe' -m pytest tests/test_benchmark_metrics.py tests/test_retrieval.py -q
```

Đầu ra này hoàn thành đối chiếu metric W03-04, không xác nhận đã chạy full
MSR-VTT, không xác nhận chất lượng baseline, không tạo HOI mAP. Việc tải dữ
liệu/weights, adapter benchmark và chạy inference chính thức thuộc công việc
tiếp theo. Khi chạy benchmark, dùng evaluator commit đã khóa; báo thêm kết
quả nội bộ nếu cần và ghi rõ pool, mapping, cutoff, thang đo, tie và coverage.
