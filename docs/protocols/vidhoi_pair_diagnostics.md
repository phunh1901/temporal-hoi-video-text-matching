# M1: chẩn đoán checkpoint theo cặp và kiểm nhãn người tương tác

Đây là bước phát triển nội bộ sau một lượt học head đã hoàn tất. Đầu vào là
ZIP kết quả có cache object/union, prototype text và initial/last checkpoint,
cùng protocol/mapping và một bộ review được chọn từ train. Không chạy encoder,
optimizer hoặc mở final; không thay nhãn hay evaluator benchmark.

## Chẩn đoán số học

`load_pair_diagnostic_source` kiểm ZIP CRC, đường dẫn, size, inventory/hash,
schema, role, ID, proposal order, positive masks và safetensors. NPZ luôn dùng
`allow_pickle=False`; không giải nén hoặc thực thi payload trong ZIP kết quả.

Chấm đủ object/union/fusion × seeds42/43/44 × initial/last: 18 trạng thái.
Khôi phục hai luồng CPU và device đã chọn trước khi tính score. Giữ nguyên
frame-level HOI mAP, global top100, tuple tie order, IoU và shared matching.
Trace giải thích canonical GT slot nằm trong/ngoài budget và prediction bị
chặn bởi matching; đối chiếu tổng TP với evaluator. Slot theo track ID không
tự là evaluator recall, vì những hộp chồng lấn có thể ghép khác identity.

Giữ riêng metric gốc của lượt nguồn và metric chấm lại. Initial/last được so
trong cùng môi trường hiện tại; khác thiết bị, Torch hoặc NumPy có thể đổi
near-tie/AP. Không ghi kết quả xấp xỉ là tái lập bit-exact hoặc sửa score gốc.

## Bộ tình huống review

Chọn theo metadata, không theo model score: cùng target và predicate, nhiều
người, có actor mang source positive và actor còn unknown. Giới hạn predicate
tiếp xúc/thao tác được khai báo trong code; đa dạng video bằng thứ tự hash
seed42, ưu tiên một case/video rồi bổ sung bằng hash. Giữ role train, loại dev
và final. Source annotation chỉ tạo candidate/badge, không suy actor khác sai.

Manifest `vidhoi_pair_review_v1` khóa source result, protocol/mapping, selection,
frame/track/box, cache row và từng ảnh/hash. PNG trung tâm giữ nguyên; ngữ cảnh
là mẫu khoảng4Hz trong tối đa±2s, với PTS giải mã thực. Ảnh ngữ cảnh có thể thu
nhỏ để xem, box hiển thị có scale/clipping được ghi riêng; không đổi input model.
Thiếu frame ở biên video hoặc thiếu ngữ cảnh/âm thanh phải được công bố.

`actor_review.html` hoạt động ngoại tuyến, giữ track màu ổn định và cho xem
chuỗi ngữ cảnh. Không hiển thị model scores; có thể hiển thị badge annotation
nguồn, vì vậy không gọi đây là review mù nhãn nguồn. Mọi nhãn người kiểm tra
khởi tạo `unknown`, kể cả actor có source positive.

## Nhãn do người kiểm tra cung cấp

Nhãn có schema `vidhoi_pair_annotations_v1`, fingerprint canonical JSON của
manifest, mã người kiểm tra, thời gian UTC, các khẳng định tự xem/không dùng AI
làm ground truth/mù model score và đủ actor của từng case. Trạng thái cho phép:
`positive`, `confirmed_negative`, `unknown`. Negative cần chỉ số ảnh ngữ cảnh
và lý do; nhiều actor có thể cùng positive. “Hold” không tự phủ định “play”.

Đây là provenance do người kiểm tra khẳng định, không phải hệ thống đã xác minh
độc lập danh tính hoặc toàn bộ thao tác xem. Không tự tạo negative cho trường
chưa đánh dấu, không dùng review để thay annotation benchmark hoặc train mới.

Notebook tự nhận duy nhất `vidhoi_pair_review_annotations.json` trong input bổ
sung, kiểm fingerprint/schema/evidence và lưu nguyên byte gốc. Nhiều phiên bản
cùng tên làm dừng thay vì chọn tùy ý. CLI cũng nhận `--annotations` cùng SHA256.

Chỉ tính margin/win/tie giữa positive và negative được người kiểm tra khẳng
định trong cùng case/predicate. Unknown không vào denominator như negative.
Báo exact tie và near tie (`abs(margin)<=1e-6`) riêng, không chỉnh ngưỡng theo
score. Không có cặp đối chứng thì status pending và metric null, không trả0
hay báo đã huấn luyện. Những phép đo này là diagnostic, không thay HOI mAP.

## Chạy và bàn giao

```powershell
uv run --frozen python scripts/run_vidhoi_pair_diagnostics.py --help
uv run --frozen python scripts/prepare_vidhoi_pair_diagnostics_kaggle.py --help
```

Config `vidhoi_pair_diagnostics_config_v1` có năm đường dẫn/hash: source_result,
protocol, mapping, review_manifest, annotations. Bộ chạy chia preflight,
diagnose, review; output có diagnostics/, review/ và receipt tổng. Export kiểm
đủ18 trạng thái, source, labels, các aggregate và inventory; trạng thái review
còn chờ nhãn là kết thúc chương trình hợp lệ, không là hoàn tất kiểm nhãn.

Builder chỉ nhận input local cụ thể, đóng gói source/cache/head/ảnh và wheels
offline Linux CPython3.11/12/13. Không cần original CLIP weights. Dùng Dataset
Private “VidHOI M1 Pair Diagnostics Input v2” và notebook “VidHOI M1 Pair Diagnostics v2”.
Add Input rồi Save Version → Save & Run All, Internet Off; T4 như lượt nguồn
giúp giảm khác biệt phần cứng, CPU cũng chạy được với chính sách báo sai khác.

ZIP kết quả nguồn được vận chuyển nguyên byte dưới tên `assets/source_result.bin`
để tránh xử lý ZIP lồng theo phần mở rộng khi Dataset được giải nén. Manifest
khóa `source_result_member` và SHA256 nguồn; loader đọc ZIP theo nội dung.
Bootstrap vẫn nhận manifest cũ dùng `.zip`, nhưng chỉ cho đúng một tên nguồn
được khai báo. Không dựng lại ZIP từ thư mục vì phải giữ chính xác hash gốc.
Khi đổi bản đóng gói, cập nhật cả notebook và input có manifest ghim tương ứng.
Với thư mục Dataset đã mở ZIP, bootstrap kiểm đủ size/path trước khi bắt đầu
chép ảnh; nội dung từng tệp vẫn được kiểm SHA256 khi chép.

Cell export kiểm lại manifest ghim và các tệp cài trước khi thêm `ROOT/src`
vào `sys.path` của kernel. `PYTHONPATH` dành cho subprocess chấm điểm không
tự làm source import được trong kernel. Sửa điều phối notebook này giữ nguyên
input v2/manifest/cache/checkpoint; không cần đổi Dataset nếu byte nguồn không đổi.

Không gói luật, nhật ký, secrets hoặc data riêng khác; không tự upload/commit.
Kết quả mới vẫn là cached oracle development, chưa là predicted HOI, full
benchmark, huấn luyện head mới hoặc bằng chứng cải thiện retrieval M2.
