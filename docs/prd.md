# PRD — Temporal HOI Video–Text Matching

**Người thực hiện:** Ngô Hoàng Phú
**Phiên bản phạm vi:** 30/09/2026, đối soát theo yêu cầu hai module và hai giai đoạn nghiên cứu.
**Vai trò:** Đặc tả yêu cầu và giao thức mục tiêu. [README](../README.md) mô tả khả năng hiện có; [Plan](../plan.md) phân bổ việc và bằng chứng. Yêu cầu trong PRD không có nghĩa đã triển khai.

## 1. Mục tiêu và câu hỏi nghiên cứu

**M1 — Zero-shot / Open-vocabulary Human-Object Interaction Detection:** phát hiện cặp người–vật và tương tác có thể được mô tả bằng văn bản, dùng một tập HOI nền tảng, không xây bộ dữ liệu và huấn luyện mô hình riêng cho mỗi hành vi.

**M2 — Video–Text Cross-modal Alignment / Retrieval:** học ánh xạ video và văn bản vào một **Shared Semantic Embedding Space**, dùng similarity để xếp hạng video/caption.

- **RQ1:** Trên cùng benchmark và giao thức lớp seen/unseen, mô hình phát hiện HOI tổng quát hóa đến tương tác chưa dùng làm nhãn huấn luyện đến đâu? Sai số nằm ở định vị cặp hay nhận biết tương tác?
- **RQ2:** So với pooling/checkpoint tham chiếu, học alignment và thông tin thời gian cải thiện retrieval hoặc giảm chi phí tính toán đến đâu khi giữ nguyên điều kiện đánh giá?

Mô hình closed-set bị giới hạn bởi tập nhãn học, nhưng không phải mọi thay đổi quy tắc đều đòi huấn luyện lại toàn bộ mô hình. Hành vi nhìn thấy, độ khớp câu và phán quyết vi phạm là các đại lượng khác nhau.

## 2. Thuật ngữ, đầu ra và giới hạn

| Khái niệm | Định nghĩa dùng trong dự án |
|---|---|
| HOI detection | Đầu ra gồm person box, object box/class, interaction label, confidence và frame/video ID. Vector HOI chỉ là biểu diễn trung gian. |
| Zero-shot HOI | Nhóm nhãn unseen không tham gia huấn luyện/chọn cấu hình cho lần đánh giá đó. Phải công bố chính xác loại giữ lại: tổ hợp tương tác, động từ hoặc vật thể. |
| Open-vocabulary | Nhận tập khái niệm qua văn bản; phải ghi rõ thành phần nào mở. Không đồng nghĩa nhận đúng mọi mô tả tùy ý. |
| Shared Semantic Embedding Space | Biểu diễn hai phương thức được học căn chỉnh; cùng số chiều và chuẩn hóa L2 chưa đủ. |
| Reproduction | Tách chạy lại evaluator trên dự đoán công bố, chạy inference checkpoint và huấn luyện lại. Ba mức không được gọi thay cho nhau. |
| Domain transfer | Chuyển môi trường/camera; không tự chứng minh khả năng nhận biết lớp unseen. |

Đầu ra M1 mục tiêu: video_id, frame_id/timestamp_s, person_bbox_xyxy, object_bbox_xyxy, object_label, interaction_label, score; track_id tùy phương pháp. Định nghĩa hệ tọa độ, đơn vị, class mapping và NMS/top-K theo evaluator. Dữ liệu dự đoán này **chưa có** trong manifest pilot.

Đầu ra M2: video_id, text_id, similarity, rank và retrieval_direction; embedding được lưu cùng encoder/checkpoint/hash/normalization. Cosine nằm trong [-1,1]; logits chia temperature không bị giới hạn trong khoảng này và không phải xác suất vi phạm.

Phạm vi chính là nghiên cứu ngoại tuyến. Không yêu cầu realtime, nhận diện danh tính, nhiều camera, tự động hiểu luật pháp hoặc xây parser ngôn ngữ tùy ý. Ứng dụng ROI/dwell và demo chỉ triển khai khi không làm chậm các mốc nghiên cứu chính.

## 3. Quyết định dữ liệu và baseline

### 3.1. Module 1: một tập nền tảng VidHOI

Chọn **VidHOI** để duy trì phạm vi HOI trong video. Không trộn Action Genome, VIRAT hoặc ShanghaiTech vào một benchmark chung. Dữ liệu demo không thay benchmark này.

[ST-HOI — Chiou et al., ACM ICMR Workshop 2021](https://github.com/coldmanck/VidHOI) là baseline lịch sử. Repository cho biết checkpoint cũ không còn được cung cấp, nhưng có dự đoán để kiểm tra evaluator. Vì vậy phải xác minh artifact trước khi cam kết chạy model; chấm lại dự đoán không phải tái lập inference/huấn luyện. Evaluator có quy ước lọc frame và oracle/predicted tracks riêng; phải khóa chúng.

**Điều kiện W03 bắt buộc:** xác minh giao thức zero-shot được công bố trên VidHOI và một baseline hiện đại có mã/weights tương thích. ST-HOI không mặc nhiên đáp ứng điều kiện này. Nếu chưa tìm được:

1. Ghi rõ phần tái lập zero-shot SOTA là **chưa đạt**, không thay bằng kết quả closed-set.
2. Chỉ chạy nghiên cứu thí điểm với giao thức nội bộ được công bố dưới đây; không so trực tiếp với số SOTA khác split.
3. Trước khi đầu tư huấn luyện lớn, ghi quyết định điều chỉnh baseline/phạm vi với người hướng dẫn; chưa tự đổi sang benchmark ảnh.

**Giao thức nội bộ dự kiến nếu cần:** unseen-composition, vật thể và động từ thành phần có mặt trong seen nhưng một số cặp tương tác bị giữ lại. Tạo danh sách lớp bằng thống kê train và seed cố định, không dựa vào score test; loại khỏi supervised training các clip/window chứa nhãn unseen, kể cả qua nhãn âm hoặc feature cache. Tạo pseudo-unseen từ phần train để chọn cấu hình, tách khỏi unseen cuối. Lưu ID và báo cáo số lớp/mẫu sau lọc; chỉ thực hiện nếu mỗi nhóm đủ mẫu để đánh giá. Đây chưa phải split đã tạo.

[SL-HOI, CVPR 2026](https://github.com/MPI-Lab/SL-HOI) là ứng viên khảo sát HOI ảnh, có công bố weights và yêu cầu CUDA; tài liệu cài đặt còn mục dependency chưa hoàn chỉnh. Không dùng kết quả trên HICO-DET thay cho kết quả VidHOI. Không tải thêm tập HOI thứ hai chỉ để làm đẹp bảng so sánh.

### 3.2. Module 2: MSR-VTT

Chọn **MSR-VTT**, protocol 9k-train/1k-test của [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip); khóa file ID/caption và commit loader/evaluator. CLIP4Clip mean pooling là baseline lịch sử. Khảo sát thêm phương pháp hiện đại theo khả năng có checkpoint, cùng split và tài nguyên; chưa coi tên CLIP/OpenCLIP/CLIP4Clip là bằng chứng đã tái lập SOTA.

Tên tham số val_csv trong ví dụ upstream có thể trỏ vào test; dự án không dùng test đó để chọn checkpoint. Để học head mới, giữ lại 10% video từ train làm validation bằng seed 42, lưu danh sách trước huấn luyện. Báo cáo đây là giao thức phát triển điều chỉnh. Nếu so với checkpoint dùng đầy đủ train, ghi rõ khác biệt; đối chứng nội bộ phải dùng cùng train/validation.

Tập test/candidate pool chính thức giữ nguyên. Khóa cách dùng caption, khử bản sao, chiều truy vấn và xử lý nhiều positive. Dữ liệu caption tiếng Anh dùng cho benchmark; caption tiếng Việt dịch hoặc demo phải báo riêng. Caption sinh từ nhãn HOI là thí nghiệm bổ sung, không gọi là MSR-VTT.

### 3.3. Pilot W02 và quyền dữ liệu

Pilot hiện có: **8 đoạn, 5 video, 22,63 MiB**, 6 dev + 2 extension đã xem, 10 câu/80 cặp. Không có test độc lập, nhãn hộp HOI hay nhãn sự kiện vi phạm đã xác minh. Giới hạn 50 MiB, không bắt buộc 80 đoạn và không đặt số lượng mới để thay yêu cầu đó.

Pilot dùng để kiểm tra I/O, schema, metric và đường chạy model; không đủ để tuyên bố chất lượng benchmark. Giữ provenance, hash, quyền dùng và phạm vi rà nhãn thực tế. Hash không chứng minh tác quyền; license code không tự áp cho footage.

## 4. Yêu cầu chức năng và bằng chứng nghiệm thu

Các mã FR được chuẩn hóa lại trong phiên bản này, không suy diễn trạng thái từ bảng FR cũ.

| Mã | Yêu cầu mục tiêu | Bằng chứng cần có |
|---|---|---|
| FR01 | Đọc clip đúng khoảng [start,end), giới hạn bộ nhớ | Test timestamp, frame thiếu/hỏng, resize; log RAM loader riêng |
| FR02 | Adapter benchmark và chống leakage | ID/split/class mapping/hash; kiểm tra trùng video và lớp held-out |
| FR03 | HOI inference đúng schema | Dự đoán hộp/nhãn/score hợp lệ; ghi oracle/predicted; evaluator chạy được |
| FR04 | Zero-shot/Open-vocabulary evaluation | Danh sách seen/unseen, nguồn supervision, mAP từng nhóm, bằng chứng không dùng unseen để chọn cấu hình |
| FR05 | Học alignment M2 | Train/validation rõ; loss hữu hạn, tham số được cập nhật, checkpoint có optimizer/config; probe nhỏ xác nhận đường học |
| FR06 | Retrieval hai chiều | R@1/5/10, MedR theo evaluator; test tính tay, candidate pool và positive mapping |
| FR07 | So sánh baseline và ablation | Cùng split/encoder/ngân sách, log thí nghiệm; chênh lệch so với số tham chiếu và nguyên nhân |
| FR08 | Tái lập và tài nguyên | Mã phiên bản, môi trường, checkpoint hash, seeds, peak RAM/VRAM, thời gian đo |
| FR09 | Demo ROI/dwell có điều kiện | Quy tắc cấu trúc, hộp/track/thời gian; thiếu bằng chứng trả unknown, không đổi thành âm tính |

FR01 có phần triển khai trong pilot; các FR khác chỉ được đánh dấu đạt khi có artifact tương ứng. Không dùng kiểm thử đơn vị để xác nhận đã hoàn thành nghiên cứu.

## 5. Huấn luyện và cấu hình khởi đầu

### 5.1. M1

Tái lập baseline theo cấu hình của tác giả trong môi trường riêng trước khi thay encoder, số frame hoặc loss. Ghi detector tiền huấn luyện và supervision có thể đã bao phủ vật thể/khái niệm unseen; zero-shot được phát biểu theo nhãn downstream, không khẳng định pretraining không rò rỉ.

Mọi cải tiến phải được học chung trên split seen, không fit classifier riêng cho từng quy tắc demo. Khi không đủ tài nguyên hoặc weights không truy cập được, ghi đúng mức chỉ chạy evaluator/prototype.

### 5.2. M2: cấu hình adapter nhẹ dự kiến

Cấu hình này là **đề xuất thử nghiệm**, chưa phải code/config đã chạy; không thay cấu hình tác giả khi báo reproduction.

- Backbone: encoder ảnh/text cùng một checkpoint đã căn chỉnh; đóng băng backbone, precompute feature theo chunk nếu vừa dung lượng.
- Video: mean pooling làm mốc; head tuyến tính làm mốc học; thử MLP chỉ sau khi mốc chạy được. D lấy theo encoder, không mặc định mọi model đều 512.
- Loss: contrastive hai chiều, với tử số cộng exp(similarity/temperature) trên positive đã biết, mẫu số trên candidate hợp lệ; trung bình hai chiều. Mask caption cùng video và positive khác đã biết để tránh false negatives; không đưa nhãn unknown vào negative có giám sát.
- Khởi đầu: AdamW, lr=1e-4, weight_decay=0.01, tối đa 10 epoch, patience=3, temperature=0.07 cố định; chọn checkpoint bằng validation R@1 text→video. Các số này cần kiểm chứng, không phải siêu tham số tối ưu.
- Batch feature mục tiêu 16 video khác nhau; đo RAM trước. Decode batch=1 không bắt buộc batch học feature=1. Contrastive chỉ có một cặp và không có negatives không phải cấu hình học hợp lệ; gradient accumulation thông thường không tự mở rộng tập negatives.
- Seed phát triển 42; khi đủ ngân sách, xác nhận bằng seeds 42/43/44, báo mean/std. Một seed phải được ghi là thăm dò.
- Không fit bằng 8 đoạn pilot rồi gọi đó là benchmark MSR-VTT. Cache feature phải lưu encoder/preprocess/split hash; thay encoder làm mất hiệu lực cache.

## 6. Metrics và đánh giá công bằng

### 6.1. M1: HOI mAP

Dùng evaluator đúng benchmark; lưu phiên bản và tham số matching/top-K/lọc frame. AP tính theo từng lớp trên detection có score; mAP lấy trung bình lớp theo quy ước evaluator. Không thay bằng accuracy phân loại hoặc Event-F1.

Báo cáo full/seen/unseen theo protocol; rare/non-rare chỉ thêm khi định nghĩa chính thức phù hợp. Lớp không có positive phải được xử lý như evaluator quy định, không tự gán AP=0. Nếu dùng frame-level mAP, không gọi kết quả là temporal tube mAP.

### 6.2. M2: retrieval

Gọi r_q là thứ hạng 1-based của positive đầu tiên cho truy vấn q:
- R@K = mean(1[r_q <= K]), K=1,5,10.
- MedR = median(r_q), thấp hơn tốt hơn.
- MRR = mean(1/r_q), cao hơn tốt hơn; là metric phụ.
- R@3 và Pairwise Accuracy là phép chẩn đoán pilot; hòa điểm pairwise không tính là thắng.

Báo riêng text→video và video→text, số query/candidate, số positive và tie-breaking. Truy vấn chưa gán nhãn loại khỏi phép đo với số lượng báo rõ; không tính là âm tính. Trong benchmark, giữ evaluator chính thức; khác định nghĩa multi-positive phải báo thành bảng riêng.

### 6.3. Ứng dụng vi phạm tùy chọn

Ghép sự kiện dự đoán theo score giảm dần với GT cùng video_id/rule_id, mỗi GT tối đa một lần, tIoU >= 0.5. Báo Event-P/R/F1; dự đoán trùng tính FP. Mẫu số bằng 0 trả None/null. FA/hour chỉ tính trên thời lượng nền đã xác minh, ghi số cảnh báo và số giờ, không ngoại suy từ vài chục giây thành độ tin cậy vận hành.

### 6.4. Ablation và kết luận

| Thí nghiệm | Biến thay đổi | Biến giữ cố định |
|---|---|---|
| B0 → B1 | Global → union crop | Encoder, pooling, head, nguồn dữ liệu |
| C00/C10/C01/C11 | Temporal tắt/bật × geometry tắt/bật | Union crop, head H, số chiều, train split, optimizer/ngân sách |
| Head ablation | Linear → MLP/alignment head mới | Cùng cấu hình C và dữ liệu |
| Oracle → predicted | Nguồn hộp/quỹ đạo | Model/evaluator và tập frame so sánh phù hợp |

C00 là mốc có head H được học; không đồng nhất với B1 frozen. Ghi số tham số, độ trễ, RAM/VRAM để tránh quy cải thiện đơn thuần do tăng dung lượng model thành đóng góp cơ chế.

Temporal module phải có timestamp/positional encoding và valid mask. Geometry chuẩn hóa theo kích thước frame; vận tốc dùng delta_position/delta_time, xử lý delta_time <= 0 và track gap. Sampling là chọn frame, không phải attention. Kiểm tra thứ tự trên ví dụ có nội dung bất đối xứng; không yêu cầu mọi chuỗi đảo ngược đều đổi output.

Ưu tiên 3 seeds cho mô hình có học và bootstrap theo video để phản ánh tương quan caption, nếu đủ ngân sách. Công bố độ bất định, kết quả âm và giới hạn; mục tiêu cải thiện không phải điều kiện bắt buộc để được ghi nhận đã thực hiện thí nghiệm.

## 7. Schema và ranh giới triển khai

Schema **hiện có** trong data/manifests:

| File | Trường thực tế chính |
|---|---|
| clips.csv | clip_id, video_path, video_id, session_id, start_s, end_s, behavior_id, rule_id, has_violation, zone_polygon, zone_status, event_start_s, event_end_s, evaluation_eligible |
| texts.csv | text_id, behavior_id, prompt_text, prompt_type, description |
| splits.csv | clip_id, split, session_id |
| relevance.csv | clip_id, text_id, is_match, reviewer, reviewed_at |
| sources.csv | video_id, video_path, source_url, attribution, license_status, license_url, source_checked_at, sha256, bytes, duration_s, fps |

Pilot loader hiện cho dev/val/test/extension; chưa hỗ trợ split train. Nhãn has_violation trống → None; relevance chỉ nhận 0/1, cặp chưa biết không có dòng. Metadata độ phân giải/fps có thể lấy từ reader; không giả định chúng là cột bắt buộc trong clips.csv.

**Phần cần triển khai W03–W04:** adapter train/validation/evaluation cho benchmark, annotation HOI/box/track/class mapping, caption IDs, seen/unseen lists; giữ nguyên annotation gốc và lưu mapping. Không đổi tên tập val chính thức thành test rồi che giấu nguồn gốc: tách tên split upstream khỏi vai trò model-selection/final-evaluation. Nếu benchmark chỉ có val công khai để chấm, dùng phần train giữ lại cho model-selection và báo rõ final-evaluation là official val.

Ranh giới code mục tiêu:
- data: đọc/biến đổi/split/adapter; không chứa mạng học.
- models và training: encoder/head/loss/optimizer/checkpoint, **chưa triển khai**.
- inference: dự đoán và xếp hạng; application tùy chọn cho ROI/dwell.
- evaluation: metric và adapter evaluator; không học head hoặc ra quyết định luật.

## 8. Tài nguyên, tính tái lập và rủi ro

- Ưu tiên máy cá nhân, đọc một clip/lần, không tải toàn bộ benchmark trong W02. Đo riêng dung lượng raw/frame/cache/checkpoint và RAM/VRAM inference/training.
- W03 phải ghi phần cứng thực, RAM/VRAM còn trống, tốc độ trên mẫu nhỏ và ước lượng toàn bộ run trước khi cam kết; không suy RAM model từ loader <200 MiB.
- Môi trường CPU Python 3.12 của dự án không mặc nhiên tương thích repository baseline cũ/CUDA. Tách môi trường thay vì ép nâng/hạ toàn bộ môi trường pilot.
- Không tự mua GPU hoặc chuyển video riêng tư lên dịch vụ ngoài. Thiếu compute thì giới hạn công bố ở mức thực sự kiểm chứng; giai đoạn 2 có thể tiếp tục sau 10 tuần.
- Lưu commit, dataset/split/checkpoint hash, seed, preprocessing, thời gian, số lần chạy, evaluator và log lỗi. Kiểm tra tái lập trong môi trường đã mô tả, dùng tolerance thích hợp; không cam kết giống tuyệt đối trên mọi thiết bị.
- Detector lỗi: phân tích riêng oracle/predicted. Nhãn thiếu: giữ unknown. Dataset không truy cập được: ghi blocked cho thí nghiệm đó và làm phần độc lập; không đổi dữ liệu để giả lập điểm chuẩn.
- ROI/dwell nếu triển khai: cấu hình cấu trúc và câu mẫu được kiểm soát, không nhận video/text tùy ý rồi hứa hiểu mọi quy tắc. Bbox không phải segmentation pixel-level; attention map không tự là bằng chứng định vị đúng.

## 9. Hai giai đoạn và điều kiện hoàn thành

**Giai đoạn 1 đạt** khi cả hai module có baseline mục tiêu, protocol/evaluator khóa, kết quả có thể đối chiếu, phân tích sai lệch/bottleneck và thí nghiệm cải tiến có kiểm soát. Baseline lịch sử không tự đáp ứng mục tiêu tái lập SOTA hiện đại; nếu ứng viên mới chưa chạy được, ghi rõ phần đó chưa hoàn thành. Cải thiện nhẹ mAP/R@K hoặc giảm MedR là mục tiêu, không hứa trước.

**Giai đoạn 2 bắt đầu** khi đã có bằng chứng giai đoạn 1 và ngân sách được xác định. Chọn một cơ chế mới từ lỗi quan sát, nêu giả thuyết, phép bác bỏ và tiêu chí đo. Đầu ra là code, đối chứng và bản thảo trung thực; tính mới, khả năng được nhận bài và xếp loại luận văn cần đánh giá bên ngoài.

**W01/W02 cập nhật 30/09/2026:** đã hoàn thiện nội dung chuẩn bị/pilot, bảng khảo sát và sơ đồ hai module; bổ sung R@K/MRR/MedR hai chiều có kiểm thử. Báo cáo sẵn để nộp, chưa xác nhận bàn giao bên ngoài. Đây không phải hoàn thành toàn bộ PRD: adapter benchmark, evaluator HOI, reproduction và training vẫn theo các tuần tiếp theo trong Plan. Xem [kiểm chứng](../outputs/w02/completion_verification.json); log cũ được giữ riêng, không ghi lùi kết quả mới.
