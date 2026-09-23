# PRD Đồ án Temporal HOI Video–Text Matching

**Phiên bản:** 0.1 — đề xuất phạm vi triển khai, ngày 23/09/2026.  
**Người thực hiện:** Ngô Hoàng Phú.  
**Thời gian đã xác nhận:** 14/09–22/11/2026, 20 giờ/tuần; có thể thuê GPU.  
**Trạng thái sản phẩm:** chuẩn bị môi trường và tài liệu. Các yêu cầu dưới đây là tiêu chí cần xây dựng, không phải khả năng đã có.

## 1. Mục đích và bài toán

Xây dựng nguyên mẫu hỗ trợ xem lại video để phát hiện một số tương tác người–vật trái với luật đã khai báo. Hệ thống liên kết hành vi trong video với mô tả văn bản, rồi xét loại luật, vùng và thời gian để tạo sự kiện có bằng chứng.

Ví dụ, cùng hành vi cầm chai có thể được phép ngoài vùng A nhưng bị cấm trong vùng A. Model nhận biết hành vi; mô-đun luật quyết định điều kiện vi phạm. Hệ thống không tự suy ra quyền truy cập của cá nhân hoặc hiểu mọi quy định an toàn từ câu bất kỳ.

Sản phẩm phục vụ trình diễn và đánh giá nghiên cứu trước giảng viên/hội đồng. Báo cáo phải chỉ rõ phần kế thừa, phần tự triển khai, lợi ích hoặc giới hạn của cải tiến và cách tái lập kết quả.

## 2. Người dùng và luồng sử dụng

| Người dùng | Nhu cầu | Đầu ra |
|---|---|---|
| Sinh viên/người nghiên cứu | So baseline và cải tiến trên cùng dữ liệu | Metric, run/config và ca lỗi truy vết được |
| Giảng viên/hội đồng | Kiểm tra đóng góp và phương pháp đánh giá | Đối chứng, ablation, giới hạn và video minh chứng |
| Người vận hành demo | Áp dụng luật lên video và vùng cụ thể | Người/vật, hành vi, thời gian và bằng chứng |

Luồng MVP:

1. Chọn video mẫu hoặc upload video tương thích.
2. Chọn luật mẫu hoặc nhập câu thuộc mẫu hỗ trợ; kiểm tra schema hiển thị hành vi, vật thể, loại luật và điều kiện.
3. Vẽ/chọn polygon vùng; xác nhận điểm neo người/vật và quy tắc vào/ra vùng.
4. Chọn B1 hoặc mô hình cải tiến, chạy ngoại tuyến và theo dõi trạng thái.
5. Xem overlay box/ID, bảng event và đoạn bằng chứng; so sánh trên cùng input.
6. Xuất JSON/CSV và video. Không có event, lỗi xử lý và thiếu bằng chứng phải được phân biệt.

## 3. Mục tiêu nghiên cứu và hoàn thành

**RQ1, chính:** temporal attention và geometry/motion có giúp nhận biết tương tác tốt hơn mean pooling trên cùng CLIP backbone, split, loss và ngân sách chọn siêu tham số không?

**RQ2, hỗ trợ:** projection/adapter nhỏ có căn chỉnh được representation temporal với text trong phạm vi dữ liệu đã chọn không?

Việc dùng projection/InfoNCE có sẵn không tự là thuật toán mới. Tính mới phải đối chiếu literature; đóng góp tối thiểu là triển khai có kiểm soát, đánh giá tái lập và phân tích bằng chứng.

| Nhóm | Tiêu chí hoàn thành |
|---|---|
| Nghiên cứu | B0/B1, phương pháp đề xuất, ablation temporal/geometry và phân tích ít nhất 10 ca lỗi |
| Dữ liệu | Nguồn/license, manifest, split, nhãn và số lượng thực; không chọn model/ngưỡng trên test |
| Sản phẩm | Video tới event bằng predicted tracks; export, 3 tình huống đối chứng và hướng dẫn |
| Tái lập | Kết quả gắn commit, config, seed, split/weight hash và predictions |
| Bàn giao | Repo, báo cáo tổng thể, 10 báo cáo tuần, slide, video dự phòng và checklist nộp |

Mục tiêu mong muốn: tăng khoảng 2 điểm phần trăm macro-AP so B1 trên validation, hoặc giảm false alarms tại recall tương đương. Đây là giả thuyết định hướng, cần chốt sau baseline tuần 3, không phải kết quả/cam kết đạt được. Kết quả âm vẫn phải báo trung thực cùng ablation.

## 4. Phạm vi MVP

### Bắt buộc

- Một camera cố định, video đã ghi; không yêu cầu streaming/realtime.
- Một dataset HOI chính và một tập video luật nội bộ có đối chứng.
- Hai đến ba luật cấm quan sát được trong một bối cảnh phòng/lab; lớp vật chốt theo dataset và detector ở tuần 2.
- Pair window, biểu diễn temporal, video-text matching và luật zone/dwell.
- Luật tiếng Việt theo mẫu hữu hạn; schema và prompt tiếng Anh đã kiểm tra. Demo nêu rõ giới hạn ngôn ngữ.
- Baseline, một cải tiến chính, đánh giá và sản phẩm có bằng chứng.

### Tùy chọn sau khi MVP ổn định

Luật thứ ba, crossing ranh giới, so sánh một backbone video-language khác, tăng tốc suy luận. Bỏ các hạng mục này trước khi ảnh hưởng báo cáo hoặc deadline.

### Ngoài phạm vi mặc định

Full dynamic scene graph/GNN, full cross-attention transformer, huấn luyện detector/foundation model mới, đa camera, định danh cá nhân/phân quyền, ngôn ngữ tùy ý, PPE vắng mặt, tự động ra quyết định kỷ luật, ứng dụng production/mobile và bảo đảm realtime.

| Luật minh họa | Điều kiện vi phạm | Âm tính khó |
|---|---|---|
| Cấm cầm vật X trong vùng A | Cầm X đủ tin cậy + đúng vùng + đủ thời lượng | Đi gần X không cầm; cầm X ngoài A |
| Cấm cầm vật Y trong vùng B | Cầm Y đủ tin cậy + đúng vùng + đủ thời lượng | Vật Y đứng yên trong B; người vào B không cầm Y |
| Cấm mang vật Y vào vùng B, tùy chọn | Mang Y + ngoài → trong B + đủ bằng chứng temporal | Vật đứng yên trong B; người vào B không mang Y |

“Cầm” và “mang” không mặc định cùng nhãn. Nếu dataset không có nhãn/chuỗi để phân biệt, giới hạn demo theo nhãn có bằng chứng. Crossing cần state qua thời gian, không thay bằng membership của một frame.

## 5. Yêu cầu chức năng

| ID | Yêu cầu | Tiêu chí nghiệm thu |
|---|---|---|
| FR01 | Nhận video, kiểm tra metadata | Đọc được frame/timestamp; file hỏng/không hỗ trợ có lỗi rõ, không tạo kết quả giả |
| FR02 | Luật mẫu | Có ID, type, subject, action, object, zone, duration; câu không hỗ trợ bị từ chối hoặc yêu cầu sửa |
| FR03 | Khai báo vùng | Polygon chuẩn hóa theo khung hình; báo polygon không hợp lệ; tái lập zone decision |
| FR04 | Detection/tracking | Có person/object box, track ID và confidence; ID không được coi là danh tính thật |
| FR05 | Pair window | Đúng track/timestamp; valid mask cho thiếu quan sát; giới hạn cặp bằng quy tắc cấu hình |
| FR06 | Baseline/cải tiến | Cùng input/tiền xử lý, ghi model/config version; không so hai split khác nhau |
| FR07 | Matching | Hiển thị similarity đúng loại; không gọi cosine là xác suất; cache text theo prompt/model |
| FR08 | Event | Xét rule/zone/dwell; deduplicate theo người–vật–luật; lưu start/end/emitted timestamp |
| FR09 | Thiếu bằng chứng | Track mất/che khuất quá giới hạn có trạng thái insufficient_evidence; không thành negative chắc chắn |
| FR10 | Bằng chứng | Event có box/ID, rule, thời gian và clip/ảnh; xem đúng thời điểm |
| FR11 | Export | JSON/CSV đúng schema, khớp UI; không có event vẫn xuất kết quả rỗng hợp lệ |
| FR12 | Nhật ký thí nghiệm | Lưu run ID, config, seed, commit, split/weight hash, metric và predictions |

Giới hạn đầu vào đề xuất để pilot: MP4 H.264 hoặc video mẫu đã kiểm thử, khoảng tối đa 2 phút/video ở 720p. Chốt lại theo pilot; chưa cam kết thời gian xử lý.

## 6. Yêu cầu chất lượng

- **Tái lập:** dependency khóa bằng uv.lock; code/config không hardcode đường dẫn máy cá nhân; ghi seed/data version.
- **Hiệu năng:** đo wall time/video, latency/clip và peak VRAM trên cấu hình cụ thể. Chỉ nói realtime sau khi tính cả đợi window và xử lý.
- **Ổn định:** thử video hỏng, không người, ngoài vùng, nhiều cặp, che khuất; không crash hoặc bỏ lỗi im lặng.
- **Giải thích:** dùng bbox, luật, score và đoạn bằng chứng; attention map không thay metric grounding.
- **Dữ liệu:** dùng video có quyền sử dụng và dữ liệu dàn dựng có đồng thuận; demo local, không cần gửi video cho dịch vụ ngoài.
- **Bàn giao:** cài từ README trên môi trường sạch, chạy bộ mẫu, có backup ngoại tuyến.

## 7. Kiến trúc và ranh giới mô-đun

```text
Video -> detector/tracker -> pair windows -> union crop CLIP + geometry/motion
      -> mean pooling (B1) hoặc temporal adapter (P) -> video projection
Luật mẫu -> schema -> prompt hành vi -> CLIP text -> matching
Schema + polygon + matching -> logic vi phạm -> smoothing/event -> demo/export
```

Detector và CLIP đóng băng; học temporal adapter và projection nhỏ. CLIP image/text đã pretrained alignment; bổ sung temporal/geometry cần kiểm chứng alignment lại. L2 normalization không thay thế học ngữ nghĩa.

Tầng 2 MVP là pair encoder; không gọi graph model khi chưa định nghĩa node/edge và message passing. Graph/GNN thuộc hướng mở rộng.

UI và CLI gọi chung `InferencePipeline`. Training/evaluation không phụ thuộc UI. Evaluator đọc predictions theo schema; notebook dùng khảo sát. Stack nền là Python 3.12, PyTorch/torchvision, OpenCLIP, OpenCV, NumPy/pandas, YAML/Pydantic, scikit-learn/matplotlib và Gradio. Detector/tracker chưa chốt; ghi weight/license/version trước tích hợp.

## 8. Dữ liệu và schema

Ưu tiên VidHOI, kiểm tra tải/lớp trong tuần 2. Action Genome là dự phòng, có nhãn tại frame lấy mẫu; không giả định có tubelet liên tục. Chỉ triển khai một adapter chính trong 10 tuần.

Mục tiêu nguồn lực: pilot 20 clip; subset HOI 300–600 clip nếu đủ điều kiện; benchmark luật khoảng 60 clip/3 luật, có thể giảm 40 clip/2 luật. Đây là số mục tiêu, phải báo số thực tế. Thêm khoảng 20 phút video bình thường liên tục cho false alarms/giờ.

Split theo video gốc/phiên quay; từng luật có positive/negative ở validation/test. Caption mẫu từ label không phải benchmark ngôn ngữ tự do. Nhãn cùng đúng là multi-positive; nhãn thiếu không mặc định negative. Kiểm tra lại ít nhất 20% annotation, ghi rõ kiểm tra độc lập hay self-review.

| Record | Trường tối thiểu |
|---|---|
| VideoManifest | video_id, source_id, session_id, path, duration_s, fps, split, annotation_version, checksum |
| PairWindow | pair_id, person_track_id, object_track_id, timestamps_s, appearance, geometry, valid_mask |
| Rule | rule_id, text_vi, rule_type, action, object, zone_id, behavior_prompt_en, min_duration_s, threshold_ref |
| Event | event_id, video_id, person/object IDs, rule_id, start_s, end_s, emitted_at_s, score, evidence_path, evidence_status |
| RunMetadata | run_id, model_id, track_mode, commit, config_hash, split_hash, weight_hash, seed, package_versions |

Rule crossing thêm hướng chuyển trạng thái. Zone chứa polygon, hệ tọa độ và điểm neo. Box khai báo xyxy/xywh và pixel/normalized. Track ID chỉ có nghĩa trong video/run; không so ID số học trực tiếp giữa GT và predicted tracks.

## 9. Thực nghiệm và metric

| Cấu hình | Vai trò |
|---|---|
| B0: CLIP toàn cảnh + mean pooling | Mốc đơn giản, ảnh hưởng nền |
| B1: union crop + mean pooling + projection học | Baseline trực tiếp |
| A1: B1 + temporal attention, không geometry | Lợi ích thời gian |
| A2: geometry + mean pooling | Lợi ích geometry/motion |
| P: geometry + temporal attention + projection | Phương pháp đề xuất |

B1/A1/A2/P cùng backbone, split, số frame, prompt, loss và ngân sách chọn siêu tham số; báo số tham số trainable. Mục tiêu 3 seed cho B1/P nếu pilot cho phép; công bố số seed thực chạy. Temporal shuffle là phép kiểm tra phụ. Zone/smoothing đánh giá riêng trên cùng model để không gán lợi ích hậu xử lý cho encoder.

Metric HOI: macro-AP đa nhãn trên lớp đã khóa, thêm micro-AP/Recall@K với candidate set và positive mask rõ ràng. Nhãn chưa đầy đủ cần tập đánh giá có nhãn tin cậy hoặc protocol phù hợp chốt trước test.

Metric event: precision/recall/F1 với one-to-one matching cùng video/rule và temporal IoU ≥ 0.5; duplicate event là FP. Đánh giá “ai” cần ghép track theo overlap; báo riêng temporal-only và temporal+spatial với IoU ≥ 0.5 trên frame có nhãn. Báo false alarms/giờ video bình thường, trễ từ GT onset tới lúc phát cảnh báo, latency/VRAM trên máy thật.

Chốt ngưỡng, dwell, smoothing, checkpoint/seed trên validation. Tách oracle khỏi predicted tracks. Test cuối tuần 8, không tuning lại trên test; nêu giới hạn mẫu/camera/lớp và kết quả âm.

## 10. Mốc và sản phẩm bàn giao

| Mốc | Điều kiện thoát |
|---|---|
| Tuần 2 | Môi trường, dữ liệu pilot, lớp/luật, split và protocol |
| Tuần 3 | B0/B1 và evaluator có validation tái lập |
| Tuần 4 | Perception, geometry, dữ liệu demo và UI khung |
| Tuần 5–6 | Cải tiến, ablation, ca lỗi; khóa kiến trúc cuối tuần 6 |
| Tuần 7 | Pipeline luật/event/demo và threshold validation |
| Tuần 8 | Test cuối, demo đầu-cuối, báo cáo nháp toàn bộ |
| Tuần 9 | Môi trường sạch, sửa báo cáo, slide và diễn tập |
| Tuần 10 | Nộp repo, báo cáo, slide, demo/backup và gói tái lập trước 22/11 |

Mỗi tuần 15 giờ chính, 3 giờ báo cáo/gặp thầy, 2 giờ dự phòng. Tuần 1 đã qua cần đối chiếu; 200 giờ là quỹ cả kỳ, không phải quỹ còn lại. Lịch chi tiết: [plan.md](../plan.md) và [Excel](Ke_hoach_do_an_10_tuan.xlsx).

## 11. Rủi ro và quyết định còn mở

| Vấn đề | Xử lý | Hạn chốt |
|---|---|---|
| Dataset/weight không truy cập | Pilot sớm, đổi dữ liệu sau tối đa 2 ngày bị chặn | Tuần 2 |
| Luật không khớp dữ liệu/detector | Giảm luật, chọn vật lớn/hành vi có nhãn | Tuần 2–4 |
| Thiếu GPU/thời gian | Feature cache, subset nhỏ, T=8; bỏ backbone phụ | Sau pilot tuần 2 |
| Không cải thiện | Kiểm tra loss/data, công bố kết quả âm và ablation | Tuần 6 |
| Tracking kém | Tách oracle/predicted, ghi ca lỗi, giới hạn cảnh | Tuần 4–8 |
| Scope tăng/trễ báo cáo | Cắt realtime/luật phụ/UI phụ; khóa tính năng | Rà mỗi tuần |

Còn cần chốt: dataset/subset; lớp/luật; detector/weight/license; GPU và ngân sách; ngày báo cáo theo lịch thầy; mẫu báo cáo trường. Không chọn mọi framework trước pilot.

## 12. Tiền đề thạc sĩ và nguồn

Giữ data adapter, manifest/split, protocol, baseline, run registry và ca lỗi. Có thể chọn một hướng thạc sĩ: dynamic graph đa đối tượng, fine-grained grounding, compositional generalization của luật, uncertainty cho điều kiện vắng mặt hoặc domain adaptation qua camera. Chọn theo bằng chứng cử nhân; không đưa tất cả vào MVP.

Nguồn định hướng: [báo cáo tuần 2](report/week2.docx), [nhận xét](report/week2-review.md), [plan.md](../plan.md). Nguồn kỹ thuật: [CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip), [VidHOI](https://github.com/coldmanck/VidHOI), [Action Genome](https://github.com/JingweiJ/ActionGenome). Số giờ, kích thước tập và mục tiêu tăng metric là đề xuất kế hoạch, không phải kết quả các công trình này.
