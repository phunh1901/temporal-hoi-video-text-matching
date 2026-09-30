# Product Requirements Document (PRD)
# Temporal HOI Video–Text Matching

**Tên đề tài:** Nghiên cứu mô hình hóa tương tác người–vật thể theo thời gian (Spatio-Temporal HOI) và so khớp video–văn bản (Video–Text Matching) phục vụ phát hiện vi phạm quy định có căn cứ bằng chứng.  
**Người thực hiện:** Ngô Hoàng Phú  
**Trạng thái tài liệu:** Tài liệu đặc tả yêu cầu kỹ thuật và phân tích đề tài nghiên cứu (PRD).

---

## 1. Tổng quan Đề tài & Bối cảnh Khoa học (Research Context)

### 1.1. Bối cảnh bài toán
Trong các hệ thống giám sát an ninh (CCTV) tại khu dân cư, chung cư và đô thị (khuôn viên, bãi đỗ xe, điểm tập kết rác, đường nội bộ), nhu cầu phát hiện các hành vi vi phạm quy định an toàn ngày càng trở nên cấp thiết. Tuy nhiên, các giải pháp thị giác máy tính truyền thống chủ yếu dựa trên bài toán **nhận dạng hành vi tập đóng (*closed-set action recognition*)**. 

Cách tiếp cận này bộc lộ những hạn chế nghiêm trọng:
- **Kém linh hoạt:** Mỗi khi có một quy định mới phát sinh (ví dụ: cấm dắt xe đạp qua sân chơi trẻ em, cấm đỗ xe sai vị trí, cấm để rác ngoài giờ quy định), hệ thống đòi hỏi phải định nghĩa lại lớp nhãn, thu thập dữ liệu chuyên biệt và huấn luyện lại toàn bộ mô hình từ đầu.
- **Thiếu tương tác Người–Vật thể theo thời gian:** Phần lớn mô hình chỉ phân loại hành động đơn lẻ hoặc nhận diện đối tượng tĩnh, không nắm bắt được mối quan hệ động (quỹ đạo tương đối, vận tốc, biến thiên vị trí) giữa Người và Vật qua chuỗi thời gian.
- **Nhầm lẫn giữa Hành vi và Phán quyết Vi phạm:** Một hành vi (ví dụ: dừng xe, cầm đồ vật) bản thân nó không phải là vi phạm; hành vi đó chỉ trở thành vi phạm khi diễn ra tại khu vực cấm (vùng không gian) và kéo dài vượt quá ngưỡng thời gian cho phép.

---

## 2. Hai Vấn đề Nghiên cứu & Phát triển Cốt lõi (Core R&D Problems)

Dự án tập trung giải quyết **hai vấn đề khoa học và kỹ thuật trọng tâm**:

### Vấn đề 1: Học biểu diễn tương tác Người + Vật thể từ một tập dữ liệu tổng quát cho trước (General HOI Representation Learning)
- **Bản chất vấn đề:** Thay vì huấn luyện mỗi hành vi tương tác trên một tập dữ liệu đặc thù riêng lẻ (task-specific training per behavior) — điều gây tốn kém tài nguyên và không có tính khái quát, đề tài nghiên cứu phương pháp **mô hình hóa và học biểu diễn tương tác Người + Vật thể theo không–thời gian (Spatio-Temporal HOI)** từ một tập dữ liệu chung/tổng quát cho trước.
- **Mục tiêu nghiên cứu:** Xây dựng cơ chế trích xuất đặc trưng ngoại hình (Appearance), hình học và chuyển động tương đối (Relative Geometry & Motion Trajectory) của cặp Người–Vật qua chuỗi khung hình (Tubelets / Pair Windows), kết hợp kỹ thuật nén thời gian (Temporal Sampling / Temporal Attention Pooling) để tạo ra vector đại diện hành vi $Z_{\text{video}}$ có khả năng khái quát hóa cho nhiều loại tương tác khác nhau mà không cần huấn luyện lại từ đầu cho từng hành vi hẹp.

### Vấn đề 2: Ánh xạ và so khớp vector biểu diễn ngữ nghĩa giữa Video và Văn bản (Cross-Modal Embedding Mapping & Matching)
- **Bản chất vấn đề:** Tồn tại khoảng cách ngữ nghĩa lớn (Semantic Gap) giữa dữ liệu chuỗi thị giác động trong video và mô tả quy tắc an toàn bằng văn bản tự nhiên. Các mô hình nền tảng thị giác–ngôn ngữ (như CLIP) vốn được tiền huấn luyện trên cặp ảnh tĩnh–văn bản, chưa được tối ưu hóa cho tương tác có chiều thời gian và cấu trúc hành vi phức tạp.
- **Mục tiêu nghiên cứu:** Nghiên cứu cơ chế **ánh xạ (mapping / projection)** và **căn chỉnh (cross-modal alignment)** giữa hai vector biểu diễn ngữ nghĩa:
  - Vector biểu diễn video $Z_{\text{video}}$: biểu diễn tương tác người–vật thể nén qua thời gian.
  - Vector biểu diễn văn bản $Z_{\text{text}}$: trích xuất từ mô tả quy tắc ngôn ngữ tự nhiên thông qua bộ mã hóa văn bản (Text Encoder).
  - Chiếu hai vector về một **không gian ngữ nghĩa chung (Shared Semantic Embedding Space)** và sử dụng hàm đo khoảng cách/độ tương đồng (Cosine Similarity / Metric Learning) để thực hiện so khớp mở (*zero-shot / open-vocabulary matching*), cho phép nhận biết hành vi tương ứng với quy tắc văn bản mà không bị ràng buộc vào các lớp phân loại đóng.

---

## 3. Các Bên Liên quan (Stakeholders)

| Bên liên quan | Vai trò & Trách nhiệm | Nhu cầu chính & Tiêu chí mong đợi |
|---|---|---|
| **Người nghiên cứu** | Trực tiếp thiết kế, triển khai mô hình, xây dựng benchmark và thực nghiệm | Mã nguồn module hóa, pipeline thử nghiệm tái lập 100%, bộ đo lường định lượng minh bạch và theo dõi ca lỗi |
| **Hội đồng Khoa học & Giảng viên** | Đánh giá tính mới, phương pháp luận và độ tin cậy học thuật | Đối chứng thực nghiệm khoa học (ablation matrix), phân tích đóng góp rõ ràng giữa phần kế thừa và phần tự phát triển, bằng chứng thực nghiệm khách quan |
| **Người đánh giá / Người vận hành** | Thử nghiệm áp dụng quy tắc lên các kịch bản video giám sát | Khai báo quy tắc bằng ngôn ngữ tự nhiên, chọn vùng đa giác (Polygon ROI), xem lại sự kiện kèm bằng chứng trực quan (BBox, Track ID, Timestamp, Clip bằng chứng) |

---

## 4. Phạm vi Dự án (Project Scope)

### 4.1. Trong phạm vi (In-Scope)
- **Môi trường & Dữ liệu:** Video giám sát từ camera cố định (CCTV), độ phân giải tiêu chuẩn (từ 640x340 đến 1100x720, 12–25 fps).
- **Mô hình hóa Tương tác (HOI):** Phát hiện và bám vết Người và Vật thể trong từng khung hình, tạo cửa sổ cặp tương tác (Pair Windows), trích xuất đặc trưng vùng bao chung (Union Crop) và quỹ đạo chuyển động.
- **Mã hóa Quy tắc An toàn:** Phân rã câu quy tắc tiếng Việt/tiếng Anh theo cấu trúc chuẩn hóa: `<Subject, Action, Object, Zone, Min Duration>`.
- **Ánh xạ Đa phương thức:** Cơ chế chiếu vector video $Z_{\text{video}}$ và vector văn bản $Z_{\text{text}}$ vào không gian chung, đo độ tương đồng ngữ nghĩa.
- **Suy luận Vi phạm Có căn cứ (Grounded Verification):** Kết hợp điểm so khớp hành vi với bộ lọc không gian (Spatial Polygon ROI) và bộ lọc thời lượng duy trì (Min Dwell Time) để xuất sự kiện vi phạm kèm bằng chứng truy vết.
- **Đánh giá Khoa học:** Đánh giá trên bộ benchmark chuẩn hóa với siêu dữ liệu công khai (`data/manifests/`), đo lường Recall@K, Pairwise Accuracy, MRR và Event-F1 theo ngưỡng $\text{tIoU} \ge 0.5$.

### 4.2. Ngoài phạm vi (Out-of-Scope)
- Không huấn luyện lại các mô hình nền tảng từ đầu (như pretraining toàn bộ CLIP hay Detector).
- Không giải quyết bài toán camera chuyển động phức tạp (PTZ) hoặc mạng lưới đa camera (Multi-camera Tracking).
- Không định danh danh tính cá nhân (Re-ID / Face Recognition) hoặc truy vết quyền hạn riêng tư cá nhân.
- Không nhận diện ngôn ngữ tự nhiên tùy ý không kiểm soát cấu trúc (unconstrained natural language).
- Không yêu cầu xử lý thời gian thực tuyệt đối (Real-time Streaming); hệ thống ưu tiên độ chính xác và tính giải thích được cho bài toán xem lại và phân tích ngoại tuyến (*offline surveillance audit*).

---

## 5. Yêu cầu Chức năng (Functional Requirements)

| Mã FR | Tên chức năng | Mô tả chi tiết | Tiêu chí nghiệm thu |
|---|---|---|---|
| **FR01** | Đọc & Chuẩn hóa Video | Tiếp nhận video giám sát, giải mã khung hình theo thời gian thực tế, lấy mẫu đồng đều ($T=8$ khung hình) và kiểm tra tính toàn vẹn metadata | Video đọc đúng timestamp, không tạo khung đen giả khi lỗi; báo lỗi rõ ràng nếu file hỏng |
| **FR02** | Khai báo & Phân rã Quy tắc | Tiếp nhận câu quy tắc an toàn và phân tích thành tuple cấu trúc: `<Chủ thể, Hành động, Đối tượng, Khu vực, Thời lượng>` | Câu quy chuẩn phân rã đúng các trường; câu sai cấu trúc bị từ chối kèm thông báo hướng dẫn |
| **FR03** | Khai báo Vùng Không gian | Cho phép định nghĩa vùng kiểm tra bằng tọa độ đa giác chuẩn hóa (Normalized Polygon ROI) | Kiểm tra điểm neo của đối tượng thuộc/nằm ngoài vùng; xử lý đa giác lồi và lõm |
| **FR04** | Nhận diện & Bám vết | Phát hiện vị trí Người và Vật thể trong khung hình, liên kết quỹ đạo (tubelets) qua thời gian | Cung cấp Bounding Box, Track ID và confidence score; không gán ID trùng lặp |
| **FR05** | Thiết lập Cặp Tương tác | Ghép nối các cặp Người–Vật trong cửa sổ thời gian (Pair Windows), trích xuất Union Crop và vector chuyển động | Đúng mốc thời gian; có valid mask đối với các khung hình bị che khuất |
| **FR06** | Ánh xạ & So khớp Đa phương thức | Chiếu biểu diễn video $Z_{\text{video}}$ và biểu diễn text $Z_{\text{text}}$ vào không gian chung; tính khoảng cách Cosine | Điểm số tương quan chuẩn hóa trong khoảng $[-1, 1]$; hỗ trợ so khớp đa nhãn (*multi-positive*) |
| **FR07** | Phán quyết Vi phạm Có căn cứ | Kết hợp điểm tương quan ngữ nghĩa với điều kiện điểm neo trong vùng ROI và thời lượng duy trì tích lũy | Sự kiện vi phạm chỉ phát sinh khi thỏa mãn đồng thời: Hành vi khớp + Đúng vùng + Đủ thời lượng |
| **FR08** | Xuất Sự kiện & Bằng chứng | Trích xuất sự kiện vi phạm gồm Bounding Box, Track ID, Rule ID, khoảng thời gian $[t_{\text{start}}, t_{\text{end}}]$ và video clip bằng chứng | Xuất file JSON/CSV chuẩn hóa; đoạn clip bằng chứng mở đúng khoảng thời gian vi phạm |
| **FR09** | Xử lý Thiếu Bằng chứng | Ghi nhận trạng thái `insufficient_evidence` khi đối tượng bị che khuất hoặc mất dấu quá ngưỡng cho phép | Không tự ý quy đổi trạng thái thiếu quan sát thành âm tính (không vi phạm) chắc chắn |

---

## 6. Yêu cầu Phi chức năng & Chất lượng Nghiên cứu (Non-Functional Requirements)

1. **Tính Tái lập Khoa học (Scientific Reproducibility):**
   - Môi trường tính toán và các gói phụ thuộc được khóa chặt chẽ thông qua `uv.lock`.
   - Toàn bộ quá trình đánh giá và benchmark được cấu hình qua file cấu hình JSON/YAML, cố định seed ngẫu nhiên, gắn liền với commit Git và hash của tập dữ liệu.
2. **Khả năng Giải thích & Minh chứng (Explainability & Grounding):**
   - Mọi phán quyết vi phạm phải đi kèm bằng chứng trực quan cụ thể (vị trí hộp bao, định danh vệt chuyển động, biểu đồ thời gian).
   - Hệ thống không đưa ra cảnh báo dưới dạng một "hộp đen" chỉ có nhãn nhị phân.
3. **Tính Ổn định & Độ tin cậy (Robustness):**
   - Xử lý mượt mà các trường hợp biên: video không có người, video không có tương tác, vật thể bị che khuất một phần, đối tượng di chuyển nhanh qua vùng cấm mà chưa đủ thời lượng duy trì.
4. **Bảo mật & Quyền riêng tư Dữ liệu:**
   - Hoạt động hoàn toàn cục bộ (*local execution*), không gửi dữ liệu video giám sát nhạy cảm lên các dịch vụ đám mây công cộng bên ngoài.

---

## 7. Kiến trúc Hệ thống & Ranh giới Mô-đun (System Architecture)

Hệ thống được tổ chức thành 3 tầng chức năng độc lập:

```text
[Video Input] ──> [Tầng 1: Perception & Tracking] ──> [Pair Windows & Tubelets]
                                                               │
                                                               ▼
[Text Rules]  ──> [Module NLP: Rule Parser]        [Tầng 2: Spatio-Temporal HOI]
                           │                                   │
                           ▼                                   ▼
                   [Z_text Vector]                    [Z_video Vector]
                           │                                   │
                           └───────────────┬───────────────────┘
                                           ▼
                    [Tầng 3: Cross-Modal Alignment & Matching]
                                           │
                                           ▼
                               [Matching Similarity]
                                           │
                       [Spatial Polygon ROI + Min Dwell Time]
                                           │
                                           ▼
                    [Sự kiện Vi phạm có Bằng chứng Grounding]
```

### Chi tiết các phân hệ:
- **Phân hệ Thị giác (Vision Subsystem):** Đóng gói trong `temporal_hoi.data`. Thực hiện giải mã video, lấy mẫu đồng đều $T$ khung hình, trích xuất đặc trưng ngoại hình và chuyển động.
- **Phân hệ Ngôn ngữ (Language Subsystem):** Tiếp nhận quy tắc, phân tích cú pháp thành schema chuẩn, sử dụng Text Encoder tiền huấn luyện để trích xuất vector ngữ nghĩa.
- **Phân hệ So khớp & Quyết định (Matching & Grounded Decision Subsystem):** Đóng gói trong `temporal_hoi.evaluation` và pipeline suy luận. Ánh xạ vector embedding, tính điểm tương đồng, kiểm tra điều kiện không–thời gian và phát sinh cảnh báo có bằng chứng.

---

## 8. Mô hình Dữ liệu & Cấu trúc Siêu dữ liệu (Data Schemas)

Dữ liệu được tổ chức chuẩn hóa trong thư mục `data/manifests/` với các thực thể chính:

| Thực thể | File lưu trữ | Các trường dữ liệu cốt lõi |
|---|---|---|
| **Video Clip** | `clips.csv` | `clip_id`, `video_id`, `camera_id`, `resolution`, `fps`, `start_s`, `end_s`, `behavior_label`, `has_violation` |
| **Quy tắc Văn bản** | `texts.csv` | `text_id`, `text_en`, `text_vi`, `subject`, `action`, `object`, `zone_type`, `is_violation_rule` |
| **Phân chia Tập** | `splits.csv` | `clip_id`, `session_id`, `split` (`dev` hoặc `extension`) |
| **Tương quan Clip–Text** | `relevance.csv` | `clip_id`, `text_id`, `relevance` (`1` là phù hợp, `0` là không phù hợp, rỗng là chưa xác định) |
| **Nguồn gốc & Bản quyền** | `sources.csv` | `source_id`, `clip_id`, `license`, `provenance_url`, `sha256_checksum` |

---

## 9. Tiêu chuẩn Đánh giá & Độ đo Nghiên cứu (Evaluation Standards)

### 9.1. Bài toán So khớp Video–Văn bản (Video–Text Matching)
Đánh giá năng lực của không gian biểu diễn đa phương thức trong việc liên kết đúng hành vi video với mô tả văn bản tương ứng:
- **Multi-positive Recall@K ($K=1, 3, 5$):** Tỷ lệ các câu truy vấn đúng được xếp hạng trong top-$K$ kết quả trả về khi một clip có thể khớp với nhiều câu mô tả hợp lệ.
- **Pairwise Accuracy:** Tỷ lệ các cặp (Clip, Text Đúng) có điểm số tương đồng cao hơn cặp đối chứng (Clip, Text Sai).
- **Mean Reciprocal Rank (MRR):** Giá trị nghịch đảo thứ hạng trung bình của kết quả đúng đầu tiên.

### 9.2. Bài toán Phát hiện Sự kiện Vi phạm Thời gian (Temporal Event Detection)
Đánh giá năng lực phát hiện chính xác thời điểm và loại vi phạm diễn ra trong video:
- **Temporal IoU (tIoU):** Đo lường mức độ trùng khớp giữa khoảng thời gian dự đoán $[t_{\text{pred\_start}}, t_{\text{pred\_end}}]$ và khoảng thời gian thực tế $[t_{\text{gt\_start}}, t_{\text{gt\_end}}]$:
  $$\text{tIoU} = \frac{|I_{\text{pred}} \cap I_{\text{gt}}|}{|I_{\text{pred}} \cup I_{\text{gt}}|}$$
- **Greedy Bipartite Matching (Ngưỡng $\text{tIoU} \ge 0.5$):** Khớp 1-to-1 giữa sự kiện dự đoán và nhãn thực tế cùng loại luật. Các dự đoán lặp lại (duplicate predictions) hoặc ngoài khoảng vi phạm bị phạt là False Positive (FP).
- **Event-Precision, Event-Recall, Event-F1:** Đánh giá mức độ chính xác và độ bao phủ của các sự kiện phát hiện được.
- **False Alarms / Hour (Tỷ lệ báo động giả mỗi giờ):** Số lượng cảnh báo sai phát sinh trên các đoạn video hoạt động bình thường, đo lường độ tin cậy thực tế của hệ thống.

---

## 10. Rủi ro Kỹ thuật & Biện pháp Giảm thiểu (Technical Risks & Mitigations)

| Rủi ro Kỹ thuật | Khả năng | Tác động | Biện pháp Giảm thiểu |
|---|---|---|---|
| **Khoảng cách Ngữ nghĩa (Cross-Modal Gap):** Vector video và vector text không căn chỉnh tốt khi chỉ dùng mô hình zero-shot tĩnh | Trung bình | Cao | Thử nghiệm lớp chiếu adapter nhẹ; kết hợp thêm đặc trưng hình học chuyển động để bổ trợ cho đặc trưng ngoại hình |
| **Nhiễu từ Bám vết Đối tượng (Tracking Failures):** Track ID bị đứt gãy hoặc nhảy vệt do che khuất | Cao | Trung bình | Tách biệt đánh giá trên quỹ đạo chuẩn (Oracle Tracks) và quỹ đạo dự đoán (Predicted Tracks); ghi nhận trạng thái `insufficient_evidence` |
| **Báo động giả do Ngưỡng nhạy cảm:** Báo vi phạm ngay khi đối tượng chỉ vừa đi lướt qua vùng cấm | Trung bình | Cao | Áp dụng cơ chế làm mịn thời gian (Temporal Dwell Smoothing) với ngưỡng thời lượng tối thiểu bắt buộc |
| **Rò rỉ Dữ liệu (Data Leakage):** Trùng lặp bối cảnh/phiên quay giữa tập phát triển và tập đánh giá | Thấp | Cao | Khóa chặt giao thức phân chia split theo `session_id` và `camera_id` gốc trong `splits.csv` |

---

## 11. Hướng phát triển Mở rộng (Future Extensions)

- **Mô hình Đồ thị Tương tác Động (Dynamic HOI Scene Graphs):** Mở rộng từ mô hình hóa cặp đơn lẻ sang đồ thị tương tác đa người – đa vật thể theo thời gian.
- **Căn chỉnh Không gian–Thời gian Mịn (Fine-Grained Spatio-Temporal Grounding):** Định vị chính xác vùng hộp bao của hành vi vi phạm ở cấp độ pixel-level hoặc spatial attention maps.
- **Tổng quát hóa Cấu trúc Luật (Compositional Rule Generalization):** Hỗ trợ các quy tắc an toàn có cấu trúc logic phức tạp hơn (điều kiện phủ định, logic kết hợp AND/OR giữa nhiều hành động).
