# M1 — Giao thức VidHOI W03, phiên bản 1

**Ngày rà:** 03/10/2026 (+07). **Mã việc:** W03-01; phần tài nguyên liên quan W03-03. Tài liệu này phân biệt quyết định đã khóa với artifact còn thiếu; chưa xác nhận một lần chạy mAP, inference hoặc training.

## 1. Kết luận khảo sát và quyết định

Đã tìm được **ACoLP, ICCV 2023**, một công bố open-set video HOI trên VidHOI, giữ lại 20% hoặc 50% **predicate** khi huấn luyện. Vì vậy không còn diễn đạt khoảng trống là “chưa tìm thấy công bố zero-shot VidHOI”. Tuy nhiên, chưa xác minh được **baseline zero-shot có bộ artifact đầy đủ để chạy lại**. Open-set trong bài định nghĩa trước danh mục hành động/HOI; đây không phải open-world không giới hạn. [Bài tác giả, §3.1 và §4](https://cse.buffalo.edu/~jsyuan/papers/2023/ICCV2023_nan.pdf).

| Phương pháp | Phạm vi đã kiểm tra | Mã/commit đã khóa | Artifact, giấy phép, tài nguyên | Quyết định W03 |
|---|---|---|---|---|
| ST-HOI, ICMRW 2021 | Video, VidHOI; mốc lịch sử closed-set | `coldmanck/VidHOI@c906d7ece7c6db423742b35859e0126373422595`, commit 04/09/2022 | Code Apache-2.0. Repo ghi checkpoint bị gỡ 21/08/2022. Có liên kết folder predictions oracle, nhưng chưa xác minh file/hash. Python 3.6, PyTorch 1.4.0, torchvision 0.5.0; SlowFast và detectron2 bản kèm repo. Tác giả ghi training 8×V100 32 GB, validation batch 1 dùng 1 GPU <4 GB. | Chọn **evaluator-only làm bước kế tiếp W04**, còn chặn bởi prediction và frame mask; inference/training chưa chạy. |
| ACoLP, ICCV 2023 | Video open-set trên VidHOI; unseen-predicate 20%/50% | `southnx/ACoLP@d4d299ee553b875206497e4dc3c806beb5ffb5cf`, commit 03/10/2023 | Không có LICENSE/requirements/evaluator/checkpoint/release trong tree đã rà. Entrypoint nhập ba module không tồn tại; model thiếu `models/mlp.py`, cần prompt/caption features ngoài repo. Paper: frozen Faster R-CNN COCO + CLIP; AdamW, 100 epochs, 4 GPUs, batch 128. Model code có CLIP ViT-L/14, timm, lavis. | Ứng viên **protocol**, chưa đủ điều kiện chọn cho inference/reproduction. |
| SAGE, arXiv 04/07/2026 | Video HOI/gaze detection và anticipation; có VidHOI | Chưa có code/commit/checkpoint xác minh được | Bài mô tả supervised detection/anticipation, không công bố split unseen-predicate trong phần đã đọc. Có nguồn project do Honda dẫn; trang project không truy cập được qua công cụ rà. Paper dùng RTX A6000 cho đo thời gian. | Có tính hiện đại trong video; chưa đáp ứng zero-shot và artifact tương thích. Không lấy điểm oracle so với detected ST-HOI. |
| HOI-DA, arXiv 12/04/2026 | Video detection/anticipation, DETAnt-HOI có điều chỉnh thời gian từ VidHOI/Action Genome | Chưa có code/commit/checkpoint xác minh được | Bài ghi code sẽ công bố; protocol DETAnt-HOI thay clip construction. Chưa xác minh split zero-shot. | Tài liệu liên quan; không thay VidHOI hiện hành bằng benchmark đã đổi protocol. |
| SL-HOI, CVPR 2026 | Open-vocabulary **ảnh**, SWiG-HOI/HICO-DET | `MPI-Lab/SL-HOI@64483cb35639198bbea66af500d1d05f8ac1e978`, commit 01/04/2026 | Code MIT theo LICENSE (GitHub API trả NOASSERTION); DINOv3 có license riêng. [HF tree](https://huggingface.co/Thatmakes11/SL-HOI-weights/tree/main) truy cập được, hiển thị tổng 5,96 GB, chưa tải/hash binary. Python 3.10, PyTorch 2.5.1, CUDA ≥12.1; requirements còn “will be provided later”. | Tham khảo thiết kế; không có adapter/split VidHOI xác minh, không dùng thay video baseline. |

Nguồn: [ST-HOI README tại commit khóa](https://github.com/coldmanck/VidHOI/blob/c906d7ece7c6db423742b35859e0126373422595/README.md), [ACoLP tree/README](https://github.com/southnx/ACoLP/tree/d4d299ee553b875206497e4dc3c806beb5ffb5cf), [SAGE paper](https://arxiv.org/html/2607.04017v1), [HOI-DA paper](https://arxiv.org/html/2604.10397v1), [SL-HOI README](https://github.com/MPI-Lab/SL-HOI/blob/64483cb35639198bbea66af500d1d05f8ac1e978/README.md), [SL-HOI LICENSE](https://github.com/MPI-Lab/SL-HOI/blob/64483cb35639198bbea66af500d1d05f8ac1e978/LICENSE).

Khảo sát gồm các nguồn trên đến ngày rà; không tuyên bố chứng minh không có phương pháp nào khác trên toàn thế giới. Việc paper ghi “SOTA” không chứng minh mô hình này là SOTA ngày 03/10/2026 hoặc đã được dự án tái lập.

## 2. Hợp đồng đã khóa

- Dataset M1 duy nhất: **VidHOI**, video HOI; không tải thêm benchmark HOI ảnh. Annotation và media gốc bất biến. Dùng `video_folder/video_id` làm khóa video, giữ `frame_id`, timestamp và fps nguồn.
- Giữ tên upstream `train`/`val`; official `val` giữ vai trò final evaluation công khai. Model-selection lấy từ train, chia theo video, seed 42. Không đổi `val` thành test rồi che nguồn.
- Tách các run **oracle GT boxes/tracks** và **predicted boxes/tracks**. Không so mAP giữa hai run khác mask, split hoặc proposal source.
- Mức kết quả ghi riêng: `protocol_audit`, `evaluator_only`, `checkpoint_inference`, `retraining`. Mức hiện tại là `protocol_audit`; không có mAP mới.
- API trung gian dự kiến: `video_id`, `frame_id`, `timestamp_s`, `person_bbox_xyxy`, `object_bbox_xyxy`, `object_label`, `interaction_label`, `score`, `track_id` tùy chọn. Box ở tọa độ pixel frame gốc; adapter phải biến đổi cả GT/prediction sang hệ của upstream evaluator và ghi phép biến đổi. Không truyền box chuẩn hóa vào evaluator mà không đối chiếu.

Danh mục nhỏ đã lưu nguyên byte: **50 predicate và 78 object names** của ST-HOI, cùng SHA-256. Đây là danh mục VidHOI sau biến đổi, không đồng nhất với 80 object categories VidOR gốc. Thứ tự JSON chưa được tự nâng thành class ID: phải đối chiếu `idx_to_pred.pkl`/`idx_to_obj.pkl`, trường ID annotation và prediction trước chạy. Danh sách 557 HOI triplets, annotation hash và split IDs **chưa có**; config dùng `null`, không sinh ID giả.

## 3. Evaluator và frame mask

[Notebook tại commit khóa](https://github.com/coldmanck/VidHOI/blob/c906d7ece7c6db423742b35859e0126373422595/vidor_eval.ipynb) có SHA-256 `211fd893d3e36159395e6dba3535ed0f39dff46dfd3c9705adb592c6498b2e0b`. Quy ước đã đọc từ code:

- Frame-level HOI mAP, AP VOC 11-point; box người/vật cùng lớp, IoU mỗi box ≥0,5; top 100 triplets/frame.
- Score xếp hạng = tổng log score HOI, proposal người, proposal vật; score ≤0 thay bằng `1e-300`. Top-K xảy ra trước lọc tập triplet.
- Full: đặt cả hai cờ rare/non-rare `False`; notebook đang lưu rare flag `True`, không chạy mặc định rồi gọi là Full. Rare `<25`, non-rare `≥25` theo count GT trong tập result, không tự dùng train frequency.
- Chỉ triplets có GT vào tập AP; lớp có GT nhưng không prediction cho AP=0. Không tự thêm lớp không có GT.
- GT pair matched set được giữ trên toàn frame; cần kiểm tra trường hợp một pair có nhiều predicates ở W04. Global sorting dùng `numpy.argsort` mặc định; ties chưa có cam kết stable, cần khóa NumPy và đối chiếu.

[Config model và helper](https://github.com/coldmanck/VidHOI/blob/c906d7ece7c6db423742b35859e0126373422595/configs/vidor/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool-spa_conf.yaml) cho nguồn mask là `val_instances_predictions_train_small_vidor_with_pseudo_labels.pth`. Target so sánh các model có pose: **21.758 frames** theo README, chưa có manifest/hash được kiểm chứng. Các count upstream 22.967→22.808→21.758 là số mô tả, chưa phải số đã audit cục bộ. README gọi bước đầu “168 fewer”, nhưng 22.967−22.808=159; lưu nguyên bất nhất này và kiểm tra bằng ID thực ở W04. Không tự tạo mask bằng count.

Config khóa lựa chọn mask mục tiêu; trạng thái `execution_ready=false`. Run oracle-all-frames nếu cần phải có protocol ID riêng, dùng `TEST_PREDICT_BOX_LISTS=['val_frame_annots.json']` và `TEST_GT_LESS_TO_ALIGN_NONGT=False`, không so ngang với target pose mask. File prediction tác giả phải ghi rõ oracle và hash; việc tải/chấm file đó chỉ chứng minh evaluator.

## 4. Giao thức zero-shot: phần đã biết và phần chờ

ACoLP công bố **unseen-predicate**, khác với fallback **unseen-composition** của PRD. Không gọi hai split này là một. Supplemental Table 1 có danh sách tên cho 20%/50%, nhưng công cụ mở PDF gặp lỗi 403; kết quả tìm kiếm có tên lặp/alias, chưa đủ căn cứ khóa class IDs. Phải lấy đầy đủ supplemental hoặc split manifest tác giả, đối chiếu canonical names (`watch`/“look at”, `play(instrument)`/“play”, v.v.), kiểm tra tập train/unseen rời nhau, rồi mới nhận protocol ACoLP đã tái lập. [Supplement chính thức](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Xi_Open_Set_Video_ICCV_2023_supplemental.pdf).

Nếu artifact ACoLP chưa đủ, fallback nội bộ được đặc tả để W04 xây khi có annotation; **chưa tạo split**:

1. Chia train videos làm supervised-train/model-selection trước, seed 42, không có video dùng ở hai phần; official val chưa mở để tune. Lưu ID/hash từng phần.
2. Đếm `(person,predicate,object)` trên supervised-train. Chọn ứng viên unseen-composition chỉ từ thống kê train, sắp thứ tự bằng SHA-256 của seed và tên triplet; loại ứng viên nếu việc giữ lại khiến predicate hoặc object mất toàn bộ seen support. Mục tiêu 20% các composition đủ điều kiện; tỷ lệ thực tế phải báo sau lọc, không ép đủ bằng class hiếm.
3. Để lớp đủ mẫu phát triển, ứng viên cần ≥25 positive keyframes và ≥2 train videos trước lọc. Đây là tiêu chuẩn nội bộ đã chọn trước chạy, không phải chuẩn upstream. Audit lại seen support sau lọc window/video.
4. Loại khỏi supervised training mọi window chứa nhãn unseen; nếu cache hoặc ngữ cảnh frame còn chứa chúng, loại cả cache/window bị ảnh hưởng. Unseen không được dùng làm negative target. Giữ bản gốc và log dropped windows/videos.
5. Tạo pseudo-unseen riêng từ phần train để tune; danh sách này rời final unseen. Model-selection dùng held-out train; final official val chỉ mở với model/prompt/preprocess đã khóa. Mọi access phải có log.
6. Nếu không còn đủ seen/pseudo-unseen support hoặc final nhóm không có GT, ghi không đánh giá được và báo coverage; không lặng lẽ đổi split sau xem score. Báo full/seen/unseen mAP trên cùng mask với số lớp/mẫu; ghi **protocol nội bộ**, không so trực tiếp số ACoLP.

Khi dùng CLIP/detector pretrained, zero-shot chỉ phát biểu theo supervision downstream đã audit, không khẳng định pretraining chưa gặp khái niệm unseen.

## 5. Artifact gate và công việc kế tiếp

| Điều kiện | Trạng thái 03/10 | Việc cần làm |
|---|---|---|
| Repo commit, evaluator source, category bytes/hash | Đã xác minh | Giữ file nhỏ trong `outputs/w03/m1_upstream/` và provenance `m1_sources.json`. |
| Annotation/split mapping/ID hash | Chưa có | Lấy annotation + frame lists; đối chiếu mapping; không tải toàn video trước khi quyết định tài nguyên. |
| Prediction oracle tác giả | Repo công bố; folder link mở được nhưng chưa xác minh listing/binary/hash | Xác minh tên, byte size, phép tải một prediction rồi hash; nếu không được ghi blocked cho evaluator run. |
| Mask proposal/pose | Chưa có file/ID/hash | Chạy kiểm toán mask riêng trước mAP; báo mọi loại frame. |
| ST-HOI trained checkpoint | Upstream ghi bị gỡ | Chưa cam kết inference; chỉ đổi trạng thái khi có checkpoint nguồn/hàm băm hợp lệ. |
| ACoLP code/split/checkpoint/evaluator | Bộ code công khai thiếu thành phần thiết yếu | Chưa được gọi là runnable baseline; không tự viết phần thiếu rồi báo reproduction tác giả. |
| Compute M1 | Máy CPU, Intel Iris Xe, không CUDA theo inventory W03 | Training 4–8 GPU của paper chưa chạy; evaluator CPU có thể cân nhắc sau artifact gate. RAM evaluator phải đo riêng. |

[Trang dữ liệu VidOR](https://xdshang.github.io/docs/vidor.html) ghi training videos khoảng 24,5G, validation 2,9G; đây là quy mô VidOR thượng nguồn, không phải lượng VidHOI đã tải. Frame cache/weights/RAM/thời gian toàn run hiện chưa đo. Media xuất phát YFCC100M; Apache/MIT của code không tự cấp quyền footage. Cần lưu điều kiện/attribution media trước sử dụng; không tuyên bố license dữ liệu đã approved chỉ từ giấy phép repo.

**Kết quả W03-01:** hoàn thành audit nguồn và đặc tả điều kiện; phát hiện công bố open-set VidHOI và rào cản artifact cụ thể. Phần chọn baseline zero-shot chạy được và khóa annotation/mask/seen-unseen ID chưa đạt. Tiếp tục phần evaluator/adapter độc lập ở W04; trước training lớn cần quyết định baseline/phạm vi cùng người hướng dẫn theo PRD.
