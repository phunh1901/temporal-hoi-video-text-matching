# W03-02 — Giao thức MSR-VTT khóa ngày 03/10/2026

W03 tạo một hợp đồng dữ liệu và đánh giá có thể tái kiểm tra. Đã thu nhận **metadata caption/split** và sửa validation sang **v2 tách nhóm URL nguồn**; chưa tải video MSR-VTT hoặc weights, chưa chạy inference hay huấn luyện benchmark. CLIP4Clip meanP là mốc lịch sử (2021). InternVideo2 Stage2-1B (2024) là ứng viên hiện đại có mã và metadata weights đã kiểm toán, không được gọi là mô hình mới nhất hoặc SOTA hiện hành năm 2026.

## 1. Khóa dữ liệu bằng artifact thực

Nguồn là [release metadata CLIP4Clip v0.0](https://github.com/ArrowLuo/CLIP4Clip/releases/tag/v0.0), được hướng dẫn trong [README tại commit khóa](https://github.com/ArrowLuo/CLIP4Clip/blob/508ffa3de39ba0563a03199c440ab602a72e9b6f/README.md). Zip thực tải 4.067.617 byte (3,879 MiB), SHA256 `93958a5f1c322b32349d0948eba0cb24c25a34d54f45bd5e037027d266ea6e56`. Metadata và code snapshot lưu ở `outputs/w03/sources/m2/`; file nguyên bản được giữ riêng khỏi dữ liệu pilot.

| Phần | Video | Caption | Vai trò |
|---|---:|---:|---|
| MSRVTT_train.9k.csv | 9.000 | 180.000 | Train tham chiếu upstream |
| Research train v2 | 8.100 | 162.000 | Train head/đối chứng nội bộ; tách nhóm URL nguồn |
| Research validation v2 | 900 | 18.000 | Chọn cấu hình/checkpoint từ train giữ lại; không chung nguồn với train |
| MSRVTT_JSFUSION_test.csv | 1.000 | 1.000 câu CSV | Final evaluation chuẩn 1k |
| Toàn bộ annotation của 1.000 test video | 1.000 | 20.000 | Extension nhiều caption, báo bảng riêng |

`MSRVTT_data.json` thực có 10.000 video và 200.000 caption, mỗi video 20 caption. V1 danh nghĩa chia mức `video_id`: xáo trộn **thứ tự ID upstream** bằng `random.Random(42).shuffle`, chọn 900 ID đầu làm validation, lưu mỗi danh sách con theo thứ tự gốc. V1 được giữ cho kiểm toán, không được chọn cho nghiên cứu head. Danh sách ID đầy đủ của v1/v2, candidate order, positive mapping, hash các file gốc và hash mapping caption ở [metadata lock](../../outputs/w03/msrvtt_protocol_lock.json). Thuật toán và hash đã khóa, không suy từ tên seed rằng mọi thư viện chia split sẽ cho cùng danh sách.

**Lỗi và sửa trong W03-06:** 10.000 `video_id` có **7.180 URL nguồn khác nhau**; một URL có thể cung cấp nhiều clip/video_id. Có **404 URL chung giữa research8100 và validation900 v1**. V2 gom toàn bộ `video_id` trong official train9k theo exactURL nguồn; tạo nhóm theo lần xuất hiện đầu trong ID order, xáo trộn nhóm bằng seed42, nhận nguyên nhóm nếu vừa số video còn thiếu tới900. Tập này đạt **8.100 train/900 validation**, **0 URL và0video_id chung giữa train và validation**. Tất cả20caption/video theo cùng nhóm. Không chia nhóm để ép đủ quota; script báo lỗi nếu quota không thể đạt bằng thuật toán đã khóa. V2 là split phát triển đã sửa và được chọn cho nghiên cứu head; baseline/treatment nội bộ phải dùng cùng v2. Không có training nào đã dùng v1.

**Giới hạn final benchmark còn lại:** official9k và test1k có **341 URL nguồn chung**. Giữ official split/candidate pool nguyên vẹn để đối chiếu benchmark, đồng thời ghi rõ test chỉ độc lập ở `video_id`, không độc lập hoàn toàn theo video nguồn. V2 sửa model-selection train/validation, không xóa được sự phụ thuộc nguồn đã có trong benchmark chính thức. Nếu cần nghiên cứu final evaluation độc lập theo nguồn thì phải thêm protocol riêng, công bố số video sau lọc và không gọi kết quả đó là benchmark9k/1k nguyên bản.

**Hai lỗi dễ làm sai mapping đã phát hiện:** có 14.632 dòng caption trùng `(video_id, caption)` vượt ngoài lần xuất hiện đầu; toàn corpus có 33.081 lần lặp chuỗi caption. Giữ tất cả dòng/ID gốc, không âm thầm khử trùng hoặc suy rằng chuỗi giống nhau là cùng video. Có **35 câu test CSV không khớp nguyên văn** một caption của video tương ứng trong JSON đầy đủ. Giữ câu CSV làm query chuẩn, cấp ID riêng `msrvtt_jsfusion_0000...0999`, lưu cờ mismatch; không thay bằng câu gần giống. Annotation caption dùng ID `msrvtt_sen_<sen_id>`.

| Thành phần khóa | SHA256 canonical JSON UTF-8 |
|---|---|
| 9k train ID order | `872783782b69a7143620be1a7ffa8ad76b6197a75c4e6075a193a781b8e0ed86` |
| 8.1k research train v1 retained | `85c26d374abf2424a12aefd782e5d3f39a66153843832a314b8f9dbbf289101c` |
| 900 research validation v1 retained | `6120e78930dd7015574362565d923e7216dca24d006e0711ad1ddc5b56f07de3` |
| 8.1k source-group train v2 selected | `722482c1fb6e560e594134ecf076a997701fbc0d10f5f0a7800ba17c0f5b1e27` |
| 900 source-group validation v2 selected | `d7ace5316b94212bd0fc58cc913d4137f8b324368632ea518e34e62d5343a900` |
| 1k test/candidate video order | `305d4c627da683d8a34009a1ba2cf44d420e8dbc783be2d3ab63033a8be7a814` |
| Toàn bộ caption mapping | `c863a2c03edaa633870a93fc8286321f3018a8298ebe59ca444a05d7c3bcf141` |
| Query CSV/mismatch flags | `f48677f8645e403a835416a6134ce1cdf0f7080c0733bc18a85ba291ff1e2d49` |
| Query → positive video mapping | `fe5c7a728714173bbdae2b3f98f22023905cbc48af587220a346de1d25596bd4` |

Đây là khóa annotation metadata đầy đủ, **không chứng minh media coverage hoặc quyền sử dụng từng footage**. License code không tự áp dụng cho dataset/media. Nguồn dữ liệu và điều kiện dùng phải kiểm tra khi chuẩn bị W04.

## 2. Candidate pool, positive và metric hai chiều

Ma trận chuẩn có **1.000 hàng text × 1.000 cột video**, cả hai theo thứ tự CSV JSFusion. Positive của text hàng i là video cột i. Text→video có 1.000 query/1.000 candidate; video→text là chuyển vị cùng ma trận, có 1.000 query video/1.000 candidate text, một positive đã biết mỗi query. [Loader upstream](https://github.com/ArrowLuo/CLIP4Clip/blob/508ffa3de39ba0563a03199c440ab602a72e9b6f/dataloaders/dataloader_msrvtt_retrieval.py) đọc câu CSV test trực tiếp; [main evaluator](https://github.com/ArrowLuo/CLIP4Clip/blob/508ffa3de39ba0563a03199c440ab602a72e9b6f/main_task_retrieval.py) chấm cả ma trận và chuyển vị.

R@1/5/10 và MedR là chính; MRR là bổ trợ. Recall upstream trả **%**, API dự án trả **tỷ lệ 0–1**; đổi thang trước đối chiếu. Giữ evaluator tham chiếu khi gọi reproduction. [compute_metrics upstream](https://github.com/ArrowLuo/CLIP4Clip/blob/508ffa3de39ba0563a03199c440ab602a72e9b6f/metrics.py) tìm mọi vị trí có score bằng score positive, nên ties có thể tạo nhiều hạng cho một query; API nội bộ ưu tiên candidate order ổn định. Trường hợp có ties phải ghi hai bảng/khác biệt, không đổi evaluator rồi báo giống chuẩn.

Extension nhiều caption dùng 20.000 text→1.000 video; chiều ngược có 20 positive/video và lấy hạng relevant đầu tiên. Positive dựa trên `video_id`, không dựa trên ngữ nghĩa do người thực nghiệm tự đoán. Pool này khác phép đo 1.000 câu CSV: không đưa metric extension vào cột official 1k. Caption tiếng Việt/dịch hoặc 8 clip pilot cũng là thí nghiệm riêng.

## 3. CLIP4Clip meanP: lịch sử và preprocessing tham chiếu

Khóa commit `508ffa3de39ba0563a03199c440ab602a72e9b6f`, code MIT. ViT-B/32, patch2d, `loose_type`, `sim_header=meanP`, `max_words=32`, `max_frames=12`, `feature_framerate=1`, `slice_framepos=2`, frame order0 là cấu hình tham chiếu đã đọc từ source. [Video preprocess](https://github.com/ArrowLuo/CLIP4Clip/blob/508ffa3de39ba0563a03199c440ab602a72e9b6f/dataloaders/rawvideo_util.py): RGB, resize bicubic cạnh ngắn224, center crop224; normalize mean `(0.48145466,0.4578275,0.40821073)` và std `(0.26862954,0.26130258,0.27577711)`. Sample1 frame/giây trước khi lấy đều tối đa12 frame. Khi thiếu frame, pad và valid mask; reproduction phải giữ quy ước nguồn. Reader pilot 8 frame/max-side320 không mặc nhiên tương đương.

[Pooling model](https://github.com/ArrowLuo/CLIP4Clip/blob/508ffa3de39ba0563a03199c440ab602a72e9b6f/modules/modeling.py): L2 normalize vector mỗi frame, mean các frame valid, normalize vector video; text là CLIP text output rồi normalize; score là dot product nhân `exp(logit_scale)`. MeanP không thêm temporal head, nhưng CLIP backbone có thể được finetune; dùng CLIP pretrained/OpenCLIP LAION cache không phải checkpoint MSR-VTT meanP đã finetune.

URL OpenAI CLIP ViT-B/32 backbone được kiểm tra bằng HEAD200, **353.976.522 byte (~337,578 MiB)**, không tải. Chưa xác minh được checkpoint fine-tuned MSR-VTT meanP trong các artifact đã rà. Ví dụ training nguồn dùng 4GPU, batch128,5epoch, PyTorch1.7.1/CUDA11.0; môi trường CPU Python3.12 hiện tại khác, không ép dependency baseline vào môi trường pilot. Không có RAM/VRAM inference/training đo thực cho baseline.

**Sửa nguy cơ test leakage:** upstream `DATALOADER_DICT['msrvtt']` có `test=None`; `val_csv` trỏ test CSV. Main gọi `eval_epoch(...test_dataloader...)` sau mỗi epoch và chọn best theo testR@1 (dòng566–569 tại commit khóa). Dự án giữ official test làm final evaluation; model selection dùng 900 video train giữ lại theo source-groupv2. Khi so checkpoint học trên9k với head học trên8.1k, phải nêu khác biệt supervision; control/treatment nội bộ đều dùng8.1k/900v2. Không chạy nguyên ví dụ `--do_train` rồi coi validation hợp lệ.

## 4. Ứng viên hiện đại và khảo sát tới ngày rà

| Phương pháp | Artifact và tính phù hợp | Quyết định W03 |
|---|---|---|
| InternVideo2 Stage2-1B,2024 | [Paper](https://arxiv.org/abs/2403.15377), [model zoo retrieval](https://github.com/OpenGVLab/InternVideo/blob/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo2/multi_modality/MODEL_ZOO.md), code/eval1k/weights có metadata thực | Chọn ứng viên đã audit; zero-shot checkpoint, không so như cùng supervision9k |
| InternVideo2.5,2025 | [Nguồn tác giả](https://github.com/OpenGVLab/InternVideo/blob/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo2.5/README.md) có weights8B MLLM/LRC; nhiệm vụ video understanding | Không tìm bằng chứng evaluator MSR-VTT9k/1k trong nguồn đã rà |
| InternVideo-Next,2025/CVPR2026 | [README nguồn](https://github.com/OpenGVLab/InternVideo/blob/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo-Next/README.md) dẫn pretrained models/code; mục tiêu video foundation không video-text supervision | Chưa audit checkpoint retrieval/same candidate pool; không nhận thay baseline |
| InternVideo3,2026 | [Nguồn tác giả](https://github.com/OpenGVLab/InternVideo/blob/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo3/README.md):8B instruct, QA/temporal grounding/agentic reasoning, weights công bố | Chưa có bằng chứng retrieval1k cùng evaluator trong nguồn đã rà |

Toàn repo hiện đại khóa commit `3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3`, Apache2.0. Rà các phiên bản mới hơn giúp tránh gọi bản2024 là mới nhất. Bảng này là khảo sát artifact theo phạm vi, không phải tổng hợp mọi paper/SOTA hoặc tuyên bố không tồn tại phương pháp phù hợp khác.

[Hugging Face metadata](https://huggingface.co/api/models/OpenGVLab/InternVideo2-Stage2_1B-224p-f4?blobs=true) xác nhận revision `4362e1f88a992e7edbfd7696f7f78b7f79426dfd`, file `InternVideo2-stage2_1b-224p-f4.pt`, **2.820.610.931 byte (~2,627GiB)**, LFS SHA256 `df8cdbe9c3c9f65fbba777afc9ee6fbf58e6d68c4aec2827cac8e2a98d0a8446`. Model là `gated=auto`; chỉ metadata công khai đã đọc, chưa kiểm tra quyền tải của người dùng/chưa chấp nhận điều khoản/chưa tải weights. Hash này là **upstream advertised**, không là hash file đã tải cục bộ.

[Cấu hình eval1k](https://github.com/OpenGVLab/InternVideo/blob/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo2/multi_modality/scripts/evaluation/stage2/zero_shot/1B/config_msrvtt.py) dùng4frame/224, BERT-large,max text40, embedding512, CUDA/BF16, FlashAttention/fusedMLP/RMSNorm, DeepSpeed; script xin1GPU/16CPU. [Requirements](https://github.com/OpenGVLab/InternVideo/blob/3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3/InternVideo2/multi_modality/requirements.txt) khóa torch1.13.1+cu117/flash_attn2.0.8/deepspeed0.10.1. Config chấm contrastive và video-text matching với `k_test=128`, `eval_x_only=False`; hai cách score phải được ghi tách.

Upstream gọi split `msrvtt_1k_test` và dùng JSON VINDLU (Google Drive), nhưng byte identity của JSON này với JSFusionCSV **chưa xác minh**: corpus path còn placeholder. W04 phải chuyển **1.000 row CSV đã khóa** sang JSON adapter, kiểm tra ID/query/positive hashes, rồi giữ preprocessing/evaluator từng model. Chưa thể đối chiếu điểm công bố chỉ vì tên split chứa1k. Zero-shot modern và supervised historical có supervision khác nhau; không tính chênh lệch điểm giữa chúng là cải tiến của dự án.

## 5. Mức thực hiện và tái kiểm tra

W03 đạt **metadata/protocol lock + audit evaluator** trên CPU. W04 inference và baseline training hiện `not_run`: thiếu video benchmark, chưa có checkpoint historical fine-tuned, modern weights gated và compute CUDA không có. Snapshot resource thực nằm trong [resource feasibility](../../outputs/w03/resource_feasibility.json); metadata weights không chứng minh peakRAM/VRAM hoặc thời gian inference. Không tự mua GPU hoặc tải model lớn trong lượt này.

Chạy lại phần khóa: `.venv/Scripts/python.exe scripts/lock_msrvtt_protocol.py` với zip nhỏ đã tải; không phát sinh network/model/media. Test chống trùngID, train/test overlap, caption thiếu/ID lạ/ID caption trùng, empty query, duplicate-string preservation, split seed42, source-group disjointness và thiếu provenance/quota không thể đạt đã **11 passed**. [Cấu hình](../../configs/w03_msrvtt_protocol.json), [script](../../scripts/lock_msrvtt_protocol.py), [test](../../tests/test_msrvtt_protocol.py) và [source audit](../../outputs/w03/m2_sources.json) là đầu ra W03-02/W03-06. Đây là kiểm chứng kỹ thuật, chưa là điểm R@K/MedR của mô hình.
