# Điều khiển ComfyUI bằng Python

`comfy_agent.py` dùng Python 3.10+ và thư viện chuẩn, không cần pip, MCP hay custom node.
API tham khảo: https://docs.comfy.org/development/comfyui-server/comms_routes

## Nạp trước preset và thời gian chạy

Cell cấu hình đầu của `comfy.ipynb` có form Colab:

- `PRELOAD_MODELS`: checkbox nạp trước (mặc định tắt).
- `MODEL_PRESET`: chọn `MiniMax H3 Image` hoặc `MiniMax H3 Video (Turbo 4 bước)`.
- `MAX_SESSION_HOURS`: số giờ hẹn dừng; mặc định `0` để không hẹn giờ. Đặt `4` nếu muốn dừng sau 4 giờ.

Khi bật nạp trước, launcher đợi ComfyUI khởi động rồi chạy warm-up **1 bước** trong
chính tiến trình ComfyUI trước khi mở link ngrok. Preset ảnh chuẩn bị diffusion model,
Qwen text encoder và image VAE; preset video chuẩn bị diffusion model, text encoder,
Turbo LoRA cùng video/audio VAE. Model cần được tải trên Drive từ trước; thiếu model
thì notebook báo và vẫn mở giao diện để xử lý. Không tự tải thêm model.

Warm-up ảnh dùng 512×512; video dùng 640×352, 124 frame. Preview chỉ nằm trong temp,
không ghi ảnh/video thử vào output Drive. Một bước chỉ để khởi tạo, không phải preset
chất lượng. Phần code được nhúng sẵn, không cần upload thêm `model_preloader.py`.

Thời gian đọc model và khởi tạo được chuyển sang lúc mở phiên, không biến mất.
ComfyUI tự quản lý RAM/VRAM; đây không phải khóa toàn bộ model trong VRAM. Đổi model,
LoRA, kích thước hoặc thiếu RAM/VRAM vẫn có thể cần nạp/compile thêm. Chọn một preset
phù hợp với tác vụ sắp dùng. Chạy lại cell cấu hình và launcher sau khi đổi lựa chọn;
dừng phiên cũ trước khi chạy lại launcher.

Giới hạn 4 giờ trước đây là do launcher đặt, không phải yêu cầu bắt buộc của Colab.
`MAX_SESSION_HOURS = 0` chỉ bỏ hẹn giờ của launcher; Colab vẫn có giới hạn tài nguyên,
thời gian sống runtime và idle riêng theo tài khoản/tình trạng dịch vụ.
Bấm Stop và Disconnect and delete runtime khi dùng xong GPU.

## Kết nối lần đầu

Giữ cell cuối của notebook Colab chạy. Trong Terminal trên máy Mac:

```sh
cd /Users/khiemp/unlock/template
python3 comfy_agent.py configure --url https://mutable-glue-usage.ngrok-free.dev
```

Nhập đúng `COMFYUI_PASSWORD` đã dùng ở Colab vào ô ẩn. Username mặc định là `comfyui`.
Script kiểm tra đăng nhập thành công rồi mới lưu `.comfy-agent.json`, quyền `600`.
File chứa mật khẩu dạng văn bản, chỉ dùng riêng trên máy và đã được thêm vào `.gitignore`.
Không chia sẻ file, không đưa mật khẩu vào prompt hoặc tham số dòng lệnh.
Khi link/mật khẩu phiên Colab thay đổi, chạy lại `configure`.

### Launcher dừng ở bước kiểm tra Basic Auth

Nếu log đã có `Starting server` rồi báo lỗi xác minh đăng nhập ngrok, ComfyUI đã
khởi động; lỗi nằm ở truy cập endpoint công khai. Bản notebook mới giữ mã HTTP và
`ERR_NGROK_...` trong thông báo thay vì gộp mọi phản hồi thành lỗi Basic Auth.
Launcher chờ tối đa khoảng 60 giây cho lỗi tunnel tạm thời; lỗi mật khẩu, quota
hoặc cấu hình được báo riêng. Không chuyển tiếp thông tin đăng nhập qua redirect.
Endpoint phải chặn truy cập không đăng nhập rồi chấp nhận đúng thông tin đăng nhập
trước khi launcher in link; nếu không, launcher vẫn cleanup và dừng.

Mở lại đúng bản `comfy.ipynb` đã cập nhật rồi Run all; không cần thêm cell.
Nếu mở từ GitHub, cần push bản mới trước. Khi có HTTP 402/403/429 hoặc mã ngrok khác,
kiểm tra Usage/Limits và endpoint ở dashboard theo mã lỗi. Log cũ chỉ báo lỗi chung
không đủ để phân biệt tunnel chưa sẵn sàng với quota/cấu hình.

Agent có thể dùng cấu hình đó mà không đọc hoặc in nội dung file:

```sh
python3 comfy_agent.py check
python3 comfy_agent.py models
python3 comfy_agent.py models checkpoints
python3 comfy_agent.py models diffusion_models
python3 comfy_agent.py nodes KSampler
python3 comfy_agent.py queue
```

Cũng có thể cung cấp `COMFYUI_URL`, `COMFYUI_USERNAME`, `COMFYUI_PASSWORD` qua môi trường
thay cho file; các giá trị môi trường được ưu tiên. Không ghi giá trị bí mật vào lịch sử shell.
Với cấu hình ở nơi khác: `python3 comfy_agent.py --config /path/to/private.json check`.

## Chạy workflow

Kiểm tra toàn bộ chu trình gửi → chạy → lưu → tải bằng một ô màu 64×64, không cần model:

```sh
python3 comfy_agent.py run workflows/connection_test_api.json --wait
```

Workflow này tạo một file `image/agent_connection_test_*.png` trong output của ComfyUI
(Drive với notebook hiện tại), đồng thời tải một bản vào `comfy-results/` trên máy.

Xuất workflow từ ComfyUI dưới dạng **API JSON** (tìm mục Export (API)/Save (API Format)
trong menu workflow; tùy phiên bản cần bật developer mode). JSON hợp lệ chứa các node
có `class_type` và `inputs`. JSON sơ đồ giao diện chứa danh sách `nodes` không dùng trực tiếp được.

```sh
python3 comfy_agent.py run workflow_api.json --wait --output comfy-results
```

Sửa tham số trên bản gửi đi, không thay đổi file nguồn. ID node phụ thuộc workflow thực tế:

```sh
python3 comfy_agent.py run workflow_api.json \
  --set '6.text="a small cabin in a pine forest, sunrise"' \
  --set '3.seed=42' --set '3.steps=20' \
  --wait --timeout 1800
```

Mỗi `--set` dùng cú pháp `NODE.INPUT=JSON`; chuỗi cần dấu ngoặc kép JSON.
Không tự áp dụng ví dụ ID 3 hoặc 6 nếu workflow dùng ID khác.
Nếu không có `--wait`, script trả `prompt_id` ngay sau khi gửi:

```sh
python3 comfy_agent.py status PROMPT_ID
python3 comfy_agent.py download PROMPT_ID --output comfy-results
```

Ảnh tải về nằm trong `comfy-results/`. Trên Drive, ảnh tạo được lưu trong
`MyDrive/ComfyUI/output/image/`. Script tự thêm `image/` cho `filename_prefix` của
node SaveImage nếu tiền tố chưa có thư mục, kể cả sau khi áp dụng `--set`.
Tiền tố đã có thư mục (như video hoặc ảnh chỉnh sửa) được dùng nguyên trạng.
Khi tạo workflow trên giao diện ComfyUI, đặt `filename_prefix` của SaveImage là
`image/tên_ảnh`; các workflow tạo ảnh mẫu trong `workflows/` đã dùng tiền tố này.
`--wait` lấy trạng thái bằng polling HTTP, mặc định 3 giây/lượt; có thể tăng `--interval`
để giảm request ngrok. Hết thời gian chờ hoặc dừng client không hủy tác vụ trên server.
Không tự retry POST `/prompt`. Nếu mất kết nối lúc gửi, kiểm tra queue/history trước khi gửi lại.
Workflow qua API không tự thay thế canvas đang mở trong trình duyệt.

## Tải model nền: không dừng ComfyUI mỗi lần

Notebook `comfy.ipynb` đã tích hợp bộ tải nền và tự bật cùng launcher.
Agent gửi manifest qua userdata có Basic Auth, worker tải file trên Colab và giữ model trên Drive.
Không có cổng công khai hoặc tunnel bổ sung. Mỗi lần chỉ tải một model để tránh tranh băng thông.
Bộ tải dùng thư viện chuẩn và curl sẵn có; không cài custom node hoặc thay bộ PyTorch.

### Chạy bản notebook mới — không thêm cell

1. Dùng đúng file `comfy.ipynb` đã cập nhật trong workspace này. Có thể upload trực tiếp file
   lên Colab; nếu mở từ GitHub thì cần push bản mới rồi mở lại đúng notebook đó.
   Tab Colab đang mở với code cũ không tự nhận thay đổi từ máy/GitHub.
2. Stop cell cuối của phiên cũ để tránh trùng cổng. Giữ runtime hiện tại nếu vẫn hoạt động.
3. Mở bản notebook mới, chọn GPU rồi **Runtime → Run all**. Bộ tải nền đã được nhúng ngay
   trong notebook, không cần upload `model_download_worker.py` hoặc thêm cell thủ công.
4. Cấp quyền Drive nếu Colab yêu cầu, nhập token và cùng `COMFYUI_PASSWORD` đã cấu hình
   trên máy. Nếu dùng Secrets thì notebook đọc tự động như trước.
5. Đợi `MODEL_DOWNLOADER_READY` và link ComfyUI. Agent có thể kiểm tra bằng `model-worker`.

Notebook lưu phần code tích hợp. Model và dữ liệu được lưu trên Drive; khi runtime mới được
cấp, chạy lại notebook sẽ bật lại worker và dùng các model đã tải trước đó.
Không cần chạy lại bộ cài chỉ để thêm checkpoint/LoRA/VAE khi phiên đang hoạt động.

### Agent tải model trong khi ComfyUI vẫn chạy

```sh
python3 comfy_agent.py model-worker
python3 comfy_agent.py install-model sdxl-base --wait
```

Không cần Stop, thêm cell hoặc khởi động lại ComfyUI cho mỗi checkpoint/LoRA/VAE mới.
Agent có thể dùng `models`/`nodes` để đọc lại danh sách sau khi tải. Trên giao diện,
có thể cần refresh danh sách model để chọn file mới. Cài thêm custom node hoặc nâng cấp
ComfyUI là việc khác và vẫn có thể cần khởi động lại.

`install-model` in `job_id` ngay khi gửi. Có thể tiếp tục theo dõi mà không gửi lại:

```sh
python3 comfy_agent.py model-status JOB_ID --wait
```

Dừng client không hủy tải; worker tiếp tục trong runtime. Khi runtime bị thu hồi thì tải dừng.
Các job đang tải dở có thể tiếp tục khi chạy lại notebook với cùng Drive; job đã báo `failed`
cần kiểm tra nguyên nhân trước khi gửi job mới. File `.part` ở `models/.agent-downloads`.

### Model khác

`models/catalog.json` có các manifest đã chuẩn bị. Thêm model vào catalog hoặc dùng:

```sh
python3 comfy_agent.py install-model --manifest model_manifest.json --wait
```

Manifest bắt buộc: `repo_id`, commit `revision` đủ 40 ký tự, `source_filename`, `filename`,
`folder`, `size` (byte), `sha256`. Lấy thông tin từ nguồn phát hành model trước khi gửi.
Chỉ hỗ trợ file `.safetensors` công khai trên Hugging Face; chưa hỗ trợ model gated,
Civitai, file GGUF hoặc cài code. Không gửi token vào manifest.
Các thư mục hỗ trợ gồm checkpoints, loras, vae, text_encoders, diffusion_models,
clip_vision, embeddings, controlnet, upscale_models, vae_approx.

Model chỉ được đưa vào thư mục đích sau khi kiểm tra đủ kích thước và SHA256.
Không ghi đè model có sẵn khác nội dung. Mỗi file tối đa 100 GB, vẫn cần đủ dung lượng Drive.

## Ảnh ngôi nhà + khinh khí cầu

Model: [SDXL Base 1.0 của Stability AI](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0),
checkpoint `sd_xl_base_1.0.safetensors`, 6.938.078.334 byte; revision và SHA256 ở catalog.

- `workflows/eye_test_house_balloon_api.json`: gửi qua script Python.
- `workflows/eye_test_house_balloon.json`: mở trên canvas, đã lưu vào userdata/workflows.

```sh
python3 comfy_agent.py install-model sdxl-base --wait
python3 comfy_agent.py run workflows/eye_test_house_balloon_api.json --wait --interval 5
```

Workflow dùng SDXL, 1152×896, 28 bước, CFG 6, DPM++ 2M/Karras, seed 20261112.
Ảnh mang tiền tố `image/eye_test_house_balloon` trong output Drive và được tải về `comfy-results/`.
Đây là ảnh phong cảnh gợi hình trong máy đo mắt, không phải ảnh chuẩn dùng cho phép đo.

Lưu/cập nhật workflow giao diện vào danh sách workflow của server:

```sh
python3 comfy_agent.py upload-userdata workflows/eye_test_house_balloon.json workflows/eye_test_house_balloon.json
```

## Prompt dùng cho agent khác

Sao chép đoạn sau và thay đường dẫn workspace nếu cần:

```text
Bạn có quyền dùng script /Users/khiemp/unlock/template/comfy_agent.py để điều khiển
ComfyUI đang chạy trên Colab theo yêu cầu tạo ảnh của tôi.

1. Đọc COMFY_AGENT_GUIDE.md cùng thư mục. Dùng python3 comfy_agent.py --help khi cần.
2. Chạy check trước. Script tự đọc cấu hình riêng .comfy-agent.json hoặc biến môi trường.
   Không đọc/in file cấu hình, mật khẩu, Authorization hoặc biến môi trường chứa bí mật.
   Nếu chưa cấu hình hoặc HTTP 401, nhờ tôi chạy configure trong Terminal với ô nhập ẩn;
   không yêu cầu tôi gửi mật khẩu vào chat.
3. Liệt kê models và xem schema node bằng nodes để biết model/input thực tế.
   Dùng workflow API JSON tôi cung cấp hoặc tạo workflow từ schema và model đang có.
   Không đoán tên checkpoint, node hoặc ID node; chỉ tải model/cài custom node khi tôi yêu cầu.
   Nếu cần tải model, kiểm tra model-worker rồi dùng install-model --wait; theo dõi job_id
   bằng model-status. Không thêm cell hoặc Stop ComfyUI cho mỗi lần tải model.
4. Sửa prompt/seed/kích thước bằng --set NODE.INPUT=JSON hoặc tạo file API JSON riêng.
   Bắt đầu bằng một ảnh để kiểm tra, sau đó mới chạy số lượng tôi yêu cầu.
5. Dùng run WORKFLOW --wait --output comfy-results. Lưu prompt_id ngay khi được trả về.
   Với tác vụ lâu, dùng status PROMPT_ID và download PROMPT_ID để tiếp tục theo dõi.
6. Nếu timeout/mất kết nối, kiểm tra queue và status trước; không gửi lại mù quáng.
   Không dừng/xóa hàng đợi của tác vụ khác. Không thay đổi dữ liệu trên Drive ngoài yêu cầu.
7. Xem ảnh đã tải về nếu có công cụ xem ảnh, đối chiếu yêu cầu, điều chỉnh có căn cứ.
   Báo file kết quả, prompt_id và các thông số chính. Nếu thiếu kết nối/model/workflow,
   nói rõ thiếu gì, không tuyên bố đã chạy thành công.

Yêu cầu tạo ảnh của tôi: [điền mô tả, số lượng, kích thước, workflow/model nếu có].
```

## MiniMax H3 Image: ảnh eye_test

Bộ file tối thiểu được lấy từ bản đóng gói chính thức Comfy-Org, tổng khoảng 37,8 GB:

| Tên trong catalog | Thư mục | File |
| --- | --- | --- |
| `minimax-h3-image` | diffusion_models | minimax_h3_image_pruned_int8_convrot.safetensors |
| `minimax-h3-text-nvfp4` | text_encoders | qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors |
| `minimax-h3-image-vae` | vae | minimax_h3_image_vae_fp16.safetensors |

Encoder NVFP4 này không yêu cầu GPU Blackwell. Không tải thêm prompt enhancer:
workflow dùng prompt tiếng Anh đã viết chi tiết. Revision, size và SHA256 nằm trong catalog.

```sh
python3 comfy_agent.py install-model minimax-h3-image --wait
python3 comfy_agent.py install-model minimax-h3-text-nvfp4 --wait
python3 comfy_agent.py install-model minimax-h3-image-vae --wait
python3 comfy_agent.py run workflows/eye_test_minimax_h3_image_api.json --wait --interval 10
```

Workflow riêng cho model Image: 2048×2048, seed 20261112, 6 bước, CFG 1, Euler/simple.
Model đã được distill CFG; không dùng negative prompt riêng hoặc thông số sampler SDXL.
`workflows/eye_test_minimax_h3_image.json` là bản mở được trên canvas, cũng được lưu vào
userdata/workflows. File ảnh đầu ra dùng tiền tố `image/eye_test_minimax_h3_image`.

Nguồn:
- https://huggingface.co/Comfy-Org/MiniMax-H3-Image
- https://huggingface.co/Comfy-Org/MiniMax-H3
- https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json

## MiniMax H3 video: animate ảnh eye_test

Dùng FL2VA INT8 cho text-to-video hoặc first/last-frame image-to-video. Bộ tải thêm
khoảng 26,3 GB; dùng lại `minimax-h3-text-nvfp4` đã cài ở phần model Image.

```sh
python3 comfy_agent.py install-model minimax-h3-video --wait
python3 comfy_agent.py install-model minimax-h3-video-vae --wait
python3 comfy_agent.py install-model minimax-h3-audio-vae --wait
python3 comfy_agent.py install-model minimax-h3-video-turbo4 --wait
python3 comfy_agent.py run workflows/eye_test_minimax_h3_video_api.json --wait --interval 10 --timeout 2400
```

Workflow dùng ảnh `eye_test_minimax_h3_image_00001_.png [output]` làm first frame,
đọc trực tiếp từ output Drive. Nếu dùng ảnh khác, đổi input `114.image` sang tên ảnh
hợp lệ. Với ảnh mới trong thư mục riêng, dùng dạng
`image/eye_test_minimax_h3_image_00001_.png [output]`.
Hậu tố `[output]` là cơ chế đường dẫn chuẩn của LoadImage.

- Video 640×640, 124 frame ở 24 fps (khoảng 5,17 giây).
- FL2VA + Turbo LoRA 4 bước, sampler `res_multistep`, scheduler `simple`, seed 20261112.
- Video VAE và audio VAE riêng; ghép video với native audio vào MP4/H.264.
- Xuất thêm ảnh xem trước đầu/giữa/cuối từ batch frame để kiểm tra chuyển động.
- Tiền tố kết quả: `video/eye_test_minimax_h3_video`.
- Bản canvas: `workflows/eye_test_minimax_h3_video.json`, được lưu trong userdata/workflows.

Đây là workflow thử ở độ phân giải vừa phải cho L4, không phải preset chất lượng tối đa.
Không cần tải Ref2VA để chạy T2V/I2V này; đó là một model riêng cho tham chiếu đa phương tiện.

Nguồn:
- https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- https://huggingface.co/Comfy-Org/MiniMax-H3
- https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json
