"""Warm one model preset inside the running ComfyUI process, using only core nodes."""
import asyncio
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


PRELOAD_PRESETS = ("MiniMax H3 Image", "MiniMax H3 Video (Turbo 4 bước)")


def build_preload_workflow(preset):
    def node(class_type, **inputs):
        return {"class_type": class_type, "inputs": inputs}

    clip = node("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                type="minimax", device="default")
    prompt = "A small white house in a green field under a clear blue sky."
    if preset == PRELOAD_PRESETS[0]:
        return {
            "1": node("UNETLoader", unet_name="minimax_h3_image_pruned_int8_convrot.safetensors",
                      weight_dtype="default"),
            "2": clip,
            "3": node("VAELoader", vae_name="minimax_h3_image_vae_fp16.safetensors"),
            "5": node("MiniMaxH3ImageConditioning", clip=["2", 0], vae=["3", 0],
                      prompt=prompt, width=512, height=512),
            "6": node("ConditioningZeroOut", conditioning=["5", 0]),
            "7": node("KSampler", model=["1", 0], seed=0, steps=1, cfg=1.0,
                      sampler_name="euler", scheduler="simple", positive=["5", 0],
                      negative=["6", 0], latent_image=["5", 1], denoise=1.0),
            "8": node("VAEDecode", samples=["7", 0], vae=["3", 0]),
            "9": node("PreviewImage", images=["8", 0]),
        }
    if preset == PRELOAD_PRESETS[1]:
        return {
            "6": node("UNETLoader", unet_name="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                      weight_dtype="default"),
            "121": node("LoraLoaderModelOnly", model=["6", 0], strength_model=1.0,
                        lora_name="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
            "13": clip,
            "11": node("VAELoader", vae_name="minimax_h3_video_vae_int8_convrot.safetensors"),
            "24": node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
            "104": node("MiniMaxH3ImageToVideo", clip=["13", 0], vae=["11", 0],
                        prompt=prompt + " Gentle breeze. Soft wind, no speech or music.",
                        width=640, height=352, length=124),
            "15": node("RandomNoise", noise_seed=0),
            "16": node("BasicGuider", model=["121", 0], conditioning=["104", 0]),
            "17": node("KSamplerSelect", sampler_name="res_multistep"),
            "9": node("BasicScheduler", model=["121", 0], scheduler="simple", steps=1, denoise=1.0),
            "14": node("SamplerCustomAdvanced", noise=["15", 0], guider=["16", 0],
                       sampler=["17", 0], sigmas=["9", 0], latent_image=["104", 1]),
            "10": node("VAEDecode", samples=["14", 0], vae=["11", 0]),
            "23": node("VAEDecodeAudio", samples=["14", 0], vae=["24", 0]),
            "preview_frame": node("ImageFromBatch", image=["10", 0], batch_index=0, length=1),
            "preview_image": node("PreviewImage", images=["preview_frame", 0]),
            "preview_audio": node("PreviewAudio", audio=["23", 0]),
        }
    raise ValueError(f"Preset nạp trước chưa được hỗ trợ: {preset}")


class PreloadRedirectDenied(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PreloadClient:
    def __init__(self, port):
        self.base_url = f"http://127.0.0.1:{port}"
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), PreloadRedirectDenied())

    def request(self, route, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url + route, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=10) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            with error:
                if route == "/prompt" and error.code == 400:
                    raise RuntimeError(
                        "Preset thiếu model hoặc node. Kiểm tra bộ model đã tải trên Drive."
                    ) from None
                raise RuntimeError(f"ComfyUI preload API trả HTTP {error.code}.") from None
        return json.loads(body) if body else None


async def preload_model_preset(process, port, preset, *, timeout=1800, request=None, log=print):
    """Run one disposable inference so model weights and LoRA are actually exercised.

    Previews live in ComfyUI temp, never in the user's persistent output folders.
    Only the submitted warm-up job may be cancelled on timeout or notebook Stop.
    """
    workflow = build_preload_workflow(preset)
    request = request or PreloadClient(port).request
    queue = await asyncio.to_thread(request, "/queue")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("ComfyUI đang có tác vụ; bỏ qua nạp trước để tránh chờ hàng đợi.")
    started = time.monotonic()
    log(f"Nạp trước {preset}: warm-up 1 bước; lần đầu đọc Drive có thể mất vài phút.")
    # Never retry POST /prompt: the server may already have accepted it.
    result = await asyncio.to_thread(request, "/prompt", {
        "prompt": workflow, "client_id": str(uuid.uuid4()),
    })
    prompt_id = result.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError("ComfyUI chưa trả prompt_id cho lượt nạp trước; kiểm tra hàng đợi.")
    log(f"PRELOAD_JOB: {prompt_id}")
    finished = False
    last_log = started
    route = "/history/" + urllib.parse.quote(prompt_id, safe="")
    try:
        while time.monotonic() - started < timeout:
            if process.poll() is not None:
                raise RuntimeError("ComfyUI đã thoát trong lúc nạp trước model.")
            history = await asyncio.to_thread(request, route)
            if prompt_id in history:
                finished = True
                status = history[prompt_id].get("status", {})
                if status.get("status_str") != "success" or not status.get("completed"):
                    raise RuntimeError("Warm-up chưa hoàn tất; xem log ComfyUI để biết lỗi model/VRAM.")
                elapsed = time.monotonic() - started
                log(f"PRELOAD_READY: {preset} — {elapsed:.1f} giây.")
                log("ComfyUI tự giữ/thu hồi model trong RAM và VRAM theo bộ nhớ còn trống.")
                return {"preset": preset, "prompt_id": prompt_id, "seconds": elapsed}
            now = time.monotonic()
            if now - last_log >= 30:
                log(f"Đang nạp trước {preset}: {now - started:.0f} giây; chờ khởi tạo/warm-up.")
                last_log = now
            await asyncio.sleep(2)
        raise TimeoutError(f"Nạp trước model vượt quá {timeout} giây.")
    finally:
        if not finished:
            try:
                await asyncio.to_thread(request, f"/api/jobs/{urllib.parse.quote(prompt_id, safe='')}/cancel", {})
            except (OSError, RuntimeError, ValueError):
                log(f"Chưa xác nhận dừng warm-up {prompt_id}; kiểm tra hàng đợi ComfyUI.")
