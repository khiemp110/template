"""Native KoboldCpp MiniMax H3 Image support; embedded into colab.ipynb.

The maintained KoboldCpp SD update (PR #2576) plus the upstream SD source sync
adds H3 Image without replacing KoboldCpp's server or its bundled StableUI.
All weights/builds are downloaded on the Colab runtime. No ComfyUI dependency.
"""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time

NATIVE_KCPP_REF = "67399ec7d0c0d7e53d0ee1e940bcebd328b65ed5"
NATIVE_SD_BASE = "73fef08224c50a49961004dc8c6fee2c0f640bcb"
NATIVE_SD_REF = "32d516c0197ed5ca31305b8575e14eb1451fad33"
NATIVE_SOURCE_SHA256 = "3b77a5a277c295a8569c8197dbd95987dab0868bca6c412bfedbf22e0d5da4d0"
NATIVE_PATCH_SHA256 = "dde88ef042473bc5e8a2fdda314f18e371ce05cf15c24c672c4499b2847da230"

MINIMAX_L4_MODELS = {
    "img_model": {
        "repo_id": "AtomicChat/MiniMax-H3-Image-GGUF",
        "revision": "5da8538f0b9a7a8a0afd1ff6ed22ade4b0134094",
        "source_filename": "MiniMax-H3-Image-AD-Q4_K.gguf",
        "size": 11571097824,
        "sha256": "df3f3734c1729765a2b7108bebd69b79d00b85fec5bceaf7363924afa84af1e7",
    },
    "img_clip": {
        "repo_id": "leejet/MiniMax-H3-GGUF",
        "revision": "d9c4c6312b4728a68a15a35626d84775a6523783",
        "source_filename": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "size": 18218065024,
        "sha256": "11e6efe70a57ce7f4838c47bdbd1a1c4b8ce10e2b7747f1b065990b70f4b05fc",
    },
    "img_vae": {
        "repo_id": "Comfy-Org/MiniMax-H3-Image",
        "revision": "0bb149f51e3156ed4b90dd45d5620346bfe1df8f",
        "source_filename": "vae/minimax_h3_image_vae_fp16.safetensors",
        "size": 1168505312,
        "sha256": "d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af",
    },
}


def minimax_model_url(model):
    return f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/{model['source_filename']}"


def set_native_log_path(path, drive_root=None):
    global NATIVE_LOG_PATH, NATIVE_LOG_STARTED, NATIVE_DRIVE_LOG_PATH
    NATIVE_LOG_PATH = Path(path)
    NATIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    NATIVE_LOG_STARTED = time.monotonic()
    NATIVE_DRIVE_LOG_PATH = (Path(drive_root) / "logs" / f"session-{time.time_ns()}.log"
                             if drive_root is not None else None)
    native_log(f"Log trực tiếp trong cell; file đầy đủ: {NATIVE_LOG_PATH}")


def native_log(message):
    started = globals().get("NATIVE_LOG_STARTED")
    prefix = f"[+{time.monotonic() - started:.1f}s] " if started is not None else ""
    line = prefix + str(message).rstrip("\r\n")
    print(line, flush=True)
    path = globals().get("NATIVE_LOG_PATH")
    if path is not None:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def save_native_log():
    source = globals().get("NATIVE_LOG_PATH")
    target = globals().get("NATIVE_DRIVE_LOG_PATH")
    if source is not None and target is not None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as error:
            print(f"Không sao lưu được log lên Drive: {error}", flush=True)


@contextmanager
def native_stage(label, heartbeat_seconds=15):
    started = time.monotonic()
    stopped = threading.Event()
    native_log(f"BẮT ĐẦU: {label}")

    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            native_log(f"{label}: đang chạy, đã {time.monotonic() - started:.0f}s; chưa kết thúc bước này.")

    ticker = threading.Thread(target=heartbeat, daemon=True)
    ticker.start()
    try:
        yield
    except BaseException as error:
        native_log(f"DỪNG/LỖI: {label} sau {time.monotonic() - started:.1f}s ({type(error).__name__}: {error})")
        raise
    else:
        native_log(f"XONG: {label} sau {time.monotonic() - started:.1f}s")
    finally:
        stopped.set()
        ticker.join(timeout=1)


def run_logged_command(arguments, *, cwd=None, label=None, heartbeat_seconds=15):
    """Pipe child output through Python so Colab actually displays it in the cell."""
    command = [str(argument) for argument in arguments]
    label = label or Path(command[0]).name
    native_log(f"Lệnh: {shlex.join(command)}")
    if cwd is not None:
        native_log(f"Thư mục: {cwd}")
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    # Also flush C/C++ stdout on Colab; Python's bufsize only affects the reader.
    executable = command
    if sys.platform == "linux" and shutil.which("stdbuf"):
        executable = ["stdbuf", "-oL", "-eL", *command]
    process = None
    try:
        with native_stage(label, heartbeat_seconds):
            process = subprocess.Popen(executable, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace",
                                       bufsize=1, env=environment, start_new_session=True)
            native_log(f"{label}: PID {process.pid}")
            for line in process.stdout:
                native_log(line)
            returncode = process.wait()
            native_log(f"{label}: exit code {returncode}")
            if returncode:
                native_log(f"Xem toàn bộ stdout/stderr tại: {globals().get('NATIVE_LOG_PATH', 'output của cell')}")
                raise subprocess.CalledProcessError(returncode, command)
            return subprocess.CompletedProcess(command, returncode)
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            process.stdout.close()
        save_native_log()


def native_file_sha256(path, *, label=None):
    path = Path(path)
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    last_report = time.monotonic()
    if label:
        native_log(f"{label}: đọc {total / 1e9:.2f} GB để kiểm tra SHA256")
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            done += len(chunk)
            if label and time.monotonic() - last_report >= 10:
                native_log(f"{label}: {done / 1e9:.2f}/{total / 1e9:.2f} GB ({100 * done / total:.1f}%)")
                last_report = time.monotonic()
    if label:
        native_log(f"{label}: đã đọc xong {total / 1e9:.2f} GB")
    return digest.hexdigest()


def native_download(url, target, expected_sha256):
    target = Path(target)
    if target.is_file() and native_file_sha256(target, label=target.name) == expected_sha256:
        native_log(f"Dùng lại mã nguồn/patch: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    run_logged_command(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
                        "--output", str(partial), url], label=f"Tải {target.name}")
    if native_file_sha256(partial, label=target.name) != expected_sha256:
        raise RuntimeError(f"SHA256 mã nguồn không khớp: {partial}")
    partial.replace(target)
    return target


def adapt_native_api_compat(source):
    # Two fork-specific LTX call sites are outside the upstream diff. Match the
    # runner-context signatures introduced by the same SD source sync.
    header = Path(source) / "otherarch/sdcpp/src/model/vae/ltx_audio_vae.hpp"
    data = header.read_bytes()
    for old, new in (
        (b"ggml_ext_pad_ext(ctx->ggml_ctx,\n                                         ctx->backend,",
         b"ggml_ext_pad_ext(ctx,"),
        (b"ggml_ext_slice(ctx->ggml_ctx, latent, 2, 0, config.latent_channels)",
         b"ggml_ext_slice(ctx, latent, 2, 0, config.latent_channels)"),
    ):
        if data.count(old) != 1:
            raise RuntimeError("KoboldCpp thay đổi API LTX ngoài phiên bản đã ghim.")
        data = data.replace(old, new)
    header.write_bytes(data)


def apply_native_sd_patch(source, original_patch):
    # Keep CRLF inside upstream's T5 hunk. read_text()/write_text() breaks it.
    patch = Path(original_patch).read_bytes()
    if hashlib.sha256(patch).hexdigest() != NATIVE_PATCH_SHA256:
        raise RuntimeError("Patch upstream sai SHA256.")
    old = b"     ggml_log_set(sd_ggml_log_callback, nullptr);"
    if patch.count(old) != 1:
        raise RuntimeError("Patch logging upstream thay đổi ngoài dự kiến.")
    # Preserve KoboldCpp's logging hook in the upscaler refactor.
    patch = patch.replace(old, b"     kcpp_sd_ggml_log_set();")
    adapted = Path(source).parent / "sdcpp-kobold.diff"
    adapted.write_bytes(patch)
    command = ["git", "apply", "--directory=otherarch/sdcpp",
               # KoboldCpp uses upstream ggml's CUDA implementation, not SD's custom kernels.
               "--exclude=otherarch/sdcpp/src/core/cuda/*", "--include=otherarch/sdcpp/src/*", str(adapted)]
    run_logged_command(command[:2] + ["--check"] + command[2:], cwd=source, label="Kiểm tra patch H3 Image")
    run_logged_command(command, cwd=source, label="Áp dụng patch H3 Image")
    adapt_native_api_compat(source)
    definition = Path(source) / "otherarch/sdcpp/src/model/integrations/minimax_h3.cpp"
    if "VERSION_MINIMAX_H3_IMAGE" not in definition.read_text():
        raise RuntimeError("Backend vừa patch chưa đăng ký H3 Image.")


def minimax_build_info():
    # This preset targets L4 (SM 8.9). CUDA/libc identify compatible cached builds;
    # do not impose RAM/VRAM thresholds or preflight hardware eligibility checks.
    native_log("Đọc phiên bản CUDA để chọn cache build.")
    toolkit = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True).stdout
    native_log(toolkit.strip())
    return {"sm": "89", "cuda": hashlib.sha256(toolkit.encode()).hexdigest()[:12],
            "libc": list(platform.libc_ver()), "source": NATIVE_KCPP_REF, "sd": NATIVE_SD_REF, "recipe": 1}


def configure_minimax_stableui(runtime):
    ui = Path(runtime) / "embd_res/kcpp_sdui.embd"
    data = ui.read_bytes()
    old = b'function oc(){return{steps:20,n:1,sampler_name:"Euler",width:512,height:512,cfg_scale:5,eta:1,flow_shift:0'
    new = b'function oc(){return{steps:6,n:1,sampler_name:"Euler",width:1024,height:1024,cfg_scale:1,eta:1,flow_shift:1'
    if data.count(old) != 1:
        raise RuntimeError("StableUI thay đổi cấu trúc cấu hình; không tự sửa preset.")
    ui.write_bytes(data.replace(old, new))


def ensure_minimax_native(cache_root, *, work_root="/content/koboldcpp-minimax-native"):
    path = globals().get("NATIVE_LOG_PATH")
    if path is None or not Path(path).is_file():
        set_native_log_path(Path(work_root) / "logs/session.log", cache_root)
    try:
        return _ensure_minimax_native(cache_root, work_root=work_root)
    finally:
        save_native_log()


def _ensure_minimax_native(cache_root, *, work_root):
    build_info = minimax_build_info()
    identity = hashlib.sha256(json.dumps(build_info, sort_keys=True).encode()).hexdigest()[:16]
    work = Path(work_root) / identity
    runtime = work / "runtime"
    stamp = runtime / "native-build.json"
    if stamp.is_file() and json.loads(stamp.read_text()) == build_info and (runtime / "koboldcpp_cublas.so").is_file():
        native_log(f"Dùng lại bản build trong runtime: {runtime}")
        return [sys.executable, str(runtime / "koboldcpp.py"), "--skiplauncher"], str(runtime)
    archive = Path(cache_root) / "runtime" / f"koboldcpp-minimax-{identity}.tar.gz"
    receipt = archive.with_suffix(".sha256.json")
    work.mkdir(parents=True, exist_ok=True)
    native_log(f"Thư mục build: {work}")
    if archive.is_file() and receipt.is_file():
        record = json.loads(receipt.read_text())
        if record.get("build") != build_info or native_file_sha256(archive, label="SHA256 cache build Drive") != record.get("sha256"):
            raise RuntimeError(f"Cache build native sai SHA256: {archive}; chuyển file lỗi đi rồi chạy lại.")
        with native_stage("Giải nén bản build từ Drive"), tarfile.open(archive) as package:
            package.extractall(runtime, filter="data")
        if not stamp.is_file() or json.loads(stamp.read_text()) != build_info or not (runtime / "koboldcpp_cublas.so").is_file():
            raise RuntimeError("Cache native thiếu file; không khởi chạy.")
        native_log("Dùng lại bản KoboldCpp native đã build trên Drive.")
    else:
        if any(shutil.which(tool) is None for tool in ("make", "g++", "git")):
            run_logged_command(["apt-get", "update", "-qq"], label="Cập nhật danh sách gói build")
            run_logged_command(["apt-get", "install", "-y", "-qq", "build-essential", "git"], label="Cài công cụ build")
        source_archive = native_download(
            f"https://codeload.github.com/wbruna/llama.cpp/tar.gz/{NATIVE_KCPP_REF}",
            work / "kobold-source.tar.gz", NATIVE_SOURCE_SHA256)
        patch = native_download(
            f"https://github.com/leejet/stable-diffusion.cpp/compare/{NATIVE_SD_BASE[:7]}...{NATIVE_SD_REF}.diff",
            work / "sdcpp.diff", NATIVE_PATCH_SHA256)
        source = work / ("llama.cpp-" + NATIVE_KCPP_REF)
        patch_stamp = source / ".minimax-patched"
        if not patch_stamp.is_file():
            if source.exists():
                shutil.rmtree(source)
            with native_stage("Giải nén mã nguồn KoboldCpp"), tarfile.open(source_archive) as package:
                package.extractall(work, filter="data")
            apply_native_sd_patch(source, patch)
            patch_stamp.write_text(NATIVE_SD_REF)
        else:
            native_log("Mã nguồn đã patch; make sẽ dùng lại các object đã build xong nếu có.")
        # Compile only this GPU architecture. Cache the completed runtime on Drive.
        sm = build_info["sm"]
        nvcc_flags = ("--forward-unknown-to-host-compiler -use_fast_math -extended-lambda "
                      f"-gencode arch=compute_{sm},code=sm_{sm} -DKCPP_LIMIT_CUDA_MAX_ARCH={sm}0")
        run_logged_command(["make", "-j2", "koboldcpp_cublas", "LLAMA_CUBLAS=1", "LLAMA_NO_LLAMAFILE=1",
                            "NVCCFLAGS=" + nvcc_flags], cwd=source, label="Build KoboldCpp CUDA + H3 Image")
        library = source / "koboldcpp_cublas.so"
        if not library.is_file():
            raise RuntimeError("Build chưa tạo koboldcpp_cublas.so.")
        with native_stage("Chuẩn bị runtime và preset StableUI"):
            runtime.mkdir(parents=True, exist_ok=True)
            for path in source.glob("*.py"):
                shutil.copy2(path, runtime / path.name)
            shutil.copy2(library, runtime / library.name)
            shutil.copytree(source / "embd_res", runtime / "embd_res", dirs_exist_ok=True)
            configure_minimax_stableui(runtime)
            stamp.write_text(json.dumps(build_info, sort_keys=True))
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".gz.part")
        with native_stage("Nén và lưu build vào Drive"), tarfile.open(partial, "w:gz") as package:
            for path in sorted(runtime.iterdir()):
                package.add(path, arcname=path.name)
        digest = native_file_sha256(partial, label="SHA256 bản build mới")
        partial.replace(archive)
        temporary_receipt = receipt.with_suffix(".json.tmp")
        temporary_receipt.write_text(json.dumps({"build": build_info, "sha256": digest}))
        temporary_receipt.replace(receipt)
        native_log(f"Đã lưu build native lên Drive: {archive}")
    return [sys.executable, str(runtime / "koboldcpp.py"), "--skiplauncher"], str(runtime)
