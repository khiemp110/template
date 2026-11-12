"""Run in Colab: %run /content/drive/MyDrive/ComfyUI/user/default/install_sdxl.py"""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

REPOSITORY = 'stabilityai/stable-diffusion-xl-base-1.0'
REVISION = '462165984030d82259a11f4367a4eed129e94a7b'
FILENAME = 'sd_xl_base_1.0.safetensors'
SIZE = 6938078334
SHA256 = '31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b'
URL = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}'
DESTINATION = Path('/content/drive/MyDrive/ComfyUI/models/checkpoints')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if not Path('/content').is_dir() or sys.platform != 'linux':
        raise RuntimeError('Chạy file này trong Google Colab, không chạy trên máy Mac.')
    if not Path('/content/drive/MyDrive').is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / FILENAME
    if target.exists():
        print('Model đã có; đang kiểm tra SHA256...', flush=True)
        if target.stat().st_size != SIZE or digest(target) != SHA256:
            raise RuntimeError(f'{target} đã tồn tại nhưng sai size/SHA256; không ghi đè. Kiểm tra file này trước.')
        print(f'MODEL_READY: {target}', flush=True)
        return
    if shutil.which('curl') is None:
        raise RuntimeError('Runtime thiếu curl; không tự cài thêm package. Dùng Colab runtime chuẩn.')
    partial = target.with_suffix(target.suffix + '.part')
    downloaded = partial.stat().st_size if partial.exists() else 0
    if downloaded > SIZE:
        raise RuntimeError(f'File tải dở lớn bất thường: {partial}; không ghi đè.')
    if shutil.disk_usage(DESTINATION).free < SIZE - downloaded:
        raise RuntimeError('Không đủ dung lượng trống cho model khoảng 6,94 GB.')
    print(f'Tải {REPOSITORY} → {target} (6,94 GB).', flush=True)
    print('Có thể tải tiếp file .part nếu bị ngắt. Không thay đổi PyTorch hoặc custom nodes.', flush=True)
    subprocess.run([
        'curl', '--fail', '--location', '--retry', '3', '--retry-delay', '3',
        '--connect-timeout', '30', '--max-time', '3600', '--continue-at', '-',
        '--output', str(partial), URL,
    ], check=True)
    if partial.stat().st_size != SIZE:
        raise RuntimeError('File tải về sai kích thước; giữ .part để kiểm tra/tải tiếp.')
    print('Đang kiểm tra SHA256 của toàn bộ model...', flush=True)
    if digest(partial) != SHA256:
        raise RuntimeError('SHA256 không khớp nguồn Stability AI; không đưa file này vào sử dụng.')
    partial.rename(target)
    print(f'MODEL_READY: {target}', flush=True)
    print('Chạy lại cell cuối của notebook để mở ComfyUI, dùng cùng COMFYUI_PASSWORD.', flush=True)


if __name__ == '__main__':
    main()
