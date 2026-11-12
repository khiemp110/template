"""Colab background downloader; requests/status travel via authenticated ComfyUI userdata."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse

MODEL_DOWNLOAD_NAMESPACE = '_agent_models'
MODEL_DOWNLOAD_FOLDERS = {
    'checkpoints', 'loras', 'vae', 'text_encoders', 'diffusion_models',
    'clip_vision', 'embeddings', 'controlnet', 'upscale_models', 'vae_approx',
}


def validate_model_request(job):
    if not isinstance(job, dict):
        raise ValueError('Job must be a JSON object.')
    for key, pattern in (
        ('id', r'[0-9a-f]{32}'),
        ('repo_id', r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+'),
        ('revision', r'[0-9a-f]{40}'),
        ('sha256', r'[0-9a-f]{64}'),
        ('filename', r'[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors'),
    ):
        if not isinstance(job.get(key), str) or not re.fullmatch(pattern, job[key]):
            raise ValueError(f'Invalid {key}.')
    source = job.get('source_filename')
    if (not isinstance(source, str) or not source.endswith('.safetensors')
            or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', part)
                   for part in source.split('/'))):
        raise ValueError('Invalid source_filename.')
    if job.get('folder') not in MODEL_DOWNLOAD_FOLDERS:
        raise ValueError('Invalid destination model folder.')
    if type(job.get('size')) is not int or not 0 < job['size'] <= 100_000_000_000:
        raise ValueError('Invalid expected size (maximum 100 GB).')
    return job


class ModelDownloadWorker(threading.Thread):
    def __init__(self, drive_root):
        super().__init__(name='comfy-model-downloader', daemon=True)
        self.drive_root = Path(drive_root).resolve()
        self.root = self.drive_root / 'user' / 'default' / MODEL_DOWNLOAD_NAMESPACE
        self.requests = self.root / 'requests'
        self.statuses = self.root / 'status'
        self.models = self.drive_root / 'models'
        self.partials = self.models / '.agent-downloads'
        self.stop_event = threading.Event()
        for directory in (self.requests, self.statuses, self.partials):
            self.confined(directory).mkdir(parents=True, exist_ok=True)
        self.last_heartbeat = 0

    def confined(self, path):
        path = Path(path)
        if not path.resolve().is_relative_to(self.drive_root):
            raise ValueError('Destination escapes the configured Drive root.')
        return path

    def write_json(self, path, value):
        path = self.confined(path)
        fd, temporary = tempfile.mkstemp(prefix='.write-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def heartbeat(self, active_job=None, force=False):
        now = time.time()
        if force or now - self.last_heartbeat >= 15:
            self.write_json(self.root / 'worker.json', {
                'protocol': 1, 'state': 'ready', 'heartbeat_at': now,
                'active_job': active_job, 'folders': sorted(MODEL_DOWNLOAD_FOLDERS),
            })
            self.last_heartbeat = now

    def status(self, job, state, **extra):
        self.write_json(self.statuses / (job['id'] + '.json'), {
            'job_id': job['id'], 'state': state, 'filename': job.get('filename'),
            'expected_bytes': job.get('size'), 'updated_at': time.time(), **extra,
        })
        self.heartbeat(job['id'])

    def verify(self, path, job):
        if path.stat().st_size != job['size']:
            raise ValueError('Downloaded file size does not match the manifest.')
        digest = hashlib.sha256()
        last_update = 0
        with path.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                if time.monotonic() - last_update > 10:
                    self.status(job, 'verifying', downloaded_bytes=job['size'])
                    last_update = time.monotonic()
        if digest.hexdigest() != job['sha256']:
            raise ValueError('SHA256 mismatch; model was not installed.')

    def install(self, job):
        target = self.confined(self.models / job['folder'] / job['filename'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            self.verify(target, job)
            self.status(job, 'completed', path=str(target), downloaded_bytes=job['size'], reused=True)
            return
        partial = self.confined(self.partials / (job['id'] + '.part'))
        already = partial.stat().st_size if partial.exists() else 0
        if already > job['size']:
            raise ValueError('Partial file is larger than expected.')
        if shutil.disk_usage(target.parent).free < job['size'] - already:
            raise ValueError('Insufficient free space for this model.')
        if shutil.which('curl') is None:
            raise ValueError('Colab runtime has no curl executable.')
        repo = urllib.parse.quote(job['repo_id'], safe='/')
        source = urllib.parse.quote(job['source_filename'], safe='/')
        # Only pinned Hugging Face HTTPS sources; clients cannot supply arbitrary URLs.
        url = f"https://huggingface.co/{repo}/resolve/{job['revision']}/{source}?download=true"
        self.status(job, 'downloading', downloaded_bytes=already)
        if already < job['size']:
            log_path = self.confined(self.partials / (job['id'] + '.log'))
            with log_path.open('ab') as log:
                process = subprocess.Popen([
                    'curl', '--fail', '--location', '--silent', '--show-error',
                    '--retry', '3', '--retry-delay', '3', '--connect-timeout', '30',
                    '--max-time', '3600', '--continue-at', '-', '--output', str(partial), url,
                ], stdout=subprocess.DEVNULL, stderr=log)
                try:
                    while process.poll() is None:
                        if self.stop_event.wait(5):
                            raise RuntimeError('Model downloader was stopped.')
                        downloaded = partial.stat().st_size if partial.exists() else 0
                        if downloaded > job['size']:
                            raise ValueError('Download exceeded the expected file size.')
                        self.status(job, 'downloading', downloaded_bytes=downloaded)
                    if process.returncode:
                        raise RuntimeError(f'curl failed with exit code {process.returncode}; partial file retained.')
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
        self.verify(partial, job)
        if target.exists():
            raise ValueError('Another file appeared at the destination; refusing to overwrite it.')
        partial.rename(target)
        self.status(job, 'completed', path=str(target), downloaded_bytes=job['size'], sha256=job['sha256'])

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.heartbeat()
                for request_path in sorted(self.requests.glob('*.json')):
                    if not re.fullmatch(r'[0-9a-f]{32}\.json', request_path.name):
                        continue
                    status_path = self.confined(self.statuses / request_path.name)
                    if status_path.exists():
                        old = json.loads(status_path.read_text(encoding='utf-8'))
                        if old.get('state') in ('completed', 'failed'):
                            continue
                    job = {'id': request_path.stem}
                    try:
                        self.confined(request_path)
                        if request_path.stat().st_size > 16384:
                            raise ValueError('Job manifest is too large.')
                        loaded = json.loads(request_path.read_text(encoding='utf-8'))
                        validate_model_request(loaded)
                        if loaded['id'] != request_path.stem:
                            raise ValueError('Job ID does not match request filename.')
                        job = loaded
                        self.install(job)
                    except Exception as error:
                        self.status(job, 'failed', error=str(error))
                    if self.stop_event.is_set():
                        return
                self.heartbeat()
            except Exception as error:
                # A temporary Drive I/O error must not crash the ComfyUI session.
                print(f'Model downloader: {type(error).__name__}; retrying Drive access.', flush=True)
            self.stop_event.wait(3)


def start_model_download_worker(drive_root='/content/drive/MyDrive/ComfyUI'):
    root = Path(drive_root).resolve()
    if not root.is_dir():
        raise RuntimeError('Mount Drive và chạy cell chuẩn bị thư mục trước.')
    previous = getattr(builtins, '_comfy_model_download_worker', None)
    if previous is not None and previous.is_alive():
        if previous.drive_root != root:
            raise RuntimeError('Một model downloader đang chạy với Drive root khác.')
        return previous
    worker = ModelDownloadWorker(root)
    worker.heartbeat(force=True)
    worker.start()
    builtins._comfy_model_download_worker = worker
    print('MODEL_DOWNLOADER_READY: agent có thể gửi yêu cầu tải model khi ComfyUI đang chạy.', flush=True)
    return worker


if __name__ == '__main__':
    if Path('/content/drive/MyDrive/ComfyUI').is_dir():
        start_model_download_worker()
