#!/usr/bin/env python3
"""Small standard-library client for a ComfyUI server behind ngrok Basic Auth."""
import argparse
import base64
import getpass
import json
import os
from pathlib import Path
import re
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_CONFIG = Path(__file__).resolve().with_name('.comfy-agent.json')
DEFAULT_MODEL_CATALOG = Path(__file__).resolve().parent / 'models' / 'catalog.json'
MODEL_DOWNLOAD_NAMESPACE = '_agent_models'


class ClientError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the Basic Auth header to a redirected destination.
        return None


def validate_url(value):
    value = value.strip().rstrip('/')
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path or any(ch.isspace() for ch in value)):
        raise ClientError('URL phải là HTTPS gốc, không chứa mật khẩu, query hoặc đường dẫn.')
    try:
        parsed.port
    except ValueError:
        raise ClientError('Cổng trong URL không hợp lệ.') from None
    return value


def tls_context():
    # Python.org macOS installs can lack a default CA bundle; use OS trust roots.
    context = ssl.create_default_context()
    if not (os.environ.get('SSL_CERT_FILE') or os.environ.get('SSL_CERT_DIR')):
        if not context.get_ca_certs():
            for candidate in ('/etc/ssl/cert.pem', '/etc/ssl/certs/ca-certificates.crt'):
                if Path(candidate).is_file():
                    context.load_verify_locations(cafile=candidate)
                    break
    return context


def load_connection(path):
    config = {}
    if path.exists():
        if os.name == 'posix' and path.stat().st_mode & 0o077:
            raise ClientError(f'File cấu hình cần quyền riêng tư: chmod 600 {path.name}')
        config = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(config, dict):
            raise ClientError('File cấu hình phải là JSON object.')
    url = os.environ.get('COMFYUI_URL') or config.get('url')
    username = os.environ.get('COMFYUI_USERNAME') or config.get('username', 'comfyui')
    password = os.environ.get('COMFYUI_PASSWORD') or config.get('password')
    if not url or not password:
        raise ClientError('Chưa cấu hình. Chạy: python3 comfy_agent.py configure --url URL_NGROK')
    if not isinstance(username, str) or not isinstance(password, str):
        raise ClientError('Tên đăng nhập và mật khẩu phải là chuỗi.')
    if ':' in username or not username.isascii() or not password.isascii():
        raise ClientError('Basic Auth cần username ASCII không chứa dấu : và mật khẩu ASCII.')
    return validate_url(url), username, password


class ComfyClient:
    def __init__(self, url, username, password, timeout=20):
        self.url = validate_url(url)
        self.timeout = timeout
        self.authorization = 'Basic ' + base64.b64encode(
            f'{username}:{password}'.encode('ascii')
        ).decode('ascii')
        self.opener = urllib.request.build_opener(
            NoRedirect(), urllib.request.HTTPSHandler(context=tls_context())
        )

    def request(self, route, payload=None, *, binary=False, raw=None):
        headers = {'Authorization': self.authorization,
                   'ngrok-skip-browser-warning': 'true',
                   'Accept': '*/*' if binary else 'application/json'}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        if raw is not None:
            if payload is not None:
                raise ClientError('Không gửi đồng thời JSON và raw body.')
            data = raw
            headers['Content-Type'] = 'application/octet-stream'
        req = urllib.request.Request(self.url + route, data=data, headers=headers)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                body = response.read()
                content_type = response.headers.get('Content-Type', '')
        except urllib.error.HTTPError as error:
            with error:
                code = error.code
                ngrok_code = error.headers.get('Ngrok-Error-Code', '')
                # /prompt validation errors contain useful node IDs and field errors.
                detail = ''
                if route == '/prompt' and code == 400:
                    try:
                        rejected = json.loads(error.read(262144))
                        detail = '\n' + json.dumps(rejected, ensure_ascii=False)
                    except (ValueError, UnicodeError):
                        pass
            hint = {401: 'Sai username/mật khẩu hoặc chưa được phép đăng nhập.',
                    403: 'Truy cập bị từ chối.',
                    404: 'Không tìm thấy API; kiểm tra URL và phiên Colab.',
                    429: 'ngrok đang giới hạn lưu lượng.',
                    502: 'Tunnel không kết nối được ComfyUI.'}.get(code, '')
            suffix = f' ({ngrok_code})' if re.fullmatch(r'ERR_NGROK_\d+', ngrok_code) else ''
            raise ClientError(f'HTTP {code}{suffix}. {hint}{detail}') from None
        except (urllib.error.URLError, OSError) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise ClientError('Không xác minh được TLS. Kiểm tra CA hệ thống hoặc SSL_CERT_FILE.') from None
            raise ClientError(f'Lỗi kết nối ({type(reason).__name__}). Kiểm tra URL và runtime Colab.') from None
        if binary:
            if 'text/html' in content_type.lower():
                raise ClientError('Server trả HTML thay vì file; kiểm tra tunnel.')
            return body
        if not body:
            return None  # Some successful control routes (for example /free) have no JSON body.
        try:
            return json.loads(body)
        except (ValueError, UnicodeError):
            raise ClientError('Server trả dữ liệu không phải JSON; kiểm tra URL/tunnel.') from None

    def check(self):
        stats = self.request('/system_stats')
        if not isinstance(stats, dict) or 'system' not in stats:
            raise ClientError('Phản hồi không phải system_stats của ComfyUI.')
        return {'connected': True, 'url': self.url,
                'system': stats['system'], 'devices': stats.get('devices', [])}

    def upload_userdata(self, local_path, remote_path, overwrite=False):
        if (not remote_path or remote_path.startswith('/') or '\\' in remote_path
                or any(part in ('', '.', '..') for part in remote_path.split('/'))):
            raise ClientError('Tên file remote phải là đường dẫn tương đối trong userdata.')
        route = '/userdata/' + urllib.parse.quote(remote_path, safe='')
        route += '?overwrite=' + ('true' if overwrite else 'false')
        return self.request(route, raw=Path(local_path).read_bytes())

    def model_worker(self):
        route = '/userdata/' + urllib.parse.quote(MODEL_DOWNLOAD_NAMESPACE + '/worker.json', safe='')
        try:
            result = self.request(route)
        except ClientError as error:
            if str(error).startswith('HTTP 404') and 'ERR_NGROK_' not in str(error):
                raise ClientError('Bộ tải model chưa bật. Chạy cell bộ tải nền trong notebook rồi mở lại phiên ComfyUI.') from None
            raise
        if (not isinstance(result, dict) or result.get('protocol') != 1
                or not isinstance(result.get('heartbeat_at'), (int, float))
                or time.time() - result['heartbeat_at'] > 90):
            raise ClientError('Bộ tải model không còn heartbeat mới; bật lại trong runtime hiện tại.')
        return result

    def install_model(self, manifest):
        from colab.model_download_worker import validate_model_request
        job = {**manifest, 'id': uuid.uuid4().hex}
        try:
            validate_model_request(job)
        except ValueError as error:
            raise ClientError(str(error)) from None
        self.model_worker()
        route = '/userdata/' + urllib.parse.quote(
            MODEL_DOWNLOAD_NAMESPACE + '/requests/' + job['id'] + '.json', safe=''
        ) + '?overwrite=false'
        self.request(route, job)
        return {'submitted': True, 'job_id': job['id'], 'filename': job['filename'], 'size': job['size']}

    def model_status(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{32}', job_id):
            raise ClientError('Model job ID phải có 32 ký tự hex.')
        route = '/userdata/' + urllib.parse.quote(
            MODEL_DOWNLOAD_NAMESPACE + '/status/' + job_id + '.json', safe=''
        )
        try:
            return self.request(route)
        except ClientError as error:
            if not str(error).startswith('HTTP 404') or 'ERR_NGROK_' in str(error):
                raise
        # Distinguish an unprocessed request from a job that never existed.
        route = '/userdata/' + urllib.parse.quote(
            MODEL_DOWNLOAD_NAMESPACE + '/requests/' + job_id + '.json', safe=''
        )
        self.request(route)
        return {'job_id': job_id, 'state': 'queued'}

    def wait_model(self, job_id, timeout):
        deadline = time.monotonic() + timeout
        previous = None
        while time.monotonic() < deadline:
            result = self.model_status(job_id)
            progress = result.get('downloaded_bytes', 0)
            marker = (result['state'], progress // (100 * 1024 * 1024))
            if marker != previous:
                print(f"{job_id}: {result['state']} {progress:,} bytes", file=sys.stderr, flush=True)
                previous = marker
            if result['state'] == 'completed':
                return result
            if result['state'] == 'failed':
                raise ClientError(json.dumps(result, ensure_ascii=False))
            self.model_worker()
            time.sleep(5)
        raise ClientError(f'Hết thời gian chờ model {job_id}; dùng model-status, không gửi lại install-model.')

    def status(self, prompt_id):
        history = self.request('/history/' + urllib.parse.quote(prompt_id, safe=''))
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get('status', {})
            if status.get('status_str') == 'error':
                state = 'failed'
            elif status.get('completed'):
                state = 'completed'
            else:
                state = 'interrupted'
            return {'prompt_id': prompt_id, 'state': state, 'history': record}
        queue = self.request('/queue')
        for key, state in (('queue_running', 'running'), ('queue_pending', 'pending')):
            if any(len(item) > 1 and item[1] == prompt_id for item in queue.get(key, [])):
                return {'prompt_id': prompt_id, 'state': state}
        return {'prompt_id': prompt_id, 'state': 'unknown'}

    def wait(self, prompt_id, timeout, interval):
        deadline = time.monotonic() + timeout
        previous = None
        while time.monotonic() < deadline:
            result = self.status(prompt_id)
            if result['state'] != previous:
                print(f"{prompt_id}: {result['state']}", file=sys.stderr, flush=True)
                previous = result['state']
            if result['state'] == 'completed':
                return result
            if result['state'] in ('failed', 'interrupted'):
                raise ClientError(json.dumps(result, ensure_ascii=False))
            time.sleep(interval)
        raise ClientError(f'Hết thời gian chờ {prompt_id}; tác vụ có thể vẫn đang chạy. Dùng status, không gửi lại run.')

    def download(self, result, directory):
        if result['state'] != 'completed':
            raise ClientError('Chỉ tải kết quả của tác vụ đã hoàn tất.')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        seen = set()
        for node_id, output in result['history'].get('outputs', {}).items():
            for key in ('images', 'gifs', 'audio', 'videos'):
                for index, item in enumerate(output.get(key, [])):
                    if not isinstance(item, dict) or not isinstance(item.get('filename'), str):
                        continue
                    descriptor = {name: item.get(name, default) for name, default in
                                  (('filename', ''), ('subfolder', ''), ('type', 'output'))}
                    signature = json.dumps(descriptor, sort_keys=True)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    basename = re.sub(r'[^\w.\-]', '_', Path(item['filename']).name)
                    prefix = re.sub(r'[^\w\-]', '_', f"{result['prompt_id']}_{node_id}_{key}_{index}")
                    target = directory / f'{prefix}_{basename}'
                    if target.exists():
                        raise ClientError(f'File đã tồn tại, không ghi đè: {target}')
                    body = self.request('/view?' + urllib.parse.urlencode(descriptor), binary=True)
                    with target.open('xb') as stream:
                        stream.write(body)
                    paths.append(str(target.resolve()))
        return {'prompt_id': result['prompt_id'], 'files': paths}


def load_workflow(path, overrides):
    workflow = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(workflow, dict) and isinstance(workflow.get('prompt'), dict):
        workflow = workflow['prompt']
    if (not isinstance(workflow, dict) or not workflow
            or not all(isinstance(node, dict) and isinstance(node.get('class_type'), str)
                       and isinstance(node.get('inputs'), dict) for node in workflow.values())):
        raise ClientError('Cần workflow dạng API JSON (mỗi node có class_type và inputs), không phải JSON giao diện.')
    for override in overrides:
        try:
            field, encoded = override.split('=', 1)
            node_id, name = field.split('.', 1)
            value = json.loads(encoded)
        except ValueError:
            raise ClientError('--set cần dạng NODE.INPUT=JSON, ví dụ 3.seed=42 hoặc 6.text=\'"a cat"\'.') from None
        if node_id not in workflow or name not in workflow[node_id]['inputs']:
            raise ClientError(f'Không tìm thấy input {node_id}.{name} trong workflow.')
        workflow[node_id]['inputs'][name] = value
    return workflow


def configure(args):
    url = validate_url(args.url)
    username = args.username
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', username):
        raise ClientError('Username không hợp lệ.')
    if not sys.stdin.isatty():
        raise ClientError('Chạy configure trong Terminal để nhập mật khẩu ẩn.')
    password = getpass.getpass('COMFYUI_PASSWORD (ẩn): ')
    if not (6 <= len(password) <= 128 and password.isascii()
            and all(ch.isprintable() and not ch.isspace() for ch in password)):
        raise ClientError('Mật khẩu cần 6–128 ký tự ASCII, không khoảng trắng; dùng đúng mật khẩu phiên Colab.')
    # Verify before replacing a previously working configuration.
    result = ComfyClient(url, username, password).check()
    args.config.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=args.config.name + '.', dir=args.config.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump({'url': url, 'username': username, 'password': password}, stream)
        os.replace(temporary, args.config)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({'connected': result['connected'], 'config': str(args.config)}, ensure_ascii=False))


def positive_number(value):
    number = float(value)
    if not 0 < number < float('inf'):
        raise argparse.ArgumentTypeError('Giá trị phải lớn hơn 0 và hữu hạn.')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest='command', required=True)
    setup = commands.add_parser('configure', help='Nhập mật khẩu ẩn, kiểm tra rồi lưu cấu hình riêng trên máy.')
    setup.add_argument('--url', required=True)
    setup.add_argument('--username', default='comfyui')
    commands.add_parser('check', help='Kiểm tra kết nối và GPU; không chạy tạo ảnh.')
    commands.add_parser('queue', help='Xem hàng đợi.')
    models = commands.add_parser('models')
    models.add_argument('folder', nargs='?', help='Ví dụ checkpoints, diffusion_models; bỏ trống để liệt kê loại.')
    nodes = commands.add_parser('nodes')
    nodes.add_argument('class_name', nargs='?')
    upload = commands.add_parser('upload-userdata', help='Lưu file/workflow vào userdata; không ghi vào thư mục model.')
    upload.add_argument('local_path', type=Path)
    upload.add_argument('remote_path')
    upload.add_argument('--overwrite', action='store_true')
    commands.add_parser('model-worker', help='Kiểm tra bộ tải model nền trong runtime Colab.')
    install = commands.add_parser('install-model', help='Yêu cầu bộ tải nền cài model; không dừng ComfyUI.')
    install.add_argument('model', nargs='?', help='Tên trong catalog, ví dụ sdxl-base.')
    install.add_argument('--catalog', type=Path, default=DEFAULT_MODEL_CATALOG)
    install.add_argument('--manifest', type=Path, help='Manifest model tùy chọn thay cho tên trong catalog.')
    install.add_argument('--wait', action='store_true')
    install.add_argument('--timeout', type=positive_number, default=3600)
    model_status = commands.add_parser('model-status')
    model_status.add_argument('job_id')
    model_status.add_argument('--wait', action='store_true')
    model_status.add_argument('--timeout', type=positive_number, default=3600)
    for command in ('status', 'download'):
        sub = commands.add_parser(command)
        sub.add_argument('prompt_id')
        if command == 'download':
            sub.add_argument('--output', type=Path, default=Path('comfy-results'))
    run = commands.add_parser('run', help='Gửi API workflow một lần; --wait để chờ và tải kết quả.')
    run.add_argument('workflow', type=Path)
    run.add_argument('--set', action='append', default=[])
    run.add_argument('--wait', action='store_true')
    run.add_argument('--timeout', type=positive_number, default=1800)
    run.add_argument('--interval', type=positive_number, default=3)
    run.add_argument('--output', type=Path, default=Path('comfy-results'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'configure':
            configure(args)
            return 0
        client = ComfyClient(*load_connection(args.config))
        if args.command == 'check':
            result = client.check()
        elif args.command == 'queue':
            result = client.request('/queue')
        elif args.command == 'upload-userdata':
            result = client.upload_userdata(args.local_path, args.remote_path, args.overwrite)
        elif args.command == 'model-worker':
            result = client.model_worker()
        elif args.command == 'model-status':
            result = client.wait_model(args.job_id, args.timeout) if args.wait else client.model_status(args.job_id)
        elif args.command == 'install-model':
            if args.manifest:
                if args.model:
                    raise ClientError('Chọn tên model hoặc --manifest, không dùng cả hai.')
                manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
            else:
                catalog = json.loads(args.catalog.read_text(encoding='utf-8'))
                if args.model not in catalog:
                    raise ClientError('Chọn model trong catalog: ' + ', '.join(catalog))
                manifest = catalog[args.model]
            if not isinstance(manifest, dict):
                raise ClientError('Manifest model phải là JSON object.')
            result = client.install_model(manifest)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if not args.wait:
                return 0
            result = client.wait_model(result['job_id'], args.timeout)
        elif args.command in ('models', 'nodes'):
            route = '/models' if args.command == 'models' else '/object_info'
            name = args.folder if args.command == 'models' else args.class_name
            if name:
                route += '/' + urllib.parse.quote(name, safe='')
            result = client.request(route)
        elif args.command == 'status':
            result = client.status(args.prompt_id)
        elif args.command == 'download':
            result = client.download(client.status(args.prompt_id), args.output)
        else:
            workflow = load_workflow(args.workflow, args.set)
            client.check()
            # Deliberately no automatic retry: a lost POST response may still have queued a job.
            try:
                result = client.request('/prompt', {'prompt': workflow, 'client_id': str(uuid.uuid4())})
            except ClientError as error:
                raise ClientError(f'{error}\nNếu mất kết nối khi gửi, xem queue trước khi chạy lại để tránh gửi trùng.') from None
            if not isinstance(result, dict) or not isinstance(result.get('prompt_id'), str):
                raise ClientError('Server không trả prompt_id: ' + json.dumps(result, ensure_ascii=False))
            print(json.dumps({'submitted': True, **result}, ensure_ascii=False), flush=True)
            if not args.wait:
                return 0
            result = client.download(client.wait(result['prompt_id'], args.timeout, args.interval), args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ClientError, OSError, ValueError) as error:
        print(f'Lỗi: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Đã dừng client; tác vụ đã gửi có thể vẫn chạy trên ComfyUI.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
