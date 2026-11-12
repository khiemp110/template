import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from colab.model_download_worker import ModelDownloadWorker, validate_model_request
from comfy_agent import ClientError, ComfyClient


class ModelDownloaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'ComfyUI'
        self.root.mkdir()
        self.worker = ModelDownloadWorker(self.root)
        self.data = b'test-model-content' * 64
        self.job = {
            'id': '1' * 32, 'repo_id': 'publisher/test-model', 'revision': 'a' * 40,
            'source_filename': 'weights/model.safetensors', 'filename': 'model.safetensors',
            'folder': 'checkpoints', 'size': len(self.data),
            'sha256': hashlib.sha256(self.data).hexdigest(),
        }

    def test_manifest_rejects_executable_traversal_and_unpinned_revision(self):
        validate_model_request(self.job)
        for key, value in (
            ('filename', '../escape.safetensors'), ('filename', 'code.py'),
            ('source_filename', '../escape.safetensors'), ('revision', 'main'),
            ('repo_id', 'https://untrusted.example/model'), ('folder', 'custom_nodes'),
            ('sha256', ''), ('size', True), ('size', 100_000_000_001),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_model_request({**self.job, key: value})

    def test_pinned_download_then_verify_and_atomic_install(self):
        data = self.data

        class Curl:
            returncode = 0

            def __init__(self, command, **kwargs):
                self.command = command
                Path(command[command.index('--output') + 1]).write_bytes(data)

            def poll(self):
                return 0

        with patch('colab.model_download_worker.subprocess.Popen', side_effect=Curl) as run:
            self.worker.install(self.job)
        command = run.call_args.args[0]
        self.assertIn('/resolve/' + self.job['revision'] + '/', command[-1])
        self.assertTrue(command[-1].startswith('https://huggingface.co/'))
        target = self.root / 'models/checkpoints/model.safetensors'
        self.assertEqual(target.read_bytes(), data)
        self.assertFalse((self.worker.partials / (self.job['id'] + '.part')).exists())
        status = json.loads((self.worker.statuses / (self.job['id'] + '.json')).read_text())
        self.assertEqual(status['state'], 'completed')
        self.assertEqual(status['sha256'], self.job['sha256'])

    def test_bad_sha_does_not_install_partial(self):
        partial = self.worker.partials / (self.job['id'] + '.part')
        partial.write_bytes(b'X' * len(self.data))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.worker.install(self.job)
        self.assertFalse((self.root / 'models/checkpoints/model.safetensors').exists())
        self.assertTrue(partial.exists())

    def test_existing_model_is_verified_and_never_overwritten(self):
        target = self.root / 'models/checkpoints/model.safetensors'
        target.parent.mkdir()
        target.write_bytes(self.data)
        with patch('colab.model_download_worker.subprocess.Popen') as run:
            self.worker.install(self.job)
            run.assert_not_called()
        target.write_bytes(b'different-existing-data')
        with self.assertRaises(ValueError):
            self.worker.install(self.job)
        self.assertEqual(target.read_bytes(), b'different-existing-data')

    def test_symlink_cannot_escape_drive_root(self):
        outside = Path(self.directory.name) / 'outside'
        outside.mkdir()
        (self.root / 'models/checkpoints').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.worker.install(self.job)
        self.assertEqual(list(outside.iterdir()), [])

    def test_job_failure_is_written_and_thread_survives(self):
        request = self.worker.requests / (self.job['id'] + '.json')
        request.write_text(json.dumps({**self.job, 'revision': 'main'}))
        self.worker.start()
        self.addCleanup(lambda: (self.worker.stop_event.set(), self.worker.join(timeout=5)))
        import time
        deadline = time.monotonic() + 5
        status = self.worker.statuses / request.name
        while time.monotonic() < deadline and not status.exists():
            time.sleep(0.01)
        self.assertEqual(json.loads(status.read_text())['state'], 'failed')
        self.assertTrue(self.worker.is_alive())

    def test_offline_tunnel_is_not_reported_as_missing_worker(self):
        client = object.__new__(ComfyClient)
        with patch.object(client, 'request', side_effect=ClientError('HTTP 404 (ERR_NGROK_3200). Offline')):
            with self.assertRaisesRegex(ClientError, 'ERR_NGROK_3200'):
                client.model_worker()


if __name__ == '__main__':
    unittest.main()
