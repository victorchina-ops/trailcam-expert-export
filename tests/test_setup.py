"""Small setup tests need no ML runtime, internet, or model downloads."""
import hashlib
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap
from trailcam import model_setup


class ModelDownloadTests(unittest.TestCase):
    def test_valid_cached_model_never_accesses_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights.bin"
            target.write_bytes(b"known-good")
            digest = hashlib.sha256(b"known-good").hexdigest()
            with patch.object(model_setup.urllib.request, "urlopen") as download:
                result = model_setup.download_verified("https://invalid.example", target, digest)
            self.assertEqual(result, target)
            download.assert_not_called()

    def test_offline_rejects_invalid_model_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights.bin"
            target.write_bytes(b"damaged")
            with patch.object(model_setup.urllib.request, "urlopen") as download:
                with self.assertRaisesRegex(RuntimeError, "offline mode"):
                    model_setup.download_verified("https://invalid.example", target, "0" * 64, offline=True)
            self.assertEqual(target.read_bytes(), b"damaged")
            download.assert_not_called()

    def test_checksum_failure_preserves_old_file_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights.bin"
            target.write_bytes(b"old-file")
            with patch.object(model_setup.urllib.request, "urlopen", return_value=io.BytesIO(b"bad-download")):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    model_setup.download_verified("https://invalid.example", target, "0" * 64, attempts=1)
            self.assertEqual(target.read_bytes(), b"old-file")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_verified_download_atomically_replaces_old_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights.bin"
            target.write_bytes(b"old-file")
            payload = b"correct-new-file"
            with patch.object(model_setup.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                model_setup.download_verified("https://invalid.example", target, hashlib.sha256(payload).hexdigest())
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(list(Path(directory).iterdir()), [target])


class InstallerTests(unittest.TestCase):
    def test_complete_environment_does_not_run_pip(self):
        with patch.object(bootstrap, "version_matches", return_value=True), \
             patch.object(bootstrap, "clip_is_compatible", return_value=True), \
             patch.object(bootstrap, "pip_install") as install:
            bootstrap.install_dependencies()
        install.assert_not_called()

    def test_only_missing_dependency_is_installed(self):
        def matches(name, version):
            return name != "Pillow"
        with patch.object(bootstrap, "version_matches", side_effect=matches), \
             patch.object(bootstrap, "clip_is_compatible", return_value=True), \
             patch.object(bootstrap, "pip_install") as install:
            bootstrap.install_dependencies()
        install.assert_called_once_with(["Pillow==12.3.0"])

    def test_failed_gpu_wheel_install_falls_back_to_cpu(self):
        def matches(name, version):
            return name not in ("torch", "torchvision")
        with patch.object(bootstrap, "version_matches", side_effect=matches), \
             patch.object(bootstrap, "clip_is_compatible", return_value=True), \
             patch.object(bootstrap, "nvidia_cuda_index", return_value=("cu128", "GPU candidate")), \
             patch.object(bootstrap, "pip_install", side_effect=[bootstrap.subprocess.CalledProcessError(1, "pip"), None]) as install:
            bootstrap.install_dependencies()
        self.assertEqual(install.call_count, 2)
        self.assertIn("https://download.pytorch.org/whl/cu128", install.call_args_list[0].args[0])
        self.assertIn("https://download.pytorch.org/whl/cpu", install.call_args_list[1].args[0])

    def test_cuda_local_version_suffix_matches(self):
        with patch.object(bootstrap, "installed_version", return_value="2.11.0+cu128"):
            self.assertTrue(bootstrap.version_matches("torch", "2.11.0"))


class LauncherTests(unittest.TestCase):
    def test_help_and_version_skip_setup_and_runtime_imports(self):
        for argument in ("--help", "--version"):
            with self.subTest(argument=argument), \
                 patch.object(bootstrap, "install_dependencies") as install, \
                 patch.object(bootstrap, "verify_runtime") as runtime, \
                 patch.object(bootstrap.venv, "EnvBuilder") as environment, \
                 patch.object(bootstrap.subprocess, "call") as child, \
                 redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as stopped:
                    bootstrap.main([argument])
                self.assertEqual(stopped.exception.code, 0)
                self.assertTrue(output.getvalue().strip())
                install.assert_not_called()
                runtime.assert_not_called()
                environment.assert_not_called()
                child.assert_not_called()

    def test_error_pause_preserves_exit_status(self):
        with patch.object(bootstrap, "disposable_windows_console", return_value=True), \
             patch("builtins.input", return_value="") as wait:
            self.assertEqual(bootstrap.finish(7), 7)
        wait.assert_called_once()

    def test_success_and_noninteractive_error_never_pause(self):
        for code, interactive in ((0, True), (1, False)):
            with patch.object(bootstrap, "disposable_windows_console", return_value=interactive), \
                 patch("builtins.input") as wait:
                self.assertEqual(bootstrap.finish(code), code)
            wait.assert_not_called()

    def test_no_pause_override_always_wins(self):
        with patch.dict(bootstrap.os.environ, {"TRAILCAM_NO_PAUSE": "1"}):
            self.assertFalse(bootstrap.disposable_windows_console())


if __name__ == "__main__":
    unittest.main()
