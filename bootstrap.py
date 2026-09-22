"""Install the local runtime if needed, then launch the image export.

Run with Python 3.11/3.12. Everything installed by this script stays in .venv;
global Python packages are never modified. Normal repeat runs do not use pip
or the network when their dependencies and model cache are already valid.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
CLIP_COMMIT = "b0c7af36eb99a5e103713e1792fc642f78059c39"
CLIP_URL = f"https://github.com/ultralytics/CLIP/archive/{CLIP_COMMIT}.zip"
PACKAGES = {
    "numpy": "2.3.5", "Pillow": "12.3.0", "ultralytics": "8.4.37",
    "paddlepaddle": "3.3.0",
}
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_matches(name: str, expected: str) -> bool:
    actual = installed_version(name)
    return actual is not None and actual.split("+", 1)[0] == expected


def clip_is_compatible() -> bool:
    try:
        dist = importlib.metadata.distribution("clip")
        origin = json.loads(dist.read_text("direct_url.json") or "{}")
        return (origin.get("vcs_info", {}).get("commit_id") == CLIP_COMMIT
                or origin.get("url", "").rstrip("/") == CLIP_URL)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return False


def pip_install(arguments: list[str]) -> None:
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *arguments]
    subprocess.run(command, check=True, cwd=ROOT)


def nvidia_cuda_index() -> tuple[str | None, str]:
    """Driver capability is a candidate only; actual kernel execution is checked later."""
    executable = shutil.which("nvidia-smi")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if candidate.is_file():
            executable = str(candidate)
    if not executable:
        return None, "No NVIDIA driver utility found; installing CPU PyTorch."
    try:
        result = subprocess.run([executable], capture_output=True, text=True, timeout=15)
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if result.returncode != 0 or not match:
            return None, "NVIDIA driver is unavailable; installing CPU PyTorch."
        capability = (int(match.group(1)), int(match.group(2)))
        if capability >= (12, 8):
            return "cu128", "NVIDIA driver supports CUDA 12.8; installing the tested GPU PyTorch build."
        if capability >= (12, 6):
            return "cu126", "NVIDIA driver supports CUDA 12.6; installing a compatible GPU PyTorch build."
        return None, "NVIDIA driver predates CUDA 12.6; CPU mode will work. Update the driver for GPU mode."
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"Could not query NVIDIA driver ({exc}); installing CPU PyTorch."


def install_dependencies(*, cpu_install: bool = False, repair_torch: bool = False) -> None:
    torch_ok = version_matches("torch", TORCH_VERSION) and version_matches("torchvision", TORCHVISION_VERSION)
    if not torch_ok or repair_torch:
        index, explanation = (None, "CPU installation requested.") if cpu_install else nvidia_cuda_index()
        print(explanation, flush=True)
        arguments = [f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
                     "--index-url", f"https://download.pytorch.org/whl/{index or 'cpu'}"]
        if repair_torch:
            arguments.append("--force-reinstall")
        try:
            pip_install(arguments)
        except subprocess.CalledProcessError:
            if index is None:
                raise
            print("GPU wheel installation failed; retrying the CPU build.", flush=True)
            pip_install([f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
                         "--index-url", "https://download.pytorch.org/whl/cpu", "--force-reinstall"])
    else:
        print("Compatible PyTorch and torchvision are already installed; skipping installation.", flush=True)

    needed = []
    for package, version in PACKAGES.items():
        # A working optional GPU Paddle distribution satisfies the same inference API.
        if package == "paddlepaddle" and version_matches("paddlepaddle-gpu", version):
            continue
        if not version_matches(package, version):
            needed.append(f"{package}=={version}")
    if not clip_is_compatible():
        needed.append(f"clip @ {CLIP_URL}")
    if needed:
        pip_install(needed)
    else:
        print("All inference dependencies are already installed; skipping pip.", flush=True)


def verify_runtime() -> dict:
    """Import together to catch broken installs and execute a real CUDA kernel."""
    import numpy
    import PIL
    import cv2
    import torch
    import torchvision
    import paddle
    import ultralytics
    import clip

    info = {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
            "paddle": paddle.__version__, "numpy": numpy.__version__,
            "pillow": PIL.__version__, "opencv": cv2.__version__, "device": "cpu"}
    if torch.cuda.is_available():
        try:
            value = (torch.ones((8, 8), device="cuda") @ torch.ones((8, 8), device="cuda")).sum().item()
            torch.cuda.synchronize()
            if value != 512:
                raise RuntimeError("CUDA kernel produced an unexpected result")
            info.update(device="cuda", gpu=torch.cuda.get_device_name(0))
        except Exception as exc:
            info["gpu_fallback_reason"] = str(exc)
    elif torch.version.cuda:
        info["gpu_fallback_reason"] = "CUDA runtime is installed, but no usable CUDA device is available."
    print(f"Runtime ready: detected {info['device']} capability; analysis applies the requested device setting.", flush=True)
    return info


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--cpu-install", action="store_true")
    parser.add_argument("--repair-torch", action="store_true")
    parser.add_argument("--check-setup", action="store_true")
    parser.add_argument("--offline", action="store_true")
    options, analysis_args = parser.parse_known_args(argv)
    if any(value in ("-h", "--help", "--version") for value in analysis_args):
        # The CLI parser uses only the standard library. Show usage before any
        # environment creation, package installation, runtime import or download.
        from trailcam.__main__ import arguments
        arguments(analysis_args)
        return 0
    if sys.version_info[:2] not in ((3, 11), (3, 12)) or sys.maxsize <= 2 ** 32:
        print("Install 64-bit Python 3.12 or 3.11, then run this launcher again.\n"
              "Windows: winget install -e --id Python.Python.3.12\n"
              "Download: https://www.python.org/downloads/windows/", file=sys.stderr)
        return 2
    python = _venv_python()
    if Path(sys.prefix).resolve() != VENV.resolve():
        if not python.is_file():
            if options.check_setup:
                print("Local .venv has not been created yet.")
                return 1
            print("Creating isolated Python environment in .venv...", flush=True)
            venv.EnvBuilder(with_pip=True).create(VENV)
        return subprocess.call([str(python), str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])], cwd=ROOT)
    os.chdir(ROOT)
    # Keep dependency caches/configuration within this project where supported.
    (ROOT / ".cache" / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
    os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if options.check_setup:
        missing = [name for name, version in PACKAGES.items()
                   if not version_matches(name, version)
                   and not (name == "paddlepaddle" and version_matches("paddlepaddle-gpu", version))]
        missing += [name for name, version in (("torch", TORCH_VERSION), ("torchvision", TORCHVISION_VERSION))
                    if not version_matches(name, version)]
        if not clip_is_compatible():
            missing.append("clip")
        print(json.dumps({"missing_or_incompatible": missing}, indent=2))
        return 1 if missing else 0
    if not options.offline:
        install_dependencies(cpu_install=options.cpu_install, repair_torch=options.repair_torch)
    runtime = verify_runtime()
    (ROOT / ".cache").mkdir(exist_ok=True)
    (ROOT / ".cache" / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    if options.setup_only:
        from trailcam.model_setup import ensure_models
        ensure_models(ROOT / "models", offline=options.offline)
        print("Setup complete. Paste images into images/ and run run_analysis.cmd.")
        return 0
    if options.offline:
        analysis_args.append("--offline")
    return subprocess.call([sys.executable, "-m", "trailcam", *analysis_args], cwd=ROOT)


def disposable_windows_console() -> bool:
    """True only for an interactive cmd /c window containing cmd and this Python.

    A pre-existing CMD shell does not have /c; a PowerShell-launched batch has
    that shell attached too. Redirected automation is never interactive.
    """
    if (os.name != "nt" or os.environ.get("TRAILCAM_NO_PAUSE") == "1"
            or os.environ.get("CI") or not sys.stdin.isatty() or not sys.stdout.isatty()):
        return False
    if not re.search(r"(?:^|\s)/c(?:\s|$)", os.environ.get("TRAILCAM_LAUNCH_COMMAND", ""), re.IGNORECASE):
        return False
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process_list = kernel32.GetConsoleProcessList
        process_list.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        process_list.restype = wintypes.DWORD
        processes = (wintypes.DWORD * 32)()
        return process_list(processes, len(processes)) == 2
    except (AttributeError, OSError):
        return False


def finish(exit_code: int) -> int:
    if exit_code and disposable_windows_console():
        try:
            input("Press Enter to close this error window...")
        except (EOFError, KeyboardInterrupt):
            pass
    return exit_code


if __name__ == "__main__":
    try:
        code = main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ImportError) as exc:
        print(f"Setup failed: {exc}\nCheck the README troubleshooting section, then rerun the launcher.", file=sys.stderr)
        code = 1
    raise SystemExit(finish(code))
