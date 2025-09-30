import importlib
import importlib.util
import os
import subprocess
import sys
from typing import Dict, Literal, Optional

from packaging.version import Version


python = sys.executable
index_url = os.environ.get("INDEX_URL", "")


def _check_torch_installed():
    try:
        import torch
        import torchvision
    except Exception:
        msg = (
            "Missing required pre-installed packages: torch, torchvision\n"
            "Install the PyTorch CUDA wheels from the appropriate index first, e.g.:\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu12x torch torchvision\n"
            "Replace the index URL and versions to match your CUDA runtime."
        )
        raise RuntimeError(msg)

    if not torch.version.cuda:
        raise RuntimeError("Detected CPU-only PyTorch. Install CUDA-enabled torch/vision/audio before installing this package.")


def get_cuda_version() -> str:
    _check_torch_installed()

    import torch
    return torch.version.cuda


def get_cuda_major() -> Optional[Literal["11", "12"]]:
    version = get_cuda_version()
    if not version:
        return None

    major = version.split(".")[0]
    if major not in ("11", "12"):
        return None
    return major


def version(package: str) -> Optional[Version]:
    try:
        return Version(importlib.import_module(package).__version__)
    except ModuleNotFoundError:
        return None


def is_installed(package: str) -> bool:
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run_python(command: str, env: Dict[str, str] = None) -> str:
    run_kwargs = {
        "args": f"\"{python}\" {command}",
        "shell": True,
        "env": os.environ if env is None else env,
        "encoding": "utf8",
        "errors": "ignore",
    }

    print(run_kwargs["args"])

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        print(f"Error running command: {command}", file=sys.stderr)
        raise RuntimeError(f"Error running command: {command}")

    return result.stdout or ""


def run_pip(command: str, env: Dict[str, str] = None) -> str:
    return run_python(f"-m pip {command}", env)
