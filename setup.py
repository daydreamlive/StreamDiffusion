import os
import re
import sys

from setuptools import find_packages, setup

# Copied from pip_utils.py to avoid import
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


def get_cuda_constraint():
    cuda_version = os.environ.get("STREAMDIFFUSION_CUDA_VERSION") or \
                    os.environ.get("CUDA_VERSION")

    if not cuda_version:
        try:
            import torch
            cuda_version = torch.version.cuda
        except Exception:
            # might not be available during wheel build, so we have to ignore
            pass

    if not cuda_version:
        return ">=11,<13"

    parts = cuda_version.split(".")
    if len(parts) < 2:
        raise RuntimeError(f"Invalid CUDA version: {cuda_version}")
    return f"~={parts[0]}.{parts[1]}"


if any(cmd in sys.argv for cmd in ("install", "develop")):
    _check_torch_installed()

_deps = [
    f"cuda-python{get_cuda_constraint()}",
    "xformers==0.0.30",
    "diffusers==0.35.0",
    "transformers==4.56.0",
    "accelerate==1.10.0",
    "huggingface_hub==0.35.0",
    "Pillow==11.0.0",
    "fire==0.6.0",
    "omegaconf==2.3.0",
    "onnx==1.18.0",
    "onnxruntime==1.22.0",
    "onnxruntime-gpu==1.22.0",
    "protobuf==4.25.3",
    "colored==2.2.4",
    "pywin32==306;sys_platform == 'win32'",
    "onnx-graphsurgeon==0.5.8",
    "controlnet-aux==0.0.10",
    "diffusers-ipadapter @ git+https://github.com/livepeer/Diffusers_IPAdapter.git@405f87da42932e30bd55ee8dca3ce502d7834a99",
    "mediapipe==0.10.21",
    "insightface==0.7.3",
    # We can't really pin torch version as it depends on CUDA, but we check if it's pre-installed above
    "torch",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ @]+)(?:[!=<>~ @].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["xformers"] = deps_list("xformers")
extras["torch"] = deps_list("torch", "accelerate")
extras["tensorrt"] = deps_list("protobuf", "cuda-python", "onnx", "onnxruntime", "onnxruntime-gpu", "colored")
extras["controlnet"] = deps_list("onnx-graphsurgeon", "controlnet-aux")
extras["ipadapter"] = deps_list("diffusers-ipadapter", "mediapipe", "insightface")

extras["dev"] = extras["xformers"] + extras["torch"] + extras["tensorrt"]

install_requires = [
    deps["fire"],
    deps["omegaconf"],
    deps["diffusers"],
    deps["transformers"],
    deps["accelerate"],
    deps["huggingface_hub"],
    deps["Pillow"],
]


setup(
    name="streamdiffusion",
    version="0.1.1",
    description="real-time interactive image generation pipeline",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning diffusion pytorch stable diffusion audioldm streamdiffusion real-time",
    license="Apache 2.0 License",
    author="Aki, kizamimi, ddPn08, Verb, ramune, teftef6220, Tonimono, Chenfeng Xu, Ararat with the help of all our contributors (https://github.com/cumulo-autumn/StreamDiffusion/graphs/contributors)",
    author_email="cumulokyoukai@gmail.com",
    url="https://github.com/cumulo-autumn/StreamDiffusion",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"streamdiffusion": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=list(install_requires),
    extras_require=extras,
)
