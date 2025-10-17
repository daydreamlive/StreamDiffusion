# Installation Guide

This guide covers the complete installation process for StreamDiffusion, including all dependencies, TensorRT acceleration, and the real-time demo interface.

## Prerequisites

Before starting, ensure you have the following installed on your system:

- **Conda** (Miniconda or Anaconda)
- **Node.js** (for the frontend interface)
- **NVIDIA GPU** with CUDA support
- **Git**

## Step 1: Clone the Repository

```bash
git clone https://github.com/livepeer/StreamDiffusion
cd StreamDiffusion
```

## Step 2: Create Conda Environment

Create and activate a new conda environment with Python 3.10:

```bash
conda create -n streamdiffusion python=3.11
conda activate streamdiffusion
```

## Step 3: Install PyTorch

Install PyTorch with CUDA support matching your system's CUDA version. Check your CUDA version with:

```bash
nvidia-smi
```

Then install PyTorch from the official website: https://pytorch.org/get-started/locally/

For CUDA 12.9:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

For other CUDA versions, adjust the URL accordingly (e.g., `cu128` for CUDA 12.8, `cu13` for CUDA 13). Note that CUDA is backwards compatible to your nvidia-smi version.

## Step 4: Install StreamDiffusion Core

Install the base StreamDiffusion package:

```bash
pip install -e .
```

## Step 5: Install Additional Dependencies

Install CUDA Python bindings and ONNX Runtime:

```bash
pip install cuda-python==12.9.0 onnxruntime
```

**Note:** Match the `cuda-python` version to your CUDA version (e.g., `12.9.0` for CUDA 12.9).

## Step 6: Install TensorRT

Run the TensorRT installation script:

```bash
python -m streamdiffusion.tools.install-tensorrt
```

This script will download and install the appropriate TensorRT version for your system.

## Step 7: Install Features

Install TensorRT acceleration, ControlNet, and IPAdapter support:

```bash
pip install -e .[tensorrt,controlnet,ipadapter]
```

## Step 8: Install Demo Requirements

Install the requirements for the real-time img2img demo:

```bash
cd demo/realtime-img2img
pip install -r requirements.txt
cd ../..
```

## Step 9: Build Depth Anything TensorRT Engine (Optional)

If you plan to use depth-based ControlNet, you'll need to build a TensorRT engine for Depth Anything.

### Download Required Files

1. Download the Depth Anything ONNX model from:
   - https://huggingface.co/yuvraj108c/Depth-Anything-2-Onnx/blob/main/depth_anything_v2_vitl.onnx

2. Copy the following files to `models/Model/`:
   - `utilities.py`
   - `export_trt.py`
   - `depth_anything_v2_vitl.onnx`

**Reference:** https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt

### Build the Engine

```bash
cd models/Model
python export_trt.py --onnx-path ./depth_anything_v2_vitl.onnx --trt-path ./depth_anything_v2_vits.engine
cd ../..
```

> Note: Thank you to yuvraj108c for easy scripts to generate the Depth Anything v2 TRT Engine. In the future this will be automated in StreamDiffusion. 

### Configure in YAML

Once built, you can reference the engine in your config files:

```yaml
controlnets:
  - type: depth
    preprocessor_params:
      engine_path: "../models/Model/depth_anything_v2_vits.engine"
```

## Step 10: Build Frontend

Build the web frontend for the real-time demo:

```bash
cd demo/realtime-img2img/frontend
npm install
npm run build
cd ../../..
```

## Step 11: Run the Demo

You're now ready to run StreamDiffusion! Start the real-time img2img demo:

```bash
cd demo/realtime-img2img
python main.py
```

The server will start, and you can access the web interface (typically at `http://localhost:7860`).

## Troubleshooting

### CUDA Version Mismatch

If you encounter CUDA-related errors, ensure that:
- Your PyTorch CUDA version matches your system CUDA version
- The `cuda-python` package version matches your CUDA version
- TensorRT is compatible with your CUDA version

### TensorRT Engine Build Failures

If TensorRT engine building fails:
1. Verify that TensorRT is properly installed: `python -c "import tensorrt; print(tensorrt.__version__)"`
2. Check that your ONNX model is compatible with your TensorRT version
3. Ensure you have enough GPU memory available

### Frontend Build Issues

If npm build fails:
1. Verify Node.js is installed: `node --version`
2. Clear npm cache: `npm cache clean --force`
3. Delete `node_modules` and `package-lock.json`, then run `npm install` again

## Next Steps

After installation, explore the documentation:

- [Configuration Guide](config.md) - Learn how to configure StreamDiffusion
- [Runtime Control](runtime_control.md) - Real-time parameter control
- [ControlNet Module](modules/controlnet.md) - Conditional guidance setup
- [IPAdapter Module](modules/ipadapter.md) - Style adaptation
- [TensorRT Acceleration](acceleration/tensorrt.md) - Optimize performance

## Verification

To verify your installation, run a simple test:

```python
import streamdiffusion
import torch

print(f"StreamDiffusion installed successfully")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

If all imports succeed and CUDA is available, your installation is complete!

