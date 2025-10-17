# TensorRT Acceleration

## Overview

TensorRT acceleration optimizes StreamDiffusion for realtime performance by compiling PyTorch models to TensorRT engines, supporting dynamic batch/resolution (384-1024), FP16, and CUDA graphs. Engines are built for UNet, VAE (encoder/decoder), ControlNet, Safety Checker. The system auto-fallbacks to PyTorch on OOM, with engine pooling for ControlNet.

Key components:
- **EngineBuilder**: Exports ONNX, optimizes, builds TRT (static/dynamic shapes).
- **EngineManager**: Manages paths, compiles/loads engines (UNet/VAE/ControlNet).
- **Runtime Engines**: UNet2DConditionModelEngine, AutoencoderKLEngine, ControlNetModelEngine (infer with shape cache).
- **Export Wrappers**: UnifiedExportWrapper for UNet+ControlNet+IPAdapter (handles kwargs, scales).
- **Utilities**: Engine class (buffers, infer), preprocess/decode helpers.

Files: [`builder.py`](../../../acceleration/tensorrt/builder.py), [`engine_manager.py`](../../../acceleration/tensorrt/engine_manager.py), [`utilities.py`](../../../acceleration/tensorrt/utilities.py), wrappers in `export_wrappers/`.

## Usage

### Engine Building

Use `EngineManager` in wrapper init (build_engines_if_missing=True):

```python
from streamdiffusion import StreamDiffusionWrapper

wrapper = StreamDiffusionWrapper(
    model_id_or_path="runwayml/stable-diffusion-v1-5",
    acceleration="tensorrt",
    engine_dir="engines",  # Output dir
    build_engines_if_missing=True  # Compile if missing
)
# Builds: unet.engine, vae_encoder.engine, vae_decoder