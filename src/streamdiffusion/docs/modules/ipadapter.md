# IPAdapter Module

## Overview

The IPAdapter Module enables image-to-image adaptation in StreamDiffusion by injecting image features (embeddings) into the UNet's attention layers, allowing style transfer, reference image guidance, or face ID consistency without full retraining. It supports multiple IPAdapters (e.g., standard, plus, face ID), dynamic scales, and efficient realtime updates via embedding caching. The module integrates seamlessly with the hook system, passing `extra_unet_kwargs` like `ipadapter_scale` to the UNet forward pass.

Key features:
- Loading and management of multiple IPAdapter models (HuggingFace or local).
- Automatic injection of custom attention processors into UNet (or TensorRT engines).
- Embedding computation from style/reference images via dedicated preprocessor.
- Face ID support using InsightFace for identity preservation.
- Streaming optimizations: Per-frame embedding updates, batch caching, weight normalization.

Core files: [`ipadapter_module.py`](../../../modules/ipadapter_module.py) for module logic, [`ipadapter_embedding.py`](../../../preprocessing/processors/ipadapter_embedding.py) for preprocessor, and [`unet_ipadapter_export.py`](../../../acceleration/tensorrt/export_wrappers/unet_ipadapter_export.py) for TensorRT export.

## Configuration

IPAdapters are configured similarly to ControlNets, via pipeline hooks in config:

- `ipadapter_model_path`: str - IPAdapter model path/ID (e.g., "h94/IP-Adapter", "h94/IP-Adapter-FaceID").
- `image_encoder_path`: str - Image encoder model path (e.g., "openai/clip-vit-large-patch14").
- `scale`: float - Adapter strength (default: 1.0; can be list for multi-adapter).
- `weight_type`: str - Weight computation ("linear" or "slerp" for multi-image blending).
- `num_image_tokens`: int - Number of image tokens (default: 4 for standard, 16 for plus).
- `is_faceid`: bool - Enable face ID mode (requires InsightFace).
- `insightface_model_name`: str - InsightFace model name for face ID (optional).
- Other: `device`, `dtype`, `cache_dir` for embeddings.

See [Config Management](../config.md) for YAML examples.

## Usage

### Initialization and Installation

```python
from streamdiffusion import StreamDiffusion
from streamdiffusion.modules import IPAdapterModule

stream = StreamDiffusion(...)
ipadapter_module = IPAdapterModule(device="cuda", dtype=torch.float16)
ipadapter_module.install(stream)  # Injects into UNet, registers hook
```

### Adding IPAdapters

```python
from streamdiffusion.modules.ipadapter_module import IPAdapterConfig

cfg = IPAdapterConfig(
    ipadapter_model_path="h94/IP-Adapter",
    image_encoder_path="openai/clip-vit-large-patch14",
    scale=0.8,
    num_image_tokens=4
)
ipadapter_module.add_ipadapter(cfg, style_image="style.jpg")  # Optional initial image
```

For Face ID:

```python
cfg_face = IPAdapterConfig(
    ipadapter_model_path="h94/IP-Adapter-FaceID",
    image_encoder_path="openai/clip-vit-large-patch14",
    scale=1.0,
    is_faceid=True,
    insightface_model_name="buffalo_l",
    style_image="reference_face.jpg"
)
ipadapter_module.add_ipadapter(cfg_face)
```

Supports TensorRT: Engines auto-substituted if available for the model.

### Updating Style Images

Efficient updates for streaming (computes embeddings, caches for reuse):

```python
# Single update
ipadapter_module.update_style_image("new_style.jpg", is_stream=True)  # Streaming mode

# Multi-image blending (weights normalized)
ipadapter_module.update_style_image(["style1.jpg", "style2.jpg"], weights=[0.6, 0.4])
```

Embeddings are computed via `IPAdapterEmbedding` preprocessor (CLIP ViT-H, 768-dim), with face ID using InsightFace for portrait adapter. Caches avoid recompute per step.

### Managing Adapters

```python
ipadapter_module.update_ipadapter_scale(0, 0.7)  # Adjust strength
ipadapter_module.remove_ipadapter(0)  # Remove
ipadapter_module.get_current_config()  # List active configs
```

### Integration in Pipeline

The module provides a `UnetHook` (via `build_unet_hook()`) that injects:

- `extra_unet_kwargs={"ipadapter_scale": scales_tensor}` into UNet call.
- Attention processors replace standard ones in UNet for feature injection.
- Multi-adapter: Layer weights blended (linear/SLERP) based on scales.

In `unet_step()` ([pipeline.py`](../../../pipeline.py)), the hook ensures adapters are applied per denoising step. For TensorRT, the export wrapper preserves processor logic in ONNX.

## TensorRT Integration

### Export

Export UNet with IPAdapter processors using `IPAdapterUNetExportWrapper`:

```python
from streamdiffusion.acceleration.tensorrt.export_wrappers import IPAdapterUNetExportWrapper
import torch.onnx

# Load UNet with IPAdapter
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
ipadapter = IPAdapter.from_pretrained("h94/IP-Adapter")  # Example
unet.set_ipadapter(ipadapter, scale=1.0)  # Inject processor

wrapper = IPAdapterUNetExportWrapper(unet)

sample_input = (sample, timestep, encoder_hidden_states, added_time_ids)  # Standard UNet inputs
torch.onnx.export(wrapper, sample_input, "unet_ipadapter.onnx",
                  input_names=["sample", "timestep", "encoder_hidden_states", "added_time_ids"],
                  output_names=["down_block_res_samples", "mid_block_res_sample", "time_embed", "time_text_embed"],
                  dynamic_axes={...},  # Batch, height, width dynamic
                  opset_version=17)
```

The wrapper handles multiple adapters, dynamic scales, and layer-specific weights during export.

### Runtime

During inference, if a TensorRT engine is loaded (via `engine_manager`), the module swaps the UNet for the engine, passing `ipadapter_scale` as extra kwarg. Embeddings are precomputed and injected via attention (preserved in TRT).

## IPAdapter Embedding Processor

The dedicated preprocessor (`ipadapter_embedding.py`) computes image embeddings:

- **Standard**: CLIP ViT-H/14 on style image → 768-dim features.
- **Face ID**: InsightFace + CLIP for identity embedding (portrait adapter).
- **Realtime**: Caches embeddings per image key, supports batch/streaming, GPU acceleration.
- Usage: Integrated automatically when `preprocessor="ipadapter_embedding"` in config.

Example standalone:

```python
from streamdiffusion.preprocessing.processors import IPAdapterEmbedding

preprocessor = IPAdapterEmbedding(pipeline_ref=stream)
embedding = preprocessor.process_image("style.jpg")  # Returns torch.Tensor
```

Supports multi-image: Averages or blends embeddings.

## Examples

### Basic Style Transfer

```python
# Config
cfg = {
    "pipeline_hooks": {
        "ipadapter": [
            {
                "ipadapter_model_path": "h94/IP-Adapter", 
                "image_encoder_path": "openai/clip-vit-large-patch14",
                "scale": 0.8, 
                "num_image_tokens": 4
            }
        ]
    }
}

stream = StreamDiffusion.from_config("config.yaml")
stream.update_style_image("art_style.jpg")
images = stream(prompt="A cat", batch_size=1)  # Cat in art style
```

### Face ID Consistency

```python
cfg = IPAdapterConfig(
    ipadapter_model_path="h94/IP-Adapter-FaceID", 
    image_encoder_path="openai/clip-vit-large-patch14",
    scale=1.0, 
    is_faceid=True,
    insightface_model_name="buffalo_l"
)
ipadapter_module.add_ipadapter(cfg, "reference_face.jpg")

# Generate consistent faces
stream.update_style_image("new_pose.jpg")  # Pose change, face preserved
images = stream(prompt="Portrait of person in different outfits")
```

### Multi-Adapter Blending

```python
ipadapter_module.add_ipadapter(IPAdapterConfig(
    ipadapter_model_path="h94/IP-Adapter", 
    image_encoder_path="openai/clip-vit-large-patch14",
    scale=0.6
))
ipadapter_module.add_ipadapter(IPAdapterConfig(
    ipadapter_model_path="h94/IP-Adapter-FaceID", 
    image_encoder_path="openai/clip-vit-large-patch14",
    scale=0.4, 
    is_faceid=True,
    insightface_model_name="buffalo_l"
))

# Blends styles with face consistency
stream.update_style_image(["style1.jpg", "face_ref.jpg"])
```

For full integration with ControlNet or multi-stage, see [Multi-Stage Processing](../pipeline.md) and [StreamParameterUpdater](../stream_parameter_updater.md).

---

*See [Index](../index.md) for all documentation. For preprocessing details, see [Realtime Processors](../preprocessing/processors.md).*