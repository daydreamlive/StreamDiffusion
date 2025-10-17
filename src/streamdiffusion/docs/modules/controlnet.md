# ControlNet Module

## Overview

The ControlNet Module enables integration of ControlNet models into the StreamDiffusion pipeline, allowing conditional guidance from external control images (e.g., edge maps, poses, depth). It supports multiple ControlNets simultaneously, each with independent preprocessors, conditioning scales, and enable/disable states. The module is designed for realtime streaming, with efficient tensor preparation, caching for SDXL conditioning, and seamless fallback between PyTorch and TensorRT engines.

Key features:
- Dynamic addition/removal/reordering of ControlNets.
- Automatic preprocessing via the [Preprocessing Orchestrator](../preprocessing/orchestrators.md).
- UNet hook integration for injecting residuals without modifying core pipeline (see [Hook-Module System](../hooks.md)).
- TensorRT acceleration for low-latency inference.
- SDXL support with optimized micro-conditioning caching.

The core implementation is in [`controlnet_module.py`](../../../modules/controlnet_module.py), with TensorRT support in [`controlnet_engine.py`](../../../acceleration/tensorrt/runtime_engines/controlnet_engine.py), export wrappers in [`controlnet_export.py`](../../../acceleration/tensorrt/export_wrappers/controlnet_export.py), and model definitions in [`controlnet_models.py`](../../../acceleration/tensorrt/models/controlnet_models.py).

## Configuration

ControlNets are configured via `ControlNetConfig`:

- `model_id`: str - Path or HuggingFace ID of the ControlNet model (e.g., "lllyasviel/sd-controlnet-canny").
- `preprocessor`: Optional[str] - Preprocessor name (e.g., "canny", "openpose"; see [Realtime Processors](../preprocessing/processors.md)).
- `conditioning_scale`: float - Guidance strength (default: 1.0).
- `enabled`: bool - Whether to use this ControlNet (default: True).
- `conditioning_channels`: Optional[int] - Input channels for model (auto-detected).
- `preprocessor_params`: Optional[Dict[str, Any]] - Params for preprocessor (e.g., thresholds).

Configs are loaded via [Config Management](../config.md) and managed by `StreamParameterUpdater` for runtime updates.

## Usage

### Initialization and Installation

The module is installed into a `StreamDiffusion` instance:

```python
from streamdiffusion import StreamDiffusion
from streamdiffusion.modules import ControlNetModule, ControlNetConfig

stream = StreamDiffusion(...)  # From config or manual setup
controlnet_module = ControlNetModule(device="cuda", dtype=torch.float16)
controlnet_module.install(stream)  # Registers UNet hook and exposes collections
```

### Adding ControlNets

Add via config or programmatically:

```python
cfg = ControlNetConfig(
    model_id="lllyasviel/sd-controlnet-canny",
    preprocessor="canny",
    conditioning_scale=1.0,
    preprocessor_params={"threshold_low": 100, "threshold_high": 200}
)
controlnet_module.add_controlnet(cfg, control_image="path/to/image.jpg")
```

Supports TensorRT: If an engine exists for the `model_id`, it auto-switches.

### Updating Control Images

Efficient per-frame updates (pipelined or sync for feedback processors):

```python
# Update single index
controlnet_module.update_control_image_efficient("new_control.jpg", index=0)

# Bulk update all
controlnet_module.update_control_image_efficient("new_stream.jpg")  # Applies to all active
```

Images are preprocessed (e.g., canny edges) and cached for batch/device alignment.

### Managing Scales and State

```python
controlnet_module.update_controlnet_scale(0, 0.8)  # Reduce strength
controlnet_module.update_controlnet_enabled(0, False)  # Disable
controlnet_module.remove_controlnet(0)  # Remove
controlnet_module.reorder_controlnets_by_model_ids(["canny", "pose"])  # Reorder
```

### Integration in Pipeline

The module registers a `UnetHook` that computes residuals per step:

- Inputs: Latent `x_t`, timesteps `t_list`, embeddings.
- Outputs: `UnetKwargsDelta` with `down_block_additional_residuals` (list of tensors) and `mid_block_additional_residual`.
- Multi-ControlNet: Residuals are summed for combined guidance.
- Caching: Prepared tensors reused across steps; SDXL cond cached per frame.

In [`pipeline.py`](../../../pipeline.py), the hook is called in `unet_step()` to augment UNet kwargs.

## TensorRT Integration

### Export

For acceleration, export ControlNet to ONNX using `SDXLControlNetExportWrapper` (handles SDXL `added_cond_kwargs`):

```python
from streamdiffusion.acceleration.tensorrt.export_wrappers import SDXLControlNetExportWrapper
import torch.onnx

controlnet = ControlNetModel.from_pretrained("model_id")
wrapper = SDXLControlNetExportWrapper(controlnet)

# Sample inputs for SDXL (7 inputs)
sample_input = (sample, timestep, encoder_hidden_states, controlnet_cond, 
                conditioning_scale, text_embeds, time_ids)

torch.onnx.export(wrapper, sample_input, "controlnet.onnx", 
                  input_names=["sample", "timestep", "encoder_hidden_states", 
                               "controlnet_cond", "conditioning_scale", 
                               "text_embeds", "time_ids"],
                  output_names=[f"down_block_{i:02d}" for i in range(9)] + ["mid_block"],
                  dynamic_axes={...})  # See controlnet_models.py for full config
```

Outputs 9 down blocks (320/640/1280 channels, progressive downsampling) + mid block.

### Runtime Engine

The `ControlNetModelEngine` loads the TRT engine:

```python
from streamdiffusion.acceleration.tensorrt.runtime_engines import ControlNetModelEngine
import polygraphy.cuda as cuda

stream = cuda.Stream()
engine = ControlNetModelEngine("controlnet.engine", stream, model_type="sdxl")

# Inference (auto-handles inputs/outputs)
down_blocks, mid_block = engine(sample, timestep, encoder_hidden_states, 
                                controlnet_cond, scale=1.0, 
                                text_embeds=..., time_ids=...)
```

- Dynamic shapes: Batch 1-4, resolutions 384-1024 (latent 48-128).
- Caching: Shape resolution and buffer allocation.
- Fallback: If no engine, uses PyTorch model.
- Model defs: `ControlNetTRT`/`ControlNetSDXLTRT` in [`controlnet_models.py`](../../../acceleration/tensorrt/models/controlnet_models.py) for builder profiles.

Engines are pooled by `model_id` in the module for auto-substitution.

## Examples

### Basic Canny Control

```python
# Config
cfg = {
    "pipeline_hooks": {
        "controlnet": [
            {"model_id": "lllyasviel/sd-controlnet-canny", "preprocessor": "canny", "conditioning_scale": 1.0}
        ]
    }
}

# Load and generate
stream = StreamDiffusion.from_config("config.yaml")
stream.update_control_image("edge_image.jpg")
images = stream(batch_size=1)  # Guided by canny edges
```

### Multi-ControlNet with TensorRT

```python
# Add multiple
controlnet_module.add_controlnet(ControlNetConfig("controlnet-canny-trt", preprocessor="canny"))
controlnet_module.add_controlnet(ControlNetConfig("controlnet-openpose-trt", preprocessor="openpose"))

# Stream with updates
while streaming:
    stream.update_control_image("current_pose.jpg")  # Updates both
    images = stream()
```

### Realtime Feedback Loop

Combine with feedback processors (e.g., latent_feedback):

```python
cfg = ControlNetConfig("controlnet-depth", preprocessor="depth_tensorrt")
controlnet_module.add_controlnet(cfg)

# In loop: Previous output feeds next input via orchestrator
stream.update_control_image(previous_image)  # Auto-preprocessed to depth
```

For full streaming setup, see [StreamDiffusionWrapper](../wrapper.md) and [Multi-Stage Processing](../pipeline.md).

---

*See [Index](../index.md) for all documentation. For acceleration details, see [TensorRT Acceleration](../acceleration/tensorrt.md).*