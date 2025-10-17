# Multi-Stage Processing

## Overview

Multi-stage processing in StreamDiffusion refers to the modular, extensible diffusion pipeline that supports realtime streaming generation. The core `StreamDiffusion` class ([`pipeline.py`](../../../pipeline.py)) orchestrates stages: preparation (prompt/timestep setup), denoising (UNet steps with CFG), and decoding (VAE), with hooks for [modules](../modules/) and [orchestrators](../preprocessing/orchestrators.md). It handles batching for efficiency, SDXL/Turbo models, LoRA/LCM, and TensorRT acceleration.

Key features:
- **Streaming**: Frame-by-frame generation with buffer for multi-frame consistency.
- **CFG Modes**: "none", "full", "self", "initialize" for guidance (self-attention based).
- **Batching**: Denoising batch for speed (use_denoising_batch), frame buffer.
- **Hooks**: Integration points for ControlNet/IPAdapter residuals, IPAdapter scales.
- **Model Detection**: Auto-detects SD1.5/SDXL/Turbo for conditional kwargs.
- **Optimizations**: EMA timing, similar image filter, SDXL cond caching.

The pipeline supports img2img/txt2img via `__call__` and `txt2img`. For wrapper usage, see [StreamDiffusionWrapper](../wrapper.md).

## Stages

### 1. Preparation (`prepare()`)

Sets up embeddings, timesteps, noise:

- **Prompt Encoding**: Text to embeds (SD1.5: 2 tensors; SDXL: 4 with pooled/time IDs).
  - CFG: Uncond/cond cat or repeat based on mode.
  - Hooks: `embedding_hooks` modify `EmbedsCtx` (e.g., blending).
- **Timesteps**: Subset from scheduler (LCM), alpha/beta scalings.
- **Noise**: Randn init, stock for self-CFG.
- **SDXL Cond**: Pooled embeds/time IDs (orig/target size, crops), cached per batch/CFG.

Example:

```python
stream.prepare(prompt="A cat", guidance_scale=7.5, cfg_type="self", seed=42)
```

### 2. Denoising (`predict_x0_batch()` / `unet_step()`)

Core generation loop:

- **Input**: Latent noise (txt2img) or encoded image (img2img).
- **Batch**: Repeat timesteps/noise for multi-frame (frame_buffer_size).
- **UNet Call**: Per step/timestep:
  - Inputs: Sample, timestep, embeds, SDXL cond (`added_cond_kwargs`).
  - Hooks: `unet_hooks` add residuals/scales (`UnetKwargsDelta` with down/mid residuals, extra like ipadapter_scale).
  - CFG: Uncond/cond blending ("full": cat batch; "self": stock noise; "initialize": uncond first).
  - Output: Model pred, denoised via scheduler (x0 from xt - beta*pred / alpha).
- **Loop**: Batch for speed (use_denoising_batch), or single-step for low VRAM.
- **TensorRT**: Auto-detects engine, passes extras/residuals.

Multi-stage: Pre-latent hooks (orchestrators), per-step UNet with modules, post-latent hooks.

Example in `__call__`:

```python
x_t_latent = torch.randn(...)  # Or encode_image(x)
x_0_pred_out = stream.predict_x0_batch(x_t_latent)
```

### 3. Decoding (`decode_image()` / VAE)

- Scales latent by vae_scale_factor, decodes to pixels.
- Post-image hooks (orchestrators for upscale/sharpen).
- Similar filter skips if duplicate (realtime opt).

Full flow in `__call__`:

```python
x = stream.image_processor.preprocess(image)  # If img2img
x = stream._apply_image_preprocessing_hooks(x)  # Orchestrators
if similar_filter: x = filter(x)
x_t_latent = encode_image(x)
x_t_latent = _apply_latent_preprocessing_hooks(x_t_latent)  # Orchestrators
x_0_pred = predict_x0_batch(x_t_latent)
x_0_pred = _apply_latent_postprocessing_hooks(x_0_pred)  # Orchestrators
x_out = decode_image(x_0_pred)
x_out = _apply_image_postprocessing_hooks(x_out)  # Orchestrators
return x_out
```

## Integration

- **Modules**: ControlNet/IPAdapter register `unet_hooks` for residuals/scales in `unet_step`.
- **Orchestrators**: Hooks call orchestrators (e.g., `_apply_latent_preprocessing_hooks` → PipelinePreprocessingOrchestrator).
- **Updater**: `StreamParameterUpdater` ([doc](../stream_parameter_updater.md)) manages prompt/seed blending, controlnet/ipadapter configs.
- **TensorRT**: UNet engine in `unet_step` (positional args + extras).
- **LoRA/LCM**: `load_lora/fuse_lora` before prepare; LCM scheduler.
- **Multi-Stage**: Stages chain hooks/orchestrators; feedback via latent_feedback.py.

CFG Modes:
- "none": No guidance (guidance_scale=1).
- "full": Uncond/cond per sample.
- "self": Stock noise for uncond (efficient).
- "initialize": Uncond first, then cond.

SDXL: Added cond (text_embeds/time_ids), detected via model_detection.py.

## Usage Examples

### Basic Txt2Img

```python
from streamdiffusion import StreamDiffusion
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
stream = StreamDiffusion(pipe, t_index_list=[0, 999], width=512, height=512)
stream.prepare("A cat", guidance_scale=7.5)
images = stream.txt2img(batch_size=1)  # Or stream()
```

### Img2Img Streaming

```python
stream.prepare("Cat in style", cfg_type="self")
while streaming:
    prev_img = ...  # From previous frame
    images = stream(prev_img)  # Encodes, denoises, decodes with hooks
```

### With Modules/Orchestrators

Modules install hooks; orchestrators chain processors in hooks.

For custom stages, extend `__call__` or add hooks.

See [Config](../config.md) for t_index_list/CFG setup.

---

*See [Index](../index.md) for all documentation. For parameters, see [StreamParameterUpdater](../stream_parameter_updater.md).*