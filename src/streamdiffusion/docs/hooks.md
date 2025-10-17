# Hook-Module System

## Overview

The Hook-Module System in StreamDiffusion provides a flexible mechanism for extending and customizing the diffusion pipeline without modifying the core implementation. Hooks are callable functions that can be injected at specific stages of the generation process, such as embedding preparation, UNet denoising steps, image/latent processing, and more. This system promotes modularity, allowing users to add features like custom conditioning, post-processing effects, or dynamic parameter adjustments.

Hooks are particularly useful for integrating advanced modules (e.g., ControlNet, IPAdapter) or implementing realtime adaptations in streaming scenarios. They operate on context objects that carry relevant tensors and metadata, enabling non-destructive modifications via return values (e.g., deltas to augment UNet kwargs).

The system is defined in [`hooks.py`](hooks.py) and integrated into the main [`pipeline.py`](pipeline.py) and [`StreamParameterUpdater`](stream_parameter_updater.py).

## Key Concepts

### Context Objects

Hooks receive and return context dataclasses that encapsulate the state at each stage. These provide access to tensors (e.g., latents, embeddings) and metadata (e.g., timesteps, dimensions).

- **`EmbedsCtx`**: Context for text embedding hooks.
  - `prompt_embeds`: Torch tensor [batch, seq_len, dim] of positive prompt embeddings.
  - `negative_prompt_embeds`: Optional torch tensor [batch, seq_len, dim] for negative prompts.
  - *Purpose*: Modify or augment embeddings before UNet input (e.g., add custom noise or blending).

- **`StepCtx`**: Context for UNet denoising step hooks.
  - `x_t_latent`: Torch tensor of current latent (possibly CFG-expanded).
  - `t_list`: Torch tensor of timesteps (possibly CFG-expanded).
  - `step_index`: Optional int for current step in total steps.
  - `guidance_mode`: String ("none", "full", "self", "initialize") indicating CFG mode.
  - `sdxl_cond`: Optional dict with SDXL micro-conditioning tensors.
  - *Purpose*: Inspect or alter state during each denoising iteration (e.g., inject dynamic guidance).

- **`UnetKwargsDelta`**: Delta object returned by UNet hooks to modify UNet call arguments.
  - `down_block_additional_residuals`: Optional list of torch tensors for down-block residuals.
  - `mid_block_additional_residual`: Optional torch tensor for mid-block residual.
  - `added_cond_kwargs`: Optional dict of additional conditioning kwargs (e.g., ControlNet outputs).
  - `extra_unet_kwargs`: Optional dict for direct UNet kwargs (e.g., scales, adapters).
  - *Purpose*: Non-invasively augment UNet forward pass without rewriting the model.

- **`ImageCtx`**: Context for image-space processing hooks (pre/post VAE).
  - `image`: Torch tensor [B, C, H, W] in pixel space.
  - `width`: Image width (int).
  - `height`: Image height (int).
  - `step_index`: Optional int for multi-step processing.
  - *Purpose*: Apply effects like sharpening or upscaling on decoded images.

- **`LatentCtx`**: Context for latent-space processing hooks.
  - `latent`: Torch tensor [B, C, H/8, W/8] in latent space.
  - `timestep`: Optional torch tensor for diffusion context.
  - `step_index`: Optional int for multi-step processing.
  - *Purpose*: Modify latents before/after UNet (e.g., noise injection or feedback loops).

### Hook Types

Hooks are defined as type aliases for clarity:

- `EmbeddingHook = Callable[[EmbedsCtx], EmbedsCtx]`: Modifies embedding contexts.
- `UnetHook = Callable[[StepCtx], UnetKwargsDelta]`: Produces deltas for UNet steps.
- `ImageHook = Callable[[ImageCtx], ImageCtx]`: Processes image tensors.
- `LatentHook = Callable[[LatentCtx], LatentCtx]`: Processes latent tensors.

Hooks can be pre- or post-processing (e.g., `_apply_image_preprocessing_hooks` in pipeline).

## Usage

### Defining a Hook

Hooks are simple callables matching the type signature. Here's an example UnetHook that adds a custom residual:

```python
from streamdiffusion.hooks import StepCtx, UnetKwargsDelta
import torch

def custom_residual_hook(ctx: StepCtx) -> UnetKwargsDelta:
    # Example: Add a simple residual based on timestep
    if ctx.step_index is not None and ctx.step_index % 5 == 0:
        residual = torch.zeros_like(ctx.x_t_latent) + 0.01 * ctx.t_list.unsqueeze(1)
        return UnetKwargsDelta(
            down_block_additional_residuals=[residual] * 4,  # Assuming 4 down blocks
            mid_block_additional_residual=residual
        )
    return UnetKwargsDelta()  # No-op delta
```

### Registering Hooks

Hooks are typically registered via configuration in `StreamParameterUpdater` or directly in the `StreamDiffusion` pipeline. For example, using config:

In your YAML config (see [Config Management](../config.md)):

```yaml
pipeline_hooks:
  unet:
    - type: "custom"  # Or module path
      class: "path.to.CustomUnetHook"
      params:
        scale: 0.5
```

Or programmatically in `StreamDiffusion`:

```python
from streamdiffusion import StreamDiffusion

stream = StreamDiffusion(...)
# Assuming pipeline supports direct registration; check pipeline.py for exact API
stream.register_hook("unet", custom_residual_hook)
```

### Integration Points

- **Embedding Stage**: Called in `prepare()` before UNet, via `_apply_embedding_hooks`.
- **UNet Steps**: Invoked per step in `unet_step()`, accumulating deltas.
- **Image/Latent Processing**: Applied in `encode_image/decode_image` and hook methods like `_apply_image_preprocessing_hooks`.
- **Multi-Stage**: Supports chaining multiple hooks; order matters (pre- then post-).

For advanced usage with modules like ControlNet, hooks handle injection of `added_cond_kwargs`. See [Pipeline Documentation](../pipeline.md) for full integration.

### Best Practices

- Keep hooks lightweight (no heavy computation; use TensorRT for speed).
- Handle batching and device consistency (contexts are on GPU).
- Return unmodified contexts/deltas for no-op cases.
- Use locks in streaming scenarios to avoid race conditions (handled internally in updater).

For examples with specific modules, refer to [ControlNet](../modules/controlnet.md) and [IPAdapter](../modules/ipadapter.md).

---

*See [Index](../index.md) for all documentation.*