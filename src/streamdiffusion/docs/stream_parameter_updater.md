# StreamParameterUpdater

## Overview

The `StreamParameterUpdater` ([`stream_parameter_updater.py`](../../../stream_parameter_updater.py)) manages dynamic runtime updates to streaming parameters in StreamDiffusion, enabling smooth transitions between prompts, seeds, ControlNets, IPAdapters, and hooks without restarting the pipeline. It uses caching for efficiency, blending (linear/SLERP) for multi-item interpolation, and thread-safe locks for realtime updates. As an [OrchestratorUser](../preprocessing/orchestrators.md#orchestratoruser), it attaches shared preprocessors.

Key features:
- **Prompt Blending**: Weighted multi-prompt embeds (cache hits for reuse, SLERP/linear).
- **Seed Blending**: Weighted noise interpolation (linear/SLERP, preserves magnitude).
- **Config Updates**: Diff-based changes to ControlNet/IPAdapter/hook setups (add/remove/update scales/enabled/params).
- **Embedding Caching**: IPAdapter style images preprocessed in parallel (sync/pipelined).
- **Timestep/Resolution**: Recalcs scalings/batches on changes (full or lightweight).
- **Normalization**: Optional weight sum-to-1 for prompts/seeds.

Updater is initialized in `StreamDiffusion` and called via `update_stream_params()` for batched updates.

## Usage

### Initialization

```python
from streamdiffusion import StreamDiffusion
stream = StreamDiffusion(...)
# Updater auto-init with stream.normalize_prompt_weights etc.
updater = stream._param_updater  # Private, use via stream.update_stream_params
```

### Prompt/Seed Blending

Multi-prompt/seed with weights:

```python
# Blended prompts
stream.update_stream_params(
    prompt_list=[("A cat", 0.7), ("A dog", 0.3)],
    prompt_interpolation_method="slerp",
    negative_prompt="blurry, low quality"
)

# Blended seeds
stream.update_stream_params(
    seed_list=[(123, 0.6), (456, 0.4)],
    seed_interpolation_method="linear"
)
```

### ControlNet/IPAdapter Updates

```python
# Update ControlNet configuration
stream.update_stream_params(
    controlnet_config=[
        {
            "model_id": "lllyasviel/sd-controlnet-canny",
            "preprocessor": "canny",
            "conditioning_scale": 0.8,
            "enabled": True
        }
    ]
)

# Update IPAdapter configuration
stream.update_stream_params(
    ipadapter_config={
        "ipadapter_model_path": "h94/IP-Adapter",
        "image_encoder_path": "openai/clip-vit-large-patch14",
        "scale": 0.7,
        "is_faceid": False
    }
)
```

### Hook Configuration Updates

```python
# Update preprocessing hooks
stream.update_stream_params(
    image_preprocessing_config=[
        {
            "type": "canny",
            "enabled": True,
            "params": {"threshold_low": 100, "threshold_high": 200}
        }
    ],
    latent_preprocessing_config=[
        {
            "type": "latent_feedback",
            "enabled": True,
            "params": {"blend_factor": 0.1}
        }
    ]
)
```

### Batch Updates

```python
# Update multiple parameters at once
stream.update_stream_params(
    guidance_scale=7.5,
    t_index_list=[0, 999],
    prompt_list=[("A beautiful landscape", 1.0)],
    controlnet_config=[...],
    image_preprocessing_config=[...]
)
```

## Advanced Features

### Weight Normalization

```python
# Enable automatic weight normalization
stream.update_stream_params(
    normalize_prompt_weights=True,
    normalize_seed_weights=True
)
```

### Cache Management

```python
# Get cache statistics
cache_info = stream._param_updater.get_cache_info()
print(f"Prompt cache hits: {cache_info['prompt_cache']['hits']}")
print(f"Seed cache misses: {cache_info['seed_cache']['misses']}")
```

### Thread Safety

All parameter updates are thread-safe and atomic. Multiple threads can call `update_stream_params()` simultaneously without race conditions.

## Integration

The updater integrates with:
- **Pipeline Hooks**: For real-time parameter application
- **Orchestrators**: For preprocessing/postprocessing updates  
- **Modules**: ControlNet and IPAdapter configuration management
- **Caching**: Efficient reuse of computed embeddings and noise

For detailed hook integration, see [Hook-Module System](hooks.md).

---

*See [Index](index.md) for all documentation.*