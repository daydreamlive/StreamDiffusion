# Config Management

## Overview

Config management in StreamDiffusion uses YAML/JSON files to define model, pipeline, blending, and module settings. The `config.py` module provides `load_config`/`save_config` for file I/O, validation for types/fields, and helpers like `create_wrapper_from_config` to instantiate `StreamDiffusionWrapper` from dicts. Supports legacy single prompts and new blending (prompt_list, seed_list), with normalization, interpolation methods.

Key functions:
- `load_config(path)`: Loads YAML/JSON, validates.
- `save_config(config, path)`: Writes validated config.
- `create_wrapper_from_config(config)`: Builds wrapper from dict, extracts params, handles blending.
- `create_prompt_blending_config`/`create_seed_blending_config`: Helpers for blending.
- `set_normalize_weights_config`: Sets normalization flags.
- Validation: Ensures model_id, controlnets/ipadapters lists, hook processors (type, enabled, params), blending lists.

Configs are loaded at startup; runtime updates via `update_stream_params` ([doc](../stream_parameter_updater.md)). Files: [`config.py`](../../../config.py).

## File Format (YAML Example)

```yaml
model_id: "runwayml/stable-diffusion-v1-5"
t_index_list: [0, 999]
width: 512
height: 512
mode: "img2img"
output_type: "pil"
device: "cuda"
dtype: "float16"
use_controlnet: true
controlnets:
  - model_id: "lllyasviel/sd-controlnet-canny"
    preprocessor: "canny"
    conditioning_scale: 1.0
    enabled: true
    preprocessor_params:
      threshold_low: 100
      threshold_high: 200
use_ipadapter: true
ipadapters:
  - ipadapter_model_path: "h94/IP-Adapter"
    image_encoder_path: "openai/clip-vit-large-patch14"
    scale: 0.8
    type: "regular"
prompt_blending:
  prompt