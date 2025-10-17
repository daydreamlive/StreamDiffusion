# Runtime Control Surface

## Overview

The runtime control surface provides real-time control over the StreamDiffusion pipeline during live streaming. This document focuses on the methods and parameters that users will actively use to control the application while it's running.

## Core Runtime Control Methods

### 1. update_stream_params()

The primary method for runtime parameter updates. All parameters are optional and only update what's specified.

```python
def update_stream_params(
    # Core generation parameters
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    delta: Optional[float] = None,
    t_index_list: Optional[List[int]] = None,
    seed: Optional[int] = None,
    
    # Prompt blending (real-time prompt control)
    prompt_list: Optional[List[Tuple[str, float]]] = None,
    negative_prompt: Optional[str] = None,
    prompt_interpolation_method: Literal["linear", "slerp"] = "slerp",
    normalize_prompt_weights: Optional[bool] = None,
    
    # Seed blending (real-time seed control)
    seed_list: Optional[List[Tuple[int, float]]] = None,
    seed_interpolation_method: Literal["linear", "slerp"] = "linear",
    normalize_seed_weights: Optional[bool] = None,
    
    # ControlNet configuration (real-time control image updates)
    controlnet_config: Optional[List[Dict[str, Any]]] = None,
    
    # IPAdapter configuration (real-time style updates)
    ipadapter_config: Optional[Dict[str, Any]] = None,
    
    # Pipeline hook configurations (real-time processing updates)
    image_preprocessing_config: Optional[List[Dict[str, Any]]] = None,
    image_postprocessing_config: Optional[List[Dict[str, Any]]] = None,
    latent_preprocessing_config: Optional[List[Dict[str, Any]]] = None,
    latent_postprocessing_config: Optional[List[Dict[str, Any]]] = None,
    
    # Safety checker
    use_safety_checker: Optional[bool] = None,
    safety_checker_threshold: Optional[float] = None,
) -> None:
```

**Real-time Usage Examples:**
```python
# Adjust generation strength
wrapper.update_stream_params(guidance_scale=8.0)

# Switch to different prompt blend
wrapper.update_stream_params(
    prompt_list=[("cat", 0.3), ("dog", 0.7)],
    negative_prompt="blurry, low quality"
)

# Change seed blend
wrapper.update_stream_params(
    seed_list=[(123, 0.2), (456, 0.8)]
)

# Update ControlNet configuration
wrapper.update_stream_params(
    controlnet_config=[{
        "model_id": "lllyasviel/sd-controlnet-canny",
        "preprocessor": "canny",
        "conditioning_scale": 1.2,
        "enabled": True
    }]
)
```

### 2. update_control_image()

Update control images for real-time ControlNet control.

```python
def update_control_image(
    index: int,                                    # ControlNet index (0, 1, 2, etc.)
    image: Union[str, Image.Image, torch.Tensor]  # New control image
) -> None:
```

**Real-time Usage:**
```python
# Update ControlNet 0 with new Canny edges
wrapper.update_control_image(0, "new_edges.jpg")

# Update ControlNet 1 with new depth map
wrapper.update_control_image(1, depth_image_tensor)

# Update from camera feed
wrapper.update_control_image(0, camera_frame)
```

### 3. update_style_image()

Update IPAdapter style reference for real-time style control.

```python
def update_style_image(
    image: Union[str, Image.Image, torch.Tensor]  # New style image
) -> None:
```

**Real-time Usage:**
```python
# Update style reference
wrapper.update_style_image("new_style.jpg")

# Update from live style feed
wrapper.update_style_image(style_camera_feed)
```

### 4. skip_diffusion

Property to enable/disable diffusion passthrough for real-time testing.

```python
# Enable passthrough mode (no diffusion, just preprocessing)
wrapper.skip_diffusion = True

# Re-enable normal diffusion
wrapper.skip_diffusion = False
```

**Use Cases:**
- Test preprocessing pipelines without diffusion overhead
- Debug control image processing
- Real-time control image updates without generation delay

## Live Stream Control Patterns

### Prompt Blending Control

Real-time prompt weight adjustment for smooth transitions:

```python
# Gradual transition from cat to dog
wrapper.update_stream_params(
    prompt_list=[("cat", 0.9), ("dog", 0.1)]  # Start mostly cat
)
# ... later ...
wrapper.update_stream_params(
    prompt_list=[("cat", 0.5), ("dog", 0.5)]  # Equal blend
)
# ... later ...
wrapper.update_stream_params(
    prompt_list=[("cat", 0.1), ("dog", 0.9)]  # Mostly dog
)
```

### Seed Blending Control

Real-time seed weight adjustment for noise variation:

```python
# Smooth noise transition
wrapper.update_stream_params(
    seed_list=[(123, 0.8), (456, 0.2)]  # Start with seed 123
)
# ... later ...
wrapper.update_stream_params(
    seed_list=[(123, 0.2), (456, 0.8)]  # Transition to seed 456
)
```

### ControlNet Real-time Updates

Update control images from live sources:

```python
# Camera-based control
def update_from_camera():
    frame = camera.get_frame()
    processed_frame = preprocess_canny(frame)
    wrapper.update_control_image(0, processed_frame)

# Multiple ControlNet updates
def update_multiple_controls():
    wrapper.update_control_image(0, canny_image)    # Canny control
    wrapper.update_control_image(1, depth_image)    # Depth control
    wrapper.update_control_image(2, pose_image)     # Pose control
```

### Style Reference Updates

Real-time style adaptation:

```python
# Update style from live feed
def update_style_from_feed():
    style_frame = style_camera.get_frame()
    wrapper.update_style_image(style_frame)

# Switch between style references
def switch_style(style_path):
    wrapper.update_style_image(style_path)
```

## Runtime State Management

### 5. get_stream_state()

Get current stream state for monitoring and debugging.

```python
def get_stream_state(
    include_caches: bool = False  # Include cache statistics
) -> Dict[str, Any]:
```

**Real-time Monitoring:**
```python
# Monitor current state
state = wrapper.get_stream_state(include_caches=True)
print(f"Current prompts: {state['prompt_list']}")
print(f"Guidance scale: {state['guidance_scale']}")
print(f"Active ControlNets: {len(state['controlnet_config'])}")

# Monitor memory usage
if state['caches']['prompt_cache_size'] > 1000:
    wrapper.clear_caches()
```

### 6. clear_caches()

Clear caches to free memory during long-running sessions.

```python
def clear_caches() -> None:
```

**Memory Management:**
```python
# Clear caches when switching to very different prompts
if prompt_change_detected:
    wrapper.clear_caches()
    wrapper.update_stream_params(prompt_list=new_prompts)
```

## Real-time Control Examples

### Interactive Prompt Control

```python
# Real-time prompt weight adjustment via UI sliders
def on_prompt_weight_change(prompt_index, new_weight):
    current_prompts = wrapper.get_stream_state()['prompt_list']
    current_prompts[prompt_index] = (current_prompts[prompt_index][0], new_weight)
    wrapper.update_stream_params(prompt_list=current_prompts)

# Real-time negative prompt updates
def on_negative_prompt_change(new_negative):
    wrapper.update_stream_params(negative_prompt=new_negative)
```

### Live Camera Control

```python
# Real-time camera control
def live_camera_control():
    while True:
        frame = camera.get_frame()
        
        # Process frame for ControlNet
        canny_frame = preprocess_canny(frame)
        wrapper.update_control_image(0, canny_frame)
        
        # Generate with current settings
        result = wrapper(frame)
        
        # Display result
        display_image(result)
```

### Dynamic Style Switching

```python
# Real-time style switching
def switch_style_dynamically(style_images):
    for style_image in style_images:
        wrapper.update_style_image(style_image)
        time.sleep(2.0)  # Hold style for 2 seconds
```

### Parameter Smoothing

```python
# Smooth parameter transitions
def smooth_guidance_transition(target_scale, steps=10):
    current_scale = wrapper.get_stream_state()['guidance_scale']
    step_size = (target_scale - current_scale) / steps
    
    for i in range(steps):
        new_scale = current_scale + (step_size * (i + 1))
        wrapper.update_stream_params(guidance_scale=new_scale)
        time.sleep(0.1)  # 100ms between updates
```

## Performance Considerations

### Update Frequency
- **Control Images**: Update as fast as source (30-60 FPS)
- **Prompts/Seeds**: Update less frequently (1-10 Hz)
- **Core Parameters**: Update sparingly (0.1-1 Hz)

### Memory Management
- Clear caches when switching between very different prompts
- Monitor memory usage with `get_stream_state(include_caches=True)`
- Use `skip_diffusion` for testing without memory overhead

### Error Handling
```python
# Safe control image updates
def safe_update_control_image(index, image):
    try:
        wrapper.update_control_image(index, image)
    except RuntimeError as e:
        print(f"ControlNet not enabled: {e}")
    except Exception as e:
        print(f"Failed to update control image: {e}")

# Safe parameter updates
def safe_update_params(**kwargs):
    try:
        wrapper.update_stream_params(**kwargs)
    except Exception as e:
        print(f"Failed to update parameters: {e}")
```

## Configuration Reference

### ControlNet Configuration

ControlNet configuration for real-time conditional guidance:

```python
controlnet_config = [
    {
        "model_id": "path/to/controlnet/model",         # Required: ControlNet model ID or path
        "preprocessor": "preprocessor_name",            # Optional: Preprocessor type from registry
        "conditioning_scale": 1.0,                      # Required: Influence strength (0.0-2.0)
        "enabled": True,                                # Optional: Enable/disable (default: True)
        "preprocessor_params": {                        # Optional: Preprocessor-specific parameters
            "param1": "value1",
            "param2": "value2"
        }
    }
]
```

**Configuration Fields:**
- `model_id`: HuggingFace model ID or local path to ControlNet model
- `preprocessor`: Preprocessor name from the processor registry (see [Processors](preprocessing/processors.md))
- `conditioning_scale`: Strength of ControlNet influence on generation
- `enabled`: Whether this ControlNet is active
- `preprocessor_params`: Parameters specific to the chosen preprocessor

### IPAdapter Configuration

IPAdapter configuration for style and reference adaptation:

```python
ipadapter_config = {
    "ipadapter_model_path": "path/to/ipadapter/model",          # Required: IPAdapter model path
    "image_encoder_path": "path/to/image/encoder",              # Required: Image encoder path
    "scale": 0.8,                                               # Required: Influence strength (0.0-1.0)
    "type": "regular",                                          # Optional: IPAdapter variant type
    "is_faceid": False,                                         # Optional: Face ID mode (default: False)
    "style_image": "path/to/style.jpg"                          # Optional: Default style reference
}
```

**Configuration Fields:**
- `ipadapter_model_path`: HuggingFace model ID or local path to IPAdapter model
- `image_encoder_path`: HuggingFace model ID or local path to CLIP image encoder
- `scale`: Strength of IPAdapter influence on generation
- `type`: IPAdapter variant (implementation-dependent)
- `is_faceid`: Enable Face ID mode for face-specific adaptation
- `style_image`: Default style reference image path

### Image Preprocessing Configuration

Configuration for image domain preprocessing hooks:

```python
image_preprocessing_config = [
    {
        "type": "processor_name",                           # Required: Processor type from registry
        "enabled": True,                                    # Optional: Enable/disable (default: True)
        "order": 0,                                         # Optional: Execution order (default: 0)
        "params": {                                         # Optional: Processor-specific parameters
            "param1": "value1",
            "param2": "value2"
        }
    }
]
```

**Configuration Fields:**
- `type`: Processor name from the processor registry (see [Processors](preprocessing/processors.md))
- `enabled`: Whether this processor is active
- `order`: Execution order in the processing chain (lower numbers execute first)
- `params`: Parameters specific to the chosen processor

### Image Postprocessing Configuration

Configuration for image domain postprocessing hooks:

```python
image_postprocessing_config = [
    {
        "type": "processor_name",                           # Required: Processor type from registry
        "enabled": True,                                    # Optional: Enable/disable (default: True)
        "order": 0,                                         # Optional: Execution order (default: 0)
        "params": {                                         # Optional: Processor-specific parameters
            "param1": "value1",
            "param2": "value2"
        }
    }
]
```

**Configuration Fields:**
- `type`: Processor name from the processor registry (see [Processors](preprocessing/processors.md))
- `enabled`: Whether this processor is active
- `order`: Execution order in the processing chain (lower numbers execute first)
- `params`: Parameters specific to the chosen processor

### Latent Preprocessing Configuration

Configuration for latent domain preprocessing hooks:

```python
latent_preprocessing_config = [
    {
        "type": "processor_name",                           # Required: Processor type from registry
        "enabled": True,                                    # Optional: Enable/disable (default: True)
        "order": 0,                                         # Optional: Execution order (default: 0)
        "params": {                                         # Optional: Processor-specific parameters
            "param1": "value1",
            "param2": "value2"
        }
    }
]
```

**Configuration Fields:**
- `type`: Processor name from the processor registry (see [Processors](preprocessing/processors.md))
- `enabled`: Whether this processor is active
- `order`: Execution order in the processing chain (lower numbers execute first)
- `params`: Parameters specific to the chosen processor

### Latent Postprocessing Configuration

Configuration for latent domain postprocessing hooks:

```python
latent_postprocessing_config = [
    {
        "type": "processor_name",                           # Required: Processor type from registry
        "enabled": True,                                    # Optional: Enable/disable (default: True)
        "order": 0,                                         # Optional: Execution order (default: 0)
        "params": {                                         # Optional: Processor-specific parameters
            "param1": "value1",
            "param2": "value2"
        }
    }
]
```

**Configuration Fields:**
- `type`: Processor name from the processor registry (see [Processors](preprocessing/processors.md))
- `enabled`: Whether this processor is active
- `order`: Execution order in the processing chain (lower numbers execute first)
- `params`: Parameters specific to the chosen processor

## Configuration Examples

### Basic Real-time Setup

```python
# Initialize with basic configs
wrapper = StreamDiffusionWrapper(
    model_id_or_path="runwayml/stable-diffusion-v1-5",
    t_index_list=[32, 45],
    use_controlnet=True,
    controlnet_config=[{
        "model_id": "lllyasviel/sd-controlnet-canny",
        "preprocessor": "canny",
        "conditioning_scale": 1.0,
        "enabled": True
    }]
)

# Runtime config updates
wrapper.update_stream_params(
    controlnet_config=[{
        "model_id": "lllyasviel/sd-controlnet-depth",
        "preprocessor": "depth",
        "conditioning_scale": 0.8,
        "enabled": True
    }]
)
```

### Advanced Multi-Module Setup

```python
# Complex runtime configuration
wrapper.update_stream_params(
    # ControlNet configuration
    controlnet_config=[
        {
            "model_id": "lllyasviel/sd-controlnet-canny",
            "preprocessor": "canny",
            "conditioning_scale": 1.0,
            "enabled": True,
            "preprocessor_params": {
                "threshold_low": 100,
                "threshold_high": 200
            }
        },
        {
            "model_id": "lllyasviel/sd-controlnet-pose",
            "preprocessor": "pose",
            "conditioning_scale": 0.7,
            "enabled": True
        }
    ],
    
    # IPAdapter configuration
    ipadapter_config={
        "ipadapter_model_path": "h94/IP-Adapter",
        "image_encoder_path": "openai/clip-vit-large-patch14",
        "scale": 0.8,
        "type": "plus",
        "is_faceid": False
    },
    
    # Image preprocessing
    image_preprocessing_config=[
        {
            "type": "resize",
            "enabled": True,
            "order": 0,
            "params": {"width": 512, "height": 512}
        },
        {
            "type": "normalize",
            "enabled": True,
            "order": 1,
            "params": {"mean": [0.5], "std": [0.5]}
        }
    ],
    
    # Latent preprocessing
    latent_preprocessing_config=[
        {
            "type": "latent_feedback",
            "enabled": True,
            "order": 0,
            "params": {"blend_factor": 0.1}
        }
    ]
)
```

## Best Practices

1. **Batch Updates**: Use `update_stream_params()` to update multiple parameters at once
2. **Error Handling**: Always wrap control calls in try-catch blocks
3. **Performance**: Use `skip_diffusion` for testing and debugging
4. **Memory**: Monitor and clear caches during long sessions
5. **Smooth Transitions**: Implement gradual parameter changes for smooth effects
6. **State Monitoring**: Use `get_stream_state()` for debugging and monitoring
7. **Config Validation**: Test configurations with `skip_diffusion=True` first
8. **Processor Ordering**: Use `order` parameter to control execution sequence
