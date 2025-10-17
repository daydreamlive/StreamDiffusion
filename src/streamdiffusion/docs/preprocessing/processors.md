# Processor System

## Overview

The processor system provides a registry-based architecture for modular image and latent processing components. Processors are executed via [Preprocessing Orchestrators](orchestrators.md) for parallel/pipelined efficiency and handle input data preparation for modules like ControlNet and IPAdapter, as well as pipeline/latent stage enhancements.

**Key Features:**
- **Registry-based**: Dynamic processor discovery and instantiation
- **Template Pattern**: Consistent interface with automatic input validation and size handling
- **GPU Acceleration**: Tensor processing with fallback to PIL
- **Pipeline Awareness**: Support for processors that need access to previous pipeline state
- **TensorRT Support**: High-performance variants for production deployment

## Architecture

### Base Classes

#### BasePreprocessor

Abstract base class implementing the template method pattern:

```python
from streamdiffusion.preprocessing.processors import BasePreprocessor

class MyProcessor(BasePreprocessor):
    def _process_core(self, image: Image.Image) -> Image.Image:
        # Implement your processing logic here
        return processed_image
```

**Key Methods:**
- `process()`: Main entry point for PIL image processing
- `process_tensor()`: GPU tensor processing
- `get_preprocessor_metadata()`: Class method returning processor metadata

#### PipelineAwareProcessor

For processors that need access to pipeline state (previous outputs):

```python
from streamdiffusion.preprocessing.processors import PipelineAwareProcessor

class FeedbackProcessor(PipelineAwareProcessor):
    def _process_core(self, image: Image.Image) -> Image.Image:
        # Access previous pipeline output via self.pipeline_ref
        prev_output = self.pipeline_ref.prev_image_result
        return blend_with_previous(image, prev_output)
```

**Features:**
- Automatic synchronous processing to avoid temporal artifacts
- Pipeline reference injection for accessing previous outputs
- Required pipeline_ref parameter validation

## Registry System

### Core Registry Functions

```python
from streamdiffusion.preprocessing.processors import (
    get_preprocessor,
    get_preprocessor_class, 
    list_preprocessors,
    register_preprocessor
)

# List available processors
available = list_preprocessors()
print(f"Available processors: {available}")

# Get processor class
ProcessorClass = get_preprocessor_class("canny")

# Get processor instance
processor = get_preprocessor("canny")

# Get pipeline-aware processor
feedback_processor = get_preprocessor("latent_feedback", pipeline_ref=stream)

# Register custom processor
register_preprocessor("my_custom", MyCustomProcessor)
```

### Processor Discovery

The registry automatically handles:
- **Conditional imports**: TensorRT and MediaPipe processors based on availability
- **Dynamic registration**: Available processors adapt to system capabilities
- **Pipeline awareness**: Automatic detection of processors requiring pipeline reference

### Metadata System

Get comprehensive processor information:

```python
# Get processor metadata
metadata = ProcessorClass.get_preprocessor_metadata()
print(f"Display name: {metadata['display_name']}")
print(f"Description: {metadata['description']}")
print(f"Parameters: {metadata['parameters']}")
print(f"Use cases: {metadata['use_cases']}")
```

**Metadata Structure:**
```python
{
    "display_name": "Human-readable name",
    "description": "Detailed description of functionality", 
    "parameters": {
        "param_name": {
            "type": "parameter_type",
            "default": "default_value",
            "description": "Parameter description",
            "range": [min_val, max_val]  # Optional
        }
    },
    "use_cases": ["list", "of", "common", "applications"]
}
```

## Processor Categories

### Standard Processors

Inherit from `BasePreprocessor` for stateless processing:
- Edge detection (Canny, HED, Lineart)
- Computer vision (Pose, Depth, Segmentation) 
- Image enhancement (Blur, Sharpen, Upscale)
- Utilities (Passthrough, External)

### Pipeline-Aware Processors  

Inherit from `PipelineAwareProcessor` for temporal processing:
- `feedback`: Frame-to-frame temporal blending
- `latent_feedback`: Latent space temporal consistency
- Custom processors requiring previous pipeline state

### TensorRT Accelerated

High-performance variants for production:
- `depth_tensorrt`: Accelerated depth estimation
- `pose_tensorrt`: Accelerated pose detection
- `realesrgan_trt`: Tiled super-resolution
- `temporal_net_tensorrt`: Temporal flow processing

## Developer API

### Creating Custom Processors

```python
from streamdiffusion.preprocessing.processors import BasePreprocessor

class CustomProcessor(BasePreprocessor):
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Custom Processor",
            "description": "My custom image processor",
            "parameters": {
                "strength": {
                    "type": "float",
                    "default": 1.0,
                    "description": "Processing strength",
                    "range": [0.0, 2.0]
                }
            },
            "use_cases": ["Custom processing", "Special effects"]
        }
    
    def _process_core(self, image: Image.Image) -> Image.Image:
        strength = self.params.get('strength', 1.0)
        # Implement custom processing
        return processed_image

# Register the processor
register_preprocessor("custom", CustomProcessor)
```

### Using Processors Programmatically

```python
# Instantiate processor with parameters
processor = get_preprocessor("blur", kernel_size=5, sigma=2.0)

# Process image
result = processor.process(input_image)

# Process tensor directly on GPU
result_tensor = processor.process_tensor(input_tensor)

# Check processor capabilities
metadata = processor.get_preprocessor_metadata()
available_params = metadata["parameters"]
```

### Pipeline Integration

```python
# Configure processor in runtime config
image_preprocessing_config = [
    {
        "type": "custom",
        "enabled": True,
        "order": 0,
        "params": {
            "strength": 1.5
        }
    }
]

# Update runtime configuration
wrapper.update_stream_params(
    image_preprocessing_config=image_preprocessing_config
)
```

## Best Practices

### Development
1. **Inherit from appropriate base class**: Use `PipelineAwareProcessor` only when you need previous pipeline state
2. **Implement metadata**: Provide comprehensive metadata for UI integration
3. **Handle parameters**: Support configuration via `self.params`
4. **GPU optimization**: Override `_process_tensor_core()` for better performance

### Deployment  
1. **Use TensorRT variants**: For production/latency-critical applications
2. **Chain via orchestrators**: For parallel processing of multiple processors
3. **Monitor dependencies**: Check processor availability before use
4. **Parameter tuning**: Use metadata to understand parameter ranges and defaults

### Integration
1. **Registry pattern**: Always use registry functions rather than direct imports
2. **Error handling**: Check processor availability with `list_processors()`
3. **Configuration**: Use structured configs for reproducible setups
4. **Testing**: Test with `skip_diffusion=True` for fast iteration

For orchestration and chaining patterns, see [Orchestrators](orchestrators.md).