# Latent Processing Modules

## Overview

The latent processing modules provide a flexible framework for processing latent representations at different stages of the StreamDiffusion pipeline. These modules operate in the latent domain (compressed representation space) and support both preprocessing (after VAE encoding, before diffusion) and postprocessing (after diffusion, before VAE decoding) operations.

## Architecture

The latent processing system consists of three main classes:

- **`LatentProcessingModule`**: Base class providing shared functionality
- **`LatentPreprocessingModule`**: Handles latent processing after VAE encoding, before diffusion
- **`LatentPostprocessingModule`**: Handles latent processing after diffusion, before VAE decoding

## Base Class: LatentProcessingModule

The `LatentProcessingModule` serves as the foundation for all latent domain processing modules.

### Key Features

- **Sequential Chain Execution**: Processes latents through a chain of processors in order
- **Processor Management**: Add, configure, and order processors dynamically
- **Orchestrator Integration**: Uses preprocessing orchestrators for efficient processing
- **Pipeline Reference**: Automatically handles pipeline reference through factory functions

### Core Methods

```python
def add_processor(self, proc_config: Dict[str, Any]) -> None:
    """Add a processor using the existing registry."""
    
def _process_latent_chain(self, input_latent: torch.Tensor) -> torch.Tensor:
    """Execute sequential chain of processors in latent domain."""
    
def _get_ordered_processors(self) -> List[Any]:
    """Return enabled processors in execution order."""
```

### Processor Configuration

Processors are added using configuration dictionaries:

```python
proc_config = {
    'type': 'processor_name',      # Required: processor type from registry
    'enabled': True,              # Optional: enable/disable processor
    'order': 0,                   # Optional: execution order
    'params': {                   # Optional: processor-specific parameters
        'param1': 'value1',
        'param2': 'value2'
    }
}
```

## LatentPreprocessingModule

Processes latent representations after VAE encoding, before diffusion.

### Timing

- **Execution Point**: After `encode_image()`, before `predict_x0_batch()`
- **Pipeline Stage**: Pre-diffusion processing stage
- **Domain**: Latent space (compressed representation)

### Key Features

- **Sequential Processing**: Uses orchestrator for sequential chain execution
- **Pipeline Integration**: Automatically handles pipeline reference
- **Hook Registration**: Registers with `latent_preprocessing_hooks`

### Installation

```python
def install(self, stream) -> None:
    """Install module by registering hook with stream and attaching orchestrator."""
    self.attach_orchestrator(stream)
    self._stream = stream  # Store stream reference
    stream.latent_preprocessing_hooks.append(self.build_latent_hook())
```

### Usage Example

```python
# Create preprocessing module
latent_preproc = LatentPreprocessingModule()

# Add processors
latent_preproc.add_processor({
    'type': 'latent_noise',
    'params': {'strength': 0.1}
})

latent_preproc.add_processor({
    'type': 'latent_scale',
    'params': {'scale_factor': 1.1}
})

# Install in stream
latent_preproc.install(stream)
```

## LatentPostprocessingModule

Processes latent representations after diffusion, before VAE decoding.

### Timing

- **Execution Point**: After `predict_x0_batch()`, before `decode_image()`
- **Pipeline Stage**: Post-diffusion processing stage
- **Domain**: Latent space (compressed representation)

### Key Features

- **Sequential Processing**: Uses orchestrator for sequential chain execution
- **Pipeline Integration**: Automatically handles pipeline reference
- **Hook Registration**: Registers with `latent_postprocessing_hooks`

### Installation

```python
def install(self, stream) -> None:
    """Install module by registering hook with stream and attaching orchestrator."""
    self.attach_orchestrator(stream)
    self._stream = stream  # Store stream reference
    stream.latent_postprocessing_hooks.append(self.build_latent_hook())
```

### Usage Example

```python
# Create postprocessing module
latent_postproc = LatentPostprocessingModule()

# Add processors
latent_postproc.add_processor({
    'type': 'latent_denoise',
    'params': {'strength': 0.05}
})

latent_postproc.add_processor({
    'type': 'latent_enhance',
    'params': {'enhancement_factor': 1.2}
})

# Install in stream
latent_postproc.install(stream)
```

## Integration with Pipeline

The latent processing modules integrate with the StreamDiffusion pipeline through hooks:

### Pipeline Integration Points

1. **Latent Preprocessing Hooks**: Applied after VAE encoding, before diffusion
2. **Latent Postprocessing Hooks**: Applied after diffusion, before VAE decoding

### Hook Execution Flow

```python
# In StreamDiffusion.__call__()
x_t_latent = self.encode_image(x)
x_t_latent = self._apply_latent_preprocessing_hooks(x_t_latent)  # LatentPreprocessingModule
x_0_pred_out = self.predict_x0_batch(x_t_latent)
x_0_pred_out = self._apply_latent_postprocessing_hooks(x_0_pred_out)  # LatentPostprocessingModule
x_output = self.decode_image(x_0_pred_out)
```

## Latent Domain Characteristics

### Latent Space Properties

- **Compressed Representation**: 4-channel latent tensors (typically 64x64 for 512x512 images)
- **Normalized Values**: Latent values are typically normalized to [-1, 1] range
- **Spatial Structure**: Maintains spatial relationships in compressed form
- **Model-Specific**: Latent dimensions depend on the VAE and model architecture

### Processing Considerations

- **Memory Efficiency**: Latent processing is more memory-efficient than image processing
- **Quality Impact**: Changes in latent space directly affect final image quality
- **Model Compatibility**: Processors must be compatible with the specific VAE model
- **Numerical Stability**: Careful handling of latent values to maintain stability

## Common Use Cases

### Latent Preprocessing

- **Noise Addition**: Add controlled noise for variation
- **Latent Scaling**: Adjust latent magnitude for different effects
- **Conditional Processing**: Apply conditional transformations based on prompts
- **Style Transfer**: Modify latent representations for style effects

### Latent Postprocessing

- **Denoising**: Remove artifacts or unwanted noise
- **Enhancement**: Improve latent quality before decoding
- **Correction**: Fix issues introduced during diffusion
- **Fine-tuning**: Apply final adjustments to latent representations

## Performance Considerations

### Sequential Processing

Latent processing modules use sequential processing:

- **Orchestrator Integration**: Uses preprocessing orchestrators for efficiency
- **Chain Execution**: Processes latents through ordered processor chains
- **Pipeline Reference**: Automatic handling of pipeline context

### Memory Management

- **Stream Reference**: Modules store stream reference for context
- **Processor Lifecycle**: Processors are managed through the orchestrator system
- **Latent Caching**: Previous latent results are cached for feedback processors

## Error Handling

The modules include robust error handling:

- **Parameter Validation**: Validates processor configuration
- **Pipeline Reference**: Automatic handling of pipeline context
- **Exception Handling**: Catches and handles processor-specific errors
- **Resource Management**: Proper cleanup of resources

## Best Practices

1. **Processor Ordering**: Use the `order` parameter to control execution sequence
2. **Latent Stability**: Ensure processors maintain numerical stability
3. **Quality Testing**: Test processors thoroughly to avoid quality degradation
4. **Memory Monitoring**: Monitor memory usage with complex processor chains
5. **Model Compatibility**: Ensure processors work with your specific VAE model

## Advanced Usage

### Custom Latent Processors

Create custom processors for specific latent operations:

```python
class CustomLatentProcessor:
    def __init__(self, custom_param):
        self.custom_param = custom_param
    
    def __call__(self, latent):
        # Custom latent processing logic
        return processed_latent
```

### Feedback Processing

Use previous latent results for feedback-based processing:

```python
# Access previous latent result in processor
def process_with_feedback(latent, prev_latent):
    # Use previous latent for feedback processing
    return processed_latent
```

## Integration with Other Modules

### ControlNet Integration

Latent processing can work alongside ControlNet modules:

```python
# Latent preprocessing before ControlNet influence
latent_preproc.add_processor({'type': 'latent_enhance'})
controlnet_module.install(stream)
```

### IPAdapter Integration

Latent processing can enhance IPAdapter effects:

```python
# Latent postprocessing after IPAdapter
ipadapter_module.install(stream)
latent_postproc.add_processor({'type': 'latent_refine'})
```
