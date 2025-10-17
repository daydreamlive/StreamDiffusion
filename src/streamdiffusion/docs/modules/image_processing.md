# Image Processing Modules

## Overview

The image processing modules provide a flexible framework for processing images at different stages of the StreamDiffusion pipeline. These modules operate in the image domain (pixel space) and support both preprocessing (before VAE encoding) and postprocessing (after VAE decoding) operations.

## Architecture

The image processing system consists of three main classes:

- **`ImageProcessingModule`**: Base class providing shared functionality
- **`ImagePreprocessingModule`**: Handles image processing before VAE encoding
- **`ImagePostprocessingModule`**: Handles image processing after VAE decoding

## Base Class: ImageProcessingModule

The `ImageProcessingModule` serves as the foundation for all image domain processing modules.

### Key Features

- **Sequential Chain Execution**: Processes images through a chain of processors in order
- **Processor Management**: Add, configure, and order processors dynamically
- **Orchestrator Integration**: Uses preprocessing orchestrators for efficient processing
- **Parameter Alignment**: Automatically aligns processor parameters with stream dimensions

### Core Methods

```python
def add_processor(self, proc_config: Dict[str, Any]) -> None:
    """Add a processor using the existing registry."""
    
def _process_image_chain(self, input_image: torch.Tensor) -> torch.Tensor:
    """Execute sequential chain of processors in image domain."""
    
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

## ImagePreprocessingModule

Processes images before VAE encoding in the pipeline.

### Timing

- **Execution Point**: After `image_processor.preprocess()`, before `similar_image_filter`
- **Pipeline Stage**: Input preprocessing stage
- **Performance**: Uses pipelined processing for optimization

### Key Features

- **Pipelined Processing**: Frame N-1 results while starting Frame N processing
- **Performance Optimization**: Uses `PipelinePreprocessingOrchestrator`
- **Fallback Support**: Falls back to synchronous processing when needed

### Installation

```python
def install(self, stream) -> None:
    """Install module by registering hook with stream and attaching orchestrators."""
    self._stream = stream
    self.attach_orchestrator(stream)  # Sequential chain processing (fallback)
    self.attach_pipeline_preprocessing_orchestrator(stream)  # Pipelined processing
    stream.image_preprocessing_hooks.append(self.build_image_hook())
```

### Usage Example

```python
# Create preprocessing module
image_preproc = ImagePreprocessingModule()

# Add processors
image_preproc.add_processor({
    'type': 'resize',
    'params': {'width': 512, 'height': 512}
})

image_preproc.add_processor({
    'type': 'normalize',
    'params': {'mean': [0.5], 'std': [0.5]}
})

# Install in stream
image_preproc.install(stream)
```

## ImagePostprocessingModule

Processes images after VAE decoding in the pipeline.

### Timing

- **Execution Point**: After `decode_image()`, before returning final output
- **Pipeline Stage**: Output postprocessing stage
- **Performance**: Uses pipelined processing for optimization

### Key Features

- **Pipelined Processing**: Frame N-1 results while starting Frame N processing
- **Performance Optimization**: Uses `PostprocessingOrchestrator`
- **Fallback Support**: Falls back to synchronous processing when needed

### Installation

```python
def install(self, stream) -> None:
    """Install module by registering hook with stream and attaching orchestrators."""
    self._stream = stream
    self.attach_preprocessing_orchestrator(stream)  # Sequential chain processing (fallback)
    self.attach_postprocessing_orchestrator(stream)  # Pipelined processing
    stream.image_postprocessing_hooks.append(self.build_image_hook())
```

### Usage Example

```python
# Create postprocessing module
image_postproc = ImagePostprocessingModule()

# Add processors
image_postproc.add_processor({
    'type': 'upscale',
    'params': {'scale_factor': 2}
})

image_postproc.add_processor({
    'type': 'sharpen',
    'params': {'strength': 0.5}
})

# Install in stream
image_postproc.install(stream)
```

## Integration with Pipeline

The image processing modules integrate with the StreamDiffusion pipeline through hooks:

### Pipeline Integration Points

1. **Image Preprocessing Hooks**: Applied after built-in preprocessing, before filtering
2. **Image Postprocessing Hooks**: Applied after VAE decoding, before final output

### Hook Execution Flow

```python
# In StreamDiffusion.__call__()
x = self.image_processor.preprocess(x, self.height, self.width)
x = self._apply_image_preprocessing_hooks(x)  # ImagePreprocessingModule
# ... VAE encoding and diffusion ...
x_output = self.decode_image(x_0_pred_out)
x_output = self._apply_image_postprocessing_hooks(x_output)  # ImagePostprocessingModule
```

## Performance Considerations

### Pipelined Processing

Both preprocessing and postprocessing modules use pipelined processing for performance:

- **Frame Overlap**: Process Frame N-1 results while starting Frame N
- **Orchestrator Integration**: Uses specialized orchestrators for efficiency
- **Fallback Support**: Graceful degradation to synchronous processing

### Memory Management

- **Stream Reference**: Modules store stream reference for dimension access
- **Parameter Alignment**: Automatic alignment with stream resolution
- **Processor Lifecycle**: Processors are managed through the orchestrator system

## Common Use Cases

### Image Preprocessing

- **Resizing**: Adjust image dimensions to match model requirements
- **Normalization**: Apply mean/std normalization
- **Color Space Conversion**: Convert between different color spaces
- **Format Conversion**: Convert between different image formats

### Image Postprocessing

- **Upscaling**: Increase image resolution
- **Enhancement**: Apply sharpening, denoising, or other enhancements
- **Format Conversion**: Convert output to desired format
- **Quality Adjustment**: Apply final quality improvements

## Error Handling

The modules include robust error handling:

- **Parameter Validation**: Validates processor configuration
- **Graceful Degradation**: Falls back to synchronous processing on errors
- **Exception Handling**: Catches and handles processor-specific errors
- **Resource Management**: Proper cleanup of resources

## Best Practices

1. **Processor Ordering**: Use the `order` parameter to control execution sequence
2. **Parameter Alignment**: Let the module handle dimension alignment automatically
3. **Performance Testing**: Test with pipelined processing for optimal performance
4. **Error Handling**: Implement proper error handling for production use
5. **Resource Management**: Monitor memory usage with complex processor chains
