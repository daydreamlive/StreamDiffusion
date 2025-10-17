# StreamDiffusion Documentation

## Getting Started

- [Installation Guide](installation.md): Complete setup instructions for StreamDiffusion with TensorRT, ControlNet, and IPAdapter.

## Core Concepts

- [Hook-Module System](hooks.md): Extensible pipeline hooks for modules.
- [Multi-Stage Processing](pipeline.md): Pipeline stages and integration.
- [StreamParameterUpdater](stream_parameter_updater.md): Runtime parameter blending/caching.
- [Runtime Control Surface](runtime_control.md): Real-time control methods for live streaming.

## Modules

- [ControlNet Module](modules/controlnet.md): Conditional guidance.
- [IPAdapter Module](modules/ipadapter.md): Style/reference adaptation.
- [Image Processing Modules](modules/image_processing.md): Image domain preprocessing and postprocessing.
- [Latent Processing Modules](modules/latent_processing.md): Latent domain preprocessing and postprocessing.

## Preprocessing

- [Orchestrators](preprocessing/orchestrators.md): Parallel/pipelined execution.
- [Processors](preprocessing/processors.md): Edge/pose/depth utilities.

## Configuration

- [Config Management](config.md): YAML/JSON loading and validation.

## Acceleration

- [TensorRT](acceleration/tensorrt.md): Engine building and runtime.

## Diagrams

- [Overall Architecture](diagrams/overall_architecture.md)
- [SDXL vs SD1.5 Comparison](diagrams/sdxl_vs_sd15.md)
- [Hooks Integration](diagrams/hooks_integration.md)
- [Orchestrator Flow](diagrams/orchestrator_flow.md)
- [Module Integration](diagrams/module_integration.md)
- [Parameter Updating](diagrams/parameter_updating.md)
- [TensorRT Pipeline](diagrams/tensorrt_pipeline.md)