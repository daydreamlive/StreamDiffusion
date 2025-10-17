# Preprocessing Orchestrators

## Overview

The Preprocessing Orchestrators manage the execution of preprocessors and postprocessors in StreamDiffusion, enabling efficient, parallelized, and pipelined processing for realtime streaming. They handle input preparation (e.g., edge detection, pose estimation), pipeline integration (e.g., latent modifications), and output enhancement (e.g., upscaling), with optimizations like caching, thread pooling, and CUDA streams to minimize latency.

Key orchestrators:
- **BaseOrchestrator**: Generic base for pipelined processing with sync fallback for feedback loops.
- **PreprocessingOrchestrator**: Handles module inputs (ControlNet/IPAdapter), parallelizes across multiple preprocessors, caches for identical frames.
- **PipelinePreprocessingOrchestrator**: Processes tensors in pipeline hooks (pre/post UNet/VAE), sequential for dependencies.
- **PostprocessingOrchestrator**: Applies enhancements to generated images, with input caching for repeated outputs.
- **OrchestratorUser**: Mixin for modules to attach shared orchestrators.

Orchestrators are lazily created and shared across modules via `StreamDiffusion` instances. Core files: [`base_orchestrator.py`](../../../preprocessing/base_orchestrator.py), [`preprocessing_orchestrator.py`](../../../preprocessing/preprocessing_orchestrator.py), [`pipeline_preprocessing_orchestrator.py`](../../../preprocessing/pipeline_preprocessing_orchestrator.py), [`postprocessing_orchestrator.py`](../../../preprocessing/postprocessing_orchestrator.py), [`orchestrator_user.py`](../../../preprocessing/orchestrator_user.py).

## BaseOrchestrator

Generic foundation for all orchestrators:

- **Pipelining**: Background thread processing for next frame while