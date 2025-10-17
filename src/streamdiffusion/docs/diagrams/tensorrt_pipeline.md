# TensorRT Pipeline

```mermaid
graph TD
    A[PyTorch Model] --> B[ONNX Export: UnifiedWrapper]
    B --> C[Optimize ONNX: Graph Surgeon]
    C --> D[Build TRT Engine: Dynamic Shapes]
    D --> E[Runtime Engine: Infer with Buffers]
    E --> F[Shape Cache: Reuse Buffers]
    F --> G[Output: Optimized Pred]
    
    H[EngineManager] -->|Compile/Load| D
    I[ControlNet/IPAdapter] -.->|Wrappers| B
    J[Config] -->|Params| H
    K[Runtime] -->|Fallback PyTorch| E