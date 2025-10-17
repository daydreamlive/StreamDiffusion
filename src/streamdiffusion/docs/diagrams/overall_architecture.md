# Overall Architecture

```mermaid
graph TB
    subgraph "Input"
        A["Input: Image/Prompt/Control Image"]
    end
    
    subgraph "Preprocessing"
        B["Preprocessing Orchestrators"]
        C["Processors: Edge Detection (Canny/HED), Pose (OpenPose), Depth (MiDaS)"]
        D["Parallel Execution via ThreadPool"]
    end
    
    subgraph "Pipeline Core"
        E["StreamDiffusion.prepare: Embeddings/Timesteps/Noise"]
        F["UNet Steps with Hooks"]
        G["ControlNet/IPAdapter Injection"]
        H["Orchestrator Calls: Latent/Image Hooks"]
    end
    
    subgraph "Decoding"
        I["VAE Decode"]
        J["Postprocessing Orchestrators"]
    end
    
    subgraph "Output"
        K["Output: Image"]
    end
    
    subgraph "Runtime Control"
        L1["StreamDiffusionWrapper"]
        L2["update_stream_params()"]
        L3["update_control_image()"]
        L4["update_style_image()"]
    end
    
    subgraph "Management"
        L["StreamParameterUpdater: Blending/Caching"]
        M["Config Loader: YAML/JSON"]
    end
    
    subgraph "Acceleration"
        N["TensorRT Engines: UNet/VAE/ControlNet"]
        O["Runtime Inference"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    
    L1 --> L2
    L1 --> L3
    L1 --> L4
    L2 -.->|"Runtime Updates"| L
    L3 -.->|"via Orchestrators"| B
    L4 -.->|"via Orchestrators"| B
    
    L -.->|"Updates"| E
    L -.->|"Updates"| F
    M -.->|"Setup"| B
    M -.->|"Setup"| J
    M -.->|"Setup"| L
    N -.->|"Optimized"| F
    N -.->|"Optimized"| I
    O -.->|"Fallback PyTorch"| F
    O -.->|"Fallback PyTorch"| I