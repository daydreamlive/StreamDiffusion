# Module Integration

```mermaid
graph TD
    A[Input Image] --> B[Image Preprocessing Hooks]
    B --> C[VAE Encode]
    C --> D[Latent Preprocessing Hooks]
    D --> E[UNet Forward]
    
    E --> F{ControlNet Active?}
    F -->|Yes| G[Add Residuals: Down/Mid Blocks]
    F -->|No| H{IPAdapter Active?}
    H -->|Yes| I[Set IPAdapter Scale Vector]
    H -->|No| J[Standard UNet Call]
    G --> J
    I --> J
    
    J --> K[Latent Postprocessing Hooks]
    K --> L[VAE Decode]
    L --> M[Image Postprocessing Hooks]
    M --> N[Output Image]
    
    O[StreamParameterUpdater] -.->|Update Scales| I
    P[Config] -->|Enable Modules| F
    P -->|Enable Modules| H
    P -->|Enable Modules| B
    P -->|Enable Modules| D
    P -->|Enable Modules| K
    P -->|Enable Modules| M