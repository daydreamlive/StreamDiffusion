# Hooks Integration

```mermaid
graph LR
    A[Image Preprocessing Hooks] --> B[Latent Preprocessing Hooks]
    B --> C[UNet Hooks: e.g., ControlNet/IPAdapter]
    C --> D[Latent Postprocessing Hooks]
    D --> E[Image Postprocessing Hooks]
    
    F[Embedding Hooks: Custom Embedding Mods] -.->|Before UNet| C
    G[Config] -->|Register Hooks| A
    G -->|Register Hooks| B
    G -->|Register Hooks| C
    G -->|Register Hooks| D
    G -->|Register Hooks| E
    G -->|Register Hooks| F