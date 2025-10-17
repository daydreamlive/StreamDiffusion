# Parameter Updating

```mermaid
graph TD
    subgraph "Runtime Update Entry Point"
        A["update_stream_params Call"]
        A --> B["Thread Lock: _update_lock"]
    end
    
    subgraph "Parallel Parameter Processing"
        B --> C1["Core Params: steps/guidance/delta/seed"]
        B --> C2["Prompt List Processing"]
        B --> C3["Seed List Processing"]
        B --> C4["ControlNet Config Processing"]
        B --> C5["IPAdapter Config Processing"]
        B --> C6["Hook Config Processing"]
        B --> C7["Timestep/Resolution Updates"]
        
        C1 --> D1["Update scheduler/guidance/delta/base_seed"]
        C2 --> D2["_update_blended_prompts: Cache/Encode/Blend"]
        C3 --> D3["_update_blended_seeds: Cache/Generate/Blend"]
        C4 --> D4["_update_controlnet_config: Diff/Add/Remove/Scale"]
        C5 --> D5["_update_ipadapter_config: Scale/Weight Type"]
        C6 --> D6["_update_hook_config: Processors/Params/Order"]
        C7 --> D7["_recalculate_timestep_dependent_params"]
    end
    
    subgraph "Blending & Caching Operations"
        D2 --> E1["_cache_prompt_embeddings"]
        D2 --> E2["_apply_prompt_blending: Linear/SLERP"]
        D3 --> E3["_cache_seed_noise"]
        D3 --> E4["_apply_seed_blending: Linear/SLERP"]
        D4 --> E5["Cache Stats: ControlNet Operations"]
        D5 --> E6["Cache Stats: IPAdapter Operations"]
        D6 --> E7["Cache Stats: Hook Operations"]
        
        E1 --> F["Update Pipeline State"]
        E2 --> F
        E3 --> F
        E4 --> F
        E5 --> F
        E6 --> F
        E7 --> F
        D1 --> F
        D7 --> F
    end
    
    subgraph "Pipeline Integration"
        F --> G["Pipeline Uses Updated Tensors/Hooks/Configs"]
    end
    
    subgraph "Shared Utilities"
        H1["Normalize Weights: Sum to 1.0 (Optional)"]
        H2["Thread-Safe Lock: Prevent Race Conditions"]
        H3["Cache Reindexing: Handle Add/Remove"]
    end
    
    B -.->|"Protect All Operations"| H2
    D2 -.->|"Use"| H1
    D3 -.->|"Use"| H1
    E1 -.->|"Use"| H3
    E3 -.->|"Use"| H3
    D4 -.->|"Use"| H3
    D5 -.->|"Use"| H3
    D6 -.->|"Use"| H3