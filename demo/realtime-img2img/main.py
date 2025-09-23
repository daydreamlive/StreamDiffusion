from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, UploadFile, File, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2

import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import os
import time
import mimetypes
import torch
import tempfile
from pathlib import Path
import yaml

from config import config, Args
from util import pil_to_frame, pt_to_frame, bytes_to_pil, bytes_to_pt
from connection_manager import ConnectionManager, ServerFullException
from img2img import Pipeline
from input_control import InputManager, GamepadInput

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120

# Import configuration from separate file to avoid circular imports
from app_config import AVAILABLE_CONTROLNETS, DEFAULT_SETTINGS

# Configure logging
def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration for the application"""
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up logger for streamdiffusion modules
    streamdiffusion_logger = logging.getLogger('streamdiffusion')
    streamdiffusion_logger.setLevel(numeric_level)
    
    # Set up logger for this application
    app_logger = logging.getLogger('realtime_img2img')
    app_logger.setLevel(numeric_level)
    
    return app_logger

# Initialize logger
logger = setup_logging(config.log_level)

# Suppress uvicorn INFO messages
if config.quiet:
    uvicorn_logger = logging.getLogger('uvicorn')
    uvicorn_logger.setLevel(logging.WARNING)
    uvicorn_access_logger = logging.getLogger('uvicorn.access')
    uvicorn_access_logger.setLevel(logging.WARNING)


class AppState:
    """Centralized application state management - SINGLE SOURCE OF TRUTH"""
    
    def __init__(self):
        # Pipeline state
        self.pipeline_lifecycle = "stopped"  # stopped, starting, running, error
        self.pipeline_active = False
        
        # Configuration state  
        self.uploaded_config = None  # Raw uploaded config
        self.runtime_config = None   # Runtime modifications to config
        self.config_needs_reload = False
        
        # Resolution state
        self.current_resolution = {"width": 512, "height": 512}
        
        # Parameter state (consolidates scattered vars from frontend)
        self.pipeline_params = {}
        
        # ControlNet state - AUTHORITATIVE SOURCE
        self.controlnet_info = {
            "enabled": False,
            "controlnets": []
        }
        
        # IPAdapter state - AUTHORITATIVE SOURCE  
        self.ipadapter_info = {
            "enabled": False,
            "has_style_image": False,
            "scale": 1.0,
            "weight_type": "linear"
        }
        
        # Pipeline hooks state - AUTHORITATIVE SOURCE
        self.pipeline_hooks = {
            "image_preprocessing": {"enabled": False, "processors": []},
            "image_postprocessing": {"enabled": False, "processors": []},
            "latent_preprocessing": {"enabled": False, "processors": []},
            "latent_postprocessing": {"enabled": False, "processors": []}
        }
        
        # Blending configurations
        self.prompt_blending = None
        self.seed_blending = None
        self.normalize_prompt_weights = True
        self.normalize_seed_weights = True
        
        # Core pipeline parameters
        self.guidance_scale = 1.1
        self.delta = 0.7
        self.num_inference_steps = 50
        self.seed = 2
        self.t_index_list = [35, 45]
        self.negative_prompt = ""
        self.skip_diffusion = False
        
        # UI state
        self.fps = 0
        self.queue_size = 0
        self.model_id = ""
        self.page_content = ""
        
        # Input source state
        self.input_sources = {
            'controlnet': {},  # {index: source_info}
            'ipadapter': None,
            'base': None
        }
        
        # Debug mode state
        self.debug_mode = False
        self.debug_pending_frame = False  # True when a frame step is requested
        
    def populate_from_config(self, config_data):
        """Populate AppState from uploaded config - SINGLE SOURCE OF TRUTH"""
        if not config_data:
            return
            
        logger.info("populate_from_config: Populating AppState from config as single source of truth")
        
        # Store the complete uploaded config to preserve ALL parameters
        self.uploaded_config = config_data
        
        # Core parameters
        self.guidance_scale = config_data.get('guidance_scale', self.guidance_scale)
        self.delta = config_data.get('delta', self.delta)
        self.num_inference_steps = config_data.get('num_inference_steps', self.num_inference_steps)
        self.seed = config_data.get('seed', self.seed)
        self.t_index_list = config_data.get('t_index_list', self.t_index_list)
        self.negative_prompt = config_data.get('negative_prompt', self.negative_prompt)
        self.skip_diffusion = config_data.get('skip_diffusion', self.skip_diffusion)
        self.model_id = config_data.get('model_id_or_path', self.model_id)
        
        # Resolution parameters
        if 'width' in config_data or 'height' in config_data:
            self.current_resolution = {
                "width": config_data.get('width', self.current_resolution["width"]),
                "height": config_data.get('height', self.current_resolution["height"])
            }
        
        # Normalization settings
        self.normalize_prompt_weights = config_data.get('normalize_weights', self.normalize_prompt_weights)
        self.normalize_seed_weights = config_data.get('normalize_weights', self.normalize_seed_weights)
        
        # ControlNet configuration
        if 'controlnets' in config_data:
            self.controlnet_info = {
                "enabled": True,
                "controlnets": []
            }
            for i, controlnet in enumerate(config_data['controlnets']):
                processed = dict(controlnet)
                processed['index'] = i
                processed['name'] = controlnet.get('model_id', '')
                processed['strength'] = controlnet.get('conditioning_scale', 0.0)
                self.controlnet_info["controlnets"].append(processed)
        else:
            self.controlnet_info = {"enabled": False, "controlnets": []}
            
        # IPAdapter configuration
        if config_data.get('use_ipadapter', False):
            self.ipadapter_info["enabled"] = True
            ipadapters = config_data.get('ipadapters', [])
            if ipadapters:
                first = ipadapters[0]
                self.ipadapter_info["scale"] = first.get('scale', 1.0)
                self.ipadapter_info["weight_type"] = first.get('weight_type', 'linear')
                # Store required model paths
                self.ipadapter_info["ipadapter_model_path"] = first.get('ipadapter_model_path')
                self.ipadapter_info["image_encoder_path"] = first.get('image_encoder_path')
                self.ipadapter_info["type"] = first.get('type', 'regular')
                self.ipadapter_info["insightface_model_name"] = first.get('insightface_model_name')
                if first.get('style_image'):
                    self.ipadapter_info["has_style_image"] = True
                    self.ipadapter_info["style_image_path"] = first['style_image']
        else:
            self.ipadapter_info = {"enabled": False, "has_style_image": False, "scale": 1.0, "weight_type": "linear"}
            
        # Pipeline hooks configuration
        for hook_type in self.pipeline_hooks.keys():
            if hook_type in config_data:
                hook_config = config_data[hook_type]
                if isinstance(hook_config, dict):
                    self.pipeline_hooks[hook_type] = {
                        "enabled": hook_config.get("enabled", False),
                        "processors": []
                    }
                    # Process processors with proper indexing
                    for index, processor in enumerate(hook_config.get("processors", [])):
                        if isinstance(processor, dict):
                            processed_processor = {
                                "index": index,
                                "name": processor.get("type", "unknown"),
                                "type": processor.get("type", "unknown"),
                                "enabled": processor.get("enabled", False),
                                "order": processor.get("order", index + 1),
                                "params": processor.get("params", {})
                            }
                            self.pipeline_hooks[hook_type]["processors"].append(processed_processor)
            else:
                self.pipeline_hooks[hook_type] = {"enabled": False, "processors": []}
        
        # Blending configurations
        self.prompt_blending = self._normalize_prompt_config(config_data)
        self.seed_blending = self._normalize_seed_config(config_data)
        
        logger.info("populate_from_config: AppState populated successfully from config")

    def _normalize_prompt_config(self, config_data):
        """Normalize prompt configuration to always return a list format"""
        if not config_data:
            return None
            
        # Check for explicit prompt_blending first
        if 'prompt_blending' in config_data:
            prompt_blending = config_data['prompt_blending']
            if isinstance(prompt_blending, dict) and 'prompt_list' in prompt_blending:
                prompt_list = prompt_blending['prompt_list']
                if isinstance(prompt_list, list) and len(prompt_list) > 0:
                    return prompt_list
            elif isinstance(prompt_blending, list) and len(prompt_blending) > 0:
                return prompt_blending
        
        # Check for direct prompt_list key  
        if 'prompt_list' in config_data:
            prompt_list = config_data['prompt_list']
            if isinstance(prompt_list, list) and len(prompt_list) > 0:
                return prompt_list
        
        # Check for simple prompt key and convert to list format
        if 'prompt' in config_data:
            prompt = config_data['prompt']
            if prompt and isinstance(prompt, str):
                return [(prompt, 1.0)]
        
        return None

    def _normalize_seed_config(self, config_data):
        """Normalize seed configuration to always return a list format"""
        if not config_data:
            return None
            
        # Check for explicit seed_blending first
        if 'seed_blending' in config_data:
            seed_blending = config_data['seed_blending']
            if isinstance(seed_blending, dict) and 'seed_list' in seed_blending:
                seed_list = seed_blending['seed_list']
                if isinstance(seed_list, list) and len(seed_list) > 0:
                    return seed_list
            elif isinstance(seed_blending, list) and len(seed_blending) > 0:
                return seed_blending
        
        # Check for direct seed_list key  
        if 'seed_list' in config_data:
            seed_list = config_data['seed_list']
            if isinstance(seed_list, list) and len(seed_list) > 0:
                return seed_list
        
        # Check for simple seed key and convert to list format
        if 'seed' in config_data:
            seed = config_data['seed']
            if seed is not None and isinstance(seed, (int, float)):
                return [(int(seed), 1.0)]
        
        return None

    def get_complete_state(self):
        """Return unified state object for frontend consumption - SINGLE SOURCE OF TRUTH"""
        return {
            # Pipeline state
            "pipeline_active": self.pipeline_active,
            "pipeline_lifecycle": self.pipeline_lifecycle,
            
            # Configuration
            "config_needs_reload": self.config_needs_reload,
            
            # Resolution
            "current_resolution": self.current_resolution,
            
            # Parameters
            "pipeline_params": self.pipeline_params,
            "controlnet": self.controlnet_info,
            "ipadapter": self.ipadapter_info,
            "prompt_blending": self.prompt_blending,
            "seed_blending": self.seed_blending,
            "normalize_prompt_weights": self.normalize_prompt_weights,
            "normalize_seed_weights": self.normalize_seed_weights,
            
            # Core parameters
            "guidance_scale": self.guidance_scale,
            "delta": self.delta,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "t_index_list": self.t_index_list,
            "negative_prompt": self.negative_prompt,
            "skip_diffusion": self.skip_diffusion,
            
            # UI state
            "fps": self.fps,
            "queue_size": self.queue_size,
            "model_id": self.model_id,
            "page_content": self.page_content,
            
            # Input sources
            "input_sources": self.input_sources,
            
            # Debug mode state
            "debug_mode": self.debug_mode,
            "debug_pending_frame": self.debug_pending_frame,
            
            # Pipeline hooks - AUTHORITATIVE SOURCE
            "image_preprocessing": self.pipeline_hooks["image_preprocessing"],
            "image_postprocessing": self.pipeline_hooks["image_postprocessing"],
            "latent_preprocessing": self.pipeline_hooks["latent_preprocessing"],
            "latent_postprocessing": self.pipeline_hooks["latent_postprocessing"],
        }
    
    def update_controlnet_strength(self, index: int, strength: float):
        """Update ControlNet strength in AppState - SINGLE SOURCE OF TRUTH"""
        if index < len(self.controlnet_info["controlnets"]):
            self.controlnet_info["controlnets"][index]["strength"] = strength
            self.controlnet_info["controlnets"][index]["conditioning_scale"] = strength
            logger.debug(f"update_controlnet_strength: Updated ControlNet {index} strength to {strength}")
        else:
            logger.warning(f"update_controlnet_strength: ControlNet index {index} out of range")
    
    def add_controlnet(self, controlnet_config: dict):
        """Add ControlNet to AppState - SINGLE SOURCE OF TRUTH"""
        index = len(self.controlnet_info["controlnets"])
        processed = dict(controlnet_config)
        processed['index'] = index
        processed['name'] = controlnet_config.get('model_id', '')
        processed['strength'] = controlnet_config.get('conditioning_scale', 0.0)
        
        self.controlnet_info["controlnets"].append(processed)
        self.controlnet_info["enabled"] = True
        logger.debug(f"add_controlnet: Added ControlNet at index {index}")
        
    def remove_controlnet(self, index: int):
        """Remove ControlNet from AppState - SINGLE SOURCE OF TRUTH"""
        if index < len(self.controlnet_info["controlnets"]):
            removed = self.controlnet_info["controlnets"].pop(index)
            # Re-index remaining controlnets
            for i, controlnet in enumerate(self.controlnet_info["controlnets"]):
                controlnet['index'] = i
            if not self.controlnet_info["controlnets"]:
                self.controlnet_info["enabled"] = False
            logger.debug(f"remove_controlnet: Removed ControlNet at index {index}")
        else:
            logger.warning(f"remove_controlnet: ControlNet index {index} out of range")
    
    def update_hook_processor(self, hook_type: str, processor_index: int, updates: dict):
        """Update pipeline hook processor in AppState - SINGLE SOURCE OF TRUTH"""
        if hook_type in self.pipeline_hooks:
            processors = self.pipeline_hooks[hook_type]["processors"]
            if processor_index < len(processors):
                processors[processor_index].update(updates)
                logger.debug(f"update_hook_processor: Updated {hook_type} processor {processor_index}")
            else:
                logger.warning(f"update_hook_processor: Processor index {processor_index} out of range for {hook_type}")
        else:
            logger.warning(f"update_hook_processor: Unknown hook type {hook_type}")
    
    def add_hook_processor(self, hook_type: str, processor_config: dict):
        """Add pipeline hook processor to AppState - SINGLE SOURCE OF TRUTH"""
        if hook_type in self.pipeline_hooks:
            index = len(self.pipeline_hooks[hook_type]["processors"])
            processed = {
                "index": index,
                "name": processor_config.get("type", "unknown"),
                "type": processor_config.get("type", "unknown"),
                "enabled": processor_config.get("enabled", True),
                "order": processor_config.get("order", index + 1),
                "params": processor_config.get("params", {})
            }
            self.pipeline_hooks[hook_type]["processors"].append(processed)
            self.pipeline_hooks[hook_type]["enabled"] = True
            logger.debug(f"add_hook_processor: Added {hook_type} processor at index {index}")
        else:
            logger.warning(f"add_hook_processor: Unknown hook type {hook_type}")
    
    def remove_hook_processor(self, hook_type: str, processor_index: int):
        """Remove pipeline hook processor from AppState - SINGLE SOURCE OF TRUTH"""
        if hook_type in self.pipeline_hooks:
            processors = self.pipeline_hooks[hook_type]["processors"]
            if processor_index < len(processors):
                removed = processors.pop(processor_index)
                # Re-index remaining processors
                for i, processor in enumerate(processors):
                    processor['index'] = i
                if not processors:
                    self.pipeline_hooks[hook_type]["enabled"] = False
                logger.debug(f"remove_hook_processor: Removed {hook_type} processor at index {processor_index}")
            else:
                logger.warning(f"remove_hook_processor: Processor index {processor_index} out of range for {hook_type}")
        else:
            logger.warning(f"remove_hook_processor: Unknown hook type {hook_type}")

    def update_parameter(self, parameter_name: str, value: float):
        """Update a single parameter in AppState - UNIFIED PARAMETER UPDATE"""
        logger.debug(f"update_parameter: Updating {parameter_name} = {value}")
        
        # Core pipeline parameters
        if parameter_name == 'guidance_scale':
            self.guidance_scale = float(value)
        elif parameter_name == 'delta':
            self.delta = float(value)
        elif parameter_name == 'num_inference_steps':
            self.num_inference_steps = int(value)
        elif parameter_name == 'seed':
            self.seed = int(value)
        elif parameter_name == 'negative_prompt':
            self.negative_prompt = str(value)
        elif parameter_name == 'skip_diffusion':
            self.skip_diffusion = bool(value)
        elif parameter_name == 't_index_list':
            if isinstance(value, list):
                self.t_index_list = value
            else:
                logger.warning(f"update_parameter: t_index_list must be a list, got {type(value)}")
        
        # IPAdapter parameters
        elif parameter_name == 'ipadapter_scale':
            self.ipadapter_info["scale"] = float(value)
        elif parameter_name == 'ipadapter_weight_type':
            # Convert numeric value to weight type string
            weight_types = ["linear", "ease in", "ease out", "ease in-out", "reverse in-out", 
                           "weak input", "weak output", "weak middle", "strong middle", 
                           "style transfer", "composition", "strong style transfer", 
                           "style and composition", "style transfer precise", "composition precise"]
            index = int(value) % len(weight_types)
            self.ipadapter_info["weight_type"] = weight_types[index]
        
        # ControlNet strength parameters
        elif parameter_name.startswith('controlnet_') and parameter_name.endswith('_strength'):
            import re
            match = re.match(r'controlnet_(\d+)_strength', parameter_name)
            if match:
                index = int(match.group(1))
                self.update_controlnet_strength(index, float(value))
        
        # ControlNet preprocessor parameters
        elif parameter_name.startswith('controlnet_') and '_preprocessor_' in parameter_name:
            import re
            match = re.match(r'controlnet_(\d+)_preprocessor_(.+)', parameter_name)
            if match:
                controlnet_index = int(match.group(1))
                param_name = match.group(2)
                if controlnet_index < len(self.controlnet_info["controlnets"]):
                    controlnet = self.controlnet_info["controlnets"][controlnet_index]
                    if 'preprocessor_params' not in controlnet:
                        controlnet['preprocessor_params'] = {}
                    controlnet['preprocessor_params'][param_name] = value
        
        # Prompt blending weights
        elif parameter_name.startswith('prompt_weight_'):
            import re
            match = re.match(r'prompt_weight_(\d+)', parameter_name)
            if match:
                index = int(match.group(1))
                if self.prompt_blending and index < len(self.prompt_blending):
                    # Update weight in prompt blending list
                    prompt_text = self.prompt_blending[index][0]
                    self.prompt_blending[index] = (prompt_text, float(value))
        
        # Seed blending weights
        elif parameter_name.startswith('seed_weight_'):
            import re
            match = re.match(r'seed_weight_(\d+)', parameter_name)
            if match:
                index = int(match.group(1))
                if self.seed_blending and index < len(self.seed_blending):
                    # Update weight in seed blending list
                    seed_value = self.seed_blending[index][0]
                    self.seed_blending[index] = (seed_value, float(value))
        
        else:
            logger.warning(f"update_parameter: Unknown parameter {parameter_name}")
            return
        
        logger.debug(f"update_parameter: Successfully updated {parameter_name} in AppState")

    def generate_pipeline_config(self):
        """Generate pipeline configuration from AppState - PRESERVES ALL ORIGINAL CONFIG"""
        logger.info("generate_pipeline_config: Generating pipeline config from AppState, preserving all original config")
        
        # Start with complete original config to preserve ALL parameters
        config = {}
        if self.uploaded_config:
            config = dict(self.uploaded_config)
        
        # Only override runtime-changeable parameters from AppState
        config.update({
            'guidance_scale': self.guidance_scale,
            'delta': self.delta,
            'num_inference_steps': self.num_inference_steps,
            'seed': self.seed,
            't_index_list': self.t_index_list,
            'negative_prompt': self.negative_prompt,
            'skip_diffusion': self.skip_diffusion,
            'width': self.current_resolution["width"],
            'height': self.current_resolution["height"],
            'output_type': 'pt',  # Force optimal tensor performance
        })
        
        # Update ControlNet configurations with current AppState values
        if self.controlnet_info["enabled"] and self.controlnet_info["controlnets"]:
            config['controlnets'] = []
            for controlnet in self.controlnet_info["controlnets"]:
                cn_config = dict(controlnet)
                # Ensure conditioning_scale reflects current strength
                cn_config['conditioning_scale'] = controlnet.get('strength', controlnet.get('conditioning_scale', 0.0))
                config['controlnets'].append(cn_config)
        elif 'controlnets' in config:
            # Remove controlnets if disabled
            del config['controlnets']
        
        # Update IPAdapter configurations with current AppState values
        if self.ipadapter_info["enabled"]:
            config['use_ipadapter'] = True
            # Preserve original ipadapters config but update runtime values
            if 'ipadapters' in config and config['ipadapters']:
                # Update existing config with current values
                config['ipadapters'][0].update({
                    'scale': self.ipadapter_info["scale"],
                    'weight_type': self.ipadapter_info["weight_type"]
                })
                # Add style image if available
                if self.ipadapter_info.get("has_style_image") and self.ipadapter_info.get("style_image_path"):
                    config['ipadapters'][0]['style_image'] = self.ipadapter_info["style_image_path"]
        elif 'use_ipadapter' in config:
            # Disable IPAdapter if not enabled in AppState
            config['use_ipadapter'] = False
        
        # Update pipeline hooks with current AppState values
        for hook_type, hook_config in self.pipeline_hooks.items():
            if hook_config["enabled"] and hook_config["processors"]:
                config[hook_type] = {
                    "enabled": True,
                    "processors": []
                }
                for processor in hook_config["processors"]:
                    proc_config = {
                        "type": processor["type"],
                        "enabled": processor["enabled"],
                        "order": processor["order"],
                        "params": processor["params"]
                    }
                    config[hook_type]["processors"].append(proc_config)
            elif hook_type in config:
                # Disable hook if not enabled in AppState
                config[hook_type] = {"enabled": False, "processors": []}
        
        # Update blending configurations with current AppState values
        if self.prompt_blending:
            config['prompt_blending'] = {
                'prompt_list': self.prompt_blending,
                'interpolation_method': 'slerp'
            }
            config['normalize_weights'] = self.normalize_prompt_weights
        elif 'prompt_blending' in config:
            del config['prompt_blending']
        
        if self.seed_blending:
            config['seed_blending'] = {
                'seed_list': self.seed_blending,
                'interpolation_method': 'linear'
            }
            # Note: seed normalization uses same normalize_weights key
            if not self.prompt_blending:  # Only set if not already set by prompt blending
                config['normalize_weights'] = self.normalize_seed_weights
        elif 'seed_blending' in config:
            del config['seed_blending']
        
        logger.info("generate_pipeline_config: Generated pipeline config preserving all original parameters")
        return config

    def update_state(self, updates):
        """Atomic state updates with validation"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"AppState update_state: Updated {key} = {value}")
            else:
                logger.warning(f"AppState update_state: Unknown state key: {key}")
    


class App:
    def __init__(self, config: Args):
        self.args = config
        self.pipeline = None  # Pipeline created lazily when needed
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.fps_counter = []
        self.last_fps_update = time.time()
        
        # Centralized state management
        self.app_state = AppState()
        
        # Initialize input manager for controller support
        self.input_manager = InputManager()
        # Initialize input source manager for modular input routing
        from input_sources import InputSourceManager
        self.input_source_manager = InputSourceManager()
        self.init_app()

    def cleanup(self):
        """Cleanup resources when app is shutting down"""
        logger.info("App cleanup: Starting application cleanup...")
        if self.pipeline:
            self.app_state.pipeline_lifecycle = "stopping"
            self._cleanup_pipeline(self.pipeline)
            self.pipeline = None
            self.app_state.pipeline_lifecycle = "stopped"
        if hasattr(self, 'input_source_manager'):
            self.input_source_manager.cleanup()
        self._cleanup_temp_files()
        logger.info("App cleanup: Completed application cleanup")

    def _handle_input_parameter_update(self, parameter_name: str, value: float) -> None:
        """Handle parameter updates from input controls - UNIFIED THROUGH APPSTATE"""
        try:
            logger.debug(f"_handle_input_parameter_update: Updating {parameter_name} = {value} via AppState")
            
            # Update AppState as single source of truth
            self.app_state.update_parameter(parameter_name, value)
            
            # Sync to pipeline if active (for real-time updates)
            if self.pipeline and hasattr(self.pipeline, 'stream'):
                self._sync_appstate_to_pipeline()
            else:
                logger.debug(f"_handle_input_parameter_update: No active pipeline, parameter stored in AppState for next pipeline creation")

        except Exception as e:
            logger.exception(f"_handle_input_parameter_update: Failed to update {parameter_name}: {e}")

    def _update_resolution(self, width: int, height: int) -> None:
        """Update resolution by recreating pipeline with new dimensions"""
        logger.info(f"_update_resolution: Updating resolution to {width}x{height}")
        
        # Update AppState first
        self.app_state.current_resolution = {"width": width, "height": height}
        
        # If no pipeline exists, just update state (will be used when pipeline is created)
        if not self.pipeline:
            logger.info("_update_resolution: No pipeline exists, resolution will apply on next pipeline creation")
            return
        
        # Set pipeline lifecycle state
        self.app_state.pipeline_lifecycle = "restarting"
        
        # Store reference to old pipeline for cleanup
        old_pipeline = self.pipeline
        
        # Clear current pipeline reference before cleanup
        self.pipeline = None
        
        # Cleanup old pipeline and free VRAM
        if old_pipeline:
            self._cleanup_pipeline(old_pipeline)
            old_pipeline = None
        
        # Create new pipeline with new resolution
        # No state restoration needed - _create_pipeline() uses AppState as single source of truth
        try:
            self.pipeline = self._create_pipeline()
            self.app_state.pipeline_lifecycle = "running"
            logger.info(f"_update_resolution: Pipeline successfully recreated with resolution {width}x{height}")
            
        except Exception as e:
            self.app_state.pipeline_lifecycle = "error"
            logger.error(f"_update_resolution: Failed to recreate pipeline: {e}")
            raise

    def _sync_appstate_to_pipeline(self):
        """Sync AppState parameters to active pipeline for real-time updates"""
        try:
            if not self.pipeline or not hasattr(self.pipeline, 'stream'):
                return
            
            # Core parameters
            self.pipeline.update_stream_params(
                guidance_scale=self.app_state.guidance_scale,
                delta=self.app_state.delta,
                num_inference_steps=self.app_state.num_inference_steps,
                seed=self.app_state.seed,
                negative_prompt=self.app_state.negative_prompt,
                t_index_list=self.app_state.t_index_list
            )
            
            # IPAdapter parameters
            if self.app_state.ipadapter_info["enabled"]:
                self.pipeline.update_stream_params(ipadapter_config={
                    'scale': self.app_state.ipadapter_info["scale"]
                })
                if hasattr(self.pipeline, 'update_ipadapter_weight_type'):
                    self.pipeline.update_ipadapter_weight_type(self.app_state.ipadapter_info["weight_type"])
            
            # ControlNet parameters
            if self.app_state.controlnet_info["enabled"] and self.app_state.controlnet_info["controlnets"]:
                controlnet_config = []
                for cn in self.app_state.controlnet_info["controlnets"]:
                    config_entry = dict(cn)
                    config_entry['conditioning_scale'] = cn['strength']
                    controlnet_config.append(config_entry)
                self.pipeline.update_stream_params(controlnet_config=controlnet_config)
            
            # Prompt blending
            if self.app_state.prompt_blending:
                self.pipeline.update_stream_params(prompt_list=self.app_state.prompt_blending)
            
            # Seed blending
            if self.app_state.seed_blending:
                self.pipeline.update_stream_params(seed_list=self.app_state.seed_blending)
            
            logger.debug("_sync_appstate_to_pipeline: Successfully synced AppState to pipeline")
            
        except Exception as e:
            logger.exception(f"_sync_appstate_to_pipeline: Failed to sync AppState to pipeline: {e}")


    


    def _get_controlnet_pipeline(self):
        """Get the ControlNet pipeline from the main pipeline structure"""
        if not self.pipeline:
            return None
            
        stream = self.pipeline.stream
        
        # Module-aware: module installs expose preprocessors on stream
        if hasattr(stream, 'preprocessors'):
            return stream
            
        # Check if stream has nested stream (IPAdapter wrapper)
        if hasattr(stream, 'stream') and hasattr(stream.stream, 'preprocessors'):
            return stream.stream
            
        # New module path on stream
        if hasattr(stream, '_controlnet_module'):
            return stream._controlnet_module
        return None


    def init_app(self):
        # Enhanced CORS for API-only development mode
        if self.args.api_only:
            # More permissive CORS for development
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # Include common Vite dev ports
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        else:
            # Standard CORS for production
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Set up input manager callback for parameter updates
        self.input_manager.set_parameter_update_callback(self._handle_input_parameter_update)

        # Register route modules
        self._register_routes()
    
    def _register_routes(self):
        """Register all route modules with dependency injection"""
        from routes import parameters, controlnet, ipadapter, inference, pipeline_hooks, websocket, input_sources, debug
        from routes.common.dependencies import get_app_instance as shared_get_app_instance, get_pipeline_class as shared_get_pipeline_class, get_default_settings as shared_get_default_settings, get_available_controlnets as shared_get_available_controlnets
        
        # Create dependency overrides to inject app instance and other dependencies
        def get_app_instance():
            return self
            
        def get_pipeline_class():
            return Pipeline
            
        def get_default_settings():
            return DEFAULT_SETTINGS
            
        def get_available_controlnets():
            return AVAILABLE_CONTROLNETS
        
        # Include routers and set up dependency overrides on the main app
        for router_module in [parameters, controlnet, ipadapter, inference, pipeline_hooks, websocket, input_sources, debug]:
            # Include the router
            self.app.include_router(router_module.router)
        
        # Set up dependency overrides on the main app (not individual routers)
        self.app.dependency_overrides[shared_get_app_instance] = get_app_instance
        self.app.dependency_overrides[shared_get_pipeline_class] = get_pipeline_class
        self.app.dependency_overrides[shared_get_default_settings] = get_default_settings
        self.app.dependency_overrides[shared_get_available_controlnets] = get_available_controlnets
        
        # Set up static files if not in API-only mode
        if not self.args.api_only:
            self.app.mount("/", StaticFiles(directory="frontend/public", html=True), name="public")


    def _create_pipeline(self):
        """Create pipeline using AppState as single source of truth"""
        logger.info("_create_pipeline: Creating pipeline using AppState as single source of truth")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16
        
        # Generate pipeline config from AppState - SINGLE SOURCE OF TRUTH
        pipeline_config = self.app_state.generate_pipeline_config()
        
        # Load config style images into InputSourceManager before creating pipeline
        self._load_config_style_images()
        
        # Create wrapper using the unified config - THIS IS NOW THE SINGLE PLACE WHERE WRAPPER IS CREATED
        from src.streamdiffusion.config import create_wrapper_from_config
        
        # Create wrapper using the unified config
        wrapper = create_wrapper_from_config(pipeline_config)
        
        # Update args with config values before passing to Pipeline
        from config import Args
        args_dict = self.args._asdict()
        if 'acceleration' in pipeline_config:
            args_dict['acceleration'] = pipeline_config['acceleration']
        if 'engine_dir' in pipeline_config:
            args_dict['engine_dir'] = pipeline_config['engine_dir']
        if 'use_safety_checker' in pipeline_config:
            args_dict['safety_checker'] = pipeline_config['use_safety_checker']
        
        updated_args = Args(**args_dict)
        
        # Create Pipeline instance with the pre-created wrapper and config
        pipeline = Pipeline(wrapper=wrapper, config=pipeline_config)
        
        logger.info("_create_pipeline: Pipeline created successfully with pre-created wrapper")
        return pipeline

    def _load_config_style_images(self):
        """Load style images from config into InputSourceManager"""
        if not self.app_state.uploaded_config:
            return
            
        try:
            # Load IPAdapter style images from config
            ipadapters = self.app_state.uploaded_config.get('ipadapters', [])
            if ipadapters:
                first_ipadapter = ipadapters[0]
                style_image_path = first_ipadapter.get('style_image')
                if style_image_path:
                    # Use the config file path as base for relative paths
                    base_config_path = getattr(self.args, 'controlnet_config', None)
                    self.input_source_manager.load_config_style_image(style_image_path, base_config_path)
        except Exception as e:
            logging.exception(f"_load_config_style_images: Error loading config style images: {e}")

    def _cleanup_temp_files(self):
        """Clean up any temporary config files"""
        if hasattr(self, '_temp_config_files'):
            import os
            for temp_path in self._temp_config_files:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass
            self._temp_config_files.clear()


    def _calculate_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate and return aspect ratio as a string"""
        import math
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        ratio_gcd = gcd(width, height)
        return f"{width//ratio_gcd}:{height//ratio_gcd}"

    def _cleanup_pipeline(self, pipeline):
        """Properly cleanup a pipeline and free VRAM"""
        if pipeline is None:
            return
        
        try:
            if hasattr(pipeline, 'cleanup'):
                pipeline.cleanup()
            del pipeline
            torch.cuda.empty_cache()
        except Exception as e:
            logging.exception(f"_cleanup_pipeline: Error during cleanup: {e}")


app = App(config).app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
    )