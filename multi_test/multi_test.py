#!/usr/bin/env python3
"""
StreamDiffusion Multi-Config Test Suite

This script processes multiple videos with multiple YAML configurations,
similar to main.py but for batch testing. It can use individual prompts
from a text file or config prompts.

Key Features:
- Memory-efficient processing with automatic cleanup between configs
- One merged video output per config (combining all prompt segments)
- Real-time memory monitoring and cleanup
- Pipeline reset between configs to prevent memory issues

Usage:
    python multi_test.py --configs ./configs --videos ./videos --output ./results
    python multi_test.py --configs ./configs --videos ./videos --prompts ./prompts.txt --output ./results
    python multi_test.py --configs ./configs --videos ./videos --output ./results --timeout_seconds 600

Based on the StreamDiffusion framework and main.py architecture.
"""

import os
import datetime
import sys
import time
import yaml
import argparse
import signal
import atexit
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

try:
    import fire
except ImportError:
    print("Error: 'fire' package not found. Please install it with: pip install fire")
    sys.exit(1)

try:
    import ffmpeg
except ImportError:
    print("Warning: ffmpeg-python library not found. Video wall creation will be disabled.")
    print("To enable video wall creation, install it with: pip install ffmpeg-python")
    ffmpeg = None

# Import enhanced video wall functions if available
try:
    from enhanced_video_wall import create_enhanced_video_with_metadata
except ImportError:
    print("Warning: Enhanced video wall module not found. Will use fallback video processing.")
    create_enhanced_video_with_metadata = None

import torch
from torchvision.io import read_video, write_video
from torchvision.transforms import functional as F
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from streamdiffusion import StreamDiffusionWrapper, load_config, create_wrapper_from_config

# Global cleanup flag
_cleanup_completed = False

def signal_handler(signum, frame):
    """Handle system signals to ensure cleanup before exit."""
    print(f"\nReceived signal {signum}, cleaning up...")
    cleanup_and_exit()
    sys.exit(1)

def cleanup_and_exit():
    """Ensure cleanup is performed before exit."""
    global _cleanup_completed
    if not _cleanup_completed:
        print("Performing final cleanup...")
        try:
            # Multiple rounds of cleanup to ensure everything is freed
            for cleanup_round in range(3):
                cleanup_gpu_memory()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # Small delay between cleanup rounds
                import time
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Warning: Final cleanup failed: {e}")
        _cleanup_completed = True
        print("Cleanup completed.")

# Register signal handlers and exit handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_and_exit)

def cleanup_gpu_memory():
    """Thorough GPU memory cleanup."""
    try:
        if torch.cuda.is_available():
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Log memory after cleanup
            memory_info = get_memory_info()
            if memory_info:
                print(f"    Memory after cleanup: GPU allocated: {memory_info['gpu_allocated']:.2f} GB, "
                      f"reserved: {memory_info['gpu_reserved']:.2f} GB, free: {memory_info['gpu_free']:.2f} GB")
    except Exception as e:
        print(f"    Warning: Memory cleanup failed: {e}")
        pass

def cleanup_pipeline(pipeline):
    """Properly cleanup a pipeline and free VRAM using StreamDiffusion's built-in cleanup"""
    if pipeline is None:
        return
        
    try:
        print("    Starting pipeline cleanup...")
        
        # Use StreamDiffusion's built-in cleanup method which properly handles:
        # - TensorRT engine cleanup
        # - ControlNet engine cleanup  
        # - Multiple garbage collection cycles
        # - CUDA cache clearing
        # - Memory tracking
        if hasattr(pipeline, 'stream') and pipeline.stream and hasattr(pipeline.stream, 'cleanup_gpu_memory'):
            pipeline.stream.cleanup_gpu_memory()
            print("    Pipeline cleanup completed using StreamDiffusion cleanup")
        elif hasattr(pipeline, 'cleanup_gpu_memory') and callable(getattr(pipeline, 'cleanup_gpu_memory')):
            pipeline.cleanup_gpu_memory()
            print("    Pipeline cleanup completed using pipeline cleanup method")
        elif hasattr(pipeline, 'cleanup') and callable(getattr(pipeline, 'cleanup')):
            pipeline.cleanup()
            print("    Pipeline cleanup completed using generic cleanup method")
        else:
            # Fallback cleanup if the method doesn't exist
            print("    StreamDiffusion cleanup method not found, using fallback cleanup")
            if hasattr(pipeline, 'stream') and pipeline.stream:
                del pipeline.stream
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"    Error during pipeline cleanup: {e}")
        # Still try to clear CUDA cache even if cleanup fails
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def preprocess_video(input_video_path: str, target_width: int, target_height: int) -> torch.Tensor:
    """Memory-efficient video preprocessing to target resolution, maintaining aspect ratio."""
    print(f"Preprocessing video: {input_video_path}")
    print(f"  Target resolution: {target_width}x{target_height}")
    
    # Load video metadata first to check size
    video_data, _, info = read_video(input_video_path, pts_unit='sec')
    original_fps = info["video_fps"]
    num_frames = video_data.shape[0]
    print(f"  Original FPS: {original_fps}")
    print(f"  Loaded video shape: {video_data.shape}")
    
    # Calculate memory usage and warn if large
    estimated_memory_gb = (num_frames * target_height * target_width * 3 * 4) / (1024**3)  # 4 bytes per float32
    print(f"  Estimated memory usage: {estimated_memory_gb:.2f} GB")
    
    if estimated_memory_gb > 4.0:
        print(f"  ⚠️  WARNING: Large video detected! Consider using batch processing for videos > 4GB")
    
    # Calculate resize parameters once
    original_height, original_width = video_data.shape[1], video_data.shape[2]
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height
    
    if original_aspect > target_aspect:
        scale_height = target_height
        scale_width = int(scale_height * original_aspect)
    else:
        scale_width = target_width
        scale_height = int(scale_width / original_aspect)
    
    print(f"  Resizing and cropping frames...")
    
    # Pre-allocate output tensor to avoid memory fragmentation
    resized_video = torch.zeros(num_frames, target_height, target_width, 3, dtype=torch.float32)
    
    # Process frames in smaller batches to reduce peak memory usage
    batch_size = min(50, num_frames)  # Process 50 frames at a time
    
    for batch_start in tqdm(range(0, num_frames, batch_size), desc="  Processing batches"):
        batch_end = min(batch_start + batch_size, num_frames)
        
        # Process batch of frames
        for i in range(batch_start, batch_end):
            # Convert to float and normalize (in-place to save memory)
            frame = video_data[i].float() / 255.0  # Shape: (H, W, C)
            frame_chw = frame.permute(2, 0, 1)
            
            # Resize maintaining aspect ratio
            resized_frame_chw = F.resize(frame_chw, [scale_height, scale_width], antialias=True)
            cropped_frame_chw = F.center_crop(resized_frame_chw, [target_height, target_width])
            final_frame = cropped_frame_chw.permute(1, 2, 0)
            
            # Store directly in pre-allocated tensor
            resized_video[i] = final_frame
            
            # Clean up intermediate tensors
            del frame, frame_chw, resized_frame_chw, cropped_frame_chw, final_frame
        
        # Force garbage collection after each batch
        import gc
        gc.collect()
    
    # Clean up original video data
    del video_data
    gc.collect()
    
    print(f"  Final processed video shape: {resized_video.shape}")
    print(f"  Memory cleanup completed")
    return resized_video

def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from text file."""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {prompts_file}")
    return prompts

def scan_completed_work(resume_dir: str) -> List[Dict]:
    """
    Load existing results from CSV and JSON metadata if available.
    
    Parameters
    ----------
    resume_dir : str
        Path to existing output directory to resume from
        
    Returns
    -------
    List[Dict]
        List of existing results (both successful and failed from CSV if available)
    """
    print(f"\n🔍 Scanning for completed work in: {resume_dir}")
    
    if not os.path.exists(resume_dir):
        print(f"❌ Resume directory does not exist: {resume_dir}")
        return []
    
    existing_results = []
    json_metadata = {}  # Store JSON metadata by video filename
    
    # First, scan for JSON metadata files
    print(f"📋 Scanning for JSON metadata files...")
    try:
        import json
        json_files = [f for f in os.listdir(resume_dir) if f.endswith('_metadata.json')]
        for json_file in json_files:
            json_path = os.path.join(resume_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Use output filename as key for easy lookup
                    output_filename = metadata.get('video_info', {}).get('output_filename', '')
                    if output_filename:
                        json_metadata[output_filename] = metadata
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load JSON metadata {json_file}: {e}")
        
        print(f"  Found {len(json_metadata)} JSON metadata files")
    except Exception as e:
        print(f"  ⚠️  Warning: Error scanning JSON files: {e}")
    
    # Try to load existing results from CSV
    csv_path = os.path.join(resume_dir, "detailed_results.csv")
    if os.path.exists(csv_path):
        print(f"📊 Loading existing results from CSV: {csv_path}")
        try:
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Load both successful AND failed results to preserve all data
                    is_successful = row['Success'] == 'Yes'
                    
                    if is_successful:
                        # Reconstruct successful result dict
                        result_dict = {
                            'config': row['Config'],
                            'video': row['Video'],
                            'model_id': row['Model ID'],
                            'resolution': row['Resolution'],
                            'total_frames': int(row['Total Frames']) if row['Total Frames'].isdigit() else 0,
                            'prompts_used': int(row['Prompts Used']) if row['Prompts Used'].isdigit() else 1,
                            'success': True,
                            'output_file': row['Output File'],
                            'fps_metrics': {
                                'overall_fps': float(row['Overall FPS']) if row['Overall FPS'] != 'N/A' else 0,
                                'min_fps': float(row['Min FPS']) if row['Min FPS'] != 'N/A' else 0,
                                'max_fps': float(row['Max FPS']) if row['Max FPS'] != 'N/A' else 0,
                                'avg_fps': float(row['Avg FPS']) if row['Avg FPS'] != 'N/A' else 0,
                                'std_dev_fps': float(row['Std Dev FPS']) if row['Std Dev FPS'] != 'N/A' else 0,
                                'cv_percent': float(row['CV %']) if row['CV %'] != 'N/A' else 0
                            }
                        }
                        
                        # Enhance with JSON metadata if available
                        output_file = row['Output File']
                        if output_file in json_metadata:
                            result_dict['json_metadata'] = json_metadata[output_file]
                            print(f"    ✅ Enhanced {output_file} with JSON metadata")
                        
                        existing_results.append(result_dict)
                    else:
                        # Reconstruct failed result dict
                        existing_results.append({
                            'config': row['Config'],
                            'video': row['Video'],
                            'model_id': row['Model ID'],
                            'resolution': row['Resolution'],
                            'total_frames': int(row['Total Frames']) if row['Total Frames'].isdigit() else 0,
                            'prompts_used': int(row['Prompts Used']) if row['Prompts Used'].isdigit() else 1,
                            'success': False,
                            'error': row['Error Message'] if row['Error Message'] else 'Unknown error'
                        })
            
            successful_count = sum(1 for r in existing_results if r['success'])
            failed_count = len(existing_results) - successful_count
            print(f"✅ Loaded {len(existing_results)} total results from CSV:")
            print(f"    - {successful_count} successful")
            print(f"    - {failed_count} failed")
        except Exception as e:
            print(f"⚠️  Warning: Could not load CSV results: {e}")
    else:
        print(f"📊 No existing CSV found - will create new results")
    
    return existing_results

def check_output_exists(output_dir: str, config_filename: str, video_filename: str, config: Dict, prompts: Optional[List[str]] = None, existing_results: Optional[List[Dict]] = None, retry_failed: bool = False) -> bool:
    """
    Check if output file already exists for this config+video combination.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    config_filename : str
        Config filename (without extension)
    video_filename : str
        Video filename (without extension)
    config : Dict
        Configuration dictionary
    prompts : Optional[List[str]]
        List of prompts (to determine filename format)
    existing_results : Optional[List[Dict]]
        List of existing results to check against (for resume functionality)
    retry_failed : bool, optional
        Whether to retry previously failed combinations, by default False
        
    Returns
    -------
    bool
        True if output file already exists or combination was already processed
    """
    # First check if this combination was already processed (from loaded CSV data)
    if existing_results:
        for result in existing_results:
            if result['config'] == config_filename and result['video'] == video_filename:
                if result['success']:
                    print(f"    ✅ Combination already completed successfully: {config_filename} + {video_filename}")
                    return True
                else:
                    if retry_failed:
                        print(f"    🔄 Retrying previously failed combination: {config_filename} + {video_filename} (Previous error: {result.get('error', 'Unknown')})")
                        return False  # Allow retry
                    else:
                        print(f"    ⚠️  Combination previously failed: {config_filename} + {video_filename} (Error: {result.get('error', 'Unknown')})")
                        return True  # Skip retry
    
    # Then check if output file exists on disk
    # Generate the expected output filename using the same logic as process_video_with_config
    config_name = config.get('model_id', 'unknown')
    # Clean up the config name to make it filesystem-safe
    if '/' in config_name:
        config_name = config_name.split('/')[-1]
    if '\\' in config_name:
        config_name = config_name.split('\\')[-1]
    # Remove file extensions
    config_name = config_name.replace('.safetensors', '').replace('.ckpt', '').replace('.pth', '')
    
    # Create expected filename
    num_prompts = len(prompts) if prompts else 1
    expected_filename = f"{config_filename}_{video_filename}_{config_name}_merged_{num_prompts}prompts.mp4"
    expected_path = os.path.join(output_dir, expected_filename)
    
    exists = os.path.exists(expected_path)
    if exists:
        print(f"    ✅ Output file already exists: {expected_filename}")
    
    return exists

def process_video_with_config(
    video: torch.Tensor,
    config: Dict,
    prompts: Optional[List[str]] = None,
    output_dir: str = "./output",
    config_filename: str = "unknown_config",
    video_filename: str = "unknown_video",
    timeout_seconds: int = 600  # 10 minutes timeout per video
) -> Optional[Dict]:
    """Process a video with a config, optionally using custom prompts with temporal splitting.
    
    Parameters
    ----------
    video : torch.Tensor
        Input video tensor
    config : Dict
        Configuration dictionary
    prompts : Optional[List[str]], optional
        List of prompts for temporal splitting, by default None
    output_dir : str, optional
        Output directory for results, by default "./output"
    config_filename : str, optional
        Name of the config file (without extension) for output filename, by default "unknown_config"
    video_filename : str, optional
        Name of the video file (without extension) for output filename, by default "unknown_video"
    timeout_seconds : int, optional
        Maximum time to spend processing this video, by default 600 (10 minutes)
    """
    
    print(f"\nProcessing with config: {config.get('model_id', 'Unknown')}")
    print(f"  Timeout set to {timeout_seconds} seconds")
    
    # Track start time for timeout
    start_time = time.time()
    
    # Clean GPU state before building pipeline
    cleanup_gpu_memory()
    log_memory_usage("before pipeline creation")
    
    stream = None
    try:
        # Check timeout before starting
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Timeout exceeded before starting processing")
        
        # Create wrapper using config system
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16
        
        overrides = {
            'device': device,
            'dtype': torch_dtype,
            'output_type': 'pt',
        }
        
        print("  Creating pipeline...")
        stream = create_wrapper_from_config(config, **overrides)
        log_memory_usage("after pipeline creation")
        
        if stream is None:
            raise RuntimeError("Failed to create pipeline - stream is None")
        
        # Check timeout after pipeline creation
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"Timeout exceeded after pipeline creation")
        
        # Debug ControlNet setup
        print(f"  Stream created successfully")
        if hasattr(stream, 'preprocessors'):
            print(f"  ControlNet preprocessors found: {len(stream.preprocessors)}")
            for idx, preproc in enumerate(stream.preprocessors):
                if preproc:
                    print(f"    Preprocessor {idx}: {preproc.__class__.__name__}")
                    if hasattr(preproc, 'params'):
                        print(f"      Params: {preproc.params}")
                else:
                    print(f"    Preprocessor {idx}: None")
        else:
            print(f"  No ControlNet preprocessors found on stream")
        
        # Check if ControlNet images are available
        if hasattr(stream, 'controlnet_images'):
            print(f"  ControlNet images available: {len(stream.controlnet_images)}")
            for idx, img in enumerate(stream.controlnet_images):
                if img is not None:
                    print(f"    ControlNet {idx} image shape: {img.shape if hasattr(img, 'shape') else 'Unknown'}")
                else:
                    print(f"    ControlNet {idx} image: None")
        
        # Check what ControlNet methods are available
        controlnet_methods = []
        if hasattr(stream, 'update_control_image'):
            controlnet_methods.append('update_control_image')
        if hasattr(stream, 'update_control_image_efficient'):
            controlnet_methods.append('update_control_image_efficient')
        if hasattr(stream, 'stream') and hasattr(stream.stream, 'update_control_image'):
            controlnet_methods.append('stream.update_control_image')
        if hasattr(stream, 'stream') and hasattr(stream.stream, 'update_control_image_efficient'):
            controlnet_methods.append('stream.update_control_image_efficient')
        
        print(f"  Available ControlNet methods: {controlnet_methods}")
        
        # Check if we have a nested stream structure
        if hasattr(stream, 'stream'):
            print(f"  Stream has nested stream object")
            if hasattr(stream.stream, 'preprocessors'):
                print(f"  Nested stream has {len(stream.stream.preprocessors)} preprocessors")
        else:
            print(f"  Stream is direct (no nested structure)")
        
        # Get base prompt from config if no custom prompts
        if not prompts:
            prompt_config = config.get('prompt_blending', {})
            if isinstance(prompt_config, dict) and 'prompt_list' in prompt_config:
                first_prompt = prompt_config['prompt_list'][0][0] if prompt_config['prompt_list'] else "a beautiful landscape"
            else:
                first_prompt = config.get('prompt', 'a beautiful landscape')
            prompts = [first_prompt]
        
        # Calculate frames per prompt for temporal splitting
        total_frames = video.shape[0]
        frames_per_prompt = total_frames // len(prompts)
        remaining_frames = total_frames % len(prompts)
        
        print(f"  Total frames: {total_frames}, Frames per prompt: {frames_per_prompt}")
        print(f"  Remaining frames: {remaining_frames} (will be distributed to first prompts)")
        
        # Process each prompt against its time segment and accumulate results
        fps_metrics = []  # Track FPS for each segment
        segment_times = []  # Track actual processing time for each segment
        all_output_frames = []  # Accumulate all segments for final merged video
        
        for i, prompt in enumerate(prompts):
            # Check timeout before processing each prompt
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Timeout exceeded while processing prompt {i+1}")
                
            print(f"  Processing prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            
            try:
                # Calculate frame range for this prompt
                start_frame = i * frames_per_prompt
                end_frame = start_frame + frames_per_prompt
                
                # Distribute remaining frames to first prompts
                if i < remaining_frames:
                    end_frame += 1
                
                # Get frames for this time segment
                segment_frames = video[start_frame:end_frame]
                print(f"    Processing frames {start_frame+1}-{end_frame} ({len(segment_frames)} frames)")
                
                # Update stream with new prompt (no pipeline restart needed)
                stream.update_prompt(prompt)
                
                # Prepare the stream if this is the first prompt
                if i == 0:
                    stream.prepare(
                        prompt=prompt,
                        negative_prompt=config.get('negative_prompt', ''),
                        num_inference_steps=config.get('num_inference_steps', 35),
                        guidance_scale=config.get('guidance_scale', 1.5),
                    )
                
                # Process frames for this time segment
                print("  Processing frames...")
                segment_start_time = time.time()
                
                # Create output tensor for this segment
                height, width = segment_frames.shape[1], segment_frames.shape[2]
                segment_result = torch.zeros(len(segment_frames), height, width, 3, dtype=torch.float32)
                
                # Warmup on first frame if this is the first prompt
                if i == 0:
                    print("  Warming up...")
                    try:
                        for _ in range(min(stream.batch_size, 3)):  # Limit warmup to prevent memory issues
                            warmup_result = stream(image=segment_frames[0].permute(2, 0, 1))
                            if warmup_result is None:
                                print("    Warning: Warmup returned None")
                    except Exception as e:
                        print(f"    Warning: Warmup failed: {e}")
                
                # Process frames for this time segment
                for j in tqdm(range(len(segment_frames)), desc="  Processing frames"):
                    # Check timeout periodically during frame processing
                    if j % 10 == 0 and time.time() - start_time > timeout_seconds:
                        raise TimeoutError(f"Timeout exceeded while processing frame {j}")
                        
                    try:
                        # Get the input frame
                        input_frame = segment_frames[j].permute(2, 0, 1)
                        
                        # Apply ControlNet preprocessing if available
                        if hasattr(stream, 'preprocessors') and stream.preprocessors:
                            # Convert frame to PIL Image for ControlNet preprocessing
                            import torchvision.transforms.functional as F
                            frame_pil = F.to_pil_image(input_frame)
                            
                            # Update control image for each ControlNet - call directly on the wrapper
                            for cn_idx in range(len(stream.preprocessors)):
                                if stream.preprocessors[cn_idx]:
                                    try:
                                        stream.update_control_image(index=cn_idx, image=frame_pil)
                                    except Exception as e:
                                        print(f"      Warning: ControlNet {cn_idx} update failed: {e}")
                        elif hasattr(stream, 'stream') and hasattr(stream.stream, 'preprocessors') and stream.stream.preprocessors:
                            # Handle nested stream structure - still call update_control_image on the wrapper
                            import torchvision.transforms.functional as F
                            frame_pil = F.to_pil_image(input_frame)
                            
                            # Update control image for each nested ControlNet - call on wrapper, not nested stream
                            for cn_idx in range(len(stream.stream.preprocessors)):
                                if stream.stream.preprocessors[cn_idx]:
                                    try:
                                        stream.update_control_image(index=cn_idx, image=frame_pil)
                                    except Exception as e:
                                        print(f"      Warning: Nested ControlNet {cn_idx} update failed: {e}")
                        
                        # Process frame through the stream - ControlNet preprocessing has been applied above
                        output_image = stream(image=input_frame)
                        
                        if output_image is None:
                            print(f"    Warning: Frame {j} returned None, skipping")
                            continue
                        
                        # Handle batch dimension if present
                        if output_image.dim() == 4:
                            segment_result[j] = output_image.squeeze(0).permute(1, 2, 0).clamp(0, 1)
                        elif output_image.dim() == 3:
                            segment_result[j] = output_image.permute(1, 2, 0).clamp(0, 1)
                        else:
                            print(f"    Warning: unexpected tensor dimensions: {output_image.shape}")
                            continue
                            
                    except Exception as e:
                        print(f"    Error processing frame {j}: {e}")
                        # Continue with next frame instead of failing completely
                        continue
                
                processing_time = time.time() - segment_start_time
                effective_fps = len(segment_frames) / processing_time
                fps_metrics.append(effective_fps)
                segment_times.append(processing_time)  # Store actual processing time
                print(f"  Processed {len(segment_frames)} frames in {processing_time:.2f}s ({effective_fps:.2f} FPS)")
                
                # Add segment frames to overall result for final merged video
                all_output_frames.append(segment_result)
                
                # Clean up segment processing memory
                del segment_result
                import gc
                gc.collect()
                
                # Clean up GPU memory after each segment
                cleanup_gpu_memory()
                log_memory_usage(f"after segment {i+1} completion")
                
            except Exception as e:
                print(f"  ERROR processing prompt {i+1}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next prompt instead of failing completely
                continue
        
        if not all_output_frames:
            raise RuntimeError("No segments were processed successfully")
        
        # Combine all segments into final merged video
        print("  Combining all prompt segments...")
        final_video = torch.cat(all_output_frames, dim=0)
        
        # Save final merged video with unique name per config
        config_name = config.get('model_id', 'unknown')
        # Clean up the config name to make it filesystem-safe
        if '/' in config_name:
            config_name = config_name.split('/')[-1]
        if '\\' in config_name:
            config_name = config_name.split('\\')[-1]
        # Remove file extensions
        config_name = config_name.replace('.safetensors', '').replace('.ckpt', '').replace('.pth', '')
        
        # Create unique filename for this config and video (merged from all prompts)
        # Include config filename, model_id, and video name for clear identification
        output_filename = f"{config_filename}_{video_filename}_{config_name}_merged_{len(prompts)}prompts.mp4"
        
        # Clean filename to ensure it's filesystem-safe
        import re
        output_filename = re.sub(r'[<>:"/\\|?*]', '_', output_filename)  # Replace invalid chars
        output_filename = output_filename[:200] + '.mp4' if len(output_filename) > 200 else output_filename  # Limit length
        
        output_video_path = os.path.join(output_dir, output_filename)
        
        # Ensure output directory exists before writing video
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Saving video to: {output_video_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  Directory exists: {os.path.exists(output_dir)}")
        
        # Convert to uint8 and save
        final_video_uint8 = (final_video * 255).clamp(0, 255).to(torch.uint8)
        
        try:
            write_video(output_video_path, final_video_uint8, fps=30)
            print(f"  ✅ Saved merged video: {output_video_path}")
        except Exception as video_error:
            print(f"  ❌ Failed to save video: {video_error}")
            print(f"     Output path: {output_video_path}")
            print(f"     Path length: {len(output_video_path)}")
            print(f"     Parent dir exists: {os.path.exists(os.path.dirname(output_video_path))}")
            print(f"     Video shape: {final_video_uint8.shape}")
            raise video_error
        finally:
            # CRITICAL: Clean up large video tensors immediately after saving
            print("  Cleaning up video tensors from system RAM...")
            try:
                del final_video_uint8
                del final_video
                del all_output_frames
                # Force immediate garbage collection
                import gc
                gc.collect()
                print("  ✅ Video tensors cleaned from system RAM")
            except Exception as cleanup_err:
                print(f"  ⚠️  Warning: Video tensor cleanup failed: {cleanup_err}")
        
        # Calculate overall FPS metrics CORRECTLY
        total_processing_time = sum(segment_times)  # Sum of actual processing times
        overall_fps = total_frames / total_processing_time if total_processing_time > 0 else 0
        min_fps = min(fps_metrics) if fps_metrics else 0
        max_fps = max(fps_metrics) if fps_metrics else 0
        avg_fps = sum(fps_metrics) / len(fps_metrics) if fps_metrics else 0
        
        # Calculate consistency metrics
        if len(fps_metrics) > 1:
            variance = sum((fps - avg_fps) ** 2 for fps in fps_metrics) / len(fps_metrics)
            std_dev_fps = variance ** 0.5
            cv_percent = (std_dev_fps / avg_fps) * 100 if avg_fps > 0 else 0
        else:
            std_dev_fps = 0
            cv_percent = 0
        
        print(f"  Overall Performance:")
        print(f"    Total processing time: {total_processing_time:.2f}s")
        print(f"    Overall FPS: {overall_fps:.2f}")
        print(f"    FPS range: {min_fps:.2f} - {max_fps:.2f}")
        print(f"    Average FPS: {avg_fps:.2f}")
        print(f"    Standard Deviation: {std_dev_fps:.2f}")
        print(f"    Coefficient of Variation: {cv_percent:.1f}%")
        
        # Create comprehensive metadata for JSON storage
        video_metadata = {
            'video_info': {
                'config_filename': config_filename,
                'video_filename': video_filename,
                'config_name': config_name,
                'output_filename': output_filename,
                'output_path': output_video_path,
                'total_frames': total_frames,
                'prompts_used': len(prompts),
                'prompts': prompts,
                'processing_date': datetime.datetime.now().isoformat(),
            },
            'config_details': {
                'model_id': config.get('model_id', 'Unknown'),
                'width': config.get('width', 'Unknown'),
                'height': config.get('height', 'Unknown'),
                'num_inference_steps': config.get('num_inference_steps', 'Unknown'),
                'guidance_scale': config.get('guidance_scale', 'Unknown'),
                'negative_prompt': config.get('negative_prompt', ''),
            },
            'performance_metrics': {
                'overall_fps': overall_fps,
                'min_fps': min_fps,
                'max_fps': max_fps,
                'avg_fps': avg_fps,
                'std_dev_fps': std_dev_fps,
                'cv_percent': cv_percent,
                'segment_fps': fps_metrics,
                'segment_times': segment_times,
                'total_processing_time': total_processing_time,
                'segments_processed': len(fps_metrics)
            },
            'technical_details': {
                'timeout_seconds': timeout_seconds,
                'start_time': start_time,
                'end_time': time.time(),
                'success': True
            }
        }
        
        # Save metadata as JSON file alongside video
        json_filename = output_filename.replace('.mp4', '_metadata.json')
        json_path = os.path.join(output_dir, json_filename)
        
        try:
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(video_metadata, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved metadata: {json_filename}")
        except Exception as json_error:
            print(f"  ⚠️  Warning: Failed to save metadata JSON: {json_error}")
        
        # Return result with FPS metrics and output file
        return {
            'output_file': output_filename,  # Just the filename, not full path
            'metadata_file': json_filename,  # JSON metadata filename
            'fps_metrics': {
                'overall_fps': overall_fps,
                'min_fps': min_fps,
                'max_fps': max_fps,
                'avg_fps': avg_fps,
                'std_dev_fps': std_dev_fps,
                'cv_percent': cv_percent,
                'segment_fps': fps_metrics,
                'segment_times': segment_times,  # Add segment times for debugging
                'total_processing_time': total_processing_time
            }
        }
        
    except TimeoutError as e:
        print(f"  TIMEOUT ERROR: {e}")
        return None
    except Exception as e:
        print(f"  ERROR processing: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Always cleanup, even if there was an error
        print("  Cleaning up pipeline...")
        try:
            if stream is not None:
                # Use the dedicated cleanup function
                cleanup_pipeline(stream)
                stream = None
                
        except Exception as cleanup_error:
            print(f"    Warning: Cleanup failed: {cleanup_error}")
        finally:
            # Force cleanup regardless of any errors
            cleanup_gpu_memory()
            print("    GPU memory cleanup completed")

def get_memory_info() -> Dict[str, float]:
    """Get current GPU and system memory information."""
    memory_info = {}
    
    # GPU memory
    if torch.cuda.is_available():
        memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / (1024**3)    # GB
        memory_info['gpu_free'] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024**3)  # GB
    
    # System memory
    try:
        import psutil
        memory_info['system_ram_used'] = psutil.virtual_memory().used / (1024**3)  # GB
        memory_info['system_ram_available'] = psutil.virtual_memory().available / (1024**3)  # GB
        memory_info['system_ram_percent'] = psutil.virtual_memory().percent
    except ImportError:
        # psutil not available, use basic info
        import os
        if hasattr(os, 'sysconf'):
            try:
                memory_info['system_ram_used'] = 'N/A (psutil not available)'
            except:
                pass
    
    return memory_info

def log_memory_usage(stage: str):
    """Log current memory usage for debugging."""
    memory_info = get_memory_info()
    if memory_info:
        gpu_info = f"GPU allocated: {memory_info['gpu_allocated']:.2f} GB, reserved: {memory_info['gpu_reserved']:.2f} GB, free: {memory_info['gpu_free']:.2f} GB"
        
        if 'system_ram_used' in memory_info and memory_info['system_ram_used'] != 'N/A (psutil not available)':
            ram_info = f"RAM used: {memory_info['system_ram_used']:.2f} GB, available: {memory_info['system_ram_available']:.2f} GB ({memory_info['system_ram_percent']:.1f}%)"
            print(f"  Memory usage at {stage}: {gpu_info}, {ram_info}")
        else:
            print(f"  Memory usage at {stage}: {gpu_info}")

def create_video_with_text(input_path: str, output_path: str, text: str, 
                          width: int, height: int, fontcolor: str = 'white') -> bool:
    """
    Create a scaled video with text overlay using ffmpeg-python
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        text: Text to overlay
        width: Target width
        height: Target height
        fontcolor: Color of the text
    
    Returns:
        True if successful, False otherwise
    """
    if ffmpeg is None:
        return False
        
    try:
        # Create the processing pipeline
        stream = ffmpeg.input(input_path)
        
        # Scale video to target size with padding
        scaled = ffmpeg.filter(
            stream, 
            'scale', 
            width, height,
            force_original_aspect_ratio='decrease'
        )
        
        padded = ffmpeg.filter(
            scaled,
            'pad',
            width, height,
            '(ow-iw)/2', '(oh-ih)/2'
        )
        
        # Add text overlay - specify font file to avoid fontconfig issues on Windows
        font_path = r'C:/Windows/Fonts/arial.ttf'
        if os.path.exists(font_path):
            with_text = ffmpeg.drawtext(
                padded,
                text=text,
                fontfile=font_path,
                fontcolor=fontcolor,
                fontsize=20,
                box=1,
                boxcolor='black@0.8',
                boxborderw=5,
                x=10,
                y='h-th-10'
            )
        else:
            # Fallback: try without fontfile (may use system default)
            with_text = ffmpeg.drawtext(
                padded,
                text=text,
                fontcolor=fontcolor,
                fontsize=20,
                box=1,
                boxcolor='black@0.8',
                boxborderw=5,
                x=10,
                y='h-th-10'
            )
        
        # Output with encoding settings
        output = ffmpeg.output(
            with_text,
            output_path,
            vcodec='libx264',
            crf=23,
            preset='medium'
        )
        
        # Run the pipeline with verbose error output
        ffmpeg.run(output, overwrite_output=True, quiet=True)
        return True
        
    except Exception as e:
        print(f"    Error processing video: {e}")
        return False

def create_placeholder_video(output_path: str, text: str, width: int, height: int, duration: float) -> bool:
    """
    Create a placeholder video with text using ffmpeg-python
    
    Args:
        output_path: Path to output video
        text: Text to display
        width: Video width
        height: Video height
        duration: Video duration in seconds
    
    Returns:
        True if successful, False otherwise
    """
    if ffmpeg is None:
        return False
        
    try:
        # Create gray color source
        color_source = ffmpeg.input(
            'color=c=gray:s={}x{}:d={}'.format(width, height, duration),
            f='lavfi'
        )
        
        # Add centered text - specify font file to avoid fontconfig issues on Windows
        font_path = r'C:/Windows/Fonts/arial.ttf'
        if os.path.exists(font_path):
            with_text = ffmpeg.drawtext(
                color_source,
                text=text,
                fontfile=font_path,
                fontcolor='white',
                fontsize=16,
                box=1,
                boxcolor='black@0.8',
                boxborderw=5,
                x='(w-text_w)/2',
                y='(h-text_h)/2'
            )
        else:
            # Fallback: try without fontfile (may use system default)
            with_text = ffmpeg.drawtext(
                color_source,
                text=text,
                fontcolor='white',
                fontsize=16,
                box=1,
                boxcolor='black@0.8',
                boxborderw=5,
                x='(w-text_w)/2',
                y='(h-text_h)/2'
            )
        
        # Output
        output = ffmpeg.output(
            with_text,
            output_path,
            vcodec='libx264',
            crf=23,
            preset='medium'
        )
        
        ffmpeg.run(output, overwrite_output=True, quiet=True)
        return True
        
    except Exception as e:
        print(f"    Error creating placeholder: {e}")
        return False

def create_video_wall(
    results: List[Dict], 
    video_files: List[Path], 
    config_files: List[Path], 
    output_dir: str
) -> Optional[str]:
    """
    Create a video wall showing original videos and processed results in a grid layout.
    
    Layout:
    - Top row: Original videos
    - Subsequent rows: Processed videos for each config
    
    Parameters
    ----------
    results : List[Dict]
        List of processing results
    video_files : List[Path]
        List of original video files
    config_files : List[Path]
        List of config files used
    output_dir : str
        Output directory for the video wall
        
    Returns
    -------
    Optional[str]
        Path to the created video wall, or None if failed
    """
    
    print("\nCreating video wall...")
    
    if ffmpeg is None:
        print("  Skipping video wall creation - ffmpeg-python not available")
        return None
    
    # Filter successful results only
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("  No successful results found for video wall")
        return None
    
    # Extract unique video and config names from successful results
    video_names = sorted(list(set([r['video'] for r in successful_results])))
    config_names = sorted(list(set([r['config'] for r in successful_results])))
    
    print(f"  Creating grid: {len(config_names)+1} rows x {len(video_names)} columns")
    print(f"  Videos: {video_names}")
    print(f"  Configs: {config_names}")
    
    # Create video wall output path
    wall_output = os.path.join(output_dir, "video_wall.mp4")
    
    # Create temporary directory for processing
    temp_dir = os.path.join(output_dir, "temp_video_wall")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Standard resolution for all videos in the wall
    wall_video_width = 512
    wall_video_height = 512
    
    try:
        # Step 1: Process all videos
        processed_videos = {}
        min_duration = float('inf')
        
        # Get minimum duration first
        print("  Getting video durations...")
        all_video_paths = []
        
        # Collect original video paths
        for video_name in video_names:
            for video_file in video_files:
                if video_file.stem == video_name:
                    all_video_paths.append(str(video_file))
                    break
        
        # Collect result video paths
        for result in successful_results:
            if 'output_file' in result and result['output_file']:
                output_file_path = os.path.join(output_dir, result['output_file'])
                if os.path.exists(output_file_path):
                    all_video_paths.append(output_file_path)
        
        # Get minimum duration
        for video_path in all_video_paths:
            try:
                probe = ffmpeg.probe(video_path)
                duration = float(probe['format']['duration'])
                min_duration = min(min_duration, duration)
            except Exception as e:
                print(f"    Warning: Could not get duration for {video_path}: {e}")
        
        if min_duration == float('inf') or min_duration < 1:
            min_duration = 10
        
        print(f"  Using duration: {min_duration:.2f} seconds")
        
        # Process original videos
        print("  Processing original videos...")
        for video_name in video_names:
            # Find the original video file
            original_video_path = None
            for video_file in video_files:
                if video_file.stem == video_name:
                    original_video_path = str(video_file)
                    break
            
            if not original_video_path:
                print(f"    Warning: Original video not found for {video_name}")
                continue
                
            scaled_path = os.path.join(temp_dir, f"scaled_original_{video_name.replace(' ', '_')}.mp4")
            text_content = f"ORIGINAL_{video_name.replace(' ', '_')}"
            
            success = create_video_with_text(
                original_video_path, 
                scaled_path, 
                text_content,
                wall_video_width, 
                wall_video_height,
                'white'
            )
            
            if success:
                processed_videos[('original', video_name)] = scaled_path
                print(f"    Processed original {video_name}")
            else:
                print(f"    Failed to process original {video_name}")
        
        # Process result videos with enhanced metadata
        print("  Processing result videos with enhanced metadata...")
        for result in successful_results:
            config_name = result['config']
            video_name = result['video']

            # Find the output file
            if 'output_file' not in result or not result['output_file']:
                print(f"    Warning: No output file for {config_name}_{video_name}")
                continue

            output_file_path = os.path.join(output_dir, result['output_file'])
            if not os.path.exists(output_file_path):
                print(f"    Warning: Output file not found: {output_file_path}")
                continue

            scaled_path = os.path.join(temp_dir, f"scaled_{config_name}_{video_name.replace(' ', '_')}.mp4")

            # Create enhanced metadata for this result
            metadata = {
                'video_info': {
                    'config_filename': config_name,
                    'output_filename': result['output_file'],
                    'total_frames': result.get('total_frames', 0)
                },
                'config_details': {
                    'model_id': result.get('model_id', 'Unknown'),
                    'width': result.get('resolution', 'Unknown').split('x')[0] if 'x' in str(result.get('resolution', '')) else 'Unknown',
                    'height': result.get('resolution', 'Unknown').split('x')[1] if 'x' in str(result.get('resolution', '')) else 'Unknown'
                },
                'performance_metrics': {
                    'overall_fps': result.get('fps_metrics', {}).get('overall_fps', 0),
                    'avg_fps': result.get('fps_metrics', {}).get('avg_fps', 0),
                    'total_processing_time': result.get('fps_metrics', {}).get('total_processing_time', 0)
                }
            }

            if create_enhanced_video_with_metadata:
                success = create_enhanced_video_with_metadata(
                    output_file_path,
                    scaled_path,
                    metadata,
                    wall_video_width,
                    wall_video_height
                )
            else:
                # Fallback to regular text overlay if enhanced function not available
                fps_metrics = result.get('fps_metrics', {})
                avg_fps = fps_metrics.get('avg_fps', 0)
                text_content = f"{config_name}_{avg_fps:.1f}_FPS"
                success = create_video_with_text(
                    output_file_path,
                    scaled_path,
                    text_content,
                    wall_video_width,
                    wall_video_height,
                    'yellow'
                )

            if success:
                processed_videos[(config_name, video_name)] = scaled_path
                fps_metrics = result.get('fps_metrics', {})
                avg_fps = fps_metrics.get('avg_fps', 0)
                print(f"    Processed {config_name}_{video_name} (FPS: {avg_fps:.1f})")
            else:
                print(f"    Failed to process {config_name}_{video_name}")
        
        # Step 2: Create the video wall grid with flipped layout
        print("  Assembling video wall with flipped layout...")
        print(f"  New layout: {len(video_names)} rows x {len(config_names) + 1} columns")
        print(f"  Each row = one original video + all its config outputs")
        print(f"  Each column = one config (including original)")

        # Collect all input streams for the grid
        input_streams = []

        # Build grid row by row - each row represents one original video
        for row_idx, video_name in enumerate(video_names):
            row_streams = []

            # For each column (config), add the corresponding video
            for col_idx, config_name in enumerate(['original'] + config_names):
                if (config_name, video_name) in processed_videos:
                    # Use existing processed video
                    stream = ffmpeg.input(processed_videos[(config_name, video_name)])
                    row_streams.append(stream)
                else:
                    # Create placeholder
                    placeholder_path = os.path.join(temp_dir, f"placeholder_{row_idx}_{col_idx}.mp4")
                    placeholder_text = f"MISSING_{config_name}_{video_name}".replace(' ', '_')

                    success = create_placeholder_video(
                        placeholder_path,
                        placeholder_text,
                        wall_video_width,
                        wall_video_height,
                        min_duration
                    )

                    if success:
                        stream = ffmpeg.input(placeholder_path)
                        row_streams.append(stream)
                    else:
                        print(f"    Failed to create placeholder for {config_name}_{video_name}")
                        return None

            # Horizontally stack this row (configs for one video)
            if len(row_streams) > 1:
                row_combined = ffmpeg.filter(row_streams, 'hstack', inputs=len(row_streams))
            else:
                row_combined = row_streams[0]

            input_streams.append(row_combined)

        # Vertically stack all rows (different videos)
        if len(input_streams) > 1:
            final_grid = ffmpeg.filter(input_streams, 'vstack', inputs=len(input_streams))
        else:
            final_grid = input_streams[0]
        
        # Trim to minimum duration and output
        trimmed = ffmpeg.filter(final_grid, 'trim', duration=min_duration)
        final_output = ffmpeg.output(
            trimmed,
            wall_output,
            vcodec='libx264',
            crf=20,
            preset='medium'
        )
        
        print("  Running ffmpeg to create final video wall...")
        ffmpeg.run(final_output, overwrite_output=True, quiet=True)
        
        print(f"  ✅ Video wall created: {wall_output}")
        
        # Clean up temporary files
        print("  Cleaning up temporary files...")
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"    Warning: Could not clean up temp directory: {e}")
        
        return wall_output
        
    except Exception as e:
        print(f"  Error creating video wall: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(
    configs: str,
    videos: str,
    output: str = "./output-test",
    prompts: Optional[str] = None,
    timeout_seconds: int = 300,  # 5 minutes timeout per video
    resume: Optional[str] = None,  # Resume from existing output directory
    retry_failed: bool = False  # Whether to retry previously failed combinations
):
    """
    Test multiple configs against multiple videos.
    
    Parameters
    ----------
    configs : str
        Directory containing YAML configuration files
    videos : str
        Directory containing video files
    output : str, optional
        Output directory for results, by default "./output-test"
    prompts : str, optional
        Text file containing individual prompts (one per line)
    timeout_seconds : int, optional
        Maximum time to spend processing each video, by default 300 (5 minutes)
    resume : str, optional
        Resume from existing output directory (full path to directory)
    retry_failed : bool, optional
        Whether to retry previously failed combinations, by default False
    """
    
    # Handle resume vs new run
    if resume:
        if not os.path.exists(resume):
            print(f"❌ Error: Resume directory does not exist: {resume}")
            return
        output_dir = resume
        print(f"🔄 Resuming from existing directory: {output_dir}")
    else:
        # Create timestamped output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"🆕 Starting new run in directory: {output_dir}")
    
    print("StreamDiffusion Multi-Config Test Suite")
    print("=" * 50)
    print(f"Configs directory: {configs}")
    print(f"Videos directory: {videos}")
    print(f"Output directory: {output_dir}")
    if prompts:
        print(f"Prompts file: {prompts}")
    if resume:
        print(f"Resume mode: ✅ Enabled")
        if retry_failed:
            print(f"Retry failed: ✅ Enabled (will retry previously failed combinations)")
        else:
            print(f"Retry failed: ❌ Disabled (will skip previously failed combinations)")
    print("=" * 50)
    
    # Load prompts if provided
    prompt_list = None
    if prompts:
        if not os.path.exists(prompts):
            print(f"Error: Prompts file not found: {prompts}")
            return
        prompt_list = load_prompts(prompts)
    
    # Scan for completed work if resuming
    existing_results = []
    if resume:
        existing_results = scan_completed_work(output_dir)
    
    # Get config files
    config_dir = Path(configs)
    config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    if not config_files:
        print(f"Error: No YAML config files found in {configs}")
        return
    
    # Get video files
    video_dir = Path(videos)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        print(f"Error: No video files found in {videos}")
        return
    
    print(f"\nFound {len(config_files)} configs and {len(video_files)} videos")
    
    # Calculate total work
    total_combinations = len(config_files) * len(video_files)
    
    print(f"\n📊 Work Summary:")
    print(f"  Total combinations: {total_combinations}")
    if resume and len(existing_results) > 0:
        print(f"  Previously completed: {len(existing_results)}")
    print(f"  Will check each combination for existing output files...")
    
    # Store results for performance summary (start with existing results)
    results = existing_results.copy()
    
    # Process each config against each video
    for config_path in config_files:
        print(f"\n{'='*60}")
        print(f"Processing config: {config_path.stem}")
        print(f"{'='*60}")
        
        # Aggressive cleanup before starting new config to ensure clean slate
        print(f"Pre-config cleanup for {config_path.stem}...")
        for cleanup_round in range(2):
            cleanup_gpu_memory()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            import time
            time.sleep(0.1)
        log_memory_usage(f"before config {config_path.stem}")
        
        try:
            config = load_config(config_path)
            print(f"Config loaded: {config.get('model_id', 'Unknown')}")
            print(f"Resolution: {config.get('width', 'Unknown')}x{config.get('height', 'Unknown')}")
        except Exception as e:
            print(f"Error loading config {config_path}: {e}")
            continue
        
        for video_path in video_files:
            print(f"\nProcessing video: {video_path.name}")
            
            # Check if output already exists (pass existing_results for resume functionality)
            if check_output_exists(output_dir, config_path.stem, video_path.stem, config, prompt_list, existing_results, retry_failed):
                print(f"  ⏭️  Skipping - already processed")
                continue
            
            try:
                print(f"  Starting video preprocessing...")
                # Preprocess video
                video = preprocess_video(
                    str(video_path),
                    config.get('width', 512),
                    config.get('height', 512)
                )
                print(f"  Video preprocessing completed, shape: {video.shape}")
                
                # Force CPU memory cleanup after video preprocessing
                import gc
                gc.collect()
                
                print(f"  Starting video processing with config...")
                # Process with config and get performance data
                result = process_video_with_config(
                    video=video,
                    config=config,
                    prompts=prompt_list,
                    output_dir=output_dir,
                    config_filename=config_path.stem,
                    video_filename=video_path.stem,
                    timeout_seconds=timeout_seconds
                )
                
                print(f"  Video processing completed, result: {'Success' if result else 'Failed'}")

                # Store video information before cleanup
                video_frames = video.shape[0]

                # Clean up after each video to prevent memory accumulation
                print(f"  Cleaning up after video {video_path.name}...")

                # Clean up video tensor from CPU memory
                del video
                import gc
                gc.collect()

                # Clean up GPU memory
                cleanup_gpu_memory()
                log_memory_usage(f"after video {video_path.name} completion")

                # If retrying, remove the old failed result to avoid duplicates
                if retry_failed:
                    results = [r for r in results if not (r['config'] == config_path.stem and r['video'] == video_path.stem)]

                # Store result for summary
                if result:
                    results.append({
                        'config': config_path.stem,
                        'video': video_path.stem,
                        'model_id': config.get('model_id', 'Unknown'),
                        'resolution': f"{config.get('width', 'Unknown')}x{config.get('height', 'Unknown')}",
                        'total_frames': video_frames,
                        'prompts_used': len(prompt_list) if prompt_list else 1,
                        'success': True,
                        'output_file': result['output_file'], # Store merged video file
                        'fps_metrics': result['fps_metrics']
                    })
                    print(f"  ✅ Successfully processed {video_path.name}")
                else:
                    results.append({
                        'config': config_path.stem,
                        'video': video_path.stem,
                        'model_id': config.get('model_id', 'Unknown'),
                        'resolution': f"{config.get('width', 'Unknown')}x{config.get('height', 'Unknown')}",
                        'total_frames': video_frames,
                        'prompts_used': len(prompt_list) if prompt_list else 1,
                        'success': False,
                        'error': 'Processing failed'
                    })
                    print(f"  ❌ Failed to process {video_path.name}")
                
            except Exception as e:
                print(f"  Failed to process {video_path.name}: {e}")
                import traceback
                traceback.print_exc()

                # Store video frames count if video was successfully loaded
                video_frames = video.shape[0] if 'video' in locals() else 0

                # Clean up video tensor even on failure
                try:
                    del video
                    import gc
                    gc.collect()
                    cleanup_gpu_memory()
                except:
                    pass  # video might not be defined if error occurred during preprocessing

                # If retrying, remove the old failed result to avoid duplicates
                if retry_failed:
                    results = [r for r in results if not (r['config'] == config_path.stem and r['video'] == video_path.stem)]
                
                results.append({
                    'config': config_path.stem,
                    'video': video_path.stem,
                    'model_id': config.get('model_id', 'Unknown'),
                    'resolution': f"{config.get('width', 'Unknown')}x{config.get('height', 'Unknown')}",
                    'total_frames': 0,
                    'prompts_used': len(prompt_list) if prompt_list else 1,
                    'success': False,
                    'error': str(e)
                })
                continue
        
        # Force cleanup between configs to ensure memory is cleared
        print(f"\nCleaning up after config {config_path.stem}...")
        try:
            # Multiple rounds of cleanup to ensure everything is freed
            for cleanup_round in range(3):  # Multiple cleanup rounds like in main.py
                cleanup_gpu_memory()
                
                # Additional cleanup to ensure no lingering references
                import gc
                gc.collect()
                
                # Force CUDA synchronization to ensure all operations are complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                # Small delay between cleanup rounds
                import time
                time.sleep(0.1)
            
            log_memory_usage(f"after config {config_path.stem} completion")
                
        except Exception as cleanup_error:
            print(f"  Warning: Config cleanup failed: {cleanup_error}")
        
        print(f"  Config {config_path.stem} cleanup completed")
        
        # Update progress tracking
        total_processed = len([r for r in results if r['success']])
        total_failed = len([r for r in results if not r['success']])
        print(f"  Progress: {total_processed + total_failed}/{total_combinations} total tests")
        print(f"    Successful: {total_processed}, Failed: {total_failed}")
    
    # Generate performance summary
    generate_performance_summary(results, output_dir, prompt_list)
    
    # Create video wall if we have successful results
    # Try enhanced video wall first (with JSON metadata), fallback to regular wall
    wall_path = None
    try:
        from enhanced_video_wall import create_enhanced_video_wall
        print("\n🎬 Attempting to create enhanced video wall with JSON metadata...")
        wall_path = create_enhanced_video_wall(output_dir, os.path.join(output_dir, "enhanced_video_wall.mp4"))
        if wall_path:
            print(f"✅ Enhanced video wall created: {wall_path}")
        else:
            print("⚠️  Enhanced video wall creation failed, falling back to regular wall")
    except ImportError:
        print("⚠️  Enhanced video wall module not available, using regular wall")
    except Exception as e:
        print(f"⚠️  Enhanced video wall creation failed: {e}, falling back to regular wall")
    
    # Fallback to regular video wall if enhanced version failed
    if not wall_path:
        wall_path = create_video_wall(results, video_files, config_files, output_dir)
    
    # Final summary
    total_successful = len([r for r in results if r['success']])
    total_failed = len([r for r in results if not r['success']])
    
    print(f"\n🎯 Final Summary:")
    print(f"  Total combinations: {total_combinations}")
    if resume:
        print(f"  Previously completed: {len(existing_results)}")
        print(f"  Newly processed: {total_successful + total_failed - len(existing_results)}")
    print(f"  Total successful: {total_successful}")
    print(f"  Total failed: {total_failed}")
    print(f"  Success rate: {(total_successful/total_combinations*100):.1f}%")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📊 Performance summary: {output_dir}/performance_summary.txt")
    print(f"📋 Detailed CSV: {output_dir}/detailed_results.csv")
    
    if wall_path:
        print(f"🎬 Video wall created: {wall_path}")
    else:
        print("🎬 Video wall creation skipped or failed")
    
    if resume:
        print(f"\n💡 To resume again later, use: --resume \"{output_dir}\"")

def generate_performance_summary(results: List[Dict], output_dir: str, prompts: Optional[List[str]] = None):
    """Generate a performance summary comparing all configs."""
    
    if not results:
        print("No results to summarize")
        return
    
    # Define successful_results early to avoid UnboundLocalError
    successful_results = [r for r in results if r['success']]
    
    summary_file = os.path.join(output_dir, "performance_summary.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("StreamDiffusion Multi-Config Performance Summary\n")
        f.write("=" * 60 + "\n\n")
        
        if prompts:
            f.write(f"Using {len(prompts)} individual prompts with temporal splitting\n\n")
        
        # Overall statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        failed_tests = total_tests - successful_tests
        
        f.write(f"Overall Results:\n")
        f.write(f"  Total tests: {total_tests}\n")
        f.write(f"  Successful: {successful_tests}\n")
        f.write(f"  Failed: {failed_tests}\n")
        f.write(f"  Success rate: {successful_tests/total_tests*100:.1f}%\n\n")
        
        # Quick Performance Summary Table
        if successful_results:
            f.write("Quick Performance Summary:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Config':<25} {'Video':<15} {'Resolution':<12} {'Overall FPS':<12} {'Avg FPS':<10} {'Min FPS':<10} {'Max FPS':<10} {'Frames':<8}\n")
            f.write("-" * 120 + "\n")
            
            for result in successful_results:
                fps = result['fps_metrics']
                f.write(f"{result['config']:<25} {result['video']:<15} {result['resolution']:<12} "
                       f"{fps['overall_fps']:<12.2f} {fps['avg_fps']:<10.2f} {fps['min_fps']:<10.2f} "
                       f"{fps['max_fps']:<10.2f} {result['total_frames']:<8}\n")
            f.write("-" * 120 + "\n\n")
        
        # Results by config
        configs = set(r['config'] for r in results)
        f.write("Results by Config:\n")
        f.write("-" * 40 + "\n")
        
        for config in sorted(configs):
            config_results = [r for r in results if r['config'] == config]
            config_success = sum(1 for r in config_results if r['success'])
            
            f.write(f"\n{config}:\n")
            f.write(f"  Tests: {len(config_results)}/{config_success} successful\n")
            f.write(f"  Model: {config_results[0]['model_id']}\n")
            f.write(f"  Resolution: {config_results[0]['resolution']}\n")
            
            # List videos processed
            for result in config_results:
                status = "✅" if result['success'] else "❌"
                f.write(f"    {status} {result['video']}")
                if result['success']:
                    f.write(f" ({result['total_frames']} frames, {result['prompts_used']} prompts)")
                    f.write(f" - Overall FPS: {result['fps_metrics']['overall_fps']:.2f}")
                    f.write(f", Min FPS: {result['fps_metrics']['min_fps']:.2f}")
                    f.write(f", Max FPS: {result['fps_metrics']['max_fps']:.2f}")
                    f.write(f", Avg FPS: {result['fps_metrics']['avg_fps']:.2f}")
                else:
                    f.write(f" - {result.get('error', 'Unknown error')}")
                f.write("\n")
        
        # Results by video
        f.write(f"\nResults by Video:\n")
        f.write("-" * 40 + "\n")
        
        videos = set(r['video'] for r in results)
        for video in sorted(videos):
            video_results = [r for r in results if r['video'] == video]
            video_success = sum(1 for r in video_results if r['success'])
            
            f.write(f"\n{video}:\n")
            f.write(f"  Tests: {len(video_results)}/{video_success} successful\n")
            
            for result in video_results:
                status = "✅" if result['success'] else "❌"
                f.write(f"    {status} {result['config']} ({result['resolution']})")
                if result['success']:
                    f.write(f" - {result['total_frames']} frames")
                    f.write(f" - Overall FPS: {result['fps_metrics']['overall_fps']:.2f}")
                    f.write(f", Min FPS: {result['fps_metrics']['min_fps']:.2f}")
                    f.write(f", Max FPS: {result['fps_metrics']['max_fps']:.2f}")
                    f.write(f", Avg FPS: {result['fps_metrics']['avg_fps']:.2f}")
                f.write("\n")
        
        # Summary of successful outputs
        if successful_results:
            f.write(f"\nGenerated Outputs:\n")
            f.write("-" * 40 + "\n")
            
            for result in successful_results:
                if 'output_file' in result and result['output_file']:
                    # Extract just the filename from the full path for display
                    output_filename = os.path.basename(result['output_file'])
                    f.write(f"✅ {output_filename}\n")
                else:
                    f.write(f"✅ {result['config']}_{result['video']}: No output files generated\n")
        
        # Performance Analysis and Rankings
        if successful_results:
            f.write(f"\nPerformance Analysis:\n")
            f.write("-" * 40 + "\n")
            
            # Overall FPS Rankings
            f.write(f"\nOverall FPS Rankings (Higher is Better):\n")
            fps_rankings = sorted(successful_results, key=lambda x: x['fps_metrics']['overall_fps'], reverse=True)
            for i, result in enumerate(fps_rankings):
                f.write(f"  {i+1:2d}. {result['config']:30s} - {result['fps_metrics']['overall_fps']:6.2f} FPS")
                f.write(f" (Avg: {result['fps_metrics']['avg_fps']:5.2f}, Range: {result['fps_metrics']['min_fps']:5.2f}-{result['fps_metrics']['max_fps']:5.2f})\n")
            
            # Average FPS Rankings
            f.write(f"\nAverage FPS Rankings (Higher is Better):\n")
            avg_fps_rankings = sorted(successful_results, key=lambda x: x['fps_metrics']['avg_fps'], reverse=True)
            for i, result in enumerate(avg_fps_rankings):
                f.write(f"  {i+1:2d}. {result['config']:30s} - {result['fps_metrics']['avg_fps']:6.2f} FPS")
                f.write(f" (Overall: {result['fps_metrics']['overall_fps']:5.2f}, Range: {result['fps_metrics']['min_fps']:5.2f}-{result['fps_metrics']['max_fps']:5.2f})\n")
            
            # Performance Statistics
            f.write(f"\nPerformance Statistics:\n")
            overall_fps_values = [r['fps_metrics']['overall_fps'] for r in successful_results]
            avg_fps_values = [r['fps_metrics']['avg_fps'] for r in successful_results]
            min_fps_values = [r['fps_metrics']['min_fps'] for r in successful_results]
            max_fps_values = [r['fps_metrics']['max_fps'] for r in successful_results]
            
            f.write(f"  Overall FPS - Best: {max(overall_fps_values):.2f}, Worst: {min(overall_fps_values):.2f}, Mean: {sum(overall_fps_values)/len(overall_fps_values):.2f}\n")
            f.write(f"  Average FPS - Best: {max(avg_fps_values):.2f}, Worst: {min(avg_fps_values):.2f}, Mean: {sum(avg_fps_values)/len(avg_fps_values):.2f}\n")
            f.write(f"  Min FPS - Best: {max(min_fps_values):.2f}, Worst: {min(min_fps_values):.2f}, Mean: {sum(min_fps_values)/len(min_fps_values):.2f}\n")
            f.write(f"  Max FPS - Best: {max(max_fps_values):.2f}, Worst: {min(max_fps_values):.2f}, Mean: {sum(max_fps_values)/len(max_fps_values):.2f}\n")
            
            # Performance by Resolution
            f.write(f"\nPerformance by Resolution:\n")
            resolutions = set(r['resolution'] for r in successful_results)
            for resolution in sorted(resolutions):
                res_results = [r for r in successful_results if r['resolution'] == resolution]
                res_overall_fps = [r['fps_metrics']['overall_fps'] for r in res_results]
                res_avg_fps = [r['fps_metrics']['avg_fps'] for r in res_results]
                
                f.write(f"  {resolution}:\n")
                f.write(f"    Configs tested: {len(res_results)}\n")
                f.write(f"    Best Overall FPS: {max(res_overall_fps):.2f} ({[r['config'] for r in res_results if r['fps_metrics']['overall_fps'] == max(res_overall_fps)][0]})\n")
                f.write(f"    Best Average FPS: {max(res_avg_fps):.2f} ({[r['config'] for r in res_results if r['fps_metrics']['avg_fps'] == max(res_avg_fps)][0]})\n")
                f.write(f"    Mean Overall FPS: {sum(res_overall_fps)/len(res_overall_fps):.2f}\n")
                f.write(f"    Mean Average FPS: {sum(res_avg_fps)/len(res_avg_fps):.2f}\n")
            
            # Performance by Video
            f.write(f"\nPerformance by Video:\n")
            videos = set(r['video'] for r in successful_results)
            for video in sorted(videos):
                vid_results = [r for r in successful_results if r['video'] == video]
                vid_overall_fps = [r['fps_metrics']['overall_fps'] for r in vid_results]
                vid_avg_fps = [r['fps_metrics']['avg_fps'] for r in vid_results]
                
                f.write(f"  {video}:\n")
                f.write(f"    Configs tested: {len(vid_results)}\n")
                f.write(f"    Best Overall FPS: {max(vid_overall_fps):.2f} ({[r['config'] for r in vid_results if r['fps_metrics']['overall_fps'] == max(vid_overall_fps)][0]})\n")
                f.write(f"    Best Average FPS: {max(vid_avg_fps):.2f} ({[r['config'] for r in vid_results if r['fps_metrics']['avg_fps'] == max(vid_avg_fps)][0]})\n")
                f.write(f"    Mean Overall FPS: {sum(vid_overall_fps)/len(vid_overall_fps):.2f}\n")
                f.write(f"    Mean Average FPS: {sum(vid_avg_fps)/len(vid_avg_fps):.2f}\n")
            
            # Best Config per Video Summary
            f.write(f"\nBest Config per Video (Overall FPS):\n")
            f.write("-" * 60 + "\n")
            for video in sorted(videos):
                vid_results = [r for r in successful_results if r['video'] == video]
                best_config = max(vid_results, key=lambda x: x['fps_metrics']['overall_fps'])
                fps = best_config['fps_metrics']
                f.write(f"  {video:<20} -> {best_config['config']:<25} ({fps['overall_fps']:6.2f} FPS, Avg: {fps['avg_fps']:5.2f})\n")
            
            f.write(f"\nBest Config per Video (Average FPS):\n")
            f.write("-" * 60 + "\n")
            for video in sorted(videos):
                vid_results = [r for r in successful_results if r['video'] == video]
                best_config = max(vid_results, key=lambda x: x['fps_metrics']['avg_fps'])
                fps = best_config['fps_metrics']
                f.write(f"  {video:<20} -> {best_config['config']:<25} ({fps['avg_fps']:6.2f} FPS, Overall: {fps['overall_fps']:5.2f})\n")
            
            # Performance Improvement Analysis
            f.write(f"\nPerformance Improvement Analysis:\n")
            f.write("-" * 60 + "\n")
            
            # Find the best overall config
            best_overall_config = max(successful_results, key=lambda x: x['fps_metrics']['overall_fps'])
            best_overall_fps = best_overall_config['fps_metrics']['overall_fps']
            
            f.write(f"Best Overall Config: {best_overall_config['config']} ({best_overall_fps:.2f} FPS)\n\n")
            f.write(f"Performance vs Best (Overall FPS):\n")
            
            for result in sorted(successful_results, key=lambda x: x['fps_metrics']['overall_fps'], reverse=True):
                if result['config'] != best_overall_config['config']:
                    improvement = ((best_overall_fps - result['fps_metrics']['overall_fps']) / result['fps_metrics']['overall_fps']) * 100
                    f.write(f"  {result['config']:<30s} - {result['fps_metrics']['overall_fps']:6.2f} FPS")
                    f.write(f" ({improvement:+.1f}% vs best)\n")
            
            # Performance vs Average
            avg_overall_fps = sum(r['fps_metrics']['overall_fps'] for r in successful_results) / len(successful_results)
            f.write(f"\nPerformance vs Average ({avg_overall_fps:.2f} FPS):\n")
            
            for result in sorted(successful_results, key=lambda x: x['fps_metrics']['overall_fps'], reverse=True):
                vs_avg = ((result['fps_metrics']['overall_fps'] - avg_overall_fps) / avg_overall_fps) * 100
                f.write(f"  {result['config']:<30s} - {result['fps_metrics']['overall_fps']:6.2f} FPS")
                f.write(f" ({vs_avg:+.1f}% vs avg)\n")
            
            # Performance Consistency Analysis
            f.write(f"\nPerformance Consistency Analysis:\n")
            f.write("-" * 60 + "\n")
            f.write("Configs ranked by FPS stability (lower variance = more stable):\n")
            
            # Calculate FPS variance for each config
            consistency_data = []
            for result in successful_results:
                segment_fps = result['fps_metrics']['segment_fps']
                if len(segment_fps) > 1:
                    mean_fps = sum(segment_fps) / len(segment_fps)
                    variance = sum((fps - mean_fps) ** 2 for fps in segment_fps) / len(segment_fps)
                    std_dev = variance ** 0.5
                    cv = (std_dev / mean_fps) * 100  # Coefficient of variation
                else:
                    variance = 0
                    std_dev = 0
                    cv = 0
                
                consistency_data.append({
                    'config': result['config'],
                    'mean_fps': result['fps_metrics']['avg_fps'],
                    'std_dev': std_dev,
                    'cv': cv,
                    'min_fps': result['fps_metrics']['min_fps'],
                    'max_fps': result['fps_metrics']['max_fps'],
                    'fps_range': result['fps_metrics']['max_fps'] - result['fps_metrics']['min_fps']
                })
            
            # Sort by coefficient of variation (lower = more stable)
            consistency_data.sort(key=lambda x: x['cv'])
            
            for i, data in enumerate(consistency_data):
                f.write(f"  {i+1:2d}. {data['config']:<30s} - CV: {data['cv']:5.1f}%")
                f.write(f" (Std: {data['std_dev']:5.2f}, Range: {data['fps_range']:5.2f})\n")
                f.write(f"      Mean: {data['mean_fps']:6.2f} FPS, Min: {data['min_fps']:5.2f}, Max: {data['max_fps']:5.2f}\n")
            
            # Recommendations
            f.write(f"\nRecommendations:\n")
            f.write("-" * 60 + "\n")
            
            # Best overall performance
            f.write(f"🏆 Best Overall Performance: {best_overall_config['config']}\n")
            f.write(f"   - Highest sustained FPS: {best_overall_config['fps_metrics']['overall_fps']:.2f}\n")
            f.write(f"   - Best for: Maximum throughput scenarios\n\n")
            
            # Most consistent performance
            most_consistent = consistency_data[0]
            f.write(f"📊 Most Consistent Performance: {most_consistent['config']}\n")
            f.write(f"   - Lowest variance: {most_consistent['cv']:.1f}% CV\n")
            f.write(f"   - Best for: Real-time applications requiring stable frame rates\n\n")
            
            # Best value (good performance + consistency)
            # Find config with good balance of performance and consistency
            balanced_configs = []
            for data in consistency_data:
                # Normalize both metrics (0-1 scale)
                perf_score = data['mean_fps'] / max(d['mean_fps'] for d in consistency_data)
                consistency_score = 1 - (data['cv'] / max(d['cv'] for d in consistency_data))
                balanced_score = (perf_score + consistency_score) / 2
                balanced_configs.append((data['config'], balanced_score, data['mean_fps'], data['cv']))
            
            balanced_configs.sort(key=lambda x: x[1], reverse=True)
            best_balanced = balanced_configs[0]
            f.write(f"⚖️  Best Balanced (Performance + Consistency): {best_balanced[0]}\n")
            f.write(f"   - Balanced score: {best_balanced[1]:.3f}\n")
            f.write(f"   - Performance: {best_balanced[2]:.2f} FPS, Consistency: {best_balanced[3]:.1f}% CV\n")
            f.write(f"   - Best for: Production environments requiring both speed and reliability\n\n")
            
            # Performance tiers
            f.write(f"📈 Performance Tiers:\n")
            fps_values = [r['fps_metrics']['overall_fps'] for r in successful_results]
            fps_values.sort(reverse=True)
            
            if len(fps_values) >= 3:
                top_tier = fps_values[:len(fps_values)//3]
                mid_tier = fps_values[len(fps_values)//3:2*len(fps_values)//3]
                bottom_tier = fps_values[2*len(fps_values)//3:]
                
                f.write(f"   🥇 Top Tier (≥{min(top_tier):.2f} FPS): {len(top_tier)} configs\n")
                f.write(f"   🥈 Mid Tier ({min(mid_tier):.2f}-{max(mid_tier):.2f} FPS): {len(mid_tier)} configs\n")
                f.write(f"   🥉 Bottom Tier (<{max(bottom_tier):.2f} FPS): {len(bottom_tier)} configs\n")
            
            f.write(f"\n💡 Usage Tips:\n")
            f.write(f"   - For maximum speed: Use {best_overall_config['config']}\n")
            f.write(f"   - For stable real-time: Use {most_consistent['config']}\n")
            f.write(f"   - For production: Use {best_balanced[0]}\n")
            f.write(f"   - Consider resolution impact: Higher resolutions generally reduce FPS\n")
            f.write(f"   - Monitor VRAM usage: Some configs may be more memory-efficient\n")
            
            # Best Configs by Use Case
            f.write(f"\n🎯 Best Configs by Use Case:\n")
            f.write("-" * 60 + "\n")
            
            # Speed-focused use cases
            f.write(f"🚀 Speed-Focused Use Cases:\n")
            speed_configs = sorted(successful_results, key=lambda x: x['fps_metrics']['overall_fps'], reverse=True)[:3]
            for i, result in enumerate(speed_configs):
                fps = result['fps_metrics']
                f.write(f"   {i+1}. {result['config']:<25} - {fps['overall_fps']:6.2f} FPS")
                f.write(f" (Avg: {fps['avg_fps']:5.2f}, CV: {fps['cv_percent']:4.1f}%)\n")
            
            # Consistency-focused use cases
            f.write(f"\n📊 Consistency-Focused Use Cases:\n")
            consistency_configs = sorted(successful_results, key=lambda x: x['fps_metrics']['cv_percent'])[:3]
            for i, result in enumerate(consistency_configs):
                fps = result['fps_metrics']
                f.write(f"   {i+1}. {result['config']:<25} - CV: {fps['cv_percent']:4.1f}%")
                f.write(f" (Avg: {fps['avg_fps']:5.2f} FPS, Overall: {fps['overall_fps']:5.2f})\n")
            
            # Balanced use cases
            f.write(f"\n⚖️  Balanced Use Cases (Speed + Consistency):\n")
            for i, (config, score, mean_fps, cv) in enumerate(balanced_configs[:3]):
                f.write(f"   {i+1}. {config:<25} - Score: {score:.3f}")
                f.write(f" (Avg: {mean_fps:5.2f} FPS, CV: {cv:4.1f}%)\n")
            
            # Resolution-specific recommendations
            f.write(f"\n🖼️  Resolution-Specific Recommendations:\n")
            for resolution in sorted(resolutions):
                res_results = [r for r in successful_results if r['resolution'] == resolution]
                best_speed = max(res_results, key=lambda x: x['fps_metrics']['overall_fps'])
                best_consistency = min(res_results, key=lambda x: x['fps_metrics']['cv_percent'])
                
                f.write(f"   {resolution}:\n")
                f.write(f"     - Best Speed: {best_speed['config']} ({best_speed['fps_metrics']['overall_fps']:.2f} FPS)\n")
                f.write(f"     - Best Consistency: {best_consistency['config']} (CV: {best_consistency['fps_metrics']['cv_percent']:.1f}%)\n")

    print(f"Performance summary saved to: {summary_file}")
    
    # Also save as CSV for easy analysis
    csv_file = os.path.join(output_dir, "detailed_results.csv")
    import csv
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Config', 'Video', 'Model ID', 'Resolution', 'Total Frames', 
            'Prompts Used', 'Success', 'Output File', 'Error Message',
            'Overall FPS', 'Min FPS', 'Max FPS', 'Avg FPS', 'Std Dev FPS', 'CV %'
        ])
        
        # Data rows
        for result in results:
            if result['success']:
                fps_metrics = result['fps_metrics']
                # Format output file path - just the filename for clarity
                output_file_str = os.path.basename(result.get('output_file', ''))
                
                writer.writerow([
                    result['config'],
                    result['video'],
                    result['model_id'],
                    result['resolution'],
                    result['total_frames'],
                    result['prompts_used'],
                    "Yes" if result['success'] else "No",
                    output_file_str,
                    result.get('error', ''),
                    f"{fps_metrics['overall_fps']:.2f}",
                    f"{fps_metrics['min_fps']:.2f}",
                    f"{fps_metrics['max_fps']:.2f}",
                    f"{fps_metrics['avg_fps']:.2f}",
                    f"{fps_metrics['std_dev_fps']:.2f}",
                    f"{fps_metrics['cv_percent']:.1f}"
                ])
            else:
                writer.writerow([
                    result['config'],
                    result['video'],
                    result['model_id'],
                    result['resolution'],
                    result['total_frames'],
                    result['prompts_used'],
                    "Yes" if result['success'] else "No",
                    "",
                    result.get('error', ''),
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A"
                ])
    
    print(f"Detailed results saved to: {csv_file}")

if __name__ == "__main__":
    try:
        fire.Fire(main)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        cleanup_and_exit()
    except Exception as e:
        print(f"\nUnexpected error in main: {e}")
        import traceback
        traceback.print_exc()
        cleanup_and_exit()
        sys.exit(1)
    finally:
        cleanup_and_exit()
