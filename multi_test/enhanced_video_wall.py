#!/usr/bin/env python3
"""
Enhanced Video Wall Creator using JSON metadata

This module creates video walls with rich metadata information from JSON files
stored alongside each processed video, providing better data for resume functionality.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import ffmpeg
except ImportError:
    print("Warning: ffmpeg-python library not found. Enhanced video wall creation will be disabled.")
    ffmpeg = None

def load_video_metadata(results_dir: str) -> Dict[str, Dict]:
    """
    Load all JSON metadata files from results directory.
    
    Parameters
    ----------
    results_dir : str
        Directory containing video results and JSON metadata
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping video filenames to their metadata
    """
    metadata_dict = {}
    
    if not os.path.exists(results_dir):
        return metadata_dict
    
    print(f"📋 Loading video metadata from: {results_dir}")
    
    try:
        json_files = [f for f in os.listdir(results_dir) if f.endswith('_metadata.json')]
        
        for json_file in json_files:
            json_path = os.path.join(results_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # Extract video filename from metadata
                video_info = metadata.get('video_info', {})
                output_filename = video_info.get('output_filename', '')
                
                if output_filename:
                    metadata_dict[output_filename] = metadata
                    
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load {json_file}: {e}")
        
        print(f"  ✅ Loaded metadata for {len(metadata_dict)} videos")
        
    except Exception as e:
        print(f"  ❌ Error loading metadata: {e}")
    
    return metadata_dict

def create_enhanced_video_with_metadata(
    input_path: str, 
    output_path: str, 
    metadata: Dict,
    width: int, 
    height: int
) -> bool:
    """
    Create a scaled video with enhanced metadata overlay using ffmpeg-python
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        metadata: Video metadata dictionary
        width: Target width
        height: Target height
    
    Returns:
        True if successful, False otherwise
    """
    if ffmpeg is None:
        return False
        
    try:
        # Extract key information from metadata
        video_info = metadata.get('video_info', {})
        config_details = metadata.get('config_details', {})
        performance = metadata.get('performance_metrics', {})
        
        config_name = video_info.get('config_filename', 'Unknown')
        model_name = config_details.get('model_id', 'Unknown').split('/')[-1]
        resolution = f"{config_details.get('width', '?')}x{config_details.get('height', '?')}"
        overall_fps = performance.get('overall_fps', 0)
        avg_fps = performance.get('avg_fps', 0)
        total_frames = video_info.get('total_frames', 0)
        processing_time = performance.get('total_processing_time', 0)
        
        # Create multi-line text overlay with rich information
        text_lines = [
            f"Config: {config_name}",
            f"Model: {model_name}",
            f"Resolution: {resolution}",
            f"Frames: {total_frames}",
            f"Overall FPS: {overall_fps:.1f}",
            f"Avg FPS: {avg_fps:.1f}",
            f"Time: {processing_time:.1f}s"
        ]
        
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
        
        # Add multi-line text overlay
        font_path = r'C:/Windows/Fonts/arial.ttf'
        
        # Start with the padded video
        current_stream = padded
        
        # Add each line of text
        for i, line in enumerate(text_lines):
            y_position = 10 + (i * 25)  # 25 pixels between lines
            
            if os.path.exists(font_path):
                current_stream = ffmpeg.drawtext(
                    current_stream,
                    text=line,
                    fontfile=font_path,
                    fontcolor='white',
                    fontsize=16,
                    box=1,
                    boxcolor='black@0.8',
                    boxborderw=3,
                    x=10,
                    y=y_position
                )
            else:
                # Fallback without fontfile
                current_stream = ffmpeg.drawtext(
                    current_stream,
                    text=line,
                    fontcolor='white',
                    fontsize=16,
                    box=1,
                    boxcolor='black@0.8',
                    boxborderw=3,
                    x=10,
                    y=y_position
                )
        
        # Output with encoding settings
        output = ffmpeg.output(
            current_stream,
            output_path,
            vcodec='libx264',
            crf=23,
            preset='medium'
        )
        
        # Run the pipeline
        ffmpeg.run(output, overwrite_output=True, quiet=True)
        return True
        
    except Exception as e:
        print(f"    Error processing enhanced video: {e}")
        return False

def create_enhanced_video_wall(
    results_dir: str,
    output_path: str,
    grid_width: int = 3,  # Kept for backward compatibility, but not used in new layout
    video_width: int = 512,
    video_height: int = 512
) -> Optional[str]:
    """
    Create an enhanced video wall using JSON metadata for rich information display.

    The layout is automatically determined: each row represents one original video,
    and columns represent different configurations (including original).

    Parameters
    ----------
    results_dir : str
        Directory containing video results and JSON metadata
    output_path : str
        Path for the output video wall
    grid_width : int, optional
        Legacy parameter kept for backward compatibility (not used in new layout)
    video_width : int, optional
        Width of each video in the wall, by default 512
    video_height : int, optional
        Height of each video in the wall, by default 512

    Returns
    -------
    Optional[str]
        Path to the created video wall, or None if failed
    """
    
    print("\n🎬 Creating enhanced video wall with JSON metadata...")
    
    if ffmpeg is None:
        print("  ❌ Skipping - ffmpeg-python not available")
        return None
    
    # Load metadata for all videos
    metadata_dict = load_video_metadata(results_dir)
    
    if not metadata_dict:
        print("  ❌ No video metadata found")
        return None
    
    # Find corresponding video files
    video_files = []
    enhanced_metadata = []
    
    for filename, metadata in metadata_dict.items():
        video_path = os.path.join(results_dir, filename)
        if os.path.exists(video_path):
            video_files.append(video_path)
            enhanced_metadata.append(metadata)
        else:
            print(f"  ⚠️  Warning: Video file not found: {filename}")
    
    if not video_files:
        print("  ❌ No video files found")
        return None
    
    print(f"  📹 Processing {len(video_files)} videos for enhanced wall")
    
    # Create temporary directory for processing
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_enhanced_wall")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Process each video with enhanced metadata overlay
        processed_videos = []
        min_duration = float('inf')
        
        print("  🔄 Processing videos with enhanced metadata...")
        for i, (video_path, metadata) in enumerate(zip(video_files, enhanced_metadata)):
            print(f"    Processing {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
            
            # Get video duration
            try:
                probe = ffmpeg.probe(video_path)
                duration = float(probe['format']['duration'])
                min_duration = min(min_duration, duration)
            except Exception as e:
                print(f"      Warning: Could not get duration: {e}")
                duration = 10
                min_duration = min(min_duration, duration)
            
            # Create enhanced video with metadata overlay
            enhanced_path = os.path.join(temp_dir, f"enhanced_{i:03d}.mp4")
            
            success = create_enhanced_video_with_metadata(
                video_path,
                enhanced_path,
                metadata,
                video_width,
                video_height
            )
            
            if success:
                processed_videos.append(enhanced_path)
                print(f"      ✅ Enhanced video created")
            else:
                print(f"      ❌ Failed to enhance video")
        
        if not processed_videos:
            print("  ❌ No videos were successfully processed")
            return None
        
        if min_duration == float('inf'):
            min_duration = 10

        print(f"  🎬 Creating video wall with flipped layout...")
        
        # Create video wall grid
        input_streams = []
        
        # Load all processed videos as input streams
        for video_path in processed_videos:
            stream = ffmpeg.input(video_path)
            input_streams.append(stream)
        
        # Pad with blank videos if needed to fill grid
        total_videos = len(input_streams)
        videos_per_row = grid_width
        rows_needed = (total_videos + videos_per_row - 1) // videos_per_row
        total_slots = rows_needed * videos_per_row
        
        # Create blank videos for empty slots
        for i in range(total_videos, total_slots):
            blank_path = os.path.join(temp_dir, f"blank_{i}.mp4")
            
            # Create a blank video
            blank_input = ffmpeg.input(
                f'color=c=gray:s={video_width}x{video_height}:d={min_duration}',
                f='lavfi'
            )
            
            blank_with_text = ffmpeg.drawtext(
                blank_input,
                text='No Video',
                fontcolor='white',
                fontsize=24,
                x='(w-text_w)/2',
                y='(h-text_h)/2'
            )
            
            blank_output = ffmpeg.output(blank_with_text, blank_path, vcodec='libx264', crf=23)
            ffmpeg.run(blank_output, overwrite_output=True, quiet=True)
            
            input_streams.append(ffmpeg.input(blank_path))
        
        # Create rows with flipped layout: each row = one original video + all its config outputs
        # This assumes videos are ordered as: original_video1, config1_video1, config2_video1, ..., original_video2, config1_video2, etc.

        # First, determine how many configs (including original) we have
        # We need to figure this out from the metadata
        if processed_videos:
            # Get unique config names from metadata
            config_names = set()
            for metadata in enhanced_metadata:
                config_names.add(metadata.get('video_info', {}).get('config_filename', 'Unknown'))
            config_names = ['original'] + sorted(list(config_names))

            # Get unique video names
            video_names = set()
            for metadata in enhanced_metadata:
                # Extract original video name from the output filename
                output_filename = metadata.get('video_info', {}).get('output_filename', '')
                if output_filename:
                    # Try to extract video name - this depends on naming convention
                    # Assuming format like: configName_videoName_merged_5prompts.mp4
                    parts = output_filename.replace('.mp4', '').split('_')
                    if len(parts) >= 2:
                        video_name = parts[1]  # Second part should be video name
                        video_names.add(video_name)
            video_names = sorted(list(video_names))

            print(f"  Detected {len(video_names)} videos and {len(config_names)} configs (including original)")
            print(f"  Flipped layout: {len(video_names)} rows x {len(config_names)} columns")

            # Reorder streams for flipped layout
            reordered_streams = []
            for video_name in video_names:
                for config_name in config_names:
                    # Find the stream for this video+config combination
                    found = False
                    for i, metadata in enumerate(enhanced_metadata):
                        video_info = metadata.get('video_info', {})
                        if (video_info.get('config_filename') == config_name or
                            (config_name == 'original' and 'original' in video_info.get('output_filename', ''))):
                            # Check if this is the right video
                            output_filename = video_info.get('output_filename', '')
                            if video_name in output_filename:
                                reordered_streams.append(input_streams[i])
                                found = True
                                break

                    if not found:
                        # Create placeholder for missing combination
                        placeholder_path = os.path.join(temp_dir, f"placeholder_{video_name}_{config_name}.mp4")
                        placeholder_text = f"MISSING_{config_name}_{video_name}".replace(' ', '_')

                        try:
                            blank_input = ffmpeg.input(
                                f'color=c=gray:s={video_width}x{video_height}:d={min_duration}',
                                f='lavfi'
                            )

                            blank_with_text = ffmpeg.drawtext(
                                blank_input,
                                text=placeholder_text,
                                fontcolor='white',
                                fontsize=24,
                                x='(w-text_w)/2',
                                y='(h-text_h)/2'
                            )

                            blank_output = ffmpeg.output(blank_with_text, placeholder_path, vcodec='libx264', crf=23)
                            ffmpeg.run(blank_output, overwrite_output=True, quiet=True)

                            reordered_streams.append(ffmpeg.input(placeholder_path))
                        except Exception as e:
                            print(f"    Failed to create placeholder: {e}")
                            return None

            # Create rows from reordered streams
            rows = []
            for row_idx in range(len(video_names)):
                start_idx = row_idx * len(config_names)
                end_idx = start_idx + len(config_names)
                row_streams = reordered_streams[start_idx:end_idx]

                if len(row_streams) > 1:
                    row_combined = ffmpeg.filter(row_streams, 'hstack', inputs=len(row_streams))
                else:
                    row_combined = row_streams[0]

                rows.append(row_combined)

            # Combine rows vertically
            if len(rows) > 1:
                final_grid = ffmpeg.filter(rows, 'vstack', inputs=len(rows))
            else:
                final_grid = rows[0]
        else:
            # Fallback to original logic if metadata parsing fails
            print("  Warning: Could not parse metadata for flipped layout, using original grid")
            rows = []
            for row_idx in range(rows_needed):
                start_idx = row_idx * videos_per_row
                end_idx = min(start_idx + videos_per_row, len(input_streams))
                row_streams = input_streams[start_idx:end_idx]

                if len(row_streams) > 1:
                    row_combined = ffmpeg.filter(row_streams, 'hstack', inputs=len(row_streams))
                else:
                    row_combined = row_streams[0]

                rows.append(row_combined)

            # Combine rows vertically
            if len(rows) > 1:
                final_grid = ffmpeg.filter(rows, 'vstack', inputs=len(rows))
            else:
                final_grid = rows[0]
        
        # Trim to minimum duration and output
        trimmed = ffmpeg.filter(final_grid, 'trim', duration=min_duration)
        final_output = ffmpeg.output(
            trimmed,
            output_path,
            vcodec='libx264',
            crf=20,
            preset='medium'
        )
        
        print("  🎬 Rendering final enhanced video wall...")
        ffmpeg.run(final_output, overwrite_output=True, quiet=True)
        
        print(f"  ✅ Enhanced video wall created: {output_path}")
        
        # Clean up temporary files
        print("  🧹 Cleaning up temporary files...")
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"    Warning: Could not clean up temp directory: {e}")
        
        return output_path
        
    except Exception as e:
        print(f"  ❌ Error creating enhanced video wall: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Example usage of enhanced video wall creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create enhanced video wall with JSON metadata")
    parser.add_argument("--results_dir", required=True, help="Directory containing video results and JSON metadata")
    parser.add_argument("--output", required=True, help="Output path for video wall")
    parser.add_argument("--grid_width", type=int, default=3, help="Number of videos per row")
    parser.add_argument("--video_width", type=int, default=512, help="Width of each video")
    parser.add_argument("--video_height", type=int, default=512, help="Height of each video")
    
    args = parser.parse_args()
    
    result = create_enhanced_video_wall(
        args.results_dir,
        args.output,
        args.grid_width,
        args.video_width,
        args.video_height
    )
    
    if result:
        print(f"\n✅ Enhanced video wall created successfully: {result}")
        return 0
    else:
        print(f"\n❌ Failed to create enhanced video wall")
        return 1

if __name__ == "__main__":
    exit(main())



