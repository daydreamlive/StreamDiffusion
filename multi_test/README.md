# StreamDiffusion Multi-Config Test Suite

This testing suite allows you to benchmark multiple StreamDiffusion configurations against multiple video files, providing comprehensive performance analysis and reports.

## Features

- **Multi-Config Testing**: Test multiple YAML configuration files against multiple video files
- **Resume Functionality**: Continue processing from where you left off after interruptions
- **Individual Prompt Processing**: Process each prompt individually and merge results into combined videos
- **RAM-Based Frame Processing**: Loads all frames into RAM for maximum speed (similar to main.py)
- **RAM-Based Video Creation**: Creates MP4 videos directly from frames in memory (no disk I/O)
- **Automatic Frame Extraction**: Uses ffmpeg to extract frames from videos for processing
- **Framerate Matching**: Output videos maintain the same framerate and timing as input videos
- **Full ControlNet Support**: Automatically loads and configures ControlNets from YAML configs
- **Performance Metrics**: Measures FPS, frame processing times, and success rates
- **Comprehensive Reporting**: Generates multiple report formats (TXT, CSV, JSON)
- **Error Handling**: Gracefully handles failures and continues with remaining tests
- **Resource Management**: Automatically cleans up temporary files and RAM cache
- **Video Merging**: Combines output from multiple prompts into single merged videos
- **JSON Metadata Generation**: Creates detailed JSON files alongside output videos containing configuration details, performance metrics, and processing information for comprehensive analysis and resume support
- **Enhanced Video Wall**: Automatically generates a video wall with rich metadata overlays (config, model, FPS, etc.) using ffmpeg, supporting flipped layouts (videos as rows, configs as columns) with fallback to basic wall
- **Retry Failed Combinations**: Supports retrying previously failed config+video pairs during resume without reprocessing successful ones, preserving all prior results
- **Advanced Performance Metrics**: Includes coefficient of variation (CV) for FPS stability, segment-level FPS analysis, and detailed rankings/recommendations in reports for better optimization insights

## Performance Optimizations

The test suite is optimized for maximum speed by:

1. **RAM-Based Processing**: All video frames are loaded into RAM once and reused across multiple prompts/configs
2. **RAM-Based Video Creation**: Creates MP4 videos directly from frames in memory (no temporary files)
3. **Frame Caching**: Frames are cached in memory to avoid reloading from disk
4. **Minimal Disk I/O**: Processing happens entirely in memory, with disk writes only for final output
5. **Efficient Memory Management**: Automatic cleanup of frame cache to prevent memory issues
6. **Batch Processing**: Multiple prompts are processed against the same frames without reloading
7. **Framerate Preservation**: Output videos maintain input video timing for seamless playback
8. **Dual Video Encoding**: Uses imageio (primary) + OpenCV (fallback) for maximum compatibility

This approach makes the test suite run significantly faster than disk-based alternatives, similar to the real-time processing in `main.py`.

## Requirements

- Python 3.7+
- StreamDiffusion installed
- ffmpeg-python (for enhanced video wall creation with metadata overlays; install with `pip install ffmpeg-python`)
- ffmpeg (for video frame extraction only)
- PyYAML
- PIL/Pillow
- **Video Creation Dependencies**:
  - `imageio` (primary video creation)
  - `imageio-ffmpeg` (for H.264 encoding)
  - `opencv-python` (fallback video creation)
- Sufficient RAM to hold all video frames (typically 2-4GB per video depending on resolution)

## Installation

1. Install the core dependencies:
```bash
pip install pyyaml pillow
```

2. Install ffmpeg (for frame extraction only):
   - **Windows**: Download from https://ffmpeg.org/download.html
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` or equivalent

3. Make sure StreamDiffusion is properly installed and accessible

## Usage

### Basic Usage

```bash
# Test with config prompts (original behavior)
python multi_test.py --configs ./myconfigdir --videos ./myinputvideos

# Test with individual prompts from file
python multi_test.py --configs ./myconfigdir --videos ./myinputvideos --prompts ./my_prompts.txt
```

### Command Line Options

- `--configs`: Directory containing YAML configuration files
- `--videos`: Directory containing video files
- `--output`: Output directory for results (default: `./output-test`)
- `--prompts`: Text file containing individual prompts (one per line, optional)
- `--timeout_seconds`: Maximum time to spend processing each video (default: 300)
- `--resume`: Resume from existing output directory (full path to directory)
- `--retry_failed`: Retry previously failed combinations during resume (default: false)

### Memory Management Options

For videos that cause CUDA out-of-memory errors, use these options:

```bash
# Process fewer frames per batch to reduce memory usage
python multi_test.py --configs ./configs --videos ./videos --batch-size 5

# Lower memory threshold for more aggressive cleanup
python multi_test.py --configs ./configs --videos ./videos --memory-threshold 1.0

# Process every 2nd frame for very long videos (reduces processing time and memory)
python multi_test.py --configs ./configs --videos ./videos --frame-skip 2

# Combine all memory management options
python multi_test.py --configs ./configs --videos ./videos --batch-size 5 --memory-threshold 1.0 --frame-skip 2
```

### Example

```bash
# Test all configs in ./configs against all videos in ./videos
python multi_test.py --configs ./configs --videos ./videos --output ./benchmark_results

# Test with individual prompts
python multi_test.py --configs ./configs --videos ./videos --prompts ./prompts.txt --output ./prompt_results

# Test with custom output directory
python multi_test.py --configs ./my_configs --videos ./my_videos --output ./my_results
```

## Resume Functionality

The test suite now supports resuming interrupted runs, allowing you to continue processing from where you left off without losing previous work.

### How Resume Works

1. **Automatic Detection**: Scans existing output directory for completed videos and JSON metadata files
2. **Smart Parsing**: Extracts config+video combinations from existing filenames and JSON metadata
3. **CSV Integration**: Loads existing results from CSV files, including both successful and failed tests
4. **JSON Enrichment**: Loads detailed performance data from JSON metadata files for enhanced analysis
5. **Skip Completed**: Only processes remaining config+video combinations, with option to retry failed ones
6. **Seamless Integration**: Updates existing reports and maintains all output files

### Resume Usage

```bash
# Start a new test run
python multi_test.py --configs ./configs --videos ./videos --output ./output-multi

# Resume from existing directory (if interrupted)
python multi_test.py --configs ./configs --videos ./videos --resume "C:\sd\StreamDiffusion\multi_test\20250903_192109"

# Resume with different prompts (will use same output directory)
python multi_test.py --configs ./configs --videos ./videos --prompts ./prompts.txt --resume "./output-multi/20250903_192109"

# Resume with different timeout for remaining work
python multi_test.py --configs ./configs --videos ./videos --timeout_seconds 600 --resume "./output-multi/20250903_192109"
```

### Resume Benefits

- **Time Saving**: No need to reprocess completed combinations
- **Memory Efficient**: Continues with existing memory management
- **Progress Preservation**: Maintains all existing output files and reports
- **Flexible**: Can change prompts or timeout for remaining work
- **Robust**: Handles various filename formats and edge cases

### Resume Output

When resuming, the test suite will show:

```
🔄 Resuming from existing directory: C:\sd\StreamDiffusion\multi_test\20250903_192109

🔍 Scanning for completed work in: C:\sd\StreamDiffusion\multi_test\20250903_192109
📊 Loading existing results from CSV: detailed_results.csv
✅ Loaded 8 successful results from CSV
🎬 Scanning video files in directory...
  📹 Analyzing: sdxl_depth_trt_ta86_cn_lcm_20250903-2317-28.5538735_ta86AllrounderXL_sdxlV1_merged_7prompts
    ✅ Found completed: sdxl_depth_trt_ta86_cn_lcm + 20250903-2317-28.5538735

📋 Resume Summary:
  Found 8 completed config+video combinations
  Found 8 results with performance data
  Completed combinations:
    ✅ sdxl_depth_trt_ta86_cn_lcm + 20250903-2317-28.5538735
    ✅ sdxl_depth_trt_ta86_cn_lcm + 20250903-2319-02.8209091
    ...

📊 Work Summary:
  Total combinations: 12
  Already completed: 8
  Remaining to process: 4
  ⏭️  Skipping 8 completed combinations
🚀 Starting processing of 4 remaining combinations...
```

### Important Notes

- **Directory Path**: Resume directory must exist and contain previous results
- **Config/Video Consistency**: Use the same config and video directories as the original run
- **Flexible Parameters**: Prompts file and timeout can be different for remaining work
- **Safety Checks**: Double-checks combinations to prevent duplicate processing
- **Progress Tracking**: Shows clear distinction between resumed and new work

## Prompt Processing Modes

### Mode 1: Config Prompts (Default)
When no `--prompts` file is provided, the test suite uses the prompts defined in your YAML config files:
```yaml
prompt: "A beautiful landscape"
negative_prompt: "low quality, bad quality, blurry"
```

### Mode 2: Individual Prompts (Temporal Splitting)
When `--prompts` file is provided, the test suite:
1. **Ignores** the `prompt` field in your YAML configs
2. **Splits the video temporally** across prompts (e.g., 30s video with 3 prompts = 10s each)
3. **Processes each prompt against its time segment** (much more efficient than full video processing)
4. **Merges all prompt outputs** into a single combined video
5. **Reports performance** for each prompt separately
6. **No pipeline restart** - uses StreamDiffusion's dynamic prompt updating

## Prompts File Format

Create a text file with one prompt per line:

```txt
A hyperrealistic close-up of a man in a crimson silk, windswept auburn hair framing a freckled face, standing on a sun-drenched beach; fine sand clinging to her bare feet, the vast ocean a turquoise expanse behind her, conveying a sense of serene solitude.
A cinematic portrait of a woman with flowing golden hair, wearing an elegant emerald dress, standing in a moonlit garden surrounded by blooming roses and twinkling fairy lights.
A dramatic close-up of a warrior with battle-scarred armor, steely blue eyes reflecting determination, standing against a stormy sky with lightning illuminating ancient castle ruins in the background.
```

## Temporal Prompt Splitting

The test suite now uses **temporal splitting** for maximum efficiency when processing multiple prompts:

### How It Works

1. **Video Segmentation**: The input video is divided into equal time segments based on the number of prompts
2. **Frame Distribution**: Each prompt processes only its assigned frames (e.g., frames 1-100 for prompt 1, frames 101-200 for prompt 2)
3. **Dynamic Prompt Updates**: Uses `stream.update_prompt()` to change prompts without restarting the pipeline
4. **Efficient Processing**: Each frame is processed only once with its corresponding prompt

### Example: 30-Second Video with 3 Prompts

- **Total Frames**: 900 frames (30fps × 30 seconds)
- **Prompt 1**: Frames 1-300 (0-10 seconds) → "Stained glass style..."
- **Prompt 2**: Frames 301-600 (10-20 seconds) → "Cinematic portrait..."
- **Prompt 3**: Frames 601-900 (20-30 seconds) → "Dramatic warrior..."

### Benefits

- **3x Faster**: Video processed once instead of three times
- **Memory Efficient**: No duplicate frame storage
- **Seamless Transitions**: Smooth prompt changes between segments
- **Professional Quality**: Each time segment gets dedicated prompt processing
- **Pipeline Optimization**: Leverages StreamDiffusion's dynamic prompt updating

## Directory Structure

```
project/
├── configs/                    # Your YAML config files
│   ├── config1.yaml
│   ├── config2.yaml
│   └── ...
├── videos/                     # Your video files
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
├── prompts.txt                 # Individual prompts (optional)
├── test_results/               # Output directory (created automatically)
│   ├── test_summary.txt
│   ├── detailed_results.csv
│   ├── performance_comparison.txt
│   ├── config1_video1_merged.mp4  # Merged video (when using prompts)
│   └── individual_results/
└── multi_test.py              # The test suite script
```

## Configuration File Format

Your YAML config files should follow the StreamDiffusion format. When using `--prompts`, the `prompt` field is ignored:

```yaml
model_id: "runwayml/stable-diffusion-v1-5"
width: 512
height: 512
t_index_list: [32, 40, 45]
acceleration: "xformers"
guidance_scale: 1.2
num_inference_steps: 50
# prompt: "This is ignored when using --prompts"
negative_prompt: "low quality, bad quality, blurry"
use_denoising_batch: true
cfg_type: "self"
seed: 42
```

### Required Fields

- `model_id`: Path to the model checkpoint
- `width`: Image width (must be multiple of 64)
- `height`: Image height (must be multiple of 64)

### Optional Fields

- `t_index_list`: Denoising timesteps (default: [32, 40, 45])
- `acceleration`: Acceleration method (default: "xformers")
- `guidance_scale`: CFG scale (default: 1.2)
- `num_inference_steps`: Number of inference steps (default: 50)
- `negative_prompt`: Negative prompt (default: "low quality, bad quality, blurry")
- `use_denoising_batch`: Use denoising batch (default: true)
- `cfg_type`: CFG type (default: "self")
- `seed`: Random seed (default: 42)

**Note**: When using `--prompts`, the `prompt` field in your config is ignored.

## ControlNet Support

The test suite automatically detects and configures ControlNets from your YAML configuration files:

### ControlNet Configuration Format

```yaml
model_id: "runwayml/stable-diffusion-v1-5"
width: 512
height: 512
acceleration: "xformers"

# ControlNet configurations
controlnets:
  - model_id: "lllyasviel/control_v11p_sd15_canny"
    preprocessor: "canny"
    conditioning_scale: 1.0
    enabled: true
    preprocessor_params:
      low_threshold: 100
      high_threshold: 200
      
  - model_id: "lllyasviel/control_v11p_sd15_depth"
    preprocessor: "depth"
    conditioning_scale: 0.8
    enabled: true
    preprocessor_params:
      depth_estimator: "dpt_large"
```

### Supported Preprocessors

- **canny**: Edge detection with configurable thresholds
- **depth**: Depth estimation using various models
- **openpose**: Human pose estimation
- **scribble**: Free-form drawing input
- **segmentation**: Semantic segmentation
- **passthrough**: Direct image input without preprocessing

### ControlNet Integration

- **Automatic Loading**: ControlNets are loaded when the pipeline is created
- **Preprocessor Setup**: Preprocessors are automatically configured with your parameters
- **Performance Impact**: ControlNet processing is included in FPS measurements
- **Memory Management**: ControlNet models are properly managed alongside the main pipeline

## Video File Support

The test suite supports common video formats:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)
- FLV (.flv)

Videos are automatically converted to frames at 30 FPS for processing.

## Framerate Matching

The test suite automatically detects and preserves the input video's framerate in all output videos:

### How It Works

1. **Automatic Detection**: Uses `ffprobe` to extract the exact framerate from input videos
2. **Timing Preservation**: Output videos maintain the same frame timing as input videos
3. **Frame Distribution**: Generated frames are distributed to match input video timing

### Example Scenarios

**Scenario 1: Input 30fps, Processing 30fps**
- Input: 100 frames at 30fps (3.33 seconds)
- Processing: Generates 100 frames
- Output: 100 frames at 30fps (3.33 seconds) - Perfect match

**Scenario 2: Input 30fps, Processing 15fps**
- Input: 100 frames at 30fps (3.33 seconds)
- Processing: Generates 50 frames
- Output: 50 frames at 30fps (3.33 seconds) - Each frame displayed for 2 input frame durations

**Scenario 3: Input 30fps, Processing 60fps**
- Input: 100 frames at 30fps (3.33 seconds)
- Processing: Generates 200 frames
- Output: 200 frames at 30fps (3.33 seconds) - Each input frame duration shows 2 generated frames

### Benefits

- **Seamless Playback**: Output videos can be played alongside input videos
- **Consistent Timing**: All output videos maintain original video timing
- **Professional Quality**: Suitable for video editing and compositing workflows
- **Frame Accuracy**: Precise frame duration calculations using ffmpeg

## Memory Management

The test suite uses intelligent memory management:

1. **Frame Loading**: Frames are loaded into RAM once per video
2. **Caching**: Frames are cached and reused across multiple configs/prompts
3. **Automatic Cleanup**: Frame cache is cleared after processing to free memory
4. **Memory Estimation**: Each frame typically uses 2-4MB depending on resolution

**Memory Requirements**: Ensure you have sufficient RAM to hold all frames from your longest video. For a 1000-frame 512x512 video, expect ~2-4GB RAM usage.

## Output Files

After running the test suite, you'll get several output files:

### 1. Test Summary (`test_summary.txt`)
- Overall test statistics
- Results grouped by configuration
- Results grouped by video
- Top 5 performing configurations
- Prompt processing information (when using `--prompts`)

### 2. Detailed Results (`detailed_results.csv`)
- CSV format with all test results
- Individual frame processing times
- Success/failure status
- Error messages for failed tests
- Prompt processing details (when using `--prompts`)

### 3. Performance Comparison (`performance_comparison.txt`)
- Performance comparison between configurations
- Average FPS for each config
- Sorted results by performance
- Individual prompt performance (when using `--prompts`)

### 4. Individual Results (`*_result.json`)
- JSON files for each config-video combination
- Detailed metrics and configuration parameters
- Frame-by-frame timing data
- Prompt-by-prompt results (when using `--prompts`)

### 4.1. Video Metadata (`*_metadata.json`)
- Comprehensive JSON files generated alongside each output video
- Contains structured data for resume functionality and analysis
- Structure:
  - **video_info**: Config filename, video filename, output filename, total frames, prompts used, processing date
  - **config_details**: Model ID, resolution (width/height), inference steps, guidance scale, negative prompt
  - **performance_metrics**: Overall FPS, min/max/avg FPS, standard deviation, CV percentage, segment FPS list, total processing time
  - **technical_details**: Timeout seconds, start/end times, success status
- Used for enhanced video wall overlays and detailed performance tracking

### 5. Merged Videos (when using `--prompts`)
- `{config}_{video}_merged.mp4`: Combined video from all prompts
- Each frame sequence from a prompt is concatenated into the final video
- Maintains input video framerate and timing

### 6. Single Prompt Videos (when not using `--prompts`)
- `{config}_{video}_output.mp4`: Output video for single prompt processing
- Maintains input video framerate and timing
- Suitable for direct comparison with input videos

### 7. Video Timing Information
All output videos automatically:
- Match the input video's framerate (e.g., 30fps, 24fps, 60fps)
- Preserve the original video's timing and duration
- Use precise frame duration calculations for professional quality

## Example Output

### With Config Prompts
```
StreamDiffusion Multi-Config Test Suite Results
============================================================

Overall Results:
  Total tests: 6
  Successful: 6
  Failed: 0
  Success rate: 100.0%

Quick Performance Summary:
----------------------------------------------------------------------------------------------------
Config                    Video         Resolution   Overall FPS  Avg FPS   Min FPS   Max FPS   Frames
----------------------------------------------------------------------------------------------------
config1                   video1.mp4    512x512     15.23        15.23     14.89     15.67     300
config1                   video2.mp4    512x512     14.89        14.89     14.50     15.28     300
config2                   video1.mp4    512x512     18.45        18.45     18.10     18.80     300
----------------------------------------------------------------------------------------------------

Results by Config:
  config1:
    Tests: 2/2 successful
    Model: runwayml/stable-diffusion-v1-5
    Resolution: 512x512
    ✅ video1.mp4 (300 frames) - Overall FPS: 15.23, Min FPS: 14.89, Max FPS: 15.67, Avg FPS: 15.23, CV: 2.5%
    ✅ video2.mp4 (300 frames) - Overall FPS: 14.89, Min FPS: 14.50, Max FPS: 15.28, Avg FPS: 14.89, CV: 2.7%

Overall FPS Rankings (Higher is Better):
  1. config2 - 18.45 FPS (Avg: 18.45, Range: 18.10-18.80)
  2. config1 - 15.23 FPS (Avg: 15.23, Range: 14.89-15.67)

Performance Statistics:
  Overall FPS - Best: 18.45, Worst: 14.89, Mean: 16.19
  Average FPS - Best: 18.45, Worst: 14.89, Mean: 16.19
  Min FPS - Best: 18.10, Worst: 14.50, Mean: 15.83
  Max FPS - Best: 18.80, Worst: 15.28, Mean: 16.58

Recommendations:
🏆 Best Overall Performance: config2
   - Highest sustained FPS: 18.45
   - Best for: Maximum throughput scenarios

📊 Most Consistent Performance: config1
   - Lowest variance: 2.5% CV
   - Best for: Real-time applications requiring stable frame rates
```

### With Individual Prompts
```
StreamDiffusion Multi-Config Test Suite Results
Using 3 individual prompts from prompts.txt
============================================================

Overall Results:
  Total tests: 6
  Successful: 6
  Failed: 0
  Success rate: 100.0%

Quick Performance Summary:
----------------------------------------------------------------------------------------------------
Config                    Video         Resolution   Overall FPS  Avg FPS   Min FPS   Max FPS   Frames
----------------------------------------------------------------------------------------------------
config1                   video1.mp4    512x512     15.23        15.23     14.89     15.67     900
config1                   video2.mp4    512x512     14.89        14.89     14.50     15.28     900
config2                   video1.mp4    512x512     18.45        18.45     18.10     18.80     900
----------------------------------------------------------------------------------------------------

Results by Config:
  config1:
    Tests: 2/2 successful
    Model: runwayml/stable-diffusion-v1-5
    Resolution: 512x512
    Prompts processed: 3/3 successful
    ✅ video1.mp4 (900 frames, 3 prompts) - Overall FPS: 15.23, Min FPS: 14.89, Max FPS: 15.67, Avg FPS: 15.23, CV: 2.5%
    ✅ video2.mp4 (900 frames, 3 prompts) - Overall FPS: 14.89, Min FPS: 14.50, Max FPS: 15.28, Avg FPS: 14.89, CV: 2.7%

Performance Consistency Analysis:
Configs ranked by FPS stability (lower variance = more stable):
  1. config1 - CV: 2.5% (Std: 0.38, Range: 0.78)
      Mean: 15.23 FPS, Min: 14.89, Max: 15.67
  2. config2 - CV: 2.1% (Std: 0.35, Range: 0.70)
      Mean: 18.45 FPS, Min: 18.10, Max: 18.80

Best Config per Video (Overall FPS):
------------------------------------------------------------
video1.mp4            -> config2                 (18.45 FPS, Avg: 18.45)
video2.mp4            -> config1                 (14.89 FPS, Avg: 14.89)

Performance Improvement Analysis:
Best Overall Config: config2 (18.45 FPS)

Performance vs Best (Overall FPS):
  config1 - 15.23 FPS (+21.1% vs best)

Recommendations:
⚖️  Best Balanced (Performance + Consistency): config2
   - Balanced score: 0.850
   - Performance: 18.45 FPS, Consistency: 2.1% CV
   - Best for: Production environments requiring both speed and reliability
```

## Performance Tips

1. **Use Temporal Splitting**: With `--prompts`, videos are processed once instead of multiple times (3x faster)
2. **Use TensorRT**: Set `acceleration: "tensorrt"` in your configs for best performance
3. **Optimize t_index_list**: Lower values (e.g., [10, 15]) for faster processing, higher values for better quality
4. **Batch Processing**: Enable `use_denoising_batch: true` for better throughput
5. **Resolution**: Lower resolutions process faster but may reduce quality
6. **Model Selection**: Smaller models (SD1.5 vs SDXL) generally process faster
7. **Prompt Length**: Shorter prompts generally process faster than very long, detailed ones
8. **RAM Optimization**: The suite automatically caches frames in RAM for maximum speed
9. **Framerate Optimization**: Output videos automatically match input timing for professional workflows
10. **Dynamic Prompt Updates**: Leverages StreamDiffusion's built-in prompt switching without pipeline restarts
11. **ControlNet Optimization**: Use fewer ControlNets and lower conditioning scales for faster processing
12. **Preprocessor Selection**: Choose efficient preprocessors (e.g., passthrough > canny > depth > openpose)

## Troubleshooting

### Common Issues

1. **ffmpeg not found**: Install ffmpeg and ensure it's in your PATH
2. **CUDA out of memory**: Use memory management options (see below)
3. **Config validation errors**: Check that required fields are present in your YAML files
4. **Model loading failures**: Verify model paths and ensure models are accessible
5. **Video merging fails**: Ensure ffmpeg supports the concat demuxer
6. **Out of memory**: Reduce video resolution or frame count, or close other applications
7. **Framerate detection fails**: Ensure ffprobe is available and input videos are valid

### CUDA Memory Issues

If you encounter `CUDA out of memory` errors, the test suite now includes several memory management features:

#### 1. Batch Processing
Process frames in smaller batches to reduce memory usage:
```bash
# Default: 10 frames per batch
python multi_test.py --configs ./configs --videos ./videos

# Reduce to 5 frames per batch for lower memory usage
python multi_test.py --configs ./configs --videos ./videos --batch-size 5

# Very conservative: 3 frames per batch
python multi_test.py --configs ./configs --videos ./videos --batch-size 3
```

#### 2. Memory Threshold Management
Set when automatic memory cleanup should occur:
```bash
# Default: Cleanup when less than 2GB free
python multi_test.py --configs ./configs --videos ./videos

# More aggressive: Cleanup when less than 1GB free
python multi_test.py --configs ./configs --videos ./videos --memory-threshold 1.0

# Very aggressive: Cleanup when less than 0.5GB free
python multi_test.py --configs ./configs --videos ./videos --memory-threshold 0.5
```

#### 3. Frame Skipping
For very long videos, process every Nth frame to reduce memory and time:
```bash
# Process every frame (default)
python multi_test.py --configs ./configs --videos ./videos

# Process every 2nd frame (2x faster, 2x less memory)
python multi_test.py --configs ./configs --videos ./videos --frame-skip 2

# Process every 3rd frame (3x faster, 3x less memory)
python multi_test.py --configs ./configs --videos ./videos --frame-skip 3
```

#### 4. Combined Memory Management
Use all options together for maximum memory efficiency:
```bash
python multi_test.py --configs ./configs --videos ./videos \
    --batch-size 3 \
    --memory-threshold 0.5 \
    --frame-skip 2
```

#### 5. Automatic Memory Recovery
The test suite now automatically:
- Monitors GPU memory usage in real-time
- Cleans up memory after each batch and prompt
- Retries failed frames after memory cleanup
- Provides detailed memory status information
- Gracefully handles out-of-memory errors without crashing

### Debug Mode

For detailed logging, you can modify the script to add more verbose output or check the individual result JSON files for specific error details.

## Advanced Usage

### Custom Frame Extraction

You can modify the `extract_frames_from_video` method to customize frame extraction parameters (FPS, format, etc.).

### Custom Metrics

Extend the `TestResult` dataclass to include additional metrics like memory usage, GPU utilization, etc.

### Parallel Processing

For faster testing, you could modify the suite to process multiple configs in parallel (requires careful resource management).

### Custom Video Merging

Modify the `merge_videos_from_prompts` method to customize how videos are combined (different frame rates, transitions, etc.).

### Framerate Customization

You can modify the `get_video_framerate` method to implement custom framerate detection logic or override framerates for specific use cases.

## Example Workflow

### With Individual Prompts

1. **Setup**: Create directories and add your configs, videos, and prompts file
2. **Run Tests**: Execute the test suite with `--prompts prompts.txt`
3. **Analyze Results**: Review the generated reports and merged videos
4. **Optimize**: Use results to tune your configurations and prompts
5. **Iterate**: Run tests again with optimized configs

### Without Individual Prompts

1. **Setup**: Create directories and add your configs and videos
2. **Run Tests**: Execute the test suite (uses config prompts)
3. **Analyze Results**: Review the generated reports and output videos
4. **Optimize**: Use results to tune your configurations
5. **Iterate**: Run tests again with optimized configs

## Contributing

Feel free to extend the test suite with additional features:
- Memory usage tracking
- GPU utilization monitoring
- Quality metrics (PSNR, SSIM)
- Automated optimization suggestions
- Integration with CI/CD pipelines
- Custom video effects and transitions
- Prompt performance analysis and optimization
- Advanced memory management strategies
- Custom framerate handling and video processing