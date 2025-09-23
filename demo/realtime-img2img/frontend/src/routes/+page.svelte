<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import ControlNetConfig from '$lib/components/ControlNetConfig.svelte';
  import IPAdapterConfig from '$lib/components/IPAdapterConfig.svelte';
  import BlendingControl from '$lib/components/BlendingControl.svelte';
  import PipelineHooksConfig from '$lib/components/PipelineHooksConfig.svelte';
  import ResolutionPicker from '$lib/components/ResolutionPicker.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import Success from '$lib/components/Success.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore } from '$lib/mediaStream';
  import { appState, startStatePolling, stopStatePolling, fetchAppState, type AppState } from '$lib/store';
  import { parseResolution, type ResolutionInfo } from '$lib/utils';
  import TextArea from '$lib/components/TextArea.svelte';
  import InputControl from '$lib/components/InputControl.svelte';
  import InputSourceSelector from '$lib/components/InputSourceSelector.svelte';

  // Reactive state derived from centralized store
  $: pipelineParams = $appState?.input_params?.properties || $appState?.pipeline_params?.properties;
  $: pipelineInfo = $appState?.info?.properties;
  $: controlnetInfo = $appState?.controlnet;
  $: ipadapterInfo = $appState?.ipadapter;
  $: imagePreprocessingInfo = $appState?.image_preprocessing;
  $: imagePostprocessingInfo = $appState?.image_postprocessing;
  $: latentPreprocessingInfo = $appState?.latent_preprocessing;
  $: latentPostprocessingInfo = $appState?.latent_postprocessing;
  $: ipadapterScale = $appState?.ipadapter?.scale || 1.0;
  $: ipadapterWeightType = $appState?.ipadapter?.weight_type || "linear";
  $: tIndexList = $appState?.t_index_list || [35, 45];
  $: guidanceScale = $appState?.guidance_scale || 1.1;
  $: delta = $appState?.delta || 0.7;
  $: numInferenceSteps = $appState?.num_inference_steps || 50;
  $: seed = $appState?.seed || 2;
  $: promptBlendingConfig = $appState?.prompt_blending;
  $: seedBlendingConfig = $appState?.seed_blending;
  $: normalizePromptWeights = $appState?.normalize_prompt_weights ?? true;
  $: normalizeSeedWeights = $appState?.normalize_seed_weights ?? true;
  $: skipDiffusion = $appState?.skip_diffusion || false;
  $: pageContent = $appState?.page_content || '';
  $: isImageMode = pipelineInfo?.input_mode?.default === PipelineMode.IMAGE;
  $: maxQueueSize = $appState?.max_queue_size || 0;
  $: currentQueueSize = $appState?.queue_size || 0;
  $: selectedModelId = $appState?.model_id || '';
  $: pipelineActive = $appState?.pipeline_active || false;
  $: fps = $appState?.fps || 0;
  $: debugMode = $appState?.debug_mode || false;
  $: debugPendingFrame = $appState?.debug_pending_frame || false;
  
  // Local UI state that doesn't come from backend
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';
  let successMessage: string = '';
  let configRefreshKey: number = 0; // Used to force component refresh when config is uploaded
  
  // Reactive key that updates when pipeline state changes (not just config uploads)
  $: pipelineStateKey = `${configRefreshKey}-${pipelineActive}-${$appState?.pipeline_lifecycle || 'stopped'}`;

  let currentResolution: ResolutionInfo;
  let apiError: string = '';
  let isRetrying: boolean = false;
  
  // Reactive resolution parsing from centralized state
  $: {
    if ($appState?.resolution) {
      currentResolution = parseResolution($appState.resolution);
    } else if ($appState?.current_resolution) {
      currentResolution = {
        width: $appState.current_resolution.width,
        height: $appState.current_resolution.height,
        aspectRatio: $appState.current_resolution.width / $appState.current_resolution.height,
        aspectRatioString: "1:1"
      };
    } else if (pipelineParams?.width?.default && pipelineParams?.height?.default) {
      // Fallback to pipeline params
      currentResolution = {
        width: Number(pipelineParams.width.default),
        height: Number(pipelineParams.height.default),
        aspectRatio: Number(pipelineParams.width.default) / Number(pipelineParams.height.default),
        aspectRatioString: "1:1"
      };
    }
  }
  
  // Panel state management
  let showPromptBlending: boolean = true; // Default to expanded since it's the unified blending interface
  let showResolutionPicker: boolean = true; // Default to expanded
  let showInputControls: boolean = true; // Standardized toggle state moved here
  let leftPanelCollapsed: boolean = false;
  let rightPanelCollapsed: boolean = false;

  // Column resizing
  let leftColumnWidth: number = 25; // Percentage
  let rightColumnWidth: number = 25; // Percentage
  let isDragging: boolean = false;
  let dragTarget: 'left' | 'right' | null = null;

  // Floating video input state
  let floatingVideoPosition = { x: 20, y: 100 };
  let isDraggingVideo = false;
  let videoOffsetX = 0;
  let videoOffsetY = 0;

  // Legacy FPS interval (kept for cleanup in onDestroy)
  let fpsInterval: number | null = null;

  onMount(() => {
    // Start centralized state polling (replaces getSettings, FPS polling, and queue polling)
    // Using 500ms for highly responsive UI updates (FPS, queue size, debug state, etc.)
    startStatePolling(500);
  });

  onDestroy(() => {
    // Stop centralized state polling
    stopStatePolling();
    if (fpsInterval) {
      clearInterval(fpsInterval);
    }
  });

  async function getSettings() {
    // Legacy fallback function - now just triggers state fetch
    try {
      apiError = '';
      isRetrying = false;
      
      // Use centralized state fetching
      const state = await fetch('/api/state').then(r => r.json());
      
      // Prompt initialization is now handled by centralized state management
      console.log('getSettings: Prompt from centralized state:', state.config_prompt);
      
      console.log('getSettings: Legacy function called - using centralized state');
      toggleQueueChecker(true);
      
    } catch (error) {
      console.error('getSettings: Failed to load settings:', error);
      apiError = error instanceof Error ? error.message : 'Failed to connect to the API. Please check if the server is running.';
    }
  }

  async function retryConnection() {
    isRetrying = true;
    await getSettings();
  }

  function handleControlNetUpdate(event: CustomEvent) {
    controlnetInfo = event.detail.controlnet;
    
    // Prompt updates are now handled by centralized state management
    if (event.detail.config_prompt) {
      console.log('handleControlNetUpdate: Config prompt updated:', event.detail.config_prompt);
    }
    
    // Update t_index_list if available
    if (event.detail.t_index_list) {
      tIndexList = [...event.detail.t_index_list];
    }
    
    console.log('handleControlNetUpdate: ControlNet updated:', controlnetInfo);
    console.log('handleControlNetUpdate: T-Index List updated:', tIndexList);
  }

  async function handleTIndexListUpdate(newTIndexList: number[]) {
    try {
      const response = await fetch('/api/params', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          t_index_list: newTIndexList
        }),
      });

      if (response.ok) {
        tIndexList = [...newTIndexList]; // Update local state
        console.log('handleTIndexListUpdate: T-Index List updated:', tIndexList);
      } else {
        const result = await response.json();
        console.error('handleTIndexListUpdate: Failed to update t_index_list:', result.detail);
      }
    } catch (error) {
      console.error('handleTIndexListUpdate: Failed to update t_index_list:', error);
    }
  }

  async function handleSkipDiffusionUpdate(enabled: boolean) {
    try {
      const response = await fetch('/api/params', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          skip_diffusion: enabled
        }),
      });

      if (response.ok) {
        skipDiffusion = enabled; // Update local state
        console.log('handleSkipDiffusionUpdate: Skip diffusion updated:', skipDiffusion);
        
        // Show success message
        successMessage = `Skip diffusion ${enabled ? 'enabled' : 'disabled'}. ${enabled ? 'Only pre/post processing will run.' : 'Full diffusion pipeline restored.'}`;
        setTimeout(() => {
          successMessage = '';
        }, 3000);
      } else {
        const result = await response.json();
        console.error('handleSkipDiffusionUpdate: Failed to update skip_diffusion:', result.detail);
        warningMessage = 'Failed to update skip diffusion: ' + result.detail;
      }
    } catch (error) {
      console.error('handleSkipDiffusionUpdate: Failed to update skip_diffusion:', error);
      warningMessage = 'Failed to update skip diffusion: ' + (error instanceof Error ? error.message : String(error));
    }
  }

  async function handleResolutionUpdate(resolution: string) {
    try {
      const response = await fetch('/api/params', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ resolution }),
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('handleResolutionUpdate: Resolution updated successfully:', result.message);
        
        // Show success toast/message instead of warning
        if (result.message) {
          successMessage = result.message;
          setTimeout(() => {
            successMessage = '';
          }, 3000);
        }
      } else {
        const result = await response.json();
        console.error('handleResolutionUpdate: Failed to update resolution:', result.detail);
        // If the pipeline isn't active and server still returns the old error, convert to friendly info
        if (!pipelineActive && /Pipeline is not initialized/i.test(result.detail || '')) {
          successMessage = 'Resolution updated and will be applied when streaming starts.';
          setTimeout(() => { successMessage = ''; }, 3000);
        } else {
          warningMessage = 'Failed to update resolution: ' + result.detail;
        }
      }
    } catch (error: unknown) {
      console.error('handleResolutionUpdate: Failed to update resolution:', error);
      warningMessage = 'Failed to update resolution: ' + (error instanceof Error ? error.message : String(error));
    }
  }

  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    // Queue checking is now handled by centralized state polling
  }

  function getSreamdata() {
    if (isImageMode) {
      return [$appState, $onFrameChangeStore?.blob];
    } else {
      return [$appState];
    }
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
    warningMessage = 'Session timed out. Please try again.';
  }
  
  // Watch for resolution changes
  let previousResolution: string = '';
  let userInitiatedResolutionChange = false;
  
  $: {
    const currentAppResolution = $appState?.resolution;
    if (currentAppResolution && currentAppResolution !== previousResolution && previousResolution !== '') {
      const nextResolution = currentAppResolution;
      previousResolution = nextResolution;
      
      // Only trigger automatic resolution update if it wasn't initiated by user action
      // This prevents double pipeline restarts when user manually updates resolution
      if (!userInitiatedResolutionChange && pipelineActive) {
        handleResolutionUpdate(nextResolution);
      } else if (!userInitiatedResolutionChange && !pipelineActive) {
        // No pipeline yet: don't call backend, just inform the user
        successMessage = 'Resolution set to ' + nextResolution.split(' ')[0] + '. It will be applied when streaming starts.';
        setTimeout(() => { successMessage = ''; }, 3000);
      }
      
      // Reset the flag after processing
      userInitiatedResolutionChange = false;
    } else if (currentAppResolution && previousResolution === '') {
      previousResolution = currentAppResolution;
    }
  }
  
  let disabled = false;
  async function toggleLcmLive() {
    try {
      if (!isLCMRunning) {
        if (isImageMode) {
          await mediaStreamActions.enumerateDevices();
          await mediaStreamActions.start();
        }
        disabled = true;
        await lcmLiveActions.start(getSreamdata);
        disabled = false;
        toggleQueueChecker(false);
      } else {
        if (isImageMode) {
          mediaStreamActions.stop();
        }
        lcmLiveActions.stop();
        toggleQueueChecker(true);
      }
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : '';
      disabled = false;
      toggleQueueChecker(true);
    }
  }

  // FPS is now handled by centralized state polling

  async function refreshBlendingConfigs() {
    try {
      const response = await fetch('/api/blending/current');
      const data = await response.json();
      
      if (data.prompt_blending) {
        promptBlendingConfig = data.prompt_blending;
        console.log('refreshBlendingConfigs: Updated prompt blending:', promptBlendingConfig);
      }
      
      if (data.seed_blending) {
        seedBlendingConfig = data.seed_blending;
        console.log('refreshBlendingConfigs: Updated seed blending:', seedBlendingConfig);
      }
      
      if (data.normalize_prompt_weights !== undefined) {
        normalizePromptWeights = data.normalize_prompt_weights;
      }
      
      if (data.normalize_seed_weights !== undefined) {
        normalizeSeedWeights = data.normalize_seed_weights;
      }
      
      console.log('refreshBlendingConfigs: Blending configs refreshed');
    } catch (error) {
      console.error('refreshBlendingConfigs: Failed to refresh blending configs:', error);
    }
  }

  async function handleImagePreprocessingRefresh() {
    try {
      const response = await fetch('/api/pipeline-hooks/image_preprocessing/info-config');
      if (response.ok) {
        const data = await response.json();
        imagePreprocessingInfo = data.image_preprocessing || null;
      }
    } catch (err) {
      console.warn('handleImagePreprocessingRefresh: Failed to refresh image preprocessing info:', err);
    }
  }

  async function handleImagePostprocessingRefresh() {
    try {
      const response = await fetch('/api/pipeline-hooks/image_postprocessing/info-config');
      if (response.ok) {
        const data = await response.json();
        imagePostprocessingInfo = data.image_postprocessing || null;
      }
    } catch (err) {
      console.warn('handleImagePostprocessingRefresh: Failed to refresh image postprocessing info:', err);
    }
  }

  async function handleLatentPreprocessingRefresh() {
    try {
      const response = await fetch('/api/pipeline-hooks/latent_preprocessing/info-config');
      if (response.ok) {
        const data = await response.json();
        latentPreprocessingInfo = data.latent_preprocessing || null;
      }
    } catch (err) {
      console.warn('handleLatentPreprocessingRefresh: Failed to refresh latent preprocessing info:', err);
    }
  }

  async function handleLatentPostprocessingRefresh() {
    try {
      const response = await fetch('/api/pipeline-hooks/latent_postprocessing/info-config');
      if (response.ok) {
        const data = await response.json();
        latentPostprocessingInfo = data.latent_postprocessing || null;
      }
    } catch (err) {
      console.warn('handleLatentPostprocessingRefresh: Failed to refresh latent postprocessing info:', err);
    }
  }

  // Pipeline configuration upload
  let fileInput: HTMLInputElement;
  let uploading = false;

  async function uploadConfig() {
    if (!fileInput.files || fileInput.files.length === 0) {
      warningMessage = 'Please select a YAML file';
      return;
    }

    const file = fileInput.files[0];
    if (!file.name.endsWith('.yaml') && !file.name.endsWith('.yml')) {
      warningMessage = 'Please select a YAML file (.yaml or .yml)';
      return;
    }

    uploading = true;

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/controlnet/upload-config', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        console.log('uploadConfig: Full response received:', result);
        console.log('uploadConfig: controls_updated flag:', result.controls_updated);
        
        // If pipeline is running, stop it first
        if (isLCMRunning) {
          console.log('uploadConfig: Stopping active pipeline before applying config...');
          await toggleLcmLive(); // Stop the current pipeline
          successMessage = 'Configuration uploaded successfully! Pipeline stopped and reset to config.';
        } else {
          successMessage = 'Configuration uploaded successfully! Pipeline will load when you start streaming.';
        }
        fileInput.value = '';
        
        // Update ControlNet info
        if (result.controlnet) {
          controlnetInfo = result.controlnet;
          console.log('uploadConfig: Updated controlnetInfo to:', controlnetInfo);
        }
        
        // Update IPAdapter info
        if (result.ipadapter) {
          ipadapterInfo = result.ipadapter;
          ipadapterScale = result.ipadapter.scale || 1.0;
        }
        // Update model badge if present
        if (result.model_id) {
          selectedModelId = result.model_id;
        }
        
        // Update streaming parameters
        if (result.t_index_list) {
          tIndexList = [...result.t_index_list];
          console.log('uploadConfig: Updated tIndexList to:', tIndexList);
        }
        if (result.guidance_scale !== undefined) {
          guidanceScale = result.guidance_scale;
          console.log('uploadConfig: Updated guidanceScale to:', guidanceScale);
        }
        if (result.delta !== undefined) {
          delta = result.delta;
          console.log('uploadConfig: Updated delta to:', delta);
        }
        if (result.num_inference_steps !== undefined) {
          numInferenceSteps = result.num_inference_steps;
          console.log('uploadConfig: Updated numInferenceSteps to:', numInferenceSteps);
        }
        if (result.seed !== undefined) {
          seed = result.seed;
          console.log('uploadConfig: Updated seed to:', seed);
        }
        
        // Config values are now handled by centralized state management
        if (result.config_values) {
          console.log('uploadConfig: Config values updated via centralized state');
        }

        // Update normalization settings
        if (result.normalize_prompt_weights !== undefined) {
          normalizePromptWeights = result.normalize_prompt_weights;
        }
        if (result.normalize_seed_weights !== undefined) {
          normalizeSeedWeights = result.normalize_seed_weights;
        }
        if (result.skip_diffusion !== undefined) {
          skipDiffusion = result.skip_diffusion;
        }
        
        // Update blending configurations
        if (result.prompt_blending) {
          promptBlendingConfig = result.prompt_blending;
          showPromptBlending = true;  // Auto-expand if config has blending data
          console.log('uploadConfig: Updated prompt blending config:', promptBlendingConfig);
        }
        if (result.seed_blending) {
          seedBlendingConfig = result.seed_blending;
          console.log('uploadConfig: Updated seed blending config:', seedBlendingConfig);
        }
        
        // Prompt and resolution updates are now handled by centralized state management
        if (result.config_prompt) {
          console.log('uploadConfig: Config prompt updated via centralized state:', result.config_prompt);
        } else if (result.prompt) {
          console.log('uploadConfig: Prompt updated via centralized state:', result.prompt);
        }
        
        if (result.negative_prompt !== undefined) {
          console.log('uploadConfig: Negative prompt updated via centralized state:', result.negative_prompt);
        }
        
        if (result.current_resolution) {
          console.log('uploadConfig: Resolution updated via centralized state:', result.current_resolution);
        }
        
        // Force complete refresh of all pipeline hook components by generating new keys
        const configUploadTimestamp = Date.now();
        
        // Update pipeline hooks info with forced refresh
        if (result.image_preprocessing) {
          imagePreprocessingInfo = result.image_preprocessing;
          console.log('uploadConfig: Updated image preprocessing info:', imagePreprocessingInfo);
        }
        if (result.image_postprocessing) {
          imagePostprocessingInfo = result.image_postprocessing;
          console.log('uploadConfig: Updated image postprocessing info:', imagePostprocessingInfo);
        }
        if (result.latent_preprocessing) {
          latentPreprocessingInfo = result.latent_preprocessing;
          console.log('uploadConfig: Updated latent preprocessing info:', latentPreprocessingInfo);
        }
        if (result.latent_postprocessing) {
          latentPostprocessingInfo = result.latent_postprocessing;
          console.log('uploadConfig: Updated latent postprocessing info:', latentPostprocessingInfo);
        }
        
        // Trigger complete re-initialization of all components by updating the config refresh key
        configRefreshKey = configUploadTimestamp;
        
        // Reset all input source selectors to defaults
        await resetAllInputSourceSelectors();
        
        // Success toast will auto-dismiss
      } else {
        warningMessage = `Error: ${result.detail || 'Failed to load configuration'}`;
      }
    } catch (error) {
      console.error('uploadConfig: Upload failed:', error);
      warningMessage = 'Upload failed. Please try again.';
    } finally {
      uploading = false;
    }
  }

  function selectFile() {
    fileInput.click();
  }

  // Column resizing functions
  function startDrag(event: MouseEvent, target: 'left' | 'right') {
    isDragging = true;
    dragTarget = target;
    event.preventDefault();
    
    document.addEventListener('mousemove', handleDrag);
    document.addEventListener('mouseup', stopDrag);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }

  function handleDrag(event: MouseEvent) {
    if (!isDragging || !dragTarget) return;
    
    const containerWidth = document.querySelector('.main-grid')?.clientWidth || 1200;
    const mouseX = event.clientX;
    const containerRect = document.querySelector('.main-grid')?.getBoundingClientRect();
    
    if (containerRect) {
      const relativeX = mouseX - containerRect.left;
      const percentage = (relativeX / containerWidth) * 100;
      
      if (dragTarget === 'left') {
        leftColumnWidth = Math.max(15, Math.min(40, percentage));
      } else if (dragTarget === 'right') {
        const rightPercentage = 100 - percentage;
        rightColumnWidth = Math.max(15, Math.min(40, rightPercentage));
      }
    }
  }

  function stopDrag() {
    isDragging = false;
    dragTarget = null;
    document.removeEventListener('mousemove', handleDrag);
    document.removeEventListener('mouseup', stopDrag);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }

  // Calculate center column width
  $: centerColumnWidth = 100 - (leftPanelCollapsed ? 0 : leftColumnWidth) - (rightPanelCollapsed ? 0 : rightColumnWidth);

  // Floating video input drag functions
  function startVideoDrag(event: MouseEvent) {
    isDraggingVideo = true;
    const target = event.currentTarget as HTMLElement;
    if (target) {
      const rect = target.getBoundingClientRect();
      videoOffsetX = event.clientX - rect.left;
      videoOffsetY = event.clientY - rect.top;
    }
    
    document.addEventListener('mousemove', handleVideoDrag);
    document.addEventListener('mouseup', stopVideoDrag);
    document.body.style.userSelect = 'none';
    event.preventDefault();
  }

  function handleVideoDrag(event: MouseEvent) {
    if (!isDraggingVideo) return;
    
    const newX = event.clientX - videoOffsetX;
    const newY = event.clientY - videoOffsetY;
    
    // Keep within viewport bounds
    const maxX = window.innerWidth - 320; // Assuming 300px width + some margin
    const maxY = window.innerHeight - 240; // Assuming 200px height + some margin
    
    floatingVideoPosition = {
      x: Math.max(0, Math.min(maxX, newX)),
      y: Math.max(0, Math.min(maxY, newY))
    };
  }

  function stopVideoDrag() {
    isDraggingVideo = false;
    document.removeEventListener('mousemove', handleVideoDrag);
    document.removeEventListener('mouseup', stopVideoDrag);
    document.body.style.userSelect = '';
  }

  function handleBaseInputSourceChanged(event: CustomEvent) {
    const { componentType, sourceType, sourceData } = event.detail;
    console.log('Main page: Base input source changed:', event.detail);
  }

  // Component references for resetting input sources
  let baseInputSourceSelector: any;
  let controlNetConfigComponent: any;
  let ipAdapterConfigComponent: any;

  // Debug control functions
  async function toggleDebugMode() {
    try {
      const endpoint = debugMode ? '/api/debug/disable' : '/api/debug/enable';
      const response = await fetch(endpoint, { method: 'POST' });
      
      if (response.ok) {
        const result = await response.json();
        console.log('toggleDebugMode: Debug mode toggled:', result.message);
      } else {
        const error = await response.json();
        warningMessage = `Failed to toggle debug mode: ${error.detail}`;
      }
    } catch (error) {
      console.error('toggleDebugMode: Error toggling debug mode:', error);
      warningMessage = 'Failed to toggle debug mode: ' + (error instanceof Error ? error.message : String(error));
    }
  }

  async function stepFrame() {
    try {
      const response = await fetch('/api/debug/step', { method: 'POST' });
      
      if (response.ok) {
        const result = await response.json();
        console.log('stepFrame: Frame step requested:', result.message);
        // Immediately refresh state to update button disabled state
        await fetchAppState();
      } else {
        const error = await response.json();
        warningMessage = `Failed to step frame: ${error.detail}`;
      }
    } catch (error) {
      console.error('stepFrame: Error stepping frame:', error);
      warningMessage = 'Failed to step frame: ' + (error instanceof Error ? error.message : String(error));
    }
  }

  // Function to reset all input source selectors
  async function resetAllInputSourceSelectors() {
    console.log('resetAllInputSourceSelectors: Resetting all input source selectors');
    
    try {
      // Reset base input source selector
      if (baseInputSourceSelector && baseInputSourceSelector.resetToDefaults) {
        baseInputSourceSelector.resetToDefaults();
      }
      
      // Reset ControlNet input source selectors (handled by ControlNetConfig component)
      if (controlNetConfigComponent && controlNetConfigComponent.resetInputSources) {
        controlNetConfigComponent.resetInputSources();
      }
      
      // Reset IPAdapter input source selector (handled by IPAdapterConfig component)
      if (ipAdapterConfigComponent && ipAdapterConfigComponent.resetInputSource) {
        ipAdapterConfigComponent.resetInputSource();
      }
      
      console.log('resetAllInputSourceSelectors: All input source selectors reset');
    } catch (error) {
      console.error('resetAllInputSourceSelectors: Error resetting input source selectors:', error);
    }
  }
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<main class="h-screen flex flex-col overflow-hidden">
  <Warning bind:message={warningMessage}></Warning>
  <Success bind:message={successMessage}></Success>
  
  <!-- Header Section -->
  <header class="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-2 flex-shrink-0">
    <div class="flex items-center justify-between">
      <div class="flex-1">
        {#if pageContent}
          <div class="text-left">
            {@html pageContent}
          </div>
        {/if}
        {#if maxQueueSize > 0}
          <p class="text-xs text-center mt-1">
            <span id="queue_size" class="font-bold">{currentQueueSize}</span> users sharing GPU.
            <a
              href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
              target="_blank"
              class="text-blue-500 underline hover:no-underline">Duplicate</a
            > to run on your own GPU.
          </p>
        {/if}
      </div>

      <!-- Pipeline Configuration and Main Controls -->
      <div class="flex items-center gap-2">
        <!-- Pipeline Configuration -->
        <Button on:click={selectFile} disabled={uploading} classList={'text-sm px-4 py-2 font-semibold'}>
          {uploading ? 'Uploading...' : 'Load YAML Config'}
        </Button>

        <input
          bind:this={fileInput}
          type="file"
          accept=".yaml,.yml"
          class="hidden"
          on:change={uploadConfig}
        />
        
        <!-- Debug Controls (only show when streaming) -->
        {#if isLCMRunning}
          <div class="flex items-center gap-1 px-3 py-1 bg-yellow-100 dark:bg-yellow-900 rounded-lg border">
            <span class="text-xs font-semibold text-yellow-800 dark:text-yellow-200">Debug:</span>
            <Button 
              on:click={toggleDebugMode} 
              classList={`text-xs px-2 py-1 ${debugMode ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-green-500 hover:bg-green-600 text-white'}`}
            >
              {debugMode ? 'Resume' : 'Pause'}
            </Button>
            {#if debugMode}
              <Button 
                on:click={stepFrame} 
                disabled={debugPendingFrame}
                classList="text-xs px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white disabled:opacity-50"
              >
                Step →
              </Button>
            {/if}
          </div>
        {/if}

        <!-- Main Control Button -->
        <Button on:click={toggleLcmLive} {disabled} classList={'text-sm px-4 py-2 font-semibold'}>
          {#if isLCMRunning}
            Stop Stream
          {:else}
            Start Stream
          {/if}
        </Button>
      </div>
    </div>

    <div class="flex items-center gap-2 justify-end">
      {#if selectedModelId}
      <div class="flex items-center gap-2 px-3 py-1 bg-blue-100 dark:bg-blue-900 rounded-lg ml-2">
        <span class="text-xs font-semibold text-blue-800 dark:text-blue-200">Model</span>
        <span class="text-sm font-medium text-blue-900 dark:text-blue-100 truncate max-w-[260px]" title={selectedModelId}>
          {selectedModelId}
        </span>
      </div>
      {/if}
    </div>
      
    
  </header>

  {#if pipelineParams}
    <!-- Main Content Grid with Resizable Columns -->
    <div class="flex-1 flex p-4 overflow-hidden main-grid" style="gap: 0;">
      
      <!-- Left Panel - Input and Basic Controls -->
      {#if !leftPanelCollapsed}
        <div
          class="flex flex-col gap-4 overflow-hidden pr-2"
          style="width: {leftColumnWidth}%; min-width: 250px;"
        >
          <!-- Panel Header -->
          <div class="flex items-center justify-between flex-shrink-0">
            <h2 class="text-lg font-semibold">Input & Controls</h2>
            <button
              on:click={() => leftPanelCollapsed = !leftPanelCollapsed}
              class="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
              title="Collapse panel"
            >
              ←
            </button>
          </div>
          
          <!-- Fixed Video Input Section (Image Mode Only) -->
          {#if isImageMode}
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 flex-shrink-0">
              <h3 class="text-md font-medium mb-3">Video Input</h3>
              <VideoInput
                width={Number(pipelineParams.width.default)}
                height={Number(pipelineParams.height.default)}
                {currentResolution}
              />
            </div>
          {/if}

          <!-- Scrollable Controls Section -->
          <div class="flex-1 overflow-y-auto space-y-4">
            <!-- Resolution Picker -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <button
                on:click={() => showResolutionPicker = !showResolutionPicker}
                class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
              >
                <h3 class="text-md font-medium">Resolution</h3>
                <span class="text-sm">{showResolutionPicker ? '−' : '+'}</span>
              </button>
              {#if showResolutionPicker}
                <div class="p-4 pt-1">
                  <ResolutionPicker {currentResolution} on:userResolutionChange={() => userInitiatedResolutionChange = true} />
                </div>
              {/if}
            </div>

            <!-- Base Pipeline Input Source -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <div class="p-4">
                <h3 class="text-md font-medium mb-3">Base Input Source</h3>
                <InputSourceSelector
                  bind:this={baseInputSourceSelector}
                  componentType="base"
                  on:sourceChanged={handleBaseInputSourceChanged}
                />
                <p class="text-xs text-gray-500 mt-3">
                  Select the input source for the main pipeline. This affects the base image that gets processed through the diffusion model.
                </p>
              </div>
            </div>

            <!-- Unified Blending Control -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <button 
                on:click={() => showPromptBlending = !showPromptBlending}
                class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
              >
                <h3 class="text-md font-medium">Blending Controls</h3>
                <span class="text-sm">{showPromptBlending ? '−' : '+'}</span>
              </button>
              {#if showPromptBlending}
                <div class="p-4 pt-1">
                  <BlendingControl
                    {promptBlendingConfig}
                    {seedBlendingConfig}
                    {normalizePromptWeights}
                    {normalizeSeedWeights}
                    currentPrompt={$appState?.config_prompt || ''}
                  />
                </div>
              {/if}
            </div>

            <!-- Input Control Section -->
            <div class="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <button 
                on:click={() => showInputControls = !showInputControls}
                class="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-t-lg"
              >
                <h3 class="text-md font-medium">Input Controls</h3>
                <span class="text-sm">{showInputControls ? '−' : '+'}</span>
              </button>
              {#if showInputControls}
                <div class="p-4 pt-1">
                  <InputControl />
                </div>
              {/if}
            </div>
          </div>
        </div>

        <!-- Left Resizer -->
        <div
          class="w-1 bg-gray-300 dark:bg-gray-600 hover:bg-blue-500 cursor-col-resize flex-shrink-0 transition-colors"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize left panel"
          on:mousedown={(e) => startDrag(e, 'left')}
          title="Drag to resize"
        ></div>
      {:else}
        <!-- Collapsed Left Panel Toggle -->
        <div class="flex flex-col items-center py-4 pr-2">
          <button
            on:click={() => leftPanelCollapsed = !leftPanelCollapsed}
            class="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 writing-mode-vertical"
            title="Expand Input & Controls"
          >
            →
          </button>
        </div>
      {/if}

      <!-- Center Panel - Main Image Output -->
      <div
        class="flex flex-col px-2"
        style="width: {centerColumnWidth}%; min-width: 300px;"
      >
        <div class="flex-1 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 flex flex-col">
          <div class="flex items-center justify-between mb-4">
            <h2 class="text-lg font-semibold">Generated Output</h2>
            <div class="flex items-center gap-4">
              {#if isLCMRunning}
                <div class="flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900 rounded-lg">
                  <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span class="text-sm font-medium text-green-800 dark:text-green-200">
                    {(fps || 0).toFixed(1)} FPS
                  </span>
                </div>
              {/if}
              {#if debugMode}
                <div class="flex items-center gap-2 px-3 py-1 bg-yellow-100 dark:bg-yellow-900 rounded-lg">
                  <div class="w-2 h-2 bg-yellow-500 rounded-full"></div>
                  <span class="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                    Debug Paused
                  </span>
                </div>
              {/if}
              <div class="text-sm text-gray-600 dark:text-gray-400">
                Status: {debugMode ? 'Debug Paused' : isLCMRunning ? 'Streaming' : 'Stopped'}
              </div>
            </div>
          </div>
          <div class="flex-1 flex items-center justify-center">
            <div class="w-full max-w-2xl">
              <ImagePlayer {currentResolution} />
            </div>
          </div>
        </div>
      </div>

      <!-- Right Panel - Advanced Controls -->
      {#if !rightPanelCollapsed}
        <!-- Right Resizer -->
        <div
          class="w-1 bg-gray-300 dark:bg-gray-600 hover:bg-blue-500 cursor-col-resize flex-shrink-0 transition-colors"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize right panel"
          on:mousedown={(e) => startDrag(e, 'right')}
          title="Drag to resize"
        ></div>

        <div
          class="flex flex-col gap-4 overflow-y-auto pl-2"
          style="width: {rightColumnWidth}%; min-width: 250px;"
        >
          <!-- Panel Header -->
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold">Advanced Settings</h2>
            <button
              on:click={() => rightPanelCollapsed = !rightPanelCollapsed}
              class="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
              title="Collapse panel"
            >
              →
            </button>
          </div>
          
          <ControlNetConfig 
            bind:this={controlNetConfigComponent}
            {controlnetInfo} 
            {tIndexList}
            {guidanceScale}
            {delta}
            {numInferenceSteps}
            on:controlnetUpdated={handleControlNetUpdate}
            on:tIndexListUpdated={(e) => handleTIndexListUpdate(e.detail)}
            on:controlnetConfigChanged={getSettings}
          ></ControlNetConfig>

          <IPAdapterConfig 
            bind:this={ipAdapterConfigComponent}
            {ipadapterInfo} 
            currentScale={ipadapterScale}
            currentWeightType={ipadapterWeightType}
            currentEnabled={ipadapterInfo?.enabled ?? true}
          ></IPAdapterConfig>

          {#key pipelineStateKey}
            <PipelineHooksConfig 
              hookType="image_preprocessing"
              hookInfo={imagePreprocessingInfo}
              {skipDiffusion}
              on:refresh={handleImagePreprocessingRefresh}
              on:skipDiffusionChanged={(e) => handleSkipDiffusionUpdate(e.detail)}
            ></PipelineHooksConfig>

            <PipelineHooksConfig 
              hookType="image_postprocessing"
              hookInfo={imagePostprocessingInfo}
              on:refresh={handleImagePostprocessingRefresh}
            ></PipelineHooksConfig>

            <PipelineHooksConfig 
              hookType="latent_preprocessing"
              hookInfo={latentPreprocessingInfo}
              on:refresh={handleLatentPreprocessingRefresh}
            ></PipelineHooksConfig>

            <PipelineHooksConfig 
              hookType="latent_postprocessing"
              hookInfo={latentPostprocessingInfo}
              on:refresh={handleLatentPostprocessingRefresh}
            ></PipelineHooksConfig>
          {/key}
        </div>
      {:else}
        <!-- Collapsed Right Panel Toggle -->
        <div class="flex flex-col items-center py-4 pl-2">
          <button
            on:click={() => rightPanelCollapsed = !rightPanelCollapsed}
            class="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
            title="Expand Advanced Settings"
          >
            ←
          </button>
        </div>
      {/if}
    </div>
  {:else if apiError}
    <!-- API Error -->
    <div class="flex-1 flex flex-col items-center justify-center gap-6 py-48 text-center">
      <div>
        <h2 class="text-2xl font-bold text-red-600 mb-2">API Connection Failed</h2>
        <p class="text-gray-600 dark:text-gray-400 mb-4 max-w-md">
          {apiError}
        </p>
        <Button 
          on:click={retryConnection} 
          disabled={isRetrying} 
          classList="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2"
        >
          {#if isRetrying}
            <Spinner classList="w-4 h-4 mr-2 animate-spin" />
            Retrying...
          {:else}
            Retry Connection
          {/if}
        </Button>
      </div>
    </div>
  {:else}
    <!-- Loading State -->
    <div class="flex-1 flex items-center justify-center">
      <div class="flex items-center gap-3 text-2xl">
        <Spinner classList={'animate-spin opacity-50'} />
        <p>Loading StreamDiffusion...</p>
      </div>
    </div>
  {/if}

  <!-- Floating Video Input (when left panel is collapsed and in image mode) -->
  {#if leftPanelCollapsed && isImageMode && pipelineParams}
    <div
      class="fixed z-50 bg-white dark:bg-gray-800 rounded-lg border-2 border-gray-300 dark:border-gray-600 shadow-lg"
      style="left: {floatingVideoPosition.x}px; top: {floatingVideoPosition.y}px; width: 320px;"
    >
      <!-- Drag Handle -->
      <div
        class="bg-gray-100 dark:bg-gray-700 px-3 py-2 rounded-t-lg cursor-move border-b border-gray-200 dark:border-gray-600 flex items-center justify-between"
        role="button"
        tabindex="0"
        on:mousedown={startVideoDrag}
      >
        <div class="flex items-center gap-2 text-sm font-medium">
          <span>📹</span>
          <span>Video Input</span>
        </div>
        <span class="text-xs text-gray-500 dark:text-gray-400">Drag to move</span>
      </div>
      
      <!-- Video Input Content -->
      <div class="p-3">
        <VideoInput
          width={Number(pipelineParams.width.default)}
          height={Number(pipelineParams.height.default)}
          {currentResolution}
        />
      </div>
    </div>
  {/if}
</main>

<style lang="postcss">
  @reference "tailwindcss";
  
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
  
  /* Custom scrollbar styling */
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 6px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    @apply bg-gray-100 dark:bg-gray-800;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    @apply bg-gray-300 dark:bg-gray-600 rounded-full;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    @apply bg-gray-400 dark:bg-gray-500;
  }

  /* Resizer styling */
  .main-grid {
    position: relative;
  }

  /* Prevent text selection during drag */
  :global(body.dragging) {
    user-select: none;
    cursor: col-resize !important;
  }

  /* Improved resizer hover effects */
  .resizer:hover {
    background-color: rgb(59 130 246) !important; /* blue-500 */
  }
</style>
