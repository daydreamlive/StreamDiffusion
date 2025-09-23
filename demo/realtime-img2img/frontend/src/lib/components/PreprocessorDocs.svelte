<script lang="ts">
  import { onMount } from 'svelte';
  import Button from './Button.svelte';

  export let visible = false;

  let preprocessorsInfo: any = null;
  let selectedPreprocessor: string | null = null;
  let configTemplate: any = null;
  let showFullTemplate = false;

  onMount(async () => {
    try {
      const response = await fetch('/api/preprocessors/info');
      const data = await response.json();
      preprocessorsInfo = data.preprocessors;
      configTemplate = data.template;
    } catch (error) {
      console.error('Failed to load preprocessor documentation:', error);
    }
  });

  function generateConfig(preprocessorType: string) {
    if (!preprocessorsInfo || !preprocessorsInfo[preprocessorType]) return '';
    
    const preprocessor = preprocessorsInfo[preprocessorType];
    const controlnet = preprocessor.example_config;
    
    const fullConfig = {
      model_id: "path/to/your/model.safetensors",
      t_index_list: [32, 45],
      width: 512,
      height: 512,
      device: "cuda",
      dtype: "float16",
      prompt: "your amazing prompt here",
      negative_prompt: "blurry, low quality, distorted",
      guidance_scale: 1.1,
      num_inference_steps: 50,
      use_denoising_batch: true,
      delta: 0.7,
      frame_buffer_size: 1,
      use_tiny_vae: true,
      acceleration: "xformers",
      cfg_type: "self",
      seed: 42,
      controlnets: [controlnet]
    };

    return `# ${preprocessor.name} Configuration Example
# Description: ${preprocessor.description}
# Use cases: ${preprocessor.use_cases.join(', ')}

${showFullTemplate ? 
  'yaml\n' + yamlStringify(fullConfig) : 
  'yaml\ncontrolnets:\n' + yamlStringify([controlnet]).split('\n').map(line => '  ' + line).join('\n')
}`;
  }

  function yamlStringify(obj: any): string {
    return JSON.stringify(obj, null, 2)
      .replace(/"/g, '')
      .replace(/,$/gm, '')
      .replace(/\{/g, '')
      .replace(/\}/g, '')
      .replace(/\[/g, '')
      .replace(/\]/g, '')
      .replace(/^\s*$/gm, '')
      .split('\n')
      .filter(line => line.trim())
      .join('\n');
  }

  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      // Could add a toast notification here
      console.log('Configuration copied to clipboard');
    });
  }

  function getParameterWidget(paramName: string, param: any) {
    if (param.type === 'int' && param.range) {
      return {
        type: 'range',
        min: param.range[0],
        max: param.range[1],
        default: param.default
      };
    }
    return { type: 'text', default: param.default };
  }
</script>

{#if visible}
  <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" on:click={() => visible = false}>
    <div class="bg-white dark:bg-gray-800 rounded-lg max-w-6xl max-h-[90vh] overflow-y-auto p-6 m-4" on:click|stopPropagation>
      <div class="flex justify-between items-center mb-6">
        <h2 class="text-2xl font-bold">ControlNet Preprocessor Documentation</h2>
        <button on:click={() => visible = false} class="text-gray-500 hover:text-gray-700 text-2xl">×</button>
      </div>

      {#if preprocessorsInfo}
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Left Column: Preprocessor List -->
          <div class="space-y-2">
            <h3 class="text-lg font-semibold mb-3">Available Preprocessors</h3>
            {#each Object.entries(preprocessorsInfo) as [key, preprocessor]}
              <button
                class="w-full text-left p-3 rounded border {selectedPreprocessor === key ? 'bg-blue-100 dark:bg-blue-900 border-blue-300' : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600'} hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                on:click={() => selectedPreprocessor = key}
              >
                <div class="font-medium">{preprocessor.name}</div>
                <div class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  {preprocessor.description.substring(0, 80)}...
                </div>
              </button>
            {/each}
          </div>

          <!-- Middle Column: Details -->
          <div class="space-y-4">
            {#if selectedPreprocessor && preprocessorsInfo[selectedPreprocessor]}
              {@const preprocessor = preprocessorsInfo[selectedPreprocessor]}
              
              <div>
                <h3 class="text-xl font-semibold">{preprocessor.name}</h3>
                <p class="text-gray-600 dark:text-gray-400 mt-2">{preprocessor.description}</p>
              </div>

              <div>
                <h4 class="font-medium mb-2">Requirements</h4>
                <ul class="text-sm text-gray-600 dark:text-gray-400">
                  {#each preprocessor.requirements as req}
                    <li>• {req}</li>
                  {/each}
                </ul>
              </div>

              <div>
                <h4 class="font-medium mb-2">Use Cases</h4>
                <div class="flex flex-wrap gap-2">
                  {#each preprocessor.use_cases as useCase}
                    <span class="px-2 py-1 bg-gray-200 dark:bg-gray-600 rounded text-sm">{useCase}</span>
                  {/each}
                </div>
              </div>

              {#if preprocessor.setup_notes}
                <div class="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded p-3">
                  <h4 class="font-medium text-yellow-800 dark:text-yellow-200">Setup Notes</h4>
                  <p class="text-sm text-yellow-700 dark:text-yellow-300 mt-1">{preprocessor.setup_notes}</p>
                </div>
              {/if}

              <div>
                <h4 class="font-medium mb-3">Parameters</h4>
                <div class="space-y-3">
                  {#each Object.entries(preprocessor.parameters) as [paramName, param]}
                    <div class="border border-gray-200 dark:border-gray-600 rounded p-3">
                      <div class="flex justify-between items-start mb-2">
                        <span class="font-medium text-sm">{paramName}</span>
                        <span class="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">{param.type}</span>
                      </div>
                      <p class="text-sm text-gray-600 dark:text-gray-400 mb-2">{param.description}</p>
                      <div class="text-xs text-gray-500 dark:text-gray-500">
                        Default: {param.default}
                        {#if param.range}
                          | Range: {param.range[0]}-{param.range[1]}
                        {/if}
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            {:else}
              <div class="text-center text-gray-500 dark:text-gray-400 py-8">
                <p>Select a preprocessor to view details</p>
              </div>
            {/if}
          </div>

          <!-- Right Column: Configuration Generator -->
          <div class="space-y-4">
            <h3 class="text-lg font-semibold">Configuration Generator</h3>
            
            {#if selectedPreprocessor}
              <div class="space-y-3">
                <div class="flex items-center gap-2">
                  <input 
                    type="checkbox" 
                    id="fullTemplate" 
                    bind:checked={showFullTemplate}
                    class="rounded"
                  >
                  <label for="fullTemplate" class="text-sm">Show full configuration</label>
                </div>

                <div class="bg-gray-900 text-gray-100 p-4 rounded text-sm font-mono overflow-x-auto">
                  <pre>{generateConfig(selectedPreprocessor)}</pre>
                </div>

                <div class="flex gap-2">
                  <Button 
                    on:click={() => copyToClipboard(generateConfig(selectedPreprocessor))}
                    classList="text-sm"
                  >
                    Copy Configuration
                  </Button>
                </div>
              </div>
            {:else}
              <div class="text-center text-gray-500 dark:text-gray-400 py-4">
                <p>Select a preprocessor to generate configuration</p>
              </div>
            {/if}

            <!-- Quick Start Guide -->
            <div class="border-t border-gray-200 dark:border-gray-600 pt-4 mt-6">
              <h4 class="font-medium mb-2">Quick Start</h4>
              <ol class="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>1. Select a preprocessor above</li>
                <li>2. Copy the generated configuration</li>
                <li>3. Save as a .yaml file</li>
                <li>4. Upload using the "Load YAML Config" button</li>
                <li>5. Start streaming to activate</li>
              </ol>
            </div>
          </div>
        </div>
      {:else}
        <div class="text-center py-8">
          <p>Loading preprocessor documentation...</p>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  pre {
    white-space: pre-wrap;
    word-wrap: break-word;
  }
</style> 