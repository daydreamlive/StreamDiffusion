import torch
import torch.nn.functional as F
from typing import Optional, Any, List
from .base import BasePreprocessor


class LatentFrequencyProcessor(BasePreprocessor):
    """
    Latent domain frequency manipulation processor
    
    Applies frequency domain filtering to latent representations, allowing users to
    control the balance between low frequencies (overall structure/composition),
    mid frequencies (textures/patterns), and high frequencies (fine details/edges).
    
    This operates in the frequency domain of latent space, which is fundamentally
    different from pixel-space frequency filtering and much more computationally efficient.
    
    Focuses solely on frequency boost/attenuation without temporal or noise effects.
    """
    
    @classmethod
    def get_preprocessor_metadata(cls):
        return {
            "display_name": "Latent Frequency Control",
            "description": "Controls frequency components in latent space for detail, texture, and structure adjustment. More efficient than pixel-space frequency filtering.",
            "parameters": {
                "low_freq_multiplier": {
                    "type": "float",
                    "default": 1.0,
                    "range": [0.0, 3.0],
                    "step": 0.01,
                    "description": "Low frequency multiplier (1.0=neutral, >1.0=boost structure, <1.0=reduce structure)"
                },
                "mid_freq_multiplier": {
                    "type": "float", 
                    "default": 1.0,
                    "range": [0.0, 3.0],
                    "step": 0.01,
                    "description": "Mid frequency multiplier (1.0=neutral, >1.0=boost textures, <1.0=reduce textures)"
                },
                "high_freq_multiplier": {
                    "type": "float",
                    "default": 1.0, 
                    "range": [0.0, 3.0],
                    "step": 0.01,
                    "description": "High frequency multiplier (1.0=neutral, >1.0=boost details, <1.0=reduce details)"
                },
                "low_mid_cutoff": {
                    "type": "float",
                    "default": 0.3,
                    "range": [0.1, 0.9], 
                    "step": 0.05,
                    "description": "Boundary between low and mid frequencies (lower = more in low band)"
                },
                "mid_high_cutoff": {
                    "type": "float",
                    "default": 0.7,
                    "range": [0.1, 0.9],
                    "step": 0.05, 
                    "description": "Boundary between mid and high frequencies (higher = more in mid band)"
                }
            },
            "use_cases": [
                "Detail enhancement/reduction", 
                "Texture control", 
                "Artistic stylization",
                "Composition refinement"
            ]
        }
    
    def __init__(self, 
                 low_freq_multiplier: float = 1.0,
                 mid_freq_multiplier: float = 1.0,
                 high_freq_multiplier: float = 1.0,
                 low_mid_cutoff: float = 0.3,
                 mid_high_cutoff: float = 0.7,
                 **kwargs):
        """
        Initialize latent frequency processor
        
        Args:
            low_freq_multiplier: Multiplier for low frequency components (1.0=neutral, 0.0-3.0)
            mid_freq_multiplier: Multiplier for mid frequency components (1.0=neutral, 0.0-3.0)
            high_freq_multiplier: Multiplier for high frequency components (1.0=neutral, 0.0-3.0)
            low_mid_cutoff: Frequency boundary between low and mid bands (0.1-0.9)
            mid_high_cutoff: Frequency boundary between mid and high bands (0.1-0.9)
            **kwargs: Additional parameters passed to BasePreprocessor
        """
        super().__init__(
            low_freq_multiplier=low_freq_multiplier,
            mid_freq_multiplier=mid_freq_multiplier,
            high_freq_multiplier=high_freq_multiplier,
            low_mid_cutoff=low_mid_cutoff,
            mid_high_cutoff=mid_high_cutoff,
            **kwargs
        )
        
        # Clamp parameters to safe ranges
        self.low_freq_multiplier = max(0.0, min(3.0, low_freq_multiplier))
        self.mid_freq_multiplier = max(0.0, min(3.0, mid_freq_multiplier))
        self.high_freq_multiplier = max(0.0, min(3.0, high_freq_multiplier))
        
        # Ensure cutoff points are valid
        self.low_mid_cutoff = max(0.1, min(0.9, low_mid_cutoff))
        self.mid_high_cutoff = max(0.1, min(0.9, mid_high_cutoff))
        
        # Ensure mid_high_cutoff > low_mid_cutoff
        if self.mid_high_cutoff <= self.low_mid_cutoff:
            self.mid_high_cutoff = min(0.9, self.low_mid_cutoff + 0.2)
    
    def _create_frequency_mask(self, shape: tuple, cutoff_low: float, cutoff_high: float) -> torch.Tensor:
        """
        Create a frequency domain mask for the given frequency band
        
        Args:
            shape: Shape of the tensor (H, W)
            cutoff_low: Low frequency cutoff (0.0-1.0)
            cutoff_high: High frequency cutoff (0.0-1.0)
            
        Returns:
            Frequency mask tensor
        """
        h, w = shape
        
        # Create frequency coordinates (DC at center after fftshift)
        freq_y = torch.fft.fftfreq(h, device=self.device).view(-1, 1)
        freq_x = torch.fft.fftfreq(w, device=self.device).view(1, -1)
        
        # Calculate distance from DC (0,0) in frequency domain
        freq_radius = torch.sqrt(freq_y**2 + freq_x**2)
        
        # Normalize to [0, 1] range - maximum frequency is 0.5 (Nyquist)
        freq_radius_norm = freq_radius / 0.5
        
        # Create band-pass mask
        if cutoff_low == 0.0 and cutoff_high == 1.0:
            # Full spectrum
            return torch.ones_like(freq_radius_norm)
        elif cutoff_low == 0.0:
            # Low-pass filter
            return (freq_radius_norm <= cutoff_high).float()
        elif cutoff_high == 1.0:
            # High-pass filter
            return (freq_radius_norm >= cutoff_low).float()
        else:
            # Band-pass filter
            return ((freq_radius_norm >= cutoff_low) & (freq_radius_norm <= cutoff_high)).float()
    
    def _apply_frequency_filtering(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency domain filtering to the latent tensor
        
        Args:
            tensor: Input latent tensor [B, C, H, W]
            
        Returns:
            Frequency-filtered latent tensor
        """
        batch_size, channels, height, width = tensor.shape
        
        # Process each item in batch
        processed_batch = []
        
        for b in range(batch_size):
            processed_channels = []
            
            for c in range(channels):
                channel_data = tensor[b, c]  # [H, W]
                
                # Apply FFT
                fft_data = torch.fft.fft2(channel_data)
                
                # Create frequency masks for the three bands
                low_mask = self._create_frequency_mask(
                    (height, width), 0.0, self.low_mid_cutoff
                )
                mid_mask = self._create_frequency_mask(
                    (height, width), self.low_mid_cutoff, self.mid_high_cutoff
                )
                high_mask = self._create_frequency_mask(
                    (height, width), self.mid_high_cutoff, 1.0
                )
                
                # Extract and multiply frequency components
                low_freq = fft_data * low_mask * self.low_freq_multiplier
                mid_freq = fft_data * mid_mask * self.mid_freq_multiplier
                high_freq = fft_data * high_mask * self.high_freq_multiplier
                
                # Combine frequency components
                combined_fft = low_freq + mid_freq + high_freq
                
                # Convert back to spatial domain
                processed_channel = torch.fft.ifft2(combined_fft).real
                processed_channels.append(processed_channel)
            
            # Stack channels back together
            processed_item = torch.stack(processed_channels, dim=0)  # [C, H, W]
            processed_batch.append(processed_item)
        
        # Stack batch back together
        result = torch.stack(processed_batch, dim=0)  # [B, C, H, W]
        return result
    
    def validate_tensor_input(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate latent tensor input - preserve batch dimensions for latent processing
        
        Args:
            latent_tensor: Input latent tensor in format [B, C, H/8, W/8]
            
        Returns:
            Validated latent tensor with preserved batch dimension
        """
        # For latent processing, we want to preserve the batch dimension
        # Only ensure correct device and dtype
        latent_tensor = latent_tensor.to(device=self.device, dtype=self.dtype)
        return latent_tensor
        
    def _ensure_target_size_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Override base class resize logic - latent tensors should NOT be resized to image dimensions
        
        For latent domain processing, we want to preserve the latent space dimensions,
        not resize to image target dimensions like image-domain processors.
        """
        # For latent frequency processing, just return the tensor as-is without any resizing
        return tensor
    
    def _process_core(self, image):
        """
        For latent frequency processing, we don't process PIL images directly.
        This method should not be called in normal latent preprocessing workflows.
        """
        raise NotImplementedError(
            "LatentFrequencyProcessor is designed for latent domain processing. "
            "Use _process_tensor_core or process_tensor for latent tensors."
        )
    
    def _process_tensor_core(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process latent tensor with frequency domain filtering
        
        Args:
            tensor: Current input latent tensor in format [B, C, H/8, W/8]
            
        Returns:
            Frequency-filtered latent tensor
        """
        # Apply frequency filtering
        filtered_tensor = self._apply_frequency_filtering(tensor)
        
        # Apply safety clamping to prevent extreme values
        filtered_tensor = torch.clamp(filtered_tensor, min=-10.0, max=10.0)
        
        # Ensure correct device and dtype
        filtered_tensor = filtered_tensor.to(device=self.device, dtype=self.dtype)
        
        return filtered_tensor

