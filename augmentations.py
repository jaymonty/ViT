# augmentations.py

import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T

# =============================================================================
# Notes on augmentation 
# -----------------------------------------------------------------------------
# • Determinism: most ops here can be made repeatable by passing a torch.Generator.
#   RFSimCLRAug.__call__ pulls a seed from that generator and re-seeds Python/NumPy/Torch
#   so torchvision RNG is fixed for that call. In SimCLR, I clone the generator to get
#   two *different* but reproducible views per sample.
#
# • SpecAugment: time_mask zeros columns (time), freq_mask zeros rows (freq).
#   I use ~10% of width/height, applied twice. If the model overfits to stripes or
#   narrowband cues, increase width/count; if too destructive, dial them down.
#   These are in-place edits on the input tensor by design.
#
# • Flips:
#   - Horizontal flip == time reversal of the *image*. It’s a strong regularizer and
#     generally fine for magnitude spectrograms.
#   - Vertical flip == frequency inversion. This is *aggressive*. If label identity
#     depends on absolute frequency placement, turn V-flip off or lower its prob.
#
# • Photometric jitter (brightness/contrast): simulates front-end gain and dynamic-range
#   changes. Jitter is applied around the per-channel mean so DC level is preserved.
#
# • Noise: small Gaussian (std=0.05 after ToTensor/Normalize) to encourage local invariance.
#   Bump slightly if embeddings collapse; reduce if I see oversmoothing.
#
# • Random erasing: acts like spatial dropout. With value='random' it fills with random
#   colors; switch to a constant (e.g., 0.0) if I want a pure “mask-out” behavior.
#
# • Normalization: using mean/std = 0.5 assumes inputs are roughly in [0,1].
#   If spectrograms come from dB ranges or different scaling, consider computing dataset
#   channel-wise means/stds and swap them in for tighter training.
#
# • Channels: pipeline is 6-channel aware (RGB spec + RGB persistence). Keep transforms
#   channel-consistent.
#
# • Performance: masks/erasing mutate tensors in-place to avoid extra allocs.
#   If I need the original tensor downstream, clone before masking.
#
# • Knobs I might tune later:
#   - blur sigma range, erase p/scale/ratio, jitter strengths
#   - number/size of SpecAugment masks
#   - flip probabilities (especially vertical)
#
# =============================================================================

def time_mask(x: torch.Tensor, mask_width: int = 30, num_masks: int = 1, generator=None) -> torch.Tensor:
    C, H, W = x.shape
    for _ in range(num_masks):
        if W <= mask_width:
            continue
       
        t0 = torch.randint(0, W - mask_width, (1,), generator=generator, device=x.device).item()
        x[:, :, t0:t0 + mask_width] = 0
    return x

def freq_mask(x: torch.Tensor, mask_height: int = 20, num_masks: int = 1, generator=None) -> torch.Tensor:
    C, H, W = x.shape
    for _ in range(num_masks):
        if H <= mask_height:
            continue
        f0 = torch.randint(0, H - mask_height, (1,), generator=generator, device=x.device).item()
        x[:, f0:f0 + mask_height, :] = 0
    return x

def add_noise(x, std=0.05):
    return x + torch.randn_like(x) * std

class RandomChannelJitter(nn.Module):
    
    def __init__(self, brightness: float = 0.4, contrast: float = 0.4):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, x: torch.Tensor, generator=None) -> torch.Tensor:
        b = torch.empty(1).uniform_(1 - self.brightness, 1 + self.brightness).item() if generator is None \
            else torch.rand(1, generator=generator).item() * (2*self.brightness) + (1 - self.brightness)
        x = x * b

        mean = x.mean(dim=(1, 2), keepdim=True)
        c = torch.empty(1).uniform_(1 - self.contrast, 1 + self.contrast).item() if generator is None \
            else torch.rand(1, generator=generator).item() * (2*self.contrast) + (1 - self.contrast)
        x = (x - mean) * c + mean
        return x
    
# Strong augmentation pipeline for RF spectrograms in SimCLR.
# Supports 6-channel inputs.
class RFSimCLRAug:

    # def __init__(self, image_size: int = 256):
    def __init__(self, image_size: int = 256, generator=None):
        self.generator = generator
        self.crop = T.RandomResizedCrop(
            size=image_size,
            scale=(0.2, 1.0),
            interpolation=T.InterpolationMode.BICUBIC
        )
        self.hflip = T.RandomHorizontalFlip(p=0.5)
        self.vflip = T.RandomVerticalFlip(p=0.5)
        self.jitter = RandomChannelJitter(brightness=0.4, contrast=0.4)
        self.blur = T.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0))
        self.erase = T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
        self.norm = T.Normalize(mean=[0.5] * 6, std=[0.5] * 6)

    def __call__(self, x: torch.Tensor, generator=None) -> torch.Tensor:

        g = generator if generator is not None else self.generator

        if g is not None:
            
            seed = int(torch.randint(0, 2**32 - 1, (1,), generator=g).item())
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        x = self.crop(x)
        x = self.hflip(x)
        x = self.vflip(x)
        x = add_noise(x, std=0.05)

        # channel jitter
        x = self.jitter(x, generator=g)

        # blur and erasing
        x = self.blur(x)
        x = self.erase(x)

        # normalize
        x = self.norm(x)

        # SpecAugment masks
        x = time_mask(x, mask_width=int(0.1 * x.shape[-1]), num_masks=2, generator=g)
        x = freq_mask(x, mask_height=int(0.1 * x.shape[-2]), num_masks=2, generator=g)

        return x



