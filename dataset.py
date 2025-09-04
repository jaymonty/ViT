# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
from PIL import Image

# =============================================================================
# Notes (dataset construction, image conversion, transforms)
# -----------------------------------------------------------------------------
# • Purpose:
#   - Take two (H, W) arrays — spectrogram & persistence — and return either:
#       (a) two SimCLR views stacked to [6, H, W] with label (train), or
#       (b) a single stacked [6, H, W] with label (eval).
#
# • PIL + dtype caveat:
#   - `_to_rgb_tensor` uses `Image.fromarray(arr).convert("RGB")` to `ToTensor()`.
#   - `Image.fromarray` expects uint8/uint16/etc. If `arr` is float (e.g., dB),
#     PIL will try to interpret it but the RGB conversion may be lossy/unexpected.
#
# • Channel semantics:
#   - I’m treating each input (spec, pers) as an RGB image (3 channels)
#     for compatibility with common image augs, then concatenating to 6 channels:
#     [spec_RGB(3), pers_RGB(3)] to [6, H, W].
#   - Keep this consistent across training/eval; if I later switch to single-channel
#     inputs, I must also change normalization stats and augmentation assumptions.
#
# • Resizing:
#   - I resize each input to RGB tensor first (bicubic) before stacking.
#     This keeps both branches identically sized; avoids mismatched shapes.
#
# • Transform application:
#   - With `transform` present: apply the RF aug pipeline to each image *independently*,
#     then stack into 6 channels for each view.
#     This guarantees the two modalities aren’t forced to share identical stochastic ops.
#   - Without `transform` (eval): simply stack once and return `[6, H, W]`.
#
# • Labels:
#   - Stored as numpy array; I cast to `int` on return. Ensure label encoding upstream
#     matches the class order I log (see data_loader `new_classes`).
#
# • Determinism:
#   - Determinism is driven by the transform’s generator (if provided) and the DataLoader
#     worker seeding. This dataset itself is stateless.
#
# =============================================================================

class RFSpectrogramDataset(Dataset):
    def __init__(self, spectrograms, persistence, labels=None, transform=None, resize_to=256):
        
        assert spectrograms.shape[0] == persistence.shape[0]
        self.spectrograms = spectrograms
        self.persistence = persistence
        self.labels = labels if labels is not None else np.zeros(len(spectrograms))
        self.transform = transform
        self.resize_to = resize_to

        # Resizing transform applied to each 1-channel image before stacking
        self.resize = T.Resize((resize_to, resize_to), interpolation=T.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.spectrograms)

    def _to_rgb_tensor(self, arr):
        
        img = Image.fromarray(arr).convert("RGB")
        img = self.resize(img)
        tensor = T.ToTensor()(img)  # [3, H, W], float32 in [0, 1]
        return tensor

    # Allows for rf augs to both images not just 1
    def __getitem__(self, idx):
        # Convert both inputs to 3-channel RGB tensors
        spec = self._to_rgb_tensor(self.spectrograms[idx])
        pers = self._to_rgb_tensor(self.persistence[idx])

        if self.transform:
            # Apply transform to each image individually
            spec_view1 = self.transform(spec)
            pers_view1 = self.transform(pers)
            spec_view2 = self.transform(spec)
            pers_view2 = self.transform(pers)

            # Stack after transforming → [6, H, W]
            view1 = torch.cat([spec_view1, pers_view1], dim=0)
            view2 = torch.cat([spec_view2, pers_view2], dim=0)
            return view1, view2, int(self.labels[idx])

        else:
            stacked = torch.cat([spec, pers], dim=0)
            return stacked, int(self.labels[idx])

