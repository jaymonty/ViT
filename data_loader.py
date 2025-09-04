# data_loader.py

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split, Dataset
from clearml import Dataset as ClearMLDataset
from dataset import RFSpectrogramDataset
from augmentations import RFSimCLRAug
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# Notes (data loading, class handling, determinism)
# -----------------------------------------------------------------------------
# • Class merges & drops:
#   - `merge_map` normalizes alias labels (e.g., 'p4','p4L' → 'Phantom') *before*
#     re-encoding. 
#   - `classes_to_exclude` filters by string name after merge. This happens prior
#     to LabelEncoder.fit, so the new label space only contains what I keep.
#
# • Label encoding flow:
#   - Original dataset ships integer labels + a stored LabelEncoder (npz with classes).
#   - I map ints to original string names, apply merges, drop excluded, then fit a fresh
#     LabelEncoder on the resulting names. `new_classes` is the canonical order for this run.
#
# • Dual-view wrapper (SimCLR):
#   - Base dataset may yield (spec, persistence, label) or (x, label). I normalize both
#     paths to a single 6-channel tensor by concatenating along channel dim.
#   - If a transform is provided, I create two *independent* views:
#       • If the transform has a persistent torch.Generator, I clone it (g1/g2) to get
#         reproducible but different views per sample.
#       • Otherwise I just call the transform twice.
#   - Return signature: (view1, view2, label). When `augment=False`, return (x, x, label)
#     so the downstream code doesn’t need a separate branch.
#
# • Determinism knobs:
#   - Pass `generator=` to DataLoader so shuffling is reproducible.
#   - Use `worker_init_fn` to seed NumPy/Python inside workers from the worker seed.
#   - For eval loaders, disable augmentations (augment=False) and use a distinct generator
#     to avoid coupling eval order to train RNG state.
#
# • Splits:
#   - I use a simple `random_split` with `val_split`. This is *not* stratified. If I see
#     class imbalance hurting probes, switch to a stratified split upstream and pass the
#     resulting Subsets into DataLoader.
#
# • DataLoader settings:
#   - `pin_memory=True` for faster host GPU transfers.
#   - `num_workers`: tune per machine.
#   - Consider `drop_last=True` for training if the final small batch degrades contrastive
#     statistics at high batch sizes. (I leave it default here.)
#
# • ClearML dataset materialization:
#   - `load_dataset_files` fetches the local path via ClearML and loads known files.
#   - Uses `allow_pickle=True` because label_encoder/metadata often contain objects.
#
# • Shapes & dtypes:
#   - The base dataset expects spectrograms/persistence as numpy arrays shaped.
#
# =============================================================================

# merge_map = {
#     'Phantom_drone': 'Phantom',
#     'p4':            'Phantom',
#     'p4L':           'Phantom',
#     'Background_RF': 'ambient'
# }

#=========== Use this for typical training
merge_map = {
    #'Phantom_drone': 'Phantom',
    'p4':            'Phantom',
    'p4L':           'Phantom',
    #'Background_RF': 'ambient'
}

# For ZERO merges
# merge_map = {
#     #'Phantom_drone': 'Phantom',
#     # 'p4':            'Phantom',
#     # 'p4L':           'Phantom',
#     #'Background_RF': 'ambient'
# }

# === EXCLUDE UNWANTED CLASSES HERE ===
classes_to_exclude = [
    'Background_RF', 
    'Phantom_drone',
] 

# Wrap a base dataset to produce two augmented views for SimCLR,
# handling datasets that return either (spec, persistence, label) or (x, label).
class SimCLRDualViewDataset(Dataset):

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        # Handle different return signatures
        if isinstance(item, tuple) and len(item) == 3:
            spec, persistence, label = item
            # convert to tensor if needed
            if not isinstance(spec, torch.Tensor):
                spec = torch.from_numpy(spec)
            if not isinstance(persistence, torch.Tensor):
                persistence = torch.from_numpy(persistence)
            x = torch.cat([spec, persistence], dim=0)
        elif isinstance(item, tuple) and len(item) == 2:
            x, label = item
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
        else:
            raise ValueError(f"Unsupported base_dataset return type: {type(item)} with len {len(item)}")

        if self.transform:
            if hasattr(self.transform, "generator") and self.transform.generator is not None:
                g1 = self.transform.generator.clone()
                g2 = self.transform.generator.clone()
                view1 = self.transform(x, generator=g1)
                view2 = self.transform(x, generator=g2)
            else:
                view1 = self.transform(x)
                view2 = self.transform(x)
            return view1, view2, label
        
        else:
            return x, x, label

def load_numpy_dataset(data_dict, resize_to=256, augment=True, val_split=0.1, generator=None):
    
    spectrograms = data_dict['spectrograms']
    persistence  = data_dict['persistence_spectra']
    # labels
    label_indices = data_dict.get('labels', None)

    if label_indices is not None:
        # --- load the original LabelEncoder to get name lookup ---
        orig_le = data_dict['label_encoder']  
        orig_classes = orig_le['classes']   

        # turn each integer to its string name
        orig_names = [orig_classes[idx] for idx in label_indices]

        # apply merge_map on the **strings**
        merged_names = [merge_map.get(name, name) for name in orig_names]
        
        #================ TO FILTER OUT CLASSES =======
        # Filter out samples with excluded class names
        filtered_indices = [i for i, name in enumerate(merged_names) if name not in classes_to_exclude]

        # Apply the filtering to all datasets
        spectrograms = spectrograms[filtered_indices]
        persistence  = persistence[filtered_indices]
        merged_names = [merged_names[i] for i in filtered_indices]
        label_indices = [label_indices[i] for i in filtered_indices]  
        #===============================================

        # fit LabelEncoder on merged names
        le = LabelEncoder().fit(merged_names)
        labels = le.transform(merged_names)
        new_classes = le.classes_.tolist()  
    else:
        labels = None
        new_classes = None

    # build the base dataset & dual‐view .
    base_dataset = RFSpectrogramDataset(
        spectrograms=spectrograms,
        persistence=persistence,
        labels=labels,
        transform=None,
        resize_to=resize_to,
    )
    
    # wrap dual views
    # transform = RFSimCLRAug(image_size=resize_to) if augment else None
    transform = RFSimCLRAug(image_size=resize_to, generator=generator) if augment else None    # For repeatability
    
    full_dataset = SimCLRDualViewDataset(base_dataset, transform=transform)

    # split out train/val
    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    return train_ds, val_ds, new_classes

# Added classes from above to handle classes merged
def get_dataloaders(data_dict, batch_size=64, resize_to=256, augment=True, val_split=0.1, num_workers=8, worker_init_fn=None, generator=None ):
    train_dataset, val_dataset, classes = load_numpy_dataset(
        data_dict=data_dict,
        resize_to=resize_to,
        augment=augment,
        val_split=val_split
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn, 
        generator=generator               
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,   
        generator=generator  
    )

    return train_loader, val_loader, classes

def load_dataset_files(dataset_id):
    dataset = ClearMLDataset.get(dataset_id=dataset_id)
    local_copy = dataset.get_local_copy()

    data = {}
    files_to_load = [
        "spectrograms.npy",
        "persistence_spectra.npy",
        "labels.npy",
        "metadata.npz",
        "label_encoder.npz"
    ]

    for filename in files_to_load:
        file_path = os.path.join(local_copy, filename)
        if os.path.exists(file_path):
            print(f"Loading {filename}")
            data[filename.replace('.npy', '').replace('.npz', '')] = np.load(file_path, allow_pickle=True)
        else:
            print(f"{filename} not found")

    return data


