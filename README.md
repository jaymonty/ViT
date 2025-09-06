# ViT


# RF Fingerprinting with **SimCLR + ViT** (+ MAE)

*Self-supervised embeddings for drone RF emissions from passive spectrum data.*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

---

## Table of Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Repo Structure](#repo-structure)
- [Data Format](#data-format)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Augmentations (RF-aware)](#augmentations-rf-aware)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Results (placeholders)](#results-placeholders)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License](#license)

---

## Overview
This repo trains a compact **Vision Transformer (ViT)** with **SimCLR** on **6-channel inputs** (RGB **spectrogram** + RGB **persistence**). A lightweight **MAE** head optionally regularizes early training. The pipeline is **deterministic** (seedable) and fully **instrumented** with ClearML.

> [!TIP]
> Six channels = 3 for spectrogram (RGB) + 3 for persistence (RGB) → tensor shape **[6, H, W]**.

---

## Highlights
- **6-channel aware** data path (spectrogram RGB ×3 + persistence RGB ×3).
- **Deterministic** augmentations: per-call `torch.Generator` reseeds Python/NumPy/Torch.
- **RF-aware augs** incl. **SpecAugment** (time/freq masks) on tensors.
- **Compact ViT** (`D=128`, `L=6`, `H=4`) — edge-friendly; projector dropped at inference.
- **MAE** regularizer (masked pixel MSE) you can anneal to **0** mid-training.
- Clear **probes & metrics**: linear/MLP, Silhouette/DB, UMAP visualization.

---

## Repo Structure
```text
augmentations.py      # RF-aware augs (seedable) + SpecAugment time/freq masks
data_loader.py        # ClearML dataset loader, dual-view SimCLR wrapper, DataLoaders
dataset.py            # (H,W) -> RGB tensors; returns [6,H,W] (train: two views; eval: one)
transformer_model.py  # PatchEmbed + ViTEncoder (CLS) + ProjectionHead + MAE module
utils.py              # CFG (hyperparams), determinism helpers
# training notebook / script  # main loop, probes, UMAP, ClearML logging









#=======================







RF Fingerprinting with SimCLR + ViT (+ MAE)

Learn robust embeddings for drone RF emissions from passive spectrum data—no labels required.
This repo trains a compact Vision Transformer (ViT) with SimCLR on 6-channel inputs (RGB spectrogram + RGB persistence). A lightweight MAE head optionally regularizes early training. Everything is deterministic (seedable) and instrumented with ClearML.

<p align="center"><i>Self-supervised representations that cluster by device type and hold up across noise, gain, and minor time/frequency shifts.</i></p>
TL;DR

Data: two grayscale planes per sample (spectrogram, persistence) → each converted to RGB and stacked ⇒ [6, H, W].

Augs: RandomResizedCrop, flips, jitter, blur, random erasing + SpecAugment (time/freq masks). All seedable.

Model: ViT encoder (D=128, L=6, H=4) with projection MLP (BN-ReLU-BN). Optional MAE reconstructs masked pixels.

Training: SimCLR on two augmented views; optional MAE on one view. Early stop & temperature tweaks to avoid collapse.

Eval: Linear/MLP probes, Silhouette/DB, UMAP. ClearML logs + best-F1 checkpoints.

Highlights

6-channel aware pipeline (spectrogram RGB ×3 + persistence RGB ×3).

Deterministic data path: per-call torch.Generator and seeded torchvision ops.

RF-aware augmentations: SpecAugment time/freq masks after normalization.

Compact ViT for edge-friendliness; projector dropped at inference.

MAE regularizer (masked pixel MSE) you can anneal to 0 mid-training.

Repo Structure
augmentations.py     # RF-aware augs (seedable) + SpecAugment
data_loader.py       # ClearML dataset loader, dual-view SimCLR dataset, DataLoaders
dataset.py           # (H,W) → RGB → Tensor; returns [6,H,W] (train: two views; eval: one)
transformer_model.py # PatchEmbed + ViTEncoder + ProjectionHead + MAE module
utils.py             # CFG + global hyperparams & determinism helpers
# training notebook / script  # main loop, probes, eval, ClearML logging

Data Format

Expected files (from ClearML dataset or local folder):
spectrograms.npy          # shape: (N, H, W), grayscale
persistence_spectra.npy   # shape: (N, H, W), grayscale
labels.npy                # shape: (N,), optional
label_encoder.npz         # contains 'classes' for original label names
metadata.npz              # optional extra metadata

During loading we:
Merge label aliases (p4/p4L → Phantom),
Exclude undesired classes (e.g., Background_RF, Phantom_drone),
Re-encode labels to a fresh canonical set (reported as class_names).

Configuration (edit in utils.py)

Key knobs:
Image/Patch: IMAGE_SIZE=252, PATCH_SIZE=14 → N=18×18=324 tokens.
Model: EMBED_DIM=128, DEPTH=6, NUM_HEADS=4.
Projector: HIDDEN_DIM=512, PROJECTION_DIM=128.
Training: EPOCHS, BATCH_SIZE, NUM_WORKERS.
SimCLR: TEMPERATURE=0.015.
MAE blend: LAMBDA_MAE=0.01 (set to 0 to disable).
LRs: lower encoder LR, higher projector LR (AdamW).

Augmentations (RF-aware)
Implemented in augmentations.py:
Spatial/appearance: RandomResizedCrop, H/V flips*, brightness/contrast jitter, Gaussian blur, Random Erasing, Normalize.
SpecAugment: time_mask (columns), freq_mask (rows), ~10% width/height ×2.
Determinism: per-call torch.Generator reseeds Python/NumPy/Torch; dual-view dataset clones the generator so views are reproducible-but-different.

Evaluation
Linear probe (LogReg) & MLP probe on clean eval loaders (no augs).
Clustering: Silhouette (↑), Davies–Bouldin (↓).
Visualization: UMAP (cosine) logged to ClearML.
Collapse check: z_var_mean on L2-normalized CLS embeddings (report scaled).

Reproducibility
Global seeding (set_seed): Python/NumPy/Torch; deterministic algorithms; TF32/Flash SDP off.
DataLoader generator + worker_init_fn to seed workers.
Aug pipeline accepts generator and reseeds torchvision ops per call.

Results:


Customize
Turn MAE off: set LAMBDA_MAE=0.0.
Soften augs: reduce mask counts/width, lower erase prob, disable V-flip.
Sharpen contrastive: lower TEMPERATURE slightly (watch for collapse).
Scale model: bump EMBED_DIM/DEPTH; track attention cost ∝ (N+1)^2.
