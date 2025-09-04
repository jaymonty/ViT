# Utils.py

import torch

# =============================================================================
# Notes (training config, determinism, schedules, logging)
# -----------------------------------------------------------------------------
# • Purpose:
#   - Centralize all training knobs in one place (CFG) so I can wire configs into
#     ClearML, the model, and the loop without hunting through the notebook.
#
# • Core training params:
#   - EPOCHS/BATCH_SIZE/NUM_WORKERS are here for a single source of truth. Workers=10
#     is a good throughput default.
#
# • Contrastive/MAE mix:
#   - `TEMPERATURE` (τ) controls SimCLR softness; lower τ = sharper distribution →
#     stronger gradients on hard negatives. I start at 0.015.
#   - `LAMBDA_MAE` blends MAE recon loss into the total. In the loop I may drop it
#     to 0 when patience stalls to focus on contrastive geometry.
#
# • Optimizer + weight decay:
#   - `BASE_LR` drives per-group LRs:
#       ENCODER_LR   = 0.1 * BASE_LR  (slow backbone updates)
#       PROJECTOR_LR = 1.0 * BASE_LR  (let the head move fast)
#     Weight decay on encoder only (common for SimCLR). Projector decay = 0.
#
# • LR schedule:
#   - Cosine anneal down to `ETA_MIN`. I also expose `WARMUP_FRAC` → derive
#     `WARMUP_EPOCHS` (used if I add a warmup scheduler).
#
# • Model geometry:
#   - `IMAGE_SIZE` and `PATCH_SIZE` define the patch grid N=(H/P)*(W/P). I’m using
#     252 // 14 = 18 → N=324 (vs the “classic” 256//16=16 → N0=256). I log an
#     attention-cost multiplier vs the 256/16 baseline: ((N+1)/(N0+1))^2.
#   - Keep EMBED_DIM/NUM_HEADS/DEPTH modest to avoid overfitting 22k images.
#
# • CFG wrapper:
#   - Duplicates top-level constants into a namespaced object so I can pass a single
#     `CFG` around. If I change a top-level constant, the class picks it up once at
#     import time; re-instantiate if I change values at runtime.
#
# • (Commented) dynamic temperature / zvm thresholds:
#   - I left a skeleton for temperature auto-tuning based on z_var_mean with EMA and
#     dimension-aware targets (1/d). If I revive it, expose the knobs here and keep
#     the scaling (1e4) consistent with what I log to ClearML.
#
# • Seeding & determinism hints (used in the main script):
#   - Set PYTHONHASHSEED, seed Python/NumPy/Torch, disable cudnn benchmark,
#     set deterministic algorithms, and disable Flash/Math SDP for attention.
#   - DataLoader: pass a `generator` plus `worker_init_fn` to seed each worker’s
#     NumPy/Python RNG. This gives me “reproducible-but-different” augmentations
#     when I clone a base generator for SimCLR view1/view2.
#
# • ClearML logging:
#   - I connect `task_params` including class names, model dims, and dataset id.
#     I also log the token budget (patch grid + attention multiplier) so scaling
#     choices are traceable across runs.
#
# =============================================================================

# === Device Config ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Training Params ===
EPOCHS = 150
BATCH_SIZE = 256 
NUM_WORKERS = 10 # zero is deterministic

# === SimCLR Params ===
TEMPERATURE =  0.015

# === MAE Weight for Joint Loss ===
LAMBDA_MAE = 0.01 

# === Optimizer Params ===
BASE_LR = 1e-2           # Starting LR
ETA_MIN = 1e-6           # Cosine LR decay target
WEIGHT_DECAY = 1e-4      # AdamW regularization
# ---
ENCODER_LR   = BASE_LR * 0.1
PROJECTOR_LR = BASE_LR


# === Warmup ===
WARMUP_FRAC = 0.05
WARMUP_EPOCHS = int(EPOCHS * WARMUP_FRAC)

# === ViT Model Dimensions ===
IMAGE_SIZE = 252 # 256
PATCH_SIZE = 14 # 16
EMBED_DIM = 128
NUM_HEADS = 4
DEPTH = 6  # ViT Layers
HIDDEN_DIM = 512 
PROJECTION_DIM = 128 

#=============================
#--- helper ---
class CFG:
    # Device
    DEVICE: str = DEVICE

    # Training
    EPOCHS: int = EPOCHS
    BATCH_SIZE: int = BATCH_SIZE
    NUM_WORKERS: int = NUM_WORKERS

    # SimCLR
    PROJECTION_DIM: int = PROJECTION_DIM
    TEMPERATURE: float = TEMPERATURE

    # MAE
    LAMBDA_MAE: float = LAMBDA_MAE

    # Optimizer
    BASE_LR: float = BASE_LR
    ETA_MIN: float = ETA_MIN
    WEIGHT_DECAY: float = WEIGHT_DECAY
    ENCODER_LR: float = ENCODER_LR
    PROJECTOR_LR: float = PROJECTOR_LR

    # Warmup
    WARMUP_FRAC: float = WARMUP_FRAC
    WARMUP_EPOCHS: int = WARMUP_EPOCHS

    # Model
    IMAGE_SIZE: int = IMAGE_SIZE
    PATCH_SIZE: int = PATCH_SIZE
    EMBED_DIM: int = EMBED_DIM
    NUM_HEADS: int = NUM_HEADS
    DEPTH: int = DEPTH
    HIDDEN_DIM: int = HIDDEN_DIM  # projector hidden
    PROJECTION_DIM: int = PROJECTION_DIM
    
    
#------------ Dynamic temp Implementation---------
    
#     # --- Temperature bounds & schedule ---
#     TAU_MIN: float = 0.01
#     TAU_MAX: float = 0.9
#     TAU_WARMUP_EPOCHS: int = 1 # I can adjust this later, but 2 for testing

#     # --- Multiplicative adjustment factors ---
#     TAU_UP: float = 1.15          # raise temp when variance is low
#     TAU_UP_HARD: float = 1.25     # stronger raise on collapse
#     TAU_DN: float = 0.93          # lower temp when variance is high
#     TAU_DN_HARD: float = 0.85     # stronger lower on very-high variance

#     # --- EMA smoothing for zvm ---
#     EMA_BETA: float = 0.9

#     # --- zvm scaling to match ClearML ---
#     ZVM_SCALE: float = 1e4

#     # --- Dimension-aware thresholds (as multiples of 1/d (d=zvm length)) ---
#     OK_LOW_MULT: float = 0.80
#     OK_HIGH_MULT: float = 1.25
#     VERY_HIGH_MULT: float = 1.60
#     BORDERLINE_MULT: float = 0.40
#     COLLAPSE_MULT: float = 0.10        # 10% of target
#     COLLAPSE_FLOOR_SCALED: float = 1.0 # never treat scaled zvm above this as “collapse”
# ---------------------------------
    
CFG = CFG()  

# # To keep code cleaner inside notebook
# def zvm_thresholds_scaled(d: int, scale: float = CFG.ZVM_SCALE) -> dict:
#     target = (1.0 / d) * scale # scale is for how I view zvm in ClearML  zvm * 1e-4
#     return {
#         "target": target,
#         "ok_low": CFG.OK_LOW_MULT * target,
#         "ok_high": CFG.OK_HIGH_MULT * target,
#         "very_high": CFG.VERY_HIGH_MULT * target,
#         "borderline": CFG.BORDERLINE_MULT * target,
#         "collapse": max(CFG.COLLAPSE_FLOOR_SCALED, CFG.COLLAPSE_MULT * target),
#     }