# transformer_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Notes (ViT encoder + SimCLR projector + MAE head)
# -----------------------------------------------------------------------------
# • Overall shape contract:
#   - Inputs are [B, 6, H, W]; PatchEmbed [B, N, D] where N=(H/P)*(W/P), D=emb_dim.
#   - ViTEncoder prepends a CLS token and adds positional embeddings [B, 1+N, D].
#   - SimCLR uses the CLS token as the global representation; projector maps CLS to z.
#
# • Patch embedding:
#   - Conv2d(kernel=patch, stride=patch) is a fast patchify; no overlap, no conv bias
#     tricks here—just a linear projection per patch. If I change IMAGE/PATCH size, the
#     number of patches N recomputes from those values; no hard-coded 256 anywhere.
#
# • Positional embeddings (dynamic sizing):
#   - `ViTEncoder.pos_embed` is sized to (1, 1+N, D) where N is computed from
#     (img_size // patch_size)**2 at model build. If I reload a checkpoint with a
#     *different* (img_size, patch_size), the shape will mismatch—either rebuild the
#     pos_embed or interpolate it. 
#
# • CLS token:
#   - I return `hidden[:, 0, :]` as the global embedding. 
#
# • Projection head (SimCLR):
#   - 2-layer MLP with BN–ReLU–BN. BN on the output is common in SimCLR-style heads
#     (stabilizes contrastive logits). 
#
# • MAE head:
#   - Randomly mask a ratio of patch tokens (default 0.6), *keep* the remainder, and
#     feed (CLS + kept) through the *shared* ViT encoder (with correct pos embeds).
#   - Map encoder outputs to decoder dim, insert learned mask tokens at the missing
#     positions, add decoder positional embeddings (learned here), run a small
#     Transformer “decoder” (actually an encoder block over the full sequence),
#     and predict pixels per patch. Loss = MSE over **masked** patches only.
#   - `to_pixels` head is created lazily once I know (patch_size^2 * in_channels).
#     This avoids hard-coding sizes and keeps MAE decoupled from ViT constructor args.
#
# • Important shape checks:
#   - PatchEmbed: [B, 6, H, W]  [B, N, D]
#   - ViTEncoder: [B, N, D]  [B, 1+N, D]
#   - SimCLR head: CLS  [B, proj_out]
#   - MAE:
#       target_patches: [B, N, p*p*C]
#       dec_out_patches: [B, N, dec_D]  to_pixels  [B, N, p*p*C]
#       mask: [B, N] with 1 at masked positions (loss over masked only)
#
# • Reuse of ViT positional embeddings in MAE:
#   - I slice `encoder.pos_embed` into CLS and patch parts, then gather the *kept*
#     patch positions to align pos embeddings with the visible tokens. This prevents
#     the “double-pos” bug and keeps positional semantics consistent across tasks.
#
# • random_masking(generator=...):
#   - Accepts an optional torch.Generator for reproducibility. It sorts random noise
#     per sample to pick which tokens to keep; returns indices to restore original order.
#
# • Stability / training tips:
#   - If SimCLR collapses (low z_var_mean), lower temperature τ a bit, or slightly
#     reduce augmentation strength (e.g., fewer masks / smaller erase). Another lever:
#     increase projector hidden dim or add a third layer.
#   - If MAE dominates gradients, lower λ_MAE; I already drop λ_MAE→0 when patience
#     stalls. Conversely, if reconstructions look too poor, try mask_ratio≈0.75 and
#     a slightly larger decoder (depth=4, dim=384).
#   - AdamW with lower LR on encoder and higher LR on projector is intentional:
#     projector can move quickly to shape contrastive geometry while the backbone
#     updates more conservatively.
#
# • Memory/perf knobs:
#   - Attention cost ~ O((N+1)^2). Track N = (H/P)*(W/P). If I bump IMAGE_SIZE or
#     shrink PATCH_SIZE, N grows quickly; keep an eye on VRAM and step time.
#   - Mixed precision is currently off (TF32/Flash disabled for determinism). If I need
#     speed, I can enable AMP once I’m done with strict reproducibility work.
#
# =============================================================================

# === Patch Embedding ===
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=6, patch_size=16, emb_dim=128, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, 256, emb_dim]
        return x

# === Vision Transformer ===
class ViTEncoder(nn.Module):
    def __init__(self, emb_dim: int, depth: int, num_heads: int, num_patches: int, mlp_ratio: float =4.0, dropout: float =0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))  # <-- dynamic
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        B, N, D = x.shape  # [B, 256, 128]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 128]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 257, 128]
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)
        #return x[:, 0]  # return [CLS] token
        return x

class ProjectionHead(nn.Module):
    # def __init__(self, emb_dim=128, projection_dim=64, hidden_dim=256):
    def __init__(self, emb_dim: int, projection_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )

    def forward(self, x):
        return self.mlp(x)

# === Combined Model ===
class ViTForSimCLR(nn.Module):
    def __init__(self, in_channels: int, img_size: int, patch_size: int, emb_dim: int, depth: int, num_heads: int, proj_hidden: int, proj_out: int):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, emb_dim, img_size)       
        num_patches = (img_size // patch_size) ** 2
        self.encoder   = ViTEncoder(emb_dim, depth, num_heads, num_patches=num_patches)
        
        
        # self.encoder = ViTEncoder(emb_dim, depth, num_heads)
        self.projector = ProjectionHead(
            emb_dim=emb_dim,
            projection_dim=proj_out,
            hidden_dim=proj_hidden,
        )

        # # expose for MAE
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim

    def forward(self, x):
        patches = self.patch_embed(x)                # [B, 256, 128]
        hidden = self.encoder(patches)               # [B, 257, 128]
        cls_token = hidden[:, 0, :]                  # now [B, 128]!
        proj      = self.projector(cls_token)        # [B, 64]
        return cls_token, proj


#=============== MAE ===============
class MAEModule(nn.Module):
    def __init__(self, emb_dim=128, decoder_dim=256, decoder_depth=2, decoder_heads=4, mask_ratio=0.6, ):
        super().__init__()
        self.mask_ratio = mask_ratio

        # Project encoder output to decoder dim
        self.enc_to_dec = nn.Linear(emb_dim, decoder_dim, bias=True)

        # Learned mask token (in decoder dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # A small transformer "decoder" (really just a transformer encoder over
        d_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(d_layer, num_layers=decoder_depth)

        # Decoder positional embedding for (CLS + all patch tokens)
        # Create this at runtime to match the exact token count of the encoder
        self.register_buffer("dec_pos_embed", None, persistent=False)

        # Final "un-patchify" head: predict pixels per patch
        # (patch_size^2 * in_channels) will be set lazily based on the model passed at forward
        self.to_pixels = None  

    # ------- helpers -------
    @staticmethod
    def patchify(x, patch_size):

        B, C, H, W = x.shape
        assert H % patch_size == 0 and W % patch_size == 0
        h = H // patch_size
        w = W // patch_size
        # unfold -> [B, C*p*p, h*w]
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)  # [B, C*p*p, N]
        patches = patches.transpose(1, 2)  # [B, N, C*p*p]
        return patches
 
    @staticmethod
    def unpatchify(patches, patch_size, H, W):

        B, N, PPc = patches.shape
        C = PPc // (patch_size * patch_size)
        h = H // patch_size
        w = W // patch_size
        assert N == h * w

        patches = patches.transpose(1, 2)  # [B, C*p*p, N]
        x = F.fold(patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
        return x

    @staticmethod
    def random_masking(x, mask_ratio, generator=None):

        B, N, D = x.shape
        device = x.device

        if generator is None:
            noise = torch.rand(B, N, device=device)
        else:
            noise = torch.rand(B, N, device=device, generator=generator)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        N_keep = int((1 - mask_ratio) * N)
        ids_keep = ids_shuffle[:, :N_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=device)
        mask[:, :N_keep] = 0
        # unshuffle to original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, ids_restore, mask, ids_keep

    def forward(self, x, vit_model: "ViTForSimCLR", generator=None):
        
        B, C, H, W = x.shape
        p = vit_model.patch_size
        emb_dim = vit_model.emb_dim
        N = (H // p) * (W // p)  # number of patches

        # Lazily create pixel head once know in_channels & patch_size
        if self.to_pixels is None:
            patch_dim = p * p * vit_model.in_channels
            self.to_pixels = nn.Linear(self.decoder.layers[0].linear1.in_features, patch_dim, bias=True).to(x.device)

        # ----- patchify pixels for ground-truth -----
        target_patches = self.patchify(x, p)  # [B, N, p*p*C]

        # ----- patch embed -----
        patch_tokens = vit_model.patch_embed(x)  # [B, N, emb_dim]

        # ----- mask -----
        x_keep, ids_restore, mask, ids_keep = self.random_masking(patch_tokens, self.mask_ratio, generator=generator)

        # ----- positional embeddings (reuse from ViT) -----
        # ViT pos_embed is [1, 257, emb_dim] (CLS + 256 patches)
        pos_full = vit_model.encoder.pos_embed  # [1, 1+N, D]
        pos_cls = pos_full[:, :1, :]           # [1, 1, D]
        pos_patches = pos_full[:, 1:, :]       # [1, N, D]
        # Select the kept positions' pos embeds
        pos_keep = torch.gather(
            pos_patches.expand(B, -1, -1),
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, emb_dim),
        )  

        # ----- prepend CLS and run through the ViT encoder -----
        cls_tok = vit_model.encoder.cls_token.expand(B, -1, -1)  # [B,1,D]
        enc_in = torch.cat([cls_tok + pos_cls, x_keep + pos_keep], dim=1)  # [B, 1+N_keep, D]
        # Call the bare encoder stack since already added pos embeds
        enc_out = vit_model.encoder.encoder(enc_in)  # [B, 1+N_keep, D]
        enc_patch_tokens = enc_out[:, 1:, :]         # drop CLS

        # ----- map to decoder dim -----
        dec_visible = self.enc_to_dec(enc_patch_tokens)  # [B, N_keep, dec_D]

        # ----- build full sequence for decoder: insert mask tokens and restore original order -----
        B, N_keep, dec_D = dec_visible.shape
        # Prepare (masked) sequence of length N in original order
        # First, create full sequence filled with mask tokens
        dec_tokens = self.mask_token.expand(B, N, dec_D).clone()  # [B, N, dec_D]

        # Fill kept positions with encoded visible tokens (in original order)
        dec_tokens.scatter_(dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dec_D), src=dec_visible)

        # ----- add decoder positional embeddings (create if missing or wrong length) -----
        need_len = 1 + N  # CLS + N patches
        if (self.dec_pos_embed is None) or (self.dec_pos_embed.shape[1] != need_len) or (self.dec_pos_embed.shape[2] != dec_D):
            # simple learnable dec pos embed
            self.dec_pos_embed = nn.Parameter(torch.zeros(1, need_len, dec_D, device=x.device))
            nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        dec_pos_cls = self.dec_pos_embed[:, :1, :]
        dec_pos_patches = self.dec_pos_embed[:, 1:, :]

        # Prepend a (zero) decoder CLS token just to mirror shapes; ignore it in loss
        dec_input = torch.cat([self.mask_token[:, :1, :].expand(B, -1, -1), dec_tokens], dim=1)  # [B, 1+N, dec_D]
        dec_input = dec_input + torch.cat([dec_pos_cls.expand(B, -1, -1), dec_pos_patches.expand(B, -1, -1)], dim=1)

        # ----- run decoder -----
        dec_out = self.decoder(dec_input)  # [B, 1+N, dec_D]
        dec_out_patches = dec_out[:, 1:, :]  # [B, N, dec_D]

        # ----- predict pixels per patch -----
        pred = self.to_pixels(dec_out_patches)  # [B, N, p*p*C]

        # ----- compute MSE on masked patches only -----
        loss = (F.mse_loss(pred, target_patches, reduction='none')).mean(dim=-1)  # [B, N]
        loss = (loss * mask).sum() / (mask.sum().clamp(min=1.0))  # scalar

        return loss



    
    
