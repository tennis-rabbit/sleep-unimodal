import torch
import ipdb
import yaml
import math
import numpy as np
from typing import List
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, f1_score
from melp.backbone.resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from melp.backbone.vit1d import vit_tiny, vit_small, vit_middle, vit_base
from melp.backbone.pooling import AttentionPool2d
from melp.models.base_pretrain_model import BasePretrainModel, PSGModalityEncoder
# from melp.utils.utils_loss import clip_loss
from melp.utils.openclip_loss import ClipLoss
from melp.paths import PROMPT_PATH, DATASET_LABELS_PATH
from melp.backbone.vit1d import TransformerBlock
from einops import rearrange
from tabulate import tabulate

def report_trainable(model: torch.nn.Module, show_table: bool = True):
    per_module = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = name.split('.')[0]                 
        per_module[key] = per_module.get(key, 0) + p.numel()

    total = sum(per_module.values())
    if show_table:
        rows = [(k, f"{v:,}") for k, v in sorted(per_module.items())]
        rows.append(("TOTAL", f"{total:,}"))
        print("\nTrainable parameters:")
        print(tabulate(rows, headers=["Module", "# params"], tablefmt="github"))
    return {"total": total, **per_module}

class MAEModel(BasePretrainModel):
    def __init__(self, 
                 psg_encoder_name: str = "vit_tiny",
                 text_encoder_name: str = "google/flan-t5-base",
                 fusion_decoder_name: str = 'cross-attn',
                 shared_emb_dim: int = 256,
                 num_leads: int = 12,
                 num_freeze_layers: int = 6,
                 init_logit_scale: float = np.log(1 / 0.07),
                 lr: float = 2e-4,
                 weight_decay: float = 0.2,
                 *args,
                 **kwargs):
        
        self.num_freeze_layers = num_freeze_layers
        super().__init__(psg_encoder_name=psg_encoder_name,
                         text_encoder_name=text_encoder_name,
                         fusion_decoder_name=fusion_decoder_name,
                         shared_emb_dim=shared_emb_dim,
                         num_leads=num_leads,
                         lr=lr,
                         weight_decay=weight_decay,
                         *args,
                         **kwargs)
        self.save_hyperparameters()
        self.proj_out = shared_emb_dim
        self.proj_hidden = 256
        self.init_text_encoder()
        

        self.cfg = [
            {
                'name': 'all',
                'freq': 128,
                'win_sec': 30,
                'in_ch': 22,
            },
        ]
        self.encoders = nn.ModuleDict()
        for mod in self.cfg:
            self.encoders[mod['name']] = PSGModalityEncoder(
                encoder_name = psg_encoder_name,
                proj_out     = self.proj_out,
                proj_hidden  = self.proj_hidden,
                freq = mod['freq'],
                win_sec = mod['win_sec'],
                channel = mod['in_ch'], 
            )
        

        lshape = []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        
        # ---- MAE components ----
        # ---- MAE components ----
        # ---- MAE components ----
        self.mask_ratio = 0.75
        dec_embed_dim   = 512
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))

        # decoder 4 block
        self.dec_proj = nn.Linear(self.proj_out, dec_embed_dim)
        self.dec_blocks = nn.ModuleList([
            TransformerBlock(dec_embed_dim, dec_embed_dim,
                             hidden_dim = dec_embed_dim*4,
                             heads = 8, dim_head = 64)
            for _ in range(4)
        ])
        self.dec_norm  = nn.LayerNorm(dec_embed_dim)

        patch_pixels = self.encoders['all'].backbone.to_patch_embedding.in_channels * \
                       self.encoders['all'].patch_size
        self.pred_head = nn.Linear(dec_embed_dim, patch_pixels)
        
        n_patch = self.encoders['all'].token_len // self.encoders['all'].patch_size
        self.pos_dec = nn.Parameter(
            torch.randn(1, n_patch, dec_embed_dim))
        
        # ---- MAE components ----
        # ---- MAE components ----
        # ---- MAE components ----
        stats = report_trainable(self, show_table=True)
    def shared_step(self, batch, batch_idx):
        sig = batch['psg']  # (B, C, L)
        bz, c, t = sig.shape

        if 'channel_mask' in batch and batch['channel_mask'] is not None:
            chmask0 = batch['channel_mask'].to(sig.dtype)           # (B, C)
            sig = sig * chmask0.unsqueeze(-1)
        else:
            eps = 1e-6
            chmask0 = (sig.abs().amax(dim=-1) > eps).to(sig.dtype)   # (B, C)
            sig = sig * chmask0.unsqueeze(-1)

        sig = sig.view(bz, c, 10, 3840).permute(0, 2, 1, 3).reshape(bz * 10, c, 3840)  # (B', C, 3840)
        Bp = sig.size(0)

        chmask = chmask0.repeat_interleave(10, dim=0).to(sig.dtype)  # (B', C)

        enc = self.encoders['all']
        vit = enc.backbone
        x = vit.to_patch_embedding(sig)                 # (B', width, N)
        x = rearrange(x, 'b c n -> b n c')              # (B', N, width)
        x = x + vit.pos_embedding
        x_vis, mask, ids_restore = self._random_mask(x) # mask: (B', N), 1=masked, 0=keep

        for i in range(vit.depth):
            x_vis = getattr(vit, f'block{i}')(x_vis)
        latent = vit.norm(x_vis)
        latent = enc.proj_head(latent)

        latent = self.dec_proj(latent)                  # (B', N_keep, dec_dim)
        Bp, N_vis, D = latent.shape
        mask_tokens = self.mask_token.repeat(Bp, mask.size(1) - N_vis, 1)
        x_full = torch.cat([latent, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x_full = x_full + self.pos_dec
        for blk in self.dec_blocks:
            x_full = blk(x_full)
        x_full = self.dec_norm(x_full)
        pred = self.pred_head(x_full)                   # (B', N, C*P)

        P = enc.patch_size                             
        target = rearrange(sig, 'b c (n p) -> b n (c p)', p=P)  # (B', N, C*P)

        mse_elem = (pred - target).pow(2)               # (B', N, C*P)

        patch_mask = mask.unsqueeze(-1).to(pred.dtype)

        N = pred.size(1)
        chan_mask_elem = chmask.unsqueeze(1).repeat(1, N, 1)      # (B', N, C)
        chan_mask_elem = chan_mask_elem.repeat_interleave(P, -1)  # (B', N, C*P)


        elem_mask = patch_mask * chan_mask_elem                   # (B', N, C*P)

        denom = elem_mask.sum().clamp_min(1.0)
        loss_recon = (mse_elem * elem_mask).sum() / denom

        metrics_dict = {
            'present_channel_ratio': chmask.mean(),
            'valid_loss_frac': (elem_mask > 0).float().mean()
        }

        return {'loss': loss_recon, 'loss_recon': loss_recon}, metrics_dict

    @torch.no_grad()
    def _random_mask(self, x):
        B, L, D = x.shape
        keep = max(1, int(L * (1 - self.mask_ratio)))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle  = torch.argsort(noise, dim=1)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)
        ids_keep     = ids_shuffle[:, :keep]
        mask = torch.ones(B, L, device=x.device)
        mask.scatter_(1, ids_keep, 0)
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1,1,D))
        return x_vis, mask, ids_restore

    def on_train_batch_end(self, *args, **kwargs) -> None:
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100)) 
