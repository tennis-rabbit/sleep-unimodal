import torch
import ipdb
import yaml
import math
import time
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
from melp.utils.openclip_loss import ClipLoss
from melp.paths import PROMPT_PATH, DATASET_LABELS_PATH


class SimCLRModel(BasePretrainModel):
    def __init__(self, 
                 psg_encoder_name: str = "resnet18",
                 text_encoder_name: str = "google/flan-t5-base",
                 fusion_decoder_name: str = 'cross-attn',
                 shared_emb_dim: int = 256,
                 
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
                         
                         lr=lr,
                         weight_decay=weight_decay,
                         *args,
                         **kwargs)
        self.save_hyperparameters()
        self.proj_out = shared_emb_dim
        self.proj_hidden = 256

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
        
    
    def mask_channel(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.3,
        mode: str = "zero",                  # "zero" | "noise" | "swap" | "mean"
        same_mask_across_batch: bool = False,
        groups: list[int] | None = None,     # e.g., [3, 19] to mask ECG(3) and EEG(19) separately
        mask_only_existing: bool = True,     # NEW: mask only among channels that are non-all-zero
        existing_eps: float = 0.0,           # NEW: a channel is "existing" if sum(|x|) > existing_eps
        min_keep_per_group: int = 1,         # ensure at least this many existing channels remain
    ) -> torch.Tensor:
        """
        Randomly mask a subset of channels in a (B, C, T) tensor, optionally restricting
        the candidates to channels that are non-all-zero ("existing") per sample/group.

        - If `same_mask_across_batch` is True and `mask_only_existing` is True,
          the function will pick masked channels from those that are existing in **all**
          samples (intersection) within each group, so the same pattern is valid for the whole batch.
        """
        assert x.dim() == 3, f"Expected x of shape (B, C, T), got {tuple(x.shape)}"
        B, C, T = x.shape
        if C == 1 or mask_ratio <= 0:
            return x

        def _mask_one_group(xg: torch.Tensor) -> torch.Tensor:
            """
            xg: (B, Cg, T) for one channel group.
            """
            Bg, Cg, Tg = xg.shape

            # Identify "existing" channels
            # existing[b, c] = True if that channel is not all-zero (or > eps) for sample b.
            if mask_only_existing:
                existing = (xg.abs().sum(dim=-1) > existing_eps)  # (B, Cg) bool
            else:
                existing = torch.ones(Bg, Cg, dtype=torch.bool, device=xg.device)

            # Helper to apply fill given a boolean mask (B, Cg, T) where True = masked
            def _apply_fill(xg: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
                mask = mask_bool.to(xg.dtype).unsqueeze(-1).expand(-1, -1, Tg)      # (B, Cg, T)
                if mode == "zero":
                    return xg * (1 - mask)
                elif mode == "mean":
                    ch_mean = xg.mean(dim=-1, keepdim=True).detach()                # (B, Cg, 1)
                    return xg * (1 - mask) + ch_mean * mask
                elif mode == "noise":
                    ch_std = xg.std(dim=-1, keepdim=True, unbiased=False).detach()  # (B, Cg, 1)
                    noise = torch.randn_like(xg) * ch_std
                    return xg * (1 - mask) + noise * mask
                elif mode == "swap":
                    # Random channel permutation per sample
                    perm = torch.argsort(torch.rand(Bg, Cg, device=xg.device), dim=1)          # (B, Cg)
                    xg_shuf = torch.gather(xg, 1, perm.unsqueeze(-1).expand(-1, -1, Tg)).detach()
                    return xg * (1 - mask) + xg_shuf * mask
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            if same_mask_across_batch:
                # ---- Shared pattern across the batch ----
                # Candidate channels = those existing in ALL samples (intersection)
                cand = existing.all(dim=0)                              # (Cg,)
                Ce = int(cand.sum().item())
                if Ce <= min_keep_per_group:
                    # Not enough common-existing channels to mask; return as-is
                    return xg
                n_mask = int(round(Ce * mask_ratio))
                # Ensure we leave at least min_keep_per_group existing channels unmasked
                n_mask = max(0, min(n_mask, Ce - min_keep_per_group))
                if n_mask == 0:
                    return xg

                # Sample n_mask channels from candidates
                scores = torch.rand(Cg, device=xg.device) - (~cand).float() * 1e9   # push non-candidates to -inf
                idx = torch.topk(scores, k=n_mask, dim=0).indices                    # (n_mask,)
                mask_bool = torch.zeros(Bg, Cg, dtype=torch.bool, device=xg.device)
                mask_bool[:, idx] = True                                            # same pattern for all samples
                return _apply_fill(xg, mask_bool)

            else:
                # ---- Independent pattern per sample ----
                # For each sample, only choose among its existing channels
                # We'll rank channels by random scores, disallowing non-existing via -inf penalty
                scores = torch.rand(Bg, Cg, device=xg.device)
                scores = scores - (~existing).float() * 1e9                          # block non-existing

                # Get ranks per row (higher score => lower rank index)
                order = torch.argsort(scores, dim=1, descending=True)                # (B, Cg)
                rank = torch.argsort(order, dim=1)                                   # (B, Cg), rank 0..Cg-1

                # Number to mask per row: round(Ce * ratio), clamped to keep at least `min_keep_per_group`
                Ce = existing.sum(dim=1)                                             # (B,)
                n_mask = (Ce.float() * mask_ratio).round().to(torch.long)            # (B,)
                n_mask = torch.clamp(n_mask, min=0)
                # Ensure we leave at least `min_keep_per_group` existing channels
                n_mask = torch.minimum(n_mask, torch.clamp(Ce - min_keep_per_group, min=0))

                # Build boolean mask per row: mask channels with rank < n_mask[row]
                # Compare rank against per-row threshold by broadcasting
                thr = n_mask.view(Bg, 1).expand(Bg, Cg)
                mask_bool = rank < thr                                               # (B, Cg) bool

                # Also ensure we never mask a non-existing channel (safety, though scores prevent it)
                mask_bool = mask_bool & existing

                # If some rows end up with all False due to small Ce, that's fine (no masking)
                return _apply_fill(xg, mask_bool)

        if groups is None:
            return _mask_one_group(x)

        # Split into groups, mask each independently, then concat back
        assert sum(groups) == C, f"`groups` sum {sum(groups)} must equal C={C}"
        xs = torch.split(x, groups, dim=1)
        xs_masked = [_mask_one_group(xg) for xg in xs]
        return torch.cat(xs_masked, dim=1)
    def shared_step(self, batch, batch_idx):
        # only used in training_step now
        # see train_config.py for the order
        
        x = batch['psg']

        ###########################
        # need further package these parameters (window size etc...)
        bz, c, t = x.shape
        x = x.view(bz, c, 10, 3840)    # -> (bz, 22, 10, 3840)
        x = x.permute(0, 2, 1, 3)      # -> (bz, 10, 22, 3840)
        x = x.reshape(bz * 10, c, 3840)  # -> (bz*10, 22, 3840)
        ############################
        
        x_aug = self.mask_channel(x, mask_ratio=0.5, mode="zero")
        raw_output = self.encoders['all'](x)
        augment_output = self.encoders['all'](x_aug)

        

        # write infonce loss for lightning 
        loss = ClipLoss(
            local_loss=True,
            gather_with_grad=True,
            cache_labels=True,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            use_horovod=False
        )

        cma_loss = loss(
            raw_output, augment_output, self.logit_scale.exp())
        
        
        cos_sim = F.cosine_similarity(raw_output, augment_output, dim=-1).mean()
        loss_dict = {
            'loss': cma_loss,
        }

        # don't write metrics for now
        metrics_dict = {
            'cos sim': cos_sim
        }


        return loss_dict, metrics_dict

    def on_train_batch_end(self, *args, **kwargs) -> None:
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100)) 
        duration = time.perf_counter() - self._step_start

        # if self.trainer.is_global_zero:
        #     print(f"[Train step] batch  took {duration:.4f} sec")
    def on_train_batch_start(self, batch, batch_idx):
        self._step_start = time.perf_counter()


    def on_validation_epoch_start(self, *args, **kargs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self, *args, **kargs):
        pass


