import torch
import ipdb
import yaml
import math
import numpy as np
from typing import List, Any
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, f1_score
from melp.backbone.resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from melp.backbone.vit1d import vit_tiny, vit_small, vit_middle, vit_base
from melp.backbone.pooling import AttentionPool2d
# from melp.utils.utils_loss import clip_loss
from melp.utils.openclip_loss import ClipLoss
from melp.paths import PROMPT_PATH, DATASET_LABELS_PATH

from lightning import LightningModule
from timm.optim import create_optimizer_v2
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
class PSGModalityEncoder(nn.Module):
    """
    function：backbone → optional down-conv → optional att-pool / proj → L2-norm
    usage：emb = encoder(x)  # (bz, proj_out)
    """
    def __init__(self, *,
                 encoder_name: str,
                 proj_out: int      = 256,
                 proj_hidden: int   = 512,
                 freq: int          = 64,
                 win_sec: int       = 30,
                 channel: int       = 11, ):
        super().__init__()
        token_len  = freq * win_sec         # e.g. 64×30 = 1920
        spacial_dim = token_len // 16       # ResNet down sample 4×4=16
        patch_size = 40
        self.token_len = token_len
        self.patch_size = patch_size
        
        # -------- build backbone --------
        if "resnet" in encoder_name:
            if encoder_name == "resnet18":
                self.backbone = ResNet18(in_ch = channel)
                in_ch = 512
            elif encoder_name == "resnet34":
                self.backbone = ResNet34(in_ch = channel); in_ch = 512
            elif encoder_name == "resnet50":
                self.backbone = ResNet50(in_ch = channel); in_ch = 2048
            elif encoder_name == "resnet101":
                self.backbone = ResNet101(in_ch = channel); in_ch = 2048

            self.downproj   = nn.Conv1d(in_ch, proj_out, kernel_size=1)
            self.att_pool   = AttentionPool2d(spacial_dim=spacial_dim,
                                              embed_dim=proj_out,
                                              num_heads=4,
                                              output_dim=proj_out)
            self.proj_head  = None         

        elif "vit" in encoder_name:
            if encoder_name == "vit_tiny":
                self.backbone = vit_tiny(num_leads=channel, seq_len = token_len, patch_size = patch_size)
            elif encoder_name == "vit_small":
                self.backbone = vit_small(num_leads=channel, seq_len = token_len, patch_size = patch_size)
            elif encoder_name == "vit_middle":
                self.backbone = vit_middle(num_leads=channel, seq_len = token_len, patch_size = patch_size)
            elif encoder_name == "vit_base":
                self.backbone = vit_base(num_leads=channel, seq_len = token_len, patch_size = patch_size)

            d_model         = self.backbone.width
            self.downproj   = None          # ViT dont use 1×1 conv
            self.att_pool   = None
            self.proj_head  = nn.Sequential(
                nn.Linear(d_model, proj_hidden),
                nn.LayerNorm(proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden, proj_out),
                nn.LayerNorm(proj_out),
            )
        # elif encoder_name.startswith('mae'):
        #     self.backbone = mae_tiny(
        #         num_leads=channel, seq_len = token_len, patch_size = patch_size, mask_ratio=0.75)

        self.avgpool = nn.AdaptiveAvgPool1d(1)  
    
    # ——— forward ———
    def forward(self, x):                    # x: (bz, C, L)
        h = self.backbone(x)                # (bz, 512, L′) or (bz, N, D)
        if self.downproj is not None:       # ResNet branch
            h = self.downproj(h)            # (bz, proj_out, L′)
            if self.att_pool is not None:
                h, _ = self.att_pool(h)     # (bz, 1, proj_out)
                h  = h.squeeze(1)           # (bz, proj_out)
            else:                           # avg-pool
                h  = self.avgpool(h).squeeze(-1)
        else:                               # ViT branch
            h = self.proj_head(h)           # (bz, proj_out)

        return F.normalize(h, dim=-1)


class BasePretrainModel(LightningModule):
    def __init__(self, 
                 psg_encoder_name: str = "resnet18",
                 text_encoder_name: str = "google/flan-t5-base",
                 fusion_decoder_name: str = 'cross-attn',
                 shared_emb_dim: int = 256,
                 lr: float = 2e-4,
                 weight_decay: float = 0.2,
                 training_steps_per_epoch: int = 1000,
                 max_epochs: int = 100,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.psg_encoder_name = psg_encoder_name
        self.text_encoder_name = text_encoder_name
        self.fusion_decoder_name = fusion_decoder_name
        self.shared_emb_dim = shared_emb_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_steps_per_epoch = training_steps_per_epoch
        self.max_epochs = max_epochs
        self.warmup_epochs = int(0.5 * self.max_epochs)
        self.proj_out = shared_emb_dim
        self.proj_hidden = 256

        
        assert self.training_steps_per_epoch > 1, "training_steps_per_epoch must be greater than 1"
    def init_text_encoder(self):
        if self.text_encoder_name == "google/flan-t5-small":
            self.lm_model = T5EncoderModel.from_pretrained(
                self.text_encoder_name, trust_remote_code=True)
            text_encoder_hidden_dim = 512
        elif self.text_encoder_name == "google/flan-t5-base":
            # self.lm_model = T5EncoderModel.from_pretrained(
            #     self.text_encoder_name, trust_remote_code=True)
            
            self.lm_model = T5EncoderModel.from_pretrained(
                "/u/ztshuai/.cache/huggingface/hub/models--google--flan-t5-base/snapshots/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2")
            text_encoder_hidden_dim = 768
        else:
            raise NotImplementedError
        # text projector
        self.proj_t = nn.Sequential(
            nn.Linear(text_encoder_hidden_dim, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.text_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/u/ztshuai/.cache/huggingface/hub/models--google--flan-t5-base/snapshots/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2")
        
    
    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=128,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output
    def encode_text(self, input_ids, attention_mask):
        if self.text_encoder_name in ["google/flan-t5-small", "google/flan-t5-base"]:
            sequence_output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            eos_mask = input_ids.eq(self.lm_model.config.eos_token_id).type_as(attention_mask).bool()
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            batch_size, _, hidden_size = sequence_output.shape
            text_emb = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
            
        proj_text_emb = self.proj_t(text_emb)
        proj_text_emb = F.normalize(proj_text_emb, dim=-1)

        return {
            'proj_text_emb': proj_text_emb,
            'text_emb': text_emb
        }
    
    def configure_optimizers(self):

        # optimizer = create_optimizer_v2(
        #     self.parameters(),
        #     "adamw",
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     eps=1e-8,
        #     betas=(0.9, 0.95),
        # )

        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.lr, 
                                      weight_decay=self.weight_decay,
                                      betas=(0.9, 0.95))

        warmup_steps = self.training_steps_per_epoch * self.warmup_epochs
        total_steps = self.training_steps_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=warmup_steps,
                T_mult=1,
                eta_min=1e-8,
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        # scheduler = {
        #     "scheduler": CosineAnnealingWarmupRestarts(
        #         optimizer,
        #         first_cycle_steps=total_steps,
        #         cycle_mult=1.0,
        #         max_lr=self.lr,
        #         min_lr=1e-8,
        #         warmup_steps=warmup_steps,
        #         gamma=1.0
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss_dict, metrics_dict = self.shared_step(batch, batch_idx)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in metrics_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        loss_dict, metrics_dict = self.shared_step(batch, batch_idx)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in metrics_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, metrics_dict = self.shared_step(batch, batch_idx)
        for k, v in loss_dict.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in metrics_dict.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict