from typing import Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
import torch.nn.functional as F
from torchmetrics import AUROC
from einops import rearrange


# class SSLFineTuner(LightningModule):
#     def __init__(self,
#         backbones,
#         use_which_backbone,
#         config = None,
#         in_features: int = 256,
#         num_classes: int = 2,
#         epochs: int = 100,
#         dropout: float = 0.0,
#         lr: float = 1e-3,
#         weight_decay: float = 1e-4,
#         scheduler_type: str = "cosine",
#         decay_epochs: Tuple = (60, 80),
#         gamma: float = 0.1,
#         final_lr: float = 1e-5,
#         use_ecg_patch: bool = False,
#         use_channel_bank: bool = True,         
#         *args, **kwargs
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.scheduler_type = scheduler_type
#         self.decay_epochs = decay_epochs
#         self.gamma = gamma
#         self.epochs = epochs
#         self.final_lr = final_lr
#         self.use_ecg_patch = use_ecg_patch
#         self.use_channel_bank = use_channel_bank   

#         self.backbones = backbones
#         self.config = config
#         self.use_which_backbone = use_which_backbone
#         self.backbone = self.backbones[self.use_which_backbone]
        
#         for p in self.backbone.parameters():
#             p.requires_grad = False

       
#         self.register_buffer("channel_bank", None, persistent=True)
#         if self.use_channel_bank:
            
#             src = getattr(self.backbone, "module", self.backbone)
#             if hasattr(src, "channel_bank") and isinstance(src.channel_bank, torch.nn.Parameter):
#                 self.channel_bank = src.channel_bank.detach().clone()
#             else:
#                 self.channel_bank = None

#         self.linear_layer = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(in_features, num_classes)
#         )

#         self.train_auc = AUROC(task="multiclass", num_classes=num_classes)
#         self.val_auc = AUROC(task="multiclass", num_classes=num_classes)
#         self.test_auc = AUROC(task="multiclass", num_classes=num_classes)
#     def _fill_zeros_with_bank(self, x: torch.Tensor) -> torch.Tensor:

#         if (self.channel_bank is None) or (x is None):
#             return x

#         C, T = x.size(1), x.size(2)
#         bank = self.channel_bank
#         if bank.dim() == 2 and (bank.size(0) >= C) and (bank.size(1) >= T):
#             bank = bank[:C, :T].to(device=x.device, dtype=x.dtype)
#         else:
#             return x 

#         zero_mask = (x.abs().sum(dim=-1) == 0)          # (B, C)
#         if zero_mask.any():
#             x = torch.where(zero_mask[..., None], bank.unsqueeze(0), x)
#         return x
#     def on_train_epoch_start(self) -> None:
#         self.backbone.eval()

#     def training_step(self, batch, batch_idx):
#         loss, logits, y = self.shared_step(batch)
#         auc = self.train_auc(logits.softmax(-1), y.long())

#         self.log("train_loss", loss, prog_bar=True)
#         self.log("train_auc_step", auc, prog_bar=True)
#         self.log("train_auc_epoch", self.train_auc)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, logits, y = self.shared_step(batch)
#         self.val_auc(logits.softmax(-1), y.long())

#         self.log("val_loss", loss, prog_bar=True, sync_dist=True)
#         self.log("val_auc", self.val_auc)

#         return loss

#     def test_step(self, batch, batch_idx):
#         loss, logits, y = self.shared_step(batch)
#         self.test_auc(logits.softmax(-1), y.long())

#         self.log("test_loss", loss, sync_dist=True)
#         self.log("test_auc", self.test_auc)

#         return loss

#     def shared_step(self, batch):
#         # Extract features from the backbone
#         with torch.no_grad():
#             ecg = batch['psg'][:,0:3,:]
#             y = batch["label"]
            
#             feats = self.backbone(ecg)
            

#         feats = feats.view(feats.size(0), -1)
#         logits = self.linear_layer(feats)
#         y = y.squeeze(1).long()
#         # print(logits.shape, y.shape)
#         loss = F.cross_entropy(logits, y)

#         return loss, logits, y

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.linear_layer.parameters(),
#             lr=self.lr,
#             weight_decay=self.weight_decay,
#         )

#         # set scheduler
#         if self.scheduler_type == "step":
#             scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
#         elif self.scheduler_type == "cosine":
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
#             )

#         return [optimizer], [scheduler]


class SSLFineTuner(LightningModule):
    def __init__(self,
        backbones,
        use_which_backbone,
        config = None,
        in_features: int = 256,
        num_classes: int = 2,
        epochs: int = 100,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        decay_epochs: Tuple = (60, 80),
        gamma: float = 0.1,
        final_lr: float = 1e-5,
        use_ecg_patch: bool = False,
        use_channel_bank: bool = True,         
        *args, **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr
        self.use_ecg_patch = use_ecg_patch
        self.use_channel_bank = use_channel_bank   

        self.backbones = backbones
        self.config = config
        self.use_which_backbone = use_which_backbone
        self.backbone = self.backbones[self.use_which_backbone]
        
        for p in self.backbone.parameters():
            p.requires_grad = False

       
        self.register_buffer("channel_bank", None, persistent=True)
        if self.use_channel_bank:
            
            src = getattr(self.backbone, "module", self.backbone)
            if hasattr(src, "channel_bank") and isinstance(src.channel_bank, torch.nn.Parameter):
                self.channel_bank = src.channel_bank.detach().clone()
            else:
                self.channel_bank = None

        self.linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

        self.train_auc = AUROC(task="multiclass", num_classes=num_classes)
        self.val_auc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_auc = AUROC(task="multiclass", num_classes=num_classes)
    def _fill_zeros_with_bank(self, x: torch.Tensor) -> torch.Tensor:

        if (self.channel_bank is None) or (x is None):
            return x

        C, T = x.size(1), x.size(2)
        bank = self.channel_bank
        if bank.dim() == 2 and (bank.size(0) >= C) and (bank.size(1) >= T):
            bank = bank[:C, :T].to(device=x.device, dtype=x.dtype)
        else:
            return x 

        zero_mask = (x.abs().sum(dim=-1) == 0)          # (B, C)
        if zero_mask.any():
            x = torch.where(zero_mask[..., None], bank.unsqueeze(0), x)
        return x
    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        auc = self.train_auc(logits.softmax(-1), y.long())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_auc_step", auc, prog_bar=True)
        self.log("train_auc_epoch", self.train_auc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.val_auc(logits.softmax(-1), y.long())

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_auc", self.val_auc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.test_auc(logits.softmax(-1), y.long())

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_auc", self.test_auc)

        return loss

    def shared_step(self, batch):
        # Extract features from the backbone
        with torch.no_grad():
            if self.use_which_backbone == 'ecg':
                x = batch['psg'][:,0:3,:]
            ###########################
            # need further package these parameters (window size etc...)
            bz, c, t = x.shape
            x = x.view(bz, c, 10, 3840)    # -> (bz, 2, 10, 3840)
            x = x.permute(0, 2, 1, 3)      # -> (bz, 10, 2, 3840)
            x = x.reshape(bz * 10, c, 3840)  # -> (bz*10, 2, 3840)
            ############################
            y = batch["label"]
            bz, _ = y.shape
            y = y.view(bz, 10, 1)    
            y = y.reshape(bz * 10, 1)  
            
            # rint(y.shape, x.shape)
            feats = self.backbone(x)
            

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        y = y.squeeze(1).long()
        # print(logits.shape, y.shape)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.linear_layer.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]