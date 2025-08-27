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



    
class UniCLIPModelEmbBank(BasePretrainModel):
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
                'name': 'ecg',
                'freq': 128,
                'win_sec': 30,
                'in_ch': 1,
            },
            {
                'name': 'eeg',
                'freq': 128,
                'win_sec': 30,
                'in_ch': 21,
            },
        ]
        self.num_channel = 22
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
        
        
        #####################################################################
        self.channel_bank = nn.Parameter(torch.zeros((self.num_channel, 128 * 30))) # need to modify if freq is not the same
        
        self.channel_embed = nn.Embedding(self.num_channel, self.proj_hidden)
        nn.init.normal_(self.channel_embed.weight, std=0.02)
        #####################################################################
        
    def _fill_zeros_with_embeddings(self, x: torch.Tensor):

        bank = self.channel_bank.to(device=x.device, dtype=x.dtype)        # (C, L)

        zero_mask = (x.abs().sum(dim=-1) == 0)                             # (B', C)


        x_filled = torch.where(zero_mask[..., None], bank.unsqueeze(0), x) # (B', C, L)
        return x_filled, zero_mask
    def shared_step(self, batch, batch_idx):
        # only used in training_step now
        # see train_config.py for the order
        
        x = batch['psg']
        # print(x.shape)
        ###########################
        # need further package these parameters (window size etc...)
        bz, c, t = x.shape
        x = x.view(bz, c, 10, 3840)    # -> (bz, 2, 10, 3840)
        x = x.permute(0, 2, 1, 3)      # -> (bz, 10, 2, 3840)
        x = x.reshape(bz * 10, c, 3840)  # -> (bz*10, 2, 3840)
        ############################
        x, zero_mask = self._fill_zeros_with_embeddings(x)
        # print(x.max(), x.min())
        ecg = x[:,0,:].unsqueeze(1)
        eeg = x[:,1:,:]
        
        
        ecg_output = self.encoders['ecg'](ecg)
        eeg_output = self.encoders['eeg'](eeg)

        

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
            ecg_output, eeg_output, self.logit_scale.exp())
        
        
        cos_sim = F.cosine_similarity(ecg_output, eeg_output, dim=-1).mean()
        loss_dict = {
            'loss': cma_loss,
            'cma_loss': cma_loss,
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
