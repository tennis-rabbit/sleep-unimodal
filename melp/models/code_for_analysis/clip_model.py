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
from melp.utils.openclip_loss import ClipLoss
from melp.paths import PROMPT_PATH, DATASET_LABELS_PATH


class CLIPModel(BasePretrainModel):
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
        self.init_text_encoder()
        
        
        self.cfg = [
            {
                'name': 'all',
                'freq': 64,
                'win_sec': 30,
                'in_ch': 21,
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
    
    
    def shared_step(self, batch, batch_idx):
        # only used in training_step now
        x = batch['psg']
        psg_output = self.encoders['all'](x)

        tokenized_input = self._tokenize(batch['report'])
        input_ids = tokenized_input['input_ids'].type_as(batch['ecg']).long()
        attention_mask = tokenized_input['attention_mask'].type_as(batch['ecg']).long()
        text_output = self.encode_text(input_ids, attention_mask)

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
            psg_output, text_output['proj_text_emb'], self.logit_scale.exp())
        
        loss_dict = {
            'loss': cma_loss,
            'cma_loss': cma_loss,
        }

        # don't write metrics for now
        metrics_dict = {}


        return loss_dict, metrics_dict

    def on_train_batch_end(self, *args, **kwargs) -> None:
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100)) 

    def on_validation_epoch_start(self, *args, **kargs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self, *args, **kargs):
        pass
