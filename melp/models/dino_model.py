import copy, math, time, random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


from melp.models.dino_utils.dino_clstoken_loss import DINOLoss 
from melp.models.dino_utils.ibot_patch_loss import iBOTPatchLoss
from melp.models.dino_utils.koleo_loss import KoLeoLoss  
from melp.models.base_pretrain_model import BasePretrainModel, PSGModalityEncoder

class DINOHead(nn.Module):
    """
    trunk_dim -> [hidden] -> bottleneck -> weight-norm prototypes (out_dim)
    """
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        num_layers = max(nlayers, 1)
        if num_layers == 1:
            self.mlp = nn.Sequential(nn.Linear(in_dim, bottleneck_dim))
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        # weight-normalized prototypes (no bias)
        self.prototypes = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # init prototype scale
        self.prototypes.weight_g.data.fill_(1.0)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)      # bottleneck norm（DINOv2 常见）
        return self.prototypes(x)       # logits -> (B, out_dim)



class DINOModel(BasePretrainModel):
    def __init__(
        self,
        psg_encoder_name: str = "resnet18",
        text_encoder_name: str = "google/flan-t5-base",
        shared_emb_dim: int = 256,

        # DINO 
        out_dim: int = 8192,           # prototypes number
        patch_out_dim: int = 8192,     # iBOT prototypes number
        student_temp: float = 0.1,
        teacher_temp_warmup: float = 0.04,
        teacher_temp_final: float = 0.07,
        teacher_temp_warmup_iters: int = 10000,

        base_momentum: float = 0.996,  # EMA
        use_koleo: bool = False,
        koleo_lambda: float = 0.02,
        ibot_lambda: float = 0.0,

        
        lr: float = 2e-4,
        weight_decay: float = 0.2,

        
        num_freeze_layers: int = 6,
        *args, **kwargs
    ):
        super().__init__(
            psg_encoder_name=psg_encoder_name,
            text_encoder_name=None,
            fusion_decoder_name=None,
            shared_emb_dim=shared_emb_dim,
            lr=lr,
            weight_decay=weight_decay,
            *args, **kwargs
        )
        self.save_hyperparameters()

        
        self.cfg = [
            dict(name='all', freq=128, win_sec=30, in_ch=22),
        ]

        # ===== student encoder =====
        self.encoders = nn.ModuleDict()
        for mod in self.cfg:
            self.encoders[mod['name']] = PSGModalityEncoder(
                encoder_name=psg_encoder_name,
                proj_out=shared_emb_dim,
                proj_hidden=256,
                freq=mod['freq'],
                win_sec=mod['win_sec'],
                channel=mod['in_ch'],
            )

        
        trunk_dim = shared_emb_dim

        # ===== heads（student）=====
        self.student_global_head = DINOHead(trunk_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3)
        self.student_patch_head  = DINOHead(trunk_dim, patch_out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3)

        
        self.teacher_encoder = copy.deepcopy(self.encoders['all'])
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        self.teacher_global_head = DINOHead(trunk_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3)
        self.teacher_patch_head  = DINOHead(trunk_dim, patch_out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3)


        self.teacher_global_head.load_state_dict(self.student_global_head.state_dict(), strict=True)
        self.teacher_patch_head.load_state_dict(self.student_patch_head.state_dict(), strict=True)

  
        for p in self.teacher_global_head.parameters():
            p.requires_grad = False
        for p in self.teacher_patch_head.parameters():
            p.requires_grad = False
        self.teacher_global_head.eval()
        self.teacher_patch_head.eval()


        # ===== losses =====
        self.dino_loss = DINOLoss(out_dim=out_dim, student_temp=student_temp, center_momentum=0.9)
        self.ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim, student_temp=student_temp, center_momentum=0.9)
        self.koleo = KoLeoLoss() if use_koleo else None
        self.koleo_lambda = koleo_lambda
        self.ibot_lambda = ibot_lambda

        # ===== teacher temperature / EMA momentum =====
        self.teacher_temp_warmup = teacher_temp_warmup
        self.teacher_temp_final  = teacher_temp_final
        self.teacher_temp_warmup_iters = teacher_temp_warmup_iters
        self.base_momentum = base_momentum

        
        self.register_buffer("seen_steps", torch.tensor(0, dtype=torch.long))

    ##############################################################################
    ##############################################################################
    def _make_views(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        输入: x (B, C, T) 5分钟片段
        输出: (globals, locals)
        - globals: 2 个视角（teacher 只看这些）
        - locals:  若干短窗视角
        这里给最简实现：用你已有切 10×30s 的逻辑，从中随机抽取 2 个当 global，另外 2~4 个当 local，并叠加轻度增广
        """
        B, C, T = x.shape
        # 切成 10 x 30s
        assert T % 3840 == 0, "Expect T to be multiple of 3840 (30s @ 128Hz)"
        num_epochs = T // 3840
        x_ = x.view(B, C, num_epochs, 3840).permute(0, 2, 1, 3).contiguous()  # (B, 10, C, 3840)

        # 随机挑 2 个 global，2-4 个 local
        idx = list(range(num_epochs))
        random.shuffle(idx)
        g_idx = idx[:2]
        l_idx = idx[2:2+random.randint(2, 4)]

        globals_ = [self._augment_global(x_[:, i]) for i in g_idx]   # each: (B, C, 3840)
        locals_  = [self._augment_local(x_[:, i])  for i in l_idx]   # each: (B, C, 3840)
        return globals_, locals_

    def _augment_global(self, x_epoch):  # x_epoch: (B, C, 3840)
 
        x = self._mask_channel(x_epoch, mask_ratio=0.2)
        noise = 0.01 * torch.randn_like(x)
        return x + noise

    def _augment_local(self, x_epoch):

        x = self._mask_channel(x_epoch, mask_ratio=0.5)
        noise = 0.02 * torch.randn_like(x)
        return x + noise

    def _mask_channel(self, x: torch.Tensor, mask_ratio=0.3):
        B, C, L = x.shape
        k = max(1, int(C * mask_ratio))
        idx = torch.randperm(C, device=x.device)[:k]
        x = x.clone()
        x[:, idx] = 0.0
        return x
    ##############################################################################
    ##############################################################################
    def _make_views_mask(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        输入: x (B, C, T) 5分钟片段
        输出: (globals, locals)
          - globals: [view1, view2]，各自形状都是 (B*10, C, 3840)
          - locals:  []（可为空）
        逻辑：
          1) 切成 30s 小窗并展平 batch 维度
          2) 分别对两份拷贝在“非全零通道”中随机屏蔽 mask_ratio 比例的通道
        """
        B, C, T = x.shape
        assert T % 3840 == 0, "Expect T to be multiple of 3840 (30s @ 128Hz)"
        num_epochs = T // 3840

        # (B, C, T) -> (B, C, 10, 3840) -> (B, 10, C, 3840) -> (B*10, C, 3840)
        x_30 = x.view(B, C, num_epochs, 3840).permute(0, 2, 1, 3).contiguous()
        x_30 = x_30.view(B * num_epochs, C, 3840)

        # 两个 view：对“非全零通道”做随机通道屏蔽（各自独立采样）
        v1 = self._mask_existing_channels(x_30, mask_ratio=getattr(self, "global_mask_ratio", 0.3), mode="zero")
        v2 = self._mask_existing_channels(x_30, mask_ratio=getattr(self, "global_mask_ratio", 0.3), mode="zero")

        globals_ = [v1]
        locals_ = [v2]  # 你如果后面要加 local，就在这里扩展
        return globals_, locals_

    def _mask_existing_channels(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.5,
        mode: str = "zero",  # "zero" | "noise" | "swap" | "mean"
    ) -> torch.Tensor:
        """
        仅在“非全零”的通道集合里随机屏蔽一定比例通道。
        x: (B2, C, L)  —— 这里的 B2 = B * num_epochs
        """
        B2, C, L = x.shape
        out = x.clone()

        # 判定每个样本的每个通道是否“存在”（不是整段全零）
        # 也可以用方差阈值：x.var(dim=-1) > eps
        existing = (x.abs().sum(dim=-1) > 0)  # (B2, C) bool

        # 每个样本按 existing 的数量来决定要 mask 的通道数
        num_exist = existing.sum(dim=-1)                     # (B2,)
        k = (num_exist.float() * mask_ratio).clamp(min=1).to(torch.long)

        # 逐样本采样通道索引（易读；如果你担心速度，可进一步矢量化）
        for b in range(B2):
            if num_exist[b] == 0:
                continue  # 全零样本，跳过
            cand = existing[b].nonzero(as_tuple=False).squeeze(1)  # 可候选通道索引
            num = min(k[b].item(), cand.numel())
            idx = cand[torch.randperm(cand.numel(), device=x.device)[:num]]

            if mode == "zero":
                out[b, idx] = 0.0
            elif mode == "noise":
                # 用该通道自身的 std 采样噪声（更稳）
                std = x[b, idx].std(dim=-1, keepdim=True).clamp_min(1e-6)
                out[b, idx] = torch.randn_like(x[b, idx]) * std
            elif mode == "swap":
                perm = idx[torch.randperm(num, device=x.device)]
                out[b, idx] = x[b, perm]
            elif mode == "mean":
                mu = x[b, idx].mean(dim=-1, keepdim=True)
                out[b, idx] = mu
            else:
                raise ValueError(f"Unknown mask mode: {mode}")

        return out

    
    def _teacher_temp(self, step: int):
        if step < self.teacher_temp_warmup_iters:
            
            alpha = step / float(max(1, self.teacher_temp_warmup_iters))
            return self.teacher_temp_warmup * (1 - alpha) + self.teacher_temp_final * alpha
        return self.teacher_temp_final

    def _momentum(self, step: int, max_steps: int):

        return 1.0 - (1.0 - self.base_momentum) * (math.cos(math.pi * step / max_steps) + 1) / 2

    @torch.no_grad()
    def _ema_update(self, m: float):

        for param_q, param_k in zip(self.encoders['all'].parameters(), self.teacher_encoder.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
        for param_q, param_k in zip(self.student_global_head.parameters(), self.teacher_global_head.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
        for param_q, param_k in zip(self.student_patch_head.parameters(), self.teacher_patch_head.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

    def _forward_encoder(self, encoder, x, return_tokens=True):
        """
        return {'cls': (B,D), 'patch': (B,N,D)}
        
        """
        try:
            out = encoder(x, return_tokens=return_tokens)
            
            if isinstance(out, dict):
                cls = out.get('cls', None)
                patch = out.get('patch', None)
            elif isinstance(out, tuple) and len(out) == 2:
                cls, patch = out
            else:
                cls, patch = out, None
        except TypeError:
            cls, patch = encoder(x), None
        return cls, patch

    # --------------- Lightning hooks ---------------
    def on_train_batch_start(self, batch, batch_idx):
        self._step_start = time.perf_counter()

    @torch.no_grad()
    def on_train_batch_end(self, *args, **kwargs):

        duration = time.perf_counter() - self._step_start

    # --------------- 训练核心（shared_step）---------------
    def shared_step(self, batch, batch_idx):
        x = batch['psg']  # (B, C, T)
        B = x.size(0)

        # 生成多视角
        globals_x, locals_x = self._make_views(x)   # list of (B,C,3840)
        
        # ===== Teacher 前向（只看全局视角；no grad、eval 行为）=====
        with torch.no_grad():
            teacher_temp = self._teacher_temp(int(self.global_step))
            teacher_out_soft_list = []
            teacher_patch_soft_list = []
            for gx in globals_x:
                cls_t, patch_t = self._forward_encoder(self.teacher_encoder, gx, return_tokens=True)
                gt_logits = self.teacher_global_head(cls_t)              # (B, out_dim)
                gt_soft   = self.dino_loss.softmax_center_teacher(gt_logits, teacher_temp)
                teacher_out_soft_list.append(gt_soft)

                if patch_t is not None:
                    pt_logits = self.teacher_patch_head(patch_t)         # (B, N, patch_out_dim)
                    pt_soft   = self.ibot_loss.softmax_center_teacher(pt_logits, teacher_temp)
                    teacher_patch_soft_list.append(pt_soft)

        # ===== Student 前向（看全局 + 局部）=====
        student_global_logits = []
        for sx in locals_x:
            cls_s, _ = self._forward_encoder(self.encoders['all'], sx, return_tokens=False)
            sg_logits = self.student_global_head(cls_s)                  # (B, out_dim)
            student_global_logits.append(sg_logits)

        # iBOT：只对 *某一个* student 视角做 masked patch（可选更多视角）
        ibot_loss_val = torch.tensor(0.0, device=x.device)
        used_mask = None
        if self.ibot_lambda > 0:
            # 选第一路 global 视角做 patch 学习
            sx = globals_x[0]
            cls_s, patch_s = self._forward_encoder(self.encoders['all'], sx, return_tokens=True)
            if patch_s is not None:
                # 随机 mask 若干 patch
                B, N, D = patch_s.shape
                mask_ratio = 0.3
                n_mask = max(1, int(N * mask_ratio))
                masks = torch.zeros(B, N, dtype=torch.bool, device=x.device)
                for b in range(B):
                    idx = torch.randperm(N, device=x.device)[:n_mask]
                    masks[b, idx] = True
                used_mask = masks

                # 只取被 mask 的位置送 head
                s_logits = self.student_patch_head(patch_s)              # (B, N, patch_out_dim)
                s_logits_masked = s_logits[masks]                        # (B*n_mask, patch_out_dim)

                # teacher patch target（取 teacher 的第一个全局视角）
                if len(teacher_patch_soft_list) > 0:
                    t_soft = teacher_patch_soft_list[0]                  # (B, N, patch_out_dim) soft target
                    t_soft_masked = t_soft[masks]                        # (B*n_mask, patch_out_dim)
                    # 这里用 forward_masked 处理权重与归一
                    ibot_loss_val = self.ibot_loss.forward_masked(
                        student_patch_tokens_masked=s_logits_masked,
                        teacher_patch_tokens_masked=t_soft_masked,
                        student_masks_flat=masks
                    )

        # ===== DINOLoss（全局）=====
        dino_loss_val = self.dino_loss(student_global_logits, teacher_out_soft_list)

        # ===== KoLeo（spread 正则），对 student 的全局“bottleneck特征” 或 直接用 encoder CLS =====
        koleo_val = torch.tensor(0.0, device=x.device)
        if self.koleo is not None and self.koleo_lambda > 0:
            # 取第一路 student 全局视角的 CLS 经过 MLP 前的特征做 spread
            # 这里偷个懒：用 student_global_head 的输入（传个 hook 会更准，这里直接再跑一次 MLP 前层不方便）
            # 简化：对 student encoder 的 cls_s 做 normalize 后用 KoLeo
            cls_s_first, _ = self._forward_encoder(self.encoders['all'], globals_x[0], return_tokens=False)
            koleo_val = self.koleo(cls_s_first)

        total_loss = dino_loss_val + self.ibot_lambda * ibot_loss_val + self.koleo_lambda * koleo_val

        # ===== 更新 center（异步归约已在 loss 内处理）=====
        with torch.no_grad():
            # teacher 全局 logits（未 softmax/center）用来更新 center
            # 这里复用上面 teacher 前向的 logits：为了简洁，这里再前向一次拿 logits
            teacher_logits_for_center = []
            for gx in globals_x:
                cls_t, _ = self._forward_encoder(self.teacher_encoder, gx, return_tokens=False)
                teacher_logits_for_center.append(self.teacher_global_head(cls_t))
            t_concat = torch.cat(teacher_logits_for_center, dim=0)
            self.dino_loss.update_center(t_concat)

            # patch center 更新（如有 patch）
            if used_mask is not None and len(globals_x) > 0:
                cls_t, patch_t = self._forward_encoder(self.teacher_encoder, globals_x[0], return_tokens=True)
                if patch_t is not None:
                    t_patch_logits = self.teacher_patch_head(patch_t)  # (B, N, P)
                    self.ibot_loss.update_center(t_patch_logits)

        # ===== 记录指标 =====
        metrics = {
            "loss/total": total_loss,
            "loss/dino": dino_loss_val,
            "loss/ibot": ibot_loss_val,
            "loss/koleo": koleo_val,
            "sched/teacher_temp": torch.tensor(self._teacher_temp(int(self.global_step)), device=x.device),
        }

        return {"loss": total_loss}, metrics

    def training_step(self, batch, batch_idx):
        loss_dict, metrics = self.shared_step(batch, batch_idx)
        for k, v in metrics.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=("total" in k), sync_dist=True)

        with torch.no_grad():
            max_steps = max(1, getattr(self.trainer, "max_steps", 100000))
            m = self._momentum(int(self.global_step), max_steps)
            self._ema_update(m)
            self.log("sched/momentum", torch.tensor(m, device=self.device), on_step=True, prog_bar=False)
        return loss_dict["loss"]


    def validation_step(self, batch, batch_idx):
        return None
