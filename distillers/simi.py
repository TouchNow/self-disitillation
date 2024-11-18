import math
import torch
from torch.nn import functional as F
import torch.nn as nn
from cskd.config import ConfigBase
from timm.models.layers import trunc_normal_
from distillers.registry import register_distiller
from cskd.models.vision_transformer import Block
from copy import deepcopy

__all__ = ["SimiKD"]


def set_module_dict(module_dict, k, v):
    if not isinstance(k, str):
        k = str(k)
    module_dict[k] = v


def get_module_dict(module_dict, k):
    if not isinstance(k, str):
        k = str(k)
    return module_dict[k]


def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Alignment(nn.Module):
    def __init__(self, feat_s, feat_t):
        super().__init__()
        assert feat_s.dim() == 3, "feat_s should be a 3D tensor"
        assert feat_t.dim() == 4, "feat_t should be a 4D tensor"
        B, Num, Embed = feat_s.shape
        B, C, H, W = feat_t.shape
        self.channel_align = nn.Sequential(
            nn.Conv2d(Embed, C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(C),
            # nn.ReLU(),
            nn.GELU(),
        )
        # self.activate = nn.Sequential(
        #     nn.LayerNorm(H * W),
        #     nn.GELU(),
        # )
        # nn.LayerNorm(H*W),

        self.spatial_align = nn.Sequential(
            nn.Linear(Num, H * W),
            # nn.ReLU(),
            nn.GELU(),
        )

    def forward(self, x, feat_t):
        B, C, H, W = feat_t.shape
        x = x.permute(0, 2, 1)  # B Embed Num
        x = self.spatial_align(x)  # B Embed HW
        x = x.reshape(B, -1, H, W)  # B Embed H W
        x = self.channel_align(x)  # B C H W
        # x = x.flatten(2)  # B C HW
        # x = self.activate(x)
        x = x.flatten(1)  # B CHW
        return x


@register_distiller
class SimiKD(nn.Module):
    def __init__(self, cfg, model, criterion, feat_t):
        super().__init__()
        self.cfg = cfg
        self.model = model  # block
        self.criterion = criterion

    def forward(self, inputs, labels, outputs_t, feat_s=None, feat_t=None):
        outputs = self.model(inputs)
        if not isinstance(outputs, torch.Tensor):
            outputs, stu_deit_logits, qk_list, vv_list = outputs
        loss_base = self.criterion(outputs, labels)
        if self.cfg.deit_loss_type == "none":  # no distill loss
            return loss_base
        loss_qk = self.attkd_loss(qk_list)
        loss_vv = self.attkd_loss(vv_list)
        loss_deit = self.get_loss_deit(stu_deit_logits, outputs_t)

        return loss_deit, loss_base, outputs, loss_qk, loss_vv

    def attkd_loss(self, relation):
        loss = []
        for i in range(len(relation) - 1):
            if i in self.cfg.feat_loc:
                temp = nn.KLDivLoss(reduction="none")(relation[i].log(), relation[-1]).sum(-1)
                loss.append(temp.mean())
        return sum(loss)

    def get_mse_loss(self, feat_s, feat_t):
        feat_t = feat_t.flatten(1)
        return nn.MSELoss()(feat_s, feat_t) / feat_s.size(0)

    def get_cosine_loss(self, feat_s, feat_t):
        feat_t = feat_t.flatten(1)
        return 1 - F.cosine_similarity(feat_s, feat_t, dim=1).mean()

    def get_loss_deit(self, stu_deit_logits, tea_global_logits):
        # deit loss
        if self.cfg.deit_loss_type == "soft":
            T = self.cfg.deit_tau
            loss_deit = (
                F.kl_div(
                    F.log_softmax(stu_deit_logits / T, dim=1),
                    F.log_softmax(tea_global_logits / T, dim=1),
                    reduction="sum",
                    log_target=True,
                )
                * (T * T)
                / stu_deit_logits.numel()
            )
        elif self.cfg.deit_loss_type == "hard":
            loss_deit = F.cross_entropy(stu_deit_logits, tea_global_logits.argmax(dim=1))
        else:
            raise NotImplementedError(self.cfg.deit_loss_type)
        return loss_deit

    def get_decay_ratio(self, epoch, max_epoch):
        x = epoch / max_epoch
        if self.cfg.cskd_decay_func == "linear":
            return 1 - x
        elif self.cfg.cskd_decay_func == "x2":
            return (1 - x) ** 2
        elif self.cfg.cskd_decay_func == "cos":
            return math.cos(math.pi * 0.5 * x)
        else:
            raise NotImplementedError(self.cfg.cskd_decay_func)
