import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from cskd.models.vision_transformer import Block
import torch.nn.functional as F
from distillers.registry import register_distiller


@register_distiller
class SimiKD(nn.Module):
    def __init__(self, cfg, model, criterion, teacher_feat_dim, embed_dim):
        super().__init__()
        self.cfg = cfg
        self.model = model  # block
        self.criterion = criterion
        self.feature_aligner = nn.Sequential(
            nn.Conv2d(teacher_feat_dim, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.GELU()
        )

    def align_features(self, x):
        # x: B C H W (教师网络特征)
        B, C, H, W = x.shape

        # 上采样到patch_size的整数倍
        target_H = 14
        target_W = 14
        if H != target_H or W != target_W:
            x = F.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False, antialias=True)
        # print(x.shape)  # torch.Size([64, 3024, 14, 14])
        # 通道对齐并转换为序列
        x = self.feature_aligner(x)  # torch.Size([64, 192, 14, 14])
        x = x.flatten(2).transpose(1, 2)  # B HW C torch.Size([64, 196, 192])
        return x

    def forward(self, labels, outputs_t=None, feat_t=None):
        x = self.align_features(feat_t)
        outputs = self.model(x)
        if self.training:
            outputs, stu_deit_logits, _, _ = outputs
            loss_base = self.criterion(outputs, labels)
            loss_deit = self.get_loss_deit(stu_deit_logits, outputs_t)
            return loss_deit, loss_base, outputs
        else:
            _, qk, vv = outputs
            return qk, vv

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
