from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLabelLoss(nn.Module):
    """
    软标签蒸馏损失：
    使用 KL 散度在温度 T 下对学生 / 教师的类别分布进行对齐。
    """

    def __init__(self, temperature: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        # KLDivLoss 期望输入为 log-prob，target 为 prob
        self.criterion = nn.KLDivLoss(reduction=reduction)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        # 按标准蒸馏公式：softmax(logits / T)，并乘以 T^2 放缩梯度
        student_log_prob = F.log_softmax(student_logits / T, dim=1)
        teacher_prob = F.softmax(teacher_logits / T, dim=1)
        loss_kl = self.criterion(student_log_prob, teacher_prob) * (T * T)
        return loss_kl


class FeatureAlignmentLoss(nn.Module):
    """
    特征对齐损失：
    学生模型经过 Adaptation Layer 后的特征 vs 教师对应特征，使用 MSE。
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        # 通道应由 Adaptation Layer 对齐；空间尺寸允许不一致，这里自动对齐到学生尺度
        if teacher_feat.shape[-2:] != student_feat.shape[-2:]:
            teacher_feat = F.interpolate(
                teacher_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.criterion(student_feat, teacher_feat)


class DistillationLoss(nn.Module):
    """
    多维度蒸馏总损失：

        Loss_total = alpha * Loss_hard (BCE+Lovasz)
                     + beta * Loss_soft (KL)
                     + gamma * Loss_feature (MSE)

    其中：
      - Loss_hard 由外部传入的 hard_loss_fn 计算（通常为 BCE+Lovasz 的组合）
      - Loss_soft 由 SoftLabelLoss 计算
      - Loss_feature 由 FeatureAlignmentLoss 计算
    """

    def __init__(
        self,
        hard_loss_fn: nn.Module,
        temperature: float = 2.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.hard_loss_fn = hard_loss_fn
        self.soft_loss = SoftLabelLoss(temperature=temperature)
        self.feature_loss = FeatureAlignmentLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            student_logits: 学生输出 logits，形状 (N, 1, H, W) 或 (N, C, H, W)
            teacher_logits: 教师输出 logits，形状同上
            student_feat: 学生 Adaptation 后的中间特征图 (N, C_f, H_f, W_f)
            teacher_feat: 教师对应的中间特征图 (N, C_f, H_f, W_f)
            target: 真实 mask，形状 (N, H, W) 或 (N, 1, H, W)

        Returns:
            total_loss, hard_loss, soft_loss, feat_loss
        """

        loss_hard = self.hard_loss_fn(student_logits, target)
        loss_soft = self.soft_loss(student_logits, teacher_logits)
        loss_feat = self.feature_loss(student_feat, teacher_feat)

        total = self.alpha * loss_hard + self.beta * loss_soft + self.gamma * loss_feat
        return total, loss_hard, loss_soft, loss_feat

