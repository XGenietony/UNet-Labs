import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from typing import Optional
from torch import Tensor


class FocalLoss(_WeightedLoss):
    r"""
    Focal Loss for binary classification with rare positive samples.

    Args:
        alpha (float): weight for positive class (0 < alpha < 1)
        gamma (float): focusing parameter
        reduction (str): 'none' | 'mean' | 'sum'
    """

    def __init__(
        self,
        alpha: float = 0.9,
        gamma: float = 2.0,
        reduction: str = 'sum'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: logits, shape (N, ...) or (N, 1, ...)
        target: {0,1}, same shape as input
        """

        # ======== 关键修复点 ========
        if input.dim() == target.dim() + 1:
            # (N, C, H, W)
            if input.size(1) == 2:
                # 取正类 logits（index=1）
                input = input[:, 1, ...]
            elif input.size(1) == 1:
                input = input.squeeze(1)
            else:
                raise ValueError(f"Unsupported channel size: {input.size(1)}")
        # ==========================

        target = target.float()

        # sigmoid 概率
        p = torch.sigmoid(input)

        # 正负样本分别计算
        pos_loss = -self.alpha * (1 - p) ** self.gamma * torch.log(p + 1e-8)
        neg_loss = -(1 - self.alpha) * p ** self.gamma * torch.log(1 - p + 1e-8)

        loss = torch.where(target == 1, pos_loss, neg_loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            num_pos = (target == 1).sum().clamp_min(1)
            loss = loss.sum() / num_pos
            # loss = loss.sum()
            return loss
        else:
            return loss

class BoundaryLoss(nn.Module):
    """
    Boundary Region Loss = BCE + Dice
    Encourage recall and region coverage
    """

    def __init__(self, dice_weight: float = 1.0, thin_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.thin_weight = thin_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: logits (N,1,H,W) or (N,H,W)
        target: {0,1}
        """
        # -------- shape align --------
        if input.dim() == target.dim() + 1:
            input = input.squeeze(1)

        target = target.float()

        # -------- BCE --------
        bce = F.binary_cross_entropy_with_logits(input, target)

        # -------- Dice --------
        prob = torch.sigmoid(input)
        prob = prob.contiguous().view(prob.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)

        intersection = (prob * target_flat).sum(dim=1)
        union = prob.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()

        # 🔥 Thin penalty（关键）
        thin_loss = prob.mean()

        return bce + self.dice_weight * dice_loss + self.thin_weight * thin_loss

def soft_skeletonize(x, iters=10):
    for _ in range(iters):
        min_pool = -F.max_pool2d(-x, 3, stride=1, padding=1)
        contour = F.relu(min_pool - x)
        x = F.relu(x - contour)
    return x

class clDiceLoss(nn.Module):
    def __init__(self, iters: int = 10):
        super().__init__()
        self.iters = iters

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: logits (N,1,H,W)
        target: {0,1}
        """
        if input.dim() == target.dim() + 1:
            input = input.squeeze(1)

        prob = torch.sigmoid(input)
        target = target.float()

        skel_pred = soft_skeletonize(prob, self.iters)
        skel_gt = soft_skeletonize(target, self.iters)

        eps = 1e-6
        tprec = (skel_pred * target).sum() / (skel_pred.sum() + eps)
        tsens = (skel_gt * prob).sum() / (skel_gt.sum() + eps)

        cldice = 2 * tprec * tsens / (tprec + tsens + eps)
        return 1 - cldice

class SkeletonLoss(nn.Module):
    """
    Skeleton Loss = Focal + clDice
    Enforce connectivity and continuity
    """

    def __init__(
        self,
        alpha: float = 0.9,
        gamma: float = 2.0,
        cldice_weight: float = 1.0
    ):
        super().__init__()
        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction='mean'
        )
        self.cldice = clDiceLoss()
        self.cldice_weight = cldice_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        focal = self.focal(input, target)
        cldice = self.cldice(input, target)
        return focal + self.cldice_weight * cldice
