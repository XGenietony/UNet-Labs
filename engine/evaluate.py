import torch
import torch.nn.functional as F
from tqdm import tqdm

from losses.dice_score import multiclass_dice_coeff, dice_coeff


def fuse_boundary_skeleton(boundary_logits, skeleton_logits, alpha=0.5, dilation_kernel=3):
    """把 boundary / skeleton 两个分支的输出融合成一张概率图。

    与各原始工程 (DCFENet-Perlin) 中的实现保持一致。
    boundary_logits / skeleton_logits: (B,1,H,W) logits。
    """
    P_boundary = torch.sigmoid(boundary_logits)
    P_skeleton = torch.sigmoid(skeleton_logits)

    if dilation_kernel > 1:
        pad = dilation_kernel // 2
        kernel = torch.ones(1, 1, dilation_kernel, dilation_kernel, device=P_skeleton.device)
        P_skeleton = F.conv2d(P_skeleton, kernel, padding=pad)
        P_skeleton = torch.clamp(P_skeleton, 0, 1)

    P_final = P_boundary + alpha * P_skeleton
    P_final = torch.clamp(P_final, 0, 1)
    return P_final


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    """训练期验证：返回平均 Dice。

    自动区分两类模型输出：
      * dict {"final","boundary","skeleton"}  → 走 boundary/skeleton 融合路径（dcfe / dual_branch）
      * 单张 logits tensor                     → 走标准 UNet 路径（unet / resnet_unet / feature_fusion）
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            out = net(image)

            # ---------- 双分支模型：dict 输出 ----------
            if isinstance(out, dict):
                mask_pred = fuse_boundary_skeleton(
                    out['boundary'], out['skeleton'], alpha=0.5, dilation_kernel=3
                )
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (mask_pred > 0.5).float()
                if mask_pred.dim() == 4 and mask_true.dim() == 3:
                    mask_true = mask_true.unsqueeze(1)
                if mask_pred.shape != mask_true.shape:
                    mask_true = mask_true[:, :mask_pred.shape[1], ...]
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                continue

            # ---------- 标准模型：单 logits 输出 ----------
            mask_pred = out
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                if mask_pred.dim() == 4 and mask_true.dim() == 3:
                    mask_true = mask_true.unsqueeze(1)
                if mask_pred.shape != mask_true.shape:
                    mask_true = mask_true[:, :mask_pred.shape[1], ...]
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
