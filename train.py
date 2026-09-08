"""统一训练入口。

通过 `--model` 选择网络，通过参数控制每个实验：
  * K 折交叉验证  : --fold N --folds K
  * 固定训练/测试 : --fixed-split
  * 随机划分      : 默认（--validation 控制验证集比例）
  * Perlin 噪声   : --perlin / --no-perlin，--perlin-prob
  * 损失          : --loss {auto,bce,ce,focal}（双分支模型固定用 composite 复合损失）
  * 多随机种子    : --seed（配合 scripts/run_seeds.sh 报告均值±标准差）

示例：
  python train.py --model dcfe        --classes 1 --fold 0 --epochs 120 -b 8 -l 1e-3 --exp dcfe_fold0
  python train.py --model unet        --classes 1 --epochs 120 -b 8 -l 1e-3 --exp unet
  python train.py --model resnet_unet --classes 2 --fixed-split --epochs 120 -b 4 -l 1e-4 --exp resnet
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from engine.evaluate import evaluate, fuse_boundary_skeleton
from models import build_model, output_type, MODEL_NAMES
from data_utils.resize_dataset import ResizeDataset, IMG_EXTS
from losses.dice_score import dice_loss
from losses.loss_function import FocalLoss, BoundaryLoss, SkeletonLoss

try:
    import wandb
except Exception:            # wandb 可选
    wandb = None

# 与各原始工程一致：强制确定性，保证可复现
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------- #
#  数据集构建（K 折 / 固定划分 / 随机划分）
# ---------------------------------------------------------------------- #
def build_datasets(args, with_skeleton):
    data_root = Path(args.data_root)
    dir_img = data_root / 'imgs'
    dir_mask = data_root / 'label'
    dir_test_img = data_root / 'test' / 'image'
    dir_test_mask = data_root / 'test' / 'label'
    common = dict(perlin=args.perlin, perlin_prob=args.perlin_prob,
                  edge_width=args.edge_width, with_skeleton=with_skeleton, size=args.size)

    if args.fold is not None:
        # K 折交叉验证：合并 train + test 后按 fold 跨步切分（与原工程一致）
        all_imgs = sorted([p for p in dir_img.iterdir() if p.suffix.lower() in IMG_EXTS]) + \
                   sorted([p for p in dir_test_img.iterdir() if p.suffix.lower() in IMG_EXTS])
        all_masks = sorted([p for p in dir_mask.iterdir() if p.suffix.lower() in IMG_EXTS]) + \
                    sorted([p for p in dir_test_mask.iterdir() if p.suffix.lower() in IMG_EXTS])
        val_idx = list(range(args.fold, len(all_imgs), args.folds))
        vs = set(val_idx)
        train_idx = [i for i in range(len(all_imgs)) if i not in vs]
        train_set = ResizeDataset(img_paths=[all_imgs[i] for i in train_idx],
                                  mask_paths=[all_masks[i] for i in train_idx],
                                  augment=True, repeat=args.repeat, **common)
        val_set = ResizeDataset(img_paths=[all_imgs[i] for i in val_idx],
                                mask_paths=[all_masks[i] for i in val_idx],
                                augment=False, repeat=1, **common)
    elif args.fixed_split:
        # 固定：imgs/label 训练，test/ 验证
        train_set = ResizeDataset(dir_img, dir_mask, augment=True, repeat=args.repeat, **common)
        val_set = ResizeDataset(dir_test_img, dir_test_mask, augment=False, repeat=1, **common)
    else:
        # 随机划分（seed 固定为 0，与原工程一致）
        dataset = ResizeDataset(dir_img, dir_mask, augment=True, repeat=args.repeat, **common)
        n_val = int(len(dataset) * args.val / 100)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val],
                                          generator=torch.Generator().manual_seed(0))
    return train_set, val_set


# ---------------------------------------------------------------------- #
#  损失
# ---------------------------------------------------------------------- #
def make_logits_criterion(loss_name, n_classes):
    if loss_name == 'focal':
        return FocalLoss()
    if loss_name == 'ce' or (loss_name == 'auto' and n_classes > 1):
        return nn.CrossEntropyLoss()
    return nn.BCEWithLogitsLoss()            # bce / auto+二分类


def loss_logits(out, true_masks, criterion, n_classes):
    """标准 U-Net 路径：CE/BCE/Focal + dice_loss。"""
    if n_classes == 1:
        loss = criterion(out.squeeze(1), true_masks.float())
        loss = loss + dice_loss(F.sigmoid(out.squeeze(1)), true_masks.float(), multiclass=False)
    else:
        loss = criterion(out, true_masks)
        loss = loss + dice_loss(
            F.softmax(out, dim=1).float(),
            F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True,
        )
    return loss


def loss_dict(out, true_masks, true_skeleton, boundary_criterion, skeleton_criterion, epoch, device):
    """双分支复合损失（移植自 DCFENet-Perlin）：
    0.5*boundary + w_skel*skeleton + w_dice*dice + 0.3*bce + 0.2*fp_penalty，
    其中 skeleton / dice 权重按 epoch/100 做 warm-up（上限 0.2 / 0.1）。
    """
    loss_boundary = boundary_criterion(out['boundary'], true_masks)
    loss_skeleton = skeleton_criterion(out['skeleton'], true_skeleton)

    w_skeleton = (epoch / 100.0) * 0.2 if epoch < 100 else 0.2
    w_dice = (epoch / 100.0) * 0.1 if epoch < 100 else 0.1

    final = out['final'].squeeze(1)
    loss_dice = dice_loss(F.sigmoid(final), true_masks.float(), multiclass=False)
    loss_bce = F.binary_cross_entropy_with_logits(
        final, true_masks.float(), pos_weight=torch.tensor(2.0, device=device))
    fp_penalty = torch.mean(torch.relu(F.sigmoid(final) - true_masks.float()))

    loss = (0.5 * loss_boundary + w_skeleton * loss_skeleton +
            w_dice * loss_dice + 0.3 * loss_bce + 0.2 * fp_penalty)
    return loss, loss_boundary, loss_skeleton, loss_dice


# ---------------------------------------------------------------------- #
#  训练主循环
# ---------------------------------------------------------------------- #
def train_model(model, device, args, dir_checkpoint):
    otype = output_type(args.model)
    is_dict = (otype == 'dict')
    n_classes = model.n_classes

    train_set, val_set = build_datasets(args, with_skeleton=is_dict)
    n_train, n_val = len(train_set), len(val_set)

    loader_args = dict(batch_size=args.batch_size, num_workers=min(8, os.cpu_count() or 1), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    experiment = None
    if wandb is not None:
        experiment = wandb.init(project='UNet-Lab', name=args.exp, resume='allow',
                                anonymous='must', mode='offline')
        experiment.config.update(vars(args))

    logging.info(f'''Starting training:
        Model:           {args.model}  (output={otype})
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Mixed Precision: {args.amp}
        Perlin:          {args.perlin} (p={args.perlin_prob})
    ''')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5, threshold=1e-3, min_lr=3e-6)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # 损失准则
    criterion = make_logits_criterion(args.loss, n_classes) if not is_dict else None
    boundary_criterion = BoundaryLoss() if is_dict else None
    skeleton_criterion = SkeletonLoss(alpha=0.9, gamma=2.0) if is_dict else None

    global_step = 0
    from tqdm import tqdm
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.long)

                assert images.shape[1] == model.n_channels, \
                    f'Network expects {model.n_channels} input channels but got {images.shape[1]}.'

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                    out = model(images)
                    if is_dict:
                        # --skeleton-gt: 'skeleton' 用真骨架 GT（推荐）；'mask' 复现原工程行为
                        if args.skeleton_gt == 'skeleton':
                            true_skeleton = batch['skeleton'].to(device=device, dtype=torch.float32)
                        else:
                            true_skeleton = true_masks.to(device=device, dtype=torch.float32)
                        loss, l_b, l_s, l_d = loss_dict(
                            out, true_masks, true_skeleton,
                            boundary_criterion, skeleton_criterion, epoch, device)
                    else:
                        loss = loss_logits(out, true_masks, criterion, n_classes)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                if experiment is not None:
                    experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})

        val_score = evaluate(model, val_loader, device, args.amp)
        scheduler.step(val_score)
        logging.info(f'Epoch {epoch}: validation Dice {float(val_score):.4f}  (mean train loss {epoch_loss / max(len(train_loader),1):.4f})')
        if experiment is not None:
            experiment.log({'validation Dice': float(val_score),
                            'learning rate': optimizer.param_groups[0]['lr'], 'epoch': epoch})

        if args.save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved to {dir_checkpoint}')


def get_args():
    p = argparse.ArgumentParser(description='UNet-Lab 统一训练入口')
    p.add_argument('--model', type=str, default='unet', choices=MODEL_NAMES,
                   help=f'选择实验模型：{MODEL_NAMES}')
    p.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='训练轮数')
    p.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='批大小')
    p.add_argument('--learning-rate', '-l', dest='lr', metavar='LR', type=float, default=1e-5, help='学习率')
    p.add_argument('--load', '-f', type=str, default=False, help='从 .pth 加载权重')
    p.add_argument('--scale', '-s', type=float, default=1.0, help='（保留）图像缩放因子')
    p.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='验证集比例(0-100)')
    p.add_argument('--amp', action='store_true', default=False, help='混合精度')
    p.add_argument('--bilinear', action='store_true', default=False, help='使用双线性上采样')
    p.add_argument('--classes', '-c', type=int, default=1, help='类别数（边界分割一般为 1）')
    p.add_argument('--weight-decay', '-w', dest='wd', metavar='WD', type=float, default=5e-6, help='权重衰减')
    p.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪阈值')
    p.add_argument('--pretrained', action='store_true', default=False, help='backbone 使用 ImageNet 预训练')
    p.add_argument('--seed', type=int, default=0, help='随机种子（多次实验报告标准差用 0/1/2）')
    p.add_argument('--exp', type=str, default='exp', help='实验名，checkpoint 存到 checkpoints/<exp>/')
    # 数据 / 划分
    p.add_argument('--data-root', type=str, default='data', help='数据根目录（含 imgs/ label/ test/）')
    p.add_argument('--fold', type=int, default=None, help='K 折的第几折（0..folds-1），不指定则不做交叉验证')
    p.add_argument('--folds', type=int, default=5, help='总折数')
    p.add_argument('--fixed-split', action='store_true', default=False, help='固定：imgs/label 训练、test/ 验证')
    p.add_argument('--size', type=int, default=512, help='统一 resize 边长')
    p.add_argument('--repeat', type=int, default=4, help='训练集每张图重复采样次数')
    p.add_argument('--edge-width', type=int, default=1, help='mask→边界 形态学宽度')
    # 增强
    p.add_argument('--perlin', dest='perlin', action='store_true', default=True, help='启用 Perlin 噪声增强（默认开）')
    p.add_argument('--no-perlin', dest='perlin', action='store_false', help='关闭 Perlin 噪声增强')
    p.add_argument('--perlin-prob', type=float, default=0.3, help='Perlin 噪声触发概率')
    # 损失
    p.add_argument('--loss', type=str, default='auto', choices=['auto', 'bce', 'ce', 'focal'],
                   help='标准模型损失（双分支模型固定 composite 复合损失）')
    p.add_argument('--skeleton-gt', type=str, default='skeleton', choices=['skeleton', 'mask'],
                   help="双分支骨架分支监督用真骨架GT('skeleton')还是复现原工程('mask')")
    p.add_argument('--no-save', dest='save_checkpoint', action='store_false', default=True, help='不保存 checkpoint')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    set_seed(args.seed)
    logging.info(f'Random seed: {args.seed}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = build_model(args.model, n_channels=3, n_classes=args.classes,
                        bilinear=args.bilinear, pretrained=args.pretrained)
    logging.info(f'Model: {args.model} | in={model.n_channels} out={model.n_classes} '
                 f'bilinear={getattr(model, "bilinear", "n/a")}')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    dir_checkpoint = Path(f'./checkpoints/{args.exp}/')
    train_model(model, device, args, dir_checkpoint)
