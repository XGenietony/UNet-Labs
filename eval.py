"""
推理 + 评测脚本 (UNet)

用法：
  # 评测单个 checkpoint
  python eval.py unet_kfold_fold0 \
      --ckpt checkpoints/unet_kfold_fold0/checkpoint_epoch10.pth \
      --tolerance-precision 0 --tolerance-recall 2

  # 遍历整个 checkpoint 目录，输出每 epoch 指标
  python eval.py unet_kfold_fold0 \
      --img-dir data/test/image --mask-dir data/test/label
"""

import argparse
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import build_model, output_type
from engine.evaluate import fuse_boundary_skeleton
from predict import mask_to_wireframe

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
EPS = 1e-7
BATCH_SIZE = 8
NUM_WORKERS = 4


# ───────────────────────────── Dataset ─────────────────────────────

class TestDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        W, H = img.size
        x = torch.from_numpy(
            np.array(img).astype(np.float32).transpose(2, 0, 1) / 255.0
        )
        x = F.interpolate(x.unsqueeze(0), size=(512, 512),
                          mode='bilinear', align_corners=False).squeeze(0)
        return x, H, W, path.name


# ───────────────────────────── 指标 ─────────────────────────────

def _dilate(arr, tol):
    if tol > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
        return cv2.dilate(arr, k)
    return arr


def tolerant_precision(pred_bin, gt_bin, tol_p):
    """预测像素落在 GT 膨胀范围内算 TP，否则算 FP。"""
    gt_dil = _dilate(gt_bin, tol_p)
    TP = int(np.sum((pred_bin == 1) & (gt_dil == 1)))
    FP = int(np.sum((pred_bin == 1) & (gt_dil == 0)))
    return TP / (TP + FP + EPS), TP, FP


def tolerant_recall(pred_bin, gt_bin, tol_r):
    """GT 像素落在预测膨胀范围内算 TP，否则算 FN。"""
    pred_dil = _dilate(pred_bin, tol_r)
    TP = int(np.sum((gt_bin == 1) & (pred_dil == 1)))
    FN = int(np.sum((gt_bin == 1) & (pred_dil == 0)))
    return TP / (TP + FN + EPS), TP, FN


def pixel_iou(pred_bin, gt_bin):
    inter = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    union = int(np.sum((pred_bin == 1) | (gt_bin == 1)))
    return inter / (union + EPS)


def pixel_dice(pred_bin, gt_bin):
    inter = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
    return (2 * inter) / (pred_bin.sum() + gt_bin.sum() + EPS)


def tolerant_f1(pred_bin, gt_bin, tol_p, tol_r):
    prec, _, _ = tolerant_precision(pred_bin, gt_bin, tol_p)
    rec,  _, _ = tolerant_recall(pred_bin, gt_bin, tol_r)
    return 2 * prec * rec / (prec + rec + EPS), prec, rec


def boundary_iou(pred_bin, gt_bin, tol=2):
    k3 = np.ones((3, 3), np.uint8)
    pb = cv2.morphologyEx(pred_bin, cv2.MORPH_GRADIENT, k3)
    gb = cv2.morphologyEx(gt_bin,   cv2.MORPH_GRADIENT, k3)
    if tol > 0:
        tk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
        pb_d = cv2.dilate(pb, tk)
        gb_d = cv2.dilate(gb, tk)
    else:
        pb_d, gb_d = pb, gb
    inter = int(np.sum((pb == 1) & (gb_d == 1))) + int(np.sum((gb == 1) & (pb_d == 1)))
    union = int(pb.sum()) + int(gb.sum())
    return inter / (union + EPS)


def ois_f1(pred_prob, gt_bin, tol_p, tol_r, n_thresh=100):
    """Per-image optimal threshold F1."""
    best = 0.0
    for t in np.linspace(0, 1, n_thresh):
        pred_bin = (pred_prob > t).astype(np.uint8)
        f1, _, _ = tolerant_f1(pred_bin, gt_bin, tol_p, tol_r)
        best = max(best, f1)
    return best


def ods_f1(pred_probs, gt_bins, tol_p, tol_r, n_thresh=100):
    """Dataset-level optimal threshold F1.
    找让所有图 per-image F1 均值最大的阈值，与 OIS 定义对齐，保证 ODS <= OIS。"""
    best = 0.0
    for t in np.linspace(0, 1, n_thresh):
        f1_mean = np.mean([
            tolerant_f1((pred_prob > t).astype(np.uint8), gt_bin, tol_p, tol_r)[0]
            for pred_prob, gt_bin in zip(pred_probs, gt_bins)
        ])
        best = max(best, float(f1_mean))
    return best


# ───────────────────────────── 主评测流程 ─────────────────────────────

def load_model(ckpt_path: Path, device, n_classes: int = 2, bilinear: bool = False,
               arch: str = 'unet'):
    net = build_model(arch, n_channels=3, n_classes=n_classes, bilinear=bilinear)
    sd = torch.load(ckpt_path, map_location=device)
    sd = {k: v for k, v in sd.items()
          if not k.endswith(('total_ops', 'total_params', 'mask_values'))}
    net.load_state_dict(sd)
    net.to(device).eval()
    return net


def eval_checkpoint(ckpt_path: Path, img_dir: Path, mask_dir: Path,
                    device, threshold=0.5, tol_p=0, tol_r=0,
                    n_classes=2, bilinear=False,
                    save_pred_dir: Path = None, arch: str = 'unet'):
    net = load_model(ckpt_path, device, n_classes, bilinear, arch)
    is_prob = (output_type(arch) == 'dict')   # 双分支输出经融合后已是概率图

    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    img_paths = [p for p in img_paths if (mask_dir / p.name).exists()]

    loader = DataLoader(
        TestDataset(img_paths),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    rows = []
    pred_probs_all, gt_bins_all = [], []

    if save_pred_dir is not None:
        save_pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch_x, batch_H, batch_W, batch_names in tqdm(
                loader, desc=ckpt_path.name, leave=False):
            batch_x   = batch_x.to(device)
            batch_out = net(batch_x)          # [B, C, 512, 512] 或 dict
            if isinstance(batch_out, dict):   # 双分支：融合成概率图
                batch_out = fuse_boundary_skeleton(batch_out['boundary'], batch_out['skeleton'])

            for i in range(len(batch_names)):
                H, W, name = int(batch_H[i]), int(batch_W[i]), batch_names[i]

                out = F.interpolate(
                    batch_out[i:i+1], size=(H, W),
                    mode='bilinear', align_corners=False
                )
                if is_prob:
                    pred_prob = out.squeeze().cpu().numpy()
                elif n_classes == 1:
                    pred_prob = torch.sigmoid(out).squeeze().cpu().numpy()
                else:
                    pred_prob = torch.softmax(out, dim=1)[:, 1].squeeze().cpu().numpy()

                pred_bin = (pred_prob > 0.5).astype(np.uint8)

                # 线框化：将预测 mask 的轮廓提取为细线，与 GT 细线结构对齐再做指标
                pred_eval = (mask_to_wireframe(pred_bin, thickness=3) > 0).astype(np.uint8)

                gt     = np.array(Image.open(mask_dir / name).convert('L'))
                gt_bin = (gt > 0).astype(np.uint8)
                if gt_bin.shape != pred_eval.shape:
                    gt_bin = cv2.resize(gt_bin, (W, H), interpolation=cv2.INTER_NEAREST)

                prec  = tolerant_precision(pred_eval, gt_bin, 1)[0]
                rec   = tolerant_recall(pred_eval, gt_bin, 2)[0]
                f1    = 2 * prec * rec / (prec + rec + EPS)
                biou  = boundary_iou(pred_bin, gt_bin, tol=1)
                piou  = pixel_iou(pred_bin, gt_bin)
                pdice = pixel_dice(pred_bin, gt_bin)
                oisf1 = ois_f1(pred_prob, gt_bin, 0, 8)

                pred_probs_all.append(pred_prob)
                gt_bins_all.append(gt_bin)
                rows.append(dict(image=name, precision=prec, recall=rec,
                                 f1=f1, ois_f1=oisf1, iou=piou,
                                 dice=pdice, boundary_iou=biou))

                if save_pred_dir is not None:
                    Image.fromarray(pred_bin * 255).save(save_pred_dir / name)

    odsf1 = ods_f1(pred_probs_all, gt_bins_all, 0, 8)
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ('precision', 'recall', 'f1', 'ois_f1', 'iou', 'dice', 'boundary_iou')}
    agg['ods_f1']  = odsf1
    agg['n_images'] = len(rows)
    return agg, rows


# ───────────────────────────── CLI ─────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='评测脚本。最简用法：python eval.py unet_kfold_fold0'
    )
    p.add_argument('exp', help='实验名，对应 checkpoints/<exp>/ 目录')
    p.add_argument('--arch', default='unet',
                   help='网络结构：unet/resnet_unet/feature_fusion/dcfe/dual_branch')
    p.add_argument('--threshold',           type=float, default=0.5)
    p.add_argument('--tolerance-precision', type=int,   default=0,
                   help='tol_p: GT 膨胀半径，影响 Precision/OIS/ODS (default: 0)')
    p.add_argument('--tolerance-recall',    type=int,   default=0,
                   help='tol_r: Pred 膨胀半径，影响 Recall/F1 (default: 0)')
    p.add_argument('--classes',             type=int,   default=2,
                   help='UNet n_classes (default: 2)')
    p.add_argument('--bilinear',            action='store_true', default=False)
    p.add_argument('--ckpt',                type=Path,  default=None,
                   help='单个权值文件；省略则遍历整个 exp 目录')
    p.add_argument('--img-dir',             type=Path,  default=Path('data/test/image'))
    p.add_argument('--mask-dir',            type=Path,  default=Path('data/test/label'))
    p.add_argument('--fold',                type=int,   default=None,
                   help='折号，写入 CSV 的 fold 列')
    p.add_argument('--output-csv',          type=Path,  default=None,
                   help='结果追加写入的 CSV；省略则自动用 results/<exp>_eval.csv')
    return p.parse_args()


def _epoch_from_path(p: Path) -> int:
    m = re.search(r'epoch(\d+)', p.stem)
    return int(m.group(1)) if m else 0


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}  Experiment: {args.exp}')

    ckpt_dir   = Path('checkpoints') / args.exp
    output_csv = args.output_csv or (Path('results') / f'{args.exp}_eval.csv')
    pred_dir   = Path('results') / f'{args.exp}_preds'
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.ckpt is not None:
        ckpt_path = args.ckpt if args.ckpt.is_absolute() else ckpt_dir / args.ckpt
        ckpts = [ckpt_path]
    else:
        ckpts = sorted(
            (p for p in ckpt_dir.glob('*.pth')),
            key=_epoch_from_path
        )
    logging.info(f'Found {len(ckpts)} checkpoints in {ckpt_dir}')

    summary_rows = []

    for ckpt in ckpts:
        epoch = _epoch_from_path(ckpt)

        agg, _ = eval_checkpoint(
            ckpt_path=ckpt,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            device=device,
            threshold=args.threshold,
            tol_p=args.tolerance_precision,
            tol_r=args.tolerance_recall,
            n_classes=args.classes,
            bilinear=args.bilinear,
            save_pred_dir=pred_dir / f'epoch{epoch}',
            arch=args.arch,
        )

        agg['epoch']      = epoch
        agg['checkpoint'] = ckpt.name
        if args.fold is not None:
            agg['fold'] = args.fold
        summary_rows.append(agg)

        logging.info(
            f"epoch={epoch:3d}  "
            f"Dice={agg['dice']:.4f}  IoU={agg['iou']:.4f}  "
            f"Prec={agg['precision']:.4f}  Rec={agg['recall']:.4f}  "
            f"F1={agg['f1']:.4f}  OIS={agg['ois_f1']:.4f}  "
            f"ODS={agg['ods_f1']:.4f}  B-IoU={agg['boundary_iou']:.4f}"
        )

    df = pd.DataFrame(summary_rows).sort_values('epoch')
    cols = ['epoch', 'checkpoint', 'dice', 'iou', 'precision', 'recall',
            'f1', 'ois_f1', 'ods_f1', 'boundary_iou', 'n_images']
    if args.fold is not None:
        cols = ['fold'] + cols
    df = df[[c for c in cols if c in df.columns]]

    df.to_csv(output_csv, index=False)
    logging.info(f'Results saved to {output_csv}')

    best = df.loc[df['ods_f1'].idxmax()]
    print(f"\n{'='*50}")
    print(f"Best epoch by ODS-F1: epoch {int(best['epoch'])}")
    for m in ('dice', 'iou', 'precision', 'recall', 'f1', 'ois_f1', 'ods_f1', 'boundary_iou'):
        print(f"  {m:<14} {float(best[m]):.4f}")


if __name__ == '__main__':
    main()
