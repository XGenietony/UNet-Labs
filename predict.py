import argparse
import logging
import os
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
from skimage.morphology import skeletonize
from typing import Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from models import build_model
from engine.evaluate import fuse_boundary_skeleton


def _order_key(p: Path) -> int:
    """批量遍历时用于排序的最佳努力整数：优先 epoch，其次 seed，再退化为文件名中的首个数字。

    兼容非 ``checkpoint_epochN`` 命名（如 ``seed0-best.pth``）；均无数字时返回 0，不会报错。
    """
    for pat in (r'epoch(\d+)', r'seed(\d+)'):
        m = re.search(pat, p.stem)
        if m:
            return int(m.group(1))
    m = re.search(r'(\d+)', p.stem)
    return int(m.group(1)) if m else 0


def _forward_prob(net, x):
    """统一前向：返回 (输出tensor, is_prob)。
    dict 输出（双分支）→ 融合成概率图 [0,1]（is_prob=True）；
    单 logits 输出 → 原样返回（is_prob=False，后续再 sigmoid/softmax）。
    """
    out = net(x)
    if isinstance(out, dict):
        return fuse_boundary_skeleton(out['boundary'], out['skeleton']), True
    return out, False


# ---------------------------------------------------------------------------
# Dataset for batch inference
# ---------------------------------------------------------------------------

class InferDataset(Dataset):
    def __init__(self, img_paths):
        self.paths = list(img_paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path   = self.paths[idx]
        img    = Image.open(path).convert('RGB')
        orig_w, orig_h = img.size
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = TF.to_tensor(img_np)   # [3, 512, 512]
        return tensor, orig_h, orig_w, str(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def preprocess_img(full_img: Image.Image) -> torch.Tensor:
    img_np = np.array(full_img.convert('RGB'))
    img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_LINEAR)
    return TF.to_tensor(img_np).unsqueeze(0)


def predict_img(net, full_img: Image.Image, device,
                out_threshold: float = 0.5) -> np.ndarray:
    net.eval()
    orig_w, orig_h = full_img.size
    img = preprocess_img(full_img).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output, is_prob = _forward_prob(net, img)
        output = F.interpolate(output, (orig_h, orig_w), mode='bilinear', align_corners=False)
        if net.n_classes > 1 and not is_prob:
            mask = output.argmax(dim=1)
        else:
            prob = output if is_prob else torch.sigmoid(output)
            mask = (prob > out_threshold).squeeze(1)
    return mask[0].cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def mask_to_wireframe(mask: np.ndarray, thickness: int = 2,
                      smooth: bool = False, min_area: int = 0) -> np.ndarray:
    """二值 mask → 线框边界图（0/255）。"""
    m    = (mask > 0).astype(np.uint8) * 255
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    wire = np.zeros_like(m)
    for cnt in cnts[-2]:
        if cv2.contourArea(cnt) < min_area:
            continue
        if smooth:
            eps = 0.002 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, eps, True)
        cv2.drawContours(wire, [cnt], -1, 255, thickness)
    return wire


def dilate_boundary(boundary, radius: int) -> np.ndarray:
    b = np.where(np.asarray(boundary) > 0, 255, 0).astype(np.uint8)
    if radius <= 0:
        return b
    k = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    return np.where(cv2.dilate(b, k) > 0, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _dilate_mask(arr: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary (0/1) uint8 array by `radius` pixels using an ellipse kernel."""
    if radius <= 0:
        return arr
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(arr, k)


def compute_precision(pred, gt, threshold=0.0, eps=1e-7):
    pb = (np.asarray(pred) > threshold).astype(np.uint8)
    gb = (np.asarray(gt)   > 0).astype(np.uint8)
    TP = np.sum((pb == 1) & (gb == 1))
    FP = np.sum((pb == 1) & (gb == 0))
    return TP / (TP + FP + eps)


def compute_recall(pred, gt, threshold=0.0, eps=1e-7):
    pb = (np.asarray(pred) > threshold).astype(np.uint8)
    gb = (np.asarray(gt)   > 0).astype(np.uint8)
    TP = np.sum((pb == 1) & (gb == 1))
    FN = np.sum((pb == 0) & (gb == 1))
    return TP / (TP + FN + eps)


def compute_f1(pred, gt, threshold=0.5, eps=1e-7):
    p = compute_precision(pred, gt, threshold, eps)
    r = compute_recall(pred, gt, threshold, eps)
    return 2 * p * r / (p + r + eps)


def compute_tolerant_precision(pred, gt, tol_p, threshold=100, eps=1e-7):
    """预测像素落在 GT 膨胀 tol_p 范围内算 TP，否则算 FP。"""
    pb     = (np.asarray(pred) > threshold).astype(np.uint8)
    gb     = (np.asarray(gt)   > 0).astype(np.uint8)
    gt_dil = _dilate_mask(gb, tol_p)
    TP = np.sum((pb == 1) & (gt_dil == 1))
    FP = np.sum((pb == 1) & (gt_dil == 0))
    return TP / (TP + FP + eps)


def compute_tolerant_recall(pred, gt, tol_r, threshold=100, eps=1e-7):
    """GT 像素落在预测膨胀 tol_r 范围内算 TP，否则算 FN。"""
    pb       = (np.asarray(pred) > threshold).astype(np.uint8)
    gb       = (np.asarray(gt)   > 0).astype(np.uint8)
    pred_dil = _dilate_mask(pb, tol_r)
    TP = np.sum((gb == 1) & (pred_dil == 1))
    FN = np.sum((gb == 1) & (pred_dil == 0))
    return TP / (TP + FN + eps)


def compute_boundary_iou_tolerant(pred, gt, threshold=0.5, tolerance=2, eps=1e-7):
    pb = (np.asarray(pred) > threshold).astype(np.uint8)
    gb = (np.asarray(gt)   > 0).astype(np.uint8)

    k        = np.ones((3, 3), np.uint8)
    pred_bnd = cv2.morphologyEx(pb, cv2.MORPH_GRADIENT, k)
    gt_bnd   = cv2.morphologyEx(gb, cv2.MORPH_GRADIENT, k)

    if tolerance > 0:
        tk       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
        pred_tol = cv2.dilate(pred_bnd, tk)
        gt_tol   = cv2.dilate(gt_bnd,  tk)
    else:
        pred_tol, gt_tol = pred_bnd, gt_bnd

    inter = (np.sum((pred_bnd == 1) & (gt_tol == 1)) +
             np.sum((gt_bnd   == 1) & (pred_tol == 1)))
    union = np.sum(pred_bnd == 1) + np.sum(gt_bnd == 1)
    return inter / (union + eps)


def compute_ois_f1(pred, gt, thresholds=np.linspace(0, 1, 150), eps=1e-7):
    gb      = (np.asarray(gt) > 0).astype(np.uint8)
    best_f1 = 0.0
    for t in thresholds:
        pb = (np.asarray(pred) > t).astype(np.uint8)
        TP = np.sum((pb == 1) & (gb == 1))
        FP = np.sum((pb == 1) & (gb == 0))
        FN = np.sum((pb == 0) & (gb == 1))
        p  = TP / (TP + FP + eps)
        r  = TP / (TP + FN + eps)
        best_f1 = max(best_f1, 2 * p * r / (p + r + eps))
    return best_f1


def compute_ods_f1(pred_list, gt_list, thresholds=np.linspace(0, 1, 150), eps=1e-7):
    best_f1 = 0.0
    for t in thresholds:
        TP = FP = FN = 0
        for pred, gt in zip(pred_list, gt_list):
            pb  = (np.asarray(pred) > t).astype(np.uint8)
            gb  = (np.asarray(gt)   > 0).astype(np.uint8)
            TP += np.sum((pb == 1) & (gb == 1))
            FP += np.sum((pb == 1) & (gb == 0))
            FN += np.sum((pb == 0) & (gb == 1))
        p       = TP / (TP + FP + eps)
        r       = TP / (TP + FN + eps)
        best_f1 = max(best_f1, 2 * p * r / (p + r + eps))
    return best_f1


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def eval_exp(in_files, out_dir, model_path, net, device,
             label_dir: str = 'data/test/label',
             threshold: float = 0.5,
             tolerance: int = 2,
             tol_p: int = 0,
             tol_r: int = 0,
             wireframe_thickness: int = 3,
             batch_size: int = 8,
             num_workers: int = 4):
    """批量推理：DataLoader 并行读图 → GPU 批量前向 → 线程池并行后处理。"""
    os.makedirs(out_dir, exist_ok=True)

    state_dict = torch.load(model_path, map_location=device)
    state_dict.pop('mask_values', None)
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.endswith(('total_ops', 'total_params'))}
    net.load_state_dict(state_dict)
    net.eval()
    logging.info(f'Model loaded from {model_path}')

    loader = DataLoader(
        InferDataset(in_files),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # 收集所有推理结果（CPU numpy）
    raw_masks = []   # list of (H×W uint8, orig_h, orig_w, path)
    with torch.no_grad():
        for tensors, orig_hs, orig_ws, paths in tqdm(loader, desc='Inference'):
            tensors = tensors.to(device=device, dtype=torch.float32)
            outputs, is_prob = _forward_prob(net, tensors)  # [B, C, 512, 512]

            for i in range(outputs.shape[0]):
                oh, ow = orig_hs[i].item(), orig_ws[i].item()
                out_i  = outputs[i:i+1]                    # [1, C, 512, 512]
                out_i  = F.interpolate(out_i, (oh, ow), mode='bilinear', align_corners=False)
                if net.n_classes > 1 and not is_prob:
                    mask = out_i.argmax(dim=1)[0]          # [H, W]
                else:
                    prob = out_i if is_prob else torch.sigmoid(out_i)
                    mask = (prob > threshold)[0, 0]        # [H, W]（out_i 恒为 [1,1,H,W]）
                raw_masks.append((mask.cpu().numpy().astype(np.uint8), oh, ow, paths[i]))

    # 后处理：wireframe + 指标（线程并行）
    sum_prec = sum_rec = sum_f1 = sum_biou = sum_ois = 0.0
    sum_tol_prec = sum_tol_rec = sum_tol_f1 = 0.0
    pred_list, label_list = [], []

    def process_one(item):
        raw_mask, oh, ow, path = item
        result     = mask_to_wireframe(raw_mask, thickness=wireframe_thickness, smooth=False)
        label_path = os.path.join(label_dir, Path(path).name)
        label_img  = np.array(Image.open(label_path).convert('L'))
        label_img  = dilate_boundary(label_img, radius=3)
        out_path   = os.path.join(out_dir, Path(path).name)
        Image.fromarray(result.astype(np.uint8)).save(out_path)
        return result.astype(np.uint8), label_img

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(tqdm(ex.map(process_one, raw_masks),
                            total=len(raw_masks), desc='Post-process'))

    for result, label_img in results:
        sum_prec += compute_precision(result, label_img, threshold=100)
        sum_rec  += compute_recall(result,    label_img, threshold=100)
        sum_f1   += compute_f1(result,        label_img, threshold=100)
        sum_biou += compute_boundary_iou_tolerant(result, label_img, threshold=100, tolerance=1)
        sum_ois  += compute_ois_f1(result, label_img)
        tp = compute_tolerant_precision(result, label_img, tol_p, threshold=100)
        tr = compute_tolerant_recall(result,    label_img, tol_r, threshold=100)
        sum_tol_prec += tp
        sum_tol_rec  += tr
        sum_tol_f1   += 2 * tp * tr / (tp + tr + 1e-7)
        pred_list.append(result)
        label_list.append(label_img)

    n      = len(in_files)
    ods_f1 = compute_ods_f1(pred_list, label_list)

    metrics = {
        'Precision':     sum_prec     / n,
        'Recall':        sum_rec      / n,
        'F1':            sum_f1       / n,
        'Tol_Precision': sum_tol_prec / n,
        'Tol_Recall':    sum_tol_rec  / n,
        'Tol_F1':        sum_tol_f1   / n,
        'Boundary_IoU':  sum_biou     / n,
        'OIS_F1':        sum_ois      / n,
        'ODS_F1':        ods_f1,
        'n_images':      n,
        'model':         model_path,
    }
    print(' | '.join(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}'
                     for k, v in metrics.items()))
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='UNet inference & evaluation')
    parser.add_argument('--arch', default='unet',
                        help='网络结构：unet/resnet_unet/feature_fusion/dcfe/dual_branch')
    parser.add_argument('--model',      '-m', default='checkpoints/unet/checkpoint_epoch120.pth')
    parser.add_argument('--input-dir',  '-i', default='data/test/image',
                        help='Directory with input images')
    parser.add_argument('--output-dir', '-o', default='results/unet',
                        help='Directory to save predictions')
    parser.add_argument('--label-dir',  '-L', default='data/test/label',
                        help='Directory with ground-truth labels')
    parser.add_argument('--threshold',  '-t', type=float, default=0.5)
    parser.add_argument('--tolerance',        type=int,   default=2,
                        help='Edge tolerance (px) for Boundary IoU')
    parser.add_argument('--tolerance-precision', type=int, default=3,
                        help='tol_p: GT 膨胀半径，影响 Tol_Precision/F1 (default: 3)')
    parser.add_argument('--tolerance-recall',    type=int, default=1,
                        help='tol_r: Pred 膨胀半径，影响 Tol_Recall/F1 (default: 1)')
    parser.add_argument('--classes',    '-c', type=int,   default=2)
    parser.add_argument('--bilinear',         action='store_true', default=False)
    parser.add_argument('--csv',              type=str,   default=None,
                        help='CSV file to append results to')
    parser.add_argument('--fold',             type=int,   default=None,
                        help='Fold index label for CSV')
    parser.add_argument('--wireframe',        type=int,   default=3,
                        help='Wireframe line thickness (px)')
    parser.add_argument('--model-dir',         type=str,   default=None,
                        help='遍历该目录下所有 .pth（兼容 epochN / seedN 等命名），与 --model 二选一')
    parser.add_argument('--batch-size',  '-b', type=int,   default=8,
                        help='推理 batch size')
    parser.add_argument('--num-workers',       type=int,   default=4,
                        help='DataLoader / 后处理线程数')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = build_model(args.arch, n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net.to(device=device)

    img_dir  = Path(args.input_dir)
    in_files = sorted(
        list(img_dir.glob('*.png'))  + list(img_dir.glob('*.jpg')) +
        list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.tif')) +
        list(img_dir.glob('*.tiff'))
    )
    if not in_files:
        logging.error(f'No images found in {img_dir}')
        raise SystemExit(1)
    logging.info(f'Found {len(in_files)} images.')

    # 收集所有要评估的 checkpoint
    if args.model_dir:
        ckpt_dir = Path(args.model_dir)
        ckpts = sorted(ckpt_dir.glob('*.pth'), key=_order_key)
        if not ckpts:
            logging.error(f'No .pth checkpoints found in {ckpt_dir}')
            raise SystemExit(1)
        logging.info(f'Found {len(ckpts)} checkpoints to evaluate.')
    else:
        ckpts = [Path(args.model)]

    all_metrics = []
    for ckpt in ckpts:
        logging.info(f'--- {ckpt.stem} : {ckpt} ---')

        out_dir_ep = os.path.join(args.output_dir, ckpt.stem) \
                     if args.model_dir else args.output_dir

        metrics = eval_exp(
            in_files=in_files,
            out_dir=out_dir_ep,
            model_path=str(ckpt),
            net=net,
            device=device,
            label_dir=args.label_dir,
            threshold=args.threshold,
            tolerance=args.tolerance,
            tol_p=args.tolerance_precision,
            tol_r=args.tolerance_recall,
            wireframe_thickness=args.wireframe,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        metrics['epoch']      = _order_key(ckpt)
        metrics['checkpoint'] = ckpt.name
        if args.fold is not None:
            metrics['fold'] = args.fold
        all_metrics.append(metrics)

    if args.csv:
        df_new = pd.DataFrame(all_metrics)
        if os.path.exists(args.csv):
            df_old = pd.read_csv(args.csv)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_csv(args.csv, index=False)
        logging.info(f'Results saved to {args.csv}')

    # 打印最优 epoch
    if len(all_metrics) > 1:
        best = max(all_metrics, key=lambda x: x['Tol_F1'])
        print(f"\nBest checkpoint: {best['checkpoint']}  Tol_F1={best['Tol_F1']:.4f}  F1={best['F1']:.4f}  ODS={best['ODS_F1']:.4f}")
