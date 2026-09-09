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


def compute_boundary_iou_tolerant(pred, gt, threshold=0.5, tolerance=3, eps=1e-7):
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


def _tolerant_f1_bin(pred_bin, gb, tol_p, tol_r, eps=1e-7):
    """给定已二值化的 pred/gt，用非对称容忍口径算 F1（与 P/R 口径一致）。"""
    gt_dil = _dilate_mask(gb, tol_p)
    TP = np.sum((pred_bin == 1) & (gt_dil == 1))
    FP = np.sum((pred_bin == 1) & (gt_dil == 0))
    prec = TP / (TP + FP + eps)
    pred_dil = _dilate_mask(pred_bin, tol_r)
    TP2 = np.sum((gb == 1) & (pred_dil == 1))
    FN  = np.sum((gb == 1) & (pred_dil == 0))
    rec = TP2 / (TP2 + FN + eps)
    return 2 * prec * rec / (prec + rec + eps)


def compute_ois_f1(pred, gt, tol_p=0, tol_r=0, thresholds=np.linspace(0, 1, 150), eps=1e-7):
    """单图 OIS-F1：在概率图上扫阈值，取该图最优的容忍 F1。"""
    gb = (np.asarray(gt) > 0).astype(np.uint8)
    best_f1 = 0.0
    for t in thresholds:
        pb = (np.asarray(pred) > t).astype(np.uint8)
        best_f1 = max(best_f1, _tolerant_f1_bin(pb, gb, tol_p, tol_r, eps))
    return best_f1


def compute_ods_f1(pred_list, gt_list, tol_p=0, tol_r=0, thresholds=np.linspace(0, 1, 150), eps=1e-7):
    """数据集级 ODS-F1：找使全体 per-image 容忍 F1 均值最大的阈值（与 OIS 口径对齐，保证 ODS<=OIS）。"""
    gbs = [(np.asarray(gt) > 0).astype(np.uint8) for gt in gt_list]
    best_f1 = 0.0
    for t in thresholds:
        f1_mean = np.mean([
            _tolerant_f1_bin((np.asarray(pred) > t).astype(np.uint8), gb, tol_p, tol_r, eps)
            for pred, gb in zip(pred_list, gbs)
        ])
        best_f1 = max(best_f1, float(f1_mean))
    return best_f1


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def _aggregate_metrics(results, tol_p, tol_r, tolerance, ois_ods_input):
    """给定每图后处理结果 [(result_bin, prob, label_img), ...]，汇总数据集级指标。

    抽出来单独成函数，便于 K 折推理时把 5 折的留出集汇成整个数据集再统一评测。
    """
    sum_biou = sum_ois = 0.0
    sum_prec = sum_rec = sum_f1 = 0.0
    pred_list, label_list = [], []
    for result, prob, label_img in results:
        prec = compute_tolerant_precision(result, label_img, tol_p, threshold=100)
        rec  = compute_tolerant_recall(result,    label_img, tol_r, threshold=100)
        sum_prec += prec
        sum_rec  += rec
        sum_f1   += 2 * prec * rec / (prec + rec + 1e-7)
        sum_biou += compute_boundary_iou_tolerant(result, label_img, threshold=100, tolerance=tolerance)
        ois_pred = prob if ois_ods_input == 'prob' else result
        sum_ois  += compute_ois_f1(ois_pred, label_img, tol_p=tol_p, tol_r=tol_r)
        pred_list.append(ois_pred)
        label_list.append(label_img)
    n = len(results)
    ods_f1 = compute_ods_f1(pred_list, label_list, tol_p=tol_p, tol_r=tol_r)
    return {
        'Precision':    sum_prec / n,
        'Recall':       sum_rec  / n,
        'F1':           sum_f1   / n,
        'Boundary_IoU': sum_biou / n,
        'OIS_F1':       sum_ois  / n,
        'ODS_F1':       ods_f1,
        'n_images':     n,
    }


def eval_exp(in_files, out_dir, model_path, net, device,
             label_dir: str = 'data/test/label',
             label_map: dict = None,
             threshold: float = 0.5,
             tolerance: int = 2,
             tol_p: int = 0,
             tol_r: int = 0,
             wireframe_thickness: int = 3,
             wireframe: bool = False,
             ois_ods_input: str = 'prob',
             collect_results: bool = False,
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
    # 同时保留概率图 prob（供 OIS/ODS 扫阈值）和二值 mask（供 P/R/F）
    raw_masks = []   # list of (prob float32 H×W, mask uint8 H×W, orig_h, orig_w, path)
    with torch.no_grad():
        for tensors, orig_hs, orig_ws, paths in tqdm(loader, desc='Inference'):
            tensors = tensors.to(device=device, dtype=torch.float32)
            outputs, is_prob = _forward_prob(net, tensors)  # [B, C, 512, 512]

            for i in range(outputs.shape[0]):
                oh, ow = orig_hs[i].item(), orig_ws[i].item()
                out_i  = outputs[i:i+1]                    # [1, C, 512, 512]
                out_i  = F.interpolate(out_i, (oh, ow), mode='bilinear', align_corners=False)
                if net.n_classes > 1 and not is_prob:
                    prob = torch.softmax(out_i, dim=1)[0, 1]   # 前景类概率，供 OIS/ODS
                    mask = out_i.argmax(dim=1)[0]              # [H, W]
                else:
                    prob = (out_i if is_prob else torch.sigmoid(out_i))[0, 0]  # [H, W]
                    mask = (prob > threshold)                 # [H, W]
                raw_masks.append((prob.cpu().numpy().astype(np.float32),
                                  mask.cpu().numpy().astype(np.uint8), oh, ow, paths[i]))

    # 后处理：wireframe + 指标（线程并行）
    def process_one(item):
        prob, raw_mask, oh, ow, path = item
        if wireframe:
            # 线框化：提取二值 mask 的轮廓为细线（复现原工程口径）
            result = mask_to_wireframe(raw_mask, thickness=wireframe_thickness, smooth=False)
        else:
            # 直接用原始二值 mask；放大到 0/255 以兼容下游 threshold=100 的指标口径
            result = (raw_mask > 0).astype(np.uint8) * 255
        # K 折时同一批图片的 label 分散在 data/label 与 data/test/label 两个目录，
        # 用 label_map（图名→label 绝对路径）精确定位；否则退化为 label_dir + 文件名
        label_path = label_map[Path(path).name] if label_map is not None \
                     else os.path.join(label_dir, Path(path).name)
        label_img  = np.array(Image.open(label_path).convert('L'))
        label_img  = dilate_boundary(label_img, radius=3)
        out_path   = os.path.join(out_dir, Path(path).name)
        Image.fromarray(result.astype(np.uint8)).save(out_path)
        return result.astype(np.uint8), prob, label_img

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        results = list(tqdm(ex.map(process_one, raw_masks),
                            total=len(raw_masks), desc='Post-process'))

    # OIS/ODS 输入：prob=喂概率图（阈值扫描真正生效），binary=喂二值结果（复现原工程退化口径）
    metrics = _aggregate_metrics(results, tol_p, tol_r, tolerance, ois_ods_input)
    metrics['model'] = model_path
    print(' | '.join(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}'
                     for k, v in metrics.items()))
    if collect_results:
        metrics['_results'] = results
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
    parser.add_argument('--threshold',  '-t', type=float, default=0.6)
    parser.add_argument('--tolerance',        type=int,   default=4,
                        help='Edge tolerance (px) for Boundary IoU')
    parser.add_argument('--tolerance-precision', type=int, default=3,
                        help='tol_p: GT 膨胀半径，影响 Precision/OIS/ODS (default: 2)')
    parser.add_argument('--tolerance-recall',    type=int, default=3,
                        help='tol_r: Pred 膨胀半径，影响 Recall/F1/OIS/ODS (default: 2)')
    parser.add_argument('--classes',    '-c', type=int,   default=2)
    parser.add_argument('--bilinear',         action='store_true', default=False)
    parser.add_argument('--csv',              type=str,   default=None,
                        help='CSV file to append results to')
    parser.add_argument('--fold',             type=int,   default=None,
                        help='Fold index label for CSV')
    parser.add_argument('--kfold',            action='store_true', default=False,
                        help='K 折留出集推理：每折模型只推理自己训练时留出的那一折，'
                             '5 折并起来覆盖整个数据集（配合 --model-dir 指向 foldN-best.pth 所在目录）')
    parser.add_argument('--folds',            type=int,   default=5,
                        help='K 折总折数（需与训练时一致，默认 5）')
    parser.add_argument('--data-root',        type=str,   default='data',
                        help='K 折模式的数据根目录（含 imgs/ label/ test/image test/label）')
    parser.add_argument('--wireframe',        type=int,   default=3,
                        help='Wireframe line thickness (px)，仅在 --use-wireframe 时生效')
    parser.add_argument('--use-wireframe',    action='store_true', default=False,
                        help='对 pred 做线框化后再评测（复现原工程口径）；默认直接用二值 mask')
    parser.add_argument('--ois-ods-input', choices=['prob', 'binary'], default='prob',
                        help="OIS/ODS 的输入：prob=概率图(阈值扫描生效)，binary=二值结果(复现原工程)")
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

    # ---------------------------------------------------------------- #
    #  K 折留出集推理：复现 train.py 的跨步切分，每折模型只评自己的留出集
    # ---------------------------------------------------------------- #
    if args.kfold:
        if not args.model_dir:
            logging.error('--kfold 需配合 --model-dir 指向 foldN-best.pth 所在目录')
            raise SystemExit(1)
        IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        root = Path(args.data_root)
        # 与 train.py 完全一致：合并 imgs + test/image（train 在前，test 在后），各自内部排序
        all_imgs  = sorted(p for p in (root / 'imgs').iterdir()        if p.suffix.lower() in IMG_EXTS) + \
                    sorted(p for p in (root / 'test' / 'image').iterdir() if p.suffix.lower() in IMG_EXTS)
        all_masks = sorted(p for p in (root / 'label').iterdir()        if p.suffix.lower() in IMG_EXTS) + \
                    sorted(p for p in (root / 'test' / 'label').iterdir() if p.suffix.lower() in IMG_EXTS)
        assert len(all_imgs) == len(all_masks), \
            f'图/标签数量不一致：{len(all_imgs)} vs {len(all_masks)}'
        label_map = {img.name: str(msk) for img, msk in zip(all_imgs, all_masks)}
        logging.info(f'K 折：合集 {len(all_imgs)} 张，{args.folds} 折跨步切分')

        model_dir = Path(args.model_dir)
        all_metrics, pooled = [], []
        for n in range(args.folds):
            ckpt = model_dir / f'fold{n}-best.pth'
            if not ckpt.exists():
                logging.error(f'缺少折模型：{ckpt}')
                raise SystemExit(1)
            idx = list(range(n, len(all_imgs), args.folds))     # 该折的留出集索引
            fold_files = [all_imgs[i] for i in idx]
            logging.info(f'--- fold{n} : {ckpt.name}  留出 {len(fold_files)} 张 ---')
            metrics = eval_exp(
                in_files=fold_files,
                out_dir=os.path.join(args.output_dir, f'fold{n}'),
                model_path=str(ckpt),
                net=net,
                device=device,
                label_map=label_map,
                threshold=args.threshold,
                tolerance=args.tolerance,
                tol_p=args.tolerance_precision,
                tol_r=args.tolerance_recall,
                wireframe_thickness=args.wireframe,
                wireframe=args.use_wireframe,
                ois_ods_input=args.ois_ods_input,
                collect_results=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            pooled.extend(metrics.pop('_results'))
            metrics['fold'] = n
            metrics['checkpoint'] = ckpt.name
            all_metrics.append(metrics)

        # 整个数据集（5 折留出集之并）统一汇总
        overall = _aggregate_metrics(pooled, args.tolerance_precision,
                                     args.tolerance_recall, args.tolerance,
                                     args.ois_ods_input)
        overall.update({'model': 'ALL_FOLDS', 'fold': 'overall', 'checkpoint': 'overall'})
        print('\n=== Overall (整个数据集，5 折留出集之并) ===')
        print(' | '.join(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}'
                         for k, v in overall.items()))
        all_metrics.append(overall)

        if args.csv:
            df_new = pd.DataFrame(all_metrics)
            if os.path.exists(args.csv):
                df_new = pd.concat([pd.read_csv(args.csv), df_new], ignore_index=True)
            df_new.to_csv(args.csv, index=False)
            logging.info(f'Results saved to {args.csv}')
        raise SystemExit(0)

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
            wireframe=args.use_wireframe,
            ois_ods_input=args.ois_ods_input,
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
        best = max(all_metrics, key=lambda x: x['F1'])
        print(f"\nBest checkpoint: {best['checkpoint']}  F1={best['F1']:.4f}  ODS={best['ODS_F1']:.4f}")
