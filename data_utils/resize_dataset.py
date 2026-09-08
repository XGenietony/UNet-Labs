"""统一的 ResizeDataset。

合并了四个原始工程 utils/resize_dataset.py 的差异，用构造参数控制：
  * augment      —— 是否做几何增强（翻转/旋转/正样本裁剪）
  * perlin       —— 是否叠加 Perlin 噪声增强（雾/霾/光照不均，Bae et al., 2018）
  * perlin_prob  —— Perlin 触发概率
  * edge_width   —— mask→边界 的形态学宽度（原 UNet=3 / DCFE=1）
  * with_skeleton—— 是否额外产出骨架 GT（双分支模型需要）
  * repeat       —— 一个 epoch 内每张图重复采样次数（配合随机增强扩样本）
  * size         —— 统一 resize 边长

__getitem__ 返回 {'image', 'mask'[, 'skeleton']}。
"""

from pathlib import Path
import random

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.morphology import skeletonize

IMG_EXTS = {'.png', '.jpg', '.tif'}


class ResizeDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir=None, mask_dir=None,
                 img_paths=None, mask_paths=None,
                 img_scale=0.5, augment=True, repeat=4,
                 perlin=True, perlin_prob=0.3,
                 edge_width=1, with_skeleton=True, size=512):
        if img_paths is not None:
            self.img_paths = list(img_paths)
            self.mask_paths = list(mask_paths)
        else:
            self.img_paths = sorted([p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS])
            self.mask_paths = sorted([p for p in Path(mask_dir).iterdir() if p.suffix.lower() in IMG_EXTS])
        assert len(self.img_paths) == len(self.mask_paths), '图像与标签数量不一致'

        self.img_scale = img_scale
        self.augment = augment
        self.repeat = repeat
        self.perlin = perlin
        self.perlin_prob = perlin_prob
        self.edge_width = edge_width
        self.with_skeleton = with_skeleton
        self.size = size
        self.n = len(self.img_paths)

    def __len__(self):
        return self.n * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % self.n          # 同一张图会被重复访问（配合随机增强）
        S = self.size
        img = np.array(Image.open(self.img_paths[real_idx]).convert('RGB'))
        mask = Image.open(self.mask_paths[real_idx]).convert('L')
        mask = self.mask_to_edge(np.array(mask), edge_width=self.edge_width)

        if self.augment:
            if random.random() > 0.9:
                img, mask = self.random_crop_with_positive(img, mask, crop_size=S)
            if random.random() > 0.7:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            if random.random() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)
            if random.random() > 0.3:
                k = random.choice([1, 2, 3])   # 90 / 180 / 270
                img = np.rot90(img, k)
                mask = np.rot90(mask, k)
            if self.perlin:
                img = np.ascontiguousarray(img)
                img = self.perlin_noise_augment(img, prob=self.perlin_prob)

        # 统一 resize（augment=False 时原图尺寸不一，也必须对齐）
        img = cv2.resize(np.ascontiguousarray(img), (S, S), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(np.ascontiguousarray(mask), (S, S), interpolation=cv2.INTER_NEAREST)

        img = img.copy()
        mask = mask.copy()

        sample = {}
        if self.with_skeleton:
            skeleton = self.boundary_to_skeleton(mask)
            sample['skeleton'] = torch.from_numpy(skeleton).float()

        img = TF.to_tensor(img)                       # [C,H,W] float32 [0,1]
        mask = (torch.from_numpy(mask).long() > 0).long()   # 二分类 0/1

        sample['image'] = img
        sample['mask'] = mask
        return sample

    # ------------------------------------------------------------------ #
    #  形态学 / GT 生成
    # ------------------------------------------------------------------ #
    def mask_to_edge(self, mask, edge_width=5, binary=True):
        mask = (mask > 0).astype(np.uint8)
        repair_k = max(3, edge_width // 2 * 2 + 1)
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (repair_k, repair_k))
        repaired = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, repair_kernel)

        k = edge_width * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(repaired, kernel, 1)
        eroded = cv2.erode(repaired, kernel, 1)

        edge = ((dilated - eroded) > 0).astype(np.uint8)
        if not binary:
            edge *= 255
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return edge

    def boundary_to_skeleton(self, boundary_mask, close_kernel=3, dilate_kernel=3):
        if boundary_mask.max() > 1:
            boundary_mask = (boundary_mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        closed = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)
        skeleton = skeletonize(closed.astype(bool)).astype(np.uint8)
        if dilate_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
            skeleton = cv2.dilate(skeleton, kernel, iterations=1)
        return skeleton

    # ------------------------------------------------------------------ #
    #  增强算子
    # ------------------------------------------------------------------ #
    def random_crop_with_positive(self, img, mask, crop_size=512, max_try=10):
        h, w = mask.shape[:2]
        th, tw = crop_size, crop_size
        if h <= th or w <= tw:
            i = max(0, (h - th) // 2)
            j = max(0, (w - tw) // 2)
            return img[i:i + th, j:j + tw], mask[i:i + th, j:j + tw]
        for _ in range(max_try):
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            mask_crop = mask[i:i + th, j:j + tw]
            if mask_crop.sum() > 0:
                return img[i:i + th, j:j + tw], mask_crop
        i = (h - th) // 2
        j = (w - tw) // 2
        return img[i:i + th, j:j + tw], mask[i:i + th, j:j + tw]

    def perlin_noise_augment(self, img, prob=0.3):
        """Perlin 噪声增强（Bae et al., Scientific Reports, 2018）。

        模拟遥感图像中的薄雾、光照不均、传感器噪声等成像条件变化。
        img: np.ndarray, uint8, (H,W,3)。
        """
        if random.random() > prob:
            return img
        H, W = img.shape[:2]
        noise = self._generate_smooth_noise(H, W, octaves=4)
        mode = random.choice(['fog', 'haze', 'illumination'])

        if mode == 'fog':
            intensity = random.uniform(0.35, 0.55)
            fog = (noise * intensity * 255).astype(np.float32)
            img = np.clip(img.astype(np.float32) + fog[:, :, None], 0, 255).astype(np.uint8)
        elif mode == 'haze':
            alpha = noise * random.uniform(0.35, 0.55)
            img = np.clip(img.astype(np.float32) * (1 - alpha[:, :, None]) + 255 * alpha[:, :, None],
                          0, 255).astype(np.uint8)
        elif mode == 'illumination':
            scale = 1.0 + noise * random.uniform(-0.65, 0.65)
            img = np.clip(img.astype(np.float32) * scale[:, :, None], 0, 255).astype(np.uint8)
        return img

    def _generate_smooth_noise(self, H, W, octaves=4):
        """多倍频程 value-noise，归一化到 [0,1]，近似 Perlin 噪声的空间连续性。"""
        noise = np.zeros((H, W), dtype=np.float32)
        amplitude = 1.0
        frequency = 1
        max_amplitude = 0.0
        for _ in range(octaves):
            h_small = max(2, H // frequency)
            w_small = max(2, W // frequency)
            small = np.random.rand(h_small, w_small).astype(np.float32)
            upsampled = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
            noise += upsampled * amplitude
            max_amplitude += amplitude
            amplitude *= 0.5
            frequency *= 2
        noise /= max_amplitude
        return noise
