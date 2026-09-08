import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from .resnet_backbone_parts import Up, OutConv

class UNet(nn.Module):
    def __init__(self, n_classes=1, n_channels=3, n_train=False, pretrained=False, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.n_classes = n_classes
        self.n_train = n_train

        # ---------------- Encoder (shared) ----------------
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.inc = nn.Sequential(
            backbone.conv1,   # 64, /2
            backbone.bn1,
            backbone.relu
        )

        self.down1 = nn.Sequential(
            backbone.maxpool,  # /4
            backbone.layer1    # 64
        )
        self.down2 = backbone.layer2  # 128, /8
        self.down3 = backbone.layer3  # 256, /16
        self.down4 = backbone.layer4  # 512, /32

        # ---------------- Branch A: Boundary Region Decoder ----------------
        self.b_up1 = Up(512, 256, 256, bilinear)
        self.b_up2 = Up(256, 128, 128, bilinear)
        self.b_up3 = Up(128, 64, 64, bilinear)
        self.b_up4 = Up(64, 64, 64, bilinear)
        self.boundary_out = OutConv(64, 1)

        # ---------------- Branch B: Skeleton / Connectivity Decoder --------
        # 刻意做“窄”，防止学成区域
        self.s_up1 = Up(512, 256, 128, bilinear)
        self.s_up2 = Up(128, 128, 64, bilinear)
        self.s_up3 = Up(64, 64, 32, bilinear)
        self.s_up4 = Up(32, 64, 32, bilinear)
        self.skeleton_out = OutConv(32, 1)

    def forward(self, x):
        # ---------------- Encoder ----------------
        x1 = self.inc(x)     # 64, H/2
        x2 = self.down1(x1)  # 64, H/4
        x3 = self.down2(x2)  # 128, H/8
        x4 = self.down3(x3)  # 256, H/16
        x5 = self.down4(x4)  # 512, H/32

        # ---------------- Boundary Region Branch ----------------
        b = self.b_up1(x5, x4)
        b = self.b_up2(b, x3)
        b = self.b_up3(b, x2)
        b = self.b_up4(b, x1)
        boundary_logits = self.boundary_out(b)

        # ---------------- Skeleton Branch ----------------
        s = self.s_up1(x5, x4)
        s = self.s_up2(s, x3)
        s = self.s_up3(s, x2)
        s = self.s_up4(s, x1)
        skeleton_logits = self.skeleton_out(s)

        # ---------------- Upsample to input size ----------------
        boundary_logits = F.interpolate(
            boundary_logits, scale_factor=2,
            mode='bilinear', align_corners=False
        )
        skeleton_logits = F.interpolate(
            skeleton_logits, scale_factor=2,
            mode='bilinear', align_corners=False
        )

        # ---------------- Skeleton-guided Fusion ----------------
        boundary_prob = torch.sigmoid(boundary_logits)
        skeleton_prob = torch.sigmoid(skeleton_logits)

        # 用 skeleton 修补连通（训练 & 推理都可用）
        if self.n_train:
            skeleton_bridge = F.max_pool2d(
                skeleton_prob, kernel_size=3, stride=1, padding=1
            )
        else:
            skeleton_bridge = skeleton_prob

        # final = boundary_prob + 0.5 * skeleton_bridge
        final = boundary_prob * (1 + 0.5 * skeleton_bridge)
        final = torch.clamp(final, 0, 1)

        return {
            "final": final,
            "boundary": boundary_logits,
            "skeleton": skeleton_logits
        }
