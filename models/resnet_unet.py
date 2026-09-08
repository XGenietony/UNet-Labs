"""ResNet34 encoder + ResNet-style decoder UNet."""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from .resnet_parts import ResDecBlock, ResUpBlock


class ResUNet(nn.Module):
    """
    Encoder : pretrained ResNet34 (ImageNet)
              stem → layer1 → layer2 → layer3 → layer4
              channels: 64(H/2) → 64(H/4) → 128(H/8) → 256(H/16) → 512(H/32)

    Decoder : ResNet-style BasicBlocks with residual connections
              dec4 → dec3 → dec2 → dec1 → dec0 → 1×1 output
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes

        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # ── Encoder ──────────────────────────────────────────────────
        self.enc_stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool    = backbone.maxpool
        self.enc_layer1 = backbone.layer1   # 64,  H/4
        self.enc_layer2 = backbone.layer2   # 128, H/8
        self.enc_layer3 = backbone.layer3   # 256, H/16
        self.enc_layer4 = backbone.layer4   # 512, H/32

        if n_channels != 3:
            self.enc_stem[0] = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # ── Decoder (ResNet-style blocks) ─────────────────────────────
        self.dec4 = ResDecBlock(512, 256, 256)   # cat(512+256) → 256, H/16
        self.dec3 = ResDecBlock(256, 128, 128)   # cat(256+128) → 128, H/8
        self.dec2 = ResDecBlock(128,  64,  64)   # cat(128+64)  →  64, H/4
        self.dec1 = ResDecBlock( 64,  64,  64)   # cat(64+64)   →  64, H/2
        self.dec0 = ResUpBlock(  64,  32)        # 64 → 32, H

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.enc_stem(x)                   # 64,  H/2
        s1 = self.enc_layer1(self.maxpool(s0))  # 64,  H/4
        s2 = self.enc_layer2(s1)                # 128, H/8
        s3 = self.enc_layer3(s2)                # 256, H/16
        s4 = self.enc_layer4(s3)                # 512, H/32

        # Decoder
        d = self.dec4(s4, s3)   # 256, H/16
        d = self.dec3(d,  s2)   # 128, H/8
        d = self.dec2(d,  s1)   # 64,  H/4
        d = self.dec1(d,  s0)   # 64,  H/2
        d = self.dec0(d)        # 32,  H

        return self.outc(d)
