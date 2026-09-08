""" Full assembly of the parts to form the complete network """

from .resnet_backbone_parts import *
from torchvision.models import resnet18, ResNet18_Weights

class TA_ASPP(nn.Module):
    def __init__(self, in_ch=512, out_ch=512, rates=(1, 6, 12, 18)):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch // len(rates), 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch // len(rates)),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])

        self.project = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # 🔥 Skeleton-aware attention
        self.skel_attn = nn.Sequential(
            nn.Conv2d(32, out_ch, 1),   # skel_feat: 32ch
            nn.Sigmoid()
        )

    def forward(self, x, skel_feat):
        # ASPP
        feats = torch.cat([b(x) for b in self.branches], dim=1)
        feats = self.project(feats)

        # Skeleton attention
        skel_attn = F.interpolate(
            self.skel_attn(skel_feat),
            size=feats.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        return feats * (1 + 0.2 * skel_attn)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_train=False, pretrained=False, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

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

        self.up1 = Up(512, 256, 256, bilinear)
        self.up2 = Up(256, 128, 128, bilinear)
        self.up3 = Up(128, 64, 64, bilinear)
        self.up4 = Up(64, 64, 64, bilinear)
        self.outc = (OutConv(64, n_classes))
        self.ta_aspp = TA_ASPP(in_ch=512, out_ch=512)

        # ==================================================
        # ① Skeleton early encoder heads（浅层结构约束）
        # ==================================================
        self.skel_enc1 = nn.Conv2d(64, 32, 1)
        self.skel_enc2 = nn.Conv2d(64, 32, 1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ==================================================
        # Skeleton early feature (用于 skip gating)
        # ==================================================
        skel_feat = self.skel_enc1(x1) + F.interpolate(
            self.skel_enc2(x2),
            size=x1.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # -------- TA-ASPP --------
        x5 = self.ta_aspp(x5, skel_feat)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        logits = F.interpolate(
            logits,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)