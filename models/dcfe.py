import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from .resnet_backbone_parts import Up, OutConv

class BoundaryAwareASPP(nn.Module):
    def __init__(self, in_ch=512, out_ch=512, rates=(1, 6, 12, 18)):
        super().__init__()

        inter_ch = out_ch // len(rates)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, inter_ch, 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(inter_ch),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, 1),
            nn.Sigmoid()
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = torch.cat([b(x) for b in self.branches], dim=1)
        attn = self.attn(feats)

        feats = feats * (1 + 0.2 * attn)
        return self.project(feats)

class BoundaryASPP(nn.Module):
    def __init__(self, in_ch=512, out_ch=512, rates=(1, 6, 12, 18)):
        super().__init__()

        inter_ch = out_ch // len(rates)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, inter_ch, 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(inter_ch),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])

        # Global context branch（非常重要）
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(inter_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[-2:]

        feats = [b(x) for b in self.branches]

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(h, w),
                           mode='bilinear', align_corners=False)

        feats.append(gp)

        feats = torch.cat(feats, dim=1)
        return self.project(feats)


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
    def __init__(self, n_classes=1, n_channels=3,
                 n_train=False, pretrained=False, bilinear=True):
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

        # ==================================================
        # ① Skeleton early encoder heads（浅层结构约束）
        # ==================================================
        self.skel_enc1 = nn.Conv2d(64, 32, 1)
        self.skel_enc2 = nn.Conv2d(64, 32, 1)

        # ==================================================
        # ② Skeleton-aware skip gating
        # ==================================================
        self.skip_gate1 = nn.Conv2d(32, 64, 1)
        self.skip_gate2 = nn.Conv2d(32, 64, 1)
        self.skip_gate3 = nn.Conv2d(32, 128, 1)
        self.skip_gate4 = nn.Conv2d(32, 256, 1)

        # ---------------- Boundary Decoder ----------------
        self.b_up1 = Up(512, 256, 256, bilinear)
        self.b_up2 = Up(256, 128, 128, bilinear)
        self.b_up3 = Up(128, 64, 64, bilinear)
        self.b_up4 = Up(64, 64, 64, bilinear)
        self.boundary_out = OutConv(64, 1)

        # ---------------- Skeleton Decoder ----------------
        self.s_up1 = Up(512, 256, 128, bilinear)
        self.s_up2 = Up(128, 128, 64, bilinear)
        self.s_up3 = Up(64, 64, 32, bilinear)
        self.s_up4 = Up(32, 64, 32, bilinear)
        self.skeleton_out = OutConv(32, 1)

        # ==================================================
        # ③ Skeleton → Boundary feature fusion（轻量）
        # ==================================================
        self.skel_fuse1 = nn.Conv2d(128, 256, 1)
        self.skel_fuse2 = nn.Conv2d(64, 128, 1)

        self.fuse_alpha = 0.2  # ★ 小权重，防止 boundary 退化
        
        # ==================================================
        # ④ TA-ASPP（Skeleton-aware global context）
        # ==================================================
        self.ta_aspp = TA_ASPP(in_ch=512, out_ch=512)
        self.baspp = BoundaryASPP(in_ch=512, out_ch=512)
        self.baaspp = BoundaryAwareASPP(in_ch=512, out_ch=512)


    def forward(self, x):
        # ---------------- Encoder ----------------
        x1 = self.inc(x)     # 64, H/2
        x2 = self.down1(x1)  # 64, H/4
        x3 = self.down2(x2)  # 128, H/8
        x4 = self.down3(x3)  # 256, H/16
        x5 = self.down4(x4)  # 512, H/32
        
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
        # x5 = self.baspp(x5)
        # x5 = self.baaspp(x5)

        # ---------------- Skeleton Branch ----------------
        s1 = self.s_up1(x5, x4)
        s2 = self.s_up2(s1, x3)
        s3 = self.s_up3(s2, x2)
        s4 = self.s_up4(s3, x1)
        skeleton_logits = self.skeleton_out(s4)

        # ==================================================
        # Skeleton-aware skip gating
        # ==================================================
        def gate(skip, gate_conv):
            g = torch.sigmoid(
                gate_conv(
                    F.interpolate(
                        skel_feat,
                        size=skip.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                )
            )
            return skip * g

        x1_g = gate(x1, self.skip_gate1)
        x2_g = gate(x2, self.skip_gate2)
        x3_g = gate(x3, self.skip_gate3)
        x4_g = gate(x4, self.skip_gate4)
        # x1_g, x2_g, x3_g, x4_g = x1, x2, x3, x4

        # ---------------- Boundary Branch ----------------
        b = self.b_up1(x5, x4_g)

        # Skeleton → Boundary feature fusion（高层）
        b = b + self.fuse_alpha * self.skel_fuse1(s1)

        b = self.b_up2(b, x3_g)
        b = b + self.fuse_alpha * self.skel_fuse2(s2)

        b = self.b_up3(b, x2_g)
        b = self.b_up4(b, x1_g)
        boundary_logits = self.boundary_out(b)

        # ---------------- Upsample ----------------
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

        if self.n_train:
            skeleton_bridge = F.max_pool2d(
                skeleton_prob, kernel_size=3, stride=1, padding=1
            )
        else:
            skeleton_bridge = skeleton_prob

        final = boundary_prob * (1 + 0.2 * skeleton_bridge)
        final = torch.clamp(final, 0, 1)

        return {
            "final": final,
            "boundary": boundary_logits,
            "skeleton": skeleton_logits
        }
