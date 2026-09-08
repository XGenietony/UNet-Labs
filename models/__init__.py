"""模型注册表 —— 通过 `--model <name>` 选择某个实验对应的网络。

统一入口：`build_model(name, n_channels, n_classes, bilinear, pretrained)`。
每个模型标注 `output_type`：
  * 'logits' —— 前向返回单张 logits tensor（标准分割头）
  * 'dict'   —— 前向返回 {"final","boundary","skeleton"}（双分支 boundary+skeleton）

原始工程对应关系：
  unet           <- Pytorch-UNet/unet/unet_model.py            （经典 U-Net）
  resnet_unet    <- Pytorch-ResNet/unet/resnet_unet.py         （ResNet34 backbone U-Net）
  feature_fusion <- Pytorch-UNetFeatureFusion/unet/unet_model.py（ResNet18 + TA-ASPP 单解码器）
  dcfe           <- Pytorch-DCFENet-Perlin/unet/unet_model.py  （ResNet18 双分支 + TA-ASPP + 骨架融合，旗舰）
  dual_branch    <- Pytorch-*/unet/unet_model_dual_branchs.py  （ResNet18 双分支，无 ASPP，消融基线）
"""

from .vanilla_unet import UNet as VanillaUNet
from .resnet_unet import ResUNet
from .feature_fusion import UNet as FeatureFusionUNet
from .dcfe import UNet as DCFENet
from .dual_branch import UNet as DualBranchUNet

# name -> (class, output_type)
_REGISTRY = {
    'unet':           (VanillaUNet,       'logits'),
    'resnet_unet':    (ResUNet,           'logits'),
    'feature_fusion': (FeatureFusionUNet, 'logits'),
    'dcfe':           (DCFENet,           'dict'),
    'dual_branch':    (DualBranchUNet,    'dict'),
}

MODEL_NAMES = list(_REGISTRY.keys())


def output_type(name: str) -> str:
    """返回 'logits' 或 'dict'。"""
    return _REGISTRY[name][1]


def build_model(name: str, n_channels: int = 3, n_classes: int = 1,
                bilinear: bool = True, pretrained: bool = False):
    """按名字构建模型，屏蔽各变体构造函数签名差异。"""
    if name not in _REGISTRY:
        raise ValueError(f'未知模型 {name!r}，可选：{MODEL_NAMES}')
    cls, _ = _REGISTRY[name]

    if name == 'unet':
        model = cls(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    elif name == 'resnet_unet':
        model = cls(n_channels=n_channels, n_classes=n_classes)
    else:  # feature_fusion / dcfe / dual_branch —— ResNet18 backbone 系列
        model = cls(n_channels=n_channels, n_classes=n_classes,
                    n_train=False, pretrained=pretrained, bilinear=bilinear)
    return model
