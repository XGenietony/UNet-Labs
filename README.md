# UNet-Lab

四个 UNet 及其变种边界/线框分割实验（原 `Pytorch-UNet` / `Pytorch-ResNet` /
`Pytorch-UNetFeatureFusion` / `Pytorch-DCFENet-Perlin`）的**统一工程**。
通过 `--model` 一个参数切换算法，配合独立标志控制 K 折交叉验证、Perlin 噪声增强、
多随机种子等实验维度。

原始各工程精心调过的**模型定义原封不动地拷贝进来**（放在 `models/`），
只在其上加一层薄的统一注册表 + 训练/评测/推理驱动，按输出类型（logits vs 双分支 dict）
分派，避免在迁移中引入细微 bug。

---

## 1. 目录结构

```
UNet-Lab/
├── train.py                # 统一训练入口（--model 选择算法）
├── eval.py                 # 遍历 checkpoint 逐 epoch 评测（推荐）
├── predict.py              # 批量推理 + 评测（另一套指标口径）
├── summarize_kfold.py      # 汇总 K 折 CSV
├── summarize_seeds.py      # 汇总多种子 CSV
├── models/                 # ← 各原始工程的模型定义（原样拷贝 + 注册表）
│   ├── __init__.py         #    build_model / output_type / MODEL_NAMES
│   ├── vanilla_unet.py     #    unet          (logits)
│   ├── resnet_unet.py      #    resnet_unet   (logits)
│   ├── feature_fusion.py   #    feature_fusion(logits)
│   ├── dcfe.py             #    dcfe          (dict: final/boundary/skeleton)
│   └── dual_branch.py      #    dual_branch   (dict)
├── data_utils/
│   └── resize_dataset.py   # 统一数据集（增强/Perlin/骨架GT 由参数控制）
├── engine/
│   └── evaluate.py         # 统一验证（logits / dict 双分支融合）
├── losses/                 # dice_score.py / loss_function.py
├── scripts/
│   ├── link_data.sh        # 建立 data 软链接
│   ├── run_kfold.sh        # K 折：逐折训练+评测+均值±std
│   └── run_seeds.sh        # 多种子：训练+评测+均值±std
├── data -> ../Pytorch-UNet/data   # 软链接，680 张图共用一份
├── checkpoints/<exp>/      # 训练输出
└── results/<exp>_eval.csv  # 评测输出
```

---

## 2. 可选模型（`--model`）

| `--model`        | 来源工程                    | 输出        | 说明 |
|------------------|-----------------------------|-------------|------|
| `unet`           | Pytorch-UNet                | logits      | 原版 milesial U-Net |
| `resnet_unet`    | Pytorch-ResNet              | logits      | ResNet34 backbone 的 ResUNet |
| `feature_fusion` | Pytorch-UNetFeatureFusion   | logits      | 特征融合 U-Net |
| `dcfe`           | Pytorch-DCFENet-Perlin      | dict        | 双分支（边界+骨架解码器）+ TA-ASPP |
| `dual_branch`    | Pytorch-DCFENet-Perlin      | dict        | 双分支变体 |

`dict` 模型输出 `{"final","boundary","skeleton"}`，评测时由
`engine.evaluate.fuse_boundary_skeleton` 融合成单张概率图。

---

## 3. 环境

基础 conda：`/environment/miniconda3`，torch 2.2.2+cu121。关键约束：

- **numpy 必须 < 2**（torch 2.2.2 与 numpy 2.x 不兼容）
- **opencv-python-headless < 5**（5.x 强依赖 numpy>=2）

一次性安装（当前基础环境已按此修好）：

```bash
pip install "opencv-python-headless<5" "numpy==1.26.4" scikit-image
```

> 建议：为本工程建一个独立 conda 环境，按 `requirements.txt` 安装，避免污染基础环境。
> 原工程 `requirements.txt` 里 torch 固定 1.13.1+cu117，若沿用请相应选择匹配的 numpy/opencv。

数据软链接（首次）：

```bash
bash scripts/link_data.sh                 # 默认链到 ../Pytorch-UNet/data
# 或 bash scripts/link_data.sh /abs/path/to/dataset
```

数据布局：`data/imgs`、`data/label`（训练），`data/test/image`、`data/test/label`（测试）。

---

## 4. 常用命令

### 4.1 单次训练

```bash
# DCFE 双分支，K 折第 0 折
python train.py --model dcfe --classes 1 --fold 0 --epochs 120 -b 8 -l 1e-3 --exp dcfe_fold0

# 原版 U-Net，随机划分（10% 验证），关闭 Perlin
python train.py --model unet --classes 1 --epochs 120 -b 8 -l 1e-3 --no-perlin --exp unet

# ResUNet，固定划分（imgs 训练 / test 验证），2 类
python train.py --model resnet_unet --classes 2 --fixed-split --epochs 120 -b 4 -l 1e-4 --exp resnet
```

checkpoint 存到 `checkpoints/<exp>/checkpoint_epoch{N}.pth`。

### 4.2 评测（逐 epoch）

```bash
python eval.py dcfe_fold0 --arch dcfe --classes 1 \
    --img-dir data/test/image --mask-dir data/test/label
```

遍历 `checkpoints/dcfe_fold0/` 下所有权重，写 `results/dcfe_fold0_eval.csv`，
并打印 ODS-F1 最优 epoch。**注意 `--arch` 必须与训练时 `--model` 一致。**

### 4.3 K 折交叉验证（一键）

```bash
MODEL=dcfe        CLASSES=1 bash scripts/run_kfold.sh
MODEL=unet        CLASSES=1 PERLIN=0 bash scripts/run_kfold.sh
MODEL=resnet_unet CLASSES=2 EPOCHS=120 BATCH=4 LR=1e-4 bash scripts/run_kfold.sh
```

逐折训练+评测，最后打印每折取 ODS-F1 最优 epoch 后的**均值 ± 标准差**。
可用环境变量：`MODEL EPOCHS BATCH LR WD CLASSES FOLDS START_FOLD SEED PERLIN EXP`。

### 4.4 多随机种子（稳健性 mean±std）

```bash
MODEL=dcfe SEEDS="0 1 2"          bash scripts/run_seeds.sh
MODEL=unet SEEDS="0 1 2" PERLIN=0 bash scripts/run_seeds.sh
```

可用环境变量：`MODEL SEEDS EPOCHS BATCH LR WD CLASSES PERLIN FIXED_SPLIT EXP`。

### 4.5 批量推理 / 可视化

```bash
python predict.py --arch dcfe --classes 1 \
    --model checkpoints/dcfe_fold0/checkpoint_epoch120.pth \
    --input-dir data/test/image --output-dir results/dcfe_pred \
    --label-dir data/test/label
```

（`predict.py` 中 `--model/-m` 是权重路径，架构选择用 `--arch`，与 `eval.py` 一致。）

---

## 5. 关键实验开关（train.py）

| 标志 | 默认 | 作用 |
|------|------|------|
| `--model {unet,resnet_unet,feature_fusion,dcfe,dual_branch}` | `unet` | 选择算法 |
| `--fold N --folds K` | — | K 折交叉验证（跨步切分 train+test 合集） |
| `--fixed-split` | off | 固定划分：imgs/label 训练、test/ 验证 |
| （默认）`--validation V` | 10 | 随机划分验证集比例（%） |
| `--perlin` / `--no-perlin` | on | Perlin 噪声增强（雾/霾/光照不均） |
| `--perlin-prob P` | 0.3 | Perlin 触发概率 |
| `--loss {auto,bce,ce,focal}` | auto | 标准模型损失（双分支固定复合损失） |
| `--skeleton-gt {skeleton,mask}` | skeleton | 双分支骨架分支监督：真骨架GT（推荐）/ 复现原工程 |
| `--classes C` | 1 | 类别数（边界分割一般为 1） |
| `--epochs -b -l -w --grad-clip --amp --bilinear --pretrained --seed` | — | 常规超参 |

> `--skeleton-gt mask` 复现了原 DCFE 工程中「骨架分支用 mask 当 GT」的行为，
> 仅用于对齐旧结果；新实验建议保持默认 `skeleton`（真骨架 GT）。

---

## 6. 备注

- 训练强制 cudnn deterministic，保证可复现。
- `eval.py` 与 `predict.py` 的指标口径不同（前者对预测做线框化后按细线对齐、
  ODS 与 OIS 定义对齐；后者是原工程口径）。论文数据以 `eval.py` 为准，两者不要混用。
