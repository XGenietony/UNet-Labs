# UNet-Lab

四个 UNet 及其变种的边界/线框分割实验统一工程，整合自
`Pytorch-UNet` / `Pytorch-ResNet` / `Pytorch-UNetFeatureFusion` /
`Pytorch-DCFENet-Perlin`。用一个 `--model` 参数切换算法，配合独立标志控制
K 折交叉验证、Perlin 噪声增强、多随机种子等实验维度。

各原始工程调好的**模型定义原样拷贝进 `models/`**，只在其上加一层薄的统一注册表
与训练/评测/推理驱动，按输出类型（logits vs 双分支 dict）分派，避免迁移时引入细微 bug。

---

## 1. 目录结构

```
UNet-Lab/
├── train.py               # 统一训练入口（--model 选择算法）
├── eval.py                # 遍历 checkpoint 逐个评测（论文口径，推荐）
├── predict.py             # 批量推理 + 评测（原工程口径），支持整目录遍历
├── summarize_kfold.py     # 汇总 K 折 CSV → 均值±std
├── summarize_seeds.py     # 汇总多种子 CSV → 均值±std
├── models/                # ← 各原始工程的模型定义（原样拷贝 + 注册表）
│   ├── __init__.py        #    build_model / output_type / MODEL_NAMES
│   ├── vanilla_unet.py    #    unet           (logits)
│   ├── resnet_unet.py     #    resnet_unet    (logits)
│   ├── feature_fusion.py  #    feature_fusion (logits)
│   ├── dcfe.py            #    dcfe           (dict: final/boundary/skeleton)
│   └── dual_branch.py     #    dual_branch    (dict)
├── data_utils/            # 统一数据集（增强 / Perlin / 骨架GT 由参数控制）
├── engine/evaluate.py     # 统一验证（logits / dict 双分支融合）
├── losses/                # dice_score.py / loss_function.py
├── scripts/
│   ├── link_data.sh       # 建立 data 软链接
│   ├── run_kfold.sh       # K 折：逐折训练+评测+均值±std
│   └── run_seeds.sh       # 多种子：训练+评测+均值±std
├── data -> ../Pytorch-UNet/data   # 软链接，多份实验共用一份数据
├── checkpoints/<exp>/     # 训练输出（已 gitignore）
└── results/               # 评测 / 推理输出（已 gitignore）
```

`data/`、`checkpoints/`、`results/` 均已排除出 git，不会被提交。

---

## 2. 可选模型（`--model` / `--arch`）

| 名称             | 来源工程                    | 输出   | 说明 |
|------------------|-----------------------------|--------|------|
| `unet`           | Pytorch-UNet                | logits | 原版 milesial U-Net |
| `resnet_unet`    | Pytorch-ResNet              | logits | ResNet34 backbone 的 ResUNet |
| `feature_fusion` | Pytorch-UNetFeatureFusion   | logits | 特征融合 U-Net |
| `dcfe`           | Pytorch-DCFENet-Perlin      | dict   | 双分支（边界+骨架解码器）+ TA-ASPP |
| `dual_branch`    | Pytorch-DCFENet-Perlin      | dict   | 双分支变体 |

> `dict` 模型输出 `{"final","boundary","skeleton"}`，评测/推理时由
> `engine.evaluate.fuse_boundary_skeleton` 融合成单张概率图。
> `train.py` 用 `--model`，`eval.py` / `predict.py` 用 `--arch`，**取值必须与训练时一致**。

---

## 3. 环境搭建

推荐为本工程建独立环境（Python 3.11），按 `requirements.txt` 安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

torch / torchvision 不在 `requirements.txt` 内，按 CUDA 版本单独装
（本项目在 torch 2.0.0+cu118 上验证）：

```bash
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
```

**libnvrtc 软链接（torch 2.0.0+cu118 装完后必做一次）**
cuDNN 运行时会 `dlopen("libnvrtc.so")`（无版本号），但 torch wheel 只带哈希命名的
`libnvrtc-*.so.11.2`。`requirements.txt` 已含 `nvidia-cuda-nvrtc-cu11`（提供标准
`libnvrtc.so.11.2`），但仍需补一个软链接指向它，否则卷积时报
`libnvrtc.so: cannot open shared object file` 并 core dump：

```bash
ln -sf ../../nvidia/cuda_nvrtc/lib/libnvrtc.so.11.2 \
  "$(python -c 'import site;print(site.getsitepackages()[0])')/torch/lib/libnvrtc.so"
```

> 该命令幂等，可重复执行。此软链接在 venv 内，不随 git / requirements 走，重建环境后需再跑一次。

**数据软链接（首次）**

```bash
bash scripts/link_data.sh                 # 默认链到 ../Pytorch-UNet/data
# 或 bash scripts/link_data.sh /abs/path/to/dataset
```

数据布局：`data/imgs`、`data/label`（训练），`data/test/image`、`data/test/label`（测试）。

---

## 4. 常用命令

### 4.1 训练

```bash
# DCFE 双分支，K 折第 0 折
python train.py --model dcfe --classes 1 --fold 0 --epochs 120 -b 8 -l 1e-3 --exp dcfe_fold0

# 原版 U-Net，随机划分（10% 验证），关闭 Perlin
python train.py --model unet --classes 1 --epochs 120 -b 8 -l 1e-3 --no-perlin --exp unet

# ResUNet，固定划分（imgs 训练 / test 验证），2 类
python train.py --model resnet_unet --classes 2 --fixed-split --epochs 120 -b 4 -l 1e-4 --exp resnet
```

checkpoint 存到 `checkpoints/<exp>/checkpoint_epoch{N}.pth`。

### 4.2 评测（逐 checkpoint，论文口径）

```bash
python eval.py dcfe_fold0 --arch dcfe --classes 1 \
    --img-dir data/test/image --mask-dir data/test/label
```

第一个位置参数是**实验名**（不带 `checkpoints/` 前缀，脚本自动补）。遍历
`checkpoints/dcfe_fold0/` 下所有 `.pth`，写 `results/dcfe_fold0_eval.csv`，并打印
ODS-F1 最优的 checkpoint。checkpoint 命名不限于 `epochN`，`seedN-best.pth`
等也能正确识别、排序、分别输出（互不覆盖）。

### 4.3 批量推理 / 可视化（原工程口径）

单个权重：

```bash
python predict.py --arch dcfe --classes 1 \
    --model checkpoints/dcfe_fold0/checkpoint_epoch120.pth \
    --input-dir data/test/image --output-dir results/dcfe_pred \
    --label-dir data/test/label
```

遍历整个目录下所有 `.pth`（与 `--model` 二选一），每个权重的结果写到
`results/dcfe_pred/<ckpt名>/` 独立子目录：

```bash
python predict.py --arch dcfe --classes 1 \
    --model-dir checkpoints/DEFCNet-seed \
    --input-dir data/test/image --output-dir results/dcfe_pred \
    --label-dir data/test/label --csv results/dcfe_pred/metrics.csv
```

> `predict.py` 中 `--model/-m` 是权重路径，`--arch` 选架构，与 `eval.py` 一致。

### 4.4 K 折交叉验证（一键）

```bash
MODEL=dcfe        CLASSES=1 bash scripts/run_kfold.sh
MODEL=unet        CLASSES=1 PERLIN=0 bash scripts/run_kfold.sh
MODEL=resnet_unet CLASSES=2 EPOCHS=120 BATCH=4 LR=1e-4 bash scripts/run_kfold.sh
```

逐折训练+评测，最后打印每折取 ODS-F1 最优 epoch 后的**均值 ± 标准差**。
可用环境变量：`MODEL EPOCHS BATCH LR WD CLASSES FOLDS START_FOLD SEED PERLIN EXP`。

### 4.5 多随机种子（稳健性 mean±std）

```bash
MODEL=dcfe SEEDS="0 1 2"          bash scripts/run_seeds.sh
MODEL=unet SEEDS="0 1 2" PERLIN=0 bash scripts/run_seeds.sh
```

可用环境变量：`MODEL SEEDS EPOCHS BATCH LR WD CLASSES PERLIN FIXED_SPLIT EXP`。

---

## 5. 关键实验开关（train.py）

| 标志 | 默认 | 作用 |
|------|------|------|
| `--model {unet,resnet_unet,feature_fusion,dcfe,dual_branch}` | `unet` | 选择算法 |
| `--fold N --folds K` | — | K 折交叉验证（跨步切分 train+test 合集） |
| `--fixed-split` | off | 固定划分：imgs/label 训练、test/ 验证 |
| `--validation V` | 10 | 随机划分验证集比例（%） |
| `--perlin` / `--no-perlin` | on | Perlin 噪声增强（雾/霾/光照不均） |
| `--perlin-prob P` | 0.3 | Perlin 触发概率 |
| `--loss {auto,bce,ce,focal}` | auto | 标准模型损失（双分支固定复合损失） |
| `--skeleton-gt {skeleton,mask}` | skeleton | 双分支骨架分支监督：真骨架GT（推荐）/ 复现原工程 |
| `--classes C` | 1 | 类别数（边界分割一般为 1） |
| `--seed S` | 0 | 随机种子（多种子报告 std 用 0/1/2） |
| `--epochs -b -l -w --grad-clip --amp --bilinear --pretrained` | — | 常规超参 |

> `--skeleton-gt mask` 复现原 DCFE 工程「骨架分支用 mask 当 GT」的行为，仅用于对齐旧结果；
> 新实验建议保持默认 `skeleton`（真骨架 GT）。

---

## 6. 备注

- 训练强制 cudnn deterministic，保证可复现。
- `eval.py` 与 `predict.py` 指标口径不同：前者对预测线框化后按细线对齐，ODS/OIS 定义对齐；
  后者是原工程口径。**论文数据以 `eval.py` 为准，两者不要混用。**
