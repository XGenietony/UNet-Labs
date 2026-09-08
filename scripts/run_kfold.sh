#!/bin/bash
# K 折交叉验证：逐折训练 + 逐折评测，汇总最优(ODS-F1) epoch 的均值±标准差。
#
# 用法（通过环境变量选择实验）：
#   MODEL=dcfe        CLASSES=1 bash scripts/run_kfold.sh
#   MODEL=unet        CLASSES=1 PERLIN=0 bash scripts/run_kfold.sh
#   MODEL=resnet_unet CLASSES=2 EPOCHS=120 BATCH=4 LR=1e-4 bash scripts/run_kfold.sh
set -e
cd "$(dirname "$0")/.."

MODEL=${MODEL:-unet}
EPOCHS=${EPOCHS:-120}
BATCH=${BATCH:-8}
LR=${LR:-1e-3}
WD=${WD:-5e-6}
CLASSES=${CLASSES:-1}
FOLDS=${FOLDS:-5}
START_FOLD=${START_FOLD:-0}
SEED=${SEED:-0}
PERLIN=${PERLIN:-1}                       # 1=开启 Perlin 增强, 0=关闭
EXP=${EXP:-${MODEL}_kfold}
TEST_IMG=${TEST_IMG:-data/test/image}
TEST_LABEL=${TEST_LABEL:-data/test/label}

PERLIN_FLAG=$([ "$PERLIN" = "1" ] && echo "--perlin" || echo "--no-perlin")

echo "=========================================="
echo " ${MODEL}  K折 (folds ${START_FOLD}..$((FOLDS-1)))  perlin=${PERLIN}"
echo "=========================================="

for fold in $(seq ${START_FOLD} $((FOLDS - 1))); do
    echo "---------- Fold ${fold}: train ----------"
    python train.py \
        --model         ${MODEL}   \
        --epochs        ${EPOCHS}  \
        --batch-size    ${BATCH}   \
        --learning-rate ${LR}      \
        --weight-decay  ${WD}      \
        --classes       ${CLASSES} \
        --seed          ${SEED}    \
        --exp           ${EXP}_fold${fold} \
        --fold          ${fold}    \
        --folds         ${FOLDS}   \
        ${PERLIN_FLAG}

    echo "---------- Fold ${fold}: eval ----------"
    python eval.py ${EXP}_fold${fold} \
        --arch     ${MODEL}    \
        --classes  ${CLASSES}  \
        --img-dir  ${TEST_IMG} \
        --mask-dir ${TEST_LABEL} \
        --fold     ${fold}
done

echo ""
echo "========== 汇总 (每折取 ODS-F1 最优 epoch) =========="
python - <<PY
import glob, pandas as pd, numpy as np
files = sorted(glob.glob("results/${EXP}_fold*_eval.csv"))
if not files:
    print("未找到 results/${EXP}_fold*_eval.csv"); raise SystemExit(0)
best = []
for f in files:
    df = pd.read_csv(f)
    best.append(df.loc[df["ods_f1"].idxmax()])
b = pd.DataFrame(best)
for col in ["dice","iou","precision","recall","f1","ois_f1","ods_f1","boundary_iou"]:
    if col in b:
        print(f"  {col:<14}: {b[col].mean():.4f} ± {b[col].std():.4f}")
PY
