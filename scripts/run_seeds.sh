#!/bin/bash
# 多随机种子实验：同一配置跑多个 seed，报告最优(ODS-F1) epoch 指标的均值±标准差，
# 用于论文里 "mean ± std" 的稳健性汇报。
#
# 用法：
#   MODEL=dcfe SEEDS="0 1 2"           bash scripts/run_seeds.sh
#   MODEL=unet SEEDS="0 1 2" PERLIN=0  bash scripts/run_seeds.sh
#   MODEL=resnet_unet CLASSES=2 EPOCHS=120 SEEDS="0 1 2 3 4" bash scripts/run_seeds.sh
set -e
cd "$(dirname "$0")/.."

MODEL=${MODEL:-unet}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-120}
BATCH=${BATCH:-8}
LR=${LR:-1e-3}
WD=${WD:-5e-6}
CLASSES=${CLASSES:-1}
PERLIN=${PERLIN:-1}
FIXED_SPLIT=${FIXED_SPLIT:-1}             # 1=固定划分(imgs 训练/test 验证), 0=随机划分
EXP=${EXP:-${MODEL}_seeds}
TEST_IMG=${TEST_IMG:-data/test/image}
TEST_LABEL=${TEST_LABEL:-data/test/label}

PERLIN_FLAG=$([ "$PERLIN" = "1" ] && echo "--perlin" || echo "--no-perlin")
SPLIT_FLAG=$([ "$FIXED_SPLIT" = "1" ] && echo "--fixed-split" || echo "")

echo "=========================================="
echo " ${MODEL}  多种子 [${SEEDS}]  perlin=${PERLIN}"
echo "=========================================="

for seed in ${SEEDS}; do
    echo "---------- seed ${seed}: train ----------"
    python train.py \
        --model         ${MODEL}   \
        --epochs        ${EPOCHS}  \
        --batch-size    ${BATCH}   \
        --learning-rate ${LR}      \
        --weight-decay  ${WD}      \
        --classes       ${CLASSES} \
        --seed          ${seed}    \
        --exp           ${EXP}_seed${seed} \
        ${SPLIT_FLAG} ${PERLIN_FLAG}

    echo "---------- seed ${seed}: eval ----------"
    python eval.py ${EXP}_seed${seed} \
        --arch     ${MODEL}    \
        --classes  ${CLASSES}  \
        --img-dir  ${TEST_IMG} \
        --mask-dir ${TEST_LABEL}
done

echo ""
echo "========== 汇总 (每个 seed 取 ODS-F1 最优 epoch) =========="
python - <<PY
import glob, pandas as pd
files = sorted(glob.glob("results/${EXP}_seed*_eval.csv"))
if not files:
    print("未找到 results/${EXP}_seed*_eval.csv"); raise SystemExit(0)
best = [pd.read_csv(f).sort_values("ods_f1").iloc[-1] for f in files]
b = pd.DataFrame(best)
print(f"seeds = {len(files)}")
for col in ["dice","iou","precision","recall","f1","ois_f1","ods_f1","boundary_iou"]:
    if col in b:
        print(f"  {col:<14}: {b[col].mean():.4f} ± {b[col].std():.4f}")
PY
