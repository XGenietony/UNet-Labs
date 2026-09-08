#!/bin/bash
# 建立 data 软链接，指向共享数据集（680 张图，各工程共用一份，避免重复拷贝）。
# 用法：
#   bash scripts/link_data.sh                       # 默认链到 ../Pytorch-UNet/data
#   bash scripts/link_data.sh /abs/path/to/dataset  # 指定数据集目录
set -e
cd "$(dirname "$0")/.."

SRC=${1:-$(cd .. && pwd)/Pytorch-UNet/data}

if [ ! -d "${SRC}" ]; then
    echo "数据源目录不存在：${SRC}" >&2
    exit 1
fi

# 校验预期子目录（imgs/label 训练，test/ 测试）
for sub in imgs label test; do
    [ -e "${SRC}/${sub}" ] || echo "警告：${SRC} 下缺少 ${sub}/" >&2
done

ln -sfn "${SRC}" data
echo "data -> $(readlink -f data)"
