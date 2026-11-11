#!/usr/bin/env bash
# 云端训练启动：先 smoke 验证环境，再正式训练。
# 用法：
#   bash cloud_deploy/run_train.sh                 # 先 smoke，再用全模块默认参数正式训练
#   SKIP_SMOKE=1 bash cloud_deploy/run_train.sh   # 跳过 smoke
#   EPOCH=100 BATCH_SIZE=48 LR=1e-5 bash cloud_deploy/run_train.sh
#   DISABLE_MODULES=1 bash cloud_deploy/run_train.sh  # 回到 S0 基线（不带 A/B/C/D）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
export KMP_DUPLICATE_LIB_OK=TRUE

# 可覆盖参数（默认与训练方案一致）
EPOCH="${EPOCH:-100}"
BATCH_SIZE="${BATCH_SIZE:-48}"
LR="${LR:-1e-5}"

# 模块参数：默认全模块 A+B+C+D；DISABLE_MODULES=1 回到 S0 基线
if [ "${DISABLE_MODULES:-0}" = "1" ]; then
  MODULE_ARGS=()
else
  MODULE_ARGS=(--use_orth --use_attr_heads --use_content_style --use_structure_anchor)
fi

# 1. smoke 验证
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "==== SMOKE ===="
  $PY train_cn_mixed.py --smoke "${MODULE_ARGS[@]}"
  echo "smoke passed."
fi

# 2. 正式训练
echo "==== FORMAL TRAIN: epoch=$EPOCH batch=$BATCH_SIZE lr=$LR modules=${MODULE_ARGS[*]} ===="
$PY train_cn_mixed.py \
  --epoch "$EPOCH" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  "${MODULE_ARGS[@]}"

echo "ALL TRAINING DONE. 检查 model_checkpoints/ 下输出的 .pt 与训练日志。"
