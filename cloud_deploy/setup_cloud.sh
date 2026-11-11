#!/usr/bin/env bash
# 云端 GPU 环境一键自检与依赖安装（AutoDL/Featurize 等 Ubuntu 实例）。
# 在项目根的 cloud_deploy/ 下执行：bash cloud_deploy/setup_cloud.sh
set -e

# 定位项目根（本脚本在 cloud_deploy/ 下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
echo "project root: $ROOT"

# 选择 python 解释器
if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
echo "python: $($PY --version 2>&1)"

# 1. GPU 检查
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: 未检测到 nvidia-smi，当前实例可能无 GPU，请确认租用的是 GPU 实例。"
  exit 1
fi
echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2. torch / CUDA 检查
echo "== PyTorch =="
if ! $PY -c "import torch" 2>/dev/null; then
  echo "ERROR: 镜像未预装 torch。请换用带 PyTorch 的 GPU 镜像，或按 cloud_gpu_deploy.md 第三节手动安装 CUDA 版 torch。"
  exit 1
fi
$PY - <<'PYEOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch 为 CPU 版或 CUDA 不可用，请安装/切换 CUDA 版 torch。")
print("cuda device:", torch.cuda.get_device_name(0))
PYEOF

# 3. 轻量依赖
echo "== install lightweight deps =="
$PY -m pip install -r "$SCRIPT_DIR/requirements-cloud.txt"

# 4. 权重
echo "== checkpoint =="
if [ -f "$ROOT/model_checkpoints/fontclip_lora_best.pt" ]; then
  echo "checkpoint exists."
else
  $PY "$SCRIPT_DIR/download_checkpoint.py"
fi

# 5. 数据就位检查
echo "== data =="
[ -d "$ROOT/cloud_fonts" ] || { echo "ERROR: 缺少 cloud_fonts/，请在项目根解压 fond_cloud_bundle.zip"; exit 1; }
[ -d "$ROOT/data_cn/mixed" ] || { echo "ERROR: 缺少 data_cn/mixed/，请在项目根解压 fond_cloud_bundle.zip"; exit 1; }
echo "cloud fonts: $(ls "$ROOT/cloud_fonts" | wc -l) files"
echo "mixed dir ready."

echo ""
echo "SETUP COMPLETE. 下一步：bash cloud_deploy/run_train.sh"
