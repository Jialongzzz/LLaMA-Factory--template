#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
config="$PWD/qwen3_full_sft.yaml"

if ! command -v python >/dev/null 2>&1 || ! command -v llamafactory-cli >/dev/null 2>&1 || ! command -v torchrun >/dev/null 2>&1; then
  echo "请先激活已安装 PyTorch、LLaMA-Factory 和 DeepSpeed 的 Conda 环境。" >&2
  exit 1
fi

gpu_count=$(python - "$config" <<'PY'
import json
import sys
from pathlib import Path

import torch
import yaml

config_path = Path(sys.argv[1])
root = config_path.parent
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
model = config["model_name_or_path"]
if model.startswith("REPLACE_WITH_"):
    sys.exit(f"请先在 {config_path} 中填写实际的本地模型路径。")
if not Path(model).is_dir():
    sys.exit(f"本地模型目录不存在：{model}")
if not (root / config["deepspeed"]).is_file():
    sys.exit(f"DeepSpeed 配置不存在：{config['deepspeed']}")
info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
for name in config["dataset"].split(","):
    name = name.strip()
    if name not in info:
        sys.exit(f"数据集未在 dataset_info.json 注册：{name}")
    data_path = root / info[name]["file_name"]
    if not data_path.is_file() or data_path.stat().st_size == 0:
        sys.exit(f"数据文件不存在或为空：{data_path}；请放入 JSONL 数据。")
count = torch.cuda.device_count()
if count < 1:
    sys.exit("未检测到可用的 CUDA GPU，请检查 PyTorch/CUDA 环境。")
print(count)
PY
)

llamafactory-cli version
export FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="$gpu_count"
echo "检测到 $gpu_count 张可见 GPU，启动训练。"
exec llamafactory-cli train "$config"
