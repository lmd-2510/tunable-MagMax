#!/usr/bin/env bash
# Prepare ai_env only. No dataset/model download, training, or experiment selection.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ACTION="${1:-check}"
PYTHON_BIN="${TUNABLE_PYTHON:-python}"
GPU_ID="${TUNABLE_GPU_ID:-0}"
[[ "$GPU_ID" == 0 || "$GPU_ID" == 1 ]] || { echo '[STOP] TUNABLE_GPU_ID must be 0 or 1.' >&2; exit 1; }
export MAGMAX_BASE_DIR="${MAGMAX_BASE_DIR:-$ROOT}"
export MAGMAX_DATA_DIR="${MAGMAX_DATA_DIR:-$ROOT/datasets}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export WANDB_MODE=disabled

case "$ACTION" in
  install)
    mkdir -p outs/mica_setup
    "$PYTHON_BIN" - <<'PY' > outs/mica_setup/torch_constraints.txt
from importlib.metadata import version, PackageNotFoundError
for name in ('torch', 'torchvision', 'torchaudio'):
    try:
        print(f'{name}=={version(name)}')
    except PackageNotFoundError:
        if name == 'torch':
            raise
PY
    "$PYTHON_BIN" -m pip install -r requirements.txt -c outs/mica_setup/torch_constraints.txt
    echo '[OK] Dependencies installed. Next: bash scripts/setup_mica.sh check'
    ;;
  check)
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys

if platform.system() != 'Linux':
    raise SystemExit('Run this preflight on the MICA Linux server.')
requirements = {}
for line in Path('requirements.txt').read_text().splitlines():
    if line.strip() and not line.startswith('#'):
        name, expected = line.strip().split('==')
        installed = metadata.version(name)
        if installed != expected:
            raise SystemExit(f'{name}: expected {expected}, installed {installed}. Run setup_mica.sh install.')
        requirements[name] = installed

import torch
import torchvision
import open_clip
import wandb
import geomloss
import optuna
import matplotlib

# Some upstream modules parse CLI arguments at import time and require a model name.
# This enables import checking only; it does not create/load a model or choose a run config.
sys.argv = [sys.argv[0], '--model', 'ViT-B-16']
import finetune_splitted
import merge_for_targetdata
from src.config import BASE_DIR, DATA_DIR
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Expected a CUDA-enabled PyTorch and one visible GPU.')
probe = torch.randn(8, 8, device='cuda', requires_grad=True)
(probe @ probe).sum().backward()
torch.cuda.synchronize()
del probe
free, total = torch.cuda.mem_get_info()
report = {
    'python': platform.python_version(), 'torch': torch.__version__,
    'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0), 'base_dir': BASE_DIR, 'data_dir': DATA_DIR,
    'dependencies': requirements,
}
out = Path('outs/mica_setup')
out.mkdir(parents=True, exist_ok=True)
(out / 'environment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('[ENV]', json.dumps(report, indent=2))
print(f'[GPU] free={free/2**30:.2f}/{total/2**30:.2f} GiB')
print('[OK] Imports/CUDA passed. This check does not launch training or merging.')
PY
    ;;
  *) echo 'Usage: bash scripts/setup_mica.sh {install|check}' >&2; exit 2 ;;
esac
