#!/bin/bash
set -euo pipefail

# Faster, quieter apt; don't reinstall pip (you already have a modern pip)
apt-get update -qq
apt-get install -y python-pip > /dev/null || true

# Make sure we don't fight Ubuntu's distutils pkgs (blinker)
python -m pip install --upgrade pip
python -m pip install --upgrade --ignore-installed blinker==1.9.0

# Install everything in one go (no build isolation needed for wheels here)
python -m pip install -r requirements.txt

# Start MLflow UI
mkdir -p /workspace/.mlflow
nohup mlflow ui --host 0.0.0.0 --port 5000 \
  --backend-store-uri /workspace/.mlflow > mlflow.log 2>&1 &

echo "📊 MLflow UI: http://localhost:5000"

# Sanity checks
python3 - <<'PY'
import torch, bitsandbytes as bnb
from transformers.utils import is_flash_attn_2_available
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| cuda ok:", torch.cuda.is_available())
print("CXX11_ABI:", "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")
print("flash-attn-2 available:", is_flash_attn_2_available())
print("bnb:", bnb.__version__)
print("bf16 supported:", torch.cuda.is_bf16_supported())
# Hard check: load the CUDA extension
try:
    import flash_attn_2_cuda as _fa2
    print("flash_attn_2_cuda import: OK")
except Exception as e:
    print("flash_attn_2_cuda import: FAILED ->", e)
PY
