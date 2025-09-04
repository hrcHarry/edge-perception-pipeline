#!/usr/bin/env bash
set -e
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U onnxruntime-gpu
python - <<'PY'
import torch, onnxruntime as ort
print("CUDA:", torch.cuda.is_available())
print("ORT providers:", ort.get_available_providers())
PY
