#!/usr/bin/env bash
# Install training dependencies into a local .venv using uv.
# Run once before run_finetune.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

echo "=== Creating .venv (Python 3.11) ==="
uv venv --python 3.11 .venv

echo "=== Installing PyTorch (CUDA 12.8 wheels) ==="
uv pip install --python .venv/bin/python \
  torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu128

echo "=== Installing Unsloth ==="
uv pip install --python .venv/bin/python \
  "unsloth[cu128-torch270] @ git+https://github.com/unslothai/unsloth.git"

echo "=== Installing remaining dependencies ==="
uv pip install --python .venv/bin/python \
  datasets \
  trl \
  peft \
  transformers \
  accelerate \
  fastapi \
  python-dotenv \
  pydantic \
  vllm

echo "=== Verifying GPU is visible to PyTorch ==="
.venv/bin/python - <<'EOF'
import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU     : {props.name}")
    print(f"VRAM    : {props.total_memory // 1024**3} GB")
else:
    print("GPU     : NOT FOUND")
EOF

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
