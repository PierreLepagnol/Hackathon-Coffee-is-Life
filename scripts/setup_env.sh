#!/usr/bin/env bash
# Install training dependencies into a local .venv using uv.
# Run once before run_finetune.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

echo "=== Creating .venv (Python 3.11) ==="
uv venv --python 3.11 .venv

echo "=== Installing PyTorch 2.11.0 + matching torchvision (CUDA 12.8 wheels) ==="
uv pip install --python .venv/bin/python \
  "torch==2.11.0" "torchvision==0.26.0" \
  --extra-index-url https://download.pytorch.org/whl/cu128

echo "=== Installing Unsloth ==="
uv pip install --python .venv/bin/python \
  "unsloth[cu128-torch270] @ git+https://github.com/unslothai/unsloth.git"

echo "=== Installing remaining dependencies ==="
uv pip install --python .venv/bin/python \
  datasets \
  "trl==0.18.2" \
  peft \
  transformers \
  accelerate \
  mergekit \
  fastapi \
  python-dotenv \
  pydantic

echo "=== Patching trl bug: _is_package_available returns tuple not bool ==="
sed -i \
  's/^\(_[a-z_]*_available\) = _is_package_available(\(.*\))$/\1, _ = _is_package_available(\2)/' \
  .venv/lib/python3.11/site-packages/trl/import_utils.py

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
