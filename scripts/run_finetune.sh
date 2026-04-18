#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-"$REPO_DIR/../llm-polars/data"}"
PREPARED_DIR="${PREPARED_DIR:-"$REPO_DIR/data/prepared"}"
OUTPUT_DIR="${OUTPUT_DIR:-"$REPO_DIR/outputs/gemma-e2b-polars"}"
MODEL_NAME="${MODEL_NAME:-"unsloth/gemma-4-E2B-it"}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-60}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-60}"
STAGE="${STAGE:-all}"

# Load HF_TOKEN from .env — check repo dir first, then one level up
if [[ -z "${HF_TOKEN:-}" ]]; then
  for env_candidate in "$REPO_DIR/.env" "$REPO_DIR/../.env"; do
    if [[ -f "$env_candidate" ]]; then
      set -a
      # shellcheck source=/dev/null
      source "$env_candidate"
      set +a
      break
    fi
  done
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Export it or add it to .env"
  exit 1
fi

cd "$REPO_DIR"

# Use venv Python if available and not already in a venv
if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/python" ]]; then
  source .venv/bin/activate
fi

echo "=== Step 1: prepare pairs.jsonl ==="
python scripts/prepare_pairs.py \
  --input "$DATA_ROOT/pairs.jsonl" \
  --output-dir "$PREPARED_DIR" \
  --skip-invalid

echo "=== Step 2: prepare github_validated_pairs.jsonl ==="
python scripts/prepare_github_pairs.py \
  --input "$DATA_ROOT/github_validated_pairs.jsonl" \
  --output-dir "$PREPARED_DIR" \
  --skip-malformed

echo "=== Step 3: merge SFT datasets ==="
mkdir -p "$PREPARED_DIR"
cat "$PREPARED_DIR/github_sft.jsonl" "$PREPARED_DIR/sft.jsonl" \
  > "$PREPARED_DIR/merged_sft.jsonl"
echo "Merged SFT records: $(wc -l < "$PREPARED_DIR/merged_sft.jsonl")"

echo "=== Step 4: merge GRPO datasets ==="
cat "$PREPARED_DIR/github_grpo.jsonl" "$PREPARED_DIR/grpo.jsonl" \
  > "$PREPARED_DIR/merged_grpo.jsonl"
echo "Merged GRPO records: $(wc -l < "$PREPARED_DIR/merged_grpo.jsonl")"

echo "=== Step 5: fine-tune (stage=$STAGE) ==="
python train_gemma_e2b_sft_grpo.py \
  --stage "$STAGE" \
  --model-name "$MODEL_NAME" \
  --sft-dataset "$PREPARED_DIR/merged_sft.jsonl" \
  --sft-messages-field messages \
  --sft-max-steps "$SFT_MAX_STEPS" \
  --grpo-dataset "$PREPARED_DIR/merged_grpo.jsonl" \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --output-dir "$OUTPUT_DIR"

echo "=== Done. Adapters saved to $OUTPUT_DIR ==="
