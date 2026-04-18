# Offline batch inference with vLLM

This repo includes `batch_infer_vllm.py` for offline JSONL batch inference.

## Input format

Each line must be a JSON object with:

- `message`: the user request
- `tables`: a JSON object describing the available datasets

Example:

```json
{"id": 1, "message": "Return the 5 newest rows from orders.", "tables": {"orders": {"columns": ["order_id", "created_at", "total"]}}}
{"id": 2, "message": "Group sales by country and sum revenue.", "tables": {"sales": {"columns": ["country", "revenue"]}}}
```

## Command

```bash
python batch_infer_vllm.py \
  --input-file data/requests.jsonl \
  --output-file data/responses.jsonl \
  --model-name Qwen/Qwen2.5-Coder-3B-Instruct
```

## Output format

The script preserves each input row and appends:

- `response`: generated Polars code
- `finish_reason`: vLLM stop reason

Example output:

```json
{"id": 1, "message": "Return the 5 newest rows from orders.", "tables": {"orders": {"columns": ["order_id", "created_at", "total"]}}, "response": "result = orders.sort(\"created_at\", descending=True).head(5)", "finish_reason": "stop"}
```

## Notes

- The prompt format matches the FastAPI app in `main.py`.
- vLLM handles continuous batching internally; `--batch-size` controls how many JSONL rows are submitted at once.
- Use `--tensor-parallel-size` when running across multiple GPUs.
