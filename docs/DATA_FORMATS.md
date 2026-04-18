# Expected data formats for Gemma E2B SFT + GRPO

This repository includes `train_gemma_e2b_sft_grpo.py`, which supports two stages:

1. **SFT**: supervised fine-tuning
2. **GRPO**: reinforcement learning with reward functions

Important Gemma 4 notes:

- use standard roles: `system`, `user`, `assistant`
- choose one template and stay consistent: `gemma-4` or `gemma-4-thinking`
- if you pre-render text yourself, strip the leading `<bos>` before SFT
- for multi-turn data, keep only the **final visible answer** in history; do not feed earlier thought blocks back into later turns
- the GRPO format shown below is a **custom reward format for this repo**, not Gemma 4's native thinking format

---

## Stage 1: SFT data

The script accepts any of the following SFT formats.

### Option A: `messages`

Best when your dataset is already in chat format.

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful benchmark assistant."},
    {"role": "user", "content": "Summarize this dataframe operation."},
    {"role": "assistant", "content": "This operation groups rows by key and computes the mean."}
  ]
}
```

### Option B: `conversations`

Useful for ShareGPT-like datasets.

```json
{
  "conversations": [
    {"role": "user", "content": "What does an inner join do?"},
    {"role": "assistant", "content": "It keeps only rows with matching keys on both sides."}
  ]
}
```

### Option C: `prompt` + `response`

Good for simple instruction datasets.

```json
{
  "prompt": "Explain the difference between a left join and an inner join.",
  "response": "A left join keeps all rows from the left table, while an inner join only keeps matching rows.",
  "system": "Answer briefly and clearly."
}
```

Notes:

- `system` is optional.
- If your system prompt lives in another column, pass `--sft-system-field`.

### Option D: `text`

Use this only if your sample is already fully templated with the exact Gemma template variant you will train and serve with.

```json
{
  "text": "<|turn>user\nWhat is a groupby?<turn|>\n<|turn>model\nA groupby splits rows into groups before aggregation.<turn|>"
}
```

Notes:

- This is the least flexible format.
- Prefer `messages` when possible.
- Make sure the text matches your chosen template variant: `gemma-4` vs `gemma-4-thinking`.
- Remove the leading `<bos>` before training if you pre-render samples yourself.

---

## Stage 2: GRPO data

GRPO needs a prompt and a gold answer for reward computation.

### Option A: `prompt` + `answer`

Recommended format.

```json
{
  "prompt": "What is the output shape of a groupby-count over 3 unique categories?",
  "answer": "3"
}
```

The script will wrap a string prompt as a user message automatically.

### Option B: chat prompt + `answer`

Recommended when you want explicit system instructions.

```json
{
  "prompt": [
    {"role": "system", "content": "Use <reasoning> and <answer> tags."},
    {"role": "user", "content": "How many rows remain after filtering 10 rows with 3 removed?"}
  ],
  "answer": "7"
}
```

### Option C: `messages`

The last assistant message is used as the gold answer.

```json
{
  "messages": [
    {"role": "system", "content": "Use <reasoning> and <answer> tags."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4"}
  ]
}
```

Important:

- the last message must be from `assistant`, or
- you must also provide an explicit `answer` column.
- the gold `answer` should contain the final answer only, not the reasoning trace.

---

## What this repo expects during GRPO

The default GRPO setup in this repo rewards outputs shaped like this:

```xml
<reasoning>
Short reasoning here.
</reasoning>
<answer>
4
</answer>
```

This is a **custom task format** used by the reward functions in `train_gemma_e2b_sft_grpo.py`.
It is not Gemma 4's native thinking-token format.

The default reward functions currently check:

- whether output is non-empty
- whether `<reasoning>` / `<answer>` tags are present
- whether the predicted answer partially matches the gold answer
- whether the predicted answer exactly matches the gold answer

---

## Minimal local examples

### Example SFT file: `data/sft.jsonl`

```json
{"prompt": "What is a left join?", "response": "A left join keeps all rows from the left table."}
{"prompt": "What does sort_values do?", "response": "It sorts rows by one or more columns."}
```

### Example GRPO file: `data/grpo.jsonl`

```json
{"prompt": "What is 5 + 7?", "answer": "12"}
{"prompt": "How many rows remain after removing 2 from 9?", "answer": "7"}
```

---

## Example commands

### SFT only

```bash
python3 train_gemma_e2b_sft_grpo.py \
  --stage sft \
  --model-name unsloth/gemma-4-E2B-it \
  --sft-dataset data/sft.jsonl \
  --output-dir outputs/gemma-e2b
```

### GRPO only

```bash
python3 train_gemma_e2b_sft_grpo.py \
  --stage grpo \
  --model-name unsloth/gemma-4-E2B-it \
  --grpo-dataset data/grpo.jsonl \
  --output-dir outputs/gemma-e2b
```

### SFT then GRPO

```bash
python3 train_gemma_e2b_sft_grpo.py \
  --stage all \
  --model-name unsloth/gemma-4-E2B-it \
  --sft-dataset data/sft.jsonl \
  --grpo-dataset data/grpo.jsonl \
  --output-dir outputs/gemma-e2b
```
