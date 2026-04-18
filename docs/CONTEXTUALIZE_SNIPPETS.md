# Contextualize snippets from a CSV sample

This repo now includes a small pipeline for this task:

> Given one row from a dataset and a generic snippet, generate a contextualized snippet using real column names, variable names, and values from that row.

The pipeline uses:

- `contextualize_snippets.py` to build a rule-first request file
- `batch_infer_vllm.py` to run offline batch generation with vLLM

## Files

- `contextualize_snippets.py`: prepares the task and can optionally call vLLM
- `batch_infer_vllm.py`: runs batch inference on a JSONL file
- `prompting.py`: switches the system prompt depending on the task

## How it works

### Step 1: rule-first preparation

`contextualize_snippets.py` reads:

- a CSV file
- one data row from that CSV
- one snippet or a file of snippets

It then builds a JSONL request with:

- the original snippet
- a rule-based draft
- the selected CSV row
- derived metadata such as:
  - `dataset_var_name`
  - `row_var_name`
  - `column_aliases`
  - `representative_columns`

### Step 2: model refinement

The prepared JSONL is passed to `batch_infer_vllm.py`.

The model receives:

- the original snippet
- the rule-based draft
- the actual CSV row
- the derived metadata

It returns only the final contextualized snippet.

## Supported snippet inputs

You can pass snippets in 2 ways.

### Option A: one snippet on the command line

```bash
uv run contextualize_snippets.py \
  --csv-file data/SampleSuperstore.csv \
  --row-index 1 \
  --snippet 'df.filter(pl.col("{metric_column}") > {value:Sales})' \
  --output-file outputs/contextualized.jsonl
```

### Option B: a snippet file

```bash
uv run contextualize_snippets.py \
  --csv-file data/SampleSuperstore.csv \
  --row-index 1 \
  --snippets-file artifacts/polars-snippets.json\
  --output-file outputs/contextualized.jsonl
```

Use `--snippets-file` with either:

- a plain text file containing one snippet
- a `.jsonl` file

Example JSONL:

```json
{"id": "s1", "snippet": "df.select(pl.col(\"{entity_column}\"))"}
{"id": "s2", "snippet": "df.filter(pl.col(\"{metric_column}\") > {value:Sales})"}
```

## Supported placeholders

The rule-first stage can replace these placeholders before the model runs.

### Generic placeholders

- `{dataset}` or `<dataset>`
- `{row}` or `<row>`
- `{entity_column}` or `<entity_column>`
- `{metric_column}` or `<metric_column>`
- `{date_column}` or `<date_column>`
- `{id_column}` or `<id_column>`

### Column and value placeholders

- `{column:Sales}`
- `{value:Sales}`
- `{column:product_name}`
- `{value:product_name}`

Both real column names and slugified aliases are supported.

For example, for `Product Name`:

- `{column:Product Name}` → `Product Name`
- `{column:product_name}` → `Product Name`
- `{value:Product Name}` → the row value
- `{value:product_name}` → the row value

## Variable naming rules

The script derives names from the dataset file.

Example:

- CSV file: `data/SampleSuperstore.csv`
- `dataset_var_name`: `sample_superstore`
- `row_var_name`: `sample_superstore_row`

It also applies simple replacements such as:

- `df` → `sample_superstore`
- `row` → `sample_superstore_row`

## Prepare-only mode

Use this if you want to inspect the generated request file before running the model.

```bash
uv run contextualize_snippets.py \
  --csv-file data/SampleSuperstore.csv \
  --row-index 1 \
  --snippet 'df.filter(pl.col("{metric_column}") > {value:Sales})' \
  --output-file outputs/requests.jsonl \
  --prepare-only
```

In this mode, `--output-file` is the prepared input JSONL.

## Full pipeline

This runs both preparation and vLLM inference.

```bash
uv run contextualize_snippets.py \
  --csv-file data/SampleSuperstore.csv \
  --row-index 1 \
  --snippet 'df.filter(pl.col("{metric_column}") > {value:Sales})' \
  --output-file outputs/contextualized.jsonl \
  --model-name Qwen/Qwen2.5-Coder-3B-Instruct
```

## Running `batch_infer_vllm.py` directly

If you already prepared the input JSONL, you can run batch inference yourself:

```bash
uv run batch_infer_vllm.py \
  --input-file outputs/requests.jsonl \
  --output-file outputs/contextualized.jsonl \
  --model-name Qwen/Qwen2.5-Coder-3B-Instruct
```

The input must contain:

- `message`
- `tables`

For this task, `tables` also contains `"__task": "contextualize_snippet"` so `prompting.py` uses the right system prompt.

## Output format

The final output JSONL preserves the original prepared record and adds:

- `response`: the contextualized snippet
- `finish_reason`: the vLLM finish reason

Example output:

```json
{
  "id": "snippet-1",
  "snippet": "df.filter(pl.col(\"{metric_column}\") > {value:Sales})",
  "rule_based_draft": "sample_superstore.filter(pl.col(\"Sales\") > 261.96)",
  "response": "sample_superstore.filter(pl.col(\"Sales\") > 261.96)",
  "finish_reason": "stop"
}
```

## Useful arguments

### `contextualize_snippets.py`

- `--csv-file`: source CSV file
- `--row-index`: 1-based data row index
- `--snippet`: one snippet inline
- `--snippets-file`: snippet file instead of inline text
- `--output-file`: prepared JSONL or final output JSONL
- `--prepare-only`: only prepare the JSONL, do not run vLLM
- `--prepared-input-file`: explicit path for the prepared JSONL
- `--keep-prepared-file`: keep the temporary prepared file in full mode

### vLLM arguments passed through

- `--model-name`
- `--batch-size`
- `--max-tokens`
- `--temperature`
- `--top-p`
- `--tensor-parallel-size`
- `--gpu-memory-utilization`
- `--max-model-len`
- `--trust-remote-code`

## Notes

- `--row-index` is counted from the first data row, not the header.
- CSV reading supports fallback encodings: `utf-8-sig`, `utf-8`, `cp1252`, `latin-1`.
- The rule-first stage is intentionally simple. The model is only used to refine or complete contextualization.
- The system prompt for this task is different from the default Polars code generation prompt.
