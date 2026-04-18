from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from prompting import MODEL_NAME


PLACEHOLDER_TOKEN_RE = re.compile(r"[^a-z0-9]+")
NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a rule-first snippet contextualization dataset and optionally run "
            "batch_infer_vllm.py on it."
        )
    )
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--row-index", type=int, default=1)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--prepared-input-file")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-prepared-file", action="store_true")

    snippet_group = parser.add_mutually_exclusive_group(required=True)
    snippet_group.add_argument("--snippet")
    snippet_group.add_argument("--snippets-file")

    parser.add_argument("--batch-script", default="batch_infer_vllm.py")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--prompt-key", default="message")
    parser.add_argument("--tables-key", default="tables")
    parser.add_argument("--response-key", default="response")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def slugify(text: str) -> str:
    camel_split = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text.strip())
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_split)
    lowered = camel_split.lower()
    normalized = PLACEHOLDER_TOKEN_RE.sub("_", lowered).strip("_")
    return normalized or "value"


def read_text_with_fallback(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def read_csv_row(path: Path, row_index: int) -> dict[str, str]:
    if row_index <= 0:
        raise ValueError("--row-index must be positive.")

    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                for current_index, row in enumerate(reader, start=1):
                    if current_index == row_index:
                        return {key: (value or "") for key, value in row.items()}
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError(f"CSV row {row_index} was not found in {path}.")


def normalize_snippet_record(record: Any, fallback_id: str) -> dict[str, Any]:
    if isinstance(record, str):
        return {"id": fallback_id, "snippet": record}
    if not isinstance(record, dict):
        raise ValueError("Snippet entries must be JSON objects or strings.")

    snippet = record.get("snippet")
    if not isinstance(snippet, str) or not snippet.strip():
        code = record.get("code")
        if isinstance(code, str) and code.strip():
            snippet = code
    if not isinstance(snippet, str) or not snippet.strip():
        raise ValueError("Snippet entries must include a non-empty 'snippet' or 'code' field.")

    enriched_record = dict(record)
    enriched_record.setdefault("id", fallback_id)
    enriched_record["snippet"] = snippet
    return enriched_record


def parse_json_snippets(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(read_text_with_fallback(path))
    if isinstance(payload, dict) and isinstance(payload.get("snippets"), list):
        raw_items = payload["snippets"]
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = [payload]

    items = [normalize_snippet_record(raw, f"snippet-{i}") for i, raw in enumerate(raw_items, start=1)]
    if not items:
        raise ValueError(f"No snippets found in {path}.")
    return items


def load_snippets(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.snippet is not None:
        return [{"id": "snippet-1", "snippet": args.snippet}]

    snippets_path = Path(args.snippets_file)
    if not snippets_path.is_file():
        raise FileNotFoundError(f"Snippets file not found: {snippets_path}")

    if snippets_path.suffix == ".jsonl":
        items: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(read_text_with_fallback(snippets_path).splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                items.append(normalize_snippet_record(record, f"snippet-{line_number}"))
            except ValueError as exc:
                raise ValueError(f"Line {line_number} in {snippets_path}: {exc}") from exc
        if not items:
            raise ValueError(f"No snippets found in {snippets_path}.")
        return items

    if snippets_path.suffix == ".json":
        return parse_json_snippets(snippets_path)

    content = read_text_with_fallback(snippets_path).strip()
    if not content:
        raise ValueError(f"Snippets file is empty: {snippets_path}")
    return [{"id": snippets_path.stem or "snippet-1", "snippet": content}]


def classify_column(name: str, value: str) -> str:
    lowered_name = name.lower()
    if lowered_name.endswith("id") or lowered_name == "row id":
        return "identifier"
    if DATE_RE.match(value):
        return "date"
    if NUMBER_RE.match(value):
        return "numeric"
    return "text"


def choose_representative_columns(row: dict[str, str]) -> dict[str, str]:
    priorities = {
        "entity_column": ["Product Name", "Customer Name", "Category", "Sub-Category", "City"],
        "metric_column": ["Sales", "Profit", "Quantity", "Discount", "Postal Code"],
        "date_column": ["Order Date", "Ship Date"],
        "id_column": ["Order ID", "Customer ID", "Product ID", "Row ID"],
    }

    chosen: dict[str, str] = {}
    for label, candidates in priorities.items():
        for candidate in candidates:
            if candidate in row:
                chosen[label] = candidate
                break

    if "entity_column" not in chosen:
        chosen["entity_column"] = next(iter(row))
    if "metric_column" not in chosen:
        for name, value in row.items():
            if classify_column(name, value) == "numeric":
                chosen["metric_column"] = name
                break
    return chosen


def build_rule_context(csv_path: Path, row_index: int, row: dict[str, str]) -> dict[str, Any]:
    dataset_name = csv_path.stem
    dataset_var_name = slugify(dataset_name)
    row_var_name = f"{dataset_var_name}_row"
    column_aliases = {name: slugify(name) for name in row}
    column_kinds = {name: classify_column(name, value) for name, value in row.items()}
    representative_columns = choose_representative_columns(row)

    return {
        "__task": "contextualize_snippet",
        "dataset_name": dataset_name,
        "dataset_var_name": dataset_var_name,
        "row_var_name": row_var_name,
        "row_index": row_index,
        "row": row,
        "column_aliases": column_aliases,
        "column_kinds": column_kinds,
        "representative_columns": representative_columns,
        "representative_values": {
            label: row[column_name]
            for label, column_name in representative_columns.items()
            if column_name in row
        },
    }


def apply_rule_replacements(snippet: str, context: dict[str, Any]) -> str:
    row = context["row"]
    replacements = {
        "{dataset}": context["dataset_var_name"],
        "<dataset>": context["dataset_var_name"],
        "{row}": context["row_var_name"],
        "<row>": context["row_var_name"],
        "{entity_column}": context["representative_columns"].get("entity_column", ""),
        "<entity_column>": context["representative_columns"].get("entity_column", ""),
        "{metric_column}": context["representative_columns"].get("metric_column", ""),
        "<metric_column>": context["representative_columns"].get("metric_column", ""),
        "{date_column}": context["representative_columns"].get("date_column", ""),
        "<date_column>": context["representative_columns"].get("date_column", ""),
        "{id_column}": context["representative_columns"].get("id_column", ""),
        "<id_column>": context["representative_columns"].get("id_column", ""),
    }

    for column_name, alias in context["column_aliases"].items():
        column_value = row[column_name]
        replacements[f"{{column:{column_name}}}"] = column_name
        replacements[f"<column:{column_name}>"] = column_name
        replacements[f"{{value:{column_name}}}"] = column_value
        replacements[f"<value:{column_name}>"] = column_value
        replacements[f"{{column:{alias}}}"] = column_name
        replacements[f"<column:{alias}>"] = column_name
        replacements[f"{{value:{alias}}}"] = column_value
        replacements[f"<value:{alias}>"] = column_value

    rewritten = snippet
    for source, target in replacements.items():
        if target:
            rewritten = rewritten.replace(source, str(target))

    rewritten = re.sub(r"\bdf\b", context["dataset_var_name"], rewritten)
    rewritten = re.sub(r"\brow\b", context["row_var_name"], rewritten)
    return rewritten


def build_message(snippet: str, rule_draft: str, context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Contextualize the snippet with the provided CSV sample.",
            "",
            "Rules:",
            "- Keep the snippet language and overall intent.",
            "- Replace generic placeholders with concrete dataset values.",
            "- Prefer exact CSV column names when referencing fields.",
            "- Prefer the provided dataset_var_name and row_var_name for new variable names.",
            "- Do not add explanations before or after the snippet.",
            "",
            "Original snippet:",
            snippet,
            "",
            "Rule-based draft:",
            rule_draft,
            "",
            "CSV row context:",
            json.dumps(context["row"], ensure_ascii=False, indent=2),
            "",
            "Column aliases:",
            json.dumps(context["column_aliases"], ensure_ascii=False, indent=2),
            "",
            "Representative columns:",
            json.dumps(context["representative_columns"], ensure_ascii=False, indent=2),
            "",
            "Return only the final contextualized snippet.",
        ]
    )


def build_request_record(
    snippet_record: dict[str, Any],
    csv_path: Path,
    row_index: int,
    row: dict[str, str],
    prompt_key: str,
    tables_key: str,
) -> dict[str, Any]:
    snippet = snippet_record["snippet"].strip()
    if not snippet:
        raise ValueError("Snippet must be non-empty.")

    context = build_rule_context(csv_path, row_index, row)
    rule_draft = apply_rule_replacements(snippet, context)
    message = build_message(snippet, rule_draft, context)

    output_record = dict(snippet_record)
    output_record["snippet"] = snippet
    output_record["row_index"] = row_index
    output_record["csv_file"] = str(csv_path)
    output_record["rule_based_draft"] = rule_draft
    output_record[prompt_key] = message
    output_record[tables_key] = context
    return output_record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_prepared_input_path(args: argparse.Namespace, output_path: Path) -> tuple[Path, bool]:
    if args.prepared_input_file:
        return Path(args.prepared_input_file), False
    if args.prepare_only:
        return output_path, False
    with tempfile.NamedTemporaryFile(
        prefix="contextualize_snippets_",
        suffix=".jsonl",
        delete=False,
        dir=output_path.parent,
    ) as handle:
        return Path(handle.name), True


def run() -> None:
    args = parse_args()
    csv_path = Path(args.csv_file)
    output_path = Path(args.output_file)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = read_csv_row(csv_path, args.row_index)
    snippets = load_snippets(args)
    records = [
        build_request_record(s, csv_path, args.row_index, row, args.prompt_key, args.tables_key)
        for s in snippets
    ]

    prepared_input_path, is_temporary = resolve_prepared_input_path(args, output_path)
    write_jsonl(prepared_input_path, records)

    if args.prepare_only:
        return

    from batch_infer_vllm import run_batch_inference_file

    try:
        run_batch_inference_file(
            prepared_input_path,
            output_path,
            model_name=args.model_name,
            prompt_key=args.prompt_key,
            tables_key=args.tables_key,
            response_key=args.response_key,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
        )
    finally:
        if is_temporary and not args.keep_prepared_file and prepared_input_path.exists():
            prepared_input_path.unlink()


if __name__ == "__main__":
    run()
