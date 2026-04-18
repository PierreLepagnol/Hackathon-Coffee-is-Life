#!/usr/bin/env python3
"""Convert llm-polars github_validated_pairs.jsonl into SFT and GRPO training files.

Input record shape:
  {
    "segment_id": "...",
    "question": "...",
    "schema": [{"name": "...", "dtype": "...", "non_null_sample": [...]}, ...],
    "answer": "...",
    "validation": {"contract_errors": [...], "referenced_columns": [...], ...},
    ...
  }

Output:
  sft.jsonl  — messages format compatible with --sft-messages-field messages
  grpo.jsonl — {"prompt": [system, user], "answer": "<polars code>"}

Records with non-empty validation.contract_errors are skipped by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SFT_SYSTEM_PROMPT = (
    "Return only valid Python Polars code. "
    "No markdown fences. "
    "Assign the final Polars DataFrame to result. "
)

GRPO_SYSTEM_PROMPT = (
    "Write your reasoning inside <reasoning>...</reasoning> and your final Polars code inside <answer>...</answer>. "
    "No markdown fences. "
    "Assign the final Polars DataFrame to result. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare github_validated_pairs.jsonl for SFT and GRPO training.")
    parser.add_argument("--input", required=True, help="Path to github_validated_pairs.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to write sft.jsonl and grpo.jsonl")
    parser.add_argument("--keep-invalid", action="store_true", help="Include records with contract_errors")
    parser.add_argument("--skip-malformed", action="store_true", help="Skip malformed records instead of aborting")
    return parser.parse_args()


def load_records(path: Path, skip_malformed: bool) -> list[tuple[int, dict]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                if skip_malformed:
                    print(f"WARNING line {line_no}: invalid JSON — skipped ({exc})", file=sys.stderr)
                    continue
                raise ValueError(f"Line {line_no}: invalid JSON: {exc}") from exc
    return records


def build_system_content(schema: list[dict], *, grpo: bool = False) -> str:
    base = GRPO_SYSTEM_PROMPT if grpo else SFT_SYSTEM_PROMPT
    return base + f"Available schema: {json.dumps(schema, ensure_ascii=False)}"


def process(
    raw_records: list[tuple[int, dict]],
    keep_invalid: bool,
    skip_malformed: bool,
) -> tuple[list[dict], list[dict], list[str]]:
    sft_records: list[dict] = []
    grpo_records: list[dict] = []
    warnings: list[str] = []

    for line_no, record in raw_records:
        segment_id = record.get("segment_id", f"line-{line_no}")

        for field in ("question", "schema", "answer"):
            if field not in record:
                msg = f"Line {line_no} ({segment_id}): missing '{field}' field."
                if skip_malformed:
                    warnings.append("SKIPPED " + msg)
                    break
                raise ValueError(msg)
        else:
            question = record["question"]
            schema = record["schema"]
            answer = record["answer"]

            if not isinstance(question, str) or not question.strip():
                msg = f"Line {line_no} ({segment_id}): 'question' must be a non-empty string."
                if skip_malformed:
                    warnings.append("SKIPPED " + msg)
                    continue
                raise ValueError(msg)

            if not isinstance(answer, str) or not answer.strip():
                msg = f"Line {line_no} ({segment_id}): 'answer' must be a non-empty string."
                if skip_malformed:
                    warnings.append("SKIPPED " + msg)
                    continue
                raise ValueError(msg)

            if not isinstance(schema, list):
                msg = f"Line {line_no} ({segment_id}): 'schema' must be a list."
                if skip_malformed:
                    warnings.append("SKIPPED " + msg)
                    continue
                raise ValueError(msg)

            contract_errors = record.get("validation", {}).get("contract_errors", [])
            if contract_errors and not keep_invalid:
                warnings.append(
                    f"SKIPPED line {line_no} ({segment_id}): "
                    f"contract_errors={json.dumps(contract_errors)}"
                )
                continue

            sft_system_msg = {"role": "system", "content": build_system_content(schema)}
            grpo_system_msg = {"role": "system", "content": build_system_content(schema, grpo=True)}
            user_msg = {"role": "user", "content": question.strip()}
            assistant_msg = {"role": "assistant", "content": answer.strip()}

            sft_records.append({"messages": [sft_system_msg, user_msg, assistant_msg]})
            grpo_records.append({"prompt": [grpo_system_msg, user_msg], "answer": answer.strip()})

    return sft_records, grpo_records, warnings


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.is_file():
        sys.exit(f"ERROR: input file not found: {input_path}")

    raw_records = load_records(input_path, args.skip_malformed)
    sft_records, grpo_records, warnings = process(raw_records, args.keep_invalid, args.skip_malformed)

    for w in warnings:
        print(w, file=sys.stderr)

    sft_path = output_dir / "github_sft.jsonl"
    grpo_path = output_dir / "github_grpo.jsonl"
    write_jsonl(sft_path, sft_records)
    write_jsonl(grpo_path, grpo_records)

    print(f"Input records : {len(raw_records)}")
    print(f"SFT records   : {len(sft_records)}  -> {sft_path}")
    print(f"GRPO records  : {len(grpo_records)}  -> {grpo_path}")
    if warnings:
        print(f"Skipped       : {len(warnings)}")


if __name__ == "__main__":
    main()
