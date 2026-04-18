#!/usr/bin/env python3
"""Convert llm-polars pairs.jsonl into SFT and GRPO training files.

Input record shape:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Output:
  sft.jsonl  — same messages format, passes through to --sft-messages-field messages
  grpo.jsonl — {"prompt": [<all turns except last>], "answer": "<last assistant content>"}
               compatible with train_gemma_e2b_sft_grpo.py GRPO Option B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pairs.jsonl for SFT and GRPO training.")
    parser.add_argument("--input", required=True, help="Path to pairs.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to write sft.jsonl and grpo.jsonl")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip malformed records instead of aborting")
    return parser.parse_args()


def load_records(path: Path, skip_invalid: bool) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_invalid:
                    print(f"WARNING line {line_no}: invalid JSON — skipped ({exc})", file=sys.stderr)
                    continue
                raise ValueError(f"Line {line_no}: invalid JSON: {exc}") from exc
            records.append((line_no, record))
    return records


def validate_messages(messages: object, line_no: int) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Line {line_no}: 'messages' must be a non-empty list.")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Line {line_no}: messages[{i}] must be an object.")
        if msg.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"Line {line_no}: messages[{i}] has unsupported role '{msg.get('role')}'.")
        if not isinstance(msg.get("content"), str):
            raise ValueError(f"Line {line_no}: messages[{i}] content must be a string.")
    last_role = messages[-1]["role"]
    if last_role != "assistant":
        raise ValueError(f"Line {line_no}: last message role must be 'assistant', got '{last_role}'.")
    return messages


def process(raw_records: list[tuple[int, dict]], skip_invalid: bool) -> tuple[list[dict], list[dict], list[str]]:
    sft_records: list[dict] = []
    grpo_records: list[dict] = []
    warnings: list[str] = []

    for line_no, record in raw_records:
        if "messages" not in record:
            msg = f"Line {line_no}: missing 'messages' field."
            if skip_invalid:
                warnings.append("SKIPPED " + msg)
                continue
            raise ValueError(msg)

        try:
            messages = validate_messages(record["messages"], line_no)
        except ValueError as exc:
            if skip_invalid:
                warnings.append(f"SKIPPED {exc}")
                continue
            raise

        answer = messages[-1]["content"].strip()
        if not answer:
            msg = f"Line {line_no}: last assistant message is empty."
            if skip_invalid:
                warnings.append("SKIPPED " + msg)
                continue
            raise ValueError(msg)

        sft_records.append({"messages": messages})
        grpo_records.append({"prompt": messages[:-1], "answer": answer})

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

    raw_records = load_records(input_path, args.skip_invalid)
    sft_records, grpo_records, warnings = process(raw_records, args.skip_invalid)

    for w in warnings:
        print(w, file=sys.stderr)

    sft_path = output_dir / "sft.jsonl"
    grpo_path = output_dir / "grpo.jsonl"
    write_jsonl(sft_path, sft_records)
    write_jsonl(grpo_path, grpo_records)

    print(f"Input records : {len(raw_records)}")
    print(f"SFT records   : {len(sft_records)}  -> {sft_path}")
    print(f"GRPO records  : {len(grpo_records)}  -> {grpo_path}")
    if warnings:
        print(f"Skipped       : {len(warnings)}")


if __name__ == "__main__":
    main()
