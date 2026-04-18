from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from contextualize_snippets import build_request_record, load_snippets, read_csv_row
from hf_utils import configure_hf_token
from prompting import MODEL_NAME, build_chat_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build contextualization requests and print the final tokenizer-rendered prompt "
            "without loading a model."
        )
    )
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--row-index", type=int, default=1)
    parser.add_argument("--output-file", help="Ignored. Accepted to mirror contextualize_snippets.py commands.")

    snippet_group = parser.add_mutually_exclusive_group(required=True)
    snippet_group.add_argument("--snippet")
    snippet_group.add_argument("--snippets-file")

    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--prompt-key", default="message")
    parser.add_argument("--tables-key", default="tables")
    parser.add_argument("--record-index", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def run() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    csv_path = Path(args.csv_file)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    row = read_csv_row(csv_path, args.row_index)
    snippets = load_snippets(args)
    records = [
        build_request_record(s, csv_path, args.row_index, row, args.prompt_key, args.tables_key)
        for s in snippets
    ]

    if args.record_index <= 0:
        raise ValueError("--record-index must be positive.")
    if args.record_index > len(records):
        raise ValueError(
            f"--record-index {args.record_index} is out of range for {len(records)} prepared records."
        )

    token = configure_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code, token=token
    )

    record = records[args.record_index - 1]
    message: Any = record[args.prompt_key]
    tables: Any = record[args.tables_key]
    prompt = build_chat_prompt(tokenizer, message, tables)
    record_id = record.get("id", f"snippet-{args.record_index}")
    prompt_token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    print(f"Loaded {len(records)} prepared record(s).")
    print(f"Showing record {args.record_index}: {record_id}")
    print(f"Prompt length: {len(prompt)} characters, {prompt_token_count} tokens")
    print()
    print(prompt)


if __name__ == "__main__":
    run()
