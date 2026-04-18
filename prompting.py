from __future__ import annotations

import json
from typing import Any, Mapping

MODEL_NAME = "google/gemma-4-E2B-it"

DEFAULT_SYSTEM_PROMPT = (
    "Return only valid Python Polars code. "
    "No markdown fences. "
    "Assign the final Polars DataFrame to result. "
)

CONTEXTUALIZE_SNIPPET_SYSTEM_PROMPT = (
    "Return only the contextualized snippet text. "
    "No markdown fences. "
    "Preserve the original snippet intent and most of its structure. "
    "Replace generic placeholders with concrete column names, variable names, and values from the provided CSV row context. "
    "Use only columns and values that exist in the provided context. "
)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python") :].strip()
    elif text.startswith("```"):
        text = text[len("```") :].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def build_system_prompt(tables: Mapping[str, Any]) -> str:
    if tables.get("__task") == "contextualize_snippet":
        return (
            CONTEXTUALIZE_SNIPPET_SYSTEM_PROMPT
            + f"Available context: {json.dumps(dict(tables), ensure_ascii=False)}"
        )
    return (
        DEFAULT_SYSTEM_PROMPT
        + f"Available datasets: {json.dumps(dict(tables), ensure_ascii=False)}"
    )


def build_chat_messages(message: str, tables: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(tables)},
        {"role": "user", "content": message},
    ]


def build_chat_prompt(tokenizer: Any, message: str, tables: Mapping[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        build_chat_messages(message, tables),
        tokenize=False,
        add_generation_prompt=True,
    )
