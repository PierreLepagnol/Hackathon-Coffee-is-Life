from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterator

from hf_utils import configure_hf_token
from prompting import MODEL_NAME, build_chat_prompt, strip_code_fence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline batch inference with vLLM.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
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
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must be a JSON object.")
            yield line_number, record


def batched(records: Iterator[tuple[int, dict[str, Any]]], batch_size: int) -> Iterator[list[tuple[int, dict[str, Any]]]]:
    batch: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def validate_record(record: dict[str, Any], line_number: int, prompt_key: str, tables_key: str) -> tuple[str, dict[str, Any]]:
    if prompt_key not in record:
        raise ValueError(f"Line {line_number} is missing '{prompt_key}'.")
    if tables_key not in record:
        raise ValueError(f"Line {line_number} is missing '{tables_key}'.")

    message = record[prompt_key]
    tables = record[tables_key]

    if not isinstance(message, str) or not message.strip():
        raise ValueError(f"Line {line_number} field '{prompt_key}' must be a non-empty string.")
    if not isinstance(tables, dict):
        raise ValueError(f"Line {line_number} field '{tables_key}' must be a JSON object.")
    return message, tables


def collect_environment_diagnostics() -> str:
    import torch

    diagnostics = [
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        f"HF_TOKEN configured: {'yes' if os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') else 'no'}",
        f"Torch: {torch.__version__}",
        f"Torch CUDA build: {torch.version.cuda or '<none>'}",
        f"torch.cuda.is_available(): {torch.cuda.is_available()}",
        f"torch.cuda.device_count(): {torch.cuda.device_count()}",
    ]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            diagnostics.append(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    return "\n".join(diagnostics)


def run_batch_inference_file(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str = MODEL_NAME,
    prompt_key: str = "message",
    tables_key: str = "tables",
    response_key: str = "response",
    batch_size: int = 128,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    trust_remote_code: bool = False,
) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    token = configure_hf_token()

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=token
        )
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize tokenizer or vLLM engine.\n"
            f"Model: {model_name}\n"
            f"tensor_parallel_size: {tensor_parallel_size}\n"
            f"gpu_memory_utilization: {gpu_memory_utilization}\n"
            f"max_model_len: {max_model_len}\n"
            f"trust_remote_code: {trust_remote_code}\n\n"
            "Environment diagnostics:\n"
            f"{collect_environment_diagnostics()}"
        ) from exc

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    with output_path.open("w", encoding="utf-8") as handle:
        for batch in batched(iter_jsonl(input_path), batch_size):
            prompts: list[str] = []
            for line_number, record in batch:
                message, tables = validate_record(record, line_number, prompt_key, tables_key)
                prompts.append(build_chat_prompt(tokenizer, message, tables))

            outputs = llm.generate(prompts, sampling_params)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a different number of outputs than prompts.")

            for (line_number, record), result in zip(batch, outputs, strict=True):
                if not result.outputs:
                    raise RuntimeError(f"No generation returned for line {line_number}.")
                best_output = result.outputs[0]
                out = dict(record)
                out[response_key] = strip_code_fence(best_output.text)
                out["finish_reason"] = None if best_output.finish_reason is None else str(best_output.finish_reason)
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def run() -> None:
    args = parse_args()
    run_batch_inference_file(
        Path(args.input_file),
        Path(args.output_file),
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


if __name__ == "__main__":
    run()
