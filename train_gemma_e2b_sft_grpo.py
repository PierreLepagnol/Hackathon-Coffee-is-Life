#!/usr/bin/env python3
"""Two-stage Gemma 4 E2B fine-tuning with Unsloth (SFT then GRPO).

See `docs/DATA_FORMATS.md` for accepted dataset shapes.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only


DEFAULT_GRPO_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. "
    "Write your reasoning inside <reasoning>...</reasoning> and your final answer inside "
    "<answer>...</answer>."
)

FORBIDDEN_HISTORY_MARKERS = (
    "<reasoning>", "</reasoning>", "<answer>", "</answer>",
    "<|think|>", "<|channel>thought", "<channel|>",
)

ALLOWED_DATASET_ROLES = {"system", "user", "assistant"}

ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gemma 4 E2B with SFT and GRPO using Unsloth.")
    parser.add_argument("--stage", choices=["sft", "grpo", "all"], default="all")
    parser.add_argument("--model-name", default="unsloth/Ministral-3B-Instruct-2410")
    parser.add_argument("--chat-template", default="mistral")
    parser.add_argument("--output-dir", default="outputs/gemma-4-e2b")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument("--sft-dataset", default=None)
    parser.add_argument("--sft-dataset-config", default=None)
    parser.add_argument("--sft-split", default="train")
    parser.add_argument("--sft-text-field", default="text")
    parser.add_argument("--sft-messages-field", default="messages")
    parser.add_argument("--sft-prompt-field", default="prompt")
    parser.add_argument("--sft-response-field", default="response")
    parser.add_argument("--sft-system-field", default=None)
    parser.add_argument("--sft-system-prompt", default=None)
    parser.add_argument("--sft-batch-size", type=int, default=1)
    parser.add_argument("--sft-gradient-accumulation", type=int, default=4)
    parser.add_argument("--sft-learning-rate", type=float, default=2e-4)
    parser.add_argument("--sft-warmup-steps", type=int, default=5)
    parser.add_argument("--sft-max-steps", type=int, default=60)

    parser.add_argument("--grpo-dataset", default=None)
    parser.add_argument("--grpo-dataset-config", default=None)
    parser.add_argument("--grpo-split", default="train")
    parser.add_argument("--grpo-prompt-field", default="prompt")
    parser.add_argument("--grpo-answer-field", default="answer")
    parser.add_argument("--grpo-messages-field", default="messages")
    parser.add_argument("--grpo-system-prompt", default=DEFAULT_GRPO_SYSTEM_PROMPT)
    parser.add_argument("--grpo-batch-size", type=int, default=4)
    parser.add_argument("--grpo-gradient-accumulation", type=int, default=2)
    parser.add_argument("--grpo-learning-rate", type=float, default=5e-5)
    parser.add_argument("--grpo-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grpo-max-steps", type=int, default=60)
    parser.add_argument("--grpo-num-generations", type=int, default=2)
    parser.add_argument("--grpo-temperature", type=float, default=1.0)
    parser.add_argument("--grpo-max-prompt-samples", type=int, default=128)
    parser.add_argument("--grpo-max-completion-length", type=int, default=None)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.stage in {"sft", "all"} and not args.sft_dataset:
        raise ValueError("--sft-dataset is required for stage=sft or stage=all.")
    if args.stage in {"grpo", "all"} and not args.grpo_dataset:
        raise ValueError("--grpo-dataset is required for stage=grpo or stage=all.")


def load_any_dataset(path_or_name: str, split: str, config_name: str | None) -> Dataset:
    path = Path(path_or_name)
    if path.exists():
        if path.is_dir():
            loaded = load_from_disk(str(path))
            if isinstance(loaded, Dataset):
                return loaded
            if split in loaded:
                return loaded[split]
            if "train" in loaded:
                return loaded["train"]
            raise ValueError(f"DatasetDict at {path} does not contain split '{split}'.")
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path), split="train")
        if suffix == ".csv":
            return load_dataset("csv", data_files=str(path), split="train")
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=str(path), split="train")
        raise ValueError(f"Unsupported local dataset format: {path}")
    if config_name:
        return load_dataset(path_or_name, name=config_name, split=split)
    return load_dataset(path_or_name, split=split)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return content_to_text(content["text"])
        if "content" in content:
            return content_to_text(content["content"])
        return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(content_to_text(item["content"]))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def strip_leading_bos(text: str, tokenizer: Any) -> str:
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token and text.startswith(bos_token):
        return text[len(bos_token):]
    return text.removeprefix("<bos>")


def validate_gemma_multiturn_history(messages: list[dict[str, Any]], context: str) -> None:
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in ALLOWED_DATASET_ROLES:
            raise ValueError(
                f"{context}: Gemma 4 datasets should use only standard roles system/user/assistant. "
                f"Found unsupported role '{role}' at turn {index}."
            )
    if len(messages) < 3:
        return
    for index, message in enumerate(messages[:-1]):
        if message.get("role") != "assistant":
            continue
        content = content_to_text(message.get("content")).lower()
        if any(marker in content for marker in FORBIDDEN_HISTORY_MARKERS):
            raise ValueError(
                f"{context}: Gemma 4 multi-turn history should keep only final visible answers. "
                f"Remove prior thought blocks from assistant turn {index}."
            )


def validate_grpo_gold_answer(answer: str, context: str) -> None:
    lowered = answer.lower()
    if any(marker in lowered for marker in FORBIDDEN_HISTORY_MARKERS):
        raise ValueError(
            f"{context}: GRPO gold answers must contain the final answer only, not reasoning or thinking tags."
        )


def prepare_sft_dataset(dataset: Dataset, tokenizer: Any, args: argparse.Namespace) -> Dataset:
    columns = set(dataset.column_names)
    if args.sft_text_field in columns:
        return dataset
    if "conversations" in columns and args.sft_messages_field not in columns:
        dataset = dataset.rename_column("conversations", args.sft_messages_field)
        columns = set(dataset.column_names)

    def render_messages(messages: list[dict[str, Any]]) -> str:
        validate_gemma_multiturn_history(messages, context="SFT sample")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return strip_leading_bos(text, tokenizer)

    def format_batch(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        batch_size = len(next(iter(examples.values())))
        rendered_texts: list[str] = []
        for index in range(batch_size):
            if args.sft_messages_field in examples:
                rendered_texts.append(render_messages(examples[args.sft_messages_field][index]))
                continue
            if "conversations" in examples:
                rendered_texts.append(render_messages(examples["conversations"][index]))
                continue

            prompt = content_to_text(examples[args.sft_prompt_field][index])
            response = content_to_text(examples[args.sft_response_field][index])
            system_prompt = args.sft_system_prompt
            if args.sft_system_field and args.sft_system_field in examples:
                system_prompt = content_to_text(examples[args.sft_system_field][index]) or system_prompt

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
            rendered_texts.append(render_messages(messages))
        return {"text": rendered_texts}

    return dataset.map(format_batch, batched=True)


def extract_prompt_and_answer_from_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if not messages:
        return [], ""
    last_message = messages[-1]
    if last_message.get("role") == "assistant":
        return messages[:-1], content_to_text(last_message.get("content"))
    return messages, ""


def prepare_grpo_dataset(dataset: Dataset, args: argparse.Namespace) -> Dataset:
    columns = set(dataset.column_names)
    if args.grpo_prompt_field in columns and args.grpo_answer_field in columns:
        pass
    elif args.grpo_messages_field not in columns:
        raise ValueError(
            "GRPO dataset must contain either prompt+answer fields or a messages field ending with an assistant answer."
        )

    def format_batch(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        batch_size = len(next(iter(examples.values())))
        prompts: list[list[dict[str, str]]] = []
        answers: list[str] = []
        for index in range(batch_size):
            if args.grpo_messages_field in examples:
                messages = examples[args.grpo_messages_field][index]
                prompt_messages, answer = extract_prompt_and_answer_from_messages(messages)
                if args.grpo_answer_field in examples:
                    answer = content_to_text(examples[args.grpo_answer_field][index]) or answer
            else:
                prompt_value = examples[args.grpo_prompt_field][index]
                answer = content_to_text(examples[args.grpo_answer_field][index])
                if isinstance(prompt_value, list):
                    prompt_messages = prompt_value
                else:
                    prompt_messages = [{"role": "user", "content": content_to_text(prompt_value)}]

            if args.grpo_system_prompt:
                has_system = bool(prompt_messages) and prompt_messages[0].get("role") == "system"
                if not has_system:
                    prompt_messages = [{"role": "system", "content": args.grpo_system_prompt}, *prompt_messages]

            validate_gemma_multiturn_history(prompt_messages, context="GRPO prompt")
            if not answer.strip():
                raise ValueError(
                    "Each GRPO sample must have a non-empty gold answer. "
                    "If you use --grpo-messages-field, the last message must be assistant or you must provide --grpo-answer-field."
                )
            validate_grpo_gold_answer(answer, context="GRPO sample")
            prompts.append(prompt_messages)
            answers.append(answer)
        return {"prompt": prompts, "answer": answers}

    return dataset.map(format_batch, batched=True, remove_columns=dataset.column_names)


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_final_answer(text: str) -> str:
    match = ANSWER_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return content_to_text(completion.get("content", completion))
    if isinstance(completion, list):
        return "\n".join(part for part in (completion_to_text(item) for item in completion) if part).strip()
    return str(completion)


def exact_match_reward(completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion, gold in zip(completions, answer):
        prediction = normalize_answer(extract_final_answer(completion_to_text(completion)))
        reference = normalize_answer(content_to_text(gold))
        rewards.append(2.0 if reference and prediction == reference else 0.0)
    return rewards


def partial_match_reward(completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion, gold in zip(completions, answer):
        prediction = normalize_answer(extract_final_answer(completion_to_text(completion)))
        reference = normalize_answer(content_to_text(gold))
        reward = 0.5 if prediction and reference and (prediction in reference or reference in prediction) else 0.0
        rewards.append(reward)
    return rewards


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        lowered = completion_to_text(completion).lower()
        has_reasoning = "<reasoning>" in lowered and "</reasoning>" in lowered
        has_answer = "<answer>" in lowered and "</answer>" in lowered
        rewards.append((0.25 if has_reasoning else 0.0) + (0.75 if has_answer else 0.0))
    return rewards


def non_empty_reward(completions: list[Any], **_: Any) -> list[float]:
    return [0.25 if completion_to_text(c).strip() else -0.25 for c in completions]


def estimate_max_prompt_length(dataset: Dataset, tokenizer: Any, sample_size: int) -> int:
    limit = min(len(dataset), sample_size)
    max_length = 0
    for index in range(limit):
        rendered_prompt = tokenizer.apply_chat_template(
            dataset[index]["prompt"], tokenize=False, add_generation_prompt=True
        )
        token_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
        max_length = max(max_length, len(token_ids))
    return max_length


def load_sft_model_and_tokenizer(args: argparse.Namespace, adapter_path: str | None = None) -> tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            random_state=args.seed,
            use_gradient_checkpointing="unsloth",
        )
    return model, tokenizer


def load_grpo_model_and_tokenizer(args: argparse.Namespace, adapter_path: str | None = None) -> tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    return model, tokenizer


def save_adapter(model: Any, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def run_sft(model: Any, tokenizer: Any, args: argparse.Namespace) -> Path:
    dataset = load_any_dataset(args.sft_dataset, args.sft_split, args.sft_dataset_config)
    dataset = prepare_sft_dataset(dataset, tokenizer, args)

    output_dir = Path(args.output_dir) / "sft"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_length=args.max_seq_length,
            per_device_train_batch_size=args.sft_batch_size,
            gradient_accumulation_steps=args.sft_gradient_accumulation,
            warmup_steps=args.sft_warmup_steps,
            max_steps=args.sft_max_steps,
            learning_rate=args.sft_learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to=args.report_to,
        ),
    )

    instruction_part = "<|turn>user\n"
    response_part = "<|turn>model\n"
    sample_text = dataset[0]["text"] if len(dataset) else ""
    if instruction_part in sample_text and response_part in sample_text:
        trainer = train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)
    else:
        print("Skipping response-only masking because the inferred Gemma markers were not found in the SFT text.")

    trainer.train()
    adapter_dir = output_dir / "adapter"
    save_adapter(model, tokenizer, adapter_dir)
    return adapter_dir


def run_grpo(model: Any, tokenizer: Any, args: argparse.Namespace) -> Path:
    dataset = load_any_dataset(args.grpo_dataset, args.grpo_split, args.grpo_dataset_config)
    dataset = prepare_grpo_dataset(dataset, args)

    effective_batch_size = args.grpo_batch_size * args.grpo_gradient_accumulation
    if effective_batch_size / args.grpo_num_generations <= 2:
        raise ValueError(
            "GRPO requires effective_batch_size / num_generations > 2. "
            "Increase --grpo-batch-size or --grpo-gradient-accumulation, or reduce --grpo-num-generations."
        )

    max_prompt_length = estimate_max_prompt_length(dataset, tokenizer, args.grpo_max_prompt_samples)
    max_completion_length = args.grpo_max_completion_length
    if max_completion_length is None:
        remaining_context = args.max_seq_length - max_prompt_length - 1
        if remaining_context < 32:
            raise ValueError("Prompts are too long for GRPO with the current --max-seq-length.")
        max_completion_length = remaining_context

    output_dir = Path(args.output_dir) / "grpo"
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, non_empty_reward, partial_match_reward, exact_match_reward],
        args=GRPOConfig(
            output_dir=str(output_dir),
            temperature=args.grpo_temperature,
            learning_rate=args.grpo_learning_rate,
            weight_decay=0.001,
            warmup_ratio=args.grpo_warmup_ratio,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            per_device_train_batch_size=args.grpo_batch_size,
            gradient_accumulation_steps=args.grpo_gradient_accumulation,
            num_generations=args.grpo_num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=args.grpo_max_steps,
            seed=args.seed,
            report_to=args.report_to,
            use_vllm=False,
            loss_type="grpo",
            mask_truncated_completions=False,
        ),
        train_dataset=dataset,
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    save_adapter(model, tokenizer, adapter_dir)
    return adapter_dir


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model = None
    tokenizer = None
    active_adapter_path = args.adapter_path

    if args.stage in {"sft", "all"}:
        model, tokenizer = load_sft_model_and_tokenizer(args, adapter_path=active_adapter_path)
        active_adapter_path = str(run_sft(model, tokenizer, args))
        model = None
        tokenizer = None

    if args.stage == "grpo":
        model, tokenizer = load_grpo_model_and_tokenizer(args, adapter_path=active_adapter_path)

    if args.stage in {"grpo", "all"}:
        if model is None or tokenizer is None:
            model, tokenizer = load_grpo_model_and_tokenizer(args, adapter_path=active_adapter_path)
        run_grpo(model, tokenizer, args)


if __name__ == "__main__":
    main()
