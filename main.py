import json
import pathlib
import sys

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "unsloth/gemma-4-E2B-it"
SNIPPETS_PATH = pathlib.Path("artifacts/contextualized.jsonl")


def load_snippets(path: pathlib.Path, max_snippets: int = 10) -> str:
    snippets = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= max_snippets:
                break
            record = json.loads(line)
            title = record.get("metadata", {}).get("snippet", {}).get("title", record["id"])
            code = record.get("response", "")
            if code:
                snippets.append(f"# {title}\n{code}")
    return "\n\n".join(snippets)


SNIPPET_EXAMPLES = load_snippets(SNIPPETS_PATH) if SNIPPETS_PATH.exists() else ""

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)

print("Model device:", next(model.parameters()).device)


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python") :].strip()
    elif text.startswith("```"):
        text = text[len("```") :].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(payload: ChatRequest) -> ChatResponse:
    print("Python version:", sys.version)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))

    system_prompt = (
        "Return only valid Python Polars code. "
        "No markdown fences. "
        "Assign the final Polars DataFrame to result. "
        f"Available datasets: {json.dumps(payload.tables, ensure_ascii=False)}"
    )
    if SNIPPET_EXAMPLES:
        system_prompt += f"\n\nHere are some contextualized Polars code examples for reference:\n\n{SNIPPET_EXAMPLES}"

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": payload.message,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return ChatResponse(response=strip_code_fence(response))
