# Refactor Plan — Coffee is Life

**Status:** Audit complete — ready to implement
**Generated:** 2025-04-18

---

## 1. Critical Issues

### 1.1 Security — Exposed SSH Private Key

**Location:** `README.md` (lines 30-36)

```markdown
private key for SSH:

----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
...
-----END OPENSSH PRIVATE KEY-----
```

**Action:**
1. Remove this block from `README.md` immediately
2. Rotate/revoke the exposed SSH key
3. If the key was committed to git history, clean it:
   ```bash
   git filter-branch --force --tree-filter 'rm -f README.md' --tag-name-filter cat -- --all
   git push origin --force --all
   ```

---

## 2. Structural Issues

### 2.1 Root is Too Crowded

The repo root mixes:
- App source files (`main.py`, `prompting.py`, `hf_utils.py`)
- Dataset pipeline scripts (`contextualize_snippets.py`, `dry_run_contextualize_snippets.py`)
- Inference script (`batch_infer_vllm.py`)
- Training script (`train_gemma_e2b_sft_grpo.py`)
- Generated outputs (`outputs/`, `artifacts/`)
- Reference corpus (`polars-code/`)

**Impact:** Hard to distinguish first-party code from external data or generated artifacts.

### 2.2 Duplicated Logic

| Duplicate | Location A | Location B |
|-----------|-----------|-------------|
| `strip_code_fence` | `main.py` | `prompting.py` |
| `build_system_prompt` | `main.py` | `prompting.py` |
| `build_chat_messages` | `main.py` | `prompting.py` |
| `build_chat_prompt` | `main.py` | `prompting.py` |

Worse: `prompting.py` is the authoritative version, but `main.py` duplicates it.

### 2.3 Large Multi-Responsibility Scripts

| Script | LOC | Responsibilities |
|--------|-----|----------------|
| `contextualize_snippets.py` | 416 | CLI parsing, CSV reading, snippet normalization, rule heuristics, prompt building, temp file lifecycle, orchestration |
| `batch_infer_vllm.py` | 241 | JSONL I/O, record validation, environment diagnostics, prompt rendering, vLLM runner |
| `train_gemma_e2b_sft_grpo.py` | 645 | Arg parsing, dataset loading/formatting, model loading, SFT, GRPO, orchestration |

Each script does too many things — hard to test, debug, or reuse a single concern.

### 2.4 Import-Time Side Effects

**Location:** `main.py` (lines 19-24)

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
```

Loading the model at import time breaks:
- Testing without a GPU
- Running offline validation
- Fast script imports for CLI tools

### 2.5 Incomplete Dependency Declaration

**Location:** `pyproject.toml`

Actual imports missing from dependencies:
- `datasets`
- `trl`
- `peft`
- `unsloth`
- `transformers`
- `vllm`

---

## 3. Best Refactor Plan

### Phase 0 — Safety

- Remove exposed SSH private key from `README.md`
- Rotate the key
- Add a short "how to get credentials" section instead

### Phase 1 — Reorganize First-Party Code

Create a package for application code:

```text
coffee_is_life/
  __init__.py
  prompting.py      # shared prompt utilities
  hf_utils.py        # HuggingFace token configuration
  api.py            # FastAPI app (thin wrapper)
  contextualization/
    __init__.py
    io.py           # JSONL, CSV, snippet loading
    rules.py        # rule-based replacements
    pipeline.py    # orchestration
  inference/
    __init__.py
    io.py           # input/output helpers
    runner.py      # vLLM execution
    diagnostics.py # environment diagnostics
  training/
    __init__.py
    args.py        # argparse configuration
    data.py        # dataset normalization
    rewards.py    # reward functions
    models.py     # model loading
    pipeline.py  # SFT + GRPO orchestration
scripts/            # thin entry points for backward compatibility
tests/             # unit tests
```

Keep current root scripts as thin wrappers that import from the package.

### Phase 2 — Deduplicate Shared Logic

- Make `coffee_is_life/prompting.py` the single source of truth
- Remove duplicate prompt helpers from `main.py`
- Move model/tokenizer loading behind a function or FastAPI lifespan hook

### Phase 3 — Split Contextualization Flow

Refactor `contextualize_snippets.py` into:
- `contextualization/io.py` — file/CSV/snippet loading
- `contextualization/rules.py` — rule-based replacement logic
- `contextualization/pipeline.py` — orchestration

Then make `dry_run_contextualize_snippets.py` reuse the same core.

### Phase 4 — Split Inference Flow

Refactor `batch_infer_vllm.py` into:
- `inference/io.py` — JSONL I/O, record validation
- `inference/runner.py` — vLLM execution
- `inference/diagnostics.py` — environment diagnostics

### Phase 5 — Split Training Script

Refactor `train_gemma_e2b_sft_grpo.py` into:
- `training/args.py` — argparse configuration
- `training/data.py` — dataset normalization (4 formats)
- `training/rewards.py` — 4 reward functions
- `training/models.py` — model loading (SFT vs GRPO)
- `training/pipeline.py` — SFT + GRPO orchestration

This is the biggest maintainability win.

### Phase 6 — Minimal Quality Gates

**Dependencies:** `pytest`, `ruff`

**Unit tests to add:**
- Prompt building from `prompting.py`
- Snippet normalization from `contextualize_snippets.py`
- Rule replacements
- JSONL validation from `batch_infer_vllm.py`
- Reward functions from training

---

## 4. Priority Order

| # | Task | Why |
|---|------|-----|
| 1 | Remove leaked SSH key | Security |
| 2 | Create package structure | Organization |
| 3 | Deduplicate prompting | Maintainability |
| 4 | Thin `main.py` | Testability |
| 5 | Split contextualization | Reusability |
| 6 | Split inference | Reusability |
| 7 | Split training | Maintainability |
| 8 | Fix dependencies | Reproducibility |
| 9 | Add tests | Confidence |

---

## 5. Suggested First Implementation

If you want me to implement this, I'd start with the **safe first pass**:

1. Remove the leaked secret from `README.md`
2. Create the `coffee_is_life/` package structure
3. Deduplicate prompting — remove helpers from `main.py`
4. Thin `main.py` — load model lazily
5. Leave `contextualize_snippets.py`, `batch_infer_vllm.py`, and `train_gemma_e2b_sft_grpo.py` as-is for now (keep them working)

This gives you:
- A cleaner package to grow into
- Fixed duplication
- Working scripts you can refactor later

---

## 6. File Inventory

| File | Type | Purpose |
|------|------|---------|
| `main.py` | App | FastAPI server — needs thinning |
| `prompting.py` | Lib | Prompt utilities — authoritative |
| `hf_utils.py` | Lib | HF token config |
| `contextualize_snippets.py` | Script | Snippet contextualization pipeline — needs split |
| `dry_run_contextualize_snippets.py` | Script | Dry-run version — should reuse contextualization/ |
| `batch_infer_vllm.py` | Script | vLLM batch inference — needs split |
| `train_gemma_e2b_sft_grpo.py` | Script | SFT + GRPO training — needs split |
| `polars-code/` | Reference | Example snippets — keep as-is |
| `artifacts/` | Generated | Dataset artifacts — keep as-is |
| `outputs/` | Generated | Training outputs — keep as-is |
| `data/` | Input | Input CSVs — keep as-is |
| `docs/` | Docs | Documentation — keep as-is |
| `scripts/` | Scripts | Shell tooling — keep as-is |

---

## 7. Next Steps

1. **Approve this plan** ��� confirm priority order
2. **Remove the SSH key** — or tell me to do it
3. **Start Phase 1** — create package structure
4. **Proceed incrementally** — one phase at a time