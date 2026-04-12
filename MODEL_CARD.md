# BlitzKode Model Card

This model card is intentionally conservative. It only states things that are directly supported by the repository contents, and it calls out where provenance is ambiguous.

## Summary

| Field | Value |
| --- | --- |
| Model name | BlitzKode |
| Served artifact in this repo | `blitzkode.gguf` |
| Serving runtime | `llama_cpp` via `server.py` |
| Likely base model family | Qwen 2.5 1.5B Instruct |
| Artifact lineage confidence | Partial, not complete |

## What Can Be Verified From The Repo

- The live app serves `blitzkode.gguf` by default from `server.py`.
- Multiple training scripts reference a Qwen 2.5 1.5B Instruct base model:
  - `scripts/train_v2.py` references `Qwen/Qwen2.5-1.5B-Instruct`
  - `scripts/train_sft.py` and `scripts/train_max.py` load a local `models/qwen1.5b` directory
- The repo contains local checkpoint directories for several training stages:
  - `checkpoints/sft-1.5b-v1`
  - `checkpoints/grpo-v1`
  - `checkpoints/dpo-v1`
  - `checkpoints/blitzkode-v2`
- The repo contains an exported merged Hugging Face directory under `exported/merged`.
- The repo contains small local dataset files under `datasets/raw/`.

## What Cannot Be Verified Cleanly

- Which exact checkpoint produced the checked-in `blitzkode.gguf`
- Which exact dataset mixture was used for the shipped GGUF artifact
- Any benchmark, accuracy, or pass-rate numbers
- Any formal safety evaluation
- Any external release history beyond what is present in the local files

The main provenance gap is that different scripts point at different checkpoints:

- `scripts/test_inference.py` defaults to `checkpoints/dpo-v1/final`
- historical code in the repo referenced `checkpoints/blitzkode-v2/checkpoint-4` for GGUF export
- the workspace contains both lines of experimentation

Because of that, this repo should be treated as a development snapshot rather than a fully reproducible model release package.

## Intended Use

- Local coding assistance
- Small-scale experimentation with a local model-serving stack
- Educational exploration of SFT, GRPO-style, and DPO-style fine-tuning scripts

## Not Intended Use

- Unsupervised production code changes
- Security-critical code generation without human review
- Medical, legal, or financial advice
- Claims of benchmarked coding performance without separate evaluation evidence

## Training Data Signals In The Repo

The repository suggests a mixed training strategy, but not a single locked recipe.

Observed data sources and patterns include:

- Small handcrafted coding prompts in `scripts/build_dataset.py`
- Larger local algorithm/data-structure samples in `scripts/train_sft.py`
- Synthetic and mixed problem generation in `scripts/build_full_dataset.py`
- Optional Hugging Face pulls mentioned in scripts, including:
  - `sahil2801/CodeAlpaca-20k`
  - `openai/gsm8k`
  - `meta-math/MetaMathQA`
  - `meta-math/MathInstruct`

Important limitation:

- These scripts do not prove that every listed source was used in the final served artifact.
- Some dataset builders mix coding and math content, so the project should not be described as a pure code-only model without extra evidence.

## Training Process Signals In The Repo

The repo contains scripts for several stages:

1. `scripts/train_sft.py`
   LoRA SFT over local coding-style prompt/response pairs.
2. `scripts/train_grpo.py`
   A GRPO-style continuation step using heuristic reward functions named `correctness_reward`, `format_reward`, and `reasoning_reward`.
3. `scripts/train_dpo.py`
   DPO over handcrafted chosen/rejected preference pairs.
4. `scripts/export_gguf.py`
   Merge adapters into the base model and prepare a directory for GGUF conversion through `llama.cpp`.

Another important limitation:

- The GRPO script uses lightweight heuristic rewards based on keyword and formatting checks. That is not the same as executing generated code against robust correctness tests.

## Evaluation

No formal model evaluation suite is stored in the top-level BlitzKode repo.

What is available instead:

- A local HTTP smoke suite for the serving layer in `tests/test_server.py`
- A direct checkpoint smoke script in `scripts/test_inference.py`

These are operational checks, not model-quality evaluations.

## Limitations

- Small-model and quantized-model constraints likely reduce long-context reliability and code accuracy.
- The live server defaults to a conservative context window for CPU inference rather than the full training context implied by the base model family.
- Repository scripts include local absolute paths and prototype variants, so portability is limited.
- Because provenance is incomplete, downstream users should not represent this artifact as fully reproducible from the repo alone.

## Recommended Release Improvements

If this project is going to be shared as a model release, the most useful next steps would be:

- add a manifest that maps `blitzkode.gguf` to a specific checkpoint and base model
- version the exact training dataset inputs or hashes
- record a minimal benchmark and safety-eval report
- add locked dependency files for both serving and training workflows
