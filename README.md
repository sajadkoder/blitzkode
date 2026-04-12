# BlitzKode

BlitzKode is a local AI coding assistant repository built around a quantized GGUF model, a FastAPI server, and a lightweight browser UI.

This repo contains three different layers of work:

- A current serving path in `server.py` plus `frontend/index.html`
- Training and export experiments under `scripts/`
- Local model artifacts, checkpoints, and a vendored `llama.cpp/` checkout

## What This Repo Actually Does

The main application serves a local coding assistant over HTTP:

- `GET /` returns the bundled frontend
- `GET /health` reports frontend/model availability
- `GET /info` returns runtime metadata
- `POST /generate` returns a single completion
- `POST /generate/stream` streams tokens as server-sent events

The current serving path uses `llama_cpp` against `blitzkode.gguf` by default, but it can be pointed at another artifact with environment variables.

## Current Entry Points

- `python server.py`
  Starts the FastAPI app that serves the current UI and API.
- `python -m unittest discover -s tests -v`
  Runs the HTTP smoke tests for the current server.
- `python scripts/test_inference.py`
  Runs a direct checkpoint smoke test for a LoRA adapter checkpoint.
- `python scripts/export_gguf.py`
  Merges a checkpoint into its base model and prepares a Hugging Face directory for GGUF conversion.

## Runtime Dependencies

Serving path:

- `fastapi`
- `uvicorn`
- `pydantic`
- `llama-cpp-python`

Training and export path:

- `torch`
- `transformers`
- `peft`
- `datasets`
- `trl`
- `gradio`

This repository does not currently include a locked dependency file for the top-level app, so install the packages needed for the path you want to run.

## Quick Start

1. Make sure the frontend file exists at `frontend/index.html`.
2. Make sure a GGUF model exists at `blitzkode.gguf`, or set `BLITZKODE_MODEL_PATH`.
3. Install the serving dependencies listed above.
4. Run:

```bash
python server.py
```

5. Open `http://localhost:7860`.

## Configuring The Server

`server.py` supports these environment variables:

| Variable | Purpose |
| --- | --- |
| `BLITZKODE_MODEL_PATH` | Override the GGUF model path |
| `BLITZKODE_FRONTEND_PATH` | Override the frontend HTML path |
| `BLITZKODE_HOST` | Bind host |
| `BLITZKODE_PORT` | Bind port |
| `BLITZKODE_THREADS` | CPU thread count passed to llama.cpp |
| `BLITZKODE_N_CTX` | Context window for inference |
| `BLITZKODE_BATCH` | Batch size passed to llama.cpp |
| `BLITZKODE_MAX_PROMPT_LENGTH` | Max accepted prompt length in characters |
| `BLITZKODE_PRELOAD_MODEL` | Preload the model on startup when set to a truthy value |

The default context window is a conservative CPU-friendly value. If you want a larger context and have enough RAM, raise `BLITZKODE_N_CTX`.

## Repo Layout

```text
BlitzKode/
|-- server.py
|-- frontend/
|   `-- index.html
|-- tests/
|   `-- test_server.py
|-- scripts/
|   |-- build_dataset.py
|   |-- build_full_dataset.py
|   |-- train_sft.py
|   |-- train_grpo.py
|   |-- train_dpo.py
|   |-- export_gguf.py
|   `-- test_inference.py
|-- checkpoints/
|-- exported/
|-- datasets/
|-- models/
|-- llama.cpp/
`-- blitzkode.gguf
```

## Training Scripts At A Glance

These scripts are not all part of the live serving path, but they explain how the project evolved:

- `scripts/build_dataset.py`
  Writes a very small seed SFT dataset.
- `scripts/build_full_dataset.py`
  Builds a larger mixed dataset from handcrafted prompts and optional Hugging Face sources.
- `scripts/train_sft.py`
  Runs LoRA SFT against a local Qwen 2.5 1.5B base model directory.
- `scripts/train_grpo.py`
  Runs a GRPO-style continuation step with simple heuristic reward functions.
- `scripts/train_dpo.py`
  Runs DPO over handcrafted preference pairs.
- `scripts/export_gguf.py`
  Merges a checkpoint and prints the correct local `llama.cpp` conversion command.

There are also older prototype apps in `scripts/` such as `final_web.py`, `optimized_web.py`, `web_chat.py`, and `web_chat_pro.py`. The current app entry point is `server.py`.

## Important Caveats

- The repo contains multiple checkpoint families: `sft-1.5b-v1`, `grpo-v1`, `dpo-v1`, and `blitzkode-v2`.
- The exact provenance of the checked-in `blitzkode.gguf` is not pinned in a manifest inside the repo.
- Several training scripts still contain hard-coded Windows paths and should be treated as local experiments, not portable tooling.
- The vendored `llama.cpp/` directory has its own tests and dependency files; those are separate from the top-level BlitzKode app.

## Tests

The current automated coverage is intentionally small and focused on the live server:

```bash
python -m unittest discover -s tests -v
```

That suite verifies:

- frontend serving
- health reporting
- prompt validation
- non-streaming generation
- streaming generation
- graceful handling when the model file is missing

## Model Documentation

See [MODEL_CARD.md](MODEL_CARD.md) for a cautious summary of what can and cannot be claimed about the model lineage, training data, intended use, and known limitations.
