#!/usr/bin/env python3
"""
Merge a LoRA checkpoint into its base model and save a Hugging Face directory
that can be converted to GGUF with llama.cpp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = REPO_ROOT / "models" / "qwen1.5b"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "exported" / "merged"
CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "checkpoints" / "dpo-v1" / "final",
    REPO_ROOT / "checkpoints" / "grpo-v1" / "final",
    REPO_ROOT / "checkpoints" / "sft-1.5b-v1" / "final",
    REPO_ROOT / "checkpoints" / "blitzkode-v2" / "checkpoint-4",
]


def pick_default_checkpoint() -> Path:
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    return CHECKPOINT_CANDIDATES[0]


def resolve_base_model(checkpoint: Path, override: Path | None) -> Path:
    if override is not None:
        return override

    try:
        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(str(checkpoint))
        configured_path = Path(peft_config.base_model_name_or_path)
        if configured_path.exists():
            return configured_path
    except Exception:
        pass

    return DEFAULT_BASE_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=pick_default_checkpoint(),
        help="Adapter checkpoint to merge before GGUF conversion.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Optional base-model directory. Defaults to the PEFT config path or models/qwen1.5b.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the merged Hugging Face model will be written.",
    )
    return parser.parse_args()


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    base_model_path = resolve_base_model(checkpoint_path, args.base_model)

    print("=" * 60)
    print("EXPORTING BLITZKODE TO GGUF")
    print("=" * 60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Base model: {base_model_path}")
    print(f"Output dir: {output_dir}")

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not base_model_path.exists():
        raise SystemExit(f"Base model not found: {base_model_path}")

    print("\n[LOADING TOKENIZER]")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)

    print("[LOADING BASE MODEL]")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("[MERGING ADAPTERS]")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[SAVING MERGED MODEL]")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    gguf_output = REPO_ROOT / "blitzkode.gguf"
    convert_script = REPO_ROOT / "llama.cpp" / "convert_hf_to_gguf.py"

    print(f"\n[DONE] Merged model saved to: {output_dir}")
    print("\nNext step:")
    print(
        f"  python {convert_script} {output_dir} --outfile {gguf_output} --outtype q4_k_m"
    )


if __name__ == "__main__":
    main()
