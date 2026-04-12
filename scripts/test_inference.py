#!/usr/bin/env python3
"""
Small local inference smoke test for a LoRA checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "checkpoints" / "dpo-v1" / "final",
    REPO_ROOT / "checkpoints" / "grpo-v1" / "final",
    REPO_ROOT / "checkpoints" / "sft-1.5b-v1" / "final",
]


def pick_default_checkpoint() -> Path:
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    return CHECKPOINT_CANDIDATES[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=pick_default_checkpoint(),
        help="Adapter checkpoint to load for the smoke test.",
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function to find the two sum of indices that add up to target.",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate.",
    )
    return parser.parse_args()


def main() -> None:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)

    peft_config = PeftConfig.from_pretrained(str(checkpoint_path))
    print(f"Loading base model: {peft_config.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    model.eval()

    print("\n" + "=" * 60)
    print("Testing model...")
    print("=" * 60)
    print(f"\nPrompt: {args.prompt}\n")
    print("Response:")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            do_sample=True,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
