#!/usr/bin/env python3
"""
Stage 2: LoRA/PEFT Fine-Tuning Stub

This script outlines how you would fine-tune an open-source LLM
(e.g., LLaMA/Mistral/Falcon) using PEFT/LoRA. It prints the planned
steps and exits. Replace with real training code when your environment
has GPUs and the required libraries installed.

Suggested dependencies (install when ready):
  pip install transformers datasets peft accelerate bitsandbytes

Usage:
  python3 src/train/lora_finetune_stub.py --model mistral-7b --data path/to/jsonl
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral-7b", help="Base model name/path")
    ap.add_argument("--data", default="data/clean", help="Training data path")
    ap.add_argument("--output", default="artifacts/lora", help="Output dir")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    args = ap.parse_args()

    print("[LoRA Stub] Configuration:")
    for k, v in vars(args).items():
        print(f"- {k}: {v}")

    print("\nPlanned steps:")
    steps = [
        "Load tokenizer and base model (8-bit if available)",
        "Prepare supervised dataset (instruction -> response)",
        "Wrap model with PEFT LoRA adapters",
        "Train with gradient accumulation and mixed precision",
        "Evaluate on held-out finance QA tasks",
        "Save adapters and training config",
    ]
    for i, s in enumerate(steps, 1):
        print(f"{i}. {s}")

    print("\nNote: This is a scaffold. Integrate with HF Transformers + PEFT to run.")


if __name__ == "__main__":
    main()

