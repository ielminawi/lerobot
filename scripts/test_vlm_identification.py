"""Diagnostic: ask the trained SmolVLA's VLM to identify a person in an image.

Bypasses the action expert entirely — just exercises the VLM + lm_head that
the text loss trained. Useful to separate "VLM didn't learn celebrities"
from "VLM learned but action expert doesn't use that knowledge."

Usage:
    python scripts/test_vlm_identification.py \
        --policy-path ETHrobotlearning/smolvla-cotrain-v2-step30k \
        --image <path or HF URL> \
        --prompt "Who is the person shown in this image?"
"""

from __future__ import annotations

import argparse
import sys
import types

# groot stub
for n in [
    "lerobot.policies.groot",
    "lerobot.policies.groot.configuration_groot",
    "lerobot.policies.groot.modeling_groot",
    "lerobot.policies.groot.processor_groot",
]:
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["lerobot.policies.groot.configuration_groot"].GrootConfig = type("GrootConfig", (), {})
sys.modules["lerobot.policies.groot.modeling_groot"].GrootPolicy = type("GrootPolicy", (), {})
sys.modules["lerobot.policies.groot.processor_groot"].make_groot_pre_post_processors = lambda *a, **k: None

import torch
from PIL import Image

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def load_image(src: str) -> Image.Image:
    """Load image from local path or HTTP URL."""
    if src.startswith("http://") or src.startswith("https://"):
        import requests
        from io import BytesIO

        return Image.open(BytesIO(requests.get(src).content)).convert("RGB")
    return Image.open(src).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", required=True, help="HF repo or local path of a trained SmolVLA checkpoint")
    p.add_argument("--image", required=True, help="Local image path or URL to a celebrity image")
    p.add_argument("--prompt", default="Who is the person shown in this image?")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-new-tokens", type=int, default=20)
    args = p.parse_args()

    print(f"Loading policy from {args.policy_path} ...")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(args.device).eval()

    # Pull out the inner HF VLM + its processor.
    vlm = policy.model.vlm_with_expert.vlm
    processor = policy.model.vlm_with_expert.processor

    print(f"Loading image: {args.image}")
    img = load_image(args.image)

    # Build the standard SmolVLM2 chat template prompt.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True)
    print(f"Prompt: {prompt_str!r}")

    inputs = processor(text=prompt_str, images=[img], return_tensors="pt")
    inputs = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    print("Generating ...")
    with torch.no_grad():
        out_ids = vlm.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode([new_tokens], skip_special_tokens=True)[0]
    print(f"\n>>> VLM says: {answer.strip()!r}")


if __name__ == "__main__":
    main()
