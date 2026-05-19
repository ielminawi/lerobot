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

    vlm = policy.model.vlm_with_expert.vlm
    processor = policy.model.vlm_with_expert.processor
    tokenizer = processor.tokenizer
    vlm_model = vlm.model
    image_token_id = vlm_model.image_token_id
    image_seq_len = vlm_model.image_seq_len

    print(f"Loading image: {args.image}")
    img = load_image(args.image)

    # MATCH TRAINING FORMAT exactly.
    # During training (_compute_text_loss):
    #   input_ids = [<image>]*image_seq_len + id_query_tokens + answer_tokens
    #   labels    = -100 on image + id_query, real labels on answer
    #   pixel_values: (B, 1, C, H, W) in [-1, 1] (lerobot prepare_images normalization)
    #
    # At inference we feed everything up to the answer and let `generate` produce
    # the answer tokens autoregressively.

    import torch.nn.functional as F
    import numpy as np

    # Image: resize to 512x512 with padding (matches lerobot's resize_imgs_with_padding),
    # normalize to [-1, 1].
    img_arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0  # (H, W, C) in [0, 1]
    img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)   # (1, C, H, W)
    img_t = F.interpolate(img_t, size=(512, 512), mode="bilinear", align_corners=False)
    img_t = img_t * 2.0 - 1.0                                          # [-1, 1]
    pixel_values = img_t.unsqueeze(1).to(args.device,
        dtype=vlm_model.vision_model.embeddings.patch_embedding.weight.dtype)

    # Tokenize the id-query prompt exactly as IdQueryTokenizerProcessorStep did.
    id_query_max_length = 24  # default in config; override with --id-query-max-length if changed
    q = tokenizer(
        args.prompt,
        max_length=id_query_max_length,
        padding="max_length",
        padding_side="right",
        truncation=True,
        return_tensors="pt",
    )
    id_query_tokens = q["input_ids"].to(args.device)
    id_query_mask = q["attention_mask"].to(args.device)

    # Build the full input: [<image>]*image_seq_len + id_query_tokens
    batch_size = 1
    image_tokens = torch.full((batch_size, image_seq_len), image_token_id,
                              dtype=torch.long, device=args.device)
    input_ids = torch.cat([image_tokens, id_query_tokens], dim=1)
    # MATCH TRAINING: attention_mask is all 1s in _compute_text_loss (including pad).
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=args.device)

    print(f"id_query_tokens decoded: {tokenizer.decode(id_query_tokens[0], skip_special_tokens=False)!r}")
    print(f"input_ids shape: {tuple(input_ids.shape)}, pixel_values shape: {tuple(pixel_values.shape)}")

    # First, do a single forward pass and look at the top-k logits at the answer
    # boundary — this tells us what the model wants to predict for token 0.
    print("Forward pass to inspect first-token logits ...")
    with torch.no_grad():
        out = vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )
    last_logits = out.logits[0, -1]  # logits at the LAST position of the prefix
    topk = torch.topk(last_logits, k=10)
    print("Top-10 token predictions for first answer position:")
    for prob_logit, tok_id in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  id={tok_id:6d}  logit={prob_logit:8.3f}  token={tokenizer.decode([tok_id])!r}")

    print("\nGenerating (greedy, max_new_tokens={}) ...".format(args.max_new_tokens))
    with torch.no_grad():
        out_ids = vlm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    new_tokens = out_ids[0, input_ids.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\n>>> VLM says: {answer.strip()!r}")
    print(f">>> raw token ids: {new_tokens.tolist()}")


if __name__ == "__main__":
    main()
