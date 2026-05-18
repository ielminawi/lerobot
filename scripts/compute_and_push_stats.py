"""Compute meta/stats.json for a LeRobot dataset missing it, then push to HF.

Usage:
    python scripts/compute_and_push_stats.py \
        --repo-id ETHrobotlearning/task3-TOY-clean-illumnation \
        --revision main

Notes:
- Iterates every frame of the dataset to compute mean/std/min/max for state
  and action, and per-channel mean/std/min/max for image features (sub-sampled
  to keep memory bounded).
- Writes ``meta/stats.json`` in the local LeRobot cache for the dataset, then
  uploads it back to the same HF dataset repo (you must be logged in with a
  token that has write access to that repo).
- Skips int/bool scalar columns like `index`, `episode_index`, `task_index`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi


# bootstrap groot stub so importing lerobot.policies.* doesn't crash on the
# upstream `groot_n1.py` × transformers incompat
import types

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

from lerobot.datasets.lerobot_dataset import LeRobotDataset


SCALAR_INT_KEYS = {"index", "episode_index", "task_index", "frame_index"}


def is_image_key(key: str) -> bool:
    return "image" in key.lower()


def per_dim_stats_1d(arr: np.ndarray) -> dict:
    """For state/action arrays of shape (N,) or (N, dim) or (N, T, dim)."""
    # Collapse any leading time dims so we get (N*, dim) or (N*,).
    if arr.ndim >= 2:
        arr = arr.reshape(-1, arr.shape[-1])  # (M, dim)
    return {
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": (arr.std(axis=0).astype(np.float32) + 1e-8).tolist(),
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "count": [int(arr.shape[0])],
    }


def per_channel_image_stats(arr: np.ndarray) -> dict:
    """For images stacked as (N, C, H, W), per-channel stats reshaped to (C, 1, 1)."""
    axes = (0, 2, 3)
    mean = arr.mean(axis=axes, keepdims=True).squeeze(0).astype(np.float32)  # (C, 1, 1)
    std = arr.std(axis=axes, keepdims=True).squeeze(0).astype(np.float32) + 1e-8
    mn = arr.min(axis=axes, keepdims=True).squeeze(0).astype(np.float32)
    mx = arr.max(axis=axes, keepdims=True).squeeze(0).astype(np.float32)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": mn.tolist(),
        "max": mx.tolist(),
        "count": [int(arr.shape[0])],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--image-subsample", type=int, default=50,
                   help="Take every Nth frame's image for image-stats computation.")
    p.add_argument("--max-image-samples", type=int, default=512,
                   help="Hard cap on number of images used for stats.")
    p.add_argument("--no-upload", action="store_true",
                   help="Compute stats and write locally but skip the HF push.")
    args = p.parse_args()

    print(f"Loading dataset {args.repo_id} (revision={args.revision}) ...")
    ds = LeRobotDataset(args.repo_id, revision=args.revision)
    print(f"  num_frames={ds.num_frames}  num_episodes={ds.num_episodes}")

    # Per-feature accumulators
    numeric_acc: dict[str, list[np.ndarray]] = defaultdict(list)  # state, action, etc.
    image_acc: dict[str, list[np.ndarray]] = defaultdict(list)    # camera images

    image_samples_taken = 0
    take_every = max(1, args.image_subsample)
    cap = args.max_image_samples

    for i in range(ds.num_frames):
        item = ds[i]
        for k, v in item.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k in SCALAR_INT_KEYS:
                continue
            if v.dtype in (torch.long, torch.int64, torch.int32, torch.bool):
                continue
            arr = v.detach().cpu().numpy()
            if is_image_key(k):
                # Sub-sample images globally
                if (i % take_every == 0) and image_samples_taken < cap:
                    # arr shape can be (C, H, W) or (T, C, H, W).
                    if arr.ndim == 4:
                        arr = arr.reshape(-1, *arr.shape[-3:])
                        image_acc[k].extend([arr[j] for j in range(arr.shape[0])])
                        image_samples_taken += arr.shape[0]
                    else:
                        image_acc[k].append(arr)
                        image_samples_taken += 1
            else:
                numeric_acc[k].append(arr)

        if i % 500 == 0:
            print(f"  processed {i}/{ds.num_frames} ...")

    print("Computing stats ...")
    stats = {}
    for k, vlist in numeric_acc.items():
        big = np.stack(vlist).astype(np.float32)
        stats[k] = per_dim_stats_1d(big)
        print(f"  {k}: shape={big.shape}  mean[0]={stats[k]['mean'][0] if isinstance(stats[k]['mean'], list) else stats[k]['mean']:.4f}")

    for k, vlist in image_acc.items():
        # All images same (C, H, W). Stack to (N, C, H, W).
        # Items may come in [0, 1] float or [0, 255] uint8 — coerce to float32 in [0,1].
        big = np.stack(vlist).astype(np.float32)
        if big.max() > 1.5:  # heuristic: still in 0-255 range
            big = big / 255.0
        stats[k] = per_channel_image_stats(big)
        print(f"  {k}: N={big.shape[0]} images, per-channel mean={stats[k]['mean']}")

    # Write to the local meta/stats.json
    out_path = Path(ds.root) / "meta" / "stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {out_path}")

    if args.no_upload:
        print("--no-upload set, done.")
        return

    print(f"Uploading meta/stats.json to {args.repo_id} on HF Hub ...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo="meta/stats.json",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add meta/stats.json computed by compute_and_push_stats.py",
    )
    print(f"Pushed. Dataset is now trainable.")


if __name__ == "__main__":
    main()
