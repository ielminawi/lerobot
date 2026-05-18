"""External celebrity-identification dataset for SmolVLA co-training (Phase 3).

Provides two classes:

- ``CelebrityIdentificationDataset``: wraps a HuggingFace celebrity-name image
  dataset (default ``tonyassi/celebrity-1000``) and yields *raw* items shaped
  like LeRobotDataset's ``__getitem__``: an image tensor + a ``task`` string
  that embeds the celebrity name so the existing SmolVLA preprocessor pipeline
  (``AnswerTokenizerProcessorStep``, ``IdQueryTokenizerProcessorStep``) picks
  it up unchanged. State and action fields are zero placeholders; the
  ``IS_CELEBRITY_ONLY`` flag tells the model to skip action loss.

- ``MixedRobotCelebrityDataset``: wraps a robot ``LeRobotDataset`` and a
  ``CelebrityIdentificationDataset``. ``__len__`` mirrors the robot dataset;
  each ``__getitem__(idx)`` decides per-sample (deterministic from idx)
  whether to return the robot frame or a random celebrity item, in the
  configured ``celebrity_mix_ratio``.

Returning raw items (image + ``task`` string) rather than pre-tokenized
tensors keeps the data flow uniform with the standard LeRobot pipeline — both
sources feed into the same preprocessor downstream. This is a deviation from
the original Phase 3 spec, made deliberately to avoid having to bypass the
preprocessor for one source while applying it to the other.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import Dataset

from lerobot.utils.constants import IS_CELEBRITY_ONLY


def _name_from_label(features, label: Any) -> str:
    """Resolve a label (int class id or str) to the celebrity-name string."""
    if isinstance(label, str):
        return label
    label_feature = features.get("label") if features is not None else None
    if label_feature is not None and hasattr(label_feature, "int2str"):
        try:
            return label_feature.int2str(int(label))
        except Exception:
            pass
    return str(label)


def _pil_to_tensor(image, channels: int = 3) -> torch.Tensor:
    """Convert a PIL image to a (C, H, W) float tensor in [0, 1]."""
    from PIL import Image as PILImage  # local import to keep import time light

    if isinstance(image, torch.Tensor):
        # assume already (C, H, W) in [0, 1]
        return image.float()
    if not isinstance(image, PILImage.Image):
        # Try to coerce numpy or bytes
        if hasattr(image, "convert"):
            pil = image
        else:
            raise TypeError(f"Unsupported image type from HF dataset: {type(image)}")
    else:
        pil = image
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    import numpy as np

    arr = np.array(pil, dtype=np.uint8, copy=True)  # writable copy, (H, W, C)
    tensor = torch.from_numpy(arr).float() / 255.0  # in [0, 1]
    tensor = tensor.permute(2, 0, 1).contiguous()  # (C, H, W)
    return tensor


class CelebrityIdentificationDataset(Dataset):
    """Indexable celebrity-name dataset that mirrors LeRobotDataset's raw output.

    Each item is a dict with the same key structure that the standard SmolVLA
    preprocessor pipeline expects from a robot frame:

      - ``observation.image``: float tensor (C, H, W) in [0, 1]
      - ``observation.state``: zero tensor (placeholder; unused for text loss)
      - ``action``: zero tensor of shape (chunk_size, action_dim) (placeholder)
      - ``task``: string containing the celebrity name in a form
        ``extract_celebrity_name()`` can parse (e.g. "identify the person on
        Taylor Swift")
      - ``is_celebrity_only``: scalar bool tensor True
      - ``index``, ``episode_index``, ``timestamp``, ``frame_index``,
        ``task_index``: zero placeholders for compatibility with code that
        reads these.
    """

    def __init__(
        self,
        dataset_name: str = "tonyassi/celebrity-1000",
        split: str = "train",
        streaming: bool = False,
        max_examples: int | None = None,
        chunk_size: int = 50,
        action_dim: int = 32,
        state_dim: int = 32,
        image_camera_key: str = "observation.image",
        task_template: str = "identify the person on {name}",
    ) -> None:
        from datasets import load_dataset  # local import: heavy

        self.dataset_name = dataset_name
        self.image_camera_key = image_camera_key
        self.task_template = task_template
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self._streaming = streaming

        if streaming:
            # Streaming dataset is not indexable; materialise the first
            # ``max_examples`` items into memory. Use a sensible default so
            # tests do not silently materialise the whole thing.
            ds_iter = load_dataset(dataset_name, split=split, streaming=True)
            self._hf_features = ds_iter.features
            n = max_examples if max_examples is not None else 64
            self._items: list[dict[str, Any]] = []
            for i, ex in enumerate(ds_iter):
                if i >= n:
                    break
                self._items.append(ex)
        else:
            hf = load_dataset(dataset_name, split=split)
            self._hf_features = hf.features
            if max_examples is not None:
                hf = hf.select(range(min(max_examples, len(hf))))
            self._items = hf

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self._items[idx]
        image_tensor = _pil_to_tensor(ex["image"])
        name = _name_from_label(self._hf_features, ex.get("label"))

        task_str = self.task_template.format(name=name)
        item: dict[str, Any] = {
            self.image_camera_key: image_tensor,
            "observation.state": torch.zeros(self.state_dim, dtype=torch.float32),
            "action": torch.zeros(self.chunk_size, self.action_dim, dtype=torch.float32),
            "task": task_str,
            IS_CELEBRITY_ONLY: torch.tensor(True, dtype=torch.bool),
            # Compatibility placeholders that LeRobotDataset items carry.
            "index": torch.tensor(int(idx), dtype=torch.long),
            "episode_index": torch.tensor(0, dtype=torch.long),
            "frame_index": torch.tensor(0, dtype=torch.long),
            "task_index": torch.tensor(0, dtype=torch.long),
            "timestamp": torch.tensor(0.0, dtype=torch.float32),
        }
        return item


class MixedRobotCelebrityDataset(Dataset):
    """Per-index random mix of a robot ``LeRobotDataset`` and a celebrity dataset.

    ``__len__`` mirrors the robot dataset (so one epoch traverses every robot
    frame exactly once). ``__getitem__(idx)``:

      - with probability ``celebrity_mix_ratio``: returns a *random* celebrity
        item, marked ``IS_CELEBRITY_ONLY=True``.
      - otherwise: returns ``robot[idx]`` with ``IS_CELEBRITY_ONLY=False``
        injected.

    Delegates ``.meta``, ``.num_frames``, ``.num_episodes``, ``.episodes`` to
    the robot dataset so the rest of the training pipeline (processors that
    read ``meta.stats``, sampler construction etc.) keeps working.
    """

    def __init__(
        self,
        robot_dataset: Dataset,
        celebrity_dataset: CelebrityIdentificationDataset,
        celebrity_mix_ratio: float = 0.0,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= celebrity_mix_ratio <= 1.0:
            raise ValueError(f"celebrity_mix_ratio must be in [0, 1], got {celebrity_mix_ratio}")
        if celebrity_mix_ratio > 0.0 and len(celebrity_dataset) == 0:
            raise ValueError("celebrity_dataset is empty but celebrity_mix_ratio > 0")
        self.robot_dataset = robot_dataset
        self.celebrity_dataset = celebrity_dataset
        self.celebrity_mix_ratio = float(celebrity_mix_ratio)
        # A per-instance RNG so behavior is deterministic across workers iff
        # seed is set per-worker; for a single-process test, default seed=0
        # gives reproducible mixing.
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.robot_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._rng.random() < self.celebrity_mix_ratio:
            celeb_idx = self._rng.randrange(len(self.celebrity_dataset))
            item = self.celebrity_dataset[celeb_idx]
            # IS_CELEBRITY_ONLY already set by the celebrity dataset.
            return item
        item = self.robot_dataset[idx]
        # Inject the flag for robot items so collated batches always have it.
        if IS_CELEBRITY_ONLY not in item:
            item = dict(item)
            item[IS_CELEBRITY_ONLY] = torch.tensor(False, dtype=torch.bool)
        return item

    # ---- attribute delegation so the training pipeline still works ----
    @property
    def meta(self):
        return self.robot_dataset.meta

    @property
    def num_frames(self):
        return getattr(self.robot_dataset, "num_frames", len(self.robot_dataset))

    @property
    def num_episodes(self):
        return getattr(self.robot_dataset, "num_episodes", 0)

    @property
    def episodes(self):
        return getattr(self.robot_dataset, "episodes", None)
