"""Phase 3 verification: celebrity dataset + mixed robot/celebrity batches.

Checks:
  1. CelebrityIdentificationDataset(streaming=True) yields raw items with the
     expected key structure (image tensor + 'task' string + is_celebrity_only).
  2. MixedRobotCelebrityDataset over a tiny mock robot dataset + the streamed
     celebrity dataset mixes the two roughly in the configured ratio.
  3. A forward pass on a hand-assembled mixed batch produces sensible
     action_loss / text_loss values and the lm_head receives non-zero grad.
  4. A controlled batch with is_celebrity_only=True for every sample produces
     action_loss == 0 (action supervision is correctly excluded for celebrity-
     only samples).

The forward-pass parts bypass the standard processor pipeline and build a
model-ready batch directly (random tokens, like the Phase 2 test). The mixed-
dataset checks exercise the *raw item* contract independently of the
processor.
"""

from __future__ import annotations

import logging
import sys
import types

# Bootstrap: stub out broken groot import — transformers>=5.4 vs groot_n1.py
# upstream incompatibility on this checkout. Phase 3 does not use groot.
_stub_groot_pkg = types.ModuleType("lerobot.policies.groot")
_stub_cfg = types.ModuleType("lerobot.policies.groot.configuration_groot")
_stub_cfg.GrootConfig = type("GrootConfig", (), {})
_stub_model = types.ModuleType("lerobot.policies.groot.modeling_groot")
_stub_model.GrootPolicy = type("GrootPolicy", (), {})
_stub_proc = types.ModuleType("lerobot.policies.groot.processor_groot")
_stub_proc.make_groot_pre_post_processors = lambda *a, **k: None
sys.modules.setdefault("lerobot.policies.groot", _stub_groot_pkg)
sys.modules.setdefault("lerobot.policies.groot.configuration_groot", _stub_cfg)
sys.modules.setdefault("lerobot.policies.groot.modeling_groot", _stub_model)
sys.modules.setdefault("lerobot.policies.groot.processor_groot", _stub_proc)
_stub_groot_pkg.GrootConfig = _stub_cfg.GrootConfig
_stub_groot_pkg.GrootPolicy = _stub_model.GrootPolicy
_stub_groot_pkg.make_groot_pre_post_processors = _stub_proc.make_groot_pre_post_processors

import torch
from torch.utils.data import Dataset

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.celebrity_dataset import (
    CelebrityIdentificationDataset,
    MixedRobotCelebrityDataset,
)
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import extract_celebrity_name
from lerobot.utils.constants import (
    ACTION,
    IS_CELEBRITY_ONLY,
    OBS_ANSWER_LABELS,
    OBS_ANSWER_TOKENS,
    OBS_ID_QUERY_ATTENTION_MASK,
    OBS_ID_QUERY_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> SmolVLAConfig:
    cfg = SmolVLAConfig(load_vlm_weights=False, device="cpu", **kwargs)
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    return cfg


def _make_pretokenized_batch(
    batch_size: int = 4,
    is_celebrity_mask: list[bool] | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a model-ready (post-preprocessor) batch like the Phase 2 test."""
    b = {
        OBS_LANGUAGE_TOKENS: torch.randint(0, 5000, (batch_size, 48), device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((batch_size, 48), dtype=torch.bool, device=device),
        OBS_ID_QUERY_TOKENS: torch.randint(0, 5000, (batch_size, 24), dtype=torch.long, device=device),
        OBS_ID_QUERY_ATTENTION_MASK: torch.ones((batch_size, 24), dtype=torch.bool, device=device),
        OBS_ANSWER_TOKENS: torch.randint(0, 5000, (batch_size, 16), dtype=torch.long, device=device),
        OBS_ANSWER_LABELS: torch.randint(0, 5000, (batch_size, 16), dtype=torch.long, device=device),
        OBS_STATE: torch.randn((batch_size, 6), device=device),
        ACTION: torch.randn((batch_size, 50, 6), device=device),
        "observation.image": torch.rand((batch_size, 3, 224, 224), device=device),
    }
    if is_celebrity_mask is not None:
        assert len(is_celebrity_mask) == batch_size
        b[IS_CELEBRITY_ONLY] = torch.tensor(is_celebrity_mask, dtype=torch.bool, device=device)
    return b


class MockRobotDataset(Dataset):
    """Minimal stand-in for LeRobotDataset for the mixing-ratio check."""

    def __init__(self, length: int = 32):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        return {
            "observation.image": torch.rand(3, 256, 256),
            "observation.state": torch.zeros(6),
            "action": torch.zeros(50, 6),
            "task": "place the coke on Taylor Swift",
            "index": torch.tensor(idx, dtype=torch.long),
            "episode_index": torch.tensor(0, dtype=torch.long),
            "frame_index": torch.tensor(idx, dtype=torch.long),
            "task_index": torch.tensor(0, dtype=torch.long),
            "timestamp": torch.tensor(float(idx), dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_celebrity_dataset_structure():
    logger.info("=" * 80)
    logger.info("Test 1: CelebrityIdentificationDataset(streaming=True) structure")
    logger.info("=" * 80)
    ds = CelebrityIdentificationDataset(
        dataset_name="tonyassi/celebrity-1000",
        streaming=True,
        max_examples=8,
        chunk_size=50,
        action_dim=6,
        state_dim=6,
        image_camera_key="observation.image",
    )
    logger.info(f"len(dataset) = {len(ds)}")
    assert len(ds) > 0, "dataset must materialize at least one example"
    ex = ds[0]
    logger.info(f"item keys: {sorted(ex.keys())}")
    for k, v in ex.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            logger.info(f"  {k}: {type(v).__name__} = {v!r}")
    # Structural checks
    assert "observation.image" in ex
    img = ex["observation.image"]
    assert img.ndim == 3 and img.shape[0] == 3, f"image must be (3, H, W), got {tuple(img.shape)}"
    assert img.dtype == torch.float32 and 0.0 <= img.min() <= img.max() <= 1.0
    assert "task" in ex and isinstance(ex["task"], str) and ex["task"].strip()
    assert ex[IS_CELEBRITY_ONLY].item() is True
    # Crucial: the task string must round-trip through extract_celebrity_name()
    name = extract_celebrity_name(ex["task"])
    logger.info(f"  extract_celebrity_name({ex['task']!r}) -> {name!r}")
    assert name is not None and name.strip(), (
        "task string must encode the celebrity name in a form the preprocessor can parse"
    )
    logger.info("✓ Test 1 PASSED\n")
    return ds


def test_mixed_dataset_ratio(celeb_ds: CelebrityIdentificationDataset):
    logger.info("=" * 80)
    logger.info("Test 2: MixedRobotCelebrityDataset ratio sampling")
    logger.info("=" * 80)
    robot_ds = MockRobotDataset(length=20)
    mix = MixedRobotCelebrityDataset(
        robot_dataset=robot_ds, celebrity_dataset=celeb_ds,
        celebrity_mix_ratio=0.3, seed=42,
    )
    logger.info(f"len(mixed) = {len(mix)} (matches robot len = {len(robot_ds)})")
    assert len(mix) == len(robot_ds)
    n = 20
    counts = {"robot": 0, "celebrity": 0}
    sample_keys = None
    for i in range(n):
        item = mix[i]
        is_celeb = bool(item[IS_CELEBRITY_ONLY].item())
        counts["celebrity" if is_celeb else "robot"] += 1
        if sample_keys is None:
            sample_keys = sorted(item.keys())
    logger.info(f"After {n} samples: {counts}")
    logger.info(f"Sample item keys: {sample_keys}")
    # Loose ratio check (n=20 is small; aim for plausibility not statistical rigor).
    assert counts["celebrity"] > 0, "expected at least one celebrity sample at mix_ratio=0.3"
    assert counts["robot"] > 0, "expected at least one robot sample at mix_ratio=0.3"
    logger.info("✓ Test 2 PASSED\n")


def test_forward_mixed_batch():
    logger.info("=" * 80)
    logger.info("Test 3: Forward pass on mixed batch (2 celebrity + 2 robot)")
    logger.info("=" * 80)
    config = _make_config(text_loss_weight=0.5, num_unfrozen_vlm_layers=2)
    policy = SmolVLAPolicy(config)
    policy.train()

    batch = _make_pretokenized_batch(batch_size=4, is_celebrity_mask=[True, False, True, False])
    loss, loss_dict = policy.forward(batch, reduction="mean")
    logger.info(f"loss = {loss.item():.4f}")
    logger.info(f"loss_dict keys: {list(loss_dict.keys())}")
    logger.info(f"  action_loss = {loss_dict['action_loss']:.4f}")
    logger.info(f"  text_loss   = {loss_dict['text_loss']:.4f}")
    logger.info(f"  total_loss  = {loss_dict['total_loss']:.4f}")
    logger.info(f"  n_robot_samples = {loss_dict.get('n_robot_samples')}")
    assert loss_dict["n_robot_samples"] == 2
    # action_loss should equal (sum of robot per-sample losses) / 2.
    # text_loss is computed over the full batch.
    expected_total = loss_dict["action_loss"] + 0.5 * loss_dict["text_loss"]
    assert abs(expected_total - loss_dict["total_loss"]) < 1e-3, (
        f"total_loss mismatch: expected {expected_total} got {loss_dict['total_loss']}"
    )
    logger.info("✓ Test 3 PASSED\n")


def test_all_celebrity_zero_action_loss():
    logger.info("=" * 80)
    logger.info("Test 4: 100% celebrity-only batch -> action_loss == 0")
    logger.info("=" * 80)
    config = _make_config(text_loss_weight=0.5, num_unfrozen_vlm_layers=2)
    policy = SmolVLAPolicy(config)
    policy.train()

    batch = _make_pretokenized_batch(batch_size=3, is_celebrity_mask=[True, True, True])
    loss, loss_dict = policy.forward(batch, reduction="mean")
    logger.info(f"action_loss = {loss_dict['action_loss']:.6f}")
    logger.info(f"text_loss   = {loss_dict['text_loss']:.4f}")
    logger.info(f"total_loss  = {loss_dict['total_loss']:.4f}")
    logger.info(f"n_robot_samples = {loss_dict.get('n_robot_samples')}")
    assert loss_dict["n_robot_samples"] == 0
    assert abs(loss_dict["action_loss"]) < 1e-6, (
        f"action_loss must be 0 when no robot samples, got {loss_dict['action_loss']}"
    )
    # Sanity: total_loss should be just 0.5 * text_loss.
    assert abs(loss_dict["total_loss"] - 0.5 * loss_dict["text_loss"]) < 1e-3
    logger.info("✓ Test 4 PASSED\n")


def test_gradient_flow_on_mixed_batch():
    logger.info("=" * 80)
    logger.info("Test 5: lm_head gradient flow on mixed batch")
    logger.info("=" * 80)
    config = _make_config(text_loss_weight=0.5, num_unfrozen_vlm_layers=2)
    policy = SmolVLAPolicy(config)
    policy.train()

    batch = _make_pretokenized_batch(batch_size=4, is_celebrity_mask=[True, False, True, False])
    loss, _ = policy.forward(batch, reduction="mean")
    loss.backward()

    info = []
    for name, p in policy.model.vlm_with_expert.named_parameters():
        if "lm_head" in name and p.requires_grad:
            g = p.grad
            info.append((name, None if g is None else float(g.abs().sum().item()),
                         None if g is None else float(g.norm().item())))
    for name, abs_sum, norm in info:
        logger.info(f"  {name}: grad_abs_sum={abs_sum}, grad_norm={norm}")
    assert info, "expected at least one trainable lm_head param"
    assert all(s is not None and s > 0 for _, s, _ in info), \
        f"lm_head grads must be non-zero, got {info}"
    logger.info("✓ Test 5 PASSED\n")


if __name__ == "__main__":
    logger.info("Phase 3 verification\n")
    try:
        celeb_ds = test_celebrity_dataset_structure()
        test_mixed_dataset_ratio(celeb_ds)
        test_forward_mixed_batch()
        test_all_celebrity_zero_action_loss()
        test_gradient_flow_on_mixed_batch()
        logger.info("=" * 80)
        logger.info("✓ ALL PHASE 3 TESTS PASSED")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}", exc_info=True)
        raise
