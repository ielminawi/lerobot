"""Test script for SmolVLA Phase 2: Text loss co-training."""

import logging
import sys
import types

# --- Bootstrap: stub out broken groot import that the local lerobot tree has
# (incompatibility between transformers>=5.4 and groot_n1.py). The Phase 2 test
# does not need groot; this just prevents the package __init__ from failing.
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
# --- end bootstrap

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_ANSWER_LABELS,
    OBS_ANSWER_TOKENS,
    OBS_ID_QUERY_ATTENTION_MASK,
    OBS_ID_QUERY_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def _make_config(**kwargs):
    """Build a SmolVLAConfig with proper input/output features so prepare_images works."""
    cfg = SmolVLAConfig(load_vlm_weights=False, device="cpu", **kwargs)
    cfg.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    cfg.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    return cfg

# Force a stdout handler on our logger even if root was already configured by
# imports (transformers/huggingface_hub call basicConfig at import time).
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)


def create_fake_batch(batch_size: int = 2, device: str = "cpu"):
    """Create a fake batch for testing with all Phase 2 fields."""
    # Shapes follow SmolVLAConfig defaults: tokenizer_max_length=48, chunk_size=50.
    batch = {
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
    return batch


def test_text_loss_enabled():
    """Test Phase 2 with text loss enabled."""
    logger.info("=" * 80)
    logger.info("Test 1: Phase 2 with text_loss_weight=0.5 (co-training enabled)")
    logger.info("=" * 80)
    
    config = _make_config(
        text_loss_weight=0.5,
        num_unfrozen_vlm_layers=2,
        id_query_prompt="Who is the person shown in this image?",
    )
    
    logger.info(f"Config: text_loss_weight={config.text_loss_weight}")
    logger.info(f"Config: num_unfrozen_vlm_layers={config.num_unfrozen_vlm_layers}")
    logger.info(f"Config: id_query_prompt='{config.id_query_prompt}'")
    
    policy = SmolVLAPolicy(config)
    device = "cpu"

    # List trainable params (filter to VLM submodule to avoid noise from the action expert)
    trainable_in_vlm = [
        name for name, p in policy.model.vlm_with_expert.named_parameters()
        if p.requires_grad
    ]
    lm_head_trainable_names = [n for n in trainable_in_vlm if "lm_head" in n]
    logger.info(f"Number of trainable params in vlm_with_expert: {len(trainable_in_vlm)}")
    logger.info(f"Trainable lm_head params: {lm_head_trainable_names}")
    assert lm_head_trainable_names, "lm_head should be trainable when text_loss_weight > 0"
    
    # Create fake batch and forward pass
    batch = create_fake_batch(batch_size=2, device=device)
    
    logger.info("Running forward pass...")
    loss, loss_dict = policy.forward(batch, reduction="mean")
    
    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Loss dict keys: {list(loss_dict.keys())}")
    
    # Check that we got both action_loss and text_loss
    assert "action_loss" in loss_dict, "action_loss should be in loss_dict"
    assert "text_loss" in loss_dict, "text_loss should be in loss_dict when enabled"
    assert "total_loss" in loss_dict, "total_loss should be in loss_dict when text loss enabled"
    
    logger.info(f"action_loss: {loss_dict['action_loss']:.4f}")
    logger.info(f"text_loss: {loss_dict['text_loss']:.4f}")
    logger.info(f"total_loss: {loss_dict['total_loss']:.4f}")
    
    # Verify that total_loss = action_loss + text_loss_weight * text_loss
    expected_total = loss_dict['action_loss'] + config.text_loss_weight * loss_dict['text_loss']
    actual_total = loss_dict['total_loss']
    assert abs(expected_total - actual_total) < 1e-4, f"Total loss mismatch: {expected_total} vs {actual_total}"
    
    logger.info("✓ Test 1 PASSED: Text loss is computed and combined correctly\n")


def test_backward_compatibility():
    """Test that Phase 2 is backward-compatible when text_loss_weight=0."""
    logger.info("=" * 80)
    logger.info("Test 2: Backward compatibility with text_loss_weight=0.0")
    logger.info("=" * 80)
    
    config = _make_config(text_loss_weight=0.0)  # Disabled
    
    logger.info(f"Config: text_loss_weight={config.text_loss_weight}")
    
    policy = SmolVLAPolicy(config)
    device = "cpu"
    
    # Check that lm_head is NOT trainable when text_loss_weight=0
    lm_head_trainable = any(
        param.requires_grad for name, param in policy.model.vlm_with_expert.named_parameters()
        if "lm_head" in name
    )
    logger.info(f"lm_head is trainable: {lm_head_trainable}")
    assert not lm_head_trainable, "lm_head should NOT be trainable when text_loss_weight=0"
    
    # Create fake batch but WITHOUT id_query fields
    batch = create_fake_batch(batch_size=2, device=device)
    # Remove text loss fields
    batch.pop(OBS_ID_QUERY_TOKENS, None)
    batch.pop(OBS_ID_QUERY_ATTENTION_MASK, None)
    batch.pop(OBS_ANSWER_TOKENS, None)
    batch.pop(OBS_ANSWER_LABELS, None)
    
    logger.info("Running forward pass without text loss fields...")
    loss, loss_dict = policy.forward(batch, reduction="mean")
    
    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Loss dict keys: {list(loss_dict.keys())}")
    
    # Check that we only got action_loss (backward compatible)
    assert "action_loss" in loss_dict, "action_loss should be in loss_dict"
    assert "text_loss" not in loss_dict, "text_loss should NOT be in loss_dict when disabled"
    assert "total_loss" not in loss_dict, "total_loss should NOT be in loss_dict when text_loss_weight=0"
    
    logger.info(f"action_loss: {loss_dict['action_loss']:.4f}")
    logger.info("✓ Test 2 PASSED: Backward compatibility maintained\n")


def test_gradient_flow():
    """Test that gradients flow through lm_head AND the unfrozen transformer layers."""
    logger.info("=" * 80)
    logger.info("Test 3: Gradient flow through lm_head and unfrozen layers")
    logger.info("=" * 80)

    config = _make_config(text_loss_weight=0.5, num_unfrozen_vlm_layers=2)
    
    policy = SmolVLAPolicy(config)
    policy.train()
    device = "cpu"
    
    # Get lm_head parameters
    lm_head_params = [
        param for name, param in policy.model.vlm_with_expert.named_parameters()
        if "lm_head" in name and param.requires_grad
    ]
    
    logger.info(f"Found {len(lm_head_params)} trainable lm_head parameters")
    assert len(lm_head_params) > 0, "Should have trainable lm_head parameters"
    
    # Create batch and forward pass
    batch = create_fake_batch(batch_size=2, device=device)
    loss, loss_dict = policy.forward(batch, reduction="mean")
    
    # Compute gradients
    logger.info("Computing gradients...")
    loss.backward()
    
    # Check that lm_head has gradients
    lm_head_grad_info = []
    for name, param in policy.model.vlm_with_expert.named_parameters():
        if "lm_head" in name and param.requires_grad:
            g = param.grad
            if g is None:
                lm_head_grad_info.append((name, None, None))
            else:
                lm_head_grad_info.append((name, float(g.abs().sum().item()), float(g.norm().item())))

    logger.info(f"Found {len(lm_head_grad_info)} trainable lm_head parameters")
    for name, abs_sum, norm in lm_head_grad_info:
        logger.info(f"  {name}: grad_abs_sum={abs_sum}, grad_norm={norm}")
    assert len(lm_head_grad_info) > 0, "Should have trainable lm_head parameters"
    assert all(info[1] is not None for info in lm_head_grad_info), \
        "lm_head should have gradients (not None)"
    # CRITICAL: gradients must be non-zero — a trainable param with zero grad means
    # the loss path is not actually connecting to it.
    assert all(info[1] > 0 for info in lm_head_grad_info), \
        f"lm_head gradients must be non-zero. Got: {lm_head_grad_info}"

    # Verify the `num_unfrozen_vlm_layers` knob actually unfroze the requested
    # layers (this previously silently did nothing due to a param-prefix typo).
    m = policy.model.vlm_with_expert
    expected_unfrozen_count = config.num_unfrozen_vlm_layers
    unfrozen_layer_indices = sorted({
        int(n.split(".layers.")[1].split(".")[0])
        for n, pp in m.vlm.named_parameters()
        if pp.requires_grad and "model.text_model.layers." in n
    })
    logger.info(f"Unfrozen transformer layer indices: {unfrozen_layer_indices}")
    logger.info(f"  (action path uses layers 0..{m.num_vlm_layers - 1}; full stack 0..{m.num_full_vlm_layers - 1})")
    assert len(unfrozen_layer_indices) == expected_unfrozen_count, (
        f"Expected {expected_unfrozen_count} unfrozen transformer layers, got {unfrozen_layer_indices}"
    )
    # Sharing check: every unfrozen layer must sit inside the action path.
    assert all(i < m.num_vlm_layers for i in unfrozen_layer_indices), (
        f"Unfrozen layers must be inside action path (< {m.num_vlm_layers}), got {unfrozen_layer_indices}"
    )
    # And those layers must have non-zero gradient (proving text loss reached them).
    transformer_grad_info = []
    for name, p in m.vlm.named_parameters():
        if "model.text_model.layers." in name and p.requires_grad and p.grad is not None:
            idx = int(name.split(".layers.")[1].split(".")[0])
            transformer_grad_info.append((idx, name, float(p.grad.abs().sum().item())))
    by_layer = {}
    for idx, _, s in transformer_grad_info:
        by_layer.setdefault(idx, 0.0)
        by_layer[idx] += s
    for idx in sorted(by_layer):
        logger.info(f"  layer {idx} total grad_abs_sum: {by_layer[idx]:.4f}")
    assert all(s > 0 for s in by_layer.values()), (
        f"Unfrozen transformer layers must receive non-zero gradient, got {by_layer}"
    )

    logger.info("✓ Test 3 PASSED: Non-zero gradients flow through lm_head AND unfrozen transformer layers\n")


if __name__ == "__main__":
    logger.info("Starting Phase 2 Co-training Verification Tests\n")
    
    try:
        test_text_loss_enabled()
        test_backward_compatibility()
        test_gradient_flow()
        
        logger.info("=" * 80)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}", exc_info=True)
        raise
