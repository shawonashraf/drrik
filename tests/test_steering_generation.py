"""Tests for SAESteering steering mechanics (no model loading, CPU only)."""

import pytest
import torch

from drrik.autoencoder import SparseAutoencoder
from drrik.steering import (
    SAESteering,
    SteeringComponent,
    SteeringVectors,
    _sample_next_token,
    make_steering_hook,
)


def _unit_vector(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _vectors_instance() -> SAESteering:
    """SAESteering built via __new__ with a vectors source (no model)."""
    comps = [SteeringComponent(5, 42, 1.5, _unit_vector(8, seed=3))]
    steering = SAESteering.__new__(SAESteering)
    steering.source = SteeringVectors(comps, hook_path="layer")
    steering.sae = None
    steering.vectors = steering.source
    steering.target_layer = None
    steering.hook_path = "layer"
    steering.device = torch.device("cpu")
    return steering


def test_constructor_requires_layer_for_sae():
    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    with pytest.raises(ValueError, match="layer is required"):
        SAESteering(sae, "meta-llama/Llama-3.1-8B-Instruct")


def test_constructor_rejects_unknown_source():
    with pytest.raises(TypeError, match="SparseAutoencoder or SteeringVectors"):
        SAESteering(42, "meta-llama/Llama-3.1-8B-Instruct")


def test_sae_only_methods_raise_on_vectors_source():
    steering = _vectors_instance()
    with pytest.raises(TypeError, match="SparseAutoencoder"):
        steering.find_steering_features("Paris", torch.zeros(2, 8))
    with pytest.raises(TypeError, match="SparseAutoencoder"):
        steering.build_token_feature_map(tokens=[1, 2])
    with pytest.raises(TypeError, match="SparseAutoencoder"):
        steering.get_steering_direction(42)


def test_hook_additive_3d():
    hook = make_steering_hook([(1.5, _unit_vector(8, seed=5))], clamp=False)
    x = torch.randn(2, 3, 8)
    out = hook(None, None, x)
    expected = x + 1.5 * _unit_vector(8, seed=5)
    assert torch.allclose(out, expected)


def test_hook_clamp_sets_exact_projection():
    v = _unit_vector(8, seed=6)
    hook = make_steering_hook([(1.5, v)], clamp=True)
    x = torch.randn(2, 3, 8)
    out = hook(None, None, x)
    proj = torch.einsum("bsh,h->bs", out, v)
    assert torch.allclose(proj, torch.full((2, 3), 1.5), atol=1e-5)


def test_hook_additive_preserves_existing_projection_plus_strength():
    v = _unit_vector(8, seed=7)
    hook = make_steering_hook([(1.5, v)], clamp=False)
    x = torch.randn(2, 3, 8)
    out = hook(None, None, x)
    old_proj = torch.einsum("bsh,h->bs", x, v)
    new_proj = torch.einsum("bsh,h->bs", out, v)
    assert torch.allclose(new_proj, old_proj + 1.5, atol=1e-5)


def test_hook_handles_2d_and_tuple_output():
    hook = make_steering_hook([(1.0, _unit_vector(8, seed=8))], clamp=False)
    x2d = torch.randn(2, 8)
    out2d = hook(None, None, x2d)
    assert out2d.shape == (2, 8)

    extra = torch.randn(2, 3, 4)
    out_tuple = hook(None, None, (x2d, extra))
    assert isinstance(out_tuple, tuple) and len(out_tuple) == 2
    assert torch.allclose(out_tuple[1], extra)


def test_repetition_penalty_suppresses_repeated_token():
    # Token 0 starts as argmax; a heavy penalty must push it out.
    # ponytail: brief's values [[2.0, 1.0, 0.5, 0.1]] can never pass — a
    # positive logit divided by any penalty stays ~0, leaving ~15% softmax
    # mass. Raised the other logits so suppression is deterministic.
    logits = torch.tensor([[20.0, 15.0, 12.0, 10.0]])
    generated = torch.tensor([[0]])
    samples = torch.tensor(
        [
            _sample_next_token(
                logits,
                1.0,
                1.0,
                repetition_penalty=1000.0,
                generated_tokens=generated,
            ).item()
            for _ in range(200)
        ]
    )
    assert (samples == 0).sum().item() == 0


def test_repetition_penalty_negative_logits():
    # All logits negative; penalizing token 0 makes it even less likely.
    logits = torch.tensor([[-2.0, -1.0, -0.5, -0.1]])
    generated = torch.tensor([[0]])
    samples = torch.tensor(
        [
            _sample_next_token(
                logits,
                1.0,
                1.0,
                repetition_penalty=1000.0,
                generated_tokens=generated,
            ).item()
            for _ in range(200)
        ]
    )
    assert (samples == 0).sum().item() == 0


def test_no_penalty_leaves_distribution_untouched():
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    a = _sample_next_token(logits, 1.0, 1.0, repetition_penalty=1.0, generator=gen_a)
    b = _sample_next_token(logits, 1.0, 1.0, generator=gen_b)
    assert a.item() == b.item()


def test_seed_reproducible_sampling():
    logits = torch.randn(1, 64)
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    a = _sample_next_token(logits, 0.8, 0.9, generator=gen_a)
    b = _sample_next_token(logits, 0.8, 0.9, generator=gen_b)
    assert a.item() == b.item()


class _StubTokenizer:
    pad_token = None
    eos_token_id = 0
    model_max_length = 1000
    all_special_ids = []
    vocab_size = 100

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def apply_chat_template(
        self, messages, return_tensors="pt", add_generation_prompt=True
    ):
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # transformers>=5 returns a BatchEncoding (mapping), not a tensor
        return {"input_ids": torch.tensor([[9, 9, 9]])}


def test_build_prompt_ids_plain_and_chat():
    from unittest.mock import MagicMock

    steering = SAESteering.__new__(SAESteering)
    steering.tokenizer = _StubTokenizer()
    steering.model = MagicMock()
    steering.model.device = torch.device("cpu")

    ids, mask = steering._build_prompt_ids("hello", system_prompt=None)
    assert ids.tolist() == [[1, 2, 3]]

    ids, _ = steering._build_prompt_ids(
        "hello", system_prompt="You are a helpful assistant."
    )
    assert ids.tolist() == [[9, 9, 9]]
