"""Tests for SAESteering steering mechanics (no model loading, CPU only)."""

import pytest
import torch

from drrik.autoencoder import SparseAutoencoder
from drrik.steering import SAESteering, SteeringComponent, SteeringVectors


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
