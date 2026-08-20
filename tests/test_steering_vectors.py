"""Tests for SteeringVectors (no model loading, CPU only)."""

import pytest
import torch

from drrik.autoencoder import SparseAutoencoder
from drrik.steering import SteeringComponent, SteeringVectors


def _unit_vector(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def test_round_trip_preserves_components(tmp_path):
    comps = [
        SteeringComponent(11, 74457, 1.408, _unit_vector(16, seed=1)),
        SteeringComponent(15, 21576, 1.545, _unit_vector(16, seed=2)),
    ]
    vectors = SteeringVectors(comps, hook_path="layer")
    path = tmp_path / "vectors.pt"
    vectors.save(path)

    loaded = SteeringVectors.from_file(path)
    assert loaded.hook_path == "layer"
    assert len(loaded.components) == 2
    for orig, new in zip(comps, loaded.components):
        assert new.layer == orig.layer
        assert new.feature_idx == orig.feature_idx
        assert new.strength == pytest.approx(orig.strength)
        assert torch.allclose(new.vector, orig.vector)


def test_round_trip_preserves_mlp_hook_path(tmp_path):
    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    vectors = SteeringVectors.from_sae(sae, layer=3, feature_indices=[1, 5])
    path = tmp_path / "vectors_mlp.pt"
    vectors.save(path)

    loaded = SteeringVectors.from_file(path)
    assert loaded.hook_path == "mlp"


def test_from_sae_builds_unit_norm_columns():
    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    vectors = SteeringVectors.from_sae(
        sae, layer=3, feature_indices=[1, 5], strengths=[2.0, 0.5]
    )
    assert vectors.hook_path == "mlp"
    assert vectors.layers == [3]
    assert vectors.activation_dim == 8
    for comp, (fid, strength) in zip(vectors.components, [(1, 2.0), (5, 0.5)]):
        assert comp.feature_idx == fid
        assert comp.strength == strength
        assert comp.vector.norm().item() == pytest.approx(1.0, abs=1e-6)
        expected = sae.decoder.weight[:, fid]
        expected = expected / expected.norm()
        assert torch.allclose(comp.vector, expected)


def test_from_sae_defaults_strengths_to_one():
    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    vectors = SteeringVectors.from_sae(sae, layer=0, feature_indices=[2])
    assert vectors.components[0].strength == 1.0


def test_validation_errors():
    with pytest.raises(ValueError, match="hook_path"):
        SteeringVectors(
            [SteeringComponent(0, 0, 1.0, _unit_vector(4))], hook_path="bogus"
        )
    with pytest.raises(ValueError, match="component"):
        SteeringVectors([])

    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    with pytest.raises(ValueError, match="same length"):
        SteeringVectors.from_sae(sae, layer=0, feature_indices=[1, 2], strengths=[1.0])
    with pytest.raises(ValueError, match="out of range"):
        SteeringVectors.from_sae(sae, layer=0, feature_indices=[999])
