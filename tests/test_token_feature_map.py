"""Self-check for SAESteering.build_token_feature_map()."""

import torch
import torch.nn as nn
from unittest.mock import MagicMock


class _FakeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class _FakeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = _FakeMLP(dim)


class _FakeInner(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(dim) for _ in range(n_layers)])


class _FakeBaseModel(nn.Module):
    """Minimal model mimicking HF layout: model.model.layers[i].mlp."""

    def __init__(self, dim, n_layers):
        super().__init__()
        self.model = _FakeInner(dim, n_layers)

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        x = torch.randn(batch, seq_len, self.model.layers[0].mlp.linear.out_features)
        for layer in self.model.layers:
            x = layer.mlp(x)
        return x


def test_build_token_feature_map():
    """Verify build_token_feature_map returns correct structure."""
    from drrik.autoencoder import SparseAutoencoder
    from drrik.steering import SAESteering

    dim = 8
    hidden = 16
    sae = SparseAutoencoder(activation_dim=dim, hidden_dim=hidden)
    fake_model = _FakeBaseModel(dim, n_layers=6)

    steering = SAESteering.__new__(SAESteering)
    steering.sae = sae
    steering.model = MagicMock()
    steering.model._module = fake_model
    steering.tokenizer = MagicMock()
    steering.tokenizer.all_special_ids = [100, 101]
    steering.tokenizer.vocab_size = 50000
    steering.target_layer = 5
    steering.device = torch.device("cpu")

    result = steering.build_token_feature_map(tokens=[10, 20, 30])

    assert set(result.keys()) == {
        10,
        20,
        30,
    }, f"Expected {{10,20,30}}, got {set(result.keys())}"
    for tid in (10, 20, 30):
        info = result[tid]
        assert "active_features" in info
        assert "mean_activations" in info
        assert isinstance(info["active_features"], list)
        assert isinstance(info["mean_activations"], dict)
    print("PASS: build_token_feature_map returns correct structure")


if __name__ == "__main__":
    test_build_token_feature_map()
