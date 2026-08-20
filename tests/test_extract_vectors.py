"""Tests for extract-vectors pure helpers (no network)."""

import pytest
import torch

from drrik.cli import extract_decoder_columns


def test_extract_decoder_columns_by_shape():
    decoder = torch.randn(8, 16)
    encoder = torch.randn(16, 8)
    state = {"decoder.weight": decoder, "encoder.weight": encoder}

    cols = extract_decoder_columns(state, activation_dim=8, feature_indices=[1, 3])
    assert len(cols) == 2
    for col, fid in zip(cols, [1, 3]):
        assert col.shape == (8,)
        assert col.norm().item() == pytest.approx(1.0, abs=1e-6)
        expected = decoder[:, fid]
        expected = expected / expected.norm()
        assert torch.allclose(col, expected)


def test_extract_decoder_columns_prefers_decoder_not_encoder():
    # Encoder is (16, 8): shape[0] != activation_dim, so it must not match.
    decoder = torch.randn(8, 16)
    state = {"enc": torch.randn(16, 8), "dec": decoder}
    cols = extract_decoder_columns(state, activation_dim=8, feature_indices=[0])
    expected = decoder[:, 0]
    expected = expected / expected.norm()
    assert torch.allclose(cols[0], expected)


def test_extract_decoder_columns_not_found():
    with pytest.raises(ValueError, match="decoder weight"):
        extract_decoder_columns(
            {"w": torch.randn(4, 4)}, activation_dim=8, feature_indices=[0]
        )


def test_load_steering_config_valid(tmp_path):
    from drrik.cli import load_steering_config

    cfg_file = tmp_path / "steering.yaml"
    cfg_file.write_text(
        "model: meta-llama/Llama-3.1-8B-Instruct\n"
        "vectors: shawon/llama-3.1-8b-instruct_eiffel_tower\n"
        'prompts: ["The weather today is"]\n'
    )
    cfg = load_steering_config(cfg_file)
    assert cfg["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg["prompts"] == ["The weather today is"]


def test_load_steering_config_missing_keys(tmp_path):
    import pytest

    from drrik.cli import load_steering_config

    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("model: x\n")
    with pytest.raises(ValueError, match="missing required keys"):
        load_steering_config(cfg_file)
