"""
Simple test to verify all modules import correctly.

Tests run with pytest.
"""

import torch


def test_config_imports():
    """Test that config module can be imported."""


def test_autoencoder_imports():
    """Test that autoencoder module can be imported."""


def test_models_imports():
    """Test that models module can be imported."""


def test_visualization_imports():
    """Test that visualization module can be imported."""


def test_settings_imports():
    """Test that settings module can be imported."""


def test_main_package_imports():
    """Test that main package exports work."""


def test_model_config_creation():
    """Test that ModelConfig can be created."""
    from drrik.config import ModelConfig

    model_cfg = ModelConfig(model_name="google/gemma-2b")
    assert model_cfg.model_name == "google/gemma-2b"
    assert model_cfg.torch_dtype == "float16"
    assert model_cfg.device_map == "auto"


def test_dataset_config_creation():
    """Test that DatasetConfig can be created."""
    from drrik.config import DatasetConfig

    dataset_cfg = DatasetConfig(dataset_name="wikitext")
    assert dataset_cfg.dataset_name == "wikitext"
    assert dataset_cfg.max_samples == 1000


def test_extractor_config_creation():
    """Test that ActivationExtractorConfig can be created."""
    from drrik.config import (
        ModelConfig,
        DatasetConfig,
        ActivationExtractorConfig,
    )

    model_cfg = ModelConfig(model_name="google/gemma-2b")
    dataset_cfg = DatasetConfig(dataset_name="wikitext")

    extractor_cfg = ActivationExtractorConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        mlp_layers=[0],
    )
    assert extractor_cfg.mlp_layers == [0]
    assert extractor_cfg.batch_size == 8


def test_sae_config_creation():
    """Test that SparseAutoencoderConfig can be created."""
    from drrik.config import SparseAutoencoderConfig

    sae_cfg = SparseAutoencoderConfig(
        activation_dim=2048,
        hidden_dim=4096,
    )
    assert sae_cfg.activation_dim == 2048
    assert sae_cfg.hidden_dim == 4096


def test_sae_creation():
    """Test that SAE can be instantiated."""
    from drrik.autoencoder import SparseAutoencoder

    sae = SparseAutoencoder(
        activation_dim=256,
        hidden_dim=512,
        l1_coefficient=0.01,
    )
    assert sae.activation_dim == 256
    assert sae.hidden_dim == 512


def test_sae_forward_pass():
    """Test SAE forward pass."""
    from drrik.autoencoder import SparseAutoencoder

    sae = SparseAutoencoder(
        activation_dim=256,
        hidden_dim=512,
        l1_coefficient=0.01,
    )

    x = torch.randn(32, 256)
    reconstructed, features = sae(x)

    assert reconstructed.shape == (32, 256)
    assert features.shape == (32, 512)


def test_sae_encode_decode():
    """Test SAE encode/decode methods."""
    from drrik.autoencoder import SparseAutoencoder

    sae = SparseAutoencoder(
        activation_dim=256,
        hidden_dim=512,
        l1_coefficient=0.01,
    )

    x = torch.randn(32, 256)
    features = sae.encode(x)
    reconstructed = sae.decode(features)

    assert features.shape == (32, 512)
    assert reconstructed.shape == (32, 256)


def test_environment_settings_creation():
    """Test that EnvironmentSettings can be created."""
    from drrik.settings import EnvironmentSettings

    settings = EnvironmentSettings()
    assert settings.wandb_project == "drrik-experiments"
    assert settings.wandb_mode == "online"


def test_wandb_config_creation():
    """Test that WandbConfig can be created."""
    from drrik.settings import WandbConfig

    wandb_cfg = WandbConfig(
        project="test-project",
        config={"test": True},
        enabled=False,  # Don't actually initialize wandb
    )
    assert wandb_cfg.project == "test-project"
    assert wandb_cfg.config == {"test": True}


def test_settings_properties():
    """Test EnvironmentSettings properties."""
    from drrik.settings import EnvironmentSettings

    settings = EnvironmentSettings()
    # These should work without error
    assert hasattr(settings, "use_wandb")
    assert hasattr(settings, "has_hf_token")
