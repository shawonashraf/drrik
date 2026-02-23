"""
Simple test to verify all modules import correctly.
"""

import sys


def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")

    try:
        print("  - drrik.config...", end=" ")
        from drrik.config import (
            Config,
            ModelConfig,
            DatasetConfig,
            ActivationExtractorConfig,
            SparseAutoencoderConfig,
            VisualizationConfig,
        )
        print("OK")

        print("  - drrik.autoencoder...", end=" ")
        from drrik.autoencoder import SparseAutoencoder
        print("OK")

        print("  - drrik.models...", end=" ")
        from drrik.models import ActivationExtractor
        print("OK")

        print("  - drrik.visualization...", end=" ")
        from drrik.visualization import FeatureVisualizer
        print("OK")

        print("  - drrik __init__...", end=" ")
        from drrik import (
            ActivationExtractor,
            SparseAutoencoder,
            FeatureVisualizer,
            Config,
        )
        print("OK")

        print("\nAll imports successful!")
        return True

    except Exception as e:
        print(f"\nImport failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_creation():
    """Test that config objects can be created."""
    print("\nTesting config creation...")

    try:
        from drrik.config import (
            ModelConfig,
            DatasetConfig,
            ActivationExtractorConfig,
            SparseAutoencoderConfig,
        )

        print("  - ModelConfig...", end=" ")
        model_cfg = ModelConfig(model_name="google/gemma-2b")
        print("OK")

        print("  - DatasetConfig...", end=" ")
        dataset_cfg = DatasetConfig(dataset_name="wikitext")
        print("OK")

        print("  - ActivationExtractorConfig...", end=" ")
        extractor_cfg = ActivationExtractorConfig(
            model=model_cfg,
            dataset=dataset_cfg,
            mlp_layers=[0],
        )
        print("OK")

        print("  - SparseAutoencoderConfig...", end=" ")
        sae_cfg = SparseAutoencoderConfig(
            activation_dim=2048,
            hidden_dim=4096,
        )
        print("OK")

        print("\nAll configs created successfully!")
        return True

    except Exception as e:
        print(f"\nConfig creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sae_creation():
    """Test that SAE can be instantiated."""
    print("\nTesting SAE creation...")

    try:
        import torch
        from drrik.autoencoder import SparseAutoencoder

        print("  - Creating SAE...", end=" ")
        sae = SparseAutoencoder(
            activation_dim=256,
            hidden_dim=512,
            l1_coefficient=0.01,
        )
        print("OK")

        print("  - Testing forward pass...", end=" ")
        x = torch.randn(32, 256)
        reconstructed, features = sae(x)
        assert reconstructed.shape == (32, 256)
        assert features.shape == (32, 512)
        print("OK")

        print("  - Testing encode/decode...", end=" ")
        features = sae.encode(x)
        reconstructed = sae.decode(features)
        assert features.shape == (32, 512)
        assert reconstructed.shape == (32, 256)
        print("OK")

        print("\nSAE tests passed!")
        return True

    except Exception as e:
        print(f"\nSAE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Drrik Framework - Import Tests")
    print("=" * 50)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_config_creation()
    all_passed &= test_sae_creation()

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
