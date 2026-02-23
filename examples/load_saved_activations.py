"""
Example: Load saved activations and train/analyze SAE.

This script demonstrates how to:
1. Load previously saved activations
2. Train an SAE on the loaded data
3. Analyze the learned features
"""

import torch
from loguru import logger

from drrik import ActivationExtractor, SparseAutoencoder, FeatureVisualizer


def main():
    """Load saved activations and train an SAE."""

    logger.info("=" * 60)
    logger.info("Loading Saved Activations Example")
    logger.info("=" * 60)

    # ========== Load Saved Activations ==========
    activations_path = "activations.pkl"  # Change to your path

    try:
        extractor = ActivationExtractor()  # Config doesn't matter for loading
        activations, metadata = extractor.load_activations(activations_path)

        logger.info(f"Loaded activations shape: {activations.shape}")
        logger.info(f"Loaded metadata keys: {metadata.keys()}")

    except FileNotFoundError:
        logger.error(f"Could not find activations at {activations_path}")
        logger.info("Run basic_usage.py first to generate activations, or update the path.")
        return

    # ========== Train SAE on Loaded Data ==========
    logger.info("\nTraining SAE on loaded activations...")

    sae = SparseAutoencoder(
        activation_dim=activations.shape[-1],
        hidden_dim=activations.shape[-1] * 8,
        l1_coefficient=0.01,
    )

    sae.fit(
        activations,
        batch_size=128,
        num_epochs=50,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    # Save the trained SAE
    sae.save("sae_from_saved_activations.pt")
    logger.info("Saved trained SAE")

    # ========== Analyze Features ==========
    logger.info("\nAnalyzing learned features...")

    visualizer = FeatureVisualizer(
        sae=sae,
        activations=activations,
        metadata=metadata,
        output_dir="./visualizations_from_saved",
    )

    visualizer.save_all(n_features=10)

    # Show some statistics
    densities = sae.get_feature_density(activations)

    logger.info("\nFeature Statistics:")
    logger.info(f"  Total features: {len(densities)}")
    logger.info(f"  Dead features: {(densities == 0).sum()}")
    logger.info(f"  Active features: {(densities > 0).sum()}")
    logger.info(f"  Median density (active): {densities[densities > 0].median():.2e}")

    # Show top activating example for most active feature
    top_feature = densities.argmax()
    top_values, top_indices = sae.get_top_activating_examples(
        activations, top_feature, k=3
    )

    logger.info(f"\nMost active feature: {top_feature}")
    logger.info(f"  Density: {densities[top_feature]:.2e}")

    if "samples_metadata" in metadata:
        logger.info("\nTop activating examples:")
        for i, (val, idx) in enumerate(zip(top_values, top_indices)):
            if idx < len(metadata["samples_metadata"]):
                text = metadata["samples_metadata"][idx]["text"][:150]
                logger.info(f"  {i+1}. ({val:.4f}): \"{text}...\"")

    logger.info("\nDone! Check ./visualizations_from_saved/ for plots.")


if __name__ == "__main__":
    main()
