"""
Basic usage example for the Drrik framework.

This script demonstrates how to:
1. Load a small language model from HuggingFace Hub
2. Run inference on a dataset and extract MLP activations
3. Train a sparse autoencoder on the activations
4. Visualize the learned features

This is suitable for models under 3B parameters on an 8GB VRAM GPU.
"""

import torch
from loguru import logger

from drrik import ActivationExtractor, SparseAutoencoder, FeatureVisualizer
from drrik.config import Config


def main():
    """Run the complete activation extraction and SAE training pipeline."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    logger.info("=" * 60)
    logger.info("Drrik Framework - Basic Usage Example")
    logger.info("=" * 60)

    # ========== Step 1: Extract MLP Activations ==========
    logger.info("\n[Step 1] Extracting MLP activations...")

    extractor = ActivationExtractor(
        model_name="google/gemma-2b",  # 2B parameters, fits on 8GB VRAM
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="train",
        mlp_layers=[0],  # Extract from first MLP layer
        num_samples=1000,  # Use 1000 samples for quick demo
        batch_size=8,
    )

    activations, metadata = extractor.extract()

    logger.info(f"Extracted activations shape: {activations.shape}")
    logger.info(f"Activation dimension: {activations.shape[-1]}")

    # Optionally save activations for later use
    # extractor.save_activations(activations, metadata, "activations.pkl")

    # ========== Step 2: Train Sparse Autoencoder ==========
    logger.info("\n[Step 2] Training Sparse Autoencoder...")

    sae = SparseAutoencoder(
        activation_dim=activations.shape[-1],
        hidden_dim=activations.shape[-1] * 8,  # 8x expansion factor
        l1_coefficient=0.01,
        normalize_decoder=True,
        pre_encoder_bias=True,
    )

    logger.info(f"SAE architecture: {activations.shape[-1]} -> {activations.shape[-1] * 8}")

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    sae.fit(
        activations,
        batch_size=64,
        num_epochs=50,  # Increase for better results
        learning_rate=1e-4,
        resample_dead_neurons=True,
        resample_interval=2000,
        device=device,
        verbose=True,
    )

    # Save the trained model
    sae.save("sae_model.pt")

    # ========== Step 3: Visualize Learned Features ==========
    logger.info("\n[Step 3] Creating visualizations...")

    visualizer = FeatureVisualizer(
        sae=sae,
        activations=activations,
        metadata=metadata,
        output_dir="./visualizations",
    )

    # Generate all standard visualizations
    visualizer.save_all(n_features=10)

    # ========== Step 4: Analyze Specific Features ==========
    logger.info("\n[Step 4] Analyzing top features...")

    # Get feature densities
    densities = visualizer.sae.get_feature_density(activations)
    n_dead = (densities == 0).sum()
    n_active = (densities > 0).sum()

    logger.info(f"Dead features: {n_dead}/{len(densities)} ({n_dead/len(densities)*100:.1f}%)")
    logger.info(f"Active features: {n_active}/{len(densities)} ({n_active/len(densities)*100:.1f}%)")
    logger.info(f"Median density: {densities[densities > 0].median():.2e}")

    # Show top activating examples for a specific feature
    top_feature_idx = densities.argmax()
    logger.info(f"\nTop feature by density: {top_feature_idx}")

    top_values, top_indices = sae.get_top_activating_examples(
        activations, top_feature_idx, k=5
    )

    logger.info("Top 5 activating examples:")
    for i, (val, idx) in enumerate(zip(top_values, top_indices)):
        if idx < len(metadata["samples_metadata"]):
            text = metadata["samples_metadata"][idx]["text"][:100]
            logger.info(f"  {i+1}. activation={val:.4f}: \"{text}...\"")

    logger.info("\n" + "=" * 60)
    logger.info("Done! Check the ./visualizations directory for plots.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
