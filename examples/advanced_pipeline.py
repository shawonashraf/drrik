"""
Advanced pipeline example demonstrating custom configurations.

This script shows how to:
1. Use custom configurations
2. Extract from multiple MLP layers
3. Use different expansion factors
4. Experiment with different datasets
"""

import torch
from loguru import logger

from drrik import ActivationExtractor, SparseAutoencoder, FeatureVisualizer
from drrik.config import (
    Config,
    ActivationExtractorConfig,
    ModelConfig,
    DatasetConfig,
    SparseAutoencoderConfig,
)


def create_custom_pipeline():
    """Create a pipeline with custom configurations."""

    logger.info("=" * 60)
    logger.info("Drrik Framework - Advanced Pipeline Example")
    logger.info("=" * 60)

    # ========== Custom Configuration ==========

    # Option 1: Use individual config classes
    model_config = ModelConfig(
        model_name="google/gemma-2b",
        torch_dtype="float16",
        device_map="auto",
    )

    dataset_config = DatasetConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="train",
        max_samples=2000,
        max_length=256,
    )

    extractor_config = ActivationExtractorConfig(
        model=model_config,
        dataset=dataset_config,
        mlp_layers=[0, 1, 2],  # Extract from multiple layers
        batch_size=16,
    )

    # Option 2: Use keyword argument overrides
    extractor = ActivationExtractor(
        config=extractor_config,
    )

    # ========== Extract Activations ==========
    logger.info("\nExtracting activations from multiple layers...")

    activations, metadata = extractor.extract()

    logger.info(f"Extracted activations shape: {activations.shape}")
    logger.info(f"Layers extracted: {metadata['layer_paths']}")

    # ========== Experiment with Different SAE Configurations ==========

    configs_to_try = [
        {
            "name": "2x_expansion",
            "hidden_dim": activations.shape[-1] * 2,
            "l1_coefficient": 0.01,
        },
        {
            "name": "4x_expansion",
            "hidden_dim": activations.shape[-1] * 4,
            "l1_coefficient": 0.01,
        },
        {
            "name": "8x_expansion_high_sparsity",
            "hidden_dim": activations.shape[-1] * 8,
            "l1_coefficient": 0.05,  # Higher sparsity
        },
    ]

    results = {}

    for config_dict in configs_to_try:
        logger.info(f"\nTraining SAE: {config_dict['name']}")

        sae = SparseAutoencoder(
            activation_dim=activations.shape[-1],
            hidden_dim=config_dict["hidden_dim"],
            l1_coefficient=config_dict["l1_coefficient"],
        )

        sae.fit(
            activations,
            batch_size=128,
            num_epochs=30,
            learning_rate=1e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
        )

        # Save model
        save_path = f"sae_{config_dict['name']}.pt"
        sae.save(save_path)

        # Compute metrics
        densities = sae.get_feature_density(activations)
        n_dead = (densities == 0).sum()
        n_active = (densities > 0).sum()

        results[config_dict["name"]] = {
            "sae": sae,
            "dead_features": n_dead,
            "active_features": n_active,
            "final_loss": sae.training_losses[-1],
            "final_l0": sae.training_l0_norms[-1],
        }

        logger.info(f"Results for {config_dict['name']}:")
        logger.info(f"  Dead features: {n_dead}/{len(densities)}")
        logger.info(f"  Active features: {n_active}/{len(densities)}")
        logger.info(f"  Final loss: {sae.training_losses[-1]:.6f}")
        logger.info(f"  Final L0: {sae.training_l0_norms[-1]:.2f}")

    # ========== Compare Results ==========
    logger.info("\n" + "=" * 60)
    logger.info("Comparison of Different SAE Configurations:")
    logger.info("=" * 60)

    for name, metrics in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Hidden dim: {metrics['sae'].hidden_dim}")
        logger.info(f"  L1 coefficient: {metrics['sae'].l1_coefficient}")
        logger.info(f"  Active features: {metrics['active_features']}")
        logger.info(f"  Final loss: {metrics['final_loss']:.6f}")
        logger.info(f"  Final L0: {metrics['final_l0']:.2f}")

    # ========== Visualize Best Model ==========
    best_model = min(results.items(), key=lambda x: x[1]["final_loss"])
    best_name, best_metrics = best_model

    logger.info(f"\nCreating visualizations for best model: {best_name}")

    visualizer = FeatureVisualizer(
        sae=best_metrics["sae"],
        activations=activations,
        metadata=metadata,
        output_dir=f"./visualizations_{best_name}",
    )

    visualizer.save_all(n_features=15)

    logger.info(f"\nVisualizations saved to ./visualizations_{best_name}/")

    return results


def explore_different_datasets():
    """Explore SAE training on different datasets."""

    logger.info("=" * 60)
    logger.info("Exploring Different Datasets")
    logger.info("=" * 60)

    datasets_to_try = [
        {"name": "wikitext", "config": "wikitext-2-raw-v1"},
        {"name": "dataset_name", "config": "standard"},  # Placeholder
    ]

    for dataset_info in datasets_to_try[:1]:  # Only use wikitext for now
        logger.info(f"\nProcessing dataset: {dataset_info['name']}")

        extractor = ActivationExtractor(
            model_name="google/gemma-2b",
            dataset_name=dataset_info["name"],
            dataset_config=dataset_info["config"],
            split="train",
            mlp_layers=[0],
            num_samples=500,
            batch_size=8,
        )

        activations, metadata = extractor.extract()

        # Train a simple SAE
        sae = SparseAutoencoder(
            activation_dim=activations.shape[-1],
            hidden_dim=activations.shape[-1] * 4,
            l1_coefficient=0.01,
        )

        sae.fit(
            activations,
            batch_size=64,
            num_epochs=20,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True,
        )

        logger.info(f"Completed {dataset_info['name']}")


if __name__ == "__main__":
    # Run the custom pipeline
    results = create_custom_pipeline()

    # Uncomment to explore different datasets
    # explore_different_datasets()
