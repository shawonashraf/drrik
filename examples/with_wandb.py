"""
Example: Using Drrik with Weights & Biases (wandb) integration.

This script demonstrates how to:
1. Use environment variables for API keys
2. Enable wandb logging for SAE training
3. Log visualizations to wandb
4. Track experiments automatically

Setup:
1. Copy .env.example to .env
2. Add your HuggingFace Hub token (for gated models)
3. Add your wandb API key
4. Run: python examples/with_wandb.py
"""

from loguru import logger

from drrik import (
    ActivationExtractor,
    SparseAutoencoder,
    FeatureVisualizer,
    get_settings,
    WandbConfig,
)


def main():
    """Run the pipeline with wandb integration."""

    # Load environment settings
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("Drrik Framework - With wandb Integration")
    logger.info("=" * 60)

    # Check settings
    logger.info("\nEnvironment Settings:")
    logger.info(f"  HF Token configured: {settings.has_hf_token}")
    logger.info(f"  wandb enabled: {settings.use_wandb}")
    logger.info(f"  wandb project: {settings.wandb_project}")
    logger.info(f"  wandb mode: {settings.wandb_mode}")

    # ========== Step 1: Create wandb config ==========
    wandb_config = WandbConfig(
        project="drrik-sae-experiments",
        entity=settings.wandb_entity,  # Uses your username if None
        name="gemma-2b-wikitext-8x-expansion",
        config={
            "model": "google/gemma-2b",
            "dataset": "wikitext",
            "mlp_layers": [0],
            "num_samples": 1000,
            "expansion_factor": 8,
            "l1_coefficient": 0.01,
        },
        tags=["gemma-2b", "wikitext", "8x", "baseline"],
        enabled=settings.use_wandb,  # Auto-disable if no API key
    )

    # ========== Step 2: Extract MLP Activations ==========
    logger.info("\n[Step 1] Extracting MLP activations...")

    extractor = ActivationExtractor(
        model_name="google/gemma-2b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="train",
        mlp_layers=[0],
        num_samples=1000,
        batch_size=8,
    )

    activations, metadata = extractor.extract()

    logger.info(f"Extracted activations shape: {activations.shape}")

    # ========== Step 3: Train SAE with wandb logging ==========
    logger.info("\n[Step 2] Training SAE with wandb logging...")

    sae = SparseAutoencoder(
        activation_dim=activations.shape[-1],
        hidden_dim=activations.shape[-1] * 8,  # 8x expansion
        l1_coefficient=0.01,
    )

    # Use wandb_config parameter - wandb will auto-initialize
    sae.fit(
        activations,
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-4,
        resample_dead_neurons=True,
        resample_interval=2000,
        wandb_config=wandb_config,  # Pass wandb config here
        wandb_enabled=True,
        verbose=True,
    )

    # Save model
    sae.save("sae_model_wandb.pt")

    # ========== Step 4: Visualize with wandb logging ==========
    logger.info("\n[Step 3] Creating visualizations with wandb logging...")

    # Initialize wandb for visualization phase
    with WandbConfig(
        project="drrik-sae-experiments",
        name="gemma-2b-wikitext-visualizations",
        config={"model": "gemma-2b", "expansion": 8},
        enabled=settings.use_wandb,
    ) as vis_wandb:
        visualizer = FeatureVisualizer(
            sae=sae,
            activations=activations,
            metadata=metadata,
            output_dir="./visualizations_wandb",
            wandb_config=vis_wandb,  # Pass wandb config for logging plots
            log_to_wandb=True,
        )

        # Generate all visualizations (will be logged to wandb)
        visualizer.save_all(n_features=10)

        logger.info("\n" + "=" * 60)
        logger.info("Done! Check wandb for logged metrics and plots.")
        if settings.use_wandb:
            logger.info(f"wandb project: {settings.wandb_project}")
        logger.info("=" * 60)


def example_custom_wandb():
    """Example showing custom wandb configuration."""

    logger.info("\n" + "=" * 60)
    logger.info("Custom wandb Configuration Example")
    logger.info("=" * 60)

    # Create custom wandb config with specific settings
    wandb_config = WandbConfig(
        project="my-custom-project",
        entity="my-team",  # Specify team/organization
        name="experiment-001",
        config={
            "description": "Testing different expansion factors",
            "architecture": "SAE-v1",
        },
        tags=["experiment", "test"],
        enabled=True,  # Force enable (requires WANDB_API_KEY env var)
    )

    # Use as context manager
    with wandb_config:
        logger.info(f"wandb initialized: {wandb_config.get_run_url()}")

        # Log some metrics
        wandb_config.log_metrics(
            {
                "test_metric_1": 1.0,
                "test_metric_2": 2.0,
            }
        )

        # Log a histogram
        import numpy as np

        data = np.random.randn(1000)
        wandb_config.log_histogram(data, "test_histogram")

        logger.info("Metrics logged to wandb")


def example_wandb_disabled():
    """Example showing how to disable wandb."""

    logger.info("\n" + "=" * 60)
    logger.info("wandb Disabled Example")
    logger.info("=" * 60)

    # Method 1: Set enabled=False in WandbConfig
    wandb_config = WandbConfig(
        enabled=False,  # Explicitly disable
    )

    # Method 2: Set wandb_enabled=False in fit()
    sae = SparseAutoencoder(
        activation_dim=2048,
        hidden_dim=4096,
    )

    # Extract some dummy activations
    import numpy as np

    activations = np.random.randn(100, 2048)

    # This won't use wandb even if API key is set
    sae.fit(
        activations,
        num_epochs=5,
        wandb_config=wandb_config,
        wandb_enabled=False,  # Disable wandb for this run
        verbose=False,
    )

    logger.info("Training completed without wandb logging")


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment to run other examples:
    # example_custom_wandb()
    # example_wandb_disabled()
