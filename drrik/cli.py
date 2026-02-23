"""
Command-line interface for the Drrik framework.

This module provides a CLI for:
- Loading models and datasets
- Extracting MLP activations
- Training sparse autoencoders
- Visualizing features
- All configurable via YAML config file
"""

from pathlib import Path
from typing import Optional

import click
import yaml
from loguru import logger

from drrik import (
    ActivationExtractor,
    SparseAutoencoder,
    FeatureVisualizer,
    WandbConfig,
    get_settings,
)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Drrik Framework - Extract interpretable features from language models."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for activations and models",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default=None,
    help="Device to use (cuda/cpu/auto)",
)
@click.option(
    "--wandb/--no-wandb",
    default=True,
    show_default=True,
    help="Enable wandb logging",
)
def extract(
    config: Path,
    output_dir: Optional[Path],
    device: Optional[str],
    wandb: bool,
):
    """
    Extract MLP activations from a language model.

    Loads a model and dataset, runs inference, and saves the MLP activations.
    """
    # Load YAML configuration
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup wandb if enabled
    wandb_config = None
    if wandb:
        wandb_config = WandbConfig(
            project=cfg.get("wandb_project", "drrik-extraction"),
            name=cfg.get("wandb_run_name"),
            config=cfg,
            enabled=get_settings().use_wandb,
        )
        wandb_config.initialize()

    logger.info("=" * 60)
    logger.info("Drrik CLI - Activation Extraction")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")
    logger.info(f"Model: {cfg['model_name']}")
    logger.info(f"Dataset: {cfg['dataset_name']}")
    logger.info(f"MLP Layers: {cfg['mlp_layers']}")
    logger.info(f"Samples: {cfg['num_samples']}")

    # Create output directory
    if output_dir is None:
        output_dir = Path("./activations_output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create extractor
    extractor = ActivationExtractor(
        model_name=cfg["model_name"],
        dataset_name=cfg["dataset_name"],
        dataset_config=cfg.get("dataset_config"),
        split=cfg.get("split", "train"),
        mlp_layers=cfg["mlp_layers"],
        num_samples=cfg["num_samples"],
        batch_size=cfg.get("batch_size", 8),
    )

    # Extract activations
    activations, metadata = extractor.extract()

    # Save activations
    import pickle

    activations_path = output_dir / "activations.pkl"
    metadata_path = output_dir / "metadata.pkl"

    with open(activations_path, "wb") as f:
        pickle.dump({"activations": activations, "metadata": metadata}, f)

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Saved activations to {activations_path}")
    logger.info(f"Saved metadata to {metadata_path}")

    # Log to wandb
    if wandb_config:
        wandb_config.log_metrics(
            {
                "n_samples": len(activations),
                "activation_dim": activations.shape[-1],
            }
        )
        wandb_config.log_model(str(activations_path), "activations")
        wandb_config.finalize()

    logger.info("Extraction complete!")


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--activations",
    "-a",
    type=click.Path(exists=True),
    default=None,
    help="Path to activations .pkl file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for models and visualizations",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default=None,
    help="Device to use (cuda/cpu/auto)",
)
@click.option(
    "--wandb/--no-wandb",
    default=True,
    show_default=True,
    help="Enable wandb logging",
)
def train(
    config: Path,
    activations: Optional[Path],
    output_dir: Optional[Path],
    device: Optional[str],
    wandb: bool,
):
    """
    Train a sparse autoencoder on extracted activations.

    Loads activations and trains a sparse autoencoder to extract interpretable features.
    """
    # Load YAML configuration
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Drrik CLI - SAE Training")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")
    logger.info(f"Activation Dim: {cfg['activation_dim']}")
    logger.info(f"Hidden Dim: {cfg['hidden_dim']}")
    logger.info(f"Expansion Factor: {cfg['hidden_dim'] / cfg['activation_dim']}")

    # Load activations
    if activations is None:
        activations = Path(
            cfg.get("activations_path", "./activations_output/activations.pkl")
        )

    from drrik.models import ActivationExtractor

    extractor = ActivationExtractor()  # Config doesn't matter for loading
    activations_data, metadata = extractor.load_activations(activations)
    activations = activations_data["activations"]

    # Setup wandb
    wandb_config = None
    if wandb:
        wandb_config = WandbConfig(
            project=cfg.get("wandb_project", "drrik-sae-training"),
            name=cfg.get("wandb_run_name"),
            config=cfg,
            enabled=get_settings().use_wandb,
        )
        wandb_config.initialize()

    # Create SAE
    sae = SparseAutoencoder(
        activation_dim=cfg["activation_dim"],
        hidden_dim=cfg["hidden_dim"],
        l1_coefficient=cfg["l1_coefficient"],
        normalize_decoder=cfg.get("normalize_decoder", True),
        pre_encoder_bias=cfg.get("pre_encoder_bias", True),
    )

    # Determine device
    if device is None:
        device = cfg.get("device", "auto")

    # Train
    logger.info("Starting training...")
    sae.fit(
        activations,
        batch_size=cfg.get("batch_size", 256),
        num_epochs=cfg["num_epochs"],
        learning_rate=cfg.get("learning_rate", 1e-4),
        validation_split=cfg.get("validation_split", 0.1),
        resample_dead_neurons=cfg.get("resample_dead_neurons", True),
        resample_interval=cfg.get("resample_interval", 10000),
        device=device,
        wandb_config=wandb_config,
        wandb_enabled=wandb,
        verbose=True,
    )

    # Save model
    if output_dir is None:
        output_dir = Path("./sae_output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "sae_model.pt"
    sae.save(model_path)
    logger.info(f"Saved model to {model_path}")

    # Log to wandb
    if wandb_config:
        wandb_config.log_model(str(model_path), "sae_model")
        wandb_config.finalize()

    logger.info("Training complete!")


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--activations",
    "-a",
    type=click.Path(exists=True),
    default=None,
    help="Path to activations .pkl file",
)
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    default=None,
    help="Path to SAE model .pt file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./visualizations",
    help="Output directory for visualizations",
)
@click.option(
    "--n-features",
    "-n",
    type=int,
    default=10,
    help="Number of top features to visualize",
)
@click.option(
    "--wandb/--no-wandb",
    default=True,
    show_default=True,
    help="Enable wandb logging",
)
def visualize(
    config: Path,
    activations: Optional[Path],
    model: Optional[Path],
    output_dir: Path,
    n_features: int,
    wandb: bool,
):
    """
    Visualize sparse autoencoder features.

    Generates plots for feature densities, top activating examples, and more.
    """
    # Load YAML configuration
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Drrik CLI - Feature Visualization")
    logger.info("=" * 60)

    # Load activations and model
    from drrik.models import ActivationExtractor

    if activations is None:
        activations = Path(
            cfg.get("activations_path", "./activations_output/activations.pkl")
        )

    extractor = ActivationExtractor()
    activations_data, metadata = extractor.load_activations(activations)
    activations = activations_data["activations"]

    if model is None:
        model = Path(cfg.get("sae_model_path", "./sae_output/sae_model.pt"))

    sae = SparseAutoencoder.load(model)

    logger.info(f"Activations shape: {activations.shape}")
    logger.info(f"SAE hidden dim: {sae.hidden_dim}")

    # Setup wandb
    wandb_config = None
    if wandb:
        wandb_config = WandbConfig(
            project=cfg.get("wandb_project", "drrik-visualization"),
            name=cfg.get("wandb_run_name"),
            config=cfg,
            enabled=get_settings().use_wandb,
        )
        wandb_config.initialize()

    # Create visualizer
    visualizer = FeatureVisualizer(
        sae=sae,
        activations=activations,
        metadata=metadata,
        output_dir=output_dir,
        wandb_config=wandb_config,
        log_to_wandb=wandb,
    )

    # Generate visualizations
    visualizer.save_all(n_features=n_features)

    if wandb_config:
        wandb_config.finalize()

    logger.info(f"Visualizations saved to {output_dir}")


@cli.command()
@click.argument(
    "config",
    type=click.Path(exists=True),
    required=False,
)
def run(config: Optional[Path]):
    """
    Run the full pipeline: extract -> train -> visualize.

    Uses a single YAML config file to configure all stages.
    """
    if config is None:
        config = Path("config.yml")

    logger.info("=" * 60)
    logger.info("Drrik CLI - Full Pipeline")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")

    # Load YAML configuration
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    # Get shared settings
    output_dir = Path(cfg.get("output_dir", "./drrik_output"))
    device = cfg.get("device", None)
    wandb_enabled = cfg.get("wandb_enabled", True)

    # Run extract
    ctx = cli.make_context(
        "extract",
        [
            "--config",
            str(config),
            "--output-dir",
            str(output_dir / "activations"),
            "--device",
            device or "auto",
        ],
    )
    with ctx:
        extract(config, output_dir / "activations", device, wandb_enabled)

    # Run train
    ctx = cli.make_context(
        "train",
        [
            "--config",
            str(config),
            "--activations",
            str(output_dir / "activations" / "activations.pkl"),
            "--output-dir",
            str(output_dir / "models"),
            "--device",
            device or "auto",
        ],
    )
    with ctx:
        train(config, None, output_dir / "models", device, wandb_enabled)

    # Run visualize
    ctx = cli.make_context(
        "visualize",
        [
            "--config",
            str(config),
            "--activations",
            str(output_dir / "activations" / "activations.pkl"),
            "--model",
            str(output_dir / "models" / "sae_model.pt"),
            "--output-dir",
            str(output_dir / "visualizations"),
            "--n-features",
            cfg.get("n_features_to_visualize", 10),
        ],
    )
    with ctx:
        visualize(
            config,
            None,
            None,
            output_dir / "visualizations",
            cfg.get("n_features_to_visualize", 10),
            wandb_enabled,
        )

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("=" * 60)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config.yml",
    help="Path for example config file",
)
def init_config(output: Path):
    """
    Generate an example YAML configuration file.

    Creates a config.yml file with all available options documented.
    """
    example_config = """
# Drrik Framework Configuration
# This file controls model loading, activation extraction, SAE training, and visualization

# ============================================================================
# Model Configuration
# ============================================================================
model_name: "google/gemma-2b"  # HuggingFace model name (<3B for 8GB VRAM)
# model_revision: "main"           # Model revision
# torch_dtype: "float16"          # Data type: float16, bfloat16, float32
# device_map: "auto"               # Device mapping: auto, cpu, cuda
# trust_remote_code: true          # Trust remote code when loading

# ============================================================================
# Dataset Configuration
# ============================================================================
# dataset_name: "wikitext"         # HuggingFace dataset name
# dataset_config: "wikitext-2-raw-v1"  # Dataset configuration name
# split: "train"                    # Dataset split: train, validation, test
# max_samples: 1000                # Number of samples to process
# max_length: 512                  # Maximum sequence length
# text_column: "text"               # Name of text column in dataset
# batch_size: 8                     # Batch size for inference

# ============================================================================
# Activation Extraction Configuration
# ============================================================================
# mlp_layers: [0]                   # List of MLP layer indices to extract from
# activation_output_dir: "./activations"  # Where to save extracted activations

# ============================================================================
# Sparse Autoencoder Configuration
# ============================================================================
# activation_dim: 2048             # Input dimension (automatically detected from activations)
# hidden_dim: 16384                # Hidden dimension (e.g., 8x expansion: 2048 * 8)
# l1_coefficient: 0.01              # L1 regularization strength for sparsity
# learning_rate: 0.0001             # Learning rate for Adam optimizer
# num_epochs: 50                    # Number of training epochs
# batch_size: 256                   # Training batch size
# validation_split: 0.1              # Fraction of data for validation
# resample_dead_neurons: true        # Whether to resample dead neurons during training
# resample_interval: 10000          # Steps between resampling checks
# normalize_decoder: true            # Normalize decoder weights to unit norm
# pre_encoder_bias: true             # Use pre-encoder bias (as in the paper)

# ============================================================================
# Visualization Configuration
# ============================================================================
# n_features_to_visualize: 10        # Number of top features to visualize
# visualization_output_dir: "./visualizations"  # Where to save plots

# ============================================================================
# Wandb Configuration (Optional)
# ============================================================================
# wandb_enabled: true                # Enable wandb logging (requires WANDB_API_KEY env var)
# wandb_project: "drrik-experiments"  # wandb project name
# wandb_run_name: null              # Specific run name (auto-generated if null)
# wandb_entity: null                 # wandb entity (username or team)

# ============================================================================
# Hardware Configuration
# ============================================================================
# device: "auto"                     # Device for training: auto, cuda, cpu

# ============================================================================
# Output Configuration
# ============================================================================
# output_dir: "./drrik_output"     # Base directory for all outputs

# ============================================================================
# Example: Quick Start Configuration
# ============================================================================
# For a quick test with a small model:
# model_name: "google/gemma-2b"
# dataset_name: "wikitext"
# num_samples: 500
# hidden_dim: 4096  # 2x expansion for quick testing
# num_epochs: 20

# ============================================================================
# Example: Production Configuration
# ============================================================================
# For a serious run with more features:
# model_name: "google/gemma-2b"
# dataset_name: "wikitext"
# num_samples: 10000
# hidden_dim: 32768  # 16x expansion for more features
# l1_coefficient: 0.01
# num_epochs: 100
"""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        f.write(example_config.strip())

    logger.info(f"Example configuration written to {output}")
    logger.info("Edit the file to customize your pipeline, then run:")
    logger.info(f"  drrik run {output}")


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
