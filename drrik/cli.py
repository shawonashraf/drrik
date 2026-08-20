"""
Command-line interface for the Drrik framework.

This module provides a Click-based CLI (``drrik`` command) for running
the complete activation extraction, SAE training, and visualization
pipeline from the command line.

Commands:
    ``drrik init-config``
        Generate an example YAML configuration file with documented options.
    ``drrik extract``
        Load a model and dataset, run inference, and save MLP activations.
    ``drrik train``
        Train a sparse autoencoder on previously extracted activations.
    ``drrik visualize``
        Generate feature analysis plots from a trained SAE.
    ``drrik run``
        Execute the full pipeline (extract -> train -> visualize) using
        a single YAML configuration file.
    ``drrik extract-vectors``
        Extract steering vectors from pre-trained SAEs and optionally
        publish them to the HF Hub.
    ``drrik steer``
        Run baseline vs steered generation from a steering config YAML.

All commands accept a YAML config file via the ``--config`` flag and
support optional Weights & Biases logging via ``--wandb``/``--no-wandb``.
"""

from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
import torch
import yaml
from loguru import logger

from drrik import (
    ActivationExtractor,
    SparseAutoencoder,
    FeatureVisualizer,
    WandbConfig,
    get_settings,
)
from drrik.steering import SAESteering, SteeringComponent, SteeringVectors


def extract_decoder_columns(
    state_dict: dict, activation_dim: int, feature_indices: List[int]
) -> List[torch.Tensor]:
    """Extract L2-normalized decoder columns from an SAE state dict.

    The decoder weight is located by shape: ``(activation_dim, D)`` with
    ``D > activation_dim`` (the encoder is the transpose-shaped matrix and
    does not match).

    Args:
        state_dict: Loaded ``ae.pt`` state dict (any key naming).
        activation_dim: Model hidden size (e.g. 4096 for Llama 3.1 8B).
        feature_indices: Feature columns to extract.

    Returns:
        List of unit-norm tensors of shape ``(activation_dim,)``.

    Raises:
        ValueError: If no decoder-shaped tensor is found.
    """
    decoder = None
    for tensor in state_dict.values():
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dim() == 2
            and tensor.shape[0] == activation_dim
            and tensor.shape[1] > activation_dim
        ):
            decoder = tensor
            break
    if decoder is None:
        raise ValueError(
            f"no decoder weight of shape ({activation_dim}, >{activation_dim}) "
            f"found in state dict keys {list(state_dict)}"
        )
    columns = []
    for fid in feature_indices:
        col = decoder[:, fid].clone().float()
        columns.append(col / col.norm())
    return columns


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
    """Extract MLP activations from a language model.

    Loads a HuggingFace model and dataset, runs inference with nnsight,
    and saves the resulting MLP layer activations as ``.npy`` files with
    accompanying ``.pkl`` metadata. Optionally logs extraction metrics
    and artifacts to Weights & Biases.

    The YAML config file must include ``model_name``, ``dataset_name``,
    ``mlp_layers``, and ``num_samples`` fields.
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
        output_dir = Path("./drrik_output/activations")
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
        batch_size=cfg.get("extraction_batch_size", 8),
    )

    # Extract activations
    activations, metadata = extractor.extract()

    # Save activations as numpy files
    activations_path = output_dir / "activations.npy"
    metadata_path = output_dir / "metadata.pkl"

    np.save(str(activations_path), activations)

    import pickle

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
        wandb_config.log_artifact(
            str(output_dir),
            f"activations-{wandb_config.get_run_id()}",
            artifact_type="activations",
        )
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
    help="Path to activations .npy file",
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
    """Train a sparse autoencoder on extracted activations.

    Loads previously extracted MLP activations, constructs a sparse
    autoencoder with the specified architecture, and trains it with
    L1-regularized reconstruction loss. The trained model is saved as
    a ``.pt`` file.

    The YAML config must include ``activation_dim``, ``hidden_dim``,
    ``l1_coefficient``, and ``num_epochs`` fields. Optional fields
    control batch size, learning rate, dead neuron resampling, and
    decoder normalization.
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
    activations_path = (
        Path(activations)
        if activations
        else Path(
            cfg.get("activations_path", "./drrik_output/activations/activations.npy")
        )
    )

    from drrik.models import ActivationExtractor

    extractor = ActivationExtractor()
    activations, metadata = extractor.load_activations(activations_path)

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

    # Train
    logger.info("Starting training...")
    sae.fit(
        activations,
        batch_size=cfg.get("training_batch_size", 256),
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
        output_dir = Path("./drrik_output/models")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "sae_model.pt"
    sae.save(model_path)
    logger.info(f"Saved model to {model_path}")

    # Log to wandb
    if wandb_config:
        wandb_config.log_artifact(
            str(output_dir),
            f"sae-model-{wandb_config.get_run_id()}",
            artifact_type="model",
        )
        wandb_config.log_artifact(
            str(activations_path.parent),
            f"activations-{wandb_config.get_run_id()}",
            artifact_type="activations",
        )
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
    help="Path to activations .npy file",
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
    """Visualize sparse autoencoder features.

    Loads a trained SAE model and its corresponding activations, then
    generates a comprehensive set of plots: feature density histograms,
    training curves, top feature rankings, activation histograms, and
    per-feature dashboards.

    The ``--n-features`` flag controls how many top features receive
    detailed dashboards. All plots are saved to ``--output-dir`` and
    optionally logged to Weights & Biases.
    """
    # Load YAML configuration
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Drrik CLI - Feature Visualization")
    logger.info("=" * 60)

    # Load activations and model
    from drrik.models import ActivationExtractor

    activations_path = (
        Path(activations)
        if activations
        else Path(
            cfg.get("activations_path", "./drrik_output/activations/activations.npy")
        )
    )

    extractor = ActivationExtractor()
    activations, metadata = extractor.load_activations(activations_path)

    if model is None:
        model = Path(cfg.get("sae_model_path", "./drrik_output/models/sae_model.pt"))

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
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="config.yml",
    help="Path to YAML configuration file",
)
@click.option(
    "--wandb/--no-wandb",
    default=None,
    help="Enable/disable wandb logging (overrides config)",
)
def run(config: Path, wandb: Optional[bool]):
    """Run the full pipeline: extract -> train -> visualize.

    Executes all three stages of the Drrik pipeline using a single
    YAML configuration file. The ``--wandb`` flag on the command line
    overrides the ``wandb_enabled`` setting in the config file.

    Outputs are organized under the configured ``output_dir``:
    - ``activations/``: Raw MLP activations and metadata
    - ``models/``: Trained SAE model checkpoint
    - ``visualizations/``: Feature analysis plots
    """
    logger.info("=" * 60)
    logger.info("Drrik CLI - Full Pipeline")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")

    # Load YAML configuration
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    # Get shared settings
    output_dir = Path(cfg.get("output_dir", "./drrik_output"))
    training_device = cfg.get("training_device", None)

    # CLI flag overrides config file
    if wandb is None:
        wandb = cfg.get("wandb_enabled", True)

    # Single shared wandb config for all stages
    wandb_config = None
    if wandb:
        wandb_config = WandbConfig(
            project=cfg.get("wandb_project", "drrik-experiments"),
            name=cfg.get("wandb_run_name"),
            config=cfg,
            enabled=get_settings().use_wandb,
        )
        wandb_config.initialize()

    # ----------------------------------------------------------------
    # Step 1: Extract activations
    # ----------------------------------------------------------------
    logger.info("Step 1/3: Extracting activations...")

    extractor = ActivationExtractor(
        model_name=cfg["model_name"],
        dataset_name=cfg["dataset_name"],
        dataset_config=cfg.get("dataset_config"),
        split=cfg.get("split", "train"),
        mlp_layers=cfg["mlp_layers"],
        num_samples=cfg["num_samples"],
        batch_size=cfg.get("extraction_batch_size", 8),
    )

    activations, metadata = extractor.extract()

    activations_dir = output_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    activations_path = activations_dir / "activations.npy"
    metadata_path = activations_dir / "metadata.pkl"

    np.save(str(activations_path), activations)

    import pickle

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Saved activations to {activations_path}")

    if wandb_config:
        wandb_config.log_metrics(
            {
                "n_samples": len(activations),
                "activation_dim": activations.shape[-1],
                "stage": "extract",
            }
        )
        wandb_config.log_artifact(
            str(activations_dir),
            f"activations-{wandb_config.get_run_id()}",
            artifact_type="activations",
        )

    # ----------------------------------------------------------------
    # Step 2: Train SAE
    # ----------------------------------------------------------------
    logger.info("Step 2/3: Training sparse autoencoder...")

    sae = SparseAutoencoder(
        activation_dim=cfg["activation_dim"],
        hidden_dim=cfg["hidden_dim"],
        l1_coefficient=cfg["l1_coefficient"],
        normalize_decoder=cfg.get("normalize_decoder", True),
        pre_encoder_bias=cfg.get("pre_encoder_bias", True),
    )

    sae.fit(
        activations,
        batch_size=cfg.get("training_batch_size", 256),
        num_epochs=cfg["num_epochs"],
        learning_rate=cfg.get("learning_rate", 1e-4),
        validation_split=cfg.get("validation_split", 0.1),
        resample_dead_neurons=cfg.get("resample_dead_neurons", True),
        resample_interval=cfg.get("resample_interval", 10000),
        device=training_device,
        wandb_config=wandb_config,
        wandb_enabled=wandb and wandb_config is not None,
        verbose=True,
    )

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    sae.save(models_dir / "sae_model.pt")
    logger.info(f"Saved model to {models_dir / 'sae_model.pt'}")

    if wandb_config:
        wandb_config.log_artifact(
            str(models_dir),
            f"sae-model-{wandb_config.get_run_id()}",
            artifact_type="model",
        )
        wandb_config.log_artifact(
            str(activations_dir),
            f"activations-{wandb_config.get_run_id()}",
            artifact_type="activations",
        )

    # ----------------------------------------------------------------
    # Step 3: Visualize
    # ----------------------------------------------------------------
    logger.info("Step 3/3: Generating visualizations...")

    sae = SparseAutoencoder.load(models_dir / "sae_model.pt")

    visualizer = FeatureVisualizer(
        sae=sae,
        activations=activations,
        metadata=metadata,
        output_dir=output_dir / "visualizations",
        wandb_config=wandb_config,
        log_to_wandb=wandb,
    )
    visualizer.save_all(n_features=cfg.get("n_features_to_visualize", 10))

    # Finalize wandb once
    if wandb_config:
        wandb_config.finalize()

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
# num_samples: 1000                # Number of samples to process
# max_length: 512                  # Maximum sequence length
# text_column: "text"               # Name of text column in dataset
# extraction_batch_size: 8              # Batch size for inference

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
# training_batch_size: 256              # Training batch size
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
# extraction_device: "cpu"            # Device for activation extraction: auto, cuda, cpu, mps
# training_device: "mps"              # Device for SAE training: auto, cuda, cpu, mps

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
    logger.info(f"  drrik run -c {output}")


@cli.command("extract-vectors")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Recipe YAML with features, sae_repo, trainer, activation_dim, concept",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(),
    default=None,
    help="Local output .pt path (default: drrik_output/vectors/<concept>.pt)",
)
@click.option(
    "--repo-id",
    default=None,
    help="HF dataset repo id to push, e.g. shawon/llama-3.1-8b-instruct_eiffel_tower",
)
def extract_vectors(config: Path, out: Optional[Path], repo_id: Optional[str]):
    """
    Extract steering vectors from pre-trained SAEs and optionally publish them.

    Downloads one ~4GB ae.pt per unique layer from the SAE repo, extracts the
    requested decoder columns (unit-norm, strength = reduced x layer), saves a
    small .pt file, and optionally pushes a parquet dataset to the HF Hub.
    """
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    with open(config) as f:
        recipe = yaml.safe_load(f)

    sae_repo = recipe["sae_repo"]
    trainer = recipe.get("trainer", "trainer_1")
    activation_dim = int(recipe["activation_dim"])
    concept = recipe["concept"]
    features = [tuple(row) for row in recipe["features"]]

    by_layer: Dict[int, List[tuple]] = {}
    for layer, fid, reduced in features:
        by_layer.setdefault(int(layer), []).append((int(fid), float(reduced)))

    components = []
    for layer in sorted(by_layer):
        logger.info(f"Downloading SAE for layer {layer} from {sae_repo} ...")
        ae_path = hf_hub_download(sae_repo, f"resid_post_layer_{layer}/{trainer}/ae.pt")
        state = torch.load(ae_path, map_location="cpu", weights_only=True)
        fids = [fid for fid, _ in by_layer[layer]]
        columns = extract_decoder_columns(state, activation_dim, fids)
        for (fid, reduced), col in zip(by_layer[layer], columns):
            components.append(SteeringComponent(layer, fid, reduced * layer, col))
        del state  # free the ~4GB before the next layer

    vectors = SteeringVectors(components, hook_path="layer")

    out_path = Path(out) if out else Path("drrik_output/vectors") / f"{concept}.pt"
    vectors.save(out_path)
    logger.info(f"Saved {len(components)} vectors to {out_path}")

    if repo_id:
        token = get_settings().huggingface_hub_token
        if not token:
            raise click.ClickException(
                "huggingface_hub_token missing from settings.yml; cannot push"
            )
        ds = Dataset.from_dict(
            {
                "layer": [c.layer for c in components],
                "feature_idx": [c.feature_idx for c in components],
                "strength": [c.strength for c in components],
                "reduced_strength": [c.strength / c.layer for c in components],
                "vector": [c.vector.tolist() for c in components],
                "concept": [concept] * len(components),
            }
        )
        ds.push_to_hub(repo_id=repo_id, token=token)
        logger.info(f"Pushed dataset to hf.co/datasets/{repo_id}")


def load_steering_config(path: Path) -> dict:
    """Load and validate a steering config YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dict.

    Raises:
        ValueError: If required keys (model, vectors, prompts) are missing.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in ("model", "vectors", "prompts") if k not in cfg]
    if missing:
        raise ValueError(f"steering config missing required keys: {missing}")
    return cfg


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Steering config YAML (model, vectors, prompts, generation)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="drrik_output/steering",
    help="Where to save the steering analysis text file",
)
def steer(config: Path, output_dir: Path):
    """
    Run baseline vs steered generation from a steering config.

    Loads steering vectors (HF dataset repo id or local .pt), generates each
    prompt at every strength scale (0.0 = baseline), prints results, and
    saves a timestamped analysis file.
    """
    cfg = load_steering_config(config)
    gen = cfg.get("generation", {})
    scales = gen.get("strength_scales", [0.0, 1.0])

    vectors_src = cfg["vectors"]
    if Path(vectors_src).exists():
        vectors = SteeringVectors.from_file(vectors_src)
    else:
        vectors = SteeringVectors.from_hf_dataset(vectors_src)

    steering = SAESteering(
        source=vectors, model_name=cfg["model"], **cfg.get("model_kwargs", {})
    )

    results = {}
    for prompt in cfg["prompts"]:
        for scale in scales:
            label = "baseline" if scale == 0.0 else f"strength_{scale}"
            logger.info(f"Generating: {prompt!r} [{label}]")
            text = steering.generate(
                prompt,
                strength_scale=scale,
                max_new_tokens=gen.get("max_new_tokens", 128),
                temperature=gen.get("temperature", 0.5),
                top_p=gen.get("top_p", 0.9),
                repetition_penalty=gen.get("repetition_penalty", 1.2),
                system_prompt=gen.get("system_prompt"),
                seed=gen.get("seed"),
            )
            key = f"{prompt} | {label}"
            results[key] = text
            print(f"\n--- {key} ---\n{text}\n")

    steering.save_steering_analysis(
        results, output_dir, prompt="; ".join(cfg["prompts"]), feature_label="steer"
    )


def main():
    """Entry point for the ``drrik`` CLI.

    Called by the ``drrik`` console script defined in ``pyproject.toml``.
    Initializes the Click command group and processes command-line
    arguments.
    """
    cli(obj={})


if __name__ == "__main__":
    main()
