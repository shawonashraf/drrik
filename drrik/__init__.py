"""
Drrik: A Framework for Monosemantic Feature Extraction from Language Models

This framework is inspired by the ``Towards Monosemanticity`` paper from
Anthropic, which applies dictionary learning to extract activations as
features from transformer-based large language models and trains sparse
autoencoders to linearize those activations.

Key Components:
    - **Model loading** from HuggingFace Hub (with gated model support)
    - **Dataset loading** and inference pipeline
    - **MLP activation collection** using nnsight
    - **Sparse Autoencoder training** for feature extraction
    - **SAE-based activation steering** to bias model outputs
    - **Visualization** of feature-specific activation vectors
    - Optional **Weights & Biases** integration for experiment tracking

Public API:
    ``ActivationExtractor``
        Loads models and datasets, extracts MLP activations via nnsight.
    ``SparseAutoencoder``
        Overcomplete SAE with L1 regularization and dead neuron resampling.
    ``FeatureVisualizer``
        Generates density histograms, training curves, and feature dashboards.
    ``SAESteering``
        Steers language model generation by adding SAE feature directions
        to MLP activations during inference.
    ``Config``
        Top-level Pydantic settings model aggregating all sub-configurations.
    ``EnvironmentSettings``
        Loads API keys and environment variables from ``.env``.
    ``WandbConfig``
        Context manager for wandb experiment tracking.
    ``get_settings``
        Returns the global ``EnvironmentSettings`` singleton.

Example Usage:
    ```python
    from drrik import ActivationExtractor, SparseAutoencoder, FeatureVisualizer

    # Extract activations
    extractor = ActivationExtractor(
        model_name="google/gemma-2b",
        dataset_name="wikitext",
        dataset_split="train",
        mlp_layers=[0, 1, 2],
        num_samples=1000
    )
    activations = extractor.extract()

    # Train sparse autoencoder (with optional wandb logging)
    sae = SparseAutoencoder(
        activation_dim=activations.shape[-1],
        hidden_dim=activations.shape[-1] * 8,  # 8x expansion
        l1_coefficient=0.01
    )
    sae.fit(activations, wandb_enabled=True)

    # Visualize features (with optional wandb logging)
    visualizer = FeatureVisualizer(sae, activations)
    visualizer.plot_feature_density()
    visualizer.plot_top_activating_examples()
    ```
"""

__version__ = "0.1.0"

from drrik.models import ActivationExtractor
from drrik.autoencoder import SparseAutoencoder
from drrik.visualization import FeatureVisualizer
from drrik.steering import SAESteering, SteeringVectors
from drrik.config import Config
from drrik.settings import EnvironmentSettings, WandbConfig, get_settings

__all__ = [
    "ActivationExtractor",
    "SparseAutoencoder",
    "FeatureVisualizer",
    "SAESteering",
    "SteeringVectors",
    "Config",
    "EnvironmentSettings",
    "WandbConfig",
    "get_settings",
]
