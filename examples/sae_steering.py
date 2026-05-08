"""
SAE-based activation steering example.

Demonstrates how to use trained Sparse Autoencoder features to steer
language model generation, following the technique from Anthropic's
``Towards Monosemanticity`` paper.

The core idea: each SAE feature has a decoder weight vector representing
a direction in MLP activation space. By adding a scaled version of this
vector to the MLP activations during generation, the model's output is
biased toward concepts associated with that feature.

This example loads a pre-trained SAE checkpoint from ``drrik_output/``
and demonstrates steering with features discovered from the training data.

Requirements:
    - Pre-trained SAE checkpoint in ``drrik_output/`` (run ``drrik run`` first)
    - GPU recommended (generation is memory-intensive)
    - HuggingFace token for gated models (set HF_TOKEN env var)

Usage:
    uv run examples/sae_steering.py
"""

import os
import warnings
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from drrik import SAESteering, SparseAutoencoder

# Suppress macOS multiprocessing semaphore leak warnings from PyTorch/nnsight
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be .* leaked semaphore objects",
)

# Default paths matching the drrik_output directory structure
DEFAULT_SAE_PATH = Path("drrik_output/models/sae_model.pt")
DEFAULT_ACTIVATIONS_PATH = Path("drrik_output/activations/activations.npy")
DEFAULT_OUTPUT_DIR = Path("drrik_output/steering_results")

# Model and SAE configuration (matches the trained checkpoint)
MODEL_NAME = "google/gemma-2b"
TARGET_LAYER = 0
STEERING_STRENGTHS = [0.0, 1.0, 2.0, 3.0, 5.0]


def load_artifacts(
    sae_path: Path = DEFAULT_SAE_PATH,
    activations_path: Path = DEFAULT_ACTIVATIONS_PATH,
) -> tuple:
    """
    Load the trained SAE model and activations from disk.

    Args:
        sae_path: Path to the saved ``.pt`` SAE checkpoint.
        activations_path: Path to the ``.npy`` activations file.

    Returns:
        Tuple of (sae, activations, activations_path).

    Raises:
        FileNotFoundError: If either file is missing.
    """
    if not sae_path.exists():
        raise FileNotFoundError(
            f"SAE model not found at {sae_path}. "
            "Run ``drrik run`` first to train a model."
        )

    logger.info(f"Loading SAE from {sae_path}")
    sae = SparseAutoencoder.load(sae_path)

    if activations_path.exists():
        logger.info(f"Loading activations from {activations_path}")
        activations = np.load(str(activations_path))
        logger.info(f"Activations shape: {activations.shape}")
    else:
        logger.warning(
            f"No activations found at {activations_path}. "
            "Feature analysis will be skipped; steering will use "
            "default features."
        )
        activations = None

    return sae, activations


def analyze_features(
    sae: SparseAutoencoder,
    activations: np.ndarray,
) -> list:
    """
    Analyze feature densities and select interesting features.

    Finds the most active features and returns them sorted by density,
    suitable for steering demonstrations.

    Args:
        sae: The trained SparseAutoencoder.
        activations: MLP activations array of shape ``(n_samples, dim)``.

    Returns:
        List of dicts with ``feature_idx`` and ``density`` keys.
    """
    densities = sae.get_feature_density(activations)
    n_dead = (densities == 0).sum()
    n_active = (densities > 0).sum()

    logger.info("Feature statistics:")
    logger.info(f"Dead features: {n_dead}/{len(densities)}")
    logger.info(f"Active features: {n_active}/{len(densities)}")

    active_mask = densities > 0
    active_indices = np.where(active_mask)[0]
    sorted_indices = active_indices[np.argsort(densities[active_indices])[::-1]]

    features = []
    for idx in sorted_indices[:20]:
        features.append(
            {
                "feature_idx": int(idx),
                "density": float(densities[idx]),
            }
        )

    logger.info("Top 10 features by density:")
    for i, f in enumerate(features[:10]):
        logger.info(f"{i + 1}. Feature {f['feature_idx']}: density={f['density']:.4f}")

    return features


def demo_baseline_vs_steered(
    steering: SAESteering,
    prompt: str,
    feature_idx: int,
    strengths: list = [],
    max_new_tokens: int = 4096,
) -> dict:
    """
    Compare baseline generation against steered outputs at multiple strengths.

    Args:
        steering: The SAESteering instance.
        prompt: The text prompt to generate from.
        feature_idx: SAE feature index to steer with.
        strengths: List of strength values to test.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Dictionary mapping strength labels to generated text.
    """
    if strengths is None:
        strengths = STEERING_STRENGTHS

    results = {}
    for s in strengths:
        label = "baseline" if s == 0.0 else f"strength_{s}"
        logger.info(f"Generating :: {label}")
        generated = steering.generate(
            text=prompt,
            feature_idx=feature_idx,
            strength=s,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
        )
        results[label] = generated

    return results


def demo_multi_feature_steering(
    steering: SAESteering,
    prompt: str,
    features: list,
    max_new_tokens: int = 4096,
) -> str:
    """
    Generate text by combining multiple feature steering directions.

    Args:
        steering: The SAESteering instance.
        prompt: The text prompt.
        features: List of feature dicts with ``feature_idx`` keys.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The generated text string.
    """
    feature_indices = [f["feature_idx"] for f in features[:2]]
    strengths = [1.0] * len(feature_indices)

    logger.info(f"Combining features {feature_indices} with strengths {strengths}")

    result = steering.generate(
        text=prompt,
        feature_indices=feature_indices,
        strengths=strengths,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.9,
    )

    return result


def demo_steering_directions(
    steering: SAESteering,
    features: list,
) -> None:
    """
    Print analysis of steering direction vectors for selected features.

    Args:
        steering: The SAESteering instance.
        features: List of feature dicts to analyze.
    """
    for f in features[:5]:
        feature_idx = f["feature_idx"]
        direction = steering.get_steering_direction(feature_idx, normalize=True)

        logger.info(f"Feature {feature_idx}:")
        logger.info(f"Dimension: {direction.shape[0]}")
        logger.info(f"L2 norm: {np.linalg.norm(direction):.4f}")
        logger.info(f"Max positive: {direction.max():.4f}")
        logger.info(f"Max negative: {direction.min():.4f}")


def main():
    """Run the SAE steering demonstration."""
    torch.manual_seed(42)

    logger.info("SAE Activation Steering Demo")

    # Load pre-trained artifacts
    sae, activations = load_artifacts()

    # Analyze features if activations are available
    features = []
    if activations is not None:
        features = analyze_features(sae, activations)

    # Initialize steering controller
    hf_token = os.environ.get("HF_TOKEN")
    steering = SAESteering(
        sae=sae,
        model_name=MODEL_NAME,
        layer=TARGET_LAYER,
        device_map="mps",
        token=hf_token,
    )

    # nnsight loads models lazily on meta — dispatch to load real weights
    steering.model.dispatch()

    # Select features for demo
    if features:
        selected = features[:3]
    else:
        selected = [{"feature_idx": i} for i in range(3)]

    logger.info("Selected features for steering:")
    for i, f in enumerate(selected):
        logger.info(f"Feature {i + 1}: idx={f['feature_idx']}")

    # Demo 1: Baseline vs steered generation
    logger.info("Demo 1: Baseline vs Steered Generation")

    prompts = [
        "The weather is",
        "The capital of France is",
        "Python is a programming language that",
    ]

    for prompt in prompts:
        logger.info(f"Prompt: '{prompt}'")
        for feat in selected:
            feature_idx = feat["feature_idx"]
            results = demo_baseline_vs_steered(
                steering, prompt, feature_idx, max_new_tokens=4096
            )
            logger.info(f"Feature :: {feature_idx}")
            for label, text in results.items():
                logger.info(f"{label}: {text}")

    # Demo 2: Multi-feature steering
    logger.info("Demo 2: Multi-Feature Steering")

    if len(selected) >= 2:
        prompt = prompts[0]
        logger.info(f"Prompt: '{prompt}'")

        result = demo_multi_feature_steering(
            steering, prompt, selected, max_new_tokens=4096
        )
        logger.info(f"Combined result: {result}")

    # Demo 3: Steering direction analysis
    logger.info("Demo 3: Steering Direction Analysis")

    demo_steering_directions(steering, selected)

    # Save analysis
    logger.info("Saving results...")

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results saved to: {DEFAULT_OUTPUT_DIR}")

    logger.success("Demo complete!")


if __name__ == "__main__":
    main()
