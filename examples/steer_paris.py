"""
Steer google/gemma-4-12B-it to talk about Paris.

Loads a pre-trained SAE, finds the feature that most activates on
Paris-related text, and uses it to steer generation so every prompt
is pulled toward Paris regardless of the original topic.

Requirements:
    - Pre-trained SAE checkpoint in ``drrik_output/``
    - GPU with sufficient VRAM for gemma-4-12B-it
    - settings.yml with HUGGINGFACE_HUB_TOKEN

Usage:
    uv run examples/steer_paris.py
"""

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from drrik import SAESteering, SparseAutoencoder
from drrik.settings import get_settings

MODEL_NAME = "google/gemma-4-12B-it"
TARGET_LAYER = 20
SAE_PATH = Path("drrik_output/models/sae_model.pt")
ACTIVATIONS_PATH = Path("drrik_output/activations/activations.npy")
OUTPUT_DIR = Path("drrik_output/steering_paris")

PROMPTS = [
    "The weather today is",
    "My favorite programming language is",
    "The best food in the world is",
    "I love spending my weekends",
    "The most beautiful city is",
]


def find_paris_feature(steering: SAESteering, activations: np.ndarray) -> int:
    """Encode Paris-related text activations through the SAE to find the
    feature that fires strongest on Paris concepts.

    Args:
        steering: The SAESteering instance.
        activations: MLP activations array.

    Returns:
        The feature index most associated with Paris.
    """
    paris_text = "Paris the capital of France Eiffel Tower Louvre Seine"
    results = steering.find_steering_features(paris_text, activations, top_k=1)
    feature_idx = results[0]["feature_idx"]
    logger.info(
        f"Paris feature: idx={feature_idx}, activation={results[0]['activation']:.4f}"
    )
    return feature_idx


def main():
    torch.manual_seed(42)

    settings = get_settings()

    logger.info(f"Loading SAE from {SAE_PATH}")
    sae = SparseAutoencoder.load(SAE_PATH)

    logger.info(f"Loading activations from {ACTIVATIONS_PATH}")
    activations = np.load(str(ACTIVATIONS_PATH))

    steering = SAESteering(
        source=sae,
        model_name=MODEL_NAME,
        layer=TARGET_LAYER,
        device_map="auto",
        token=settings.huggingface_hub_token,
    )
    steering.model.dispatch()

    paris_feature = find_paris_feature(steering, activations)

    results = {}
    for prompt in PROMPTS:
        logger.info(f"Prompt: '{prompt}'")

        baseline = steering.generate(prompt, max_new_tokens=50)
        steered = steering.generate(
            prompt, feature_idx=paris_feature, strength=4.0, max_new_tokens=50
        )

        logger.info(f"Baseline: {baseline}")
        logger.info(f"Steered:  {steered}")

        results[f"{prompt}_baseline"] = baseline
        results[f"{prompt}_steered"] = steered

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    steering.save_steering_analysis(
        results,
        OUTPUT_DIR,
        prompt="; ".join(PROMPTS),
        feature_label=f"paris_feature_{paris_feature}",
    )
    logger.info(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
