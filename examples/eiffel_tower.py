"""
Steer Llama 3.1 8B Instruct to talk about the Eiffel Tower.

Reproduces the Eiffel Tower Llama demo (dlouapre/eiffel-tower-llama) using
pre-extracted steering vectors from Anthropic's Llama 3.1 8B SAEs, published
as the HF dataset ``shawon/llama-3.1-8b-instruct_eiffel_tower``.

Requirements:
    - ~20GB free RAM for Llama 3.1 8B in fp16 (Apple Silicon: use
      device_map="auto"; fall back to "cpu" if MPS runs out of memory)
    - The vectors dataset (run ``drrik extract-vectors`` once to create it)
    - settings.yml with HUGGINGFACE_HUB_TOKEN (Llama is gated)

Usage:
    uv run examples/eiffel_tower.py
"""

from loguru import logger

from drrik import SAESteering
from drrik.steering import SteeringVectors

DATASET = "shawon/llama-3.1-8b-instruct_eiffel_tower"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."
OUTPUT_DIR = "drrik_output/steering_eiffel"

PROMPTS = [
    "The weather today is",
    "My favorite programming language is",
    "The best food in the world is",
    "I love spending my weekends",
    "The most beautiful city is",
]


def main():
    logger.info(f"Loading steering vectors from {DATASET}")
    vectors = SteeringVectors.from_hf_dataset(DATASET)

    logger.info(f"Loading {MODEL_NAME}")
    steering = SAESteering(source=vectors, model_name=MODEL_NAME)

    results = {}
    for prompt in PROMPTS:
        logger.info(f"Prompt: {prompt!r}")
        for scale, label in [(0.0, "baseline"), (1.0, "steered")]:
            text = steering.generate(
                prompt,
                strength_scale=scale,
                max_new_tokens=128,
                temperature=0.5,
                repetition_penalty=1.2,
                system_prompt=SYSTEM_PROMPT,
                seed=16,
            )
            results[f"{prompt} | {label}"] = text
            logger.info(f"{label}: {text[:200]}")

    steering.save_steering_analysis(
        results, OUTPUT_DIR, prompt="; ".join(PROMPTS), feature_label="eiffel_tower"
    )
    logger.info(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
