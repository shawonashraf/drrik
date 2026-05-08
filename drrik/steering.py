"""
SAE-based activation steering for language model outputs.

This module implements the activation steering technique described in
Anthropic's `Towards Monosemanticity
<https://transformer-circuits.pub/2023/monosemantic-features/index.html>`__
paper.  The core idea is to use trained SAE features to steer language
model generation by adding scaled decoder weight vectors to MLP
activations during inference.

Steering Mechanism:
    Each SAE feature corresponds to a direction in activation space given
    by its decoder weight vector.  By adding
    ``strength * decoder_weight[:, feature_idx]`` to the MLP activations
    at a specific layer during generation, the model's output distribution
    is biased toward outputs associated with that feature.

    This follows the intervention procedure from the paper:

    1. Register a forward hook on the target MLP module.
    2. At each generation step the hook adds a scaled feature direction
       to the MLP output before it re-enters the residual stream.
    3. The hook is removed when generation completes.

Shared Utilities:
    :func:`resolve_module_path` is also imported by
    :mod:`drrik.models` so that dotted/bracket path resolution lives in
    one place.

Example:
    ```python
    from drrik import SparseAutoencoder, SAESteering

    sae = SparseAutoencoder.load("sae_model.pt")
    steering = SAESteering(sae, model_name="google/gemma-2b", layer=5)

    # Generate with steering
    output = steering.generate(
        "The weather is",
        feature_idx=42,
        strength=3.0,
        max_new_tokens=50,
    )
    ```
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import AutoTokenizer

from drrik.autoencoder import SparseAutoencoder
from drrik.settings import get_settings


def resolve_module_path(model, path: str):
    """Resolve a dotted/bracket path to the actual nnsight module.

    Shared by :class:`~drrik.models.ActivationExtractor` and
    :class:`SAESteering` so that path-resolution logic lives in one
    place.

    Parses a path string like ``model.layers[0].mlp`` by splitting on
    dots, handling both plain attribute access and integer bracket
    indexing.

    Args:
        model: The root model object (e.g. an nnsight
            :class:`~nnsight.LanguageModel`).
        path: A dot-separated module path with optional bracket
            indexing, e.g. ``model.layers[2].mlp``.

    Returns:
        The resolved module object.

    Example:
        ```python
        from drrik.steering import resolve_module_path
        mlp = resolve_module_path(model, "model.layers[3].mlp")
        ```
    """
    obj = model
    for part in re.split(r"\.", path):
        match = re.match(r"(\w+)\[(\d+)\]", part)
        if match:
            obj = getattr(obj, match.group(1))[int(match.group(2))]
        else:
            obj = getattr(obj, part)
    return obj


def _sample_next_token(
    logits: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """Apply temperature scaling, top-p (nucleus) filtering, and sample.

    Used by both :meth:`SAESteering._generate_with_hooks` and
    :meth:`SAESteering._generate_baseline` to avoid duplicating the
    sampling logic.

    Args:
        logits: Raw logits of shape ``(1, vocab_size)``.
        temperature: Sampling temperature.  Higher values produce
            more random outputs.
        top_p: Nucleus sampling threshold (0–1).  Only tokens
            within the cumulative probability mass are considered.

    Returns:
        Sampled token tensor of shape ``(1, 1)``.
    """
    logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float("-inf")

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


class SAESteering:
    """
    Steer language model generation using trained SAE features.

    Wraps a HuggingFace language model (via nnsight) and applies SAE
    feature interventions during generation.  At each generation step,
    a forward hook on the target MLP module adds a scaled decoder
    weight vector, biasing the residual stream toward the chosen
    feature direction.

    The HuggingFace auth token is read automatically from
    ``HUGGINGFACE_HUB_TOKEN`` (via :func:`~drrik.settings.get_settings`)
    when the *token* parameter is not explicitly provided.

    Attributes:
        sae: The trained :class:`~drrik.autoencoder.SparseAutoencoder`
            providing feature directions.
        model: The nnsight :class:`~nnsight.LanguageModel` wrapper.
        tokenizer: The HuggingFace tokenizer for the model.
        target_layer: The layer index where interventions are applied.
        device: The device the SAE parameters live on.

    Example:
        ```python
        sae = SparseAutoencoder.load("sae_model.pt")
        steering = SAESteering(sae, model_name="google/gemma-2b", layer=5)

        # Steered generation
        result = steering.generate(
            "The sky is",
            feature_idx=128,
            strength=2.5,
            max_new_tokens=50,
        )

        # Baseline (no steering)
        baseline = steering.generate("The sky is", max_new_tokens=50)

        # Multi-feature steering
        result = steering.generate(
            "The sky is",
            feature_indices=[10, 128],
            strengths=[1.0, 2.5],
        )
        ```
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        model_name: str,
        layer: int,
        revision: str = "main",
        torch_dtype: str = "float16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        token: Optional[str] = None,
    ):
        """
        Initialize the SAE steering controller.

        If ``token`` is not provided, it is read from the environment via
        ``get_settings()`` (i.e. ``HUGGINGFACE_HUB_TOKEN`` in ``.env``).

        Args:
            sae: A trained SparseAutoencoder whose decoder weights provide
                 steering directions.
            model_name: HuggingFace model identifier for generation (e.g.,
                       ``"google/gemma-2b"``).
            layer: The transformer layer index where MLP activations are
                   intercepted and modified.
            revision: Model revision to load.
            torch_dtype: Weight dtype for model loading.
            device_map: Device mapping strategy.
            trust_remote_code: Whether to trust remote code from the repo.
            token: HuggingFace token for gated models. Falls back to
                   ``HUGGINGFACE_HUB_TOKEN`` from ``.env`` when ``None``.
        """
        if token is None:
            settings = get_settings()
            token = settings.huggingface_hub_token

        self.sae = sae
        self.target_layer = layer
        self.device = next(sae.parameters()).device

        logger.info(f"Loading model '{model_name}' for steering at layer {layer}")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        load_dtype = dtype_map.get(torch_dtype, torch.float16)

        model_kwargs = {
            "revision": revision,
            "torch_dtype": load_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if token:
            model_kwargs["token"] = token

        self.model = LanguageModel(model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            token=token,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded on {self.model.device}")

    def _tokenize(self, text: str) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Tokenize text and move tensors to the model device.

        Centralises tokenization so that :meth:`generate` and
        :meth:`_generate_baseline` don't duplicate this work.

        Args:
            text: Raw prompt string.

        Returns:
            A tuple of ``(input_ids, attention_mask)`` tensors on
            the model's device.  ``attention_mask`` may be ``None``
            if the tokenizer does not produce one.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        return input_ids, attention_mask

    def _generate_with_hooks(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        steering_direction: torch.Tensor,
        strength: float,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """
        Generate text using forward-hook-based MLP intervention.

        Registers a forward hook on the target MLP module that adds
        ``strength * steering_direction`` to the output at every
        forward pass.  The hook is removed in a ``finally`` block to
        guarantee cleanup even on error.

        Generation is performed token-by-token using the underlying
        HuggingFace model (``self.model._module``) with top-p
        (nucleus) sampling.

        Args:
            input_ids: Tokenized input tensor of shape ``(1, seq_len)``.
            attention_mask: Optional attention mask tensor.
            steering_direction: The combined SAE decoder weight vector
                of shape ``(activation_dim,)``.
            strength: Multiplicative steering magnitude (typically ``1.0``
                because strengths are baked into *steering_direction* by
                the caller).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            The decoded text string (special tokens stripped).
        """
        generated_tokens = input_ids.clone()
        steering_direction = steering_direction.to(generated_tokens.device)
        target_module = resolve_module_path(
            self.model, f"model.layers[{self.target_layer}].mlp"
        )
        hook_handle = None

        try:

            def steering_hook(module, input_args, output):
                if isinstance(output, torch.Tensor):
                    output = output + strength * steering_direction
                elif isinstance(output, tuple):
                    output = tuple(
                        o + strength * steering_direction
                        if isinstance(o, torch.Tensor)
                        else o
                        for o in output
                    )
                return output

            hook_handle = target_module.register_forward_hook(steering_hook)

            base_model = self.model._module
            device = generated_tokens.device

            for _ in tqdm(range(max_new_tokens), desc="Steering generation"):
                if generated_tokens.shape[-1] >= self.tokenizer.model_max_length:
                    break

                outputs = base_model(
                    input_ids=generated_tokens,
                    attention_mask=attention_mask,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]
                next_token = _sample_next_token(logits, temperature, top_p)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones(1, 1, device=device)], dim=-1
                    )

        finally:
            if hook_handle is not None:
                hook_handle.remove()

        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def generate(
        self,
        text: str,
        feature_idx: Optional[int] = None,
        strength: float = 1.0,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        feature_indices: Optional[List[int]] = None,
        strengths: Optional[List[float]] = None,
    ) -> str:
        """
        Generate text with optional SAE feature steering.

        This is the main entry point for generation.

        - **No features** → calls :meth:`_generate_baseline` (plain
          autoregressive generation).
        - **Single feature** → pass ``feature_idx`` and ``strength``.
        - **Multiple features** → pass ``feature_indices`` and
          ``strengths`` as equal-length lists; the weighted decoder
          directions are summed into a single combined vector.

        Args:
            text: The prompt text to generate from.
            feature_idx: Index of a single SAE feature to steer with.
            strength: Steering magnitude for a single feature
                (default ``1.0``).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            feature_indices: List of feature indices for multi-feature
                steering.
            strengths: List of steering magnitudes, one per entry in
                ``feature_indices``.  Defaults to ``[1.0, …]`` if
                ``None``.

        Returns:
            The generated text string (special tokens stripped).

        Raises:
            ValueError: If ``feature_indices`` and ``strengths`` have
                mismatched lengths, or a feature index is out of range.

        Example:
            ```python
            # Single feature
            steering.generate("Hello", feature_idx=42, strength=2.0)

            # Multi-feature
            steering.generate(
                "Hello",
                feature_indices=[10, 42],
                strengths=[1.0, 3.0],
            )

            # Baseline
            steering.generate("Hello", max_new_tokens=100)
            ```
        """
        if feature_idx is None and feature_indices is None:
            return self._generate_baseline(text, max_new_tokens, temperature, top_p)

        if feature_idx is not None:
            feature_indices = [feature_idx]
            strengths = [strength]
        elif strengths is None:
            strengths = [1.0] * len(feature_indices)

        if len(feature_indices) != len(strengths):
            raise ValueError("feature_indices and strengths must have the same length")

        combined_direction = torch.zeros(self.sae.activation_dim, device=self.device)
        for fid, s in zip(feature_indices, strengths):
            if fid < 0 or fid >= self.sae.hidden_dim:
                raise ValueError(
                    f"feature_idx {fid} out of range [0, {self.sae.hidden_dim})"
                )
            combined_direction += s * self.sae.decoder.weight[:, fid]

        input_ids, attention_mask = self._tokenize(text)

        return self._generate_with_hooks(
            input_ids,
            attention_mask,
            combined_direction,
            1.0,
            max_new_tokens,
            temperature,
            top_p,
        )

    def _generate_baseline(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """
        Generate text without any SAE steering (baseline).

        Runs the same token-by-token autoregressive loop as
        :meth:`_generate_with_hooks` but without registering a
        steering hook.

        Args:
            text: The prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            The generated text string (special tokens stripped).
        """
        input_ids, attention_mask = self._tokenize(text)

        base_model = self.model._module
        generated_tokens = input_ids.clone()

        with torch.no_grad():
            for _ in tqdm(range(max_new_tokens), desc="Baseline generation"):
                if generated_tokens.shape[-1] >= self.tokenizer.model_max_length:
                    break

                outputs = base_model(
                    input_ids=generated_tokens,
                    attention_mask=attention_mask,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]
                next_token = _sample_next_token(logits, temperature, top_p)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(1, 1, device=generated_tokens.device),
                        ],
                        dim=-1,
                    )

        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def compare_steering(
        self,
        text: str,
        feature_idx: int,
        strengths: Optional[List[float]] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> Dict[str, str]:
        """
        Compare baseline generation against steered generations at different strengths.

        Runs one generation per strength value (including ``0.0`` for a
        no-steering baseline) and returns a dict mapping descriptive
        labels to the generated text.

        Args:
            text: The prompt text.
            feature_idx: SAE feature index to steer with.
            strengths: List of strength values to test.  Defaults to
                ``[0.0, 1.0, 2.0, 3.0, 5.0]``.
            max_new_tokens: Maximum tokens to generate per run.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Dictionary mapping strength labels to generated text.
            Keys are ``"baseline"`` (for strength ``0.0``) and
            ``"strength_1.0"``, ``"strength_2.0"``, etc.

        Example:
            ```python
            results = steering.compare_steering(
                "The sky is",
                feature_idx=42,
                strengths=[0.0, 1.0, 3.0],
            )
            for label, text in results.items():
                print(f"{label}: {text}")
            ```
        """
        if strengths is None:
            strengths = [0.0, 1.0, 2.0, 3.0, 5.0]

        results = {}

        for s in strengths:
            if s == 0.0:
                label = "baseline"
            else:
                label = f"strength_{s}"

            logger.info(f"Generating with {label} (strength={s})")
            generated = self.generate(
                text,
                feature_idx=feature_idx,
                strength=s,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results[label] = generated

        return results

    def find_steering_features(
        self,
        text: str,
        activations: np.ndarray,
        top_k: int = 20,
        min_activation: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-k SAE features that most activate on given activations.

        Encodes the provided activations through the SAE and returns the
        features with the highest mean activation values, sorted by
        magnitude.  Useful for identifying which features are relevant
        to a given input before steering.

        Args:
            text: The prompt text (used for logging only).
            activations: MLP activations array of shape
                ``(n_samples, activation_dim)``.
            top_k: Number of top features to return.
            min_activation: Minimum mean activation threshold to
                consider a feature.

        Returns:
            List of dicts sorted by descending activation, each
            containing:

            - ``feature_idx`` (``int``): The feature index.
            - ``activation`` (``float``): Mean activation value.
            - ``normalized_activation`` (``float``): Activation
              divided by the decoder weight L2 norm.
            - ``decoder_weight_norm`` (``float``): L2 norm of the
              decoder weight vector.
        """
        self.sae.eval()
        device = self.device

        with torch.no_grad():
            if isinstance(activations, np.ndarray):
                acts_tensor = torch.from_numpy(activations).float().to(device)
            else:
                acts_tensor = activations.to(device)

            features = self.sae.encode(acts_tensor)

            avg_activations = features.mean(dim=0).cpu().numpy()
            decoder_norms = self.sae.decoder.weight.norm(dim=0).cpu().numpy()

            active_mask = avg_activations >= min_activation
            active_indices = np.where(active_mask)[0]

            sorted_indices = active_indices[
                np.argsort(avg_activations[active_indices])[::-1]
            ][:top_k]

            results = []
            for idx in sorted_indices:
                results.append(
                    {
                        "feature_idx": int(idx),
                        "activation": float(avg_activations[idx]),
                        "normalized_activation": float(
                            avg_activations[idx] / (decoder_norms[idx] + 1e-8)
                        ),
                        "decoder_weight_norm": float(decoder_norms[idx]),
                    }
                )

        logger.info(f"Found {len(results)} active features for text: {text[:100]}")
        return results

    def get_steering_direction(
        self,
        feature_idx: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Get the steering direction vector for a given feature.

        The steering direction is the SAE decoder weight column for the
        specified feature.  This vector represents the direction in MLP
        activation space that the feature encodes.

        Args:
            feature_idx: Index of the SAE feature.
            normalize: If ``True``, L2-normalise the direction vector.

        Returns:
            NumPy array of shape ``(activation_dim,)`` containing the
            steering direction.

        Raises:
            ValueError: If ``feature_idx`` is out of range
                ``[0, sae.hidden_dim)``.
        """
        if feature_idx < 0 or feature_idx >= self.sae.hidden_dim:
            raise ValueError(
                f"feature_idx {feature_idx} out of range [0, {self.sae.hidden_dim})"
            )

        direction = self.sae.decoder.weight[:, feature_idx].detach().cpu().numpy()

        if normalize:
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

        return direction

    def save_steering_analysis(
        self,
        results: Dict[str, str],
        output_dir: Union[str, Path],
        prompt: str = "",
        feature_label: str = "",
    ) -> Path:
        """
        Save steering comparison results to disk.

        Writes a text file containing the prompt and all generated
        outputs (baseline and steered) for later review.  The
        filename includes a timestamp (and optional feature label)
        to prevent overwriting previous analyses.

        Args:
            results: Dictionary mapping labels to generated text
                (as returned by :meth:`compare_steering`).
            output_dir: Directory to save the analysis file.
            prompt: The original prompt text.
            feature_label: Optional label included in the filename
                (e.g. ``"feature_42"``).

        Returns:
            Path to the saved analysis file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{feature_label}" if feature_label else ""
        filepath = output_dir / f"steering_analysis{suffix}_{timestamp}.txt"

        with open(filepath, "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write("=" * 80 + "\n\n")

            for label, text in results.items():
                f.write(f"--- {label} ---\n")
                f.write(f"{text}\n\n")
                f.write("-" * 40 + "\n")

        logger.info(f"Saved steering analysis to {filepath}")
        return filepath
