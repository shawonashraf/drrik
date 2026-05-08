"""
SAE-based activation steering for language model outputs.

This module implements the activation steering technique described in Anthropic's
``Towards Monosemanticity`` paper. The core idea is to use trained SAE features
to steer language model generation by adding scaled decoder weight vectors to
MLP activations during inference.

Steering Mechanism:
    Each SAE feature corresponds to a direction in activation space given by
    its decoder weight vector. By adding ``strength * decoder_weight[:, feature_idx]``
    to the MLP activations at a specific layer during generation, the model's
    output distribution is biased toward outputs associated with that feature.

    This follows the intervention procedure from the paper:
    1. Extract MLP activations at a target layer during generation
    2. Add a scaled feature direction to the activations
    3. Continue the forward pass with the modified activations

Example:
    ```python
    from drrik import SparseAutoencoder
    from drrik.steering import SAESteering

    sae = SparseAutoencoder.load("path/to/sae_model.pt")
    steering = SAESteering(sae, model_name="google/gemma-2b", layer=5)

    # Generate with steering
    output = steering.generate(
        "The weather is",
        feature_idx=42,
        strength=3.0,
        num_steps=20,
    )
    ```
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import re

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from nnsight import LanguageModel

from loguru import logger

from drrik.autoencoder import SparseAutoencoder


class SAESteering:
    """
    Steer language model generation using trained SAE features.

    This class wraps a HuggingFace language model (via nnsight) and applies
    SAE feature interventions during generation. At each generation step,
    the MLP activations at a target layer are modified by adding a scaled
    decoder weight vector corresponding to the chosen feature.

    The steering direction is the SAE decoder weight for a given feature,
    which represents the direction in activation space that the feature
    encodes. Adding this direction to the MLP activations biases the model
    toward outputs associated with that feature.

    Attributes:
        sae: The trained SparseAutoencoder providing feature directions.
        model: The nnsight LanguageModel wrapper for generation.
        tokenizer: The HuggingFace tokenizer for the model.
        target_layer: The layer index where interventions are applied.
        device: The device the model runs on.

    Example:
        ```python
        sae = SparseAutoencoder.load("sae_model.pt")
        steering = SAESteering(sae, model_name="google/gemma-2b", layer=5)

        # Positive steering
        result = steering.generate(
            "The sky is",
            feature_idx=128,
            strength=2.5,
            max_new_tokens=50,
        )

        # Compare with and without steering
        baseline = steering.generate("The sky is", max_new_tokens=50)
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
            token: HuggingFace token for gated models.
        """
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

    def _get_mlp_output(
        self,
        layer_output,
    ) -> torch.Tensor:
        """
        Extract the MLP output tensor from a layer's trace object.

        For most architectures (Gemma, Llama), the MLP output is the direct
        output of the MLP module. This method handles the nnsight proxy
        tensor returned during tracing.

        Args:
            layer_output: The nnsight proxy tensor for the MLP output.

        Returns:
            The MLP activation tensor.
        """
        return layer_output

    def steer_generation(
        self,
        text: str,
        feature_idx: int,
        strength: float = 1.0,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Generate text with SAE feature steering applied.

        During generation, at each step the MLP activations at the target
        layer are modified by adding ``strength * decoder_weight[:, feature_idx]``.
        This biases the model's next-token distribution toward outputs
        associated with the given feature.

        The steering is applied using nnsight's tracing mechanism to
        intercept and modify the MLP output before it passes to the
        residual stream.

        Args:
            text: The prompt text to generate from.
            feature_idx: Index of the SAE feature to use for steering.
            strength: Magnitude of steering. Higher values produce stronger
                     bias. Typical range: 0.0 to 5.0.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature for generation.
            top_p: Nucleus sampling threshold.
            attention_mask: Optional attention mask from tokenization.

        Returns:
            The generated text string.

        Raises:
            ValueError: If ``feature_idx`` is out of range for the SAE.
        """
        if feature_idx < 0 or feature_idx >= self.sae.hidden_dim:
            raise ValueError(
                f"feature_idx {feature_idx} out of range [0, {self.sae.hidden_dim})"
            )

        # Get the steering direction from the SAE decoder weights
        # Shape: (activation_dim,)
        steering_direction = self.sae.decoder.weight[:, feature_idx]

        # Tokenize the prompt
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        if attention_mask is None:
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)

        # Get the steering direction on the model's device
        steering_direction = steering_direction.to(self.model.device)

        with torch.no_grad():
            current_ids = input_ids
            current_mask = attention_mask

            for step in tqdm(range(max_new_tokens), desc="Steering generation"):
                # Check if we've generated enough tokens
                if current_ids.shape[-1] >= self.tokenizer.model_max_length:
                    break

                # Use nnsight to trace through the model and intercept MLP output
                with self.model.trace(current_ids, attention_mask=current_mask):
                    # Navigate to the target layer's MLP module
                    module = self.model
                    for part in re.split(
                        r"\.", f"model.layers[{self.target_layer}].mlp"
                    ):
                        match = re.match(r"(\w+)\[(\d+)\]", part)
                        if match:
                            module = getattr(module, match.group(1))[
                                int(match.group(2))
                            ]
                        else:
                            module = getattr(module, part)

                    # Save the MLP output for intervention
                    mlp_output = module.output.save()

                # Get the actual MLP activations
                mlp_acts = mlp_output

                # Apply steering: add scaled feature direction
                # The steering direction is added to the MLP output
                # Shape: (batch, seq_len, hidden_dim) + (hidden_dim,)
                if mlp_acts.dim() == 3:
                    mlp_acts[:, :, :] = (
                        mlp_acts[:, :, :] + strength * steering_direction
                    )
                elif mlp_acts.dim() == 2:
                    mlp_acts[:] = mlp_acts[:] + strength * steering_direction

                # Get the next token logits from the model's LM head
                # We need to re-run with the modified activations
                # Since nnsight doesn't support in-place modification of proxy tensors
                # directly, we use a different approach: modify the module's forward

                break  # Exit the trace loop

            # Since nnsight's proxy tensors can't be modified in-place during trace,
            # we use a hook-based approach instead

        # Use hook-based intervention for proper steering
        result = self._generate_with_hooks(
            input_ids,
            attention_mask,
            steering_direction,
            strength,
            max_new_tokens,
            temperature,
            top_p,
        )

        return result

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

        Registers a forward hook on the target layer's MLP module to
        modify activations during each forward pass. This is the
        recommended approach for steering with nnsight.

        Args:
            input_ids: Tokenized input tensor.
            attention_mask: Attention mask tensor.
            steering_direction: The SAE decoder weight vector for the feature.
            strength: Steering magnitude.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            The generated text string.
        """
        generated_tokens = input_ids.clone()
        steering_direction = steering_direction.to(generated_tokens.device)

        target_module = None
        hook_handle = None

        try:
            # Navigate to the target layer's MLP
            target_module = self.model
            for part in re.split(r"\.", f"model.layers[{self.target_layer}].mlp"):
                match = re.match(r"(\w+)\[(\d+)\]", part)
                if match:
                    target_module = getattr(target_module, match.group(1))[
                        int(match.group(2))
                    ]
                else:
                    target_module = getattr(target_module, part)

            def steering_hook(module, input_args, output):
                """Forward hook that adds steering direction to MLP output."""
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

            # Register the hook
            hook_handle = target_module.register_forward_hook(steering_hook)

            # Generate token by token using the underlying transformers model
            # nnsight's LanguageModel wraps a transformers model
            base_model = self.model._module
            device = generated_tokens.device

            for step in tqdm(range(max_new_tokens), desc="Steering generation"):
                # Forward pass
                outputs = base_model(
                    input_ids=generated_tokens,
                    attention_mask=attention_mask
                    if attention_mask is not None
                    else None,
                    use_cache=True,
                )

                # Get logits for the last position
                logits = outputs.logits[:, -1, :]

                # Apply temperature
                logits = logits / temperature

                # Apply top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens below cumulative probability threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices left by 1 to mark the threshold token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                # Index into sorted_indices to get the actual indices to remove
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

                # Sample next token
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Append token (next_token is already (1, 1) from multinomial on 2D logits)
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones(1, 1, device=device)], dim=-1
                    )

        finally:
            # Clean up hook
            if hook_handle is not None:
                hook_handle.remove()

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )

        return generated_text

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

        This is the main entry point for generation. If ``feature_idx`` or
        ``feature_indices`` is provided, steering is applied. Otherwise,
        a baseline (unsteered) generation is produced.

        Multiple features can be combined by passing ``feature_indices``
        and ``strengths`` as lists. The steering directions are summed
        with their respective weights.

        Args:
            text: The prompt text to generate from.
            feature_idx: Index of a single SAE feature to steer with.
            strength: Steering magnitude for a single feature.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            feature_indices: List of feature indices for multi-feature steering.
            strengths: List of steering magnitudes corresponding to
                      ``feature_indices``.

        Returns:
            The generated text string.

        Raises:
            ValueError: If neither ``feature_idx`` nor ``feature_indices``
                       is provided for steered generation, or if the lists
                       have mismatched lengths.
        """
        # If no steering requested, do baseline generation
        if feature_idx is None and feature_indices is None:
            return self._generate_baseline(text, max_new_tokens, temperature, top_p)

        # Handle single feature
        if feature_idx is not None:
            feature_indices = [feature_idx]
            strengths = [strength]
        elif strengths is None:
            strengths = [1.0] * len(feature_indices)

        if len(feature_indices) != len(strengths):
            raise ValueError("feature_indices and strengths must have the same length")

        # Compute combined steering direction
        combined_direction = torch.zeros(self.sae.activation_dim, device=self.device)
        for fid, s in zip(feature_indices, strengths):
            if fid < 0 or fid >= self.sae.hidden_dim:
                raise ValueError(
                    f"feature_idx {fid} out of range [0, {self.sae.hidden_dim})"
                )
            combined_direction += s * self.sae.decoder.weight[:, fid]

        return self._generate_with_hooks(
            self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)[
                "input_ids"
            ].to(self.model.device),
            self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            .get("attention_mask", None)
            .to(self.model.device)
            if self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).get("attention_mask", None)
            is not None
            else None,
            combined_direction,
            1.0,  # strength is already baked into combined_direction
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

        Args:
            text: The prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            The generated text string.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        base_model = self.model._module
        generated_tokens = input_ids.clone()

        with torch.no_grad():
            for step in tqdm(range(max_new_tokens), desc="Baseline generation"):
                if generated_tokens.shape[-1] >= self.tokenizer.model_max_length:
                    break

                outputs = base_model(
                    input_ids=generated_tokens,
                    attention_mask=attention_mask,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]
                logits = logits / temperature

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

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
        strengths: List[float] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> Dict[str, str]:
        """
        Compare baseline generation against steered generations at different strengths.

        Runs one baseline generation and multiple steered generations at
        increasing strength levels, returning all results for comparison.

        Args:
            text: The prompt text.
            feature_idx: SAE feature index to steer with.
            strengths: List of strength values to test. Defaults to
                      ``[0.0, 1.0, 2.0, 3.0, 5.0]``.
            max_new_tokens: Maximum tokens to generate per run.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Dictionary mapping strength labels to generated text strings.
            Keys are ``"baseline"``, ``"strength_1.0"``, ``"strength_2.0"``, etc.
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
        max_new_tokens: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-k SAE features that most activate on given text.

        Encodes the provided activations through the SAE and returns the
        features with the highest activation values, sorted by magnitude.
        This helps identify which features are relevant to a given input.

        Args:
            text: The prompt text (for logging purposes).
            activations: MLP activations array of shape ``(n_samples, activation_dim)``.
            top_k: Number of top features to return.
            min_activation: Minimum activation threshold to consider.
            max_new_tokens: Max tokens for any generation during analysis.

        Returns:
            List of dicts, each containing:
            - ``feature_idx``: The feature index.
            - ``activation``: The activation value.
            - ``normalized_activation``: Activation normalized by decoder weight norm.
            - ``decoder_weight_norm``: L2 norm of the decoder weight vector.
        """
        self.sae.eval()
        device = self.device

        with torch.no_grad():
            if isinstance(activations, np.ndarray):
                acts_tensor = torch.from_numpy(activations).float().to(device)
            else:
                acts_tensor = activations.to(device)

            features = self.sae.encode(acts_tensor)

            # Average activation across samples
            avg_activations = features.mean(dim=0).cpu().numpy()

            # Get decoder weight norms
            decoder_norms = self.sae.decoder.weight.norm(dim=0).cpu().numpy()

            # Filter by minimum activation
            active_mask = avg_activations >= min_activation
            active_indices = np.where(active_mask)[0]

            # Sort by activation value
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

        The steering direction is the SAE decoder weight vector for the
        specified feature. This vector represents the direction in MLP
        activation space that the feature encodes.

        Args:
            feature_idx: Index of the SAE feature.
            normalize: Whether to L2-normalize the direction vector.

        Returns:
            NumPy array of shape ``(activation_dim,)`` containing the
            steering direction.

        Raises:
            ValueError: If ``feature_idx`` is out of range.
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
    ) -> Path:
        """
        Save steering comparison results to disk.

        Writes a text file containing the prompt and all generated
        outputs (baseline and steered) for later review.

        Args:
            results: Dictionary mapping labels to generated text.
            output_dir: Directory to save the analysis file.
            prompt: The original prompt text.

        Returns:
            Path to the saved analysis file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / "steering_analysis.txt"

        with open(filepath, "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write("=" * 80 + "\n\n")

            for label, text in results.items():
                f.write(f"--- {label} ---\n")
                f.write(f"{text}\n\n")
                f.write("-" * 40 + "\n")

        logger.info(f"Saved steering analysis to {filepath}")
        return filepath
