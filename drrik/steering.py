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
    steering = SAESteering(source=sae, model_name="google/gemma-2b", layer=5)

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply repetition penalty, temperature, top-p, and sample a token.

    Used by both :meth:`SAESteering._generate_with_hooks` and
    :meth:`SAESteering._generate_baseline`.

    Args:
        logits: Raw logits of shape ``(1, vocab_size)``.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold (0–1).
        repetition_penalty: CTRL-style penalty for already-generated
            tokens. ``1.0`` disables it. Positive logits of seen tokens
            are divided by the penalty, negative logits multiplied.
        generated_tokens: All tokens generated so far (any shape); its
            unique ids receive the penalty. ``None`` disables it.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Sampled token tensor of shape ``(1, 1)``.
    """
    if (
        repetition_penalty != 1.0
        and generated_tokens is not None
        and generated_tokens.numel() > 0
    ):
        seen = generated_tokens.unique()
        logits = logits.clone()
        penalized = logits[:, seen]
        penalized = torch.where(
            penalized > 0,
            penalized / repetition_penalty,
            penalized * repetition_penalty,
        )
        logits[:, seen] = penalized

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

    return torch.multinomial(
        F.softmax(logits, dim=-1), num_samples=1, generator=generator
    )


@dataclass(frozen=True)
class SteeringComponent:
    """One steering direction: a unit-norm vector, its layer, and magnitude.

    Attributes:
        layer: Transformer layer index where the vector is injected.
        feature_idx: SAE feature index the vector was extracted from.
        strength: Absolute magnitude added along the unit vector.
        vector: Unit-norm direction of shape ``(activation_dim,)``.
    """

    layer: int
    feature_idx: int
    strength: float
    vector: torch.Tensor


class SteeringVectors:
    """Container of steering components with a shared hook path.

    Holds pre-extracted, unit-norm SAE decoder columns ready for injection
    during generation.  The hook path records where the source SAE was
    trained: ``"layer"`` (residual stream, e.g. Anthropic's
    ``resid_post_layer_*`` SAEs) or ``"mlp"`` (MLP output, e.g. SAEs
    trained by Drrik's own pipeline).

    Example:
        ```python
        vectors = SteeringVectors.from_hf_dataset(
            "shawon/llama-3.1-8b-instruct_eiffel_tower"
        )
        steering = SAESteering(source=vectors, model_name="meta-llama/Llama-3.1-8B-Instruct")
        ```
    """

    def __init__(self, components: List[SteeringComponent], hook_path: str = "layer"):
        """
        Args:
            components: Non-empty list of steering components.
            hook_path: ``"layer"`` (hook ``model.layers[N]`` output) or
                ``"mlp"`` (hook ``model.layers[N].mlp`` output).

        Raises:
            ValueError: If ``components`` is empty or ``hook_path`` is invalid.
        """
        if hook_path not in ("layer", "mlp"):
            raise ValueError(f"hook_path must be 'layer' or 'mlp', got {hook_path!r}")
        if not components:
            raise ValueError("SteeringVectors requires at least one component")
        self.components = list(components)
        self.hook_path = hook_path

    @property
    def layers(self) -> List[int]:
        """Sorted unique layer indices across all components."""
        return sorted({c.layer for c in self.components})

    @property
    def activation_dim(self) -> int:
        """Dimension of the steering vectors (all components share it)."""
        return self.components[0].vector.shape[0]

    @classmethod
    def from_sae(
        cls,
        sae: SparseAutoencoder,
        layer: int,
        feature_indices: List[int],
        strengths: Optional[List[float]] = None,
    ) -> "SteeringVectors":
        """Build components from a Drrik-trained SAE's decoder columns.

        Args:
            sae: A trained :class:`~drrik.autoencoder.SparseAutoencoder`.
            layer: Layer index the SAE was trained on.
            feature_indices: Feature indices to extract.
            strengths: Magnitude per feature. Defaults to all ``1.0``.

        Returns:
            ``SteeringVectors`` with ``hook_path="mlp"``.

        Raises:
            ValueError: If lengths mismatch or an index is out of range.
        """
        if strengths is None:
            strengths = [1.0] * len(feature_indices)
        if len(feature_indices) != len(strengths):
            raise ValueError("feature_indices and strengths must have the same length")

        components = []
        for fid, s in zip(feature_indices, strengths):
            if fid < 0 or fid >= sae.hidden_dim:
                raise ValueError(
                    f"feature_idx {fid} out of range [0, {sae.hidden_dim})"
                )
            v = sae.decoder.weight[:, fid].detach().clone().float()
            v = v / v.norm()
            components.append(SteeringComponent(layer, int(fid), float(s), v))
        return cls(components, hook_path="mlp")

    @classmethod
    def from_hf_dataset(cls, source: str) -> "SteeringVectors":
        """Load components from an HF dataset (or local parquet file).

        The dataset must contain ``layer``, ``feature_idx``, ``strength``,
        and ``vector`` (list of floats) columns, as produced by
        ``drrik extract-vectors``.

        Args:
            source: HF dataset repo id (e.g.
                ``"shawon/llama-3.1-8b-instruct_eiffel_tower"``) or a
                local path to a parquet file.

        Returns:
            ``SteeringVectors`` with ``hook_path="layer"``.
        """
        from datasets import load_dataset

        p = Path(source)
        if p.exists():
            ds = load_dataset("parquet", data_files=str(p))["train"]
        else:
            ds = load_dataset(source)["train"]

        components = [
            SteeringComponent(
                layer=int(row["layer"]),
                feature_idx=int(row["feature_idx"]),
                strength=float(row["strength"]),
                vector=torch.tensor(row["vector"], dtype=torch.float32),
            )
            for row in ds
        ]
        return cls(components, hook_path="layer")

    def save(self, path: Union[str, Path]) -> None:
        """Save components and hook path to a ``.pt`` file.

        Args:
            path: Destination path. Parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hook_path": self.hook_path,
                "components": [
                    {
                        "layer": c.layer,
                        "feature_idx": c.feature_idx,
                        "strength": c.strength,
                        "vector": c.vector.cpu(),
                    }
                    for c in self.components
                ],
            },
            path,
        )
        logger.info(f"Saved {len(self.components)} steering vectors to {path}")

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SteeringVectors":
        """Load components saved by :meth:`save` (hook path restored).

        Args:
            path: Path to a ``.pt`` file written by :meth:`save`.

        Returns:
            The reconstructed ``SteeringVectors``.
        """
        state = torch.load(Path(path), weights_only=True)
        components = [
            SteeringComponent(
                layer=int(c["layer"]),
                feature_idx=int(c["feature_idx"]),
                strength=float(c["strength"]),
                vector=c["vector"].float(),
            )
            for c in state["components"]
        ]
        return cls(components, hook_path=state["hook_path"])


def make_steering_hook(components: List[Tuple[float, torch.Tensor]], clamp: bool):
    """Build a forward hook that injects steering vectors into a layer output.

    For each ``(strength, vector)`` pair the hook adds ``strength * vector``
    to the hidden states.  With ``clamp=True`` the hidden state's existing
    projection onto the vector is removed first, so the component along the
    vector becomes exactly ``strength`` (the scheme used in Anthropic's
    Golden Gate demo and the Eiffel Tower Llama reproduction).

    Handles 2D ``(batch, hidden)`` and 3D ``(batch, seq, hidden)`` hidden
    states and tuple module outputs.

    Args:
        components: ``(strength, unit_vector)`` pairs for one layer.
        clamp: Whether to remove the existing projection before adding.

    Returns:
        A forward-hook callable for ``register_forward_hook``.
    """

    def hook(module, input_args, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not isinstance(hidden, torch.Tensor):
            return output

        was_2d = hidden.dim() == 2
        if was_2d:
            hidden = hidden.unsqueeze(1)

        for strength, vector in components:
            v = vector.to(dtype=hidden.dtype, device=hidden.device)
            if clamp:
                proj = torch.einsum("bsh,h->bs", hidden, v).unsqueeze(-1) * v
                hidden = hidden - proj
            hidden = hidden + strength * v

        if was_2d:
            hidden = hidden.squeeze(1)
        if rest is not None:
            return (hidden,) + rest
        return hidden

    return hook


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
        source: The steering source this instance was built from — a
            :class:`~drrik.autoencoder.SparseAutoencoder` or a
            :class:`SteeringVectors`.
        sae: The trained :class:`~drrik.autoencoder.SparseAutoencoder`
            providing feature directions, or ``None`` when built from
            :class:`SteeringVectors`.
        vectors: The :class:`SteeringVectors` pre-extracted components,
            or ``None`` when built from a ``SparseAutoencoder``.
        model: The nnsight :class:`~nnsight.LanguageModel` wrapper.
        tokenizer: The HuggingFace tokenizer for the model.
        target_layer: The layer index where interventions are applied
            (``None`` for :class:`SteeringVectors`; layers come from
            the components).
        hook_path: ``"mlp"`` for SAE sources; the source's
            ``hook_path`` for :class:`SteeringVectors`.
        device: The device the steering directions live on.

    Example:
        ```python
        sae = SparseAutoencoder.load("sae_model.pt")
        steering = SAESteering(source=sae, model_name="google/gemma-2b", layer=5)

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

        Or from pre-extracted :class:`SteeringVectors`:

        ```python
        vectors = SteeringVectors.from_hf_dataset("user/dataset")
        steering = SAESteering(source=vectors, model_name="google/gemma-2b")
        ```
    """

    def __init__(
        self,
        source: Union[SparseAutoencoder, SteeringVectors],
        model_name: str,
        layer: Optional[int] = None,
        revision: str = "main",
        torch_dtype: str = "float16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        token: Optional[str] = None,
    ):
        """
        Initialize the SAE steering controller.

        Args:
            source: Steering direction source — either a trained
                :class:`~drrik.autoencoder.SparseAutoencoder` (then
                ``layer`` is required) or a
                :class:`SteeringVectors` of pre-extracted components
                (then ``layer`` is ignored; layers come from the
                components).
            model_name: HuggingFace model identifier for generation.
            layer: Transformer layer index for a ``SparseAutoencoder``
                source. Ignored for ``SteeringVectors``.
            revision: Model revision to load.
            torch_dtype: Weight dtype for model loading.
            device_map: Device mapping strategy.
            trust_remote_code: Whether to trust remote code from the repo.
            token: HuggingFace token for gated models. Falls back to
                ``HUGGINGFACE_HUB_TOKEN`` from ``settings.yml`` when ``None``.

        Raises:
            TypeError: If ``source`` is neither type.
            ValueError: If ``source`` is a ``SparseAutoencoder`` and
                ``layer`` is ``None``.
        """
        if isinstance(source, SparseAutoencoder):
            if layer is None:
                raise ValueError("layer is required when source is a SparseAutoencoder")
            self.sae: Optional[SparseAutoencoder] = source
            self.vectors: Optional[SteeringVectors] = None
            self.target_layer: Optional[int] = layer
            self.hook_path = "mlp"
            self.device = next(source.parameters()).device
        elif isinstance(source, SteeringVectors):
            self.sae = None
            self.vectors = source
            self.target_layer = None
            self.hook_path = source.hook_path
            self.device = source.components[0].vector.device
        else:
            raise TypeError(
                "source must be a SparseAutoencoder or SteeringVectors, "
                f"got {type(source).__name__}"
            )
        self.source = source

        if token is None:
            settings = get_settings()
            token = settings.huggingface_hub_token

        if layer is not None:
            logger.info(f"Loading model '{model_name}' for steering at layer {layer}")
        else:
            layers = sorted({c.layer for c in source.components})
            logger.info(
                f"Loading model '{model_name}' for steering at layer(s) {layers}"
            )

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

    def _build_prompt_ids(
        self, text: str, system_prompt: Optional[str]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Tokenize a prompt, optionally through the chat template.

        Args:
            text: User prompt text.
            system_prompt: When set, the prompt is rendered as a
                system+user chat via ``apply_chat_template`` (required
                for instruct models).

        Returns:
            ``(input_ids, attention_mask)`` on the model device.
        """
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            return input_ids.to(self.model.device), None
        return self._tokenize(text)

    def _generate_with_hooks(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        layer_components: Dict[int, List[Tuple[float, torch.Tensor]]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        clamp: bool,
        seed: Optional[int],
    ) -> str:
        """
        Generate text with forward-hook steering on one or more layers.

        Registers one :func:`make_steering_hook` per layer in
        *layer_components*; each hook injects its ``(strength, vector)``
        pairs at every forward pass.  All hooks are removed in a
        ``finally`` block to guarantee cleanup even on error.

        Generation is performed token-by-token using the underlying
        HuggingFace model (``self.model._module``) with top-p
        (nucleus) sampling.

        Args:
            input_ids: Tokenized input tensor of shape ``(1, seq_len)``.
            attention_mask: Optional attention mask tensor.
            layer_components: Map of layer index to ``(strength, vector)``
                pairs injected at that layer.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repetition_penalty: CTRL-style repetition penalty (1.0 off).
            clamp: Remove existing projection before adding (see
                :func:`make_steering_hook`).
            seed: Optional RNG seed for reproducible sampling.

        Returns:
            The decoded text string (special tokens stripped).
        """
        generated_tokens = input_ids.clone()
        device = generated_tokens.device
        hook_handles = []
        # ponytail: device-bound generator for reproducibility; torch.manual_seed
        # (global) if per-device generators ever misbehave on MPS
        generator = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None
            else None
        )

        try:
            for layer, comps in layer_components.items():
                if self.hook_path == "layer":
                    path = f"model.layers[{layer}]"
                else:
                    path = f"model.layers[{layer}].mlp"
                target_module = resolve_module_path(self.model, path)
                hook_handles.append(
                    target_module.register_forward_hook(
                        make_steering_hook(comps, clamp)
                    )
                )

            base_model = self.model._module

            for _ in tqdm(range(max_new_tokens), desc="Steering generation"):
                if generated_tokens.shape[-1] >= self.tokenizer.model_max_length:
                    break

                outputs = base_model(
                    input_ids=generated_tokens,
                    attention_mask=attention_mask,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]
                next_token = _sample_next_token(
                    logits,
                    temperature,
                    top_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=generated_tokens,
                    generator=generator,
                )

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones(1, 1, device=device)], dim=-1
                    )
        finally:
            for handle in hook_handles:
                handle.remove()

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
        clamp: bool = False,
        repetition_penalty: float = 1.0,
        system_prompt: Optional[str] = None,
        strength_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate text with optional SAE feature steering.

        This is the main entry point for generation.

        - **No features** → calls :meth:`_generate_baseline` (plain
          autoregressive generation).
        - **Single feature** → pass ``feature_idx`` and ``strength``.
        - **Multiple features** → pass ``feature_indices`` and
          ``strengths`` as equal-length lists; each decoder direction
          is injected as its own component.
        - **SteeringVectors source** → all pre-extracted components
          are injected at their layers, scaled by ``strength_scale``;
          feature selection params are rejected.

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
            clamp: Remove the hidden state's existing projection onto
                each steering vector before adding it, so the
                component along the vector becomes exactly ``strength``
                (the Golden Gate / Eiffel Tower scheme).
            repetition_penalty: CTRL-style repetition penalty for
                already-generated tokens (``1.0`` disables).
            system_prompt: When set, render the prompt as a
                system+user chat via ``apply_chat_template`` (required
                for instruct models).
            strength_scale: Multiplier on all steering strengths
                (applies to both SAE and SteeringVectors sources).
                ``0.0`` with a vectors source runs the baseline loop.
            seed: Optional RNG seed for reproducible sampling.

        Returns:
            The generated text string (special tokens stripped).

        Raises:
            ValueError: If ``feature_indices`` and ``strengths`` have
                mismatched lengths, a feature index is out of range, or
                feature selection params are passed with a
                :class:`SteeringVectors` source.

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

            # Chat-templated, clamped, reproducible
            steering.generate(
                "Hello",
                feature_idx=42,
                system_prompt="You are a helpful assistant.",
                clamp=True,
                seed=0,
            )

            # Baseline
            steering.generate("Hello", max_new_tokens=100)
            ```
        """
        if self.vectors is not None:
            if feature_idx is not None or feature_indices is not None:
                raise ValueError(
                    "feature selection is fixed by the SteeringVectors source; "
                    "use strength_scale to adjust magnitude"
                )
            layer_components = {}
            for c in self.vectors.components:
                layer_components.setdefault(c.layer, []).append(
                    (c.strength * strength_scale, c.vector)
                )
            if strength_scale == 0.0:
                return self._generate_baseline(
                    text,
                    max_new_tokens,
                    temperature,
                    top_p,
                    repetition_penalty,
                    seed,
                    system_prompt,
                )
        else:
            if feature_idx is None and feature_indices is None:
                return self._generate_baseline(
                    text,
                    max_new_tokens,
                    temperature,
                    top_p,
                    repetition_penalty,
                    seed,
                    system_prompt,
                )
            if feature_idx is not None:
                feature_indices = [feature_idx]
                strengths = [strength]
            elif strengths is None:
                strengths = [1.0] * len(feature_indices)
            if len(feature_indices) != len(strengths):
                raise ValueError(
                    "feature_indices and strengths must have the same length"
                )
            pairs = []
            for fid, s in zip(feature_indices, strengths):
                if fid < 0 or fid >= self.sae.hidden_dim:
                    raise ValueError(
                        f"feature_idx {fid} out of range [0, {self.sae.hidden_dim})"
                    )
                pairs.append(
                    (s * strength_scale, self.sae.decoder.weight[:, fid].detach())
                )
            layer_components = {self.target_layer: pairs}

        input_ids, attention_mask = self._build_prompt_ids(text, system_prompt)

        return self._generate_with_hooks(
            input_ids,
            attention_mask,
            layer_components,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
            clamp,
            seed,
        )

    def _generate_baseline(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
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
            repetition_penalty: CTRL-style repetition penalty (1.0 off).
            seed: Optional RNG seed for reproducible sampling.
            system_prompt: When set, render the prompt through the
                chat template (see :meth:`_build_prompt_ids`).

        Returns:
            The decoded text string (special tokens stripped).
        """
        input_ids, attention_mask = self._build_prompt_ids(text, system_prompt)

        base_model = self.model._module
        generated_tokens = input_ids.clone()
        device = generated_tokens.device
        generator = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None
            else None
        )

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
                next_token = _sample_next_token(
                    logits,
                    temperature,
                    top_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=generated_tokens,
                    generator=generator,
                )

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones(1, 1, device=device)], dim=-1
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
        if self.sae is None:
            raise TypeError(
                "find_steering_features requires a SparseAutoencoder source; "
                "this instance was built from SteeringVectors (no encoder)"
            )
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

    def build_token_feature_map(
        self,
        tokens: Optional[List[int]] = None,
        top_k_features: int = 10,
        min_activation: float = 1e-3,
        batch_size: int = 64,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Build a token-to-feature activation map.

        For each input token (or batch of tokens), this method passes it
        through the model at the target layer to capture MLP activations,
        then encodes those activations through the SAE encoder to determine
        which features are activated by each token.  The result is a
        dictionary mapping token IDs to their active SAE features and
        mean activation magnitudes.

        This provides a lookup table for understanding what semantic
        content each token is associated with in terms of learned SAE
        features — useful for interpreting which features respond to
        specific tokens without having to steer by them during generation.

        Args:
            tokens: List of token IDs to evaluate.  If ``None``, evaluates
                the first N tokens from the tokenizer's vocabulary
                (default ``min(1024, len(tokenizer) - num_special_tokens)``).
                Special/padding tokens are skipped.
            top_k_features: Maximum number of features per token to report
                when there are more than this many above the activation
                threshold.  Default ``10``.
            min_activation: Minimum mean activation value for a feature
                to be considered "active" by a token.  Features below this
                threshold are excluded from the mapping.  Default ``1e-3``.
            batch_size: Batch size for processing tokens through the model.
                Used when ``tokens`` is provided; otherwise tokens are
                processed individually.  Default ``64``.

        Returns:
            Dictionary mapping token IDs to dictionaries containing:

            - ``active_features`` (``List[int]``): Indices of features whose
              mean activation exceeds ``min_activation``, sorted by
              descending activation strength.
            - ``mean_activations`` (``Dict[int, float]``): Mean activation
              value per active feature.
            - ``decoder_norms`` (``Dict[int, float]``): L2 norm of the
              decoder weight vector for each active feature.

        Example:
            ```python
            steering = SAESteering(
                source=sae, model_name="google/gemma-2b", layer=5
            )
            token_map = steering.build_token_feature_map()

            # Check which features respond to a specific token
            for token_id in range(100):
                if token_id in token_map:
                    active = token_map[token_id]["active_features"]
                    logger.info(f"Token {token_id} activates features: {active[:5]}")

            # Find tokens that strongly activate feature 42
            matching_tokens = [
                tid for tid, info in token_map.items()
                if 42 in info["mean_activations"]
                and info["mean_activations"][42] > 0.1
            ]
            ```
        """
        if self.sae is None:
            raise TypeError(
                "build_token_feature_map requires a SparseAutoencoder source; "
                "this instance was built from SteeringVectors (no encoder)"
            )
        # Determine tokens to evaluate
        n_special = len(self.tokenizer.all_special_ids)
        vocab_size = self.tokenizer.vocab_size

        if tokens is None:
            eval_tokens = list(
                range(
                    n_special,
                    min(vocab_size - 1, n_special + max(1024, vocab_size // 2)),
                )
            )
        else:
            eval_tokens = [t for t in tokens if t not in self.tokenizer.all_special_ids]

        logger.info(f"Evaluating {len(eval_tokens)} tokens for feature mapping")
        self.sae.eval()
        device = self.device

        token_map: Dict[int, Dict[str, Any]] = {}

        # Use forward hook on the target MLP to capture activations at layer `target_layer`
        base_model = self.model._module
        target_module = resolve_module_path(
            base_model, f"model.layers[{self.target_layer}].mlp"
        )
        captured_activations: Optional[torch.Tensor] = None

        def activation_hook(module, input_args, output):
            nonlocal captured_activations
            if isinstance(output, torch.Tensor):
                if output.dim() == 3 and output.shape[1] == 1:
                    captured_activations = output.squeeze(1).clone()
                elif output.dim() == 2:
                    captured_activations = output.clone()

        hook_handle = target_module.register_forward_hook(activation_hook)
        try:
            # Process tokens in batches
            for i in range(0, len(eval_tokens), batch_size):
                batch_tokens = eval_tokens[i : i + batch_size]
                input_ids = torch.tensor([[t] for t in batch_tokens], device=device)

                with torch.no_grad():
                    _ = base_model(input_ids=input_ids)

                    if captured_activations is None:
                        logger.warning(
                            "No activations captured from hook — model may not produce 2D output"
                        )
                        continue

                # Encode through SAE encoder to get feature activations
                features = self.sae.encode(captured_activations).detach().cpu().numpy()

                for token_id, feat_row in zip(batch_tokens, features):
                    mean_act = float(np.mean(feat_row))
                    if mean_act >= min_activation:
                        active_features = sorted(
                            int(idx)
                            for idx, val in enumerate(feat_row)
                            if val > 0 and val > min_activation
                        )[:top_k_features]

                        token_map[token_id] = {
                            "active_features": active_features,
                            "mean_activations": {
                                int(fid): float(feat_row[fid])
                                for fid in active_features
                            },
                            "decoder_norms": {},
                        }

            logger.info(f"Built token→feature map with {len(token_map)} tokens")

        finally:
            hook_handle.remove()

        return token_map

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
        if self.sae is None:
            raise TypeError(
                "get_steering_direction requires a SparseAutoencoder source; "
                "this instance was built from SteeringVectors (no encoder)"
            )
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
