"""
Model and activation extraction module using nnsight.

This module provides the ``ActivationExtractor`` class which orchestrates
the full activation extraction pipeline:

1. **Model Loading**: Loads a HuggingFace language model via nnsight's
   ``LanguageModel`` wrapper, supporting gated models with authentication
   via ``HUGGINGFACE_HUB_TOKEN`` (loaded through :func:`drrik.settings.get_settings`).
2. **Dataset Loading**: Loads and tokenizes a HuggingFace dataset for
   inference.
3. **Activation Extraction**: Runs forward passes through the model,
   capturing MLP layer outputs at specified layers using nnsight's
   tracing context. Module paths are resolved with the shared
   :func:`~drrik.steering.resolve_module_path` utility.
4. **Persistence**: Saves and loads extracted activations as ``.npy``
   files with accompanying ``.pkl`` metadata.

Supported Architectures:
    The module auto-detects MLP layer paths for common transformer
    architectures including Gemma, Llama, Phi, GPT-2, Pythia, and
    BERT. Unknown architectures fall back to common naming patterns.
"""

from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
import pickle

from drrik.steering import resolve_module_path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from nnsight import LanguageModel

from loguru import logger

from drrik.config import ActivationExtractorConfig, ModelConfig, DatasetConfig
from drrik.settings import get_settings


class ActivationExtractor:
    """
    Extract MLP activations from language models using nnsight.

    Handles model and dataset loading from HuggingFace Hub, runs
    inference with nnsight's tracing context to capture MLP outputs,
    and provides persistence via ``.npy`` / ``.pkl`` files.

    The HuggingFace auth token is read automatically from the
    ``HUGGINGFACE_HUB_TOKEN`` environment variable (or ``.env`` file)
    via :func:`~drrik.settings.get_settings`.

    Attributes:
        config: The resolved :class:`~drrik.config.ActivationExtractorConfig`.
        model: The loaded nnsight ``LanguageModel`` (``None`` until
            :meth:`load_model` is called).
        tokenizer: The HuggingFace tokenizer (``None`` until
            :meth:`load_model` is called).
        dataset: The loaded HuggingFace ``Dataset`` (``None`` until
            :meth:`load_dataset` is called).

    Example:
        ```python
        extractor = ActivationExtractor(
            model_name="google/gemma-2b",
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            mlp_layers=[0],
            num_samples=1000,
        )
        activations, metadata = extractor.extract()
        extractor.save_activations(activations, metadata, "./output")
        ```
    """

    def __init__(self, config: Optional[ActivationExtractorConfig] = None, **kwargs):
        """
        Initialize the ActivationExtractor.

        Accepts either a pre-built config object or flat keyword
        arguments that are automatically sorted into
        :class:`~drrik.config.ModelConfig`,
        :class:`~drrik.config.DatasetConfig`, and the remaining
        extraction-level settings.

        Args:
            config: A fully constructed
                :class:`~drrik.config.ActivationExtractorConfig`.
                If ``None``, the config is built from ``**kwargs``.
            **kwargs: Flat config overrides. Recognised keys are
                distributed to the appropriate sub-config:

                - *Model*: ``model_name``, ``revision``,
                  ``torch_dtype``, ``device_map``,
                  ``trust_remote_code``
                - *Dataset*: ``dataset_name``, ``dataset_config``,
                  ``split``, ``num_samples``, ``text_column``,
                  ``max_length``
                - *Extraction*: ``mlp_layers``, ``batch_size``,
                  ``output_dir``
        """
        if config is None:
            model_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "model_name",
                    "revision",
                    "torch_dtype",
                    "device_map",
                    "trust_remote_code",
                )
            }
            dataset_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "dataset_name",
                    "dataset_config",
                    "split",
                    "num_samples",
                    "text_column",
                    "max_length",
                )
            }
            remaining_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in model_kwargs and k not in dataset_kwargs
            }

            model = ModelConfig(**model_kwargs) if model_kwargs else None
            dataset = DatasetConfig(**dataset_kwargs) if dataset_kwargs else None

            config = ActivationExtractorConfig(
                model=model,
                dataset=dataset,
                **remaining_kwargs,
            )

        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self._activations = []
        self._metadata = []

    def load_model(self) -> LanguageModel:
        """
        Load the model from HuggingFace Hub using nnsight.

        If the model is already loaded, returns the cached instance.
        The HuggingFace token is read from ``HUGGINGFACE_HUB_TOKEN``
        via :func:`~drrik.settings.get_settings`.

        Returns:
            The loaded nnsight :class:`~nnsight.LanguageModel` wrapper.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self.model is not None:
            return self.model

        try:
            logger.info(f"Loading model: {self.config.model.model_name}")

            # Get HF token from settings
            settings = get_settings()
            hf_token = settings.huggingface_hub_token

            if hf_token:
                logger.info("Using HuggingFace Hub token for authentication")

            # Load tokenizer with token
            tokenizer_kwargs = {
                "revision": self.config.model.revision,
                "trust_remote_code": self.config.model.trust_remote_code,
            }
            if hf_token:
                tokenizer_kwargs["token"] = hf_token

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name, **tokenizer_kwargs
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.config.model.torch_dtype, torch.float16)

            # Load model with nnsight (with token if available)
            nnsight_kwargs = {
                "revision": self.config.model.revision,
                "torch_dtype": torch_dtype,
                "trust_remote_code": self.config.model.trust_remote_code,
                "device_map": self.config.model.device_map,
            }
            if hf_token:
                nnsight_kwargs["token"] = hf_token

            self.model = LanguageModel(self.config.model.model_name, **nnsight_kwargs)

            logger.info(f"Model loaded successfully on {self.model.device}")
            return self.model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def load_dataset(self) -> Dataset:
        """
        Load the dataset from HuggingFace Hub.

        If the dataset is already loaded, returns the cached instance.
        The HuggingFace token is forwarded for gated datasets when
        available.

        Returns:
            The loaded HuggingFace :class:`~datasets.Dataset`.

        Raises:
            RuntimeError: If dataset loading fails.
        """
        if self.dataset is not None:
            return self.dataset

        try:
            logger.info(
                f"Loading dataset: {self.config.dataset.dataset_name} "
                f"({self.config.dataset.split} split)"
            )

            # Get HF token from settings (for gated datasets)
            settings = get_settings()
            hf_token = settings.huggingface_hub_token

            # Load dataset with token if available
            load_kwargs = {
                "path": self.config.dataset.dataset_name,
                "name": self.config.dataset.dataset_config,
                "split": self.config.dataset.split,
            }
            if hf_token:
                load_kwargs["token"] = hf_token

            self.dataset = load_dataset(**load_kwargs)

            logger.info(f"Dataset loaded with {len(self.dataset)} examples")
            return self.dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Dataset loading failed: {e}") from e

    def _get_mlp_layer_name(self, layer_idx: int) -> str:
        """
        Get the nnsight module path to an MLP layer.

        Auto-detects the correct path pattern based on the model name
        in the config.  For unknown architectures a warning is logged
        and the default Gemma/Llama pattern is returned.

        Args:
            layer_idx: Zero-indexed transformer layer number.

        Returns:
            A dotted/bracket module path string, e.g.
            ``"model.layers[3].mlp"``.
        """
        model_name_lower = self.config.model.model_name.lower()

        # Gemma/GPT-style: model.layers.N.mlp
        if any(name in model_name_lower for name in ["gemma", "gpt-2", "pythia"]):
            return f"model.layers[{layer_idx}].mlp"

        # Llama-style: model.layers.N.mlp
        if "llama" in model_name_lower:
            return f"model.layers[{layer_idx}].mlp"

        # Phi-style: model.layers.N.mlp
        if "phi" in model_name_lower:
            return f"model.layers[{layer_idx}].mllp"  # Note: mlp can be 'mlp' or 'mlp'

        # BERT-style: bert.encoder.layer.N.output.dense
        if "bert" in model_name_lower:
            return f"bert.encoder.layer[{layer_idx}].output"

        # Default: try common patterns
        common_patterns = [
            f"model.layers[{layer_idx}].mlp",
            f"model.layers[{layer_idx}].ffn",
            f"transformer.h[{layer_idx}].mlp",
            f"layers[{layer_idx}].mlp",
        ]

        logger.warning(
            f"Unknown model architecture '{self.config.model.model_name}', "
            f"trying common patterns"
        )
        return common_patterns[0]

    def _resolve_layer_path(self, path: str):
        """Resolve a dotted/bracket path to the actual nnsight module.

        Delegates to the shared :func:`~drrik.steering.resolve_module_path`
        utility so that path-resolution logic is maintained in one place.

        Args:
            path: A dot-separated module path with optional bracket
                indexing, e.g., ``model.layers[2].mlp``.

        Returns:
            The resolved nnsight module object corresponding to the
            given path.
        """
        return resolve_module_path(self.model, path)

    def extract(
        self,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract MLP activations from the model.

        Loads the model and dataset (if not already loaded), tokenizes
        the data, and runs batched forward passes using nnsight's
        tracing context.  For each sample, the **last-token** activation
        from each requested MLP layer is collected and concatenated.

        Args:
            num_samples: Override for the number of samples to process.
                If ``None``, uses ``config.dataset.num_samples``.

        Returns:
            A tuple of:

            - **activations** (:class:`numpy.ndarray`) — shape
              ``(n_samples, activation_dim * n_layers)``.  When a
              single layer is requested the shape simplifies to
              ``(n_samples, activation_dim)``.
            - **metadata** (:class:`dict`) — contains the serialised
              config, sample count, activation dimension, layer paths,
              and per-sample metadata (index, truncated text, input
              ids).

        Raises:
            RuntimeError: If extraction fails at any stage.
        """
        try:
            # Load model and dataset
            self.load_model()
            self.load_dataset()

            n_samples = num_samples or self.config.dataset.num_samples
            logger.info(
                f"Extracting activations from {len(self.config.mlp_layers)} MLP layers "
                f"for {n_samples} samples"
            )

            # Prepare dataset
            dataset = self.dataset.select(range(min(n_samples, len(self.dataset))))

            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples[self.config.dataset.text_column],
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.dataset.max_length,
                    return_tensors="pt",
                )

            tokenized = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing",
            )

            # Get MLP layer names
            layer_paths = [
                self._get_mlp_layer_name(layer_idx)
                for layer_idx in self.config.mlp_layers
            ]

            logger.info(f"Extracting from layers: {layer_paths}")

            # Collect activations using nnsight
            self._activations = []
            self._metadata = []

            batch_size = self.config.batch_size
            n_batches = (len(tokenized) + batch_size - 1) // batch_size

            with torch.no_grad():
                for batch_idx in tqdm(range(n_batches), desc="Extracting activations"):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(tokenized))

                    batch = tokenized[start_idx:end_idx]
                    input_ids = torch.tensor(batch["input_ids"])
                    attention_mask = torch.tensor(batch["attention_mask"])

                    layer_outputs = []

                    # Use nnsight to extract activations
                    with self.model.trace(input_ids, attention_mask=attention_mask):
                        for layer_path in layer_paths:
                            module = self._resolve_layer_path(layer_path)
                            output = module.output.save()
                            layer_outputs.append(output)

                    # Process outputs
                    batch_input_ids = input_ids.cpu().numpy()
                    for sample_idx in range(len(input_ids)):
                        sample_activations = []
                        for layer_output in layer_outputs:
                            # Get activation after non-linearity (if applicable)
                            # Shape: (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
                            act = layer_output

                            if act.dim() == 3:
                                act = act[sample_idx]  # (seq_len, hidden_dim)
                            elif act.dim() == 2:
                                act = act  # Already (seq_len, hidden_dim)

                            # Use the last token's activation (common practice)
                            act = act[-1]  # (hidden_dim,)

                            sample_activations.append(act.cpu().numpy())

                        # Concatenate activations from all layers
                        self._activations.append(np.concatenate(sample_activations))

                        # Store metadata
                        self._metadata.append(
                            {
                                "sample_idx": start_idx + sample_idx,
                                "text": dataset[start_idx + sample_idx][
                                    self.config.dataset.text_column
                                ][:200],
                                "input_ids": batch_input_ids[sample_idx],
                            }
                        )

            activations = np.array(self._activations)
            logger.info(f"Extracted activations shape: {activations.shape}")

            metadata = {
                "config": self.config.model_dump(),
                "n_samples": len(self._activations),
                "activation_dim": activations.shape[-1],
                "layer_paths": layer_paths,
                "samples_metadata": self._metadata,
            }

            return activations, metadata

        except Exception as e:
            logger.error(f"Failed to extract activations: {e}")
            raise RuntimeError(f"Activation extraction failed: {e}") from e

    def save_activations(
        self,
        activations: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Save extracted activations to disk.

        Writes two files to *output_dir*:

        - ``activations.npy`` — the activation array.
        - ``metadata.pkl`` — the metadata dictionary.

        Args:
            activations: Activation array of shape
                ``(n_samples, dim)``.
            metadata: Metadata dictionary (as returned by
                :meth:`extract`).
            output_dir: Target directory.  If ``None``, uses
                ``config.output_dir``.

        Returns:
            Path to the saved ``activations.npy`` file.

        Raises:
            ValueError: If *output_dir* is ``None`` and no output
                directory is configured.
        """
        if output_dir is None:
            output_dir = self.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        activations_path = output_dir / "activations.npy"
        np.save(str(activations_path), activations)

        metadata_path = output_dir / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved activations to {activations_path}")
        return activations_path

    def load_activations(
        self,
        filepath: Union[str, Path],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load saved activations from disk.

        Supports two formats:

        - ``.npy`` — loads the array directly and looks for a
          companion ``metadata.pkl`` in the same directory.
        - ``.pkl`` (legacy) — loads a dict with ``"activations"``
          and ``"metadata"`` keys.

        Args:
            filepath: Path to an ``.npy`` or ``.pkl`` file.

        Returns:
            A tuple of (activations array, metadata dict).

        Raises:
            FileNotFoundError: If the file (or companion metadata) does
                not exist.
        """
        filepath = Path(filepath)

        if filepath.suffix == ".npy":
            activations = np.load(str(filepath))
            metadata_path = filepath.parent / "metadata.pkl"
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            activations = data["activations"]
            metadata = data["metadata"]

        logger.info(f"Loaded activations from {filepath}")
        return activations, metadata
