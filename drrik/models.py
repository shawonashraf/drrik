"""
Model and activation extraction module using nnsight.

This module provides functionality to:
1. Load language models from HuggingFace Hub
2. Load datasets from HuggingFace Hub
3. Extract MLP activations using the nnsight library
"""

from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
import pickle

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from nnsight import NNsight

from loguru import logger

from drrik.config import ActivationExtractorConfig
from drrik.settings import get_settings


class ActivationExtractor:
    """
    Extract MLP activations from language models using nnsight.

    This class handles:
    - Loading models from HuggingFace Hub
    - Loading datasets from HuggingFace Hub
    - Running inference and collecting MLP layer activations
    - Saving/loading extracted activations

    Example:
        extractor = ActivationExtractor(
            model_name="google/gemma-2b",
            dataset_name="wikitext",
            mlp_layers=[0],
            num_samples=1000
        )
        activations, metadata = extractor.extract()
    """

    def __init__(self, config: Optional[ActivationExtractorConfig] = None, **kwargs):
        """
        Initialize the ActivationExtractor.

        Args:
            config: Configuration object. If None, uses defaults.
            **kwargs: Additional config overrides (e.g., model_name="gpt2")
        """
        if config is None:
            config = ActivationExtractorConfig()

        # Apply any keyword argument overrides
        if kwargs:
            config = config.model_copy(update=kwargs)

        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self._activations = []
        self._metadata = []

    def load_model(self) -> NNsight:
        """
        Load the model from HuggingFace Hub using nnsight.

        Returns:
            The loaded nnsight model wrapper

        Raises:
            RuntimeError: If model loading fails
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

            self.model = NNsight(self.config.model.model_name, **nnsight_kwargs)

            logger.info(f"Model loaded successfully on {self.model.device}")
            return self.model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def load_dataset(self) -> Dataset:
        """
        Load the dataset from HuggingFace Hub.

        Returns:
            The loaded dataset

        Raises:
            RuntimeError: If dataset loading fails
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
        Get the nnsight path to an MLP layer.

        Different model architectures have different naming conventions.
        This method attempts to find the correct path for common architectures.

        Args:
            layer_idx: The layer index

        Returns:
            The module path string for nnsight
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

    def extract(
        self,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract MLP activations from the model.

        This method:
        1. Loads the model and dataset
        2. Tokenizes the dataset
        3. Runs inference with nnsight to collect MLP activations
        4. Returns the activations and metadata

        Args:
            num_samples: Override for the number of samples to process

        Returns:
            Tuple of (activations array, metadata dict)

        Raises:
            RuntimeError: If extraction fails
        """
        try:
            # Load model and dataset
            self.load_model()
            self.load_dataset()

            n_samples = num_samples or self.config.dataset.max_samples
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
                    input_ids = torch.stack(batch["input_ids"])
                    attention_mask = torch.stack(batch["attention_mask"])

                    # Use nnsight to extract activations
                    with self.model.trace(
                        input_ids, attention_mask=attention_mask
                    ) as tracer:
                        # Collect outputs from each MLP layer
                        layer_outputs = []
                        for layer_path in layer_paths:
                            output = tracer[layer_path].output.save()
                            layer_outputs.append(output)

                        # Run the model
                        tracer.invoke(input_ids, attention_mask=attention_mask)

                    # Process outputs
                    batch_input_ids = input_ids.cpu().numpy()
                    for sample_idx in range(len(input_ids)):
                        sample_activations = []
                        for layer_output in layer_outputs:
                            # Get activation after non-linearity (if applicable)
                            # Shape: (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
                            act = layer_output.value

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
        filepath: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Save extracted activations to disk.

        Args:
            activations: The activations array
            metadata: Metadata dictionary
            filepath: Path to save to. If None, uses config output_dir

        Returns:
            Path where activations were saved
        """
        if filepath is None:
            filepath = self.config.output_dir / "activations.pkl"

        filepath = Path(filepath)

        data = {"activations": activations, "metadata": metadata}

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved activations to {filepath}")
        return filepath

    def load_activations(
        self,
        filepath: Union[str, Path],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load saved activations from disk.

        Args:
            filepath: Path to load from

        Returns:
            Tuple of (activations array, metadata dict)
        """
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        logger.info(f"Loaded activations from {filepath}")
        return data["activations"], data["metadata"]
