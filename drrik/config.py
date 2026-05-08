"""
Configuration module for the Drrik framework.

This module provides Pydantic-based configuration classes for setting up
activation extraction and sparse autoencoder training. All configuration
models use Pydantic v2 with validated fields and sensible defaults.

Configuration Hierarchy:
    - ``ModelConfig``: Language model loading parameters
    - ``DatasetConfig``: Dataset loading and preprocessing parameters
    - ``ActivationExtractorConfig``: Combines model, dataset, and extraction settings
    - ``SparseAutoencoderConfig``: SAE architecture and training hyperparameters
    - ``VisualizationConfig``: Plot generation settings
    - ``Config``: Top-level settings that aggregate all sub-configurations

Environment variables can override any setting using the ``DRIK_`` prefix
with ``__`` as nested delimiter (e.g., ``DRIK_RANDOM_SEED=123``).
"""

from typing import List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Configuration for loading a language model from HuggingFace Hub.

    Controls which model is loaded, its data type, device placement, and
    whether remote code execution is allowed during loading.

    Attributes:
        model_name: HuggingFace Hub model identifier. Should be under 3B
            parameters for machines with 8GB VRAM.
        revision: Git revision (branch, tag, or commit hash) to load.
        torch_dtype: Floating point precision for model weights. One of
            ``float16``, ``bfloat16``, or ``float32``.
        device_map: Strategy for distributing the model across devices.
            Use ``auto`` for automatic placement, ``cpu`` for CPU-only,
            or ``cuda`` for GPU.
        trust_remote_code: Whether to allow execution of custom code
            from the model repository. Required for some architectures.
    """

    model_name: str = Field(
        default="google/gemma-2b",
        description="Name of the model on HuggingFace Hub (must be <3B for 8GB VRAM)",
    )
    revision: str = Field(default="main", description="Model revision to use")
    torch_dtype: str = Field(
        default="float16",
        description="Data type for model weights (float16, bfloat16, float32)",
    )
    device_map: str = Field(
        default="auto", description="Device mapping strategy (auto, cpu, cuda)"
    )
    trust_remote_code: bool = Field(
        default=True, description="Whether to trust remote code when loading the model"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Warn if the model might be too large for 8GB VRAM.

        Checks the model name for common size indicators (7b, 13b, etc.)
        and issues a warning if the model is likely to exceed available
        memory on consumer GPUs.

        Args:
            v: The model name string to validate.

        Returns:
            The unmodified model name string.
        """
        large_models = ["7b", "13b", "30b", "70b", "mixtral", "llama-3"]
        v_lower = v.lower()
        if any(model in v_lower for model in large_models):
            import warnings

            warnings.warn(
                f"Model '{v}' may be too large for 8GB VRAM. "
                "Consider using a smaller model (<3B parameters)."
            )
        return v


class DatasetConfig(BaseModel):
    """Configuration for loading a dataset from HuggingFace Hub.

    Specifies which dataset to load, how many samples to process,
    and how to tokenize the text data.

    Attributes:
        dataset_name: HuggingFace Hub dataset identifier (e.g.,
            ``wikitext``, ``openwebtext``).
        dataset_config: Optional sub-configuration name for datasets
            with multiple variants (e.g., ``wikitext-2-raw-v1``).
        split: Dataset split to load (``train``, ``validation``, or
            ``test``).
        num_samples: Number of samples to process. Must be at least 1.
            This field has no default and **must** be explicitly provided.
        text_column: Name of the column containing the text data.
        max_length: Maximum token sequence length. Longer sequences
            are truncated.
    """

    dataset_name: str = Field(
        default="wikitext", description="Name of the dataset on HuggingFace Hub"
    )
    dataset_config: Optional[str] = Field(
        default="wikitext-2-raw-v1", description="Dataset configuration name"
    )
    split: str = Field(
        default="train", description="Dataset split to use (train, validation, test)"
    )
    num_samples: int = Field(ge=1, description="Number of samples to process")
    text_column: str = Field(
        default="text", description="Name of the column containing text data"
    )
    max_length: int = Field(
        default=512, ge=1, description="Maximum sequence length for tokenization"
    )


class ActivationExtractorConfig(BaseModel):
    """Configuration for MLP activation extraction using nnsight.

    Combines model and dataset configurations with extraction-specific
    parameters such as which MLP layers to target and batch size for
    inference.

    Attributes:
        model: Model configuration. If ``None``, must be provided via
            kwargs when creating an ``ActivationExtractor``.
        dataset: Dataset configuration. If ``None``, must be provided
            via kwargs when creating an ``ActivationExtractor``.
        mlp_layers: List of zero-indexed MLP layer indices to extract
            activations from. Defaults to ``[0]`` (first layer only).
        batch_size: Number of samples processed per inference batch.
            Lower values reduce memory usage.
        output_dir: Directory to save extracted activations as ``.npy``
            files. If ``None``, activations are not automatically saved.
    """

    model: Optional[ModelConfig] = Field(default=None)
    dataset: Optional[DatasetConfig] = Field(default=None)

    mlp_layers: List[int] = Field(
        default_factory=lambda: [0],
        description="List of MLP layer indices to extract activations from",
    )
    batch_size: int = Field(default=8, ge=1, description="Batch size for inference")
    output_dir: Optional[Path] = Field(
        default=None,
        description="Directory to save extracted activations (if None, don't save)",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_output_dir(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        """Convert a string path to a Path object.

        Ensures that ``output_dir`` is always stored as a ``Path``
        instance, even when loaded from a YAML config that provides
        a plain string.

        Args:
            v: The output directory value (string, Path, or None).

        Returns:
            ``Path`` instance if a string was provided, otherwise
            the original value.
        """
        return Path(v) if isinstance(v, str) else v


class SparseAutoencoderConfig(BaseModel):
    """Configuration for training a Sparse Autoencoder.

    Based on the architecture from Anthropic's ``Towards Monosemanticity``
    paper. The SAE learns an overcomplete dictionary of features from
    MLP activations, with L1 regularization to encourage sparsity.

    Attributes:
        activation_dim: Dimensionality of the input MLP activations.
            Should match the hidden size of the transformer model.
        hidden_dim: Dimensionality of the sparse hidden layer. An
            overcomplete representation (e.g., 8x expansion) allows
            the SAE to learn more features than the original space.
        l1_coefficient: Strength of L1 regularization. Higher values
            produce sparser representations but may reduce reconstruction
            quality.
        learning_rate: Learning rate for the Adam optimizer.
        batch_size: Number of samples per training batch.
        num_epochs: Total number of training epochs.
        resample_dead_neurons: Whether to periodically resample neurons
            that stop firing during training. This helps prevent feature
            death, especially with high sparsity.
        resample_interval: Number of training steps between dead neuron
            resampling checks.
        normalize_decoder: Whether to normalize decoder weight columns
            to unit norm after each optimizer step. Prevents scaling
            collapse as described in the paper.
        pre_encoder_bias: Whether to include a learnable bias that is
            subtracted before encoding and added back after decoding.
            Initialized to the median of the training data.
    """

    activation_dim: int = Field(
        default=2048, ge=1, description="Dimension of the input MLP activations"
    )
    hidden_dim: int = Field(
        default=4096,  # 2x expansion by default
        ge=1,
        description="Dimension of the sparse hidden layer (overcomplete basis)",
    )
    l1_coefficient: float = Field(
        default=0.01, ge=0.0, description="L1 regularization coefficient for sparsity"
    )
    learning_rate: float = Field(
        default=1e-4, ge=0.0, description="Learning rate for Adam optimizer"
    )
    batch_size: int = Field(default=256, ge=1, description="Batch size for training")
    num_epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    resample_dead_neurons: bool = Field(
        default=True, description="Whether to resample dead neurons during training"
    )
    resample_interval: int = Field(
        default=10000, ge=1, description="Steps between neuron resampling checks"
    )
    normalize_decoder: bool = Field(
        default=True, description="Whether to normalize decoder weights to unit norm"
    )
    pre_encoder_bias: bool = Field(
        default=True, description="Whether to use pre-encoder bias (as in the paper)"
    )


class VisualizationConfig(BaseModel):
    """Configuration for feature visualization.

    Controls the appearance and output settings for all plots generated
    by ``FeatureVisualizer``, including density histograms, activation
    histograms, and feature dashboards.

    Attributes:
        output_dir: Directory where visualization files are saved.
            Created automatically if it does not exist.
        dpi: Resolution (dots per inch) for saved figure images.
            Higher values produce sharper images but larger files.
        style: Seaborn plot style. Common options include
            ``whitegrid``, ``darkgrid``, ``white``, ``dark``, ``ticks``.
        color_palette: Seaborn color palette name or matplotlib
            colormap for plot coloring.
        num_examples_per_feature: Number of top-activating examples
            shown per feature in detailed plots.
    """

    output_dir: Path = Field(
        default=Path("./visualizations"),
        description="Directory to save visualization outputs",
    )
    dpi: int = Field(default=150, ge=50, description="DPI for saved figures")
    style: str = Field(default="whitegrid", description="Seaborn style to use")
    color_palette: str = Field(default="husl", description="Color palette for plots")
    num_examples_per_feature: int = Field(
        default=10,
        ge=1,
        description="Number of top activating examples to show per feature",
    )


class Config(BaseSettings):
    """Main configuration class for the Drrik framework.

    Aggregates all sub-configurations (extractor, autoencoder, visualization)
    into a single object. Settings can be loaded from environment variables
    (with ``DRIK_`` prefix) or from a YAML config file via the CLI.

    Attributes:
        extractor: Activation extraction configuration.
        autoencoder: Sparse autoencoder configuration.
        visualization: Feature visualization configuration.
        random_seed: Global random seed for reproducibility. Applied
            to numpy, torch, and random at pipeline start.
        log_level: Logging verbosity. One of ``DEBUG``, ``INFO``,
            ``WARNING``, ``ERROR``, ``CRITICAL``.

    Example:
        Load from environment variables:

        ```bash
        export DRIK_RANDOM_SEED=42
        export DRIK_AUTOENCODER__HIDDEN_DIM=16384
        config = Config()
        ```

        Or construct programmatically:

        ```python
        config = Config(
            extractor=ActivationExtractorConfig(...),
            autoencoder=SparseAutoencoderConfig(activation_dim=2048),
        )
        ```
    """

    extractor: ActivationExtractorConfig = Field(
        default_factory=ActivationExtractorConfig
    )
    autoencoder: SparseAutoencoderConfig = Field(
        default_factory=SparseAutoencoderConfig
    )
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    model_config = ConfigDict(
        env_prefix="DRIK_",
        env_nested_delimiter="__",
    )

    def create_output_dirs(self) -> None:
        """Create output directories if they don't exist.

        Ensures that both the extractor output directory (if configured)
        and the visualization output directory exist on disk, creating
        any missing parent directories as needed.
        """
        if self.extractor.output_dir:
            self.extractor.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualization.output_dir.mkdir(parents=True, exist_ok=True)
