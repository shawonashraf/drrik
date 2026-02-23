"""
Configuration module for the Drrik framework.

This module provides Pydantic-based configuration classes for setting up
activation extraction and sparse autoencoder training.
"""

from typing import List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Configuration for loading a language model from HuggingFace Hub."""

    model_name: str = Field(
        default="google/gemma-2b",
        description="Name of the model on HuggingFace Hub (must be <3B for 8GB VRAM)"
    )
    revision: str = Field(
        default="main",
        description="Model revision to use"
    )
    torch_dtype: str = Field(
        default="float16",
        description="Data type for model weights (float16, bfloat16, float32)"
    )
    device_map: str = Field(
        default="auto",
        description="Device mapping strategy (auto, cpu, cuda)"
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code when loading the model"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Warn if model might be too large for 8GB VRAM."""
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
    """Configuration for loading a dataset from HuggingFace Hub."""

    dataset_name: str = Field(
        default="wikitext",
        description="Name of the dataset on HuggingFace Hub"
    )
    dataset_config: Optional[str] = Field(
        default="wikitext-2-raw-v1",
        description="Dataset configuration name"
    )
    split: str = Field(
        default="train",
        description="Dataset split to use (train, validation, test)"
    )
    max_samples: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of samples to process"
    )
    text_column: str = Field(
        default="text",
        description="Name of the column containing text data"
    )
    max_length: int = Field(
        default=512,
        ge=1,
        description="Maximum sequence length for tokenization"
    )


class ActivationExtractorConfig(BaseModel):
    """Configuration for MLP activation extraction using nnsight."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    mlp_layers: List[int] = Field(
        default_factory=lambda: [0],
        description="List of MLP layer indices to extract activations from"
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for inference"
    )
    output_dir: Optional[Path] = Field(
        default=None,
        description="Directory to save extracted activations (if None, don't save)"
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_output_dir(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        """Convert string path to Path object."""
        return Path(v) if isinstance(v, str) else v


class SparseAutoencoderConfig(BaseModel):
    """
    Configuration for training a Sparse Autoencoder.

    Based on the architecture from the "Towards Monosemanticity" paper.
    """

    activation_dim: int = Field(
        default=2048,
        ge=1,
        description="Dimension of the input MLP activations"
    )
    hidden_dim: int = Field(
        default=4096,  # 2x expansion by default
        ge=1,
        description="Dimension of the sparse hidden layer (overcomplete basis)"
    )
    l1_coefficient: float = Field(
        default=0.01,
        ge=0.0,
        description="L1 regularization coefficient for sparsity"
    )
    learning_rate: float = Field(
        default=1e-4,
        ge=0.0,
        description="Learning rate for Adam optimizer"
    )
    batch_size: int = Field(
        default=256,
        ge=1,
        description="Batch size for training"
    )
    num_epochs: int = Field(
        default=100,
        ge=1,
        description="Number of training epochs"
    )
    resample_dead_neurons: bool = Field(
        default=True,
        description="Whether to resample dead neurons during training"
    )
    resample_interval: int = Field(
        default=10000,
        ge=1,
        description="Steps between neuron resampling checks"
    )
    normalize_decoder: bool = Field(
        default=True,
        description="Whether to normalize decoder weights to unit norm"
    )
    pre_encoder_bias: bool = Field(
        default=True,
        description="Whether to use pre-encoder bias (as in the paper)"
    )


class VisualizationConfig(BaseModel):
    """Configuration for feature visualization."""

    output_dir: Path = Field(
        default=Path("./visualizations"),
        description="Directory to save visualization outputs"
    )
    dpi: int = Field(
        default=150,
        ge=50,
        description="DPI for saved figures"
    )
    style: str = Field(
        default="whitegrid",
        description="Seaborn style to use"
    )
    color_palette: str = Field(
        default="husl",
        description="Color palette for plots"
    )
    num_examples_per_feature: int = Field(
        default=10,
        ge=1,
        description="Number of top activating examples to show per feature"
    )


class Config(BaseSettings):
    """
    Main configuration class for the Drrik framework.

    This class combines all sub-configurations and can be loaded from
    environment variables or a config file.
    """

    extractor: ActivationExtractorConfig = Field(default_factory=ActivationExtractorConfig)
    autoencoder: SparseAutoencoderConfig = Field(default_factory=SparseAutoencoderConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    class Config:
        env_prefix = "DRIK_"
        env_nested_delimiter = "__"

    def create_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        if self.extractor.output_dir:
            self.extractor.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualization.output_dir.mkdir(parents=True, exist_ok=True)
