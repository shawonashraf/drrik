"""
Settings and environment configuration for the Drrik framework.

This module manages:
- HuggingFace Hub API tokens for gated models
- Weights & Biases (wandb) API keys for experiment tracking
- Other environment-based configuration
"""

import os
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from loguru import logger


class EnvironmentSettings(BaseSettings):
    """
    Environment settings for API keys and tokens.

    These settings are loaded from environment variables or a .env file.
    Create a .env file in the project root with your credentials.

    Example .env file:
        HUGGINGFACE_HUB_TOKEN=hf_...
        WANDB_API_KEY=...
        WANDB_PROJECT=drrik-experiments
        WANDB_ENTITY=your-username

    Attributes:
        huggingface_hub_token: HuggingFace Hub API token for gated models
        wandb_api_key: Weights & Biases API key for experiment tracking
        wandb_project: Default wandb project name
        wandb_entity: Default wandb entity (username or team)
        wandb_mode: wandb mode ('online', 'offline', or 'disabled')
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    huggingface_hub_token: Optional[str] = Field(
        default=None,
        description="HuggingFace Hub API token for accessing gated models. "
        "Get your token at: https://huggingface.co/settings/tokens",
    )

    wandb_api_key: Optional[str] = Field(
        default=None,
        description="Weights & Biases API key for experiment tracking. "
        "Get your key at: https://wandb.ai/settings",
    )

    wandb_project: str = Field(
        default="drrik-experiments", description="Default wandb project name"
    )

    wandb_entity: Optional[str] = Field(
        default=None, description="Default wandb entity (username or team name)"
    )

    wandb_mode: str = Field(
        default="online",
        description="wandb mode: 'online' to sync, 'offline' to save locally, "
        "'disabled' to disable wandb",
    )

    @field_validator("wandb_mode")
    @classmethod
    def validate_wandb_mode(cls, v: str) -> str:
        """Validate wandb mode."""
        valid_modes = ["online", "offline", "disabled"]
        v = v.lower()
        if v not in valid_modes:
            raise ValueError(f"wandb_mode must be one of {valid_modes}, got '{v}'")
        return v

    @field_validator("huggingface_hub_token")
    @classmethod
    def validate_hf_token(cls, v: Optional[str]) -> Optional[str]:
        """Log if HF token is provided."""
        if v:
            logger.info("HuggingFace Hub token is configured")
        return v

    @field_validator("wandb_api_key")
    @classmethod
    def validate_wandb_key(cls, v: Optional[str]) -> Optional[str]:
        """Log if wandb key is provided."""
        if v:
            logger.info("Weights & Biases API key is configured")
        return v

    @property
    def use_wandb(self) -> bool:
        """Check if wandb should be enabled based on API key and mode."""
        return self.wandb_api_key is not None and self.wandb_mode != "disabled"

    @property
    def has_hf_token(self) -> bool:
        """Check if HF token is available."""
        return self.huggingface_hub_token is not None

    def get_hf_auth(self) -> Optional[tuple]:
        """
        Get HuggingFace authentication tuple.

        Returns:
            Tuple of (token, ) or None if no token is set
        """
        if self.has_hf_token:
            return (True, self.huggingface_hub_token)
        return None


class WandbConfig:
    """
    Configuration for wandb experiment tracking.

    This class handles wandb initialization, logging, and cleanup.
    It's designed to be used as a context manager for automatic cleanup.

    Example:
        from drrik.settings import WandbConfig

        # Use as context manager
        with WandbConfig(
            project="my-sae-experiment",
            config={"model": "gemma-2b", "expansion": 8}
        ) as wandb_logger:
            wandb_logger.log_metrics({"loss": 0.5, "l0_norm": 10})

        # Or manually initialize/finalize
        wandb_config = WandbConfig(project="my-sae-experiment")
        wandb_config.initialize()
        wandb_config.log_metrics({"loss": 0.5})
        wandb_config.finalize()
    """

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        settings: Optional[EnvironmentSettings] = None,
        enabled: bool = True,
    ):
        """
        Initialize wandb configuration.

        Args:
            project: wandb project name (uses settings default if None)
            entity: wandb entity (uses settings default if None)
            name: Run name (auto-generated if None)
            config: Configuration dict to log
            tags: List of tags for the run
            settings: EnvironmentSettings instance (uses global if None)
            enabled: If False, disables wandb even if configured
        """
        self.settings = settings or EnvironmentSettings()
        self.enabled = enabled and self.settings.use_wandb

        if self.enabled and not self.settings.wandb_api_key:
            logger.warning(
                "wandb is enabled but no API key is set. "
                "Set WANDB_API_KEY environment variable or disable wandb."
            )
            self.enabled = False

        self.project = project or self.settings.wandb_project
        self.entity = entity or self.settings.wandb_entity
        self.name = name
        self.config = config or {}
        self.tags = tags

        self._initialized = False
        self._run = None

    def initialize(self) -> bool:
        """
        Initialize wandb run.

        Returns:
            True if wandb was successfully initialized, False otherwise
        """
        if not self.enabled:
            logger.info("wandb is disabled")
            return False

        if self._initialized:
            logger.warning("wandb is already initialized")
            return True

        try:
            import wandb

            # Set API key
            os.environ["WANDB_API_KEY"] = self.settings.wandb_api_key
            os.environ["WANDB_MODE"] = self.settings.wandb_mode

            # Initialize run
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config=self.config,
                tags=self.tags,
                reinit=True,  # Allow re-initialization
            )

            self._initialized = True
            logger.info(f"wandb initialized: {self.get_run_url()}")
            return True

        except ImportError:
            logger.warning(
                "wandb package not installed. " "Install it with: pip install wandb"
            )
            self.enabled = False
            return False
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            self.enabled = False
            return False

    def finalize(self) -> None:
        """Finalize wandb run."""
        if self._initialized:
            try:
                import wandb

                wandb.finish()
                logger.info("wandb run finalized")
            except Exception as e:
                logger.error(f"Error finalizing wandb: {e}")
            finally:
                self._initialized = False
                self._run = None

    def log_metrics(
        self,
        metrics: dict,
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names to values
            step: Current step (for logging to specific step)
            commit: Whether to commit the log
        """
        if self._initialized:
            try:
                import wandb

                wandb.log(metrics, step=step, commit=commit)
            except Exception as e:
                logger.error(f"Error logging to wandb: {e}")

    def log_histogram(
        self,
        values,
        name: str,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a histogram to wandb.

        Args:
            values: Array-like values to histogram
            name: Name of the histogram
            step: Current step
        """
        if self._initialized:
            try:
                import wandb
                import numpy as np

                wandb.log({name: wandb.Histogram(np.array(values))}, step=step)
            except Exception as e:
                logger.error(f"Error logging histogram to wandb: {e}")

    def log_model(self, model_path: str, name: str = "model") -> None:
        """
        Log a model artifact to wandb.

        Args:
            model_path: Path to the model file
            name: Artifact name
        """
        if self._initialized:
            try:
                import wandb

                artifact = wandb.Artifact(name, type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                logger.info(f"Logged model artifact: {name}")
            except Exception as e:
                logger.error(f"Error logging model to wandb: {e}")

    def get_run_url(self) -> Optional[str]:
        """Get the wandb run URL."""
        if self._run:
            return self._run.get_url()
        return None

    def get_run_id(self) -> Optional[str]:
        """Get the wandb run ID."""
        if self._run:
            return self._run.id
        return None

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()
        return False


# Global settings instance
_global_settings: Optional[EnvironmentSettings] = None


def get_settings() -> EnvironmentSettings:
    """
    Get the global EnvironmentSettings instance.

    Returns:
        The global settings (creates one if it doesn't exist)
    """
    global _global_settings
    if _global_settings is None:
        _global_settings = EnvironmentSettings()
    return _global_settings


def reload_settings() -> EnvironmentSettings:
    """
    Reload settings from environment variables and .env file.

    Returns:
        The reloaded settings
    """
    global _global_settings
    _global_settings = EnvironmentSettings()
    return _global_settings
