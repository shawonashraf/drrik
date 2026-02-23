"""
Visualization module for sparse autoencoder features.

This module provides tools for visualizing:
- Feature densities
- Top activating examples for each feature
- Activation histograms
- Decoder weight distributions
- Training curves
"""

from typing import Optional, List, Union, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from loguru import logger

from drrik.autoencoder import SparseAutoencoder
from drrik.settings import WandbConfig


class FeatureVisualizer:
    """
    Visualize sparse autoencoder features and activations.

    This class provides methods to create various plots that help understand
    the learned features, similar to the visualizations in the
    "Towards Monosemanticity" paper.

    Example:
        visualizer = FeatureVisualizer(sae, activations, metadata)
        visualizer.plot_feature_density()
        visualizer.plot_top_features(n_features=10)
        visualizer.plot_feature_examples(feature_idx=0)
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        activations: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path] = "./visualizations",
        style: str = "whitegrid",
        dpi: int = 150,
        wandb_config: Optional[WandbConfig] = None,
        log_to_wandb: bool = True,
    ):
        """
        Initialize the visualizer.

        Args:
            sae: Trained sparse autoencoder
            activations: The MLP activations used to train the SAE
            metadata: Optional metadata dictionary (e.g., from ActivationExtractor)
            output_dir: Directory to save plots
            style: Seaborn style to use
            dpi: DPI for saved figures
            wandb_config: Optional WandbConfig for logging visualizations
            log_to_wandb: If True, log plots to wandb when wandb_config is provided
        """
        self.sae = sae
        self.activations = activations
        self.metadata = metadata or {}
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.wandb_config = wandb_config
        self.log_to_wandb = log_to_wandb

        # Set style
        sns.set_style(style)
        sns.set_palette("husl")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compute feature activations
        self._compute_features()

    def _log_figure_to_wandb(self, fig: plt.Figure, name: str) -> None:
        """
        Log a matplotlib figure to wandb.

        Args:
            fig: The matplotlib figure
            name: Name for the plot in wandb
        """
        if self.wandb_config and self.log_to_wandb:
            try:
                import wandb

                wandb.log({name: wandb.Image(fig)})
                logger.debug(f"Logged {name} to wandb")
            except Exception as e:
                logger.warning(f"Failed to log {name} to wandb: {e}")

    def _compute_features(self) -> None:
        """Compute sparse feature activations from the SAE."""
        import torch

        self.sae.eval()
        with torch.no_grad():
            activations_tensor = torch.from_numpy(self.activations).float()
            self.features = self.sae.encode(activations_tensor).cpu().numpy()

        logger.info(f"Computed features with shape: {self.features.shape}")

    def plot_feature_density(
        self,
        bins: int = 50,
        log_scale: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the distribution of feature densities.

        Feature density is the fraction of examples on which a feature fires.
        This histogram helps understand the sparsity distribution.

        Args:
            bins: Number of histogram bins
            log_scale: Whether to use log scale on x-axis
            save_path: Optional path to save the figure

        Returns:
            The matplotlib Figure object
        """
        # Compute feature densities
        densities = (self.features > 0).mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Remove zero-density features (dead neurons)
        active_densities = densities[densities > 0]

        # Plot histogram on log scale
        if log_scale:
            bins_log = np.logspace(-8, 0, bins)
            ax.hist(active_densities, bins=bins_log, edgecolor="black", alpha=0.7)
            ax.set_xscale("log")
        else:
            ax.hist(active_densities, bins=bins, edgecolor="black", alpha=0.7)

        ax.set_xlabel("Feature Density (fraction of examples)", fontsize=12)
        ax.set_ylabel("Number of Features", fontsize=12)
        ax.set_title("Distribution of Feature Densities", fontsize=14)

        # Add statistics
        n_dead = (densities == 0).sum()
        n_active = len(active_densities)

        stats_text = f"Dead features: {n_dead} ({n_dead/len(densities)*100:.1f}%)\n"
        stats_text += (
            f"Active features: {n_active} ({n_active/len(densities)*100:.1f}%)\n"
        )
        stats_text += f"Median density: {np.median(active_densities):.2e}"

        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path or self.output_dir:
            path = save_path or self.output_dir / "feature_density.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved feature density plot to {path}")

        # Log to wandb
        self._log_figure_to_wandb(fig, "feature_density")

        return fig

    def plot_activation_histogram(
        self,
        feature_idx: int,
        bins: int = 100,
        log_y: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot activation histogram for a specific feature.

        Args:
            feature_idx: Index of the feature to visualize
            bins: Number of histogram bins
            log_y: Whether to use log scale on y-axis
            save_path: Optional path to save the figure

        Returns:
            The matplotlib Figure object
        """
        feature_acts = self.features[:, feature_idx]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        counts, bin_edges, patches = ax.hist(
            feature_acts[feature_acts > 0],  # Only non-zero activations
            bins=bins,
            edgecolor="black",
            alpha=0.7,
        )

        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel("Activation Value", fontsize=12)
        ax.set_ylabel("Count (log scale)" if log_y else "Count", fontsize=12)
        ax.set_title(f"Activation Histogram for Feature {feature_idx}", fontsize=14)

        # Add statistics
        density = (feature_acts > 0).mean()
        max_act = feature_acts.max()
        mean_act = feature_acts[feature_acts > 0].mean() if density > 0 else 0

        stats_text = f"Density: {density:.2e}\n"
        stats_text += f"Max activation: {max_act:.4f}\n"
        stats_text += f"Mean (active): {mean_act:.4f}"

        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        plt.tight_layout()

        if save_path or self.output_dir:
            path = (
                save_path
                or self.output_dir / f"activation_histogram_feature_{feature_idx}.png"
            )
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved activation histogram to {path}")

        return fig

    def plot_training_curves(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot training loss and L0 norm curves.

        Args:
            save_path: Optional path to save the figure

        Returns:
            The matplotlib Figure object
        """
        if not self.sae.training_losses:
            logger.warning("No training data available")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.sae.training_losses) + 1)

        # Loss curve
        ax1.plot(epochs, self.sae.training_losses, "b-", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training Loss", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # L0 norm curve
        ax2.plot(epochs, self.sae.training_l0_norms, "r-", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("L0 Norm (avg # active features)", fontsize=12)
        ax2.set_title("Sparsity (L0 Norm)", fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path or self.output_dir:
            path = save_path or self.output_dir / "training_curves.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved training curves to {path}")

        return fig

    def plot_top_features(
        self,
        n_features: int = 10,
        by: str = "density",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot top N features by various metrics.

        Args:
            n_features: Number of top features to show
            by: Metric to sort by ("density", "max_activation", "mean_activation")
            save_path: Optional path to save the figure

        Returns:
            The matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Compute metric for each feature
        if by == "density":
            values = (self.features > 0).mean(axis=0)
            ylabel = "Feature Density"
            title = f"Top {n_features} Features by Density"
        elif by == "max_activation":
            values = self.features.max(axis=0)
            ylabel = "Max Activation"
            title = f"Top {n_features} Features by Max Activation"
        elif by == "mean_activation":
            values = self.features.mean(axis=0)
            ylabel = "Mean Activation"
            title = f"Top {n_features} Features by Mean Activation"
        else:
            raise ValueError(f"Unknown metric: {by}")

        # Get top features
        top_indices = np.argsort(values)[-n_features:][::-1]
        top_values = values[top_indices]

        # Plot bar chart
        _ = ax.barh(range(n_features), top_values[::-1])
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([f"Feature {i}" for i in top_indices[::-1]])
        ax.set_xlabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        if save_path or self.output_dir:
            path = save_path or self.output_dir / f"top_features_{by}.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved top features plot to {path}")

        return fig

    def plot_feature_examples(
        self,
        feature_idx: int,
        k: int = 10,
        show_text: bool = True,
        max_text_length: int = 200,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot top k activating examples for a specific feature.

        Args:
            feature_idx: Index of the feature to visualize
            k: Number of examples to show
            show_text: Whether to show the text examples
            max_text_length: Maximum length of text to display
            save_path: Optional path to save the figure

        Returns:
            The matplotlib Figure object
        """
        # Get top activating examples
        top_values, top_indices = self.sae.get_top_activating_examples(
            self.activations, feature_idx, k
        )

        fig, ax = plt.subplots(figsize=(12, k * 0.5))

        # Plot bar chart
        _ = ax.barh(range(k), top_values[::-1])
        ax.set_yticks(range(k))
        ax.set_xlabel("Activation Value", fontsize=12)
        ax.set_title(
            f"Top {k} Activating Examples for Feature {feature_idx}", fontsize=14
        )
        ax.grid(True, axis="x", alpha=0.3)

        # Add text examples if available
        if show_text and "samples_metadata" in self.metadata:
            samples_metadata = self.metadata["samples_metadata"]

            labels = []
            for i, idx in enumerate(top_indices[::-1]):
                if idx < len(samples_metadata):
                    text = samples_metadata[idx].get("text", "")
                    # Clean up text
                    text = text.replace("\n", " ").strip()
                    if len(text) > max_text_length:
                        text = text[:max_text_length] + "..."
                    labels.append(f"{i+1}. {text}")
                else:
                    labels.append(f"{i+1}. (No metadata)")

            ax.set_yticklabels(labels, fontsize=9)

        plt.tight_layout()

        if save_path or self.output_dir:
            path = save_path or self.output_dir / f"feature_{feature_idx}_examples.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved feature examples to {path}")

        return fig

    def plot_decoder_weights(
        self,
        feature_indices: Optional[List[int]] = None,
        n_features: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize decoder weights for specific features.

        This shows how each feature maps back to the MLP activation space.

        Args:
            feature_indices: Specific feature indices to plot
            n_features: Number of features to plot (if feature_indices is None)
            save_path: Optional path to save the figure

        Returns:
            The matplotlib Figure object
        """
        import torch

        if feature_indices is None:
            # Get features by density
            densities = (self.features > 0).mean(axis=0)
            feature_indices = np.argsort(densities)[-n_features:][::-1]

        n_features_plot = len(feature_indices)
        fig, axes = plt.subplots(1, n_features_plot, figsize=(3 * n_features_plot, 4))

        if n_features_plot == 1:
            axes = [axes]

        with torch.no_grad():
            for i, feat_idx in enumerate(feature_indices):
                decoder_weight = self.sae.decoder.weight[:, feat_idx].cpu().numpy()

                axes[i].hist(decoder_weight, bins=50, edgecolor="black", alpha=0.7)
                axes[i].set_title(f"Feature {feat_idx}", fontsize=12)
                axes[i].set_xlabel("Weight Value", fontsize=10)
                axes[i].set_ylabel("Count", fontsize=10)
                axes[i].grid(True, alpha=0.3)

        plt.suptitle("Decoder Weight Distributions", fontsize=14)
        plt.tight_layout()

        if save_path or self.output_dir:
            path = save_path or self.output_dir / "decoder_weights.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved decoder weights plot to {path}")

        return fig

    def create_feature_dashboard(
        self,
        feature_idx: int,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard for a single feature.

        Includes activation histogram, top examples, and decoder weights.

        Args:
            feature_idx: Index of the feature to visualize
            save_path: Optional path to save the figure

        Returns:
            The matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Activation histogram
        ax1 = fig.add_subplot(gs[0, 0])
        feature_acts = self.features[:, feature_idx]
        ax1.hist(feature_acts[feature_acts > 0], bins=50, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Activation Value", fontsize=11)
        ax1.set_ylabel("Count", fontsize=11)
        ax1.set_title(f"Activation Distribution (Feature {feature_idx})", fontsize=12)
        ax1.set_yscale("log")

        # 2. Top activating examples
        ax2 = fig.add_subplot(gs[0, 1])
        top_values, top_indices = self.sae.get_top_activating_examples(
            self.activations, feature_idx, 10
        )
        ax2.barh(range(10), top_values[::-1])
        ax2.set_xlabel("Activation Value", fontsize=11)
        ax2.set_title("Top 10 Activating Examples", fontsize=12)
        ax2.set_yticks(range(10))
        ax2.grid(True, axis="x", alpha=0.3)

        # 3. Decoder weights
        ax3 = fig.add_subplot(gs[1, :])
        import torch

        with torch.no_grad():
            decoder_weight = self.sae.decoder.weight[:, feature_idx].cpu().numpy()

        ax3.hist(decoder_weight, bins=100, edgecolor="black", alpha=0.7)
        ax3.set_xlabel("Weight Value", fontsize=11)
        ax3.set_ylabel("Count", fontsize=11)
        ax3.set_title("Decoder Weight Distribution", fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Add statistics box
        density = (feature_acts > 0).mean()
        max_act = feature_acts.max()

        stats_text = f"Feature {feature_idx} Statistics:\n"
        stats_text += f"Density: {density:.2e}\n"
        stats_text += f"Max Activation: {max_act:.4f}\n"
        stats_text += f"Decoder Norm: {np.linalg.norm(decoder_weight):.4f}"

        fig.text(
            0.5,
            0.02,
            stats_text,
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(f"Feature {feature_idx} Dashboard", fontsize=16)

        if save_path or self.output_dir:
            path = save_path or self.output_dir / f"feature_{feature_idx}_dashboard.png"
            plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved feature dashboard to {path}")

        return fig

    def save_all(self, n_features: int = 10) -> None:
        """
        Generate and save all standard visualizations.

        Args:
            n_features: Number of features to include in detailed plots
        """
        logger.info("Generating all visualizations...")

        self.plot_feature_density()
        self.plot_training_curves()

        for metric in ["density", "max_activation"]:
            self.plot_top_features(n_features=n_features, by=metric)

        # Create dashboards for top features by density
        densities = (self.features > 0).mean(axis=0)
        top_features = np.argsort(densities)[-n_features:]

        for feat_idx in top_features:
            self.create_feature_dashboard(feat_idx)

        logger.info(f"All visualizations saved to {self.output_dir}")
