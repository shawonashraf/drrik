"""
Sparse Autoencoder implementation for feature extraction.

This module implements a sparse autoencoder as described in the
"Towards Monosemanticity" paper from Anthropic.

The autoencoder learns an overcomplete basis of features that are
more interpretable than the original MLP neuron activations.
"""

from typing import Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from loguru import logger

from drrik.settings import WandbConfig


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for extracting interpretable features from MLP activations.

    Architecture:
        Input (activation_dim) -> [bias] -> Encoder (hidden_dim) -> ReLU -> [bias]
        -> Decoder (activation_dim) -> Output

    The decoder weights are normalized to unit norm to prevent scaling collapse.
    L1 regularization on hidden activations encourages sparsity.

    Based on the architecture described in:
    "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
    https://transformer-circuits.pub/2023/monosemantic-features/index.html

    Attributes:
        encoder: Linear layer from activation_dim to hidden_dim
        decoder: Linear layer from hidden_dim to activation_dim
        pre_encoder_bias: Learnable bias subtracted from input
        post_decoder_bias: Learnable bias added to output (tied to pre_encoder_bias)

    Example:
        sae = SparseAutoencoder(
            activation_dim=2048,
            hidden_dim=4096,  # 2x expansion
            l1_coefficient=0.01
        )
        sae.fit(activations, num_epochs=100)
        features = sae.encode(activations)
        reconstructed = sae.decode(features)
    """

    def __init__(
        self,
        activation_dim: int,
        hidden_dim: int,
        l1_coefficient: float = 0.01,
        normalize_decoder: bool = True,
        pre_encoder_bias: bool = True,
    ):
        """
        Initialize the Sparse Autoencoder.

        Args:
            activation_dim: Dimension of input MLP activations
            hidden_dim: Dimension of sparse hidden layer (overcomplete)
            l1_coefficient: L1 regularization strength
            normalize_decoder: Whether to normalize decoder columns to unit norm
            pre_encoder_bias: Whether to use pre-encoder bias (as in the paper)
        """
        super().__init__()

        self.activation_dim = activation_dim
        self.hidden_dim = hidden_dim
        self.l1_coefficient = l1_coefficient
        self.normalize_decoder = normalize_decoder
        self.pre_encoder_bias = pre_encoder_bias

        # Encoder: input -> hidden
        self.encoder = nn.Linear(activation_dim, hidden_dim, bias=False)

        # Decoder: hidden -> output
        # Initialize with small random weights
        self.decoder = nn.Linear(hidden_dim, activation_dim, bias=False)

        # Initialize decoder weights to unit norm (Kaiming uniform)
        nn.init.kaiming_uniform_(self.decoder.weight, a=np.sqrt(5))
        with torch.no_grad():
            self.decoder.weight.copy_(
                self.decoder.weight / self.decoder.weight.norm(dim=0, keepdim=True)
            )

        # Initialize encoder weights
        nn.init.kaiming_uniform_(self.encoder.weight, a=np.sqrt(5))

        # Biases
        if pre_encoder_bias:
            # Pre-encoder bias (subtracted from input, added to output)
            # Initialize to geometric median of data (will be set during fit)
            self.bias = nn.Parameter(torch.zeros(activation_dim))
        else:
            self.register_parameter("bias", None)

        # Encoder bias
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Training metrics
        self.training_losses = []
        self.training_l0_norms = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse features.

        Args:
            x: Input tensor of shape (batch_size, activation_dim)

        Returns:
            Sparse feature activations of shape (batch_size, hidden_dim)
        """
        # Subtract bias if using pre-encoder bias
        if self.bias is not None:
            x = x - self.bias

        # Apply encoder
        hidden = F.linear(x, self.encoder.weight, self.encoder_bias)

        # ReLU activation for sparsity
        features = F.relu(hidden)

        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.

        Args:
            features: Sparse feature tensor of shape (batch_size, hidden_dim)

        Returns:
            Reconstructed activations of shape (batch_size, activation_dim)
        """
        # Apply decoder (weights are normalized to unit norm)
        output = F.linear(features, self.decoder.weight)

        # Add bias if using
        if self.bias is not None:
            output = output + self.bias

        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, activation_dim)

        Returns:
            Tuple of (reconstructed, features)
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features

    def loss(
        self, x: torch.Tensor, reconstructed: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss with L1 regularization.

        Loss = MSE(reconstructed, x) + lambda * L1(features)

        Args:
            x: Input tensor
            reconstructed: Reconstructed tensor
            features: Sparse feature activations

        Returns:
            Total loss
        """
        mse_loss = F.mse_loss(reconstructed, x)
        l1_loss = self.l1_coefficient * features.abs().sum(dim=1).mean()
        return mse_loss + l1_loss

    def normalize_decoder_weights(self) -> None:
        """Normalize decoder weights to unit norm."""
        if self.normalize_decoder:
            with torch.no_grad():
                self.decoder.weight.copy_(
                    self.decoder.weight
                    / self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
                )

    def resample_dead_neurons(
        self,
        activations: torch.Tensor,
        dead_threshold: float = 1e-8,
    ) -> int:
        """
        Resample dead neurons that haven't fired recently.

        This follows the procedure from the paper:
        1. Identify neurons that haven't fired above threshold
        2. Compute loss on a batch of data
        3. Resample dead neurons to fit poorly reconstructed examples

        Args:
            activations: Batch of activations to use for resampling
            dead_threshold: Activation threshold below which a neuron is considered dead

        Returns:
            Number of neurons resampled
        """
        with torch.no_grad():
            # Get feature activations
            features = self.encode(activations)

            # Find dead neurons (those with very low activation)
            neuron_activity = features.abs().sum(dim=0)  # (hidden_dim,)
            dead_mask = neuron_activity < dead_threshold
            n_dead = dead_mask.sum().item()

            if n_dead == 0:
                return 0

            logger.info(f"Resampling {n_dead} dead neurons")

            # Compute reconstruction loss for each sample
            reconstructed, _ = self.forward(activations)
            sample_losses = F.mse_loss(
                reconstructed, activations, reduction="none"
            ).sum(dim=1)

            # Sample based on loss (higher loss = more likely to be resampled)
            probs = sample_losses**2
            probs = probs / probs.sum()

            # For each dead neuron, resample from a high-loss example
            for dead_idx in torch.where(dead_mask)[0]:
                # Sample an example
                sample_idx = torch.multinomial(probs, 1).item()
                example = activations[sample_idx]

                # Normalize example
                example_normalized = example / (example.norm() + 1e-8)

                # Set decoder weight
                self.decoder.weight[:, dead_idx] = example_normalized

                # Set encoder weight (smaller scale to prevent immediate firing)
                avg_encoder_norm = (
                    self.encoder.weight[:, ~dead_mask].norm(dim=0).mean().item()
                    if (~dead_mask).any() > 0
                    else 0.1
                )
                self.encoder.weight[dead_idx, :] = (
                    example_normalized * avg_encoder_norm * 0.2
                )

                # Reset encoder bias
                self.encoder_bias[dead_idx] = 0

            # Re-normalize decoder
            self.normalize_decoder_weights()

        return n_dead

    def fit(
        self,
        activations: np.ndarray,
        batch_size: int = 256,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        validation_split: float = 0.1,
        resample_dead_neurons: bool = True,
        resample_interval: int = 10000,
        device: Optional[str] = None,
        verbose: bool = True,
        wandb_config: Optional[WandbConfig] = None,
        wandb_enabled: bool = True,
    ) -> "SparseAutoencoder":
        """
        Train the sparse autoencoder on MLP activations.

        Args:
            activations: Training data of shape (n_samples, activation_dim)
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for Adam optimizer
            validation_split: Fraction of data to use for validation
            resample_dead_neurons: Whether to resample dead neurons during training
            resample_interval: Steps between resampling checks
            device: Device to train on (cuda/cpu). If None, auto-detect
            verbose: Whether to show progress bars
            wandb_config: Optional WandbConfig for experiment tracking
            wandb_enabled: If True and wandb_config is None, create default WandbConfig

        Returns:
            self (trained model)
        """
        # Setup wandb if enabled
        wandb_logger = None
        if wandb_enabled:
            if wandb_config is None:
                wandb_logger = WandbConfig(
                    config={
                        "activation_dim": self.activation_dim,
                        "hidden_dim": self.hidden_dim,
                        "expansion_factor": self.hidden_dim / self.activation_dim,
                        "l1_coefficient": self.l1_coefficient,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                    }
                )
            else:
                wandb_logger = wandb_config

            if wandb_logger:
                wandb_logger.initialize()

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self = self.to(device)

        # Convert to tensor
        if isinstance(activations, np.ndarray):
            activations = torch.from_numpy(activations).float()

        n_samples = len(activations)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        # Split data
        train_data = activations[:n_train]
        val_data = activations[n_train:]

        # Create data loaders
        train_dataset = TensorDataset(train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Initialize bias to median if using
        if self.bias is not None and self.pre_encoder_bias:
            with torch.no_grad():
                # Approximate geometric median using median for simplicity
                median = train_data.median(dim=0).values
                self.bias.data = median

        # Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
        )

        # Training loop
        self.training_losses = []
        self.training_l0_norms = []

        n_steps_since_resample = 0
        global_step = 0

        if verbose:
            pbar = tqdm(total=num_epochs, desc="Training SAE")
        else:
            pbar = None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_l0 = 0.0

            for batch in train_loader:
                batch = batch[0].to(device)

                # Forward pass
                reconstructed, features = self.forward(batch)

                # Compute loss
                loss = self.loss(batch, reconstructed, features)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Remove gradients parallel to decoder weights (Adam optimization trick from paper)
                if self.normalize_decoder:
                    with torch.no_grad():
                        # Project gradients to be orthogonal to decoder weights
                        decoder_w = self.decoder.weight
                        decoder_grad = self.decoder.weight.grad

                        # Compute projection
                        projection = (decoder_grad * decoder_w).sum(
                            dim=0, keepdim=True
                        ) * decoder_w
                        self.decoder.weight.grad = decoder_grad - projection

                optimizer.step()

                # Normalize decoder weights
                self.normalize_decoder_weights()

                # Track metrics
                epoch_loss += loss.item()
                epoch_l0 += (features > 0).sum(dim=1).float().mean().item()

                n_steps_since_resample += 1
                global_step += 1

                # Resample dead neurons
                if (
                    resample_dead_neurons
                    and n_steps_since_resample >= resample_interval
                ):
                    self.resample_dead_neurons(batch)
                    n_steps_since_resample = 0

            # Average metrics
            avg_loss = epoch_loss / len(train_loader)
            avg_l0 = epoch_l0 / len(train_loader)

            self.training_losses.append(avg_loss)
            self.training_l0_norms.append(avg_l0)

            # Validation
            with torch.no_grad():
                val_data_tensor = val_data.to(device)
                val_reconstructed, val_features = self.forward(val_data_tensor)
                val_loss = F.mse_loss(val_reconstructed, val_data_tensor).item()
                val_l0 = (val_features > 0).sum(dim=1).float().mean().item()

            # Log to wandb
            if wandb_logger:
                wandb_logger.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "train_l0_norm": avg_l0,
                        "val_loss": val_loss,
                        "val_l0_norm": val_l0,
                        "learning_rate": learning_rate,
                    },
                    step=epoch + 1,
                )

            if verbose:
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.6f}",
                            "val_loss": f"{val_loss:.6f}",
                            "L0": f"{avg_l0:.2f}",
                            "val_L0": f"{val_l0:.2f}",
                        }
                    )
                else:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs}: "
                        f"loss={avg_loss:.6f}, val_loss={val_loss:.6f}, "
                        f"L0={avg_l0:.2f}, val_L0={val_l0:.2f}"
                    )

        if pbar:
            pbar.close()

        # Finalize wandb and log summary metrics
        if wandb_logger:
            # Log final feature densities
            densities = self.get_feature_density(activations.cpu().numpy())
            n_dead = (densities == 0).sum()
            n_active = (densities > 0).sum()

            wandb_logger.log_metrics(
                {
                    "final_train_loss": self.training_losses[-1],
                    "final_val_loss": val_loss,
                    "final_l0_norm": self.training_l0_norms[-1],
                    "final_val_l0_norm": val_l0,
                    "n_dead_features": int(n_dead),
                    "n_active_features": int(n_active),
                    "feature_sparsity": float(n_active / len(densities)),
                }
            )

            # Log feature density histogram
            wandb_logger.log_histogram(densities, "feature_density_histogram")

            logger.info(f"wandb run URL: {wandb_logger.get_run_url()}")

            # Finalize wandb (close the run)
            wandb_logger.finalize()

        logger.info("Training complete!")
        return self

    def get_feature_density(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute the density (fraction of non-zero activations) for each feature.

        Args:
            activations: Data to compute density on

        Returns:
            Array of feature densities of shape (hidden_dim,)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(activations, np.ndarray):
                activations = torch.from_numpy(activations).float()

            features = self.encode(activations)
            density = (features > 0).float().mean(dim=0).cpu().numpy()

        return density

    def get_top_activating_examples(
        self,
        activations: np.ndarray,
        feature_idx: int,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the top k examples that activate a given feature the most.

        Args:
            activations: Data to search through
            feature_idx: Index of the feature to analyze
            k: Number of top examples to return

        Returns:
            Tuple of (top activations values, top indices)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(activations, np.ndarray):
                activations = torch.from_numpy(activations).float()

            features = self.encode(activations)
            feature_activations = features[:, feature_idx].cpu().numpy()

            top_k_indices = np.argsort(feature_activations)[-k:][::-1]
            top_k_values = feature_activations[top_k_indices]

        return top_k_values, top_k_indices

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model state to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "activation_dim": self.activation_dim,
            "hidden_dim": self.hidden_dim,
            "l1_coefficient": self.l1_coefficient,
            "state_dict": self.state_dict(),
            "training_losses": self.training_losses,
            "training_l0_norms": self.training_l0_norms,
        }

        torch.save(state, filepath)
        logger.info(f"Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SparseAutoencoder":
        """Load model state from disk."""
        filepath = Path(filepath)
        state = torch.load(filepath, weights_only=False)

        model = cls(
            activation_dim=state["activation_dim"],
            hidden_dim=state["hidden_dim"],
            l1_coefficient=state["l1_coefficient"],
        )
        model.load_state_dict(state["state_dict"])
        model.training_losses = state.get("training_losses", [])
        model.training_l0_norms = state.get("training_l0_norms", [])

        logger.info(f"Loaded model from {filepath}")
        return model
