# Drrik

Drrik (দৃক) is a Sanskrit (সংস্কৃত) word which stands for knowledge, eye, and direction. Drrik is a framework for extracting interpretable features from the MLP layers of transformer-based Large Language Models using Sparse Autoencoders, inspired by the [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) paper from Anthropic.

## Overview

This framework implements a pipeline for:

1. **Loading models** from HuggingFace Hub
2. **Running inference** on datasets from HuggingFace Hub
3. **Collecting MLP activations** using the [`nnsight`](https://github.com/ndif-team/nnsight) library
4. **Training Sparse Autoencoders** to extract interpretable features
5. **Visualizing features** with matplotlib and seaborn
6. **Tracking experiments** with Weights & Biases (optional)

## Installation

### Requirements

- Python 3.12+
- CUDA-capable GPU (recommended, 8GB VRAM sufficient for <3B models)
- 8GB+ RAM

### Setup with UV (recommended)

```bash
# Install uv if you don't have it
pip install uv

# Clone and install
git clone <repo-url>
cd drrik
uv sync
```

### Setup with pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

### Option 1: Using the CLI (Recommended)

The CLI provides a simple way to run the full pipeline with a YAML configuration file.

```bash
# Generate an example config file
drrik init-config -o config.yml

# Edit config.yml to customize your settings

# Run the full pipeline (extract -> train -> visualize)
drrik run config.yml

# Or run individual steps
drrik extract -c config.yml
drrik train -c config.yml
drrik visualize -c config.yml
```

### Option 2: Python API

```python
from drrik import ActivationExtractor, SparseAutoencoder, FeatureVisualizer

# 1. Extract MLP activations
extractor = ActivationExtractor(
    model_name="google/gemma-2b",  # 2B parameters, fits on 8GB VRAM
    dataset_name="wikitext",
    mlp_layers=[0],
    num_samples=1000,
)
activations, metadata = extractor.extract()

# 2. Train Sparse Autoencoder
sae = SparseAutoencoder(
    activation_dim=activations.shape[-1],
    hidden_dim=activations.shape[-1] * 8,  # 8x expansion
    l1_coefficient=0.01,
)
sae.fit(activations, num_epochs=50)

# 3. Visualize features
visualizer = FeatureVisualizer(sae, activations, metadata)
visualizer.save_all(n_features=10)
```

### CLI Commands

The `drrik` CLI provides several commands:

- `drrik init-config` - Generate an example YAML configuration file
- `drrik extract` - Extract MLP activations from a model
- `drrik train` - Train a sparse autoencoder
- `drrik visualize` - Generate feature visualizations
- `drrik run` - Run the full pipeline

Each command supports additional options:

```bash
drrik extract --config config.yml --output-dir ./outputs --device cuda
drrik train --config config.yml --activations ./outputs/activations.pkl
drrik visualize --config config.yml --n-features 20 --no-wandb
```

### Example Scripts

```bash
# Basic usage example
python examples/basic_usage.py

# Advanced pipeline with custom configs
python examples/advanced_pipeline.py

# Load saved activations
python examples/load_saved_activations.py

# Using wandb integration
python examples/with_wandb.py
```

## Project Structure

```
drrik/
├── drrik/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Pydantic configuration classes
│   ├── models.py             # Activation extraction using nnsight
│   ├── autoencoder.py        # Sparse Autoencoder implementation
│   ├── visualization.py      # Feature visualization tools
│   ├── settings.py           # Environment settings and wandb config
│   └── cli.py                # Command-line interface
├── examples/
│   ├── basic_usage.py        # Basic pipeline example
│   ├── advanced_pipeline.py  # Advanced configuration examples
│   ├── load_saved_activations.py
│   └── with_wandb.py         # wandb integration example
├── tests/
│   ├── conftest.py           # Pytest configuration
│   ├── test_imports.py       # Core functionality tests
│   └── test_settings.py      # Settings and wandb tests
├── .env.example              # Environment variables template
├── config.yml                # Example YAML configuration
├── pyproject.toml           # Project dependencies
└── README.md
```

## Configuration

### YAML Configuration (CLI)

The CLI uses YAML configuration files for easy setup:

```yaml
# Model configuration
model_name: "google/gemma-2b"
torch_dtype: "float16"
device_map: "auto"

# Dataset configuration
dataset_name: "wikitext"
dataset_config: "wikitext-2-raw-v1"
split: "train"
num_samples: 1000
batch_size: 8

# Activation extraction
mlp_layers: [0]

# Sparse Autoencoder configuration
activation_dim: 2048
hidden_dim: 16384  # 8x expansion
l1_coefficient: 0.01
learning_rate: 0.0001
num_epochs: 50
validation_split: 0.1
resample_dead_neurons: true

# Visualization
n_features_to_visualize: 10

# Wandb integration (optional)
wandb_enabled: true
wandb_project: "drrik-experiments"

# Output
output_dir: "./drrik_output"
```

Generate an example config with: `drrik init-config -o config.yml`

### Python API Configuration

The framework uses Pydantic for configuration. Key configuration classes:

#### ActivationExtractorConfig

```python
from drrik.config import ActivationExtractorConfig, ModelConfig, DatasetConfig

config = ActivationExtractorConfig(
    model=ModelConfig(
        model_name="google/gemma-2b",
        torch_dtype="float16",
    ),
    dataset=DatasetConfig(
        dataset_name="wikitext",
        max_samples=1000,
        max_length=512,
    ),
    mlp_layers=[0, 1, 2],
    batch_size=8,
)
```

#### SparseAutoencoderConfig

```python
from drrik.config import SparseAutoencoderConfig

sae_config = SparseAutoencoderConfig(
    activation_dim=2048,
    hidden_dim=4096,  # 2x expansion
    l1_coefficient=0.01,
    learning_rate=1e-4,
    resample_dead_neurons=True,
)
```

## Environment Variables

For API keys and optional settings, create a `.env` file (see `.env.example`):

```bash
# HuggingFace Hub token (for gated models)
HF_TOKEN=your_token_here

# Weights & Biases API key (optional, for experiment tracking)
WANDB_API_KEY=your_wandb_key_here

# Wandb settings
WANDB_PROJECT=drrik-experiments
WANDB_ENTITY=your_username
WANDB_MODE=online  # or 'offline' to disable
```

## Key Features

### Supported Models

Any HuggingFace transformer model with MLP layers. Recommended for 8GB VRAM:
- `google/gemma-2b` (2B parameters)
- `microsoft/phi-2` (2.7B parameters)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B parameters)

### Supported Datasets

Any dataset from HuggingFace Datasets. Common choices:
- `wikitext` - Wikipedia text
- `pile` - The Pile dataset
- `c4` - Colossal Clean Crawled Corpus

### SAE Features

Following the Anthropic paper:
- **Overcomplete basis**: Hidden dimension > activation dimension
- **L1 sparsity**: Encourages sparse feature activations
- **Decoder normalization**: Prevents scaling collapse
- **Pre-encoder bias**: As used in the paper
- **Dead neuron resampling**: Reinitializes inactive neurons during training

### Wandb Integration

Optional wandb integration for experiment tracking:

```python
from drrik import WandbConfig, get_settings

settings = get_settings()
wandb_config = WandbConfig(
    project="drrik-experiments",
    name="my-experiment",
    config={"model": "gemma-2b", "expansion": 8},
    enabled=settings.use_wandb,  # Auto-disables if no API key
)

# Use in training
sae.fit(activations, wandb_config=wandb_config, wandb_enabled=True)

# Use in visualization
visualizer = FeatureVisualizer(
    sae=sae,
    activations=activations,
    wandb_config=wandb_config,
    log_to_wandb=True,
)
```

The framework automatically logs:
- Training metrics (loss, L0 norm, dead neurons)
- Learning rate changes
- Activation histograms
- Feature visualizations

## Visualization Outputs

The framework generates several visualizations:

1. **Feature Density Histogram** - Distribution of feature firing rates
2. **Training Curves** - Loss and L0 norm over training
3. **Top Features** - Features ranked by density/activation
4. **Feature Dashboards** - Comprehensive view per feature
5. **Activation Histograms** - Distribution of activations per feature

All plots can be saved locally and optionally logged to wandb.

## API Reference

### ActivationExtractor

```python
extractor = ActivationExtractor(
    model_name: str = "google/gemma-2b",
    dataset_name: str = "wikitext",
    mlp_layers: List[int] = [0],
    num_samples: int = 1000,
    batch_size: int = 8,
)

activations, metadata = extractor.extract()
```

### SparseAutoencoder

```python
sae = SparseAutoencoder(
    activation_dim: int,
    hidden_dim: int,
    l1_coefficient: float = 0.01,
)

sae.fit(
    activations: np.ndarray,
    batch_size: int = 256,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    resample_dead_neurons: bool = True,
    wandb_config: Optional[WandbConfig] = None,
    wandb_enabled: bool = False,
)

features = sae.encode(activations)
reconstructed = sae.decode(features)
```

### FeatureVisualizer

```python
visualizer = FeatureVisualizer(
    sae: SparseAutoencoder,
    activations: np.ndarray,
    metadata: Optional[Dict] = None,
    output_dir: str = "./visualizations",
    wandb_config: Optional[WandbConfig] = None,
    log_to_wandb: bool = False,
)

visualizer.plot_feature_density()
visualizer.plot_top_features(n_features=10)
visualizer.create_feature_dashboard(feature_idx=0)
visualizer.save_all(n_features=10)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_imports.py

# Run with verbose output
pytest -v

# Skip slow tests
pytest -m "not slow"
```

## References

- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html) - Anthropic
- [nnsight Library](https://github.com/ndif-team/nnsight) - For activation extraction
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_models/index.html) - Anthropic

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
