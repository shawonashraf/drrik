# Drrik

Drrik (দৃক) is a Sanskrit (সংস্কৃত) word which stands for knowledge, eye, and direction. Drrik is a framework for extracting interpretable features from the MLP layers of transformer-based Large Language Models using Sparse Autoencoders, inspired by the [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) paper from Anthropic.

## Overview

This framework implements a pipeline for:

1. **Loading models** from HuggingFace Hub
2. **Running inference** on datasets from HuggingFace Hub
3. **Collecting MLP activations** using the [`nnsight`](https://github.com/ndif-team/nnsight) library
4. **Training Sparse Autoencoders** to extract interpretable features
5. **Visualizing features** with matplotlib and seaborn

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

### Basic Usage

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

### Run Example Scripts

```bash
# Basic usage example
python examples/basic_usage.py

# Advanced pipeline with custom configs
python examples/advanced_pipeline.py

# Load saved activations
python examples/load_saved_activations.py
```

## Project Structure

```
drrik/
├── drrik/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Pydantic configuration classes
│   ├── models.py             # Activation extraction using nnsight
│   ├── autoencoder.py        # Sparse Autoencoder implementation
│   └── visualization.py      # Feature visualization tools
├── examples/
│   ├── basic_usage.py        # Basic pipeline example
│   ├── advanced_pipeline.py  # Advanced configuration examples
│   └── load_saved_activations.py
├── visualizations/           # Output directory for plots
├── pyproject.toml           # Project dependencies
└── README.md
```

## Configuration

The framework uses Pydantic for configuration. Key configuration classes:

### ActivationExtractorConfig

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

### SparseAutoencoderConfig

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

## Visualization Outputs

The framework generates several visualizations:

1. **Feature Density Histogram** - Distribution of feature firing rates
2. **Training Curves** - Loss and L0 norm over training
3. **Top Features** - Features ranked by density/activation
4. **Feature Dashboards** - Comprehensive view per feature
5. **Activation Histograms** - Distribution of activations per feature

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
)

visualizer.plot_feature_density()
visualizer.plot_top_features(n_features=10)
visualizer.create_feature_dashboard(feature_idx=0)
visualizer.save_all(n_features=10)
```

## References

- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html) - Anthropic
- [nnsight Library](https://github.com/ndif-team/nnsight) - For activation extraction
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_models/index.html) - Anthropic

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
