# Drrik

[![Tests](https://github.com/rashomon-gh/drrik/actions/workflows/tests.yml/badge.svg)](https://github.com/rashomon-gh/drrik/actions/workflows/tests.yml) [![Documentation](https://github.com/rashomon-gh/drrik/actions/workflows/gh_pages.yml/badge.svg)](https://github.com/rashomon-gh/drrik/actions/workflows/gh_pages.yml)

Drrik (দৃক) is a Sanskrit (সংস্কৃত) word which stands for knowledge, eye, and direction. Drrik is a framework for extracting interpretable features from the MLP layers of transformer-based Large Language Models using Sparse Autoencoders, inspired by the [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) paper from Anthropic.

## Installation

### Setup with UV (recommended)

```bash
uv sync
source .venv/bin/activate
```

### Setup with pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

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

### Activation Steering

After training an SAE, use the learned feature directions to steer model generation:

```python
from drrik import SAESteering, SparseAutoencoder

sae = SparseAutoencoder.load("sae_model.pt")
steering = SAESteering(sae, model_name="google/gemma-2b", layer=0)

# Generate with a single feature
result = steering.generate(
    "The weather is",
    feature_idx=42,
    strength=3.0,
    max_new_tokens=50,
)

# Compare baseline vs steered at multiple strengths
comparison = steering.compare_steering(
    "The weather is",
    feature_idx=42,
    strengths=[0.0, 1.0, 2.0, 5.0],
)

# Combine multiple features
result = steering.generate(
    "The weather is",
    feature_indices=[10, 42],
    strengths=[1.0, 2.0],
)

# Baseline (no steering)
baseline = steering.generate("The weather is", max_new_tokens=50)
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

# SAE-based activation steering
python examples/sae_steering.py
```

## Configuration

### YAML Configuration (CLI)

The CLI uses YAML configuration files for easy setup:

```yaml
# Model configuration
model_name: "google/gemma-2b"
torch_dtype: "float16"
device_map: "cpu"

# Dataset configuration
dataset_name: "wikitext"
dataset_config: "wikitext-2-raw-v1"
split: "train"
num_samples: 1000
max_length: 512
extraction_batch_size: 8

# Activation extraction
mlp_layers: [0]

# Sparse Autoencoder configuration
activation_dim: 2048
hidden_dim: 16384  # 8x expansion
l1_coefficient: 0.01
learning_rate: 0.0001
num_epochs: 50
training_batch_size: 256
validation_split: 0.1
resample_dead_neurons: true

# Visualization
n_features_to_visualize: 10

# Hardware configuration
extraction_device: "cpu"
training_device: "mps"

# Wandb integration (optional)
wandb_enabled: false
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
        num_samples=1000,
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

## Configuration

API keys are loaded from a `settings.yml` file (see `settings.example.yml`). Copy and fill in your credentials:

```bash
cp settings.example.yml settings.yml
```

```yaml
# settings.yml
huggingface_hub_token: hf_your_token_here
wandb_api_key: your_wandb_key_here
wandb_project: drrik-experiments
```

Environment variables (`HUGGINGFACE_HUB_TOKEN`, `WANDB_API_KEY`, `WANDB_PROJECT`) override the file when set.

## Key Features

### Supported Models

Any HuggingFace transformer model with MLP layers. 

> [!IMPORTANT]
>  **For Apple Silicon users**: Models like gemma-2b have internal weight matrices that exceed MPS kernel limits. Use `device_map: "cpu"` for activation extraction and `training_device: "mps"` for SAE training or, use a smaller batch size and hidden dimension (meaning a smaller expansion factor).

### Supported Datasets

Any dataset from HuggingFace Datasets.

### SAE Features

Following the Anthropic paper:
- **Overcomplete basis**: Hidden dimension > activation dimension
- **L1 sparsity**: Encourages sparse feature activations
- **Decoder normalization**: Prevents scaling collapse
- **Pre-encoder bias**: As used in the paper
- **Dead neuron resampling**: Reinitializes inactive neurons during training using a sliding window of recent batches for robust detection

### Activation Steering

Use trained SAE features to steer language model generation by adding scaled decoder weight vectors to MLP activations during inference:
- **Single-feature steering**: Bias output toward one learned feature direction
- **Multi-feature steering**: Combine multiple feature directions with individual strengths
- **Comparison tools**: Compare baseline vs steered outputs across strength levels
- **Feature analysis**: Find top-activating features for a given input

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

The full API documentation with Google-style docstrings is available in the generated docs:

```bash
# Regenerate docs (requires pdoc)
.venv/bin/pdoc drrik -o docs --docformat google

# View locally
open docs/drrik.html
```

### ActivationExtractor

```python
extractor = ActivationExtractor(
    model_name="google/gemma-2b",
    dataset_name="wikitext",
    num_samples=1000,
    mlp_layers=[0],
    batch_size=8,
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
    resample_interval: int = 10000,
    dead_threshold: float = 1e-8,
    window_size: int = 100,
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

### SAESteering

```python
from drrik import SAESteering, SparseAutoencoder

sae = SparseAutoencoder.load("sae_model.pt")
steering = SAESteering(
    source=sae,
    model_name="google/gemma-2b",
    layer=0,               # target MLP layer
    device_map="auto",
    # token is read from HUGGINGFACE_HUB_TOKEN in .env by default
)

# Steered generation
output = steering.generate("The sky is", feature_idx=42, strength=2.0)

# Find top features for an input
features = steering.find_steering_features("text", activations, top_k=20)

# Get raw steering direction vector
direction = steering.get_steering_direction(42, normalize=True)

# Build token-to-feature mapping
token_map = steering.build_token_feature_map(tokens=[10, 20, 30])

# Save comparison results
steering.save_steering_analysis(results, "./output", prompt="The sky is")
```

## Visible steering (Eiffel Tower recipe)

Steering small models (e.g. Gemma 2B) with self-trained SAEs rarely produces
visible effects — the AxBench benchmark found SAE steering on Gemma 2B/9B
nearly ineffective compared to plain prompting. For a dramatic, visible effect
use the proven recipe from the
[Eiffel Tower Llama](https://dlouapre-eiffel-tower-llama.hf.space/)
reproduction:

- **Model:** `meta-llama/Llama-3.1-8B-Instruct` (~20GB RAM in fp16)
- **Vectors:** Anthropic's pre-trained Llama 3.1 8B SAE features, pre-extracted
  and published as the dataset `shawon/llama-3.1-8b-instruct_eiffel_tower`
- **Mechanism:** multi-layer injection (layers 11/15/19/23) with clamping,
  temperature 0.5, repetition penalty 1.2, chat template

One-time setup (downloads ~4GB of SAE weights per layer, extracts 8 vectors,
publishes the dataset):

```bash
drrik extract-vectors -c examples/eiffel_tower_recipe.yaml \
    --repo-id shawon/llama-3.1-8b-instruct_eiffel_tower
```

Run the demo (baseline vs steered for 5 off-topic prompts):

```bash
uv run examples/eiffel_tower.py
```

Or interactively (installs gradio):

```bash
uv sync --extra app
uv run examples/eiffel_tower_app.py
```

Programmatic use:

```python
from drrik import SAESteering
from drrik.steering import SteeringVectors

vectors = SteeringVectors.from_hf_dataset(
    "shawon/llama-3.1-8b-instruct_eiffel_tower"
)
steering = SAESteering(source=vectors, model_name="meta-llama/Llama-3.1-8B-Instruct")

steered = steering.generate(
    "The weather today is",
    strength_scale=1.0,
    temperature=0.5,
    repetition_penalty=1.2,
    system_prompt="You are a helpful assistant.",
    seed=16,
)
baseline = steering.generate("The weather today is", strength_scale=0.0, seed=16)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Skip slow tests
pytest -m "not slow"
```

## References

- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html) - Anthropic
- [nnsight Library](https://github.com/ndif-team/nnsight) - For activation extraction
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_models/index.html) - Anthropic

## License

AGPL-3.0

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
