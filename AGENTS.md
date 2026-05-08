# AGENTS.md

Agent documentation for the Drrik framework.

## Agent Guidelines

**Verification Required**: Agents must only write code that can be verified through:

- Official documentation (linked references)
- Web search results (with sources cited)
- Existing codebase patterns

**Key Documentation Sources**:

- [PyTorch 2.11](https://docs.pytorch.org/docs/2.11/index.html)
- [nnsight](https://nnsight.net/documentation/)

**No Speculative Code**: Do not write code based on internal reasoning alone. Every technical decision must be grounded in verifiable sources.

**When in doubt**: Search first, code second. If documentation cannot be found, ask the user rather than guessing.

## Project Overview

Drrik is a PyTorch framework for extracting interpretable features from transformer MLP layers using Sparse Autoencoders (SAEs), inspired by Anthropic's [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) paper.

**Language:** Python 3.12+  
**License:** AGPL-3.0

## Architecture

```
drrik/
├── __init__.py          # Public API exports
├── config.py            # Pydantic configuration models
├── settings.py          # Environment variables, wandb integration
├── models.py            # ActivationExtractor (nnsight-based)
├── autoencoder.py       # SparseAutoencoder implementation
├── visualization.py     # FeatureVisualizer for plots
└── cli.py               # Click-based CLI (drrik command)
```

## Key Components

### [models.py](drrik/models.py)
- **ActivationExtractor**: Loads HuggingFace models/datasets, extracts MLP activations using nnsight
- Supports gated models via `HF_TOKEN` env var
- Auto-detects MLP layer paths for common architectures (Gemma, Llama, Phi, BERT)

### [autoencoder.py](drrik/autoencoder.py)
- **SparseAutoencoder**: Overcomplete basis SAE with L1 regularization
- Implements paper architecture: pre-encoder bias, decoder normalization, dead neuron resampling
- Training loop with validation, wandb logging, gradient tricks

### [visualization.py](drrik/visualization.py)
- **FeatureVisualizer**: Generates plots for feature analysis
- Creates density histograms, training curves, top features, dashboards

### [config.py](drrik/config.py)
- Pydantic configs: `ModelConfig`, `DatasetConfig`, `ActivationExtractorConfig`, `SparseAutoencoderConfig`
- Type-safe, validated configuration with defaults

### [settings.py](drrik/settings.py)
- `EnvironmentSettings`: Loads HF/WANDB API keys from .env
- `WandbConfig`: Context manager for wandb experiment tracking

### [cli.py](drrik/cli.py)
- Commands: `init-config`, `extract`, `train`, `visualize`, `run`
- YAML-driven pipeline execution

## Common Tasks

### Adding New Model Architecture Support
Edit [models.py:175](drrik/models.py#L175) in `_get_mlp_layer_name()` to add the MLP path pattern.

### Modifying SAE Architecture
Edit [autoencoder.py:58](drrik/autoencoder.py#L58) in `SparseAutoencoder.__init__()`.

### Adding New Visualizations
Add method to [visualization.py:26](drrik/visualization.py#L26) in `FeatureVisualizer` class.

### Adding CLI Commands
Add Click command to [cli.py:28](drrik/cli.py#L28) using the `@cli.command()` decorator.

## Environment Setup

Required env vars (see [.env.example](.env.example)):
- `HF_TOKEN` - HuggingFace token for gated models
- `WANDB_API_KEY` - Optional, for experiment tracking

## Linting & Formatting

Uses **ruff** for linting and formatting, enforced via pre-commit hooks.

```bash
# Manual linting with auto-fix
ruff check --fix .

# Manual formatting
ruff format .

# Install pre-commit hooks (runs automatically on git commit)
pre-commit install

# Run all pre-commit hooks manually
pre-commit run --all-files
```

Pre-commit config: [.pre-commit-config.yaml](.pre-commit-config.yaml) (ruff v0.4.10)

## Running Tests

```bash
pytest                    # All tests
pytest -m "not slow"      # Skip slow tests
```

## Dependencies

Core: torch, transformers, nnsight, datasets, pydantic, click, wandb, matplotlib, seaborn

Dev: pytest (testing), ruff (linting/formatting), pre-commit (git hooks)

## Entry Points

- **CLI**: `drrik` command → [cli.py:598](drrik/cli.py#L598)
- **Python API**: `from drrik import ActivationExtractor, SparseAutoencoder, FeatureVisualizer`
