# AGENTS.md

Agent documentation for the Drrik framework.

## Agent Guidelines

**Codebase First**: Always analyze the existing codebase (imports, patterns, nearby code) before considering external sources. Most answers can be inferred from how the project already does things.

**External Verification (fallback)**: Only when the codebase doesn't provide enough context, verify assumptions through:

- Official documentation (linked references)
- Web search results (with sources cited)

**Key Documentation Sources**:

- [PyTorch 2.11](https://docs.pytorch.org/docs/2.11/index.html)
- [nnsight 0.5.15](https://nnsight.net/documentation/)

**No Speculative Code**: Do not write code based on internal reasoning alone. If the codebase is unclear and documentation cannot be found, ask the user rather than guessing.

## Important Technical Notes

### nnsight API (v0.5.15)
- Use `LanguageModel` (not `NNsight`) for loading HuggingFace models. `NNsight` extends `Envoy` directly and doesn't handle HuggingFace kwargs like `revision`.
- Access model layers through the model object (e.g., `model.model.layers[0].mlp.output.save()`), not through the tracer. The tracer context manager is not subscriptable.
- `.save()` returns tensors directly — no `.value` accessor needed.
- Initialize variables (like output lists) **before** the `with model.trace(...)` block, not inside it.

### Device Configuration
- `device_map` controls where the HuggingFace model is loaded (used by nnsight/transformers).
- `extraction_device` and `training_device` are separate — extraction may need CPU on Apple Silicon due to MPS kernel limits on large weight matrices, while SAE training can use MPS.
- Use `self.to(device)` (not `self = self.to(device)`) for moving `nn.Module` to device — reassignment breaks in-place movement.

### Config Key Naming
- `extraction_batch_size` for activation extraction batching.
- `training_batch_size` for SAE training batching.
- `num_samples` in `DatasetConfig` — no default, must be explicitly provided. Replaced the old `max_samples` field.

### Pydantic v2 Migration
- Use `model_config = ConfigDict(...)` instead of `class Config:` inner class.
- `ActivationExtractorConfig.model` and `ActivationExtractorConfig.dataset` are `Optional` — no default factories.

### ActivationExtractor Initialization
- `ActivationExtractor.__init__` accepts flat kwargs (e.g., `model_name`, `num_samples`) and builds nested `ModelConfig`/`DatasetConfig` automatically.
- When passing a `config` object, nested configs must be fully constructed (no partial overrides via `model_copy(update=kwargs)`).

### Dead Neuron Resampling (Sliding Window)
- `SparseAutoencoder.fit()` tracks neuron activity across a sliding window of recent batches (`window_size` parameter, default 100).
- Dead neurons are identified by average activity across the window, not a single batch.
- `resample_dead_neurons()` accepts an optional `dead_mask` parameter for pre-computed masks.
- Window is cleared after resampling to start fresh tracking.
- New `fit()` parameters: `dead_threshold` (default `1e-8`), `window_size` (default `100`).

### Wandb API
- Use `run.url` property instead of the deprecated `get_url()` method.

### datasets Library
- `datasets.Dataset.map()` stores results as Python lists even with `return_tensors="pt"` in the tokenizer. Use `torch.tensor()` (not `torch.stack()`) to convert.

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
docs/
├── drrik.html           # Generated API documentation (pdoc)
├── index.html           # Docs index page
└── search.js            # Full-text search index
```

## Key Components

### [models.py](drrik/models.py)
- **ActivationExtractor**: Loads HuggingFace models/datasets, extracts MLP activations using nnsight
- Supports gated models via `HF_TOKEN` env var
- Auto-detects MLP layer paths for common architectures (Gemma, Llama, Phi, BERT)

### [autoencoder.py](drrik/autoencoder.py)
- **SparseAutoencoder**: Overcomplete basis SAE with L1 regularization
- Implements paper architecture: pre-encoder bias, decoder normalization, sliding window dead neuron resampling
- Training loop with validation, wandb logging, gradient tricks

### [visualization.py](drrik/visualization.py)
- **FeatureVisualizer**: Generates plots for feature analysis
- Creates density histograms, training curves, top features, dashboards

### [config.py](drrik/config.py)
- Pydantic configs: `ModelConfig`, `DatasetConfig`, `ActivationExtractorConfig`, `SparseAutoencoderConfig`
- Type-safe, validated configuration — `DatasetConfig.num_samples` has no default and must be explicitly provided

### [settings.py](drrik/settings.py)
- `EnvironmentSettings`: Loads HF/WANDB API keys from .env
- `WandbConfig`: Context manager for wandb experiment tracking

### [cli.py](drrik/cli.py)
- Commands: `init-config`, `extract`, `train`, `visualize`, `run`
- YAML-driven pipeline execution

## Common Tasks

### Adding New Model Architecture Support
Edit [models.py:224](drrik/models.py#L224) in `_get_mlp_layer_name()` to add the MLP path pattern.

### Modifying SAE Architecture
Edit [autoencoder.py:78](drrik/autoencoder.py#L78) in `SparseAutoencoder.__init__()`.

### Modifying Dead Neuron Resampling
Edit [autoencoder.py:226](drrik/autoencoder.py#L226) in `resample_dead_neurons()` and [autoencoder.py:305](drrik/autoencoder.py#L305) in `fit()`.

### Adding New Visualizations
Add method to [visualization.py:40](drrik/visualization.py#L40) in `FeatureVisualizer` class.

### Adding CLI Commands
Add Click command to [cli.py:42](drrik/cli.py#L42) using the `@cli.command()` decorator.

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

Dev: pytest (testing), ruff (linting/formatting), pre-commit (git hooks), pdoc (docs generation)

## Docstring & Documentation Conventions

- All modules, classes, and public methods use **Google-style docstrings** (Args, Returns, Raises, Example sections).
- Docstrings describe the "why" and "how", not just the "what". Include cross-references to the paper where relevant.
- HTML API documentation is generated with **pdoc** and lives in `docs/`.

### Regenerating Docs

```bash
.venv/bin/pdoc drrik -o docs --docformat google
```

## Entry Points

- **CLI**: `drrik` command → [cli.py:719](drrik/cli.py#L719)
- **Python API**: `from drrik import ActivationExtractor, SparseAutoencoder, FeatureVisualizer`
- **API Docs**: [docs/drrik.html](docs/drrik.html)
