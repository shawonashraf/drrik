# Eiffel Tower Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Drrik's SAE steering produce visible output effects by reproducing the Eiffel Tower Llama recipe (Llama 3.1 8B Instruct + Anthropic's pre-trained SAE vectors + improved steering mechanism).

**Architecture:** New `SteeringVectors` class holds pre-extracted unit-norm steering components (loadable from an HF dataset, local file, or an existing `SparseAutoencoder`). `SAESteering` accepts either source, gains multi-layer hooks, clamping, repetition penalty, chat-template prompts, `strength_scale`, and `seed`. Two new CLI commands: `extract-vectors` (one-time 4GB SAE pull → tiny HF dataset) and `steer` (run a recipe). Phase 2 adds a Gradio app.

**Tech Stack:** Python 3.12, PyTorch, nnsight, transformers, datasets, click, pytest, ruff, gradio (optional extra)

**Spec:** `docs/superpowers/specs/2026-08-19-eiffel-tower-steering-design.md`

## Global Constraints

- Python 3.12+, uv-managed env: run tests with `uv run pytest`, lint with `uv run ruff check --fix .` and `uv run ruff format .`
- Google-style docstrings on all public classes/methods (project convention)
- No new core dependencies; `gradio` goes in an optional `app` extra only
- HF token comes from `get_settings().huggingface_hub_token` (`settings.yml`) — never from shell env vars
- Dataset naming: `{model_name}_{sae_vectors}` under HF user `shawon`; this recipe's dataset is `shawon/llama-3.1-8b-instruct_eiffel_tower`
- Recipe values (from the official demo): temperature 0.5, repetition_penalty 1.2, system prompt `"You are a helpful assistant."`, seed 16, 8 features in layers 11/15/19/23 with reduced strengths `[0.128, 0.255, 0.132, 0.103, 0.459, 0.466, 0.228, 0.043]`
- Existing tests must stay green; existing SAE-path behavior of `SAESteering` is unchanged (new params all have backward-compatible defaults)
- Commit after every task

---

### Task 1: `SteeringComponent` + `SteeringVectors` core

**Files:**
- Modify: `drrik/steering.py` (add after `_sample_next_token`, before `class SAESteering`)
- Modify: `drrik/__init__.py` (export `SteeringVectors`)
- Test: `tests/test_steering_vectors.py` (create)

**Interfaces:**
- Consumes: `SparseAutoencoder` (existing, `drrik/autoencoder.py`) — uses `sae.decoder.weight` (shape `(activation_dim, hidden_dim)`) and `sae.hidden_dim`
- Produces:
  - `SteeringComponent(layer: int, feature_idx: int, strength: float, vector: torch.Tensor)` — frozen dataclass, `vector` unit-norm shape `(activation_dim,)`
  - `SteeringVectors(components: List[SteeringComponent], hook_path: str = "layer")` with `.components`, `.hook_path` (`"layer"` | `"mlp"`), `.layers` (sorted unique list), `.activation_dim` (int)
  - `SteeringVectors.from_sae(sae, layer, feature_indices, strengths=None) -> SteeringVectors` (hook_path `"mlp"`)
  - `SteeringVectors.save(path) -> None` / `SteeringVectors.from_file(path) -> SteeringVectors` (round-trips `hook_path`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_steering_vectors.py`:

```python
"""Tests for SteeringVectors (no model loading, CPU only)."""

import pytest
import torch

from drrik.autoencoder import SparseAutoencoder
from drrik.steering import SteeringComponent, SteeringVectors


def _unit_vector(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def test_round_trip_preserves_components(tmp_path):
    comps = [
        SteeringComponent(11, 74457, 1.408, _unit_vector(16, seed=1)),
        SteeringComponent(15, 21576, 1.545, _unit_vector(16, seed=2)),
    ]
    vectors = SteeringVectors(comps, hook_path="layer")
    path = tmp_path / "vectors.pt"
    vectors.save(path)

    loaded = SteeringVectors.from_file(path)
    assert loaded.hook_path == "layer"
    assert len(loaded.components) == 2
    for orig, new in zip(comps, loaded.components):
        assert new.layer == orig.layer
        assert new.feature_idx == orig.feature_idx
        assert new.strength == pytest.approx(orig.strength)
        assert torch.allclose(new.vector, orig.vector)


def test_round_trip_preserves_mlp_hook_path(tmp_path):
    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    vectors = SteeringVectors.from_sae(sae, layer=3, feature_indices=[1, 5])
    path = tmp_path / "vectors_mlp.pt"
    vectors.save(path)

    loaded = SteeringVectors.from_file(path)
    assert loaded.hook_path == "mlp"


def test_from_sae_builds_unit_norm_columns():
    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    vectors = SteeringVectors.from_sae(
        sae, layer=3, feature_indices=[1, 5], strengths=[2.0, 0.5]
    )
    assert vectors.hook_path == "mlp"
    assert vectors.layers == [3]
    assert vectors.activation_dim == 8
    for comp, (fid, strength) in zip(vectors.components, [(1, 2.0), (5, 0.5)]):
        assert comp.feature_idx == fid
        assert comp.strength == strength
        assert comp.vector.norm().item() == pytest.approx(1.0, abs=1e-6)
        expected = sae.decoder.weight[:, fid]
        expected = expected / expected.norm()
        assert torch.allclose(comp.vector, expected)


def test_from_sae_defaults_strengths_to_one():
    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    vectors = SteeringVectors.from_sae(sae, layer=0, feature_indices=[2])
    assert vectors.components[0].strength == 1.0


def test_validation_errors():
    with pytest.raises(ValueError, match="hook_path"):
        SteeringVectors([SteeringComponent(0, 0, 1.0, _unit_vector(4))], hook_path="bogus")
    with pytest.raises(ValueError, match="component"):
        SteeringVectors([])

    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    with pytest.raises(ValueError, match="same length"):
        SteeringVectors.from_sae(sae, layer=0, feature_indices=[1, 2], strengths=[1.0])
    with pytest.raises(ValueError, match="out of range"):
        SteeringVectors.from_sae(sae, layer=0, feature_indices=[999])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_steering_vectors.py -v`
Expected: FAIL with `ImportError: cannot import name 'SteeringComponent'`

- [ ] **Step 3: Implement `SteeringComponent` and `SteeringVectors`**

In `drrik/steering.py`, add `from dataclasses import dataclass` to the imports, then insert after `_sample_next_token` (before `class SAESteering`):

```python
@dataclass(frozen=True)
class SteeringComponent:
    """One steering direction: a unit-norm vector, its layer, and magnitude.

    Attributes:
        layer: Transformer layer index where the vector is injected.
        feature_idx: SAE feature index the vector was extracted from.
        strength: Absolute magnitude added along the unit vector.
        vector: Unit-norm direction of shape ``(activation_dim,)``.
    """

    layer: int
    feature_idx: int
    strength: float
    vector: torch.Tensor


class SteeringVectors:
    """Container of steering components with a shared hook path.

    Holds pre-extracted, unit-norm SAE decoder columns ready for injection
    during generation.  The hook path records where the source SAE was
    trained: ``"layer"`` (residual stream, e.g. Anthropic's
    ``resid_post_layer_*`` SAEs) or ``"mlp"`` (MLP output, e.g. SAEs
    trained by Drrik's own pipeline).

    Example:
        ```python
        vectors = SteeringVectors.from_hf_dataset(
            "shawon/llama-3.1-8b-instruct_eiffel_tower"
        )
        steering = SAESteering(source=vectors, model_name="meta-llama/Llama-3.1-8B-Instruct")
        ```
    """

    def __init__(self, components: List[SteeringComponent], hook_path: str = "layer"):
        """
        Args:
            components: Non-empty list of steering components.
            hook_path: ``"layer"`` (hook ``model.layers[N]`` output) or
                ``"mlp"`` (hook ``model.layers[N].mlp`` output).

        Raises:
            ValueError: If ``components`` is empty or ``hook_path`` is invalid.
        """
        if hook_path not in ("layer", "mlp"):
            raise ValueError(f"hook_path must be 'layer' or 'mlp', got {hook_path!r}")
        if not components:
            raise ValueError("SteeringVectors requires at least one component")
        self.components = list(components)
        self.hook_path = hook_path

    @property
    def layers(self) -> List[int]:
        """Sorted unique layer indices across all components."""
        return sorted({c.layer for c in self.components})

    @property
    def activation_dim(self) -> int:
        """Dimension of the steering vectors (all components share it)."""
        return self.components[0].vector.shape[0]

    @classmethod
    def from_sae(
        cls,
        sae: SparseAutoencoder,
        layer: int,
        feature_indices: List[int],
        strengths: Optional[List[float]] = None,
    ) -> "SteeringVectors":
        """Build components from a Drrik-trained SAE's decoder columns.

        Args:
            sae: A trained :class:`~drrik.autoencoder.SparseAutoencoder`.
            layer: Layer index the SAE was trained on.
            feature_indices: Feature indices to extract.
            strengths: Magnitude per feature. Defaults to all ``1.0``.

        Returns:
            ``SteeringVectors`` with ``hook_path="mlp"``.

        Raises:
            ValueError: If lengths mismatch or an index is out of range.
        """
        if strengths is None:
            strengths = [1.0] * len(feature_indices)
        if len(feature_indices) != len(strengths):
            raise ValueError("feature_indices and strengths must have the same length")

        components = []
        for fid, s in zip(feature_indices, strengths):
            if fid < 0 or fid >= sae.hidden_dim:
                raise ValueError(
                    f"feature_idx {fid} out of range [0, {sae.hidden_dim})"
                )
            v = sae.decoder.weight[:, fid].detach().clone().float()
            v = v / v.norm()
            components.append(SteeringComponent(layer, int(fid), float(s), v))
        return cls(components, hook_path="mlp")

    def save(self, path: Union[str, Path]) -> None:
        """Save components and hook path to a ``.pt`` file.

        Args:
            path: Destination path. Parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hook_path": self.hook_path,
                "components": [
                    {
                        "layer": c.layer,
                        "feature_idx": c.feature_idx,
                        "strength": c.strength,
                        "vector": c.vector.cpu(),
                    }
                    for c in self.components
                ],
            },
            path,
        )
        logger.info(f"Saved {len(self.components)} steering vectors to {path}")

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SteeringVectors":
        """Load components saved by :meth:`save` (hook path restored).

        Args:
            path: Path to a ``.pt`` file written by :meth:`save`.

        Returns:
            The reconstructed ``SteeringVectors``.
        """
        state = torch.load(Path(path), weights_only=False)
        components = [
            SteeringComponent(
                layer=int(c["layer"]),
                feature_idx=int(c["feature_idx"]),
                strength=float(c["strength"]),
                vector=c["vector"].float(),
            )
            for c in state["components"]
        ]
        return cls(components, hook_path=state["hook_path"])
```

- [ ] **Step 4: Export from package root**

In `drrik/__init__.py`, add `SteeringVectors` to the `drrik.steering` import and to `__all__` (match the existing style — `SAESteering` is already exported from `drrik.steering`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_steering_vectors.py -v`
Expected: all 6 tests PASS

- [ ] **Step 6: Lint, full test suite, commit**

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pytest
git add drrik/steering.py drrik/__init__.py tests/test_steering_vectors.py
git commit -m "feat: add SteeringVectors container for pre-extracted steering directions"
```

---

### Task 2: `SteeringVectors.from_hf_dataset`

**Files:**
- Modify: `drrik/steering.py` (add classmethod to `SteeringVectors`)
- Test: `tests/test_steering_vectors.py` (append)

**Interfaces:**
- Consumes: `SteeringComponent` (Task 1), `datasets` (existing dependency)
- Produces: `SteeringVectors.from_hf_dataset(source: str) -> SteeringVectors` — `source` is an HF dataset repo id **or** a local path to a parquet file; always sets `hook_path="layer"`. Expected parquet columns: `layer` (int), `feature_idx` (int), `strength` (float), `vector` (list of floats). Optional columns (`reduced_strength`, `concept`) are ignored.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_steering_vectors.py`:

```python
def test_from_hf_dataset_local_parquet(tmp_path):
    from datasets import Dataset

    parquet_path = tmp_path / "vectors.parquet"
    ds = Dataset.from_dict(
        {
            "layer": [11, 15],
            "feature_idx": [74457, 21576],
            "strength": [1.408, 1.545],
            "reduced_strength": [0.128, 0.103],
            "vector": [
                [1.0] + [0.0] * 7,
                [0.0, 1.0] + [0.0] * 6,
            ],
            "concept": ["eiffel_tower", "eiffel_tower"],
        }
    )
    ds.to_parquet(str(parquet_path))

    vectors = SteeringVectors.from_hf_dataset(str(parquet_path))
    assert vectors.hook_path == "layer"
    assert vectors.layers == [11, 15]
    assert vectors.activation_dim == 8
    assert vectors.components[0].feature_idx == 74457
    assert vectors.components[0].strength == pytest.approx(1.408)
    assert vectors.components[1].vector[1].item() == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_steering_vectors.py::test_from_hf_dataset_local_parquet -v`
Expected: FAIL with `AttributeError: ... from_hf_dataset`

- [ ] **Step 3: Implement `from_hf_dataset`**

Add to `SteeringVectors` in `drrik/steering.py`:

```python
    @classmethod
    def from_hf_dataset(cls, source: str) -> "SteeringVectors":
        """Load components from an HF dataset (or local parquet file).

        The dataset must contain ``layer``, ``feature_idx``, ``strength``,
        and ``vector`` (list of floats) columns, as produced by
        ``drrik extract-vectors``.

        Args:
            source: HF dataset repo id (e.g.
                ``"shawon/llama-3.1-8b-instruct_eiffel_tower"``) or a
                local path to a parquet file.

        Returns:
            ``SteeringVectors`` with ``hook_path="layer"``.
        """
        from datasets import load_dataset

        p = Path(source)
        if p.exists():
            ds = load_dataset("parquet", data_files=str(p))["train"]
        else:
            ds = load_dataset(source, repo_type="dataset")["train"]

        components = [
            SteeringComponent(
                layer=int(row["layer"]),
                feature_idx=int(row["feature_idx"]),
                strength=float(row["strength"]),
                vector=torch.tensor(row["vector"], dtype=torch.float32),
            )
            for row in ds
        ]
        return cls(components, hook_path="layer")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_steering_vectors.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Lint, commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add drrik/steering.py tests/test_steering_vectors.py
git commit -m "feat: add SteeringVectors.from_hf_dataset loader"
```

---

### Task 3: `SAESteering` constructor — `source` union + validation

**Files:**
- Modify: `drrik/steering.py:181-249` (`__init__`), `drrik/steering.py:579-648` (`find_steering_features` guard), `drrik/steering.py:650-794` (`build_token_feature_map` guard), module/class docstring examples
- Modify: `examples/sae_steering.py:244` (`sae=sae` → `source=sae`)
- Modify: `examples/steer_paris.py:73` (`sae=sae` → `source=sae`)
- Modify: `README.md:303,405` (`sae=sae` → `source=sae` in code blocks)
- Test: `tests/test_steering_generation.py` (create)

**Interfaces:**
- Consumes: `SparseAutoencoder`, `SteeringVectors` (Tasks 1-2)
- Produces:
  - `SAESteering(source, model_name, layer=None, ...)` — `source` is `SparseAutoencoder` (then `layer` required) or `SteeringVectors` (then `layer` ignored). Instance attrs: `.source`, `.sae` (or `None`), `.vectors` (or `None`), `.target_layer` (or `None`), `.hook_path` (`"mlp"` for SAE source, `source.hook_path` for vectors)
  - SAE-only methods (`find_steering_features`, `build_token_feature_map`) raise `TypeError` when `self.sae is None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_steering_generation.py`:

```python
"""Tests for SAESteering steering mechanics (no model loading, CPU only)."""

import pytest
import torch

from drrik.autoencoder import SparseAutoencoder
from drrik.steering import SAESteering, SteeringComponent, SteeringVectors


def _unit_vector(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _vectors_instance() -> SAESteering:
    """SAESteering built via __new__ with a vectors source (no model)."""
    comps = [SteeringComponent(5, 42, 1.5, _unit_vector(8, seed=3))]
    steering = SAESteering.__new__(SAESteering)
    steering.source = SteeringVectors(comps, hook_path="layer")
    steering.sae = None
    steering.vectors = steering.source
    steering.target_layer = None
    steering.hook_path = "layer"
    steering.device = torch.device("cpu")
    return steering


def test_constructor_requires_layer_for_sae():
    sae = SparseAutoencoder(activation_dim=8, hidden_dim=16)
    with pytest.raises(ValueError, match="layer is required"):
        SAESteering(sae, "meta-llama/Llama-3.1-8B-Instruct")


def test_constructor_rejects_unknown_source():
    with pytest.raises(TypeError, match="SparseAutoencoder or SteeringVectors"):
        SAESteering(42, "meta-llama/Llama-3.1-8B-Instruct")


def test_sae_only_methods_raise_on_vectors_source():
    steering = _vectors_instance()
    with pytest.raises(TypeError, match="SparseAutoencoder"):
        steering.find_steering_features("Paris", torch.zeros(2, 8))
    with pytest.raises(TypeError, match="SparseAutoencoder"):
        steering.build_token_feature_map(tokens=[1, 2])
```

Note: `test_constructor_requires_layer_for_sae` and `test_constructor_rejects_unknown_source` must raise **before** any model download — the validation code goes at the top of `__init__`, before `LanguageModel(...)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_steering_generation.py -v`
Expected: FAIL — `test_constructor_requires_layer_for_sae` fails (no validation; it tries to load the model or accepts `layer=None`), `test_constructor_rejects_unknown_source` fails, `test_sae_only_methods_raise_on_vectors_source` fails (no guard)

- [ ] **Step 3: Rewrite `__init__`**

In `drrik/steering.py`, replace the `__init__` signature and its first ~15 lines (up to and including `self.device = next(sae.parameters()).device`) with:

```python
    def __init__(
        self,
        source: Union[SparseAutoencoder, SteeringVectors],
        model_name: str,
        layer: Optional[int] = None,
        revision: str = "main",
        torch_dtype: str = "float16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        token: Optional[str] = None,
    ):
        """
        Initialize the SAE steering controller.

        Args:
            source: Steering direction source — either a trained
                :class:`~drrik.autoencoder.SparseAutoencoder` (then
                ``layer`` is required) or a
                :class:`SteeringVectors` of pre-extracted components
                (then ``layer`` is ignored; layers come from the
                components).
            model_name: HuggingFace model identifier for generation.
            layer: Transformer layer index for a ``SparseAutoencoder``
                source. Ignored for ``SteeringVectors``.
            revision: Model revision to load.
            torch_dtype: Weight dtype for model loading.
            device_map: Device mapping strategy.
            trust_remote_code: Whether to trust remote code from the repo.
            token: HuggingFace token for gated models. Falls back to
                ``HUGGINGFACE_HUB_TOKEN`` from ``settings.yml`` when ``None``.

        Raises:
            TypeError: If ``source`` is neither type.
            ValueError: If ``source`` is a ``SparseAutoencoder`` and
                ``layer`` is ``None``.
        """
        if isinstance(source, SparseAutoencoder):
            if layer is None:
                raise ValueError(
                    "layer is required when source is a SparseAutoencoder"
                )
            self.sae: Optional[SparseAutoencoder] = source
            self.vectors: Optional[SteeringVectors] = None
            self.target_layer: Optional[int] = layer
            self.hook_path = "mlp"
            self.device = next(source.parameters()).device
        elif isinstance(source, SteeringVectors):
            self.sae = None
            self.vectors = source
            self.target_layer = None
            self.hook_path = source.hook_path
            self.device = source.components[0].vector.device
        else:
            raise TypeError(
                "source must be a SparseAutoencoder or SteeringVectors, "
                f"got {type(source).__name__}"
            )
        self.source = source
```

Keep the rest of `__init__` (token fallback, `LanguageModel` load, tokenizer) unchanged. Update the class docstring example and the module docstring example (`SAESteering(sae, model_name=...)` → `SAESteering(source=sae, model_name=...)`, and add one `SteeringVectors` example).

- [ ] **Step 4: Add guards to SAE-only methods**

At the top of `find_steering_features` (before `self.sae.eval()`):

```python
        if self.sae is None:
            raise TypeError(
                "find_steering_features requires a SparseAutoencoder source; "
                "this instance was built from SteeringVectors (no encoder)"
            )
```

At the top of `build_token_feature_map` (before the token-determination code):

```python
        if self.sae is None:
            raise TypeError(
                "build_token_feature_map requires a SparseAutoencoder source; "
                "this instance was built from SteeringVectors (no encoder)"
            )
```

- [ ] **Step 5: Update keyword callers**

- `examples/sae_steering.py:244`: `sae=sae,` → `source=sae,`
- `examples/steer_paris.py:73`: `sae=sae,` → `source=sae,`
- `README.md` lines 303 and 405: `sae=sae,` → `source=sae,`

(Do NOT touch `sae=sae` in `FeatureVisualizer(...)` calls — different class.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_steering_generation.py tests/test_steering_vectors.py tests/test_token_feature_map.py -v`
Expected: all PASS (the `__new__`-based token-map test still works — it sets `steering.sae` directly)

- [ ] **Step 7: Lint, full suite, commit**

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pytest
git add drrik/steering.py examples/sae_steering.py examples/steer_paris.py README.md tests/test_steering_generation.py
git commit -m "feat: SAESteering accepts SteeringVectors via source arg"
```

---

### Task 4: Steering hook factory — multi-layer + clamping

**Files:**
- Modify: `drrik/steering.py` (add module-level `make_steering_hook` after `SteeringVectors`; rewrite `_generate_with_hooks` to use it)
- Test: `tests/test_steering_generation.py` (append)

**Interfaces:**
- Consumes: `SteeringComponent` (Task 1)
- Produces:
  - `make_steering_hook(components: List[Tuple[float, torch.Tensor]], clamp: bool) -> Callable` — module-level; `components` is a list of `(strength, unit_vector)` for ONE layer; returns a forward hook compatible with `nn.Module.register_forward_hook`
  - `SAESteering._generate_with_hooks(input_ids, attention_mask, layer_components, max_new_tokens, temperature, top_p, repetition_penalty, clamp, seed) -> str` where `layer_components: Dict[int, List[Tuple[float, torch.Tensor]]]` maps layer index → `(strength, vector)` pairs. (Full wiring of the generation loop lands in Task 6; this task implements the hook + the loop body.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_steering_generation.py`:

```python
from drrik.steering import make_steering_hook


def test_hook_additive_3d():
    hook = make_steering_hook([(1.5, _unit_vector(8, seed=5))], clamp=False)
    x = torch.randn(2, 3, 8)
    out = hook(None, None, x)
    expected = x + 1.5 * _unit_vector(8, seed=5)
    assert torch.allclose(out, expected)


def test_hook_clamp_sets_exact_projection():
    v = _unit_vector(8, seed=6)
    hook = make_steering_hook([(1.5, v)], clamp=True)
    x = torch.randn(2, 3, 8)
    out = hook(None, None, x)
    proj = torch.einsum("bsh,h->bs", out, v)
    assert torch.allclose(proj, torch.full((2, 3), 1.5), atol=1e-5)


def test_hook_additive_preserves_existing_projection_plus_strength():
    v = _unit_vector(8, seed=7)
    hook = make_steering_hook([(1.5, v)], clamp=False)
    x = torch.randn(2, 3, 8)
    out = hook(None, None, x)
    old_proj = torch.einsum("bsh,h->bs", x, v)
    new_proj = torch.einsum("bsh,h->bs", out, v)
    assert torch.allclose(new_proj, old_proj + 1.5, atol=1e-5)


def test_hook_handles_2d_and_tuple_output():
    hook = make_steering_hook([(1.0, _unit_vector(8, seed=8))], clamp=False)
    x2d = torch.randn(2, 8)
    out2d = hook(None, None, x2d)
    assert out2d.shape == (2, 8)

    extra = torch.randn(2, 3, 4)
    out_tuple = hook(None, None, (x2d, extra))
    assert isinstance(out_tuple, tuple) and len(out_tuple) == 2
    assert torch.allclose(out_tuple[1], extra)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_steering_generation.py -k hook -v`
Expected: FAIL with `ImportError: cannot import name 'make_steering_hook'`

- [ ] **Step 3: Implement `make_steering_hook`**

Add to `drrik/steering.py` (module level, after the `SteeringVectors` class):

```python
def make_steering_hook(
    components: List[Tuple[float, torch.Tensor]], clamp: bool
):
    """Build a forward hook that injects steering vectors into a layer output.

    For each ``(strength, vector)`` pair the hook adds ``strength * vector``
    to the hidden states.  With ``clamp=True`` the hidden state's existing
    projection onto the vector is removed first, so the component along the
    vector becomes exactly ``strength`` (the scheme used in Anthropic's
    Golden Gate demo and the Eiffel Tower Llama reproduction).

    Handles 2D ``(batch, hidden)`` and 3D ``(batch, seq, hidden)`` hidden
    states and tuple module outputs.

    Args:
        components: ``(strength, unit_vector)`` pairs for one layer.
        clamp: Whether to remove the existing projection before adding.

    Returns:
        A forward-hook callable for ``register_forward_hook``.
    """

    def hook(module, input_args, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not isinstance(hidden, torch.Tensor):
            return output

        was_2d = hidden.dim() == 2
        if was_2d:
            hidden = hidden.unsqueeze(1)

        for strength, vector in components:
            v = vector.to(dtype=hidden.dtype, device=hidden.device)
            if clamp:
                proj = torch.einsum("bsh,h->bs", hidden, v).unsqueeze(-1) * v
                hidden = hidden - proj
            hidden = hidden + strength * v

        if was_2d:
            hidden = hidden.squeeze(1)
        if rest is not None:
            return (hidden,) + rest
        return hidden

    return hook
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_steering_generation.py -k hook -v`
Expected: all 4 hook tests PASS

- [ ] **Step 5: Lint, commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add drrik/steering.py tests/test_steering_generation.py
git commit -m "feat: add make_steering_hook with clamping support"
```

---

### Task 5: `_sample_next_token` — repetition penalty + seed

**Files:**
- Modify: `drrik/steering.py:100-131` (`_sample_next_token`)
- Test: `tests/test_steering_generation.py` (append)

**Interfaces:**
- Consumes: nothing new
- Produces: `_sample_next_token(logits, temperature, top_p, repetition_penalty=1.0, generated_tokens=None, generator=None) -> torch.Tensor` — CTRL-style penalty: for token ids in `generated_tokens`, positive logits are divided by `repetition_penalty`, negative logits multiplied by it. `generator` is forwarded to `torch.multinomial` for reproducibility.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_steering_generation.py`:

```python
from drrik.steering import _sample_next_token


def test_repetition_penalty_suppresses_repeated_token():
    # Token 0 starts as argmax; a heavy penalty must push it out.
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
    generated = torch.tensor([[0]])
    samples = torch.stack(
        [
            _sample_next_token(
                logits, 1.0, 1.0,
                repetition_penalty=1000.0,
                generated_tokens=generated,
            ).item()
            for _ in range(200)
        ]
    )
    assert (samples == 0).sum().item() == 0


def test_repetition_penalty_negative_logits():
    # All logits negative; penalizing token 0 makes it even less likely.
    logits = torch.tensor([[-2.0, -1.0, -0.5, -0.1]])
    generated = torch.tensor([[0]])
    samples = torch.stack(
        [
            _sample_next_token(
                logits, 1.0, 1.0,
                repetition_penalty=1000.0,
                generated_tokens=generated,
            ).item()
            for _ in range(200)
        ]
    )
    assert (samples == 0).sum().item() == 0


def test_no_penalty_leaves_distribution_untouched():
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    a = _sample_next_token(logits, 1.0, 1.0, repetition_penalty=1.0, generator=gen_a)
    b = _sample_next_token(logits, 1.0, 1.0, generator=gen_b)
    assert a.item() == b.item()


def test_seed_reproducible_sampling():
    logits = torch.randn(1, 64)
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    a = _sample_next_token(logits, 0.8, 0.9, generator=gen_a)
    b = _sample_next_token(logits, 0.8, 0.9, generator=gen_b)
    assert a.item() == b.item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_steering_generation.py -k "repetition or seed or penalty" -v`
Expected: FAIL with `TypeError: _sample_next_token() got an unexpected keyword argument 'repetition_penalty'`

- [ ] **Step 3: Implement**

Replace `_sample_next_token` in `drrik/steering.py`:

```python
def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply repetition penalty, temperature, top-p, and sample a token.

    Used by both :meth:`SAESteering._generate_with_hooks` and
    :meth:`SAESteering._generate_baseline`.

    Args:
        logits: Raw logits of shape ``(1, vocab_size)``.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold (0–1).
        repetition_penalty: CTRL-style penalty for already-generated
            tokens. ``1.0`` disables it. Positive logits of seen tokens
            are divided by the penalty, negative logits multiplied.
        generated_tokens: All tokens generated so far (any shape); its
            unique ids receive the penalty. ``None`` disables it.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Sampled token tensor of shape ``(1, 1)``.
    """
    if (
        repetition_penalty != 1.0
        and generated_tokens is not None
        and generated_tokens.numel() > 0
    ):
        seen = generated_tokens.unique()
        logits = logits.clone()
        penalized = logits[:, seen]
        penalized = torch.where(
            penalized > 0,
            penalized / repetition_penalty,
            penalized * repetition_penalty,
        )
        logits[:, seen] = penalized

    logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float("-inf")

    return torch.multinomial(
        F.softmax(logits, dim=-1), num_samples=1, generator=generator
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_steering_generation.py -v`
Expected: all tests in the file PASS

- [ ] **Step 5: Lint, commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add drrik/steering.py tests/test_steering_generation.py
git commit -m "feat: add repetition penalty and seeded sampling to _sample_next_token"
```

---

### Task 6: `generate()` — new params, chat template, multi-layer wiring

**Files:**
- Modify: `drrik/steering.py` — `generate()`, `_generate_with_hooks()`, `_generate_baseline()`, add `_build_prompt_ids()`
- Test: `tests/test_steering_generation.py` (append)

**Interfaces:**
- Consumes: `make_steering_hook` (Task 4), `_sample_next_token` (Task 5), `resolve_module_path` (existing)
- Produces:
  - `SAESteering.generate(text, feature_idx=None, strength=1.0, max_new_tokens=50, temperature=0.8, top_p=0.9, feature_indices=None, strengths=None, clamp=False, repetition_penalty=1.0, system_prompt=None, strength_scale=1.0, seed=None) -> str`
  - `SAESteering._build_prompt_ids(text: str, system_prompt: Optional[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`
  - Behavior: SAE source → single-layer components at `target_layer`, hook `"mlp"` (existing behavior, new params default to off). Vectors source → all components scaled by `strength_scale`; `strength_scale=0.0` runs the baseline loop; passing `feature_idx`/`feature_indices` with a vectors source raises `ValueError`. `seed` calls `torch.manual_seed(seed)` before the loop (ponytail: global seed — the generation loop is the only RNG consumer in the call; per-device generators if that ever bites).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_steering_generation.py`:

```python
class _StubTokenizer:
    pad_token = None
    eos_token_id = 0
    model_max_length = 1000
    all_special_ids = []
    vocab_size = 100

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def apply_chat_template(self, messages, return_tensors="pt", add_generation_prompt=True):
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return torch.tensor([[9, 9, 9]])


def test_build_prompt_ids_plain_and_chat():
    from unittest.mock import MagicMock

    steering = SAESteering.__new__(SAESteering)
    steering.tokenizer = _StubTokenizer()
    steering.model = MagicMock()
    steering.model.device = torch.device("cpu")

    ids, mask = steering._build_prompt_ids("hello", system_prompt=None)
    assert ids.tolist() == [[1, 2, 3]]

    ids, _ = steering._build_prompt_ids("hello", system_prompt="You are a helpful assistant.")
    assert ids.tolist() == [[9, 9, 9]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_steering_generation.py::test_build_prompt_ids_plain_and_chat -v`
Expected: FAIL with `AttributeError: 'SAESteering' object has no attribute '_build_prompt_ids'`

- [ ] **Step 3: Implement `_build_prompt_ids`**

Add to `SAESteering` (next to `_tokenize`):

```python
    def _build_prompt_ids(
        self, text: str, system_prompt: Optional[str]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Tokenize a prompt, optionally through the chat template.

        Args:
            text: User prompt text.
            system_prompt: When set, the prompt is rendered as a
                system+user chat via ``apply_chat_template`` (required
                for instruct models).

        Returns:
            ``(input_ids, attention_mask)`` on the model device.
        """
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            return input_ids.to(self.model.device), None
        return self._tokenize(text)
```

Add `Tuple` to the `typing` import if not already present.

- [ ] **Step 4: Rewrite `generate()`**

Replace the body of `generate()` (keep the docstring, extend it with the new params):

```python
        if self.vectors is not None:
            if feature_idx is not None or feature_indices is not None:
                raise ValueError(
                    "feature selection is fixed by the SteeringVectors source; "
                    "use strength_scale to adjust magnitude"
                )
            layer_components = {}
            for c in self.vectors.components:
                layer_components.setdefault(c.layer, []).append(
                    (c.strength * strength_scale, c.vector)
                )
            if strength_scale == 0.0:
                return self._generate_baseline(
                    text, max_new_tokens, temperature, top_p,
                    repetition_penalty, seed, system_prompt,
                )
        else:
            if feature_idx is None and feature_indices is None:
                return self._generate_baseline(
                    text, max_new_tokens, temperature, top_p,
                    repetition_penalty, seed, system_prompt,
                )
            if feature_idx is not None:
                feature_indices = [feature_idx]
                strengths = [strength]
            elif strengths is None:
                strengths = [1.0] * len(feature_indices)
            if len(feature_indices) != len(strengths):
                raise ValueError(
                    "feature_indices and strengths must have the same length"
                )
            pairs = []
            for fid, s in zip(feature_indices, strengths):
                if fid < 0 or fid >= self.sae.hidden_dim:
                    raise ValueError(
                        f"feature_idx {fid} out of range [0, {self.sae.hidden_dim})"
                    )
                pairs.append(
                    (s * strength_scale, self.sae.decoder.weight[:, fid].detach())
                )
            layer_components = {self.target_layer: pairs}

        input_ids, attention_mask = self._build_prompt_ids(text, system_prompt)

        return self._generate_with_hooks(
            input_ids,
            attention_mask,
            layer_components,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
            clamp,
            seed,
        )
```

- [ ] **Step 5: Rewrite `_generate_with_hooks`**

Replace the method (signature + body). Keep the docstring, update Args:

```python
    def _generate_with_hooks(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        layer_components: Dict[int, List[Tuple[float, torch.Tensor]]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        clamp: bool,
        seed: Optional[int],
    ) -> str:
        """Generate text with forward-hook steering on one or more layers.

        Args:
            input_ids: Tokenized input of shape ``(1, seq_len)``.
            attention_mask: Optional attention mask.
            layer_components: Map of layer index to ``(strength, vector)``
                pairs injected at that layer.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repetition_penalty: CTRL-style repetition penalty (1.0 off).
            clamp: Remove existing projection before adding (see
                :func:`make_steering_hook`).
            seed: Optional RNG seed for reproducible sampling.

        Returns:
            The decoded text (special tokens stripped).
        """
        generated_tokens = input_ids.clone()
        device = generated_tokens.device
        hook_handles = []
        # ponytail: device-bound generator for reproducibility; torch.manual_seed
        # (global) if per-device generators ever misbehave on MPS
        generator = (
            torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        )

        try:
            for layer, comps in layer_components.items():
                if self.hook_path == "layer":
                    path = f"model.layers[{layer}]"
                else:
                    path = f"model.layers[{layer}].mlp"
                target_module = resolve_module_path(self.model, path)
                hook_handles.append(
                    target_module.register_forward_hook(make_steering_hook(comps, clamp))
                )

            base_model = self.model._module

            for _ in tqdm(range(max_new_tokens), desc="Steering generation"):
                if generated_tokens.shape[-1] >= self.tokenizer.model_max_length:
                    break

                outputs = base_model(
                    input_ids=generated_tokens,
                    attention_mask=attention_mask,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]
                next_token = _sample_next_token(
                    logits,
                    temperature,
                    top_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=generated_tokens,
                    generator=generator,
                )

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones(1, 1, device=device)], dim=-1
                    )
        finally:
            for handle in hook_handles:
                handle.remove()

        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
```

- [ ] **Step 6: Update `_generate_baseline`**

New signature: `_generate_baseline(self, text, max_new_tokens, temperature, top_p, repetition_penalty=1.0, seed=None, system_prompt=None)`. Body changes:

```python
        input_ids, attention_mask = self._build_prompt_ids(text, system_prompt)

        base_model = self.model._module
        generated_tokens = input_ids.clone()
        device = generated_tokens.device
        generator = (
            torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        )

        with torch.no_grad():
            for _ in tqdm(range(max_new_tokens), desc="Baseline generation"):
                if generated_tokens.shape[-1] >= self.tokenizer.model_max_length:
                    break

                outputs = base_model(
                    input_ids=generated_tokens,
                    attention_mask=attention_mask,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]
                next_token = _sample_next_token(
                    logits,
                    temperature,
                    top_p,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=generated_tokens,
                    generator=generator,
                )

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones(1, 1, device=device)], dim=-1
                    )

        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/ -v`
Expected: ALL tests pass (new + existing). The old positional call sites of `_generate_with_hooks`/`_generate_baseline` are all internal and updated in this task.

- [ ] **Step 8: Lint, commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add drrik/steering.py tests/test_steering_generation.py
git commit -m "feat: multi-layer steering with clamping, repetition penalty, chat prompts, seed"
```

---

### Task 7: `drrik extract-vectors` CLI command

**Files:**
- Modify: `drrik/cli.py` (add `extract_decoder_columns` helper + `extract-vectors` command)
- Create: `examples/eiffel_tower_recipe.yaml`
- Test: `tests/test_extract_vectors.py` (create)

**Interfaces:**
- Consumes: `SteeringVectors` (Task 1), `hf_hub_download` (huggingface_hub, already available via transformers/nnsight deps), `get_settings` (existing)
- Produces:
  - `extract_decoder_columns(state_dict: dict, activation_dim: int, feature_indices: List[int]) -> List[torch.Tensor]` — module-level in `cli.py`; finds the decoder weight by shape `(activation_dim, D)` with `D > activation_dim`, returns L2-normalized columns
  - CLI: `drrik extract-vectors --config <recipe.yaml> [--out <path.pt>] [--repo-id <user/name>]`
  - Recipe YAML schema: `model_name`, `sae_repo`, `trainer`, `activation_dim`, `concept`, `features: [[layer, feature_idx, reduced_strength], ...]`
  - Pushed dataset columns: `layer`, `feature_idx`, `strength` (absolute = reduced × layer), `reduced_strength`, `vector`, `concept`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_extract_vectors.py`:

```python
"""Tests for extract-vectors pure helpers (no network)."""

import pytest
import torch

from drrik.cli import extract_decoder_columns


def test_extract_decoder_columns_by_shape():
    decoder = torch.randn(8, 16)
    encoder = torch.randn(16, 8)
    state = {"decoder.weight": decoder, "encoder.weight": encoder}

    cols = extract_decoder_columns(state, activation_dim=8, feature_indices=[1, 3])
    assert len(cols) == 2
    for col, fid in zip(cols, [1, 3]):
        assert col.shape == (8,)
        assert col.norm().item() == pytest.approx(1.0, abs=1e-6)
        expected = decoder[:, fid]
        expected = expected / expected.norm()
        assert torch.allclose(col, expected)


def test_extract_decoder_columns_prefers_decoder_not_encoder():
    # Encoder is (16, 8): shape[0] != activation_dim, so it must not match.
    decoder = torch.randn(8, 16)
    state = {"enc": torch.randn(16, 8), "dec": decoder}
    cols = extract_decoder_columns(state, activation_dim=8, feature_indices=[0])
    expected = decoder[:, 0]
    expected = expected / expected.norm()
    assert torch.allclose(cols[0], expected)


def test_extract_decoder_columns_not_found():
    with pytest.raises(ValueError, match="decoder weight"):
        extract_decoder_columns({"w": torch.randn(4, 4)}, activation_dim=8, feature_indices=[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract_vectors.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_decoder_columns'`

- [ ] **Step 3: Implement helper + command**

In `drrik/cli.py`, add near the top (after imports):

```python
def extract_decoder_columns(
    state_dict: dict, activation_dim: int, feature_indices: List[int]
) -> List[torch.Tensor]:
    """Extract L2-normalized decoder columns from an SAE state dict.

    The decoder weight is located by shape: ``(activation_dim, D)`` with
    ``D > activation_dim`` (the encoder is the transpose-shaped matrix and
    does not match).

    Args:
        state_dict: Loaded ``ae.pt`` state dict (any key naming).
        activation_dim: Model hidden size (e.g. 4096 for Llama 3.1 8B).
        feature_indices: Feature columns to extract.

    Returns:
        List of unit-norm tensors of shape ``(activation_dim,)``.

    Raises:
        ValueError: If no decoder-shaped tensor is found.
    """
    decoder = None
    for tensor in state_dict.values():
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dim() == 2
            and tensor.shape[0] == activation_dim
            and tensor.shape[1] > activation_dim
        ):
            decoder = tensor
            break
    if decoder is None:
        raise ValueError(
            f"no decoder weight of shape ({activation_dim}, >{activation_dim}) "
            f"found in state dict keys {list(state_dict)}"
        )
    return [
        (decoder[:, fid].clone().float() / decoder[:, fid].norm())
        for fid in feature_indices
    ]
```

Add the command (after the existing `run` command, before `def main()`):

```python
@cli.command("extract-vectors")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Recipe YAML with features, sae_repo, trainer, activation_dim, concept",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(),
    default=None,
    help="Local output .pt path (default: drrik_output/vectors/<concept>.pt)",
)
@click.option(
    "--repo-id",
    default=None,
    help="HF dataset repo id to push, e.g. shawon/llama-3.1-8b-instruct_eiffel_tower",
)
def extract_vectors(config: Path, out: Optional[Path], repo_id: Optional[str]):
    """
    Extract steering vectors from pre-trained SAEs and optionally publish them.

    Downloads one ~4GB ae.pt per unique layer from the SAE repo, extracts the
    requested decoder columns (unit-norm, strength = reduced x layer), saves a
    small .pt file, and optionally pushes a parquet dataset to the HF Hub.
    """
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    with open(config) as f:
        recipe = yaml.safe_load(f)

    sae_repo = recipe["sae_repo"]
    trainer = recipe.get("trainer", "trainer_1")
    activation_dim = int(recipe["activation_dim"])
    concept = recipe["concept"]
    features = [tuple(row) for row in recipe["features"]]

    by_layer: Dict[int, List[tuple]] = {}
    for layer, fid, reduced in features:
        by_layer.setdefault(int(layer), []).append((int(fid), float(reduced)))

    components = []
    for layer in sorted(by_layer):
        logger.info(f"Downloading SAE for layer {layer} from {sae_repo} ...")
        ae_path = hf_hub_download(
            sae_repo, f"resid_post_layer_{layer}/{trainer}/ae.pt"
        )
        state = torch.load(ae_path, map_location="cpu", weights_only=False)
        fids = [fid for fid, _ in by_layer[layer]]
        columns = extract_decoder_columns(state, activation_dim, fids)
        for (fid, reduced), col in zip(by_layer[layer], columns):
            components.append(
                SteeringComponent(layer, fid, reduced * layer, col)
            )
        del state  # free the ~4GB before the next layer

    vectors = SteeringVectors(components, hook_path="layer")

    out_path = Path(out) if out else Path("drrik_output/vectors") / f"{concept}.pt"
    vectors.save(out_path)
    logger.info(f"Saved {len(components)} vectors to {out_path}")

    if repo_id:
        token = get_settings().huggingface_hub_token
        if not token:
            raise click.ClickException(
                "huggingface_hub_token missing from settings.yml; cannot push"
            )
        ds = Dataset.from_dict(
            {
                "layer": [c.layer for c in components],
                "feature_idx": [c.feature_idx for c in components],
                "strength": [c.strength for c in components],
                "reduced_strength": [
                    c.strength / c.layer for c in components
                ],
                "vector": [c.vector.tolist() for c in components],
                "concept": [concept] * len(components),
            }
        )
        ds.push_to_hub(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            allow_patterns=["*.parquet"],
        )
        logger.info(f"Pushed dataset to hf.co/datasets/{repo_id}")
```

Add imports at the top of `cli.py` if missing: `from drrik.steering import SteeringComponent, SteeringVectors` (check existing imports first — `SparseAutoencoder` is already imported from `drrik.autoencoder`).

Create `examples/eiffel_tower_recipe.yaml`:

```yaml
# Eiffel Tower steering recipe (from the official eiffel-tower-llama demo)
# Strengths are REDUCED values; absolute strength = reduced x layer.
model_name: meta-llama/Llama-3.1-8B-Instruct
sae_repo: andyrdt/saes-llama-3.1-8b-instruct
trainer: trainer_1
activation_dim: 4096
concept: eiffel_tower
features:
  - [11, 74457, 0.128]
  - [11, 18894, 0.255]
  - [11, 61463, 0.132]
  - [15, 21576, 0.103]
  - [19, 93, 0.459]
  - [23, 111898, 0.466]
  - [23, 40788, 0.228]
  - [23, 21334, 0.043]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extract_vectors.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Verify CLI wiring**

Run: `uv run drrik extract-vectors --help`
Expected: shows the `--config`, `--out`, `--repo-id` options (no download happens)

- [ ] **Step 6: Lint, full suite, commit**

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pytest
git add drrik/cli.py examples/eiffel_tower_recipe.yaml tests/test_extract_vectors.py
git commit -m "feat: add drrik extract-vectors command and Eiffel Tower recipe"
```

---

### Task 8: `drrik steer` CLI + `examples/eiffel_tower.py`

**Files:**
- Modify: `drrik/cli.py` (add `steer` command + `load_steering_config` helper)
- Create: `examples/eiffel_tower.py`
- Test: `tests/test_extract_vectors.py` (append config test — rename file mentally: it now covers both new commands' pure helpers)

**Interfaces:**
- Consumes: `SteeringVectors` (Tasks 1-2), `SAESteering` (Tasks 3-6)
- Produces:
  - `load_steering_config(path: Path) -> dict` — validates required keys `model`, `vectors`, `prompts`; returns the parsed dict
  - CLI: `drrik steer --config <steering.yaml> [--output-dir <path>]`
  - Steering config schema: `model` (str), `vectors` (HF repo id or local .pt path), `prompts` (list of str), optional `generation` dict (`max_new_tokens`, `temperature`, `top_p`, `repetition_penalty`, `system_prompt`, `strength_scales`, `seed`), optional `model_kwargs` dict forwarded to `SAESteering`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extract_vectors.py`:

```python
def test_load_steering_config_valid(tmp_path):
    from drrik.cli import load_steering_config

    cfg_file = tmp_path / "steering.yaml"
    cfg_file.write_text(
        "model: meta-llama/Llama-3.1-8B-Instruct\n"
        "vectors: shawon/llama-3.1-8b-instruct_eiffel_tower\n"
        'prompts: ["The weather today is"]\n'
    )
    cfg = load_steering_config(cfg_file)
    assert cfg["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg["prompts"] == ["The weather today is"]


def test_load_steering_config_missing_keys(tmp_path):
    import pytest

    from drrik.cli import load_steering_config

    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("model: x\n")
    with pytest.raises(ValueError, match="missing required keys"):
        load_steering_config(cfg_file)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract_vectors.py -k steering_config -v`
Expected: FAIL with `ImportError: cannot import name 'load_steering_config'`

- [ ] **Step 3: Implement helper + command**

In `drrik/cli.py`:

```python
def load_steering_config(path: Path) -> dict:
    """Load and validate a steering config YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dict.

    Raises:
        ValueError: If required keys (model, vectors, prompts) are missing.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in ("model", "vectors", "prompts") if k not in cfg]
    if missing:
        raise ValueError(f"steering config missing required keys: {missing}")
    return cfg


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Steering config YAML (model, vectors, prompts, generation)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="drrik_output/steering",
    help="Where to save the steering analysis text file",
)
def steer(config: Path, output_dir: Path):
    """
    Run baseline vs steered generation from a steering config.

    Loads steering vectors (HF dataset repo id or local .pt), generates each
    prompt at every strength scale (0.0 = baseline), prints results, and
    saves a timestamped analysis file.
    """
    cfg = load_steering_config(config)
    gen = cfg.get("generation", {})
    scales = gen.get("strength_scales", [0.0, 1.0])

    vectors_src = cfg["vectors"]
    if Path(vectors_src).exists():
        vectors = SteeringVectors.from_file(vectors_src)
    else:
        vectors = SteeringVectors.from_hf_dataset(vectors_src)

    steering = SAESteering(
        source=vectors, model_name=cfg["model"], **cfg.get("model_kwargs", {})
    )

    results = {}
    for prompt in cfg["prompts"]:
        for scale in scales:
            label = "baseline" if scale == 0.0 else f"strength_{scale}"
            logger.info(f"Generating: {prompt!r} [{label}]")
            text = steering.generate(
                prompt,
                strength_scale=scale,
                max_new_tokens=gen.get("max_new_tokens", 128),
                temperature=gen.get("temperature", 0.5),
                top_p=gen.get("top_p", 0.9),
                repetition_penalty=gen.get("repetition_penalty", 1.2),
                system_prompt=gen.get("system_prompt"),
                seed=gen.get("seed"),
            )
            key = f"{prompt[:40]} | {label}"
            results[key] = text
            print(f"\n--- {key} ---\n{text}\n")

    steering.save_steering_analysis(
        results, output_dir, prompt="; ".join(cfg["prompts"]), feature_label="steer"
    )
```

Add `SAESteering` to the `drrik.steering` import in `cli.py` if not already imported.

Create `examples/eiffel_tower.py`:

```python
"""
Steer Llama 3.1 8B Instruct to talk about the Eiffel Tower.

Reproduces the Eiffel Tower Llama demo (dlouapre/eiffel-tower-llama) using
pre-extracted steering vectors from Anthropic's Llama 3.1 8B SAEs, published
as the HF dataset ``shawon/llama-3.1-8b-instruct_eiffel_tower``.

Requirements:
    - ~20GB free RAM for Llama 3.1 8B in fp16 (Apple Silicon: use
      device_map="auto"; fall back to "cpu" if MPS runs out of memory)
    - The vectors dataset (run ``drrik extract-vectors`` once to create it)
    - settings.yml with HUGGINGFACE_HUB_TOKEN (Llama is gated)

Usage:
    uv run examples/eiffel_tower.py
"""

from loguru import logger

from drrik import SAESteering
from drrik.steering import SteeringVectors

DATASET = "shawon/llama-3.1-8b-instruct_eiffel_tower"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."
OUTPUT_DIR = "drrik_output/steering_eiffel"

PROMPTS = [
    "The weather today is",
    "My favorite programming language is",
    "The best food in the world is",
    "I love spending my weekends",
    "The most beautiful city is",
]


def main():
    logger.info(f"Loading steering vectors from {DATASET}")
    vectors = SteeringVectors.from_hf_dataset(DATASET)

    logger.info(f"Loading {MODEL_NAME}")
    steering = SAESteering(source=vectors, model_name=MODEL_NAME)

    results = {}
    for prompt in PROMPTS:
        logger.info(f"Prompt: {prompt!r}")
        for scale, label in [(0.0, "baseline"), (1.0, "steered")]:
            text = steering.generate(
                prompt,
                strength_scale=scale,
                max_new_tokens=128,
                temperature=0.5,
                repetition_penalty=1.2,
                system_prompt=SYSTEM_PROMPT,
                seed=16,
            )
            results[f"{prompt} | {label}"] = text
            logger.info(f"{label}: {text[:200]}")

    steering.save_steering_analysis(
        results, OUTPUT_DIR, prompt="; ".join(PROMPTS), feature_label="eiffel_tower"
    )
    logger.info(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extract_vectors.py -v`
Expected: all tests PASS

- [ ] **Step 5: Verify CLI wiring**

Run: `uv run drrik steer --help`
Expected: shows `--config` and `--output-dir` options

- [ ] **Step 6: Lint, full suite, commit**

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pytest
git add drrik/cli.py examples/eiffel_tower.py tests/test_extract_vectors.py
git commit -m "feat: add drrik steer command and Eiffel Tower example"
```

---

### Task 9: Phase 2 — Gradio app + optional extra

**Files:**
- Modify: `pyproject.toml` (add `app` optional-dependencies)
- Create: `examples/eiffel_tower_app.py`
- Test: `tests/test_app_import.py` (create)

**Interfaces:**
- Consumes: `SAESteering`, `SteeringVectors` (Tasks 1-6)
- Produces: `examples/eiffel_tower_app.py` with module-level constants (`DATASET`, `MODEL_NAME`, `SYSTEM_PROMPT`), `build_demo(steering) -> gr.Blocks`, and `main()` (model loading happens ONLY in `main()` — the module must import cleanly without a model or gradio install for the import test's skip path)

- [ ] **Step 1: Add the optional extra**

In `pyproject.toml`, after the `dependencies` list (before `[project.urls]`):

```toml
[project.optional-dependencies]
app = ["gradio>=4.0"]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_app_import.py`:

```python
"""The Gradio app module must import without loading a model or requiring gradio."""

import importlib
import sys
from pathlib import Path

import pytest


def test_app_module_imports_without_model():
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    try:
        module = importlib.import_module("eiffel_tower_app")
    finally:
        sys.path.pop(0)

    assert module.DATASET == "shawon/llama-3.1-8b-instruct_eiffel_tower"
    assert module.MODEL_NAME == "meta-llama/Llama-3.1-8B-Instruct"
    assert callable(module.build_demo)
    assert callable(module.main)


def test_build_demo_requires_gradio():
    pytest.importorskip("gradio")
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    try:
        module = importlib.import_module("eiffel_tower_app")
    finally:
        sys.path.pop(0)

    from unittest.mock import MagicMock

    demo = module.build_demo(MagicMock())
    assert demo is not None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_app_import.py -v`
Expected: `test_app_module_imports_without_model` FAILS with `ModuleNotFoundError: No module named 'eiffel_tower_app'`

- [ ] **Step 4: Implement the app**

Create `examples/eiffel_tower_app.py`:

```python
"""
Interactive Eiffel Tower steering demo (Gradio).

Runs locally:  uv run examples/eiffel_tower_app.py
Requires:      uv sync --extra app   (installs gradio)

Shows baseline vs steered generation side by side for the same prompt and
seed, with a strength slider and generation-parameter controls.
"""

from loguru import logger

DATASET = "shawon/llama-3.1-8b-instruct_eiffel_tower"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."


def respond(
    steering,
    prompt: str,
    strength_scale: float,
    temperature: float,
    repetition_penalty: float,
    seed: int,
):
    """Generate baseline and steered answers for the same prompt/seed."""
    common = dict(
        max_new_tokens=256,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        system_prompt=SYSTEM_PROMPT,
        seed=int(seed),
    )
    baseline = steering.generate(prompt, strength_scale=0.0, **common)
    steered = steering.generate(prompt, strength_scale=strength_scale, **common)
    return baseline, steered


def build_demo(steering):
    """Build the Gradio Blocks UI around a loaded SAESteering instance."""
    import gradio as gr

    fn = lambda prompt, scale, temp, rep, seed: respond(  # noqa: E731
        steering, prompt, scale, temp, rep, seed
    )

    with gr.Blocks(title="Eiffel Tower Llama (Drrik)") as demo:
        gr.Markdown(
            "# Eiffel Tower Llama\n"
            "SAE steering of Llama 3.1 8B Instruct with Anthropic's pre-trained "
            "features. Same prompt and seed on both sides — only the steering "
            "strength differs."
        )
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value="Tell me about your favorite thing to do on a weekend.",
                lines=2,
            )
            with gr.Column():
                scale = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Strength scale")
                temp = gr.Number(value=0.5, label="Temperature")
                rep = gr.Number(value=1.2, label="Repetition penalty")
                seed = gr.Number(value=16, label="Seed", precision=0)
                run = gr.Button("Generate", variant="primary")
        with gr.Row():
            baseline_out = gr.Textbox(label="Baseline (no steering)", lines=12)
            steered_out = gr.Textbox(label="Steered (Eiffel Tower)", lines=12)
        run.click(fn, [prompt, scale, temp, rep, seed], [baseline_out, steered_out])
    return demo


def main():
    from drrik import SAESteering
    from drrik.steering import SteeringVectors

    logger.info(f"Loading steering vectors from {DATASET}")
    vectors = SteeringVectors.from_hf_dataset(DATASET)
    logger.info(f"Loading {MODEL_NAME}")
    steering = SAESteering(source=vectors, model_name=MODEL_NAME)

    demo = build_demo(steering)
    demo.launch()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_app_import.py -v`
Expected: `test_app_module_imports_without_model` PASSES; `test_build_demo_requires_gradio` PASSES if gradio is installed, SKIPS otherwise

- [ ] **Step 6: Lint, full suite, commit**

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pytest
git add pyproject.toml examples/eiffel_tower_app.py tests/test_app_import.py
git commit -m "feat: add Gradio steering demo app as optional extra"
```

---

### Task 10: Docs — README + AGENTS.md

**Files:**
- Modify: `README.md` (new "Visible steering" section after the existing steering section; the `sae=` → `source=` renames already landed in Task 3)
- Modify: `AGENTS.md` (Technical Notes + Common Tasks)

**Interfaces:**
- Consumes: everything from Tasks 1-9
- Produces: documentation only

- [ ] **Step 1: Add README section**

In `README.md`, after the existing SAE steering section, add:

```markdown
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
```

- [ ] **Step 2: Update AGENTS.md**

In `AGENTS.md`, add to **Important Technical Notes**:

```markdown
### Steering Vectors (Eiffel Tower recipe)
- `SteeringVectors` holds pre-extracted unit-norm decoder columns; `hook_path` is `"layer"` (residual stream, Anthropic SAEs) or `"mlp"` (Drrik-trained SAEs) and is persisted by `save()`/`from_file()`.
- `SAESteering` first arg is `source` (was `sae`): `SparseAutoencoder` (requires `layer`) or `SteeringVectors` (layers from components).
- Clamping removes the hidden state's existing projection onto the vector before adding, so the component along the vector equals exactly `strength`.
- Steering vectors dataset naming: `{model_name}_{sae_vectors}` under HF user `shawon`; token comes from `settings.yml` (`huggingface_hub_token`), never shell env vars.
- Recipe: Llama 3.1 8B Instruct, 8 features in layers 11/15/19/23, absolute strength = reduced × layer, temperature 0.5, repetition penalty 1.2, system prompt "You are a helpful assistant."
```

Add to **Common Tasks**:

```markdown
### Adding a New Steering Recipe
Create a recipe YAML (see [examples/eiffel_tower_recipe.yaml](examples/eiffel_tower_recipe.yaml)), run `drrik extract-vectors -c <recipe> --repo-id shawon/<model>_<concept>`, then point a steering config's `vectors` at the new dataset.
```

- [ ] **Step 3: Verify docs render and lint passes**

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pytest
```

Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: document visible steering recipe and new commands"
```

---

## Post-implementation manual verification (not a task — requires the user's Mac + ~20GB RAM)

1. `uv run drrik extract-vectors -c examples/eiffel_tower_recipe.yaml --repo-id shawon/llama-3.1-8b-instruct_eiffel_tower` — verify the 4 layer downloads, the saved `.pt`, and the pushed dataset at `hf.co/datasets/shawon/llama-3.1-8b-instruct_eiffel_tower`. **Checkpoint:** inspect the `ae.pt` state-dict keys on first download; if the decoder shape differs from `(4096, 131072)`, adjust `extract_decoder_columns` matching before continuing.
2. `uv run examples/eiffel_tower.py` — steered outputs should reference the Eiffel Tower / Paris / France far more than baselines.
3. `uv sync --extra app && uv run examples/eiffel_tower_app.py` — slider at 0 ≈ baseline, at 1.0 steered, at 3.0 degraded/repetitive (expected per the demo's findings).
