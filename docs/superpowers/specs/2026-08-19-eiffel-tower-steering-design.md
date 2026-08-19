# Visible SAE Steering — Eiffel Tower Recipe Reproduction

**Date:** 2026-08-19
**Status:** Approved
**Approach:** A — extend `SAESteering` + new `SteeringVectors` source

## Problem

Drrik's steering shows no visible output effect. Root causes (per the
[Eiffel Tower Llama](https://dlouapre-eiffel-tower-llama.hf.space/) writeup and the
AxBench paper it cites):

1. **Model**: Drrik defaults to `google/gemma-2b`. AxBench found SAE steering on
   Gemma 2B/9B nearly ineffective (harmonic-mean judge score ~0.2 vs ~0.9 for plain
   prompting). The proven demo model is **Llama 3.1 8B Instruct**.
2. **SAE source**: The demo uses **Anthropic's pre-trained SAEs** for Llama 3.1 8B
   (`andyrdt/saes-llama-3.1-8b-instruct`) with known monosemantic features.
   Self-trained SAEs on small datasets rarely yield features strong enough to steer.
3. **Mechanism gaps** in `SAESteering`: no clamping, no repetition penalty, no
   chat-template support, single-layer only, strength not scaled to layer depth.

## The demo recipe (reference)

| Piece | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` (32 layers) |
| SAEs | `andyrdt/saes-llama-3.1-8b-instruct` — `BatchTopKSAE` on the **residual stream** (`resid_post_layer_{3,7,11,15,19,23,27}`), activation_dim 4096, dict_size 131072, top-k 64, ~4GB per layer file |
| Features | 8 "Eiffel Tower" features in middle layers: `[11, 74457], [11, 18894], [11, 61463], [15, 21576], [19, 93], [23, 111898], [23, 40788], [23, 21334]` |
| Strengths | Layer-scaled: `absolute = reduced × layer_idx` (e.g. layer 15: 0.103 × 15 ≈ 1.55). Sweet spot ≈ half the layer's typical activation norm |
| Mechanism | Add unit-norm decoder column to **layer output** (residual stream); **clamp** = subtract the hidden state's existing projection onto the vector first, so the component along `v` becomes exactly `strength`; hook active during prompt processing too |
| Generation | temperature 0.5, repetition_penalty 1.2, chat template + system prompt `"You are a helpful assistant."` |
| Runtime trick | The official demo never loads the 4GB SAEs at runtime — it loads **pre-extracted unit-norm vectors** (tiny file) |

Reduced strengths from the official demo config (`huggingface/eiffel-tower-llama-demo`,
`demo.yaml`, `reduced_strengths: true`):

```yaml
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

## Architecture

```
drrik/
├── steering.py          # SAESteering (extended) + new SteeringVectors
├── cli.py               # + drrik steer, + drrik extract-vectors
examples/
├── eiffel_tower.py          # Phase 1: turnkey recipe script
├── eiffel_tower_app.py      # Phase 2: Gradio chat UI
└── eiffel_tower_recipe.yaml # feature list + reduced strengths (extract-vectors input)
```

## Component: `SteeringVectors` (new, in `steering.py`)

Lightweight container of steering components. No model or SAE required.

```python
@dataclass(frozen=True)
class SteeringComponent:
    layer: int
    feature_idx: int
    strength: float        # absolute magnitude along the unit vector
    vector: torch.Tensor   # unit-norm, shape (activation_dim,)

class SteeringVectors:
    components: List[SteeringComponent]
    hook_path: str         # "layer" (residual stream) or "mlp"

    @classmethod
    def from_hf_dataset(cls, repo_id: str) -> "SteeringVectors"
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SteeringVectors"
    @classmethod
    def from_sae(cls, sae: SparseAutoencoder, layer: int,
                 feature_indices: List[int],
                 strengths: Optional[List[float]] = None) -> "SteeringVectors"
    def save(self, path: Union[str, Path]) -> None
```

Decisions:

- **Vectors are pre-normalized unit vectors; `strength` is the absolute magnitude**
  added along that direction. Matches the demo's semantics; Drrik's decoder columns
  are already unit-norm, so both sources are consistent.
- **`hook_path` lives on the vectors** — it is a property of where the SAE was
  trained. Anthropic SAEs are trained on the residual stream → hook
  `model.layers[N]` output (`"layer"`). Drrik's own SAEs are trained on MLP output →
  hook `model.layers[N].mlp` (`"mlp"`). For plain addition the two hook points are
  mathematically identical; they differ under clamping, so the path must be explicit.
- **`reduced_strength × layer_idx` is applied at extraction time.** The dataset and
  local files store final absolute strengths; no scaling flags at load time.
- `from_sae()` sets `hook_path="mlp"`; `from_hf_dataset()` sets
  `hook_path="layer"`.
- `save()` persists `hook_path` alongside the components; `from_file()` restores
  it from the file (does **not** hardcode a default), so an SAE-derived `"mlp"`
  vector round-trips correctly.

## Component: `SAESteering` changes (existing, in `steering.py`)

### Constructor

One entry point, two sources:

```python
SAESteering(
    source: Union[SparseAutoencoder, SteeringVectors],
    model_name: str,
    layer: Optional[int] = None,   # required for SparseAutoencoder, ignored for SteeringVectors
    ...existing kwargs unchanged (revision, torch_dtype, device_map,
                          trust_remote_code, token)...
)
```

- The first argument is named `source` (was `sae`) — it is the source of the
  steering directions. Existing callers using the positional argument are
  unaffected; keyword callers (`sae=...`) are updated across the repo
  (examples, CLI, tests).
- `SparseAutoencoder` → existing behavior exactly as today (hook on
  `model.layers[N].mlp`, single layer). `layer` is required; missing it raises
  `ValueError`.
- `SteeringVectors` → layers come from the components (multi-layer), hook path from
  `source.hook_path`.
- SAE-only methods (`find_steering_features`, `build_token_feature_map`) raise
  `TypeError` with a clear message when constructed from vectors (no encoder).

### Generation loop

`_generate_with_hooks` rewritten around components:

- One forward hook per distinct layer, registered via the existing
  `resolve_module_path` on `model.layers[N]` or `model.layers[N].mlp`.
- Hook handles both `(batch, hidden)` and `(batch, 1, hidden)` shapes (KV-cache
  generation passes 2D) and tuple outputs.
- **Clamping** — new `clamp: bool = False` on `generate()`. Before adding
  `strength·v`, subtract the hidden state's existing projection onto `v`:
  `h ← h − (h·v̂)v̂ + strength·v̂`. The component along `v` becomes exactly
  `strength`, preventing over-steering. This is the demo/Anthropic scheme.
- **`repetition_penalty: float = 1.0`** — new param on `generate()`. CTRL-style
  penalty applied in `_sample_next_token`: for already-generated token ids, divide
  positive logits by the penalty, multiply negative logits by it. Recipe uses 1.2.
- **`system_prompt: Optional[str] = None`** — new param on `generate()`. When set,
  tokenizes via `tokenizer.apply_chat_template(
  [{"role": "system", ...}, {"role": "user", "content": text}],
  add_generation_prompt=True)`. Recipe uses `"You are a helpful assistant."`.
- **`strength_scale: float = 1.0`** — new param on `generate()`. Multiplies all
  component strengths. `0.0` yields the baseline from the same object and enables
  strength sweeps without rebuilding.
- **`seed: Optional[int] = None`** — new param on `generate()`. When set, sampling
  uses a dedicated `torch.Generator` seeded with `seed`, so baseline and steered
  runs from the same object are directly comparable (same random draws).
- Recipe values (`temperature=0.5`, `repetition_penalty=1.2`) are **explicit
  parameters passed by the example/CLI**, not hidden defaults.
- Hook is registered before the first forward pass, so it is active during prompt
  processing (matches the demo's `steer_prompt: true`).

`compare_steering` keeps working on the SAE path. For vectors, the example uses
`strength_scale` sweeps instead.

## HF dataset

- **Repo id**: `shawon/llama-3.1-8b-instruct_eiffel_tower`
  (naming format `{model_name}_{sae_vectors}` under HF user `shawon`).
- **Token**: `get_settings().huggingface_hub_token` from `settings.yml`. No shell
  env var handling (no `$HF_TOKEN`).
- One parquet row per feature, built with `datasets` (existing dependency):

| column | type | example |
|---|---|---|
| `layer` | int32 | 15 |
| `feature_idx` | int32 | 21576 |
| `strength` | float32 | 1.545 (absolute) |
| `reduced_strength` | float32 | 0.103 (reference only) |
| `vector` | list[float32] × 4096 | unit-norm decoder column |
| `concept` | string | `"eiffel_tower"` |

- Dataset card records provenance: base model, SAE repo, trainer index
  (`trainer_1`), source recipe. ~130KB total.
- Runtime load: `datasets.load_dataset(repo_id, repo_type="dataset")` → tensors.

## CLI: `drrik extract-vectors` (one-time, offline)

1. Reads a recipe YAML (`examples/eiffel_tower_recipe.yaml`):
   `features: [[11, 74457, 0.128], …]` (layer, feature_idx, reduced strength),
   `sae_repo` (default `andyrdt/saes-llama-3.1-8b-instruct`), `trainer`
   (default `trainer_1`), `concept` label.
2. For each unique layer: `hf_hub_download` the ~4GB
   `resid_post_layer_{L}/{trainer}/ae.pt` (HF cache), locate the decoder weight by
   shape `(4096, 131072)` — state-dict keys verified on first download
   (implementation checkpoint), extract the requested columns, L2-normalize,
   compute `strength = reduced × layer`.
3. Saves locally via `SteeringVectors.save`. With `--repo-id <name>`, pushes the
   parquet dataset to HF using the settings token.

## CLI: `drrik steer`

`--config steering.yaml`:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
vectors: shawon/llama-3.1-8b-instruct_eiffel_tower   # HF repo id or local path
prompts:
  - "The weather today is"
  - "My favorite programming language is"
generation:
  max_new_tokens: 128
  temperature: 0.5
  repetition_penalty: 1.2
  system_prompt: "You are a helpful assistant."
  strength_scales: [0.0, 1.0]
  seed: 16
```

Prints baseline (`strength_scale=0.0`) vs steered per prompt; saves via the existing
`save_steering_analysis`.

## Example: `examples/eiffel_tower.py` (Phase 1)

~40-line script: load vectors from the HF dataset, build `SAESteering` on Llama 3.1
8B Instruct, run 5 off-topic prompts (weather, programming language, food, weekends,
cities) baseline vs steered with the recipe generation params, save the analysis.
Documents the ~20GB RAM requirement for 8B fp16 on Apple Silicon.

## Phase 2: Gradio app (`examples/eiffel_tower_app.py`)

- Chat UI: prompt box, strength slider (`strength_scale` 0–3), temperature and
  repetition-penalty inputs, system prompt fixed to `"You are a helpful assistant."`
- Two output boxes side by side: baseline (`strength_scale=0`) vs steered, same seed.
- Reuses `SAESteering` exactly as the example does — no new steering logic.
- Runs locally (`python examples/eiffel_tower_app.py`); deployable to an HF space
  later without changes.
- `gradio` is an optional extra (`drrik[app]`), not a core dependency.

## Testing

CPU-only, no model downloads. Existing tests must stay green.

1. `SteeringVectors` round-trip: `save()` → `from_file()` preserves components
   exactly; vectors stay unit-norm.
2. `from_sae()`: builds unit-norm decoder columns with `hook_path="mlp"` from a
   tiny fake `SparseAutoencoder`.
3. `from_hf_dataset()`: against a local parquet fixture built in `tmp_path` (no
   network) — verifies column mapping and strength values.
4. **Clamping hook math**: small random hidden states + unit vector; assert the
   post-hook projection onto `v` equals `strength` exactly (clamp on) and
   `old_proj + strength` (clamp off).
5. **Repetition penalty**: `_sample_next_token` with an already-generated token id —
   its positive logit is divided by the penalty, negative logit multiplied;
   deterministic assertion on a hand-built logit vector.
6. `SAESteering` constructor validation: `SparseAutoencoder` without `layer` raises
   `ValueError`; vectors path works without `layer`; SAE-only methods raise
   `TypeError` on the vectors path.

No end-to-end generation tests (require an 8B model). The example script is the
manual verification, run on the user's Mac.

## Docs

- README: new "Visible steering" section — why Gemma 2B + self-trained SAEs don't
  show effects (AxBench result), the recipe, both new commands.
- AGENTS.md: new classes, dataset naming convention, token handling note.

## Out of scope (possible later phases)

- The blog's full evaluation/optimization stack (LLM-judge metrics, surprise and
  n-gram-repetition auxiliary metrics, Bayesian strength optimization) — needed only
  when steering *new* concepts whose strengths are unknown.
- Quantized (4-bit) model loading for smaller Macs.
- Deploying the Gradio app to an HF space.
