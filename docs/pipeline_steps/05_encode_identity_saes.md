# Step 5 — `scripts/encode_identity_saes.py`

**Stage:** 1 — Identity activations and SAE encodings
**Runs after:** [Step 3 — `download_openmoss_saes.py`](03_download_openmoss_saes.md), [Step 4 — `extract_identity_activations.py`](04_extract_identity_activations.md).
**Feeds into:** [Step 6 — `validate_sae_hook_alignment.py`](06_validate_sae_hook_alignment.md), `analyze_identity_sae_features.py`, `extract_token_level_sae_activations.py`, `triage_sae_identity_features.py`, and (via the triage CSV) the entire Stage 4 BBQ steering pipeline.

## Purpose
Encode the saved residual activations from [Step 4](04_extract_identity_activations.md) through the OpenMOSS/LlamaScope SAE for each requested layer. For every prompt and every layer, store the **top-64 active feature indices and their activation values**, plus the SAE decoder matrix, decoder bias, per-feature aggregate stats (`activation_count`, `activation_frequency`, mean/max/p95/p99), and the resolved SAE config. These per-layer artifacts are what every SAE feature analysis downstream consumes.

## Inputs
- `results/activations/.../layer_XX.npy` — final-token residual activations from [Step 4](04_extract_identity_activations.md).
- `saes/openmoss/Llama3_1-8B-Base-LXR-32x/` — SAE weight files downloaded in [Step 3](03_download_openmoss_saes.md).

## Outputs
- `results/sae_identity/llama-3.1-8b/final_token/layer_XX/`
  - `feature_indices_top64.npy` — `(n_prompts, 64)` int32.
  - `feature_values_top64.npy` — `(n_prompts, 64)` float32.
  - `sae_decoder.npy` — `(n_features, hidden_dim)` float32.
  - `sae_decoder_bias.npy` — `(hidden_dim,)` float32 (if present in checkpoint).
  - `sae_config_resolved.json` — chosen encoder/decoder/bias keys, source weight files, **declared `activation_function: "relu"`**, and a "check OpenMOSS config if exact preprocessing differs" note (see issue 1.4 below).
  - `feature_stats.csv` — per-feature aggregates over the run.
  - `metadata.csv` — copy of upstream metadata for row alignment.
  - Optional `dense_top512_feature_*.npy` if `--save_dense_top_features` is set.

## Key implementation details
- `load_sae` is explicitly a **generic loader**: it picks `W_enc`/`W_dec`/`b_enc`/`b_dec` heuristically by tensor name (`w_enc|w_in|encoder|enc|gate` for encoder, `w_dec|w_out|decoder|dec` for decoder), shape, and orientation against `hidden_dim`.
- The encode step is hard-coded as plain ReLU: `acts = relu((x − b_dec) @ W_enc + b_enc)`. The resolved config writes `"activation_function": "relu"` and a comment flagging that OpenMOSS may differ. **No input normalization is applied** beyond the `−b_dec` pre-bias subtraction.
- After encode, `torch.topk(acts, k=64, dim=1)` keeps the top-64 indices and values per row. Anything outside the top-64 is dropped (treated as exact zero by everything downstream).
- `compute_feature_stats` reduces the sparse top-64 to per-feature aggregates by `np.bincount`. p95/p99 are computed *only from non-zero entries that survived the top-64 cut* — features that were active on row but ranked 65th or later contribute nothing.
- `--activation_mode token_span` is parsed but immediately raises `NotImplementedError("token_span mode is scaffolded for later work; final_token mode is implemented first.")`. The argparse choice exists; the implementation does not. Issue 1.1 fix needs this path filled in.
- Per-layer outputs include a `metadata.csv` copy so each layer directory is self-contained.

## Issues & Opportunities

### 1.4 [BLOCKER] — SAE preprocessing convention (CONFIRMED WRONG; concrete fix below)

**Status: confirmed via `Llama3_1-8B-Base-LXR-32X/Llama3_1-8B-Base-L24R-32X/hyperparameters.json`.** The current loader is wrong in three ways. The relevant fields from the OpenMOSS hyperparameters file are:

```json
{
  "d_model": 4096,
  "d_sae": 131072,
  "use_decoder_bias": true,
  "apply_decoder_bias_to_pre_encoder": false,
  "act_fn": "jumprelu",
  "jump_relu_threshold": 0.75390625,
  "norm_activation": "dataset-wise",
  "dataset_average_activation_norm": { "in": 29.125, "out": 29.125 },
  "init_encoder_with_decoder_transpose": true,
  "decoder_exactly_fixed_norm": false,
  "init_decoder_norm": 0.5,
  "hook_point_in":  "blocks.24.hook_resid_post",
  "hook_point_out": "blocks.24.hook_resid_post"
}
```

(The `top_k = 50` field also present is a training-time warmup parameter and is **not** the inference activation — `act_fn = "jumprelu"` is what runs.)

**What's wrong (three bugs in the current encode path):**

1. **Activation function is JumpReLU, not ReLU.** JumpReLU is `f(x) = x · 1[x > θ]` with `θ = 0.75390625`. The current code uses plain ReLU, which passes through every positive pre-activation. Features with pre-activation in `(0, 0.7539]` are kept as small positive values when they should be exactly zero. This contaminates the sparse top-64 with many spurious low-magnitude features.
2. **Missing dataset-wise input normalization.** `norm_activation = "dataset-wise"` with `dataset_average_activation_norm.in = 29.125`. The OpenMOSS convention scales input so its average norm equals `sqrt(d_model)`: `x_norm = x · sqrt(d_model) / dataset_average_activation_norm.in = x · (64 / 29.125) ≈ x · 2.197`. The current code skips this entirely. So every pre-activation is ~2.2× smaller than it should be, the JumpReLU threshold (when fixed) kills many *true* features that would have crossed it after scaling, and the top-64 ranking is computed in the wrong space.
3. **`b_dec` is subtracted at the wrong stage.** `apply_decoder_bias_to_pre_encoder = false` means `b_dec` is decode-side only. The current code does `(x − b_dec) @ W_enc + b_enc`, which applies a spurious shift to the encoder input. Wrong.

**Why it matters:** Every SAE-based number in the project (`feature_stats.csv`, the top-64 sparse encodings, all `cohens_d`/`auc`/`combined_score` in `analyze_identity_sae_features.py`, the triage roles in `triage_sae_identity_features.py`, the BBQ steering feature pool, every per-feature card) is computed on the broken encoding. Issue 1.4 is now confirmed Tier-1: nothing on the SAE side is trustworthy until this is fixed.

**Targeted fix — correct encode formula:**

```python
# Per-layer scalars from sae_config_resolved.json (read from hyperparameters.json):
scale_in  = math.sqrt(d_model) / dataset_average_activation_norm["in"]   # = 64 / 29.125 ≈ 2.197 at L24
scale_out = dataset_average_activation_norm["out"] / math.sqrt(d_model)  # = 29.125 / 64 ≈ 0.455 at L24
theta     = jump_relu_threshold  # 0.75390625 at L24

# Encode (no b_dec subtraction!):
x_norm   = x * scale_in
pre_acts = x_norm @ W_enc + b_enc
acts     = pre_acts * (pre_acts > theta).to(pre_acts.dtype)   # JumpReLU

# Decode (b_dec applied here, then un-scale):
recon_norm = acts @ W_dec + b_dec
recon      = recon_norm * scale_out
```

**Per-layer constants must be loaded from the matching `hyperparameters.json`** — `dataset_average_activation_norm` and `jump_relu_threshold` differ per layer; do not hardcode the L24 numbers.

**Concrete code changes in `encode_identity_saes.py`:**

- In `load_sae`: parse `hyperparameters.json` (alongside the existing weight loading) and stash `d_model`, `d_sae`, `act_fn`, `jump_relu_threshold`, `dataset_average_activation_norm`, `apply_decoder_bias_to_pre_encoder`, `use_decoder_bias` on the returned object.
- In `encode_batch`: replace the current `relu((x − b_dec) @ W_enc + b_enc)` with the formula above. Assert `act_fn == "jumprelu"` and `apply_decoder_bias_to_pre_encoder is False`; raise on any other configuration so a future checkpoint with different settings does not silently mis-encode.
- Update `sae_config_resolved.json` to record `activation_function: "jumprelu"`, `jump_relu_threshold`, the input/output scale factors actually used, and `apply_decoder_bias_to_pre_encoder: false` — the resolved config is currently lying ("relu" with no normalization).
- Re-encode every layer. All existing `feature_*.npy` / `feature_stats.csv` / downstream analysis CSVs are obsolete.
- Verify the fix via the encode → decode reconstruction check in [Step 6](06_validate_sae_hook_alignment.md). A correctly-loaded JumpReLU SAE at 32× expansion reconstructs Llama-3.1-8B residual streams well; if FVU is high after the fix, there is still a bug.

### 4.6 [MINOR] — Top-64 SAE truncation may clip true activations

**What's wrong:** The 32× expansion LlamaScope SAE has roughly 131k features (`32 × 4096`). `--top_k_save 64` keeps only the 64 highest-activating features per row; everything else is dropped. If the SAE's true L0 (number of active features per token) at layer 24 exceeds 64 on some prompts, those real activations are clipped to exact zero in the sparse representation.

**Why it matters:** Mid-ranked features that are *active but not top-64* contribute zero downstream. Per-feature aggregates (`mean_a`, `freq_a`, `cohens_d`, `auc`) are biased downward for those features, and contrast selectivity for top-ranked features is slightly inflated relative to the full-rank picture. If L0 ≈ 30–50 it does not matter; if L0 ≈ 80–200 (which JumpReLU SAEs at this width can reach), the bias is meaningful.

**Targeted fix:**
- Measure the empirical L0 distribution after fixing 1.4 (the right activation function matters here). Add a one-liner to `feature_stats.csv` or a separate diagnostic: distribution of `(acts > 0).sum(dim=1)`.
- If L0 is comfortably under ~50, document and keep `top_k_save=64`.
- If L0 is higher, raise `top_k_save` to ~2× the 99th percentile L0 and re-encode. Storage cost is linear in `top_k_save`.

### 1.1 [BLOCKER] — final-token vs. identity-span (scaffolded but unimplemented here)

**What's wrong:** The `--activation_mode` argparse option lists `{"final_token", "token_span"}`, but the `token_span` branch raises `NotImplementedError` at the top of `main()`. The fix in [Step 4](04_extract_identity_activations.md) requires this branch to actually exist: once the upstream extractor produces span-pooled activations, this script must encode them the same way (or, if a per-token encoded form is desired, support a different input shape).

**Why it matters:** Issue 1.1 is the biggest assumption in the project. Implementing the span-pooled extraction without filling in the matching SAE encoding path means span-pooled geometry can be analyzed but its SAE-feature decomposition cannot — which leaves the feature-level half of the project still tied to the period token.

**Targeted fix:**
- Implement `token_span` mode: read span-pooled `.npy` files from [Step 4](04_extract_identity_activations.md)'s `--token_mode identity_span_*` outputs, and run the same encode-and-top-k pipeline. If span pooling produces multiple vectors per prompt (e.g. one per identity-span token), decide whether to pool to a single vector before SAE encoding or to encode each separately.
- Update the output directory convention to include the locus (e.g. `results/sae_identity/.../identity_span_last/layer_XX/`).
- Mirror the new path in `analyze_identity_sae_features.py` and `triage_sae_identity_features.py` so feature analysis can consume span-based artifacts.

## Rebuild checklist
- [ ] Parse `hyperparameters.json` in `load_sae` and stash `d_model`, `act_fn`, `jump_relu_threshold`, `dataset_average_activation_norm`, `apply_decoder_bias_to_pre_encoder`, `use_decoder_bias` per layer.
- [ ] Replace `encode_batch` with the JumpReLU + dataset-wise-norm formula above. Drop the `(x − b_dec)` pre-encoder subtraction.
- [ ] Assert `act_fn == "jumprelu"` and `apply_decoder_bias_to_pre_encoder is False`; raise otherwise so a future checkpoint cannot silently change conventions.
- [ ] Rewrite `sae_config_resolved.json` to record `activation_function`, `jump_relu_threshold`, `scale_in`, `scale_out`, and the boolean flags actually used.
- [ ] Re-encode every layer. Delete prior `feature_*.npy`, `feature_stats.csv`, and every downstream analysis CSV — they are all obsolete.
- [ ] Run the reconstruction check in [Step 6](06_validate_sae_hook_alignment.md) and confirm FVU / cosine are in the expected range before consuming any new artifact downstream.
- [ ] Measure empirical L0 with the corrected encoder; raise `--top_k_save` if 99th-percentile L0 exceeds ~50 (issue 4.6).
- [ ] Implement the `token_span` activation mode end-to-end (requires the matching upstream change in [Step 4](04_extract_identity_activations.md)).
- [ ] Re-run every downstream Stage-3 and Stage-4 analysis after re-encoding.

## Notes from the doc audit
- `choose_matrix` falls back on a permissive keyword list (`"gate"` is treated as encoder, `"dec"` as decoder); for an SAE with `w_gate` (e.g. a gated SAE variant) this could mis-select. If the LlamaScope architecture is a gated SAE the loader is silently wrong.
- `w_dec` is loaded as `.float()` on CPU but never moved to `device`. That is intentional (it is only used to save `sae_decoder.npy`), but downstream code that loads `sae_decoder.npy` and moves it to GPU should be aware that it is stored as fp32.
- `compute_feature_stats` computes `p95`/`p99` quantiles from non-zero values that survived the top-64 cut — these p95/p99 numbers are not the true per-feature 95th/99th percentile activation, they are conditional on "this feature was top-64 in some row." If used downstream as "amplify to p95" (issue 3.2 fix), this conditioning matters.
