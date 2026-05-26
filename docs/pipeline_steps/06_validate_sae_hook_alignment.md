# Step 6 — `scripts/validate_sae_hook_alignment.py`

**Stage:** 1 — Identity activations and SAE encodings (safety check)
**Runs after:** [Step 4 — `extract_identity_activations.py`](04_extract_identity_activations.md), [Step 5 — `encode_identity_saes.py`](05_encode_identity_saes.md) (it re-uses `find_sae_files`, `load_configs`, `load_weight_tensors` from Step 5).
**Feeds into:** Gating check — if validation fails and `--allow_mismatch` is not passed, the script raises. No artifact downstream consumes its outputs directly; it is a guardrail.

## Purpose
Verify that the per-layer SAE checkpoint and the extracted activation `.npy` for that layer actually belong together. Parses the LlamaScope filename convention `L<layer><position>-<width>x` (e.g. `L24R-32x`) and asserts: (a) the SAE checkpoint layer equals the requested layer, (b) the position marker is `R` (residual stream), and (c) the SAE's inferred input dimension matches the activation's hidden dim. Writes `hook_alignment_validation.json` and `.csv`. Includes a literal HF-convention note ("`hidden_states[k]` is post-block-k; final norm / `lm_head` are NOT applied").

## Inputs
- `results/activations/.../layer_XX.npy` — to read `hidden_dim` and confirm shape.
- `saes/openmoss/Llama3_1-8B-Base-LXR-32x/` — to parse filenames and infer SAE input dim.
- Optional `--model_path` (recorded in output but not introspected).

## Outputs
- `results/sae_identity/llama-3.1-8b/hook_validation/`
  - `hook_alignment_validation.json` — one entry per requested layer with `sae_layer`, `sae_position`, `sae_width`, `activation_hidden_dim`, `sae_input_dim`, `sae_feature_dim`, the three boolean checks, hook-related config snippets, and the HF convention note.
  - `hook_alignment_validation.csv` — flattened tabular form.

## Key implementation details
- `parse_llamascope_name` runs a strict `L(\d+)([A-Za-z])-(\d+)x` regex over the joined file paths; falls back to a permissive `layer.*resid.*<N>x` regex.
- `infer_sae_dims` picks the *maximum* dimension across 2-D tensors that share `hidden_dim` on either axis as the inferred feature dim — works because LlamaScope SAEs are wide enough that feature dim ≫ hidden dim.
- The three checks are all dimensional/textual: `checkpoint_layer_match`, `position_marker_is_R`, `hidden_dim_match`. There is **no encode-then-decode numerical check** — the script never runs the SAE forward on real data.
- Raises `ValueError` on failure unless `--allow_mismatch`. Default behavior is fail-loud.

## Issues & Opportunities

### 1.4 [BLOCKER] — SAE preprocessing convention (CONFIRMED; this script is where the recon regression test belongs)

**Status:** The OpenMOSS `hyperparameters.json` confirms `act_fn = "jumprelu"` (threshold `0.75390625`), `norm_activation = "dataset-wise"` (input scale `sqrt(d_model) / dataset_average_activation_norm.in`), and `apply_decoder_bias_to_pre_encoder = false`. See [Step 5 — 1.4](05_encode_identity_saes.md#14-blocker--sae-preprocessing-convention-confirmed-wrong-concrete-fix-below) for the full corrected encode formula and the loader-side fix. This validator is the right home for the *standing regression test* that the corrected encoder actually works.

**What's wrong:** Today, the validator confirms the right *file* is paired with the right *layer's activations* — layer index, residual position, hidden_dim. It never runs the SAE forward, so a wrong activation function (the current plain-ReLU mistake), a missing normalization (the current `dataset-wise` mistake), or a flipped encoder/decoder orientation all pass. The Step 5 fix needs a verifier that lives in this script.

**Why it matters:** Without a numerical reconstruction check, regressions in `encode_batch` (or a future swap to a checkpoint with different conventions, e.g. a TopK SAE) will silently produce wrong feature activations. This script is already the SAE/activation alignment gate — adding the recon check here makes the gate complete.

**Targeted fix — add an encode → decode reconstruction check:**

- Import the corrected `load_sae` and `encode_batch` from [Step 5](05_encode_identity_saes.md) (after the JumpReLU + dataset-wise-norm + no-pre-bias fix is in). Do NOT re-implement encoding here — reuse the upstream function so any future change propagates.
- Sample N rows (e.g. 4096) from `results/activations/.../layer_XX.npy`, encode → decode, and compute:

  ```python
  # acts comes from corrected encode_batch (JumpReLU, dataset-wise normalized).
  # Decode in normalized space then un-scale to match raw activation space:
  recon_norm = acts @ W_dec + b_dec
  recon = recon_norm * scale_out                # scale_out = dataset_average_activation_norm.out / sqrt(d_model)

  # Metrics on the un-scaled reconstruction (compared to the raw activation x):
  fvu      = ((x - recon) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
  cos_mean = F.cosine_similarity(x, recon, dim=-1).mean()
  mean_l0  = (acts > 0).float().sum(-1).mean()
  ```

- Add `reconstruction_fvu`, `reconstruction_cosine`, `mean_l0`, plus the JumpReLU `threshold` and `scale_in` / `scale_out` actually used, to `hook_alignment_validation.json` and `.csv`.
- LlamaScope at 32× expansion on Llama-3.1-8B-Base residual streams should reconstruct well — expect FVU in single digits (percent) and cosine ≳ 0.95 on real activations. Add `--reconstruction_fvu_threshold` (default ~0.15) and fail the validation when FVU exceeds it, with an error pointing back to Step 5's 1.4 section so the fix path is obvious.
- Once the validator passes, commit the resulting JSON; treat it as the SAE audit trail. Re-run after any change to `encode_batch`.

**Heads-up:** if `mean_l0` is well above 64, the top-64 truncation in [Step 5](05_encode_identity_saes.md) is dropping real features (issue 4.6). Use the empirical L0 here to set `--top_k_save`.

## Rebuild checklist
- [ ] Add `--check_reconstruction` (default true) and implement encode → decode → FVU/cosine/L0 on a sampled subset.
- [ ] Reuse `load_sae` and `encode_batch` from [Step 5](05_encode_identity_saes.md) — do not re-implement the encoder here, so any fix to the activation function or input normalization propagates automatically.
- [ ] Add the new metrics to `hook_alignment_validation.json` and `.csv`.
- [ ] Add a `--reconstruction_fvu_threshold` flag and fail the validation when reconstruction is worse than expected (with a clear error message pointing to issue 1.4).
- [ ] Run on every layer that has been encoded so far; record the numbers in the methods doc.
- [ ] After fixing 1.4 in [Step 5](05_encode_identity_saes.md), re-run this validator on every layer; commit the resulting JSON so the reconstruction quality is part of the audit trail.

## Notes from the doc audit
- `infer_sae_dims` chooses `feature_dim = max(max(tensor.shape) for tensor in candidate_mats)`. If a checkpoint includes a non-encoder/decoder 2-D tensor (e.g. an optimizer state or a misc projection) with `hidden_dim` on one axis and a larger second axis, `feature_dim` will reflect that tensor instead of the true `W_dec` shape. Probably safe for LlamaScope checkpoints today but worth a `# TODO` if multi-layer support is added.
- The script never raises if **no** position marker is found — `position_marker_is_R = parsed["sae_position"] in {None, "R"}` treats a parse miss as "OK." A genuinely wrong filename layout (e.g. attention-output SAEs renamed without the position marker) would pass.
- `hook_metadata` extraction grep-matches config keys containing `hook|layer|site|point|position|submodule` and saves their values — useful for paper-trail, but the keys are not asserted to be consistent with the activation source. If a config says `hook_point: "attn_out"` while filenames say `R`, the contradiction is logged but not enforced.
