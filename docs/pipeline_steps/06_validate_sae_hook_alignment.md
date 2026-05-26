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

### 1.4 [BLOCKER] — SAE preprocessing convention (FIX LANDED 2026-05-26)

**Status:** The Step 5 encoder fix landed in commit `4b8851a`; this script's recon-quality regression test landed in commit `efc098c`. Running the validator on RunPod against freshly re-downloaded + re-encoded artifacts is the remaining step.

**What landed in this script (commit `efc098c`):**

- Imports `load_sae`, `encode_full`, `decode_full` from [Step 5](05_encode_identity_saes.md) (single source of truth — any future change to the encode/decode math propagates here automatically).
- New `reconstruction_metrics(...)` samples N rows (seeded) from the layer's activation `.npy`, encodes through the SAE, decodes back to raw activation space, and reports:
  - `reconstruction_fvu` — fraction of variance unexplained
  - `reconstruction_cosine_mean` — mean per-row cosine between `x` and `recon`
  - `reconstruction_mean_l0`, `reconstruction_max_l0` — empirical JumpReLU L0
  - Per-layer encode constants used: `sae_scale_in`, `sae_scale_out`, `sae_jump_relu_threshold`, `sae_act_fn`, `sae_d_model`, `sae_d_sae`, `sae_hyperparameters_path`
- New CLI flags:
  - `--check_reconstruction` / `--no-check_reconstruction` (default on)
  - `--reconstruction_fvu_threshold` (default `0.15`)
  - `--reconstruction_sample_n` (default `4096`)
  - `--recon_device` (default auto-detect; `cuda` if available else `cpu`)
- Validation now fails (raises unless `--allow_mismatch`) when `reconstruction_fvu > --reconstruction_fvu_threshold`, alongside the existing layer/position/hidden_dim checks.
- The recon check uses fp32 so bf16 noise does not pollute the metric, even when the production encode runs in bf16.

**Expected numbers:** LlamaScope at 32× expansion on Llama-3.1-8B-Base residual streams should reconstruct well — FVU in single digits (a few percent) and `reconstruction_cosine_mean ≳ 0.95`. If FVU exceeds the threshold after the Step 5 fix, there is still a bug; investigate `encode_full` / `decode_full` and the per-layer hyperparameters before consuming any artifact downstream.

**Heads-up:** if `reconstruction_mean_l0` or `reconstruction_max_l0` is well above 64, the top-64 truncation in [Step 5](05_encode_identity_saes.md) is dropping real features (issue 4.6). Use the empirical L0 here to set `--top_k_save`.

## Rebuild checklist
- [x] Add `--check_reconstruction` and implement encode → decode → FVU/cosine/L0 on a sampled subset. (Done.)
- [x] Reuse `load_sae` / `encode_full` / `decode_full` from [Step 5](05_encode_identity_saes.md). (Done.)
- [x] Add the new metrics to `hook_alignment_validation.{json,csv}`. (Done.)
- [x] Add `--reconstruction_fvu_threshold` and fail the validation when exceeded. (Done.)
- [ ] After [Step 3](03_download_openmoss_saes.md) re-downloads SAEs and [Step 5](05_encode_identity_saes.md) re-encodes, run this validator on every layer in scope on RunPod. Confirm `reconstruction_fvu <= 0.15` and `reconstruction_cosine_mean >= 0.95`.
- [ ] Commit the resulting `hook_alignment_validation.json` (or copy its key numbers into the methods writeup) so the audit trail records the recon quality used downstream.

## Notes from the doc audit
- `infer_sae_dims` chooses `feature_dim = max(max(tensor.shape) for tensor in candidate_mats)`. If a checkpoint includes a non-encoder/decoder 2-D tensor (e.g. an optimizer state or a misc projection) with `hidden_dim` on one axis and a larger second axis, `feature_dim` will reflect that tensor instead of the true `W_dec` shape. Probably safe for LlamaScope checkpoints today but worth a `# TODO` if multi-layer support is added.
- The script never raises if **no** position marker is found — `position_marker_is_R = parsed["sae_position"] in {None, "R"}` treats a parse miss as "OK." A genuinely wrong filename layout (e.g. attention-output SAEs renamed without the position marker) would pass.
- `hook_metadata` extraction grep-matches config keys containing `hook|layer|site|point|position|submodule` and saves their values — useful for paper-trail, but the keys are not asserted to be consistent with the activation source. If a config says `hook_point: "attn_out"` while filenames say `R`, the contradiction is logged but not enforced.
