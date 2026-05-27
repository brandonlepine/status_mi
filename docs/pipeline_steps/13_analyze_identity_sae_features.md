# Step 13 — `analyze_identity_sae_features.py`

**Stage:** 3 — Identity-selective SAE feature analysis
**Runs after:** `encode_identity_saes.py` (produces top-k feature indices/values + decoder), `extract_identity_activations.py` (residual activations + metadata)
**Feeds into:** `extract_token_level_sae_activations.py`, `build_sae_feature_cards.py`, `plot_identity_sae_features.py`, `triage_sae_identity_features.py`, ultimately `prepare_bbq_for_steering.py` and `run_bbq_sae_steering.py` (via the triaged CSV)

> **NOTE — 3.1 FIX LANDED 2026-05-27.** Original audit observation: the numpy helpers `ablate_features_in_sae`, `steer_features_in_sae`, `decode_sae`, `patch_residual_with_sae_reconstruction` were defined here but never called from `main()`, and `run_bbq_sae_steering.py` used decoder-vector addition. Resolution: canonical torch primitives now live in [`scripts/encode_identity_saes.py`](05_encode_identity_saes.md) alongside `encode_full` / `decode_full` (commit `11d4a4d`), and [`run_bbq_sae_steering.py`](20_run_bbq_sae_steering.md) consumes them via `install_feature_intervention_hook` (commit `84c87b5`). The numpy helpers in this file are retained as analysis-side diagnostics and have docstrings pointing at the canonical torch versions.

## Purpose

The bridge from "geometric contrast directions" to "individual SAE features." For each layer, this script computes per-feature identity selectivity and per-contrast selectivity statistics (Cohen's d, AUC, mean/freq comparisons), decoder-row alignment with the difference-of-means contrast direction, a reconstruction analysis (how much of the contrast direction is captured by top-k decoder rows), and an intervention-candidate shortlist that downstream scripts treat as the feature pool.

## Inputs

- `results/sae_identity/llama-3.1-8b/final_token/layer_XX/feature_indices_top64.npy` and `feature_values_top64.npy` — top-k SAE encodings.
- `results/sae_identity/llama-3.1-8b/final_token/layer_XX/sae_decoder.npy` — decoder rows `(n_features, hidden_dim)`.
- `results/activations/llama-3.1-8b/identity_prompts_final_token/layer_XX.npy` — raw final-token residual activations.
- `results/activations/.../metadata.csv` — row-aligned prompt metadata (`identity_id`, `axis`, `family`, `template_id`, `required_form`, `canonical_label`).
- `DEFAULT_CONTRASTS` (21 entries, lines 22-44) — the contrast registry, or `--contrasts_csv` override.

## Outputs (under `<output_dir>/`, default `.../final_token/analysis/`)

- `feature_identity_selectivity.csv` — per (layer, identity, feature) `mean_identity`, `mean_other_same_axis`, `cohens_d`, `auc`, `freq_*`.
- `feature_selectivity.csv` — per (layer, contrast, feature) Cohen's d / AUC / diff_mean and ranks.
- `decoder_direction_alignment.csv` — per (layer, contrast, feature) `cosine_with_direction`, `signed_dot`, norm.
- `feature_selectivity_alignment_joined.csv` — selectivity + alignment merged, plus `combined_score = z(|d|) + z(|cos|) + z(|auc − 0.5|)`.
- `direction_reconstruction.csv` — for `k ∈ {5, 10, 20, 50, 100, 200}` × `{decoder_alignment, selectivity, combined_score, random_baseline}`, cosine and "fraction norm captured" of the reconstructed direction.
- `intervention_candidate_features.csv` — top-N per contrast by `combined_score`, with `direction_side`, `recommended_intervention="ablate"`.
- `run_config.json`.

## Key implementation details

- Residualization (default `family_residualized`) is applied to the **activations** (`residualize()`, line 102): subtract per-family mean, add back the global mean. This affects the contrast `direction` and `evaluate_direction` scores — but the SAE encodings in `long_df` come from `encode_identity_saes.py`, which encoded **raw** activations. The two representations are mixed. See issue 5.4.
- Contrast direction (`compute_direction`, line 121) is the unit-normalized centered `mean(A) − mean(B)`, sign-flipped so identity_a scores higher.
- `feature_selectivity_for_contrast` (line 199): first filters to top `5 × top_n` features by `|diff_mean|`, then computes Cohen's d / AUC only on that subset, then keeps top `top_n` by `|d|`. This selection screen biases the reported `d`/`auc` upward. See issue 2.5.
- `reconstruct_direction` (line 323) does `coeff = basis @ direction; recon = coeff @ basis` — i.e. `BᵀB d` with `B` having unit-norm but **not orthogonal** rows. This is not an orthogonal projection. See issue 5.1.
- `combined_score` (line 481) sums three z-scored magnitudes: `|d|`, `|cos|`, `|auc − 0.5|`. Since `d` and `auc` measure the same A/B separation, this double-weights selectivity vs alignment. See issue 5.3.
- Output CSVs are **appended** (`append_csv`) per layer/contrast, so the existing run requires `--overwrite` to rebuild cleanly.
- The four numpy intervention helpers (lines 414-434) — `ablate_features_in_sae`, `steer_features_in_sae`, `decode_sae`, `patch_residual_with_sae_reconstruction` — are still defined here for analysis-side use, but the canonical torch primitives now live in [`scripts/encode_identity_saes.py`](05_encode_identity_saes.md) and are the ones consumed by [Step 20](20_run_bbq_sae_steering.md) under `--intervention_modes`. See issue 3.1 below.

## Issues & Opportunities

> **Upstream callout — issue 1.4 (FIX LANDED; regenerate inputs).** The encoder fix in [Step 5](05_encode_identity_saes.md#14-blocker--sae-preprocessing-convention-fix-landed-2026-05-26) landed in commit `4b8851a` (JumpReLU at θ=0.7539, dataset-wise input normalization, `b_dec` decode-only). Every existing `feature_*_selectivity.csv`, `decoder_direction_alignment.csv`, `direction_reconstruction.csv`, and `intervention_candidate_features.csv` was produced by the broken encoder and must be regenerated: re-run [Step 5](05_encode_identity_saes.md) with `--overwrite`, [Step 6](06_validate_sae_hook_alignment.md) to confirm `reconstruction_fvu <= 0.15`, then this script. The script logic below is unchanged; only its inputs were wrong.

### 2.1 [MAJOR] — Headline reconstruction AUC / Cohen's d are in-sample (PARTIAL FIX LANDED 2026-05-27; held-out math bundled with 2.5)

**Status:** Commit `51aa571` renamed the affected columns in `direction_reconstruction.csv` so the in-sample status is explicit. The held-out reconstruction math is scope-deferred to the 2.5 winner's-curse fix because both require held-out feature SELECTION (re-ranking by Cohen's d on non-held-out rows), not just held-out direction estimation. Bundling them avoids re-touching the selection code twice.

**What landed:**
- `direction_reconstruction.csv` columns renamed: `auc → auc_in_sample`, `cohens_d → cohens_d_in_sample`, `full_direction_auc → full_direction_auc_in_sample`, `full_direction_cohens_d → full_direction_cohens_d_in_sample`. Inline comment in `reconstruction_rows` documents the scope split.

**Remaining work (folded into 2.5):**
- For each held-out template family `f`: re-derive the contrast direction on non-`f` rows, re-rank features by Cohen's d / decoder alignment / combined score using the non-`f` direction, reconstruct using the new top-`k`, evaluate on `f`. Write `direction_reconstruction_holdout.csv` and a per-(contrast, method, k) summary.
- The `random_baseline` selection method is direction-independent and gets the held-out treatment for free (just re-run the reconstruction on held-out rows).

### 2.5 [MAJOR] — Selection-induced bias ("winner's curse") in feature effect sizes

**What's wrong:** `feature_selectivity_for_contrast` filters to the top `5 · top_n` features by `|diff_mean|`, then computes Cohen's d and AUC only on that surviving subset, then re-ranks and keeps the top `top_n` by `|d|`. Because `diff_mean` and `d` are highly correlated, the reported `d`/`auc` are conditioned on having survived a selection screen and are inflated.

**Why it matters:** Every downstream "this feature has Cohen's d = X" number in `feature_selectivity.csv` (and the `combined_score` derived from it, and the triage thresholds keyed off `|d|`, and the steering pool) is a post-selection estimate without a confirmation set.

**Targeted fix:** Either (a) compute `d`/`auc` for **all** features (cheap on sparse `long_df`) or (b) split prompts into a selection set and a confirmation set per identity pair, screen on the first and report effect sizes on the second. At minimum, drop the `5 · top_n` `|diff_mean|` prefilter — the cost of computing AUC on `n_features` is tractable. Reflect this in `feature_selectivity.csv` schema by labeling current columns as "screening-set" and adding "holdout" columns.

### 5.1 [MAJOR] — Direction reconstruction treats decoder rows as an orthonormal basis

**What's wrong:** `reconstruct_direction` (line 323) computes `basis = decoder_normed[feature_ids]`, then `coeff = basis @ direction; recon = coeff @ basis`. This is `BᵀB d`. With `B` having unit-norm rows but **not** orthogonal rows (related identity features generally are not orthogonal), the orthogonal projection of `d` onto `span(B)` is actually `Bᵀ(BBᵀ)⁻¹B d`. So `fraction_norm_captured = ||recon||²` is not bounded in `[0, 1]` and is not a fraction of anything, and `cosine_with_full_direction` is taken against a non-projection vector.

**Why it matters:** The reconstruction table is meant to answer "how much of the identity direction do `k` SAE features capture" — a natural and reviewable claim. As written the headline number in `direction_reconstruction.csv` does not have the interpretation the column name implies.

**Targeted fix:** Replace the `coeff @ basis` line with a proper least-squares solve: `coeff, *_ = np.linalg.lstsq(basis.T, direction, rcond=None); recon = coeff @ basis`. Equivalently, orthonormalize `B` via QR (`Q, _ = np.linalg.qr(basis.T); recon = (Q.T @ direction) @ Q.T`). Then `fraction_norm_captured = ||recon||² / ||direction||²` (and direction is unit-norm so this is `||recon||²` in `[0, 1]`).

### 5.3 [MINOR] — `combined_score` sums three near-duplicate, equally-weighted metrics

**What's wrong:** `combined_score = zscore(|cohens_d|) + zscore(|cos|) + zscore(|auc − 0.5|)` (line 481). Cohen's d and AUC both measure the same A/B distribution separation and are monotonically related, so the score effectively double-weights selectivity relative to decoder alignment.

**Why it matters:** Propagates into `intervention_candidate_features.csv`, into `extract_token_level_sae_activations.py:select_features` (which picks features by `combined_score`), into `build_sae_feature_cards.py`, and into the triage. The "top by combined_score" features are systematically biased toward selectivity-only features over alignment-only features.

**Targeted fix:** Pick **one** selectivity metric (d **or** AUC) and combine with decoder cosine. E.g. `combined_score = 0.5 zscore(|d|) + 0.5 zscore(|cos|)`. Document the weighting choice in `run_config.json`. Re-run downstream feature selection.

### 5.4 [MINOR] — Residualized direction vs raw-encoded SAE features inconsistency

**What's wrong:** The contrast `direction` is computed from `family_residualized` activations (line 466), but the SAE feature values in `long_df` come from `encode_identity_saes.py`, which encoded **raw** (non-residualized) activations. `decoder_alignment` then takes the cosine between a raw-space decoder row and a residualized-space direction, and `combined_score` mixes a residualized-direction cosine with a raw-SAE-activation `cohens_d`.

**Why it matters:** The two quantities live in slightly different spaces. The "decoder alignment" claim is implicitly "this decoder row points in the residualized direction" but the decoder was trained on raw residuals.

**Targeted fix:** Pick one representation and stick with it. Two consistent choices:
- **Raw end-to-end:** drop residualization in this script, accept that template/family variance is baked into the direction (and document it).
- **Residualized end-to-end:** re-encode the residualized activations through the SAE (call `relu((x_resid - b_dec) @ w_enc + b_enc)`), and rebuild `long_df` from those activations. The encoder lives in `encode_identity_saes.py`; either import it or factor the encode step into a shared helper.

Add the chosen mode to `run_config.json` and assert downstream.

### 3.1 (load-bearing context) [BLOCKER] — Feature-level intervention helpers exist here but are never used (FIX LANDED 2026-05-27)

**Status:** Closed across commits `11d4a4d` (canonical torch primitives in `encode_identity_saes.py`) and `84c87b5` (BBQ steering hook wired through them). The legacy numpy helpers in this file are kept for analysis-side use and explicitly point at the canonical implementations in their docstrings.

**What landed:**
- Canonical torch primitives in [Step 5 — `encode_identity_saes.py`](05_encode_identity_saes.md): `ablate_features`, `clamp_features`, `steer_features`, `patched_residual_with_intervention(h, sae, intervention_fn)`. The wrapper handles the full encode → modify-latent-f → decode → patch loop and accounts for the audit 1.4 dataset-wise normalization (the patch math operates in un-normalized residual space; `scale_out` is folded in by `decode_full`).
- [Step 20 — `run_bbq_sae_steering.py`](20_run_bbq_sae_steering.md) now exposes `--intervention_modes` values `ablate`, `clamp`, `steer` (in addition to legacy `add_vector` / `ablate_projection`), with `ablate` as the default. The hook dispatch (`make_batched_hook_fn` / `make_hook_fn`) routes feature modes through the encode → modify → decode → patch loop on the actual SAE; legacy modes still build the decoder-direction vector for the audit 5.5 baseline.
- The numpy helpers in this file (`ablate_features_in_sae`, `steer_features_in_sae`, `decode_sae`, `patch_residual_with_sae_reconstruction`) now carry docstrings pointing at the canonical torch versions and explaining that `decode_sae` here is pure normalized-space decode — the corrected residual-space decode is `decode_full` in Step 5.

**Original audit (preserved):** Lines 414-434 of this file defined feature-level intervention helpers that were exactly the primitives needed for the encode → modify-latent-f → decode → patch loop, but `main()` did not call any of them and `run_bbq_sae_steering.py` used decoder-vector addition. As long as steering was "add a unit-norm direction at one token," the SAE contributed only a direction; the headline causal claim could not be supported.

### 5.10 [MINOR] — Heavy code duplication across analysis scripts (FIX LANDED 2026-05-27)

**Status:** All shared helpers — `cohens_d`, `cosine`, `normalize`, `compute_direction`, `evaluate_projection`, `residualize`, `OKABE_ITO`, `save_fig`, `CenterOnlyScaler` + `make_scaler` — now live in `scripts/common.py` (commit `e50bbd1`). The canonical contrast list lives in `scripts/contrast_registry.py` (commit `1e242c9`; audit 4.1). This script's local copies are gone; any remaining definitions are thin adapter wrappers that preserve the prior return-tuple shapes while routing through `common.py`. Net change across the 8 consumer scripts: 358 lines added to `common.py`, 369 lines removed elsewhere.

## Rebuild checklist

- [ ] Audit `DEFAULT_CONTRASTS` (lines 22-44) against `bbq_identity_normalized_forms.csv`. Drop or rename `ses_low_income`, `ses_high_socioeconomic_status`; replace with `ses_low_income → ses_low` or `ses_poor`, and add the contrasts that should have been there. Tie this to the shared registry (5.10).
- [ ] Decide on a single representation: residualize end-to-end (re-encode through SAE on residualized activations) **or** keep everything raw. Document in `run_config.json`. (5.4)
- [ ] Replace `feature_selectivity_for_contrast`'s `|diff_mean|` prefilter with full computation, or split prompts into selection/confirmation halves and add holdout columns. (2.5)
- [ ] Replace `reconstruct_direction` with a proper least-squares projection (`np.linalg.lstsq` or QR). Re-derive `fraction_norm_captured` as the normalized squared norm. (5.1)
- [ ] Rebalance `combined_score`: use one of `|d|`/`|auc-0.5|` plus `|cos|` with a documented weighting. (5.3)
- [x] Extract canonical SAE intervention primitives into a shared module so `run_bbq_sae_steering.py` can consume them. (3.1) *(Done 2026-05-27: torch primitives in `scripts/encode_identity_saes.py`, commits `11d4a4d` + `84c87b5`.)*
- [ ] Convert `load_contrasts`'s silent `print` into a `warnings.warn` (or fail) so missing identity IDs cannot be masked. (4.1)
- [ ] Re-run with `--overwrite` after the above; downstream scripts (token-level extraction, feature cards, triage) must be re-run to pick up the new CSVs.

## Notes from the doc audit

- `intervention_candidate_features.csv` hardcodes `"recommended_intervention": "ablate"` for every row (line 408). With 3.1 landed (2026-05-27) and `--intervention_modes ablate` as the new default in [Step 20](20_run_bbq_sae_steering.md), this field is now consistent with the production intervention. The field is still informational — the steering runner reads `feature_ids` from the triage CSV, not this column — but it now describes what actually runs.
- `load_contrasts` skips silently with only a `print` (line 98). Combined with the missing-ID problem in `DEFAULT_CONTRASTS`, the SES axis effectively runs with two contrasts instead of four, with no error.
- `decoder_direction_alignment.csv` only writes the top-N by each of three ranks per contrast (lines 483-487), so a feature that is mid-rank in every contrast may not appear in this CSV at all — `triage_sae_identity_features.py` joins on this file and the missing rows fall back to `max_abs_decoder_cosine = 0`. Either widen the write here or use the full `alignment` frame.
