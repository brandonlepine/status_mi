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

- The script is **raw end-to-end** (audit 5.4 closed 2026-05-27 in commit `ebfdff7`). Contrast `direction`, `decoder_alignment`, `feature_selectivity_for_contrast`, and `reconstruction_rows` all operate on the same raw activations the SAE was trained on. The `--residualization` flag and the `residualize(x, …)` call were removed; `run_config.json` now records `representation: "raw"` with an audit note. Prior default (`family_residualized`) mixed a residualized-space direction with raw-SAE-space Cohen's d.
- Contrast direction (`compute_direction`, line 121) is the unit-normalized centered `mean(A) − mean(B)`, sign-flipped so identity_a scores higher.
- `feature_selectivity_for_contrast` (line 199): computes Cohen's d and AUC analytically for **all** `n_features` via `compute_cohens_d_and_auc_for_all_features` (Cohen's d from sum/sum_sq/count using sample variance; AUC via the sparse 4-bucket decomposition), then ranks and keeps top `top_n` by `|d|`. The prior code prefiltered to `5 × top_n` by `|diff_mean|` before computing d/AUC, which biased the reported effect sizes upward. Audit 2.5 closed 2026-05-27.
- `reconstruct_direction` solves the least-squares problem `argmin_c ||basis.T @ c − direction||²` via `np.linalg.lstsq` and sets `recon = c @ basis` — the true orthogonal projection of `direction` onto `span(rows of basis)`. `fraction_norm_captured = ||recon||² / ||direction||²` is now correctly in `[0, 1]` and equals `cosine_with_full_direction²` (projection identity). Audit 5.1 closed 2026-05-27 in commit `1a569c3`. The prior `coeff = basis @ direction; recon = coeff @ basis` (= `BᵀB d`) was only the orthogonal projection when `B Bᵀ = I`, which decoder rows don't satisfy.
- `combined_score = 0.5·z(|cohens_d|) + 0.5·z(|cosine_with_direction|)` (audit 5.3 closed 2026-05-27 in commit `3b48e5b`). The prior formula added `z(|auc − 0.5|)` as a third term, which double-weighted selectivity (d and auc are monotonically related) vs decoder alignment. The weights are surfaced in `run_config.json` under `combined_score_weights` / `combined_score_formula`.
- Output CSVs are **appended** (`append_csv`) per layer/contrast, so the existing run requires `--overwrite` to rebuild cleanly.
- The four numpy intervention helpers (lines 414-434) — `ablate_features_in_sae`, `steer_features_in_sae`, `decode_sae`, `patch_residual_with_sae_reconstruction` — are still defined here for analysis-side use, but the canonical torch primitives now live in [`scripts/encode_identity_saes.py`](05_encode_identity_saes.md) and are the ones consumed by [Step 20](20_run_bbq_sae_steering.md) under `--intervention_modes`. See issue 3.1 below.

## Issues & Opportunities

> **Upstream callout — issue 1.4 (FIX LANDED; regenerate inputs).** The encoder fix in [Step 5](05_encode_identity_saes.md#14-blocker--sae-preprocessing-convention-fix-landed-2026-05-26) landed in commit `4b8851a` (JumpReLU at θ=0.7539, dataset-wise input normalization, `b_dec` decode-only). Every existing `feature_*_selectivity.csv`, `decoder_direction_alignment.csv`, `direction_reconstruction.csv`, and `intervention_candidate_features.csv` was produced by the broken encoder and must be regenerated: re-run [Step 5](05_encode_identity_saes.md) with `--overwrite`, [Step 6](06_validate_sae_hook_alignment.md) to confirm `reconstruction_fvu <= 0.15`, then this script. The script logic below is unchanged; only its inputs were wrong.

### 2.1 [MAJOR] — Headline reconstruction AUC / Cohen's d are in-sample (FIX LANDED 2026-05-27/28)

**Status:** Commit `51aa571` renamed the in-sample columns; commit `304ddb6` (2026-05-28) added the held-out reconstruction. Both the contrast direction (2.1) and the feature selection (2.5) are now re-derived per held-out fold.

**What landed:**
- `direction_reconstruction.csv` columns renamed `*_in_sample` (commit `51aa571`); retained as the in-sample diagnostic.
- New `reconstruction_holdout_rows` (commit `304ddb6`): for each held-out prompt `family` `f`, re-derive the contrast direction on non-`f` rows, re-rank features by Cohen's d / decoder alignment / combined score using the non-`f` direction, reconstruct using the new top-`k`, and evaluate on `f`. `summarize_reconstruction_holdout` aggregates to one row per (layer, contrast, method, k). Writes `direction_reconstruction_holdout.csv` (per fold) + `direction_reconstruction_holdout_summary.csv` (headline). The `random_baseline` method gets the held-out treatment for free. If `metadata.csv` lacks a `family` column the pass is skipped with a one-time warning.

**Validation (synthetic, 15/15, decoder = identity):** with a true A/B signal held-out selectivity/combined AUC ≈ 0.99 and beats the random baseline; **with pure noise the held-out AUC collapses to ≈ 0.54 (chance)** — the in-sample optimism is removed; folds == families and each is evaluated only on its held-out rows.

### 2.5b held-out feature selection (identity screen)

The held-out reconstruction above re-selects features per fold on train rows, which closes the identity-screen half of audit 2.5 (held-out feature *selection*). The per-feature Cohen's d in `feature_selectivity.csv` remains an in-sample descriptive screen statistic by design; the held-out generalization evidence lives in `direction_reconstruction_holdout_summary.csv`.

### 2.5 [MAJOR] — Selection-induced bias ("winner's curse") in feature effect sizes (FIX LANDED 2026-05-27)

**Status:** Closed across commits `4481445` (prefilter removed) and `304ddb6` (held-out feature selection). The `|diff_mean|` prefilter (5×top_n in `feature_selectivity_for_contrast`, 3×top_n in `identity_selectivity`) is gone — Cohen's d and AUC are now computed for every feature in closed form, then the ranking selects top `top_n`. The held-out-confirmation half is closed via the leave-one-family-out reconstruction (see 2.1 above and the 2.5b note below).

**What landed:**
- New helper `compute_cohens_d_and_auc_for_all_features(long_df, mask_a, mask_b, df_groups, prefix_a, prefix_b)` computes both metrics for every row of `df_groups` (i.e. every feature) without a screening prefilter.
- Cohen's d is derived from `(sum, sum_sq, count)` per (feature, group) — sample variance with `ddof=1`, matching `common.cohens_d` to floating-point precision.
- AUC uses a sparse 4-bucket decomposition that exploits the fact that `long_df` only holds positive activations:
  - **B1 (both zero):** AUC = 0.5 (all pairs tie at 0).
  - **B2 (a-only nonzeros):** AUC = `(k_a + 0.5·(n_a − k_a)) / n_a` (a's zeros tie b's zeros; a's nonzeros beat b's zeros).
  - **B3 (b-only nonzeros):** AUC = `0.5·(n_b − k_b) / n_b` (a's zeros tie b's zeros; a's zeros lose to b's nonzeros).
  - **B4 (both groups have nonzeros):** direct comparison of the (typically small) nonzero arrays plus the closed-form contributions from the three zero-pair buckets.
- B1-B3 are fully vectorized; only B4 features (typically a small subset for identity-selective SAE features) loop. `summarize_feature_groups` was extended to also return `sum_sq_{prefix}` so variance is derivable per feature.
- `feature_selectivity_for_contrast` and `identity_selectivity` were both refactored to use the helper. Output schemas (`feature_selectivity.csv`, `feature_identity_selectivity.csv`) are unchanged.

**Validation:** Synthetic sparse data covering all four AUC buckets — vectorized helpers match the per-feature reference loop (`sklearn.roc_auc_score` + `common.cohens_d`) to ~1e-16. Hidden low-variance high-d features (e.g. `diff_mean = 0.005` with tiny pooled SD) now surface at the top of the ranking; under the old prefilter they were silently discarded.

**Held-out selection (closed 2026-05-28, commit `304ddb6`):** the leave-one-family-out `reconstruction_holdout_rows` re-ranks features on the train rows of each fold and evaluates the reconstruction on the held-out family — the proper winner's-curse correction for feature selection, sharing the held-out reconstruction plumbing (audit 2.1). See the 2.1 section above. The per-feature Cohen's d in `feature_selectivity.csv` stays an in-sample descriptive screen statistic; the held-out evidence is `direction_reconstruction_holdout_summary.csv`.

**Original audit (preserved):** `feature_selectivity_for_contrast` filtered to the top `5 · top_n` features by `|diff_mean|`, then computed Cohen's d and AUC only on that surviving subset, then re-ranked and kept the top `top_n` by `|d|`. Because `diff_mean` and `d` are highly correlated, the reported `d`/`auc` were conditioned on having survived a selection screen and inflated. Every downstream "this feature has Cohen's d = X" number — in `feature_selectivity.csv`, in the `combined_score` derived from it, in the triage thresholds keyed off `|d|`, and in the steering pool — was a post-selection estimate without a confirmation set.

### 5.1 [MAJOR] — Direction reconstruction treats decoder rows as an orthonormal basis (FIX LANDED 2026-05-27)

**Status:** Closed in commit `1a569c3`. `reconstruct_direction` now solves the least-squares problem `argmin_c ||basis.T @ c − direction||²` via `np.linalg.lstsq`; the minimizer `recon = c @ basis` is the true orthogonal projection of `direction` onto `span(rows of basis)`. Output schema of `direction_reconstruction.csv` unchanged — only the values become correct (and bounded).

**What landed:**
- `fraction_norm_captured = ||recon||² / ||direction||²` is now in `[0, 1]` by construction (orthogonal projection always satisfies `||proj|| ≤ ||original||`).
- `cosine_with_full_direction = sqrt(fraction_norm_captured)` follows from the projection identity `direction · recon = ||recon||²`. The two columns are now algebraically related (squaring one gives the other), which is the right invariant.
- Defensive: `fraction` is computed with `||direction||²` in the denominator rather than assuming the direction is unit-norm.

**Validation:**
- 200 random trials over `k ∈ [3, 50]`, `d_model = 256`: new fraction always in `[0, 1]`; projection identity holds to ~1e-15.
- Direction lying in `span(basis)` → `fraction = 1.0` exactly.
- Orthonormal basis (the degenerate case where the old formula was correct) → new matches old to numerical zero.
- Pathological case (10 basis rows with average mutual cosine 0.77, direction along the cluster): old gives `fraction = 74.7` (nonsensical), new gives `0.984` (correct: direction lies almost entirely in span).

**Downstream re-validation:**
- [Step 16 — `plot_identity_sae_features.py`](16_plot_identity_sae_features.md) reads `cosine_with_full_direction` and `fraction_norm_captured` directly from the CSV; the curves will now be bounded and interpretable. No code change needed there.
- [Step 17 — `triage_sae_identity_features.py`](17_triage_sae_identity_features.md): `aggregate_signal_metrics` references the fraction column; re-validate its thresholds against the (now-correctly-bounded) values.

**Original audit (preserved):** `reconstruct_direction` computed `basis = decoder_normed[feature_ids]`, then `coeff = basis @ direction; recon = coeff @ basis` — i.e. `BᵀB d`. With `B` having unit-norm rows but **not** orthogonal rows (related identity features generally are not orthogonal), the orthogonal projection of `d` onto `span(B)` is actually `Bᵀ(BBᵀ)⁻¹B d`. So `fraction_norm_captured = ||recon||²` was not bounded in `[0, 1]` and was not a fraction of anything; `cosine_with_full_direction` was taken against a non-projection vector. The reconstruction table is meant to answer "how much of the identity direction do `k` SAE features capture" — a natural and reviewable claim — and the headline number in `direction_reconstruction.csv` did not have the interpretation the column name implied.

### 5.3 [MINOR] — `combined_score` sums three near-duplicate, equally-weighted metrics (FIX LANDED 2026-05-27)

**Status:** Closed in commit `3b48e5b`. Selectivity uses Cohen's d only; alignment uses `|cosine_with_direction|`; weights are 50/50 and documented in `run_config.json`.

**What landed:**
- New formula: `combined_score = 0.5·zscore(|cohens_d|) + 0.5·zscore(|cosine_with_direction|)`.
- `run_config.json` now contains `combined_score_weights` (`{"cohens_d": 0.5, "decoder_cosine": 0.5}`), `combined_score_formula` (the literal expression as a string), and `combined_score_audit_note` (rationale + reference to audit 5.3). Any downstream consumer that loads the config can see the choice.
- Schema of `feature_selectivity_alignment_joined.csv` unchanged.

**Validation (synthetic):** Pathological pair — Feature A has high d+auc but low cosine, Feature B has low d+auc but high cosine. Old formula favored A by +2.00 z-units (d and auc both swing it); new formula puts them at parity (50/50 weight). On a realistic 500-feature sweep with d↔auc correlated, Spearman ρ(old, new) = 0.92 — high but not 1, so the rebalance correctly shifts ~8% of the ranking toward alignment-strong features.

**Downstream:** `plot_identity_sae_features.py`, `triage_sae_identity_features.py`, `build_sae_feature_cards.py`, and `run_bbq_sae_steering.py` all sort by `combined_score`; the corrected ranking propagates on next regeneration. The "top by combined_score" features under the old formula were systematically biased toward selectivity-strong features over alignment-strong features.

**Original audit (preserved):** `combined_score = zscore(|cohens_d|) + zscore(|cos|) + zscore(|auc − 0.5|)`. Cohen's d and AUC both measure the same A/B distribution separation and are monotonically related, so the score effectively double-weighted selectivity relative to decoder alignment.

### 5.4 [MINOR] — Residualized direction vs raw-encoded SAE features inconsistency (FIX LANDED 2026-05-27)

**Status:** Closed in commit `ebfdff7` (`scripts/analyze_identity_sae_features.py`). The script is now raw end-to-end — the simpler of the two consistent choices the audit proposed.

**What landed:**
- `--residualization` CLI flag removed. `RESIDUALIZATION_GROUPS` constant and the local `residualize()` adapter removed. The `residualize(x, …)` call in `main()` removed.
- `decoder_alignment` and `reconstruction_rows` no longer take a `residualization` parameter, and the `residualization` column is gone from `decoder_direction_alignment.csv` and `direction_reconstruction.csv`. Verified by grep that no downstream consumer (`plot_identity_sae_features.py`, `triage_sae_identity_features.py`, `build_sae_feature_cards.py`, `extract_token_level_sae_activations.py`, `run_bbq_sae_steering.py`) reads the dropped column.
- `run_config.json` now records `"representation": "raw"` and a `representation_audit_note` explaining the choice.

**Caveat (template/family variance baked into direction):** Without residualization, template-specific or family-specific variance is part of the contrast direction. For the SAE-features path this is acceptable because the SAE was trained on raw activations and "sees" the same variance; the alternative (residualize → re-encode through SAE) would require importing `encode_full` from Step 5 and regenerating `long_df` from residualized activations. If that becomes the preferred path later, it's a deliberate addition rather than a flag toggle.

**Original audit (preserved):** The contrast `direction` was computed from `family_residualized` activations (line 466), but the SAE feature values in `long_df` came from `encode_identity_saes.py`, which encoded **raw** (non-residualized) activations. `decoder_alignment` then took the cosine between a raw-space decoder row and a residualized-space direction, and `combined_score` mixed a residualized-direction cosine with a raw-SAE-activation `cohens_d`. The two quantities lived in slightly different spaces. The "decoder alignment" claim implicitly said "this decoder row points in the residualized direction" but the decoder was trained on raw residuals.

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
- [x] Decide on a single representation: residualize end-to-end (re-encode through SAE on residualized activations) **or** keep everything raw. Document in `run_config.json`. (5.4) *(Done 2026-05-27: commit `ebfdff7` — raw end-to-end; flag removed; recorded as `representation: "raw"` in run_config.json.)*
- [x] Replace `feature_selectivity_for_contrast`'s `|diff_mean|` prefilter with full computation. (2.5) *(Done 2026-05-27: commit `4481445` — analytical d/AUC for all features; identity_selectivity also fixed.)* **Still open:** holdout-set confirmation columns, bundled with 2.1 winner's-curse correction.
- [x] Replace `reconstruct_direction` with a proper least-squares projection (`np.linalg.lstsq` or QR). Re-derive `fraction_norm_captured` as the normalized squared norm. (5.1) *(Done 2026-05-27: commit `1a569c3`.)*
- [x] Rebalance `combined_score`: use one of `|d|`/`|auc-0.5|` plus `|cos|` with a documented weighting. (5.3) *(Done 2026-05-27: commit `3b48e5b` — 0.5·z(|d|) + 0.5·z(|cos|), weights in run_config.json.)*
- [x] Extract canonical SAE intervention primitives into a shared module so `run_bbq_sae_steering.py` can consume them. (3.1) *(Done 2026-05-27: torch primitives in `scripts/encode_identity_saes.py`, commits `11d4a4d` + `84c87b5`.)*
- [ ] Convert `load_contrasts`'s silent `print` into a `warnings.warn` (or fail) so missing identity IDs cannot be masked. (4.1)
- [ ] Re-run with `--overwrite` after the above; downstream scripts (token-level extraction, feature cards, triage) must be re-run to pick up the new CSVs.

## Notes from the doc audit

- `intervention_candidate_features.csv` hardcodes `"recommended_intervention": "ablate"` for every row (line 408). With 3.1 landed (2026-05-27) and `--intervention_modes ablate` as the new default in [Step 20](20_run_bbq_sae_steering.md), this field is now consistent with the production intervention. The field is still informational — the steering runner reads `feature_ids` from the triage CSV, not this column — but it now describes what actually runs.
- `load_contrasts` skips silently with only a `print` (line 98). Combined with the missing-ID problem in `DEFAULT_CONTRASTS`, the SES axis effectively runs with two contrasts instead of four, with no error.
- `decoder_direction_alignment.csv` only writes the top-N by each of three ranks per contrast (lines 483-487), so a feature that is mid-rank in every contrast may not appear in this CSV at all — `triage_sae_identity_features.py` joins on this file and the missing rows fall back to `max_abs_decoder_cosine = 0`. Either widen the write here or use the full `alignment` frame.
