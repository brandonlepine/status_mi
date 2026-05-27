# Step 8 — `analyze_identity_geometry_diagnostics.py`

**Stage:** 2 — Identity-geometry analyses (second pass / surface-form controls)
**Runs after:** `extract_identity_activations.py` (Stage 1); typically also after [Step 7](07_analyze_identity_geometry.md), though it does not consume Step 7's outputs.
**Feeds into:** Diagnostic figures, downstream SAE-feature analysis (which by default uses `family_residualized` activations), [Step 11](11_plot_identity_directional_visualizations.md), [Step 12](12_plot_identity_directional_followups.md).

## Purpose
Asks the central robustness question for the geometry pipeline: **does identity structure survive controls for prompt surface form?** Computes variance decomposition (η²) by metadata factor, residualizes activations against `family` / `template_id` / `required_form`, then re-runs PCA, identity probes, surface-form probes, and contrasts under each residualization. Produces per-layer plots and per-axis PCA scatters at `selected_layers_for_plots` (default `0,8,16,24,32`).

## Inputs
- `results/activations/llama-3.1-8b/identity_prompts_final_token/layer_XX.npy` (per-layer final-token residuals)
- `results/activations/.../identity_prompts_final_token/metadata.csv`
- Optionally `--geometry_dir` (default sibling of `--output_dir`) if any first-pass outputs are reused

## Outputs
- `diagnostics/variance_decomposition.csv` — one row per `(layer, factor)` with η² for `family`, `template_id`, `required_form`, `axis`, `identity_id`
- `diagnostics/pca_residualized/{raw, family_residualized, template_residualized, required_form_residualized}/pca_layer_XX.csv` + `pca_explained_variance.csv`
- `diagnostics/probes/axis_probe_residualized_scores.csv`, `identity_within_axis_probe_residualized_scores.csv`, `surface_form_probe_scores.csv`
- `diagnostics/contrasts/contrast_full_residualized_scores.csv` (full-data AUC/d), `contrast_family_holdout_residualized_scores.csv`
- `diagnostics/figures/...` — variance-decomposition curves, probe macro-F1 by layer × residualization, contrast AUC by layer, per-axis PCA scatters / progression panels, optional UMAP
- `diagnostics/run_config.json`

## Key implementation details
- **Residualization variants.** `RESIDUALIZATIONS = {"raw": None, "family_residualized": "family", "template_residualized": "template_id", "required_form_residualized": "required_form"}`. `residualize(x, metadata, group_col)` subtracts the per-`group_col` mean and adds back the global mean, yielding a per-group-mean-zero activation.
- **Variance decomposition.** `variance_decomposition_layer` computes between-group SS / total SS as `eta_squared = ss_factor / ss_total` for each factor in `FACTORS = ["family", "template_id", "required_form", "axis", "identity_id"]`. One row per (layer, factor).
- **Probe model.** `LogisticRegression(class_weight="balanced", solver="saga" by default, max_iter=500)` via the same `crossval_probe` pattern as [Step 7](07_analyze_identity_geometry.md) (`make_probe_features` does `StandardScaler` + `PCA(probe_pca_dim=64)`).
- **Surface-form probes (raw only).** Predict `required_form` and `family` from activations, grouped by `identity_id`. These are the *positive controls* for surface leakage — if identity probes are high but surface-form probes on raw activations are also high, the identity probes may be picking up surface form.
- **Contrasts.** Same `CONTRASTS` list as [Step 7](07_analyze_identity_geometry.md); for each residualization computes both full-data and family-holdout AUC/d.
- **Resume + partial runs.** `--resume` reuses existing per-layer outputs. `--skip_*` and `--only_*` flags allow rerunning a single phase (e.g. `--only_variance`, `--skip_umap`, `--skip_surface_form_probes`).
- **Plots.** Static matplotlib (Okabe-Ito palette) for variance decomposition, probe curves, contrast AUC, and per-axis PCA scatters at the layers in `--selected_layers_for_plots`.

## Strengths to preserve (audit Section 6)

These are explicitly called out as keepers in `docs/issues_and_opportunities.md` §6 and should survive any refactor:

- **Surface-form residualization diagnostics.** Residualizing by `family` / `template_id` / `required_form` and re-running PCA / probes / contrasts directly attacks the "are we measuring identity or template?" confound. This is the right instinct and a genuine project strength.
- **Family-holdout / family-to-family generalization.** Training a direction on some families and evaluating on held-out ones is a real generalization test. (Make these the headline numbers — see 2.1 below.)
- **Variance decomposition (η²)** by metadata factor is a clean, honest way to show how much variance identity explains relative to surface form.

Do not collapse the residualization grid into a single "best" residualization — the comparison across `raw`, `family`, `template_id`, `required_form` is itself a result.

## Issues & Opportunities

### 2.1 [MAJOR] — Headline contrast AUC / Cohen's d are in-sample (FIX LANDED 2026-05-27)

**Status:** Commit `e15e62f`. Same fix shape as [Step 7](07_analyze_identity_geometry.md#21-major--headline-contrast-auc--cohens-d-are-in-sample-fix-landed-2026-05-27).

**What landed:**
- `CONTRAST_COLUMNS` split into `CONTRAST_IN_SAMPLE_COLUMNS` and `CONTRAST_HOLDOUT_COLUMNS`; the two CSVs now have distinct schemas (in-sample writes `auc_in_sample`/`cohens_d_in_sample`, held-out writes `auc`/`cohens_d`). Backwards-compatible alias `CONTRAST_COLUMNS = CONTRAST_HOLDOUT_COLUMNS`.
- `run_contrasts` populates the renamed in-sample columns.
- New `_write_holdout_summary` helper emits `contrast_family_holdout_residualized_summary.csv` — the headline number per (layer, residualization, contrast).
- The residualization comparison story is now honest: differences between residualizations are visible on the held-out summary, not on a metric that the difference-of-means direction is guaranteed to maximize.

**Remaining work (followup plotting):** `plot_identity_directional_followups.py` still computes in-sample AUC per residualization for the projection histogram titles. Those numbers correctly describe the projection distributions being plotted (they're inherently in-sample), but the title text should explicitly call them in-sample to avoid confusion. That's a small docs-side label tweak, not a logic change.

### 2.2 [BLOCKER] — No null model for the central claims (PROBE NULL LANDED 2026-05-27; η² / contrast / SVD nulls still pending)

**Status:** The probe permutation null landed in the diagnostics script. The other parts of audit 2.2 (η² under shuffled labels, contrast AUC null, shared-subspace SVD null) are still open.

**What landed in this script:**
- `_run_cv_folds_diag` extracts the per-fold loop; observed and null share one implementation.
- `crossval_probe` accepts `n_permutations` and `null_rng_seed`. Same shuffle strategy as [Step 7](07_analyze_identity_geometry.md): global y shuffle per replicate, GroupKFold split structure preserved.
- New CLI: `--n_permutations` (default `20`) and `--null_random_seed` (defaults to `--random_seed`).
- All three diagnostic probes (axis prediction, identity-within-axis prediction, surface-form probes for `required_form`/`family`/`template_id`) carry the null through. Output rows gain `null_n_permutations`, `null_accuracy_mean`, `null_accuracy_sd`, `null_macro_f1_mean`, `null_macro_f1_sd`, `accuracy_z`, `macro_f1_z`, `accuracy_p_value`, `macro_f1_p_value` (Phipson-Smyth smoothed).

**Original audit (preserved for context):** The residualized probe and contrast numbers were reported absolutely, with no permutation null. Across four residualizations the implicit comparison "raw vs residualized" was also uncalibrated — it was unclear how much the residualization-induced drop in AUC exceeds what shuffling `identity_id` labels would produce.

**Remaining work:**
- Add a permutation null for the variance decomposition (η²) under shuffled identity labels.
- Add a permutation null for the per-contrast AUC / family-holdout AUC (similar shuffle, applied to the contrast-direction projection).
- Add a null for the shared-subspace SVD spectrum in [Step 9](09_analyze_shared_social_subspace.md): random unit vectors vs shuffled-identity directions.
- Probe nulls are surfaced in the existing CSV columns; the remaining nulls will need new sidecar CSVs (`variance_decomposition_null.csv`, `contrast_null_distribution.csv`).

### 2.8 [MINOR] — Probe dimensionality reduction leaks across CV folds (VERIFIER LANDED 2026-05-27)

Same design and same fix as [Step 7](07_analyze_identity_geometry.md#28-minor--probe-dimensionality-reduction-leaks-across-cv-folds-verifier-landed-2026-05-27). The diagnostics script also fits `StandardScaler + PCA` once globally in `make_probe_features`. New `crossval_probe_fold_internal_pca_diag` mirrors the LogisticRegression configuration (`solver`, `max_iter`, `n_jobs`) so the global-vs-fold-internal comparison is apples-to-apples. `--verify_fold_internal_pca <layer>` runs the verifier at `residualization == "raw"` across all three probe configurations (axis, identity-within-axis, surface-form) and writes `probes/pca_leakage_verification.csv`.

### 4.1 [MAJOR] — Contrast lists reference identities that do not exist (FIX LANDED 2026-05-27)

**Status:** Same fix as [Step 7](07_analyze_identity_geometry.md#41-major--contrast-lists-reference-identities-that-do-not-exist-fix-landed-2026-05-27). This script's `CONTRASTS` literal is gone; `resolve_contrasts_from_registry(metadata, args.output_dir)` populates the module-level list at `main()` startup and writes `output_dir/contrasts/contrasts_skipped.csv`. The shared registry lives in `scripts/contrast_registry.py`. **No startup assertion** — partial-axis runs work and the skipped CSV preserves the audit trail. SES axis runs all 4 contrasts now.

### 5.9 [MINOR] — PCA on StandardScaler-ed activations changes the geometry

**What's wrong:** Same as in [Step 7](07_analyze_identity_geometry.md): `StandardScaler` before `PCA` for both `pca_residualized/...` outputs and `make_probe_features`.

**Why it matters:** Explained-variance ratios describe standardized space; comparing them across residualizations is still meaningful, but absolute values should be stated as "post-z-score."

**Targeted fix:** Add `--scaling {standardize, center_only}` and record in `run_config.json`. Show that residualization conclusions are stable under both.

### 5.10 [MINOR] — Heavy code duplication across analysis scripts

**What's wrong:** `cohens_d`, `contrast_direction`, `residualize`, `OKABE_ITO`, `CONTRASTS`, `evaluate_contrast_scores` are copy-pasted from [Step 7](07_analyze_identity_geometry.md). The residualization implementation here is the canonical one — `analyze_shared_social_subspace.py` and the directional plotting scripts copy it.

**Why it matters:** A future fix to residualization (e.g. changing how the global-mean offset is handled) requires editing four files in sync.

**Targeted fix:** Move `residualize` and the residualization map into `status_mi/common.py` as the source of truth; import from there in this script and in [Step 9](09_analyze_shared_social_subspace.md), [Step 11](11_plot_identity_directional_visualizations.md), [Step 12](12_plot_identity_directional_followups.md).

## Rebuild checklist
- [ ] Import `CONTRASTS` / `residualize` / `cohens_d` / `evaluate_contrast_scores` from the shared module; remove the local copies.
- [ ] Add permutation-null support to `crossval_probe` and to `run_contrasts` (held-out and full variants); write `*_null_mean`, `*_null_sd`, `*_z_score`, `*_p_value` columns.
- [ ] Rewire `plot_contrast_full_auc_residualized_by_layer` to read held-out AUC by default; rename the in-sample plot `*_in_sample_diagnostic`.
- [ ] Add `--scaling {standardize, center_only}`; rerun with `center_only` and confirm residualization conclusions hold.
- [ ] Fit `StandardScaler` + `PCA` inside each CV fold in `make_probe_features`, or document the leakage and provide a one-layer sensitivity check.
- [ ] Preserve the η² / surface-form-probe / family-holdout machinery — these are project strengths.

## Notes from the doc audit
- `--n_splits` defaults to 3 here but 5 in [Step 7](07_analyze_identity_geometry.md). When comparing probe macro-F1 across the two scripts (which is tempting because both publish per-layer probe accuracy), the difference in fold count is a real confound.
- The diagnostics script supports `--solver saga` and `--max_iter 500` whereas Step 7 hardcodes `lbfgs` and `max_iter=2000`; document this in the rebuild plan.
