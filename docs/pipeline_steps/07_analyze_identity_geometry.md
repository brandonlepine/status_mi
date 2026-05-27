# Step 7 — `analyze_identity_geometry.py`

**Stage:** 2 — Identity-geometry analyses (first pass)
**Runs after:** `extract_identity_activations.py` (Stage 1)
**Feeds into:** [Step 8](08_analyze_identity_geometry_diagnostics.md), [Step 9](09_analyze_shared_social_subspace.md), [Step 10](10_plot_identity_geometry.md), [Step 11](11_plot_identity_directional_visualizations.md), [Step 12](12_plot_identity_directional_followups.md), and `analyze_identity_sae_features.py`.

## Purpose
First-pass characterization of identity geometry from the final-token residual stream at every layer. Produces the headline numbers that downstream scripts and plots consume: PCA scores, group means (identity, identity×family, axis), logistic probe accuracies (axis prediction and identity-within-axis prediction), family-stability cosines for each identity, and the **contrast direction** metrics (`auc_all`, `cohens_d_all`, family-holdout AUC/d) that the rest of the project treats as evidence that identities are linearly separable.

## Inputs
- `results/activations/llama-3.1-8b/identity_prompts_final_token/layer_XX.npy` (per-layer final-token residuals; memmapped float32)
- `results/activations/.../identity_prompts_final_token/metadata.csv` (row-aligned with `.npy`; columns `prompt_id, prompt, identity_id, axis, canonical_label, template_id, family, required_form, form_used`)

## Outputs
- `results/geometry/.../pca/pca_layer_XX.csv`, `pca/pca_explained_variance.csv`
- `results/geometry/.../means/identity_means_layer_XX.npy` (+ metadata), `identity_family_means_*`, `axis_means_*`
- `results/geometry/.../probes/axis_probe_scores.csv`, `identity_within_axis_probe_scores.csv`
- `results/geometry/.../family_stability/family_cosines_layer_XX.csv`, `family_cosines_summary.csv`
- `results/geometry/.../contrasts/contrast_scores.csv` (in-sample), `contrast_family_holdout_scores.csv`
- `results/geometry/.../contrasts/contrast_projection_scores_layer_XX.csv` (only for `PROJECTION_LAYERS = {0, 8, 16, 24, 32}`)
- `results/geometry/.../run_config.json`

## Key implementation details
- **Token convention.** Inherits the final-non-pad-token activation from `extract_identity_activations.py`. Every templated prompt ends in `.`, so this is effectively the period token (see issue 1.1 — flagged in [Step 8](08_analyze_identity_geometry_diagnostics.md) and the audit).
- **PCA pipeline.** `run_pca` applies `StandardScaler().fit_transform(x_sample)` then `PCA(n_components=10)` on a stratified sample (`stratified_sample_indices` strata on `axis||family`). Both raw and probe-feature pipelines z-score before reducing.
- **Probe features.** `make_probe_features` fits `StandardScaler` + `PCA(n_components=256)` **once on the entire layer** before `crossval_probe`, which then runs `GroupKFold` (`n_splits=min(5, n_groups)`) on a `LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')`.
- **Probe tasks.** Axis prediction grouped by `template_id` and by `family`; identity-within-axis prediction grouped by `template_id`.
- **Contrast direction.** `contrast_direction(x_centered, mask_a, mask_b)` returns the unit-normalized centered difference of means `(mean_a − mean_b) / ||·||`. `evaluate_contrast_scores` scores every prompt by projection onto this direction, then computes ROC AUC and Cohen's d **on the same A/B prompts that defined it**.
- **Family holdout.** For each `heldout_family`, refits the direction on `~heldout` rows, evaluates AUC/d on the held-out rows only — written to `contrast_family_holdout_scores.csv`.
- **Hardcoded contrast list.** `CONTRASTS` at the top of the file enumerates 17 identity pairs. Pairs whose identities are absent from `metadata["identity_id"].unique()` hit `if identity_a not in identity_set: continue` and are silently dropped.

## Issues & Opportunities

### 2.1 [MAJOR] — Headline contrast AUC / Cohen's d are in-sample (FIX LANDED 2026-05-27)

**Status:** Commit `e15e62f`. The headline metric is now held-out; in-sample is preserved as a diagnostic so the over-fitting gap can still be quoted.

**What landed:**
- `CONTRAST_COLUMNS` renamed `auc_all` → `auc_in_sample`, `cohens_d_all` → `cohens_d_in_sample` (in-sample status now loud in the schema).
- `run_contrasts` writes the renamed columns to `contrast_scores.csv`.
- New `contrast_holdout_summary.csv` aggregates `contrast_family_holdout_scores.csv` into a headline row per (layer, contrast): `auc_mean`, `auc_sd`, `auc_min`, `auc_max`, `cohens_d_mean`, `cohens_d_sd`, `n_families`. Cite this in the methods doc, not `contrast_scores.csv`.
- [Step 10](10_plot_identity_geometry.md) — `plot_contrasts` now plots held-out AUC / Cohen's d as the headline `contrast_auc_by_layer.png` / `contrast_cohens_d_by_layer.png` (with "HEADLINE" in the title). The in-sample plots get an `_in_sample` suffix and "DIAGNOSTIC" in the title so anyone glancing knows they overstate separation.

**Original audit (preserved for context):** `run_contrasts` defined the contrast direction as `mean(A) − mean(B)` on **all** prompts of identities A and B, then evaluated `auc_all` / `cohens_d_all` by projecting **those same prompts** onto that direction. Difference-of-means is by construction the linear maximizer of mean separation between the two groups, so this number was optimistically biased. "Identity contrasts are linearly decodable with AUC ≈ X" was a tautology, not evidence.

### 2.2 [BLOCKER] — No null model for probes (PROBE NULL LANDED 2026-05-27; SVD null still pending)

**Status:** The probe permutation null landed; the shared-subspace SVD null (the other half of audit 2.2 in [Step 9](09_analyze_shared_social_subspace.md)) is still open.

**What landed in this script:**
- `_run_cv_folds` extracts the inner per-fold loop so the observed and null replicates share one implementation.
- `crossval_probe` accepts `n_permutations` and `null_rng_seed`. When `n_permutations > 0`, `y` is globally shuffled per replicate and the GroupKFold split structure is preserved across all replicates (audit-recommended).
- New CLI: `--n_permutations` (default `20`; bump to `>=100` for the headline number) and `--null_random_seed` (defaults to `--random_seed`).
- Output rows gain `null_n_permutations`, `null_accuracy_mean`, `null_accuracy_sd`, `null_macro_f1_mean`, `null_macro_f1_sd`, `accuracy_z`, `macro_f1_z`, `accuracy_p_value`, `macro_f1_p_value`. The p-value uses the Phipson-Smyth `(1 + n_above) / (1 + N)` smoothing so `p > 0` always.
- When `n_permutations == 0`, the null fields are written as `NaN` so downstream CSV readers can tell that no null was computed.

**Validation:** synthetic test cases confirm correct behavior (commit message documents these). Perfect-feature inputs hit p at the n-perm floor (1/(N+1)); noise inputs sit within the null with z ≈ 0 and high p; same seed reproduces the same null bit-for-bit.

**Original audit (preserved for context):** `crossval_probe` reported accuracy / macro-F1 against chance only via `n_classes` arithmetic. With group structure / template leakage present in the data, the observed accuracy could reflect non-identity structure; without a calibrated null, "identity is linearly decodable from layer-X activations" is not a defensible claim.

**Remaining work:** the shared-subspace SVD spectrum in [Step 9](09_analyze_shared_social_subspace.md) still lacks its own null (random-direction baseline + shuffled-identity SVD spectrum). That fix is a separate commit.

### 2.8 [MINOR] — Probe dimensionality reduction leaks across CV folds (VERIFIER LANDED 2026-05-27)

**Status:** The design is preserved (fit StandardScaler + PCA once globally; refitting per fold is computationally intractable across `n_folds × n_residualizations × n_probe_configs × n_layers`). A verifier is now in place to empirically vindicate the choice on a chosen layer.

**What landed:**
- `make_probe_features` docstring rewritten to explain the speed tradeoff, the technical leakage, and how to verify it on a chosen layer.
- New `crossval_probe_fold_internal_pca`: identical CV structure to `crossval_probe`, but `StandardScaler` and `PCA` are fit inside each fold on the train rows only.
- New CLI: `--verify_fold_internal_pca <layer_index>` (default off). When set, runs every probe configuration on that layer a second time through `crossval_probe_fold_internal_pca` and writes `probes/pca_leakage_verification.csv` with side-by-side `global_pca_*` and `fold_internal_pca_*` accuracy and macro-F1 means/SDs, plus `accuracy_delta` and `macro_f1_delta`. Synthetic tests confirm the verifier reproduces the global-PCA numbers when the design is sound.

**How to read the verification CSV:** if `|accuracy_delta|` is smaller than the per-fold `global_pca_accuracy_sd`, the fold-internal-PCA path produces statistically indistinguishable numbers and the global-PCA shortcut is defensible. A `|accuracy_delta|` of, say, +0.05 on a 0.04-SD probe would indicate real leakage; in that case the cost of fold-internal PCA becomes justifiable for the headline run even if not for development iterations.

**Original audit (preserved):** `make_probe_features` fits `StandardScaler` + `PCA(n_components=probe_pca_dim)` once on the entire layer before `crossval_probe` does `GroupKFold`. The unsupervised PCA basis is thus fit on data that include the held-out fold; a careful reviewer would flag it. PCA is unsupervised so the leakage is mild in principle, but unverified.

### 4.1 [MAJOR] — Contrast lists reference identities that do not exist (silently skipped)

**What's wrong:** `CONTRASTS` includes pairs whose identity IDs are not in `bbq_identity_normalized_forms.csv`. Concretely, `ses_low_income` (used in `ses_low_income_vs_ses_rich` and `ses_low_income_vs_ses_high_socioeconomic_status`) and `ses_high_socioeconomic_status` do not exist; the dataset's SES identities are `ses_low`, `ses_high`, `ses_poor`, `ses_rich`, `ses_lower_class`, etc. The loop runs `if identity_a not in identity_set or identity_b not in identity_set: continue` — these pairs are dropped with no warning.

**Why it matters:** The SES axis runs with fewer contrasts than the code implies. Anything that aggregates "all SES contrasts" understates coverage; rerunning after a dataset edit will silently change the count.

**Targeted fix:** (1) Replace the literal `CONTRASTS` list with references to a single validated contrast registry (per 5.10) keyed by IDs that exist in `bbq_identity_normalized_forms.csv`. (2) Make the skip loud — log a warning per missing identity and write a `contrasts_skipped.csv` listing skipped pairs. (3) Add a startup assertion that every contrast identity is present.

### 5.9 [MINOR] — PCA on StandardScaler-ed activations changes the geometry

**What's wrong:** `run_pca` and `make_probe_features` both apply `StandardScaler` (per-dim z-scoring) before PCA. Residual-stream dimensions carry meaningfully unequal scale (rogue / high-norm dimensions carry real signal); z-scoring upweights low-variance dimensions, so the resulting `explained_variance_ratio` describes standardized space, not activation space.

**Why it matters:** Defensible for visualization, but should be stated and ideally compared with centered-only PCA. The probe choice matters less (logistic regression is scale-tolerant) but should be consistent and explicit.

**Targeted fix:** Add a `--scaling {standardize, center_only}` flag; default to `center_only` for PCA, keep `standardize` available. Or write both variants and document the choice in `run_config.json`.

### 5.10 [MINOR] — Heavy code duplication across analysis scripts

**What's wrong:** `cohens_d`, `contrast_direction`, `evaluate_contrast_scores`, `CONTRASTS`, Okabe-Ito palettes, and `cosine` are reimplemented in this script and in `analyze_identity_geometry_diagnostics.py`, `analyze_shared_social_subspace.py`, `analyze_identity_sae_features.py`, `plot_identity_directional_visualizations.py`, and `plot_identity_directional_followups.py`. Each copy can drift independently.

**Why it matters:** Sign conventions, normalization, and contrast lists silently diverge between scripts — a class of "results differ between scripts" bug that is invisible until someone checks.

**Targeted fix:** Extract into `status_mi/common.py`: `cohens_d`, `contrast_direction`, `evaluate_contrast_scores`, `residualize`, `OKABE_ITO`, `save_fig`, and the validated contrast registry. Import from there in every analysis/plot script.

## Rebuild checklist
- [ ] Move `CONTRASTS` into a shared, validated registry (`status_mi/common.py`) and import it here; raise a hard error if any referenced identity is missing from `bbq_identity_normalized_forms.csv`.
- [ ] Demote `auc_all` / `cohens_d_all` to a `*_in_sample` suffix and document it; rewire downstream plotting (Steps 10–12) to read the family-holdout CSV by default.
- [ ] Add a label-permutation null to `crossval_probe` (configurable `--n_permutations`, default 100) and write `accuracy_null_mean/sd`, `z_score`, `p_value` columns to both probe CSVs.
- [ ] Fit `StandardScaler` + `PCA` inside each CV fold in `make_probe_features` (or pre-register the leakage and add a one-layer sensitivity check).
- [ ] Add `--scaling {standardize, center_only}` and record it in `run_config.json`.
- [ ] Extract shared helpers (`cohens_d`, `contrast_direction`, `evaluate_contrast_scores`) into `status_mi/common.py`.

## Notes from the doc audit
- The "skip" branch for missing identities (`if identity_a not in identity_set or identity_b not in identity_set: continue`, line ~609) has no logging at all — not even a `print`. Easiest fix is to wrap it in a `logger.warning` and append to a `skipped_contrasts.csv` per layer.
- `PROJECTION_LAYERS = {0, 8, 16, 24, 32}` is hardcoded — should be a flag if other layers ever become relevant for per-prompt projection scatterplots.
