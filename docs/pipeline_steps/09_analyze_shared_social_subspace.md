# Step 9 — `analyze_shared_social_subspace.py`

**Stage:** 2 — Identity-geometry analyses (shared-subspace decomposition)
**Runs after:** `extract_identity_activations.py` (Stage 1); independently of [Step 7](07_analyze_identity_geometry.md) and [Step 8](08_analyze_identity_geometry_diagnostics.md), although it relies on the same residualization machinery.
**Feeds into:** `triage_sae_identity_features.py` (consumes `shared_pc_loading_score`), shared-subspace figures, [Step 12](12_plot_identity_directional_followups.md) (for direction-stability comparisons).

## Purpose
Decomposes the set of identity contrast directions into a **shared social subspace** (top SVD components) plus contrast-specific residuals. For each (layer, residualization), stacks unit-normalized centered difference-of-means directions for ~19 identity contrasts into a matrix `D ∈ ℝ^(C×hidden_dim)`, runs SVD `D = UΣVᵀ`, treats the top-k right singular vectors as the shared basis, then decomposes each contrast direction into `shared = Vₖᵀ(Vₖ d)` and `residual = d − shared`, and re-evaluates separation metrics for both components.

## Inputs
- `results/activations/llama-3.1-8b/identity_prompts_final_token/layer_XX.npy`
- `results/activations/.../identity_prompts_final_token/metadata.csv`

## Outputs (under `results/geometry/.../shared_subspace_decomposition/`)
- `metrics/shared_subspace_spectrum.csv` — singular values + per-component explained-variance ratios per (layer, residualization)
- `metrics/decomposition_metrics.csv` — per (layer, residualization, contrast, k): `shared_norm`, `residual_norm`, `cosine_with_full`, plus AUC / Cohen's d / midpoint accuracy of each component evaluated on the endpoint identities
- `metrics/axis_sharedness_summary.csv` — per-axis aggregates (mean fraction shared, mean full/shared/residual AUC and d)
- `metrics/shared_pc_identity_rankings.csv`, `metrics/shared_pc_top_bottom.csv` — projection of identity centroids onto each PC; top/bottom identities per PC
- `metrics/contrast_pc_loadings.csv` — each contrast's loading on each PC
- `metrics/cross_axis_projection_summary.csv`, `metrics/cross_axis_identity_projections.csv` — every identity centroid projected onto each contrast direction (cross-axis sweep)
- `figures/{spectrum, decomposition, axis_summary, pc_interpretation, pc_loadings, cross_axis, paper_panels}/...`

## Key implementation details
- **Direction construction.** For each `(identity_a, identity_b)` in `DEFAULT_CONTRASTS`, compute centered `mean(A) − mean(B)`, unit-normalize, sign-flip so `identity_a` has the larger projection. This is the same construction as [Step 7](07_analyze_identity_geometry.md), just stacked.
- **SVD on a thin matrix.** `D` has shape `(C, 4096)` with `C ≤ 21`; the singular spectrum lives in `min(C, 4096) = C` dimensions. PCs are inherently low-dimensional and *any* set of unit vectors will have *some* spectrum — this is the heart of issue 2.2.
- **Decomposition.** For each contrast direction `d` and each `k`, project onto `span(Vₖ)`: `shared = Vₖᵀ Vₖ d`, `residual = d − shared`. `fraction_norm = ||shared||² / ||d||²`. Cosine with full direction reported. The component is then evaluated on endpoint prompts via `evaluate_component` (projecting endpoint activations onto the component direction, taking AUC/d on identity_a vs identity_b prompts).
- **PC interpretation.** Each PC is projected against identity centroids; `shared_pc_top_bottom.csv` lists the most positive / negative identities per PC. This is the input that lets `triage_sae_identity_features.py` build a per-feature `shared_pc_loading_score`.
- **Cross-axis projection.** Each identity centroid (across all axes) is projected onto each contrast direction. `cross_axis_projection_summary.csv` summarizes per (contrast, projected_axis): mean / sd / range of projections, plus max/min identity within that axis on that contrast.
- **Residualization sweep.** `DEFAULT_RESIDUALIZATIONS = ["raw", "family_residualized", "template_residualized", "required_form_residualized"]`; each (layer, residualization) gets its own decomposition.

## Issues & Opportunities

### 2.1 [MAJOR] — Shared / residual components evaluated in-sample (FIX LANDED 2026-05-27)

**Status:** Commit `51aa571`. Genuinely held-out at every step: direction estimation, SVD basis, and evaluation.

**What landed:**
- `DECOMPOSITION_COLUMNS` split into in-sample (suffixed `_in_sample`) and held-out schemas; `_suffix_metric_keys` helper drives the rename so the same `evaluate_component` function backs both paths.
- New `decomposition_rows_holdout`: for each held-out template family `f`, re-derives every contrast direction on the non-`f` rows, re-SVDs the basis from those held-out-trained directions, decomposes each direction onto the held-out basis, and evaluates shared / residual components on the `f` rows. Writes `decomposition_metrics_holdout.csv` (one row per fold).
- New `write_decomposition_holdout_summary`: aggregates per-fold rows into one row per (contrast, k, component_type): `auc_mean`, `auc_sd`, `cohens_d_mean`, `cohens_d_sd`, `n_folds`. Written to `decomposition_metrics_holdout_summary.csv`. This is the audit-recommended headline; cite it in the methods doc, not the in-sample `decomposition_metrics.csv`.
- `aggregate_axis_sharedness` output columns are renamed `mean_*_auc → mean_*_auc_in_sample` (same for `d`) so the per-axis summary's in-sample status is unambiguous. `plot_axis_summary` title and filename now carry "DIAGNOSTIC (in-sample)".
- Paper-panel plot in `plot_decomposition_paper_panel` reads `auc_in_sample` (the renamed column) for the in-sample quad.

**Original audit (preserved):** `evaluate_component` projected the endpoint A/B prompts (the *same* prompts whose means defined `d`) onto each component (shared or residual) and reported AUC/d. Because `d` was built from these prompts and the shared/residual components are linear functions of `d` and a fixed basis, the separation metrics were conditioned on the data they evaluated. The shared-subspace paper-worthy claim ("a low-rank subspace recovers most of the identity contrast across axes") is now a defensible generalization claim from the held-out summary.

### 2.2 [BLOCKER] — No null model for the SVD spectrum / "shared subspace" claim (FIX LANDED 2026-05-27)

**Status:** Closed in commit `c4071cd`. Audit 2.2 is now FULLY landed (probe null + SVD null).

**What landed:**

Two null methods both run per (layer, residualization), gated behind `--n_nulls_svd` (default 200):

1. **`null_directions_shuffle_identities`** — for each (identity_a, identity_b) contrast, randomly permute prompts BETWEEN the two endpoint identities (preserving marginal n_a / n_b). Compute the diff-of-means direction on the shuffled labels. Tests H0 = "the difference between identity_a's and identity_b's prompts reflects no identity-specific structure beyond random relabeling."
2. **`null_directions_random_half_split`** — for each contrast, take ALL prompts on that axis (not just the two endpoint identities) and randomly split into two halves whose sizes match the original n_a / n_b. Stronger null. Tests H0 = "the contrast direction reflects no axis-specific structure beyond a random 50/50 split of the axis."

For each null replicate: build directions, stack, SVD; record per-PC singular values + **participation ratio** + **top-k explained variance** (both audit-recommended concentration metrics).

**New CSVs:**
- `metrics/shared_subspace_spectrum_null_summary.csv` — per (layer, residualization, null_method, component): observed_singular_value, null_mean, null_sd, null_p5, null_p50, null_p95, n_null_replicates, `observed_exceeds_p95`. **PCs whose observed value exceeds null p95 are the audit-defensible "shared" components.**
- `metrics/shared_subspace_concentration_null.csv` — per (layer, residualization, null_method): observed_participation_ratio, observed_top_k_variance, the matching null distribution stats, and `observed_pr_more_concentrated_than_p5` and `observed_top_k_exceeds_p95` flags.
- `metrics/shared_subspace_spectrum_null_replicates.csv` (optional, gated by `--save_null_svd_replicates`): per-replicate, per-component singular values for downstream plotting.

**New CLI:** `--n_nulls_svd` (default 200), `--null_svd_random_seed` (defaults to `--random_seed`), `--null_svd_top_k` (default 5), `--save_null_svd_replicates`.

**Decision rule for the paper:** report "the top-k singular components are shared" where k is the largest index for which `observed_exceeds_p95` is true under both null methods. Cite the concentration metrics (participation ratio, top-k variance) alongside.

**Synthetic validation (in commit message):**
- Strong rank-1 shared subspace: observed PC1 sigma 2.21 vs null p95 1.63 → EXCEEDS_P95 (correctly flagged); PR 1.05 vs null p5 ~2.88 → MORE_CONCENTRATED.
- No shared structure (each axis along its own random direction): observed PC1 sigma 1.15 vs null p95 1.22-1.24 → below_p95 (correctly NOT flagged); PR 4.76 vs null p5 ~4.55 → NOT_MORE_CONCENTRATED.

The null discriminates signal from noise.

**Original audit (preserved):** The reported singular spectrum and PC structure were not compared with any null. Any set of ~19 unit vectors in 4096-d has *some* concentration in its leading SVD components; the question is whether the observed concentration exceeds chance.

### 4.1 [MAJOR] — `DEFAULT_CONTRASTS` references identities that do not exist (FIX LANDED 2026-05-27)

**Status:** `DEFAULT_CONTRASTS` is now imported from `scripts/contrast_registry.py` (commit `6eafb4d`); the typos are fixed in the registry so SES has 4 valid contrasts. `KEY_CONTRASTS` and `SELECTED_CROSS_AXIS_ORDERINGS` also imported from the registry. `load_contrasts(path, metadata, output_dir=...)` writes `metrics/contrasts_skipped.csv` with per-row reasons and emits warnings per skipped pair. **No startup assertion** — partial-axis runs work.

**Original audit (preserved):** `DEFAULT_CONTRASTS` included `ses_low_income_vs_ses_rich` and `ses_low_income_vs_ses_high_socioeconomic_status` — neither `ses_low_income` nor `ses_high_socioeconomic_status` was present in `bbq_identity_normalized_forms.csv`. Pairs whose identities were missing were filtered out before SVD, silently reducing `C`. This changed the dimensionality of the shared subspace as a function of which contrasts happened to exist in the data; the realized `C` was never reported.

**Remaining tightening (optional):** Record the realized `C` (post-validation) in `run_config.json` and in each spectrum row so paper claims about the "rank" of the shared subspace cite a specific, reproducible number.

### 5.10 [MINOR] — Heavy code duplication across analysis scripts (FIX LANDED 2026-05-27)

**Status:** All shared helpers — `cohens_d`, `cosine`, `normalize`, `compute_direction`, `evaluate_projection`, `residualize`, `OKABE_ITO`, `save_fig`, `CenterOnlyScaler` + `make_scaler` — now live in `scripts/common.py` (commit `e50bbd1`). The canonical contrast list lives in `scripts/contrast_registry.py` (commit `1e242c9`; audit 4.1). This script's local copies are gone; any remaining definitions are thin adapter wrappers that preserve the prior return-tuple shapes while routing through `common.py`. Net change across the 8 consumer scripts: 358 lines added to `common.py`, 369 lines removed elsewhere.

## Rebuild checklist
- [ ] Source `DEFAULT_CONTRASTS` from the shared registry; assert every identity exists at startup.
- [ ] Add a null-spectrum generator: shuffled-identity SVD spectra (≥100 reps); write `shared_subspace_spectrum_null.csv` and a per-PC `p_value` / `z_score` column on the observed spectrum.
- [ ] Add a held-out split (cross-family) for the decomposition: re-fit `d` and `V` on train families, evaluate components on the held-out family; write `decomposition_metrics_heldout.csv`.
- [ ] Record the realized contrast count `C` in `run_config.json` and in each output CSV.
- [ ] Import `residualize`, `cohens_d`, `compute_direction`, `OKABE_ITO` from `status_mi/common.py`.

## Notes from the doc audit
- The `evaluate_component` AUC for "residual" components is effectively asking "how much identity-axis separation is left after stripping the shared subspace" — that comparison is the most paper-worthy output of this script and should be the headline plot once held-out evaluation is in place.
- The `KEY_CONTRASTS` and `SELECTED_CROSS_AXIS_ORDERINGS` lists at the top of the file hardcode which contrasts get spotlighted in the "paper panel" figures; flag this in the rebuild so the registry edit propagates to these too.
