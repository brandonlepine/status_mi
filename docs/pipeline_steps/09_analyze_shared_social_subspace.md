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

### 2.2 [BLOCKER] — No null model for the SVD spectrum / "shared subspace" claim (STILL OPEN)

**Status:** The probe-side half of audit 2.2 landed in commits touching [Step 7](07_analyze_identity_geometry.md) and [Step 8](08_analyze_identity_geometry_diagnostics.md). This SVD-side half is still pending.

**What's wrong:** The reported singular spectrum and PC structure are not compared with any null. Any set of ~19 unit vectors in 4096-d has *some* concentration in its leading SVD components; the question is whether the observed concentration exceeds chance. The script writes `shared_subspace_spectrum.csv` without a matched-null spectrum.

**Why it matters:** The "shared social subspace" claim is unsupported until a null exists. This is the central conceptual risk for this analysis.

**Targeted fix:** Build a null distribution of SVD spectra by (a) shuffling identity assignments within each axis and re-deriving the contrast directions, or (b) splitting each axis's prompts randomly into two halves and computing differences of half-means. Run ≥100 nulls. Report concentration metrics (participation ratio, variance in top-k) against the null. Only PCs whose singular value exceeds, e.g., the 95th percentile of the null are reported as "shared."

### 4.1 [MAJOR] — `DEFAULT_CONTRASTS` references identities that do not exist (silently skipped)

**What's wrong:** `DEFAULT_CONTRASTS` includes `ses_low_income_vs_ses_rich` and `ses_low_income_vs_ses_high_socioeconomic_status` — neither `ses_low_income` nor `ses_high_socioeconomic_status` is present in `bbq_identity_normalized_forms.csv`. Pairs whose identities are missing are filtered out before SVD, silently reducing `C`. This changes the dimensionality of the shared subspace as a function of which contrasts happen to exist in the data.

**Why it matters:** The "rank" and "spectrum" of the shared subspace depend on `C`. Silent dropping of contrasts changes the reported spectrum without flagging it.

**Targeted fix:** Source `DEFAULT_CONTRASTS` from the shared, validated registry; fail loudly if any contrast identity is missing. Record the realized `C` (post-validation) in `run_config.json` and in each spectrum row.

### 5.10 [MINOR] — Heavy code duplication across analysis scripts

**What's wrong:** Independent copies of `cohens_d`, contrast lists, sign-flip convention, `residualize`, `compute_direction`, palettes. The `DEFAULT_CONTRASTS` here uses a different schema (`(name, identity_a, identity_b, axis)` 4-tuples) than [Step 7](07_analyze_identity_geometry.md)'s 2-tuples.

**Why it matters:** Two parallel contrast registries with different schemas guarantee drift.

**Targeted fix:** One registry in `status_mi/common.py` that exposes both views (`as_pairs()` and `as_named()`). One implementation of `residualize` and one of `compute_direction` (with documented sign convention).

## Rebuild checklist
- [ ] Source `DEFAULT_CONTRASTS` from the shared registry; assert every identity exists at startup.
- [ ] Add a null-spectrum generator: shuffled-identity SVD spectra (≥100 reps); write `shared_subspace_spectrum_null.csv` and a per-PC `p_value` / `z_score` column on the observed spectrum.
- [ ] Add a held-out split (cross-family) for the decomposition: re-fit `d` and `V` on train families, evaluate components on the held-out family; write `decomposition_metrics_heldout.csv`.
- [ ] Record the realized contrast count `C` in `run_config.json` and in each output CSV.
- [ ] Import `residualize`, `cohens_d`, `compute_direction`, `OKABE_ITO` from `status_mi/common.py`.

## Notes from the doc audit
- The `evaluate_component` AUC for "residual" components is effectively asking "how much identity-axis separation is left after stripping the shared subspace" — that comparison is the most paper-worthy output of this script and should be the headline plot once held-out evaluation is in place.
- The `KEY_CONTRASTS` and `SELECTED_CROSS_AXIS_ORDERINGS` lists at the top of the file hardcode which contrasts get spotlighted in the "paper panel" figures; flag this in the rebuild so the registry edit propagates to these too.
