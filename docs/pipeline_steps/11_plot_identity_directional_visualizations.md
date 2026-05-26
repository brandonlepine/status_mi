# Step 11 — `plot_identity_directional_visualizations.py`

**Stage:** 2 — Identity-geometry analyses (directional plotting)
**Runs after:** `extract_identity_activations.py` (Stage 1). Notably, **does not consume** [Step 7](07_analyze_identity_geometry.md)'s contrast CSVs — it re-loads activations and re-computes its own contrast directions.
**Feeds into:** Figures only; also writes its own `metrics/` CSVs that [Step 12](12_plot_identity_directional_followups.md) and the paper-panel composites may reference.

## Purpose
Theory-driven visualization of identity contrast representations: per-layer per-contrast projection distributions, family-to-family generalization heatmaps, direction-cosine summaries across layers, centroid-distance scatters, and 2-D plane scatters built from two orthogonalized contrast directions. Re-derives directions independently (rather than reading them from [Step 7](07_analyze_identity_geometry.md)), so the contrast list, residualization pipeline, and sign convention all live in this file.

## Inputs
- `results/activations/llama-3.1-8b/identity_prompts_final_token/layer_XX.npy` (re-loads activations)
- `results/activations/.../identity_prompts_final_token/metadata.csv`

## Outputs (under `results/geometry/.../directional_visualizations/`)
- `metrics/layerwise_contrast_metrics.csv` — per (layer, residualization, contrast): full-data AUC/d, family-holdout AUC/d
- `metrics/family_to_family_generalization.csv` — train direction on family A, evaluate on family B
- `metrics/centroid_ordering.csv` — every same-axis identity's projection onto a focal contrast direction with bootstrap CIs
- `direction_cosines/*.csv` — inter-contrast and inter-layer direction cosines
- Figures: projection-distribution plots, layer-AUC curves, direction-cosine heatmaps, centroid-distance scatters, 2-D plane scatters (e.g. `sexual_orientation` plane = `gay − straight` × orthogonalized `bisexual − straight`).

## Key implementation details
- **Re-implements direction computation.** `compute_direction`, `cohens_d`, `residualize`, and contrast-evaluation helpers are local copies — independent of [Step 7](07_analyze_identity_geometry.md) and [Step 8](08_analyze_identity_geometry_diagnostics.md). The 4-tuple `DEFAULT_CONTRASTS = [(name, identity_a, identity_b, axis), ...]` schema mirrors [Step 9](09_analyze_shared_social_subspace.md) but is a separate copy.
- **PLANE_SPECS.** `PLANE_SPECS` hardcodes which two contrasts define the 2-D plane scatter per axis (e.g. `race_ethnicity: [(Black − White), (Asian − White)]`, `sexual_orientation: [(gay − straight), (bisexual − straight)]`). The second direction is Gram-Schmidt orthogonalized against the first.
- **Residualization sweep.** `DEFAULT_RESIDUALIZATIONS` matches [Step 8](08_analyze_identity_geometry_diagnostics.md): `raw, family_residualized, template_residualized, required_form_residualized`.
- **Per-prompt projection scatters.** Each prompt is scored by projection onto the contrast direction; scatters show by-identity distributions, optionally split by family.

## Issues & Opportunities

### 2.1 (visualization consequence) [MAJOR] — In-sample AUC plotted as the headline contrast curve

**What's wrong:** `metrics/layerwise_contrast_metrics.csv` contains both the full-data and family-holdout AUC; the layer-curve figure here uses the full-data column by default. The full-data number is computed on the same prompts that defined the direction — identical to the bias described in [Step 7](07_analyze_identity_geometry.md) issue 2.1.

**Why it matters:** This script's layer-AUC plot is one of the geometry-pipeline's headline visuals. Plotting in-sample AUC overstates separability.

**Targeted fix:** Plot the family-holdout AUC by default (use the `family_to_family_generalization.csv` aggregates if needed). Demote the in-sample curve to a diagnostic. Mirror the fix in [Step 10](10_plot_identity_geometry.md) and [Step 12](12_plot_identity_directional_followups.md).

### 5.10 [MAJOR severity-by-impact, listed as MINOR in source] — Heavy code duplication; this script *re-computes* contrast directions independently

**What's wrong:** This script reimplements `compute_direction`, `cohens_d`, the residualization map, the contrast list (in a different schema than [Step 7](07_analyze_identity_geometry.md)), and the family-holdout evaluator. Because the re-implementation is independent, any drift in sign convention, normalization, or residualization between this script and [Step 7](07_analyze_identity_geometry.md)/[Step 8](08_analyze_identity_geometry_diagnostics.md) is invisible until someone compares numbers across CSVs.

**Why it matters:** When a reviewer asks "why does the AUC for `race_black_vs_race_white` at layer 24 in `geometry/figures/contrast_auc_by_layer.png` differ from `directional_visualizations/figures/layer_auc_curve.png`?", the answer should be "different evaluation set," not "different sign convention." The duplication makes this almost impossible to audit.

**Targeted fix:** Extract `compute_direction`, `residualize`, `cohens_d`, `evaluate_contrast_scores`, and the contrast registry into `status_mi/common.py`. This script should call the same `compute_direction(x, metadata, identity_a, identity_b, residualization=...)` as [Step 7](07_analyze_identity_geometry.md) and [Step 9](09_analyze_shared_social_subspace.md). Verify on one (layer, contrast) that the three scripts produce identical AUCs.

## Rebuild checklist
- [ ] Replace local `compute_direction`, `cohens_d`, `residualize`, `DEFAULT_CONTRASTS`, `OKABE_ITO`, `MARKERS`, `LINESTYLES`, palette helpers with imports from `status_mi/common.py`.
- [ ] Switch the layer-AUC curve to read family-holdout AUC by default; keep in-sample as a diagnostic variant with an explicit suffix.
- [ ] Add a regression test (one layer, one contrast) that compares this script's `compute_direction` output to [Step 7](07_analyze_identity_geometry.md)'s, after the shared-module refactor.
- [ ] Surface `PLANE_SPECS` in `run_config.json` so 2-D plane figures are reproducible after edits to the spec.
- [ ] Confirm the contrasts listed in `PLANE_SPECS` all exist in the validated contrast registry (e.g. `gender_woman` / `gender_man` should be present; see the `gender_identity` plane spec).

## Notes from the doc audit
- The `PLANE_SPECS["gender_identity"]` entry uses `gender_woman` and `gender_man`. These identity IDs should be audited against `bbq_identity_normalized_forms.csv` before the contrast registry refactor — they may or may not exist (the dataset uses `gender_woman`/`gender_man` for gender identity, but the workflow doc emphasizes `gender_transgender*` / `gender_cisgender*` as the main contrasts). If `gender_woman`/`gender_man` are missing, this plane silently degrades.
- `metrics/family_to_family_generalization.csv` already contains the held-out evidence the project needs for the "identity geometry generalizes" claim — it just needs to be promoted into the headline visualization.
