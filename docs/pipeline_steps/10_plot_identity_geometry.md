# Step 10 — `plot_identity_geometry.py`

**Stage:** 2 — Identity-geometry analyses (plotting)
**Runs after:** [Step 7](07_analyze_identity_geometry.md)
**Feeds into:** Figures only (no downstream analysis script consumes these PNG/PDFs).

## Purpose
Pure plotting layer over the first-pass geometry CSVs. Reads `pca/`, `probes/`, `family_stability/`, and `contrasts/` outputs from [Step 7](07_analyze_identity_geometry.md) and produces PCA scatters per layer, axis-centroid scatters, probe-score line plots, family-stability heatmaps, contrast-AUC-by-layer curves, and (optionally) UMAPs. No new metrics — what is plotted is what was already computed.

## Inputs
- `results/geometry/llama-3.1-8b/identity_prompts_final_token/pca/pca_layer_XX.csv`, `pca_explained_variance.csv`
- `results/geometry/.../probes/axis_probe_scores.csv`, `identity_within_axis_probe_scores.csv`
- `results/geometry/.../family_stability/family_cosines_layer_XX.csv`, `family_cosines_summary.csv`
- `results/geometry/.../contrasts/contrast_scores.csv` (in-sample `auc_all` / `cohens_d_all`), `contrast_family_holdout_scores.csv`, `contrast_projection_scores_layer_XX.csv`

## Outputs
- `results/geometry/.../figures/...` — PCA scatters (`pca_by_identity/`), probe-score curves, family-stability heatmaps, contrast-AUC-by-layer, optional UMAP scatters
- Static matplotlib + optional seaborn / umap-learn

## Key implementation details
- **Pure consumer.** Does not load activations; only reads CSVs/`npy` summaries produced by [Step 7](07_analyze_identity_geometry.md).
- **Plot defaults.** `--layers` defaults to `"0,8,16,24,32"`; `--max_points_per_plot 15000`. Okabe-Ito palette and the same marker list (`o, s, ^, D, P, X, v, <, >, *, h, 8`) and linestyles as the analysis scripts.
- **UMAP.** Optional, gated by `--make_umap` and presence of `umap-learn`. Default neighbors=30, min_dist=0.1, metric=`cosine`.
- **Headline contrast plot.** The contrast-AUC-by-layer figure (the one that gets used as the geometry headline) is read from `contrast_scores.csv`, i.e. the in-sample `auc_all` — this is the downstream-of-2.1 visualization issue (see below).

## Issues & Opportunities

### 2.1 (visualization consequence) [MAJOR] — In-sample contrast AUC is plotted as the headline

**What's wrong:** This script's contrast-AUC-by-layer plot is derived from `contrast_scores.csv` (the `auc_all` / `cohens_d_all` columns from [Step 7](07_analyze_identity_geometry.md)). Those values are computed on the same prompts used to define the direction; the chart is therefore a chart of an in-sample, optimistically biased number. [Step 7](07_analyze_identity_geometry.md)'s `contrast_family_holdout_scores.csv` already contains the honest cross-family number — it is simply not plotted as the headline.

**Why it matters:** "Identity contrasts are linearly decodable from layer X with AUC ≈ Y" is the visual that anchors the geometry results. If `Y` is in-sample, the chart overstates the finding. Switching the plot source is a low-cost, high-leverage fix.

**Targeted fix:** Replace the contrast-AUC plot's source with `contrast_family_holdout_scores.csv` (group by `(layer, contrast_name)`, take mean / median over `heldout_family`). Keep the in-sample version available as a clearly labeled diagnostic figure (`contrast_auc_by_layer_in_sample.png`). Apply the same change in [Step 11](11_plot_identity_directional_visualizations.md) and [Step 12](12_plot_identity_directional_followups.md).

### 5.10 [MINOR] — Heavy code duplication across analysis scripts (FIX LANDED 2026-05-27)

**Status:** All shared helpers — `cohens_d`, `cosine`, `normalize`, `compute_direction`, `evaluate_projection`, `residualize`, `OKABE_ITO`, `save_fig`, `CenterOnlyScaler` + `make_scaler` — now live in `scripts/common.py` (commit `e50bbd1`). The canonical contrast list lives in `scripts/contrast_registry.py` (commit `1e242c9`; audit 4.1). This script's local copies are gone; any remaining definitions are thin adapter wrappers that preserve the prior return-tuple shapes while routing through `common.py`. Net change across the 8 consumer scripts: 358 lines added to `common.py`, 369 lines removed elsewhere.

## Rebuild checklist
- [ ] Switch the contrast-AUC-by-layer figure to read `contrast_family_holdout_scores.csv` by default; rename the in-sample figure with an `_in_sample` suffix.
- [ ] Replace the local `OKABE_ITO` / `MARKERS` / `LINESTYLES` / `AXES_TO_PLOT` / `safe_read_csv` definitions with imports from the shared module.
- [ ] Confirm `pca_by_identity/` scatters are drawn from the appropriate residualization (currently they use the raw-pipeline PCA from [Step 7](07_analyze_identity_geometry.md); the residualized PCA scatters live under [Step 8](08_analyze_identity_geometry_diagnostics.md)'s `diagnostics/pca_residualized/...` and should be linked to from the index page if one exists).
- [ ] Add a `--use_holdout_for_contrast_plot/--use_in_sample_for_contrast_plot` flag so the choice is auditable from `run_config` / cmdline history.

## Notes from the doc audit
- This script does not write a `run_config.json` of its own — only the analysis scripts do. The flags used to produce a given figure are not recorded next to the figure. Add one for reproducibility.
- The output dir defaults to `geometry/.../figures` which is inside the same tree as [Step 7](07_analyze_identity_geometry.md)'s CSV outputs; if you ever delete `geometry/.../` to rerun analysis, figures vanish too. Consider a sibling `geometry_figures/` directory.
