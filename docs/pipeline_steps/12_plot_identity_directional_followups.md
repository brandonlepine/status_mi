# Step 12 — `plot_identity_directional_followups.py`

**Stage:** 2 — Identity-geometry analyses (followup plotting / paper-summary panels)
**Runs after:** `extract_identity_activations.py` (Stage 1); often after [Step 11](11_plot_identity_directional_visualizations.md), though it does not strictly depend on that script's outputs (it re-derives directions itself).
**Feeds into:** Figures only; produces the 6-panel "paper-ready summary panel" composite that is the most likely candidate for a paper figure.

## Purpose
A still-larger followup-plotting script that re-derives contrast directions and produces five focused visualization classes plus a paper-summary panel:
- centroid-ordering plots (every same-axis identity on a focal contrast direction, with bootstrap CIs);
- family-to-family generalization heatmaps;
- direction-stability curves (cosine between adjacent layers, and vs a reference layer, default 24);
- 2-D plane scatters using two orthogonalized contrast directions;
- a 6-panel paper summary combining variance decomposition, key-contrast AUC, two axis planes, centroid ordering, and direction-cosine heatmap.

## Inputs
- `results/activations/llama-3.1-8b/identity_prompts_final_token/layer_XX.npy`
- `results/activations/.../identity_prompts_final_token/metadata.csv`
- Variance decomposition from [Step 8](08_analyze_identity_geometry_diagnostics.md) (`diagnostics/variance_decomposition.csv`) for the paper-panel's left-most subplot

## Outputs (under `results/geometry/.../directional_followups/`)
- Figures: centroid-ordering panels per `CENTROID_ORDERING_CONTRASTS`, family-to-family heatmaps per `KEY_CONTRASTS`, direction-stability line charts (`adjacent_layer_cosine`, `cosine_vs_layer_24`), 2-D plane scatters, paper summary panel
- May write its own `metrics/` CSVs for the per-contrast paper-panel inputs

## Key implementation details
- **Re-derives directions.** Like [Step 11](11_plot_identity_directional_visualizations.md), this script reimplements `compute_direction`, `cohens_d`, `residualize`, and contrast-evaluation helpers. The `DEFAULT_CONTRASTS` 4-tuple list is a third independent copy.
- **Featured contrast lists.**
  - `KEY_CONTRASTS` (6 entries) drives the headline curves: `sexuality_gay_vs_sexuality_straight`, `race_black_vs_race_white`, `gender_transgender_vs_gender_cisgender`, `appearance_obese_vs_appearance_thin`, `ses_low_income_vs_ses_rich`, `disability_disabled_vs_disability_able_bodied`.
  - `RESIDUALIZATION_COMPARISON_CONTRASTS` (5 entries) drives the per-residualization small multiples.
  - `CENTROID_ORDERING_CONTRASTS` (11 entries) drives the per-axis centroid-ordering panels.
- **Direction stability.** Cosine of the same contrast direction between adjacent layers and vs the layer-24 direction. Useful for picking the intervention layer.
- **Paper summary panel.** 6-panel composite — variance decomposition (top-left), key-contrast AUC by layer (top-right), two axis-plane scatters (middle row), centroid ordering (bottom-left), direction-cosine heatmap (bottom-right).
- **Residualization sweep.** Same four-residualization grid as upstream.

## Issues & Opportunities

### 2.1 (visualization consequence) [MAJOR] — Key contrast AUC by layer (PARTIAL FIX LANDED 2026-05-27)

**Status:** The headline AUC plot is now sourced from held-out CSV. The centroid-ordering CIs are NOT yet bootstrapped against held-out direction estimation — that's the remaining piece.

**What landed:**
- `plot_layerwise_figures` now reads `family_to_family_generalization.csv` (filtered to `train_family LIKE "all_except_*"` — the leave-one-family-out rows) and aggregates per `(layer, contrast_name, residualization)` as both `mean` and `median` across held-out families.
- New headline plots: `key_contrasts_auc_by_layer_residualization_comparison.png` (mean) and `..._median.png`. Titles include "HEADLINE — across held-out template families".
- In-sample variants moved to `..._in_sample.png` with "DIAGNOSTIC" in the title and a note that the metric overstates separation.
- Cohen's d variant: same swap, gated on `cohens_d` column being present in `family_to_family_generalization.csv`.

**Remaining work (smaller follow-up):** centroid-ordering CIs are still bootstrapped over the full-data direction. The honest version bootstraps over (a) family-held-out direction estimation, and (b) the prompts being projected. Substantial change to `centroid_ordering_rows` and the corresponding plotter; deferred until you say.

**Original audit (preserved):** The paper-summary panel's "key contrast AUC by layer" subplot was derived from the full-data direction (`compute_direction` on all prompts) projected onto all prompts. The cross-family / cross-template held-out variant existed upstream but was not the default source for the paper panel.

### 4.1 [MAJOR] — `DEFAULT_CONTRASTS` references identities that do not exist (FIX LANDED 2026-05-27)

**Status:** `DEFAULT_CONTRASTS` and `KEY_CONTRASTS` now import from `scripts/contrast_registry.py`. The `ses_low_income_vs_ses_rich` references in `RESIDUALIZATION_COMPARISON_CONTRASTS` and `CENTROID_ORDERING_CONTRASTS` are rewritten to `ses_low_vs_ses_rich` to match the registry's typo-fixed contrast names. The SES row of the paper-summary panel now points at a contrast that actually exists in the upstream analysis CSVs.

**Original audit (preserved):** `DEFAULT_CONTRASTS` listed `ses_low_income_vs_ses_rich` and `ses_low_income_vs_ses_high_socioeconomic_status`. `KEY_CONTRASTS` and `CENTROID_ORDERING_CONTRASTS` *both* included `ses_low_income_vs_ses_rich`. Because `ses_low_income` was not in the dataset, the SES row of the paper-panel was missing or visually inconsistent.

### 5.10 [MINOR] — Heavy code duplication across analysis scripts (FIX LANDED 2026-05-27)

**Status:** All shared helpers — `cohens_d`, `cosine`, `normalize`, `compute_direction`, `evaluate_projection`, `residualize`, `OKABE_ITO`, `save_fig`, `CenterOnlyScaler` + `make_scaler` — now live in `scripts/common.py` (commit `e50bbd1`). The canonical contrast list lives in `scripts/contrast_registry.py` (commit `1e242c9`; audit 4.1). This script's local copies are gone; any remaining definitions are thin adapter wrappers that preserve the prior return-tuple shapes while routing through `common.py`. Net change across the 8 consumer scripts: 358 lines added to `common.py`, 369 lines removed elsewhere.

## Rebuild checklist
- [ ] Import `compute_direction`, `cohens_d`, `residualize`, `evaluate_contrast_scores`, palette, markers, contrast registry from `status_mi/common.py`.
- [ ] Switch the paper-summary AUC panel to cross-family held-out AUC; keep in-sample available as a diagnostic.
- [ ] Audit `KEY_CONTRASTS`, `RESIDUALIZATION_COMPARISON_CONTRASTS`, `CENTROID_ORDERING_CONTRASTS` against the validated registry; replace `ses_low_income` with an existing ID; fail loudly if any featured contrast cannot be resolved.
- [ ] Re-derive centroid-ordering bootstrap CIs using a held-out-family direction (not the full-data direction).
- [ ] Surface `KEY_CONTRASTS` and `CENTROID_ORDERING_CONTRASTS` selections in a `run_config.json` next to the figures so the panel is reproducible.

## Notes from the doc audit
- The paper-summary panel pulls variance decomposition from [Step 8](08_analyze_identity_geometry_diagnostics.md)'s `diagnostics/variance_decomposition.csv` but the rest of the panel from activations directly. This means the panel can show stale variance numbers if [Step 8](08_analyze_identity_geometry_diagnostics.md) is rerun without rerunning this script — add a mtime check or a `--require_fresh_inputs` flag.
- The direction-stability curves (adjacent-layer cosine, cosine vs layer 24) are an excellent input to choosing the intervention layer for Stage 4 steering. Worth surfacing this prominently in the paper section discussing layer choice.
