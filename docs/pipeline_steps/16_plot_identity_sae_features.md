# Step 16 — `plot_identity_sae_features.py`

**Stage:** 3 — Identity-selective SAE feature analysis (presentation layer)
**Runs after:** `analyze_identity_sae_features.py`
**Feeds into:** Human inspection only; no downstream computation consumes these figures.

## Purpose

Bulk static plotting over the CSVs produced by `analyze_identity_sae_features.py` — selectivity heatmaps (top features × contrasts by Cohen's d), selectivity-vs-decoder-alignment scatters per key contrast, reconstruction-curve plots, and supporting feature-profile panels. Presents the same numbers that `triage_sae_identity_features.py` aggregates, but in figure form for paper/report use.

## Inputs

- `analysis/feature_selectivity.csv`
- `analysis/decoder_direction_alignment.csv`
- `analysis/feature_selectivity_alignment_joined.csv`
- `analysis/direction_reconstruction.csv`
- `analysis/feature_identity_selectivity.csv`
- (Optionally) `analysis/intervention_candidate_features.csv` for ordering and labels.

## Outputs (under `<output_dir>/`, default `.../final_token/figures/`)

- `selectivity/top_feature_selectivity_heatmap_layer{XX}.{png,pdf}` — top-80 features × contrast Cohen's d heatmap.
- `alignment/selectivity_vs_decoder_alignment_{contrast}_layer{XX}.{png,pdf}` — per-`KEY_CONTRASTS` scatter of `cohens_d` vs `cosine_with_direction`, with top-10 by `combined_score` labelled.
- `reconstruction/...`, `feature_profiles/...`, `feature_cards/...` (subdirs created by `prepare_output`).

## Key implementation details

- `KEY_CONTRASTS` (lines 25-32) is a fixed subset of six contrasts used to gate the scatter plots. Includes `ses_low_income_vs_ses_rich` — which references an identity ID missing from the dataset (see issue 4.1). The contrast row will not exist in the joined CSV, so the plot is silently skipped.
- All readers use `safe_read` (line 67), which returns an empty DataFrame on missing/empty files; plots are skipped silently in that case.
- The selectivity heatmap (line 73) ranks features by absolute `cohens_d` and takes the top 80 — the exact post-selection-bias numbers from `feature_selectivity.csv`.
- `save_fig` emits PNG and PDF together at 220 DPI.

## Issues & Opportunities

### 5.10 [MINOR] — Heavy code duplication across analysis scripts

**What's wrong:** `save_fig`, Okabe-Ito-style palettes (where used), and contrast-list constants are independently re-implemented across `analyze_identity_geometry.py`, `analyze_identity_geometry_diagnostics.py`, `plot_identity_directional_visualizations.py`, `plot_identity_directional_followups.py`, and here. The `KEY_CONTRASTS` list in this file is yet another contrast constant, separate from `DEFAULT_CONTRASTS` in `analyze_identity_sae_features.py` and from the BBQ contrast mapping in `prepare_bbq_for_steering.py`.

**Why it matters:** If `ses_low_income_vs_ses_rich` is renamed or dropped (per the 4.1 fix), this script will silently produce one fewer plot with no error. If the upstream `combined_score` definition changes (5.3), the "top-10 labelled" points in the scatter (line 110) will shift without notice.

**Targeted fix:** Centralize. Move `save_fig`, `KEY_CONTRASTS` (as `HEADLINE_CONTRASTS`, validated against the canonical contrast registry), and the palette into a shared module. The contrast list here should be a curated subset of the validated registry, not a hand-rolled set of strings. See the parallel notes in step 13 (5.10).

### Inherited issues (do not require changes to this file)

- **2.5 (post-selection bias)** — The heatmap and scatters faithfully display the inflated Cohen's d numbers. Once the upstream fix lands, the plots will improve automatically.
- **5.1 (reconstruction projection math)** — *FIX LANDED 2026-05-27 upstream (commit `1a569c3`).* The reconstruction-curve plots inherit `fraction_norm_captured` and `cosine_with_full_direction` from `direction_reconstruction.csv`; the upstream now writes the true least-squares projection (`fraction ∈ [0, 1]`, `cosine² = fraction`). Plot code unchanged — re-run with `--overwrite` against re-generated CSVs.
- **5.3 (`combined_score` double-weighting)** — *FIX LANDED 2026-05-27 upstream (commit `3b48e5b`).* The top-10-by-`combined_score` labels in the alignment scatter (line 110) now reflect the rebalanced formula `0.5·z(|cohens_d|) + 0.5·z(|cosine_with_direction|)`. Plot code unchanged; re-run with `--overwrite` against regenerated CSVs.

## Rebuild checklist

- [ ] After the contrast-registry audit (4.1), update `KEY_CONTRASTS` to reference only valid IDs, ideally by importing from the shared registry.
- [ ] After the `combined_score` fix (5.3) and selection-bias fix (2.5), rerun this script with `--overwrite`. No structural changes needed here.
- [ ] Move `save_fig` and the figure-style boilerplate into the shared module (5.10).
- [ ] Add an explicit "this plot would have been produced for contrast X but the contrast row is missing" log so silent skips become visible during the rebuild.

## Notes from the doc audit

- The `feature_cards/` subdir is created by `prepare_output` (line 55) but this script's `plot_*` functions write to `selectivity/`, `alignment/`, `reconstruction/`, `feature_profiles/` only. The `feature_cards/` subdir is left empty unless an external step writes there.
- `prepare_output` requires `--overwrite` to clobber a non-empty output dir; partial reruns are not supported.
