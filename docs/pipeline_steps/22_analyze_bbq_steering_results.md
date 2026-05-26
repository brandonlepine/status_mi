# Step 22 — `scripts/analyze_bbq_steering_results.py`

**Stage:** 4 — BBQ steering and feature-level causal analysis
**Runs after:** [Step 20 — `run_bbq_sae_steering.py`](20_run_bbq_sae_steering.md).
**Feeds into:** Sanity inspection only. **Marked LEGACY in `docs/bbq_steering_pipeline.md`** — explicitly not the final feature-level analysis. Use [Step 23 — `analyze_bbq_feature_level_causal_effects.py`](23_analyze_bbq_feature_level_causal_effects.md) for substantive causal claims.

## Purpose
Aggregate-level interpretation pass over `results_parts/*.parquet`. Reads every part file, merges with the prepared BBQ table for metadata, enriches with derived deltas, then aggregates at three coarse levels: **axis**, **contrast**, and **feature_set × alpha**. Computes bootstrap CIs (default 500 samples), writes an overview README, a smoke-test limitations document, and a folder of overview figures. Useful for sign-convention sanity checks and broad axis-level behavioral plots; not sufficient for individual-feature claims.

## Inputs
- `steering/results_parts/*.parquet` (or `.csv`) — from Step 20.
- Optionally, an already-merged `steering_results_merged.csv` via `--results_csv` (lets the analyzer re-run without re-loading raw parts).
- `prepared/bbq_prepared_examples.parquet` — for metadata enrichment.

## Outputs
```
analysis/
  steering_results_merged.csv     # full merged long table after enrichment
  coverage_report.csv             # n examples per (axis × contrast × alpha × position)
  README_interpretation.md        # sign-convention notes and how to read the plots
  SMOKE_TEST_LIMITATIONS.md       # explicit caveats for smoke runs
  interpretation_summary_by_axis.csv
  interpretation_summary_by_contrast.csv
  interpretation_summary_by_feature_set_alpha.csv
  figures/                        # ambiguous-bias-by-alpha, answer-logprob-shifts, axis-level-bias, etc.
  logs/analysis.log
```

## Key implementation details
- Loads every `part_*.parquet`/`part_*.csv` under `steering_dir/results_parts/` and concatenates.
- Re-derives `feature_role` from `feature_roles_json` (takes index 0), `feature_id` from `feature_ids_json` (takes index 0 for single-element rows).
- Computes group means + bootstrap CIs at each of the three aggregation levels.
- Writes a hand-authored `README_interpretation.md` and `SMOKE_TEST_LIMITATIONS.md` covering the axis-matching change, the bundle vs. per-feature distinction, sign conventions, and that the legacy aggregator collapses across alpha-sign, feature direction, and position. The presence of the limitations doc is itself a signal that the script is not for headline claims.
- No FDR correction, no per-feature CIs, no identity-level disambiguation.

## Issues & Opportunities

### 5.10 [MINOR] — Heavy code duplication across analysis scripts

**What's wrong:** This script reimplements `read_table`, `bool_series`, `parse_json_list`, `first_role`, axis-from-identity / axis-from-contrast helpers, `cohens_d`-style aggregations, and the Okabe-Ito-ish plotting helpers that also live in `analyze_bbq_feature_level_causal_effects.py`, `analyze_identity_sae_features.py`, `analyze_identity_geometry.py`, and `plot_identity_directional_*.py`. They are functionally equivalent today but drift silently — a sign-flip convention change in one script will not reach the others. Within Stage 4, the two analyzers (this and Step 23) each redefine `enrich_results`-style logic.

**Why it matters:** When the per-feature analyzer (Step 23) changes a delta definition (e.g. polarity-signing `stereotype_preference_delta` per issue 4.3), this aggregator will silently disagree. Reviewers comparing axis-level plots from this script with the per-feature analyzer's outputs will get confused.

**Targeted fix:** Extract a `status_mi/common.py` module containing: `read_table`, `parse_json_list`, `first_role`, axis maps, `bootstrap_ci`, `sign_flip_pvalue`, `fdr_bh`, the BBQ delta definitions (`stereotype_preference_delta`, ...), and the shared plotting palette. Have both this script and Step 23 import from there. Once that lands, this script's redundant helpers can be deleted.

### Inherited from upstream

All Step 20 issues propagate: `bias_margin_delta` etc. inherit the `first_token` scoring degeneracy (1.3), the controls-disabled run (2.3), the decoder-direction-not-feature intervention (3.1), the regex-located positions (3.3), the axis-fallback contrasts (3.4), the uniform alpha (3.2), and bundle averaging (3.5). All Step 18 issues propagate too — most importantly the absent polarity sign (4.3): this script aggregates `bias_margin_delta` across `question_polarity` without sign-correcting, so axis- and contrast-level summaries are polarity-confounded the same way Step 23's `effect_label` is.

### Status

The legacy aggregator is deliberately preserved as a smoke-and-overview tool. The recommendation in `docs/bbq_steering_pipeline.md` is that **Step 23 supersedes this script for any feature-level or identity-level claim**. The substantive issues to fix here are either:
(a) cross-cutting (1.3, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5, 4.3) — fix once in upstream / Step 23, this script benefits automatically, and
(b) the local code-duplication (5.10) — fix by consolidation.

## Rebuild checklist
- [ ] Decide: keep this script as the high-level overview aggregator, or delete it once Step 23's plots are sufficient. If keeping:
- [ ] Move shared helpers into `status_mi/common.py` and import them here. Remove the duplicates.
- [ ] Make the polarity sign explicit in every aggregation by depending on the `bias_polarity_sign` column added by Step 18.
- [ ] Add `mapped_contrast_confidence` to every aggregation key so the user can filter axis-level summaries to `exact`-only rows.
- [ ] Add a banner at the top of `README_interpretation.md` linking to Step 23 and clarifying that this aggregator does not produce feature-level causal estimates.

## Notes from the doc audit
- `coverage_report.csv` is the most useful artifact this script produces and is not duplicated in Step 23 in the same form. Worth preserving even after the rest of the script is retired.
- The hand-authored `README_interpretation.md` documents sign conventions that are correct for the *current* delta definitions but will need re-writing once polarity-signing (issue 4.3) lands.
- No `--smoke` flag here, but the `SMOKE_TEST_LIMITATIONS.md` is written unconditionally. Consider making it conditional or rename it to `AGGREGATION_LIMITATIONS.md`.
