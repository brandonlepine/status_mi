# Step 17 — `triage_sae_identity_features.py`

**Stage:** 3 — Identity-selective SAE feature analysis (glue)
**Runs after:** `analyze_identity_sae_features.py`, `extract_token_level_sae_activations.py`, and optionally `analyze_shared_social_subspace.py`
**Feeds into:** `prepare_bbq_for_steering.py` (contrast→axis map), `run_bbq_sae_steering.py` (the feature pool, via `keep_for_intervention`), `analyze_bbq_feature_level_causal_effects.py` (metadata merge)

> This is the single most consequential glue script in the project. It is the only place where per-feature signal, membership, identity, token-localization, and shared-subspace metrics are joined into one table, scored, and converted into a **role label** and a `keep_for_intervention` flag. Every causal claim downstream rests on the features this script keeps. The role definitions are heuristic — see issue 5.2 — but the **selection** they imply is load-bearing.

## Purpose

For each feature × layer, build a per-feature aggregate table that combines every upstream signal source, derive four hand-weighted summary scores (`contrast_specificity_score`, `sharedness_score`, `template_artifact_score`, `polysemanticity_score`), and run a rule-based cascade that assigns each feature a `provisional_role` and a `keep_for_intervention` flag. Emits the catalog (`intervention_candidate_features_triaged.csv`) that the BBQ steering pipeline reads as its feature pool.

## Inputs

- `analysis/feature_selectivity_alignment_joined.csv`, `feature_selectivity.csv`, `feature_identity_selectivity.csv`, `decoder_direction_alignment.csv`, `intervention_candidate_features.csv`, `direction_reconstruction.csv` — from `analyze_identity_sae_features.py`.
- `feature_cards/token_level/layer_XX/token_feature_activations.csv` and `feature_top_tokens.csv` — from `extract_token_level_sae_activations.py`.
- (Optional) `shared_subspace_decomposition/metrics/*.csv` — from `analyze_shared_social_subspace.py`, used to derive `shared_pc_loading_score`.

## Outputs (under `<output_dir>/`, default `.../final_token/triage/`)

- `feature_triage.csv` — every feature × every aggregated metric × `provisional_role` × `role_confidence` × `keep_for_intervention` × `intervention_priority` × `reason`.
- `intervention_candidate_features_triaged.csv` — per-(layer, contrast, feature) rows from `intervention_candidate_features.csv` merged with triage role/keep columns. **This is the file the BBQ pipeline reads.**
- `feature_triage_summary.csv` — per-role aggregate counts and means.
- `role_counts.csv` — per (layer, role) counts.
- `triage_index.html` — filterable HTML table of kept features with links to feature cards.
- `figures/{role_counts, score_distributions, role_by_axis_heatmap, keep_for_intervention_by_contrast, scatter_selectivity_vs_artifact, scatter_sharedness_vs_specificity}.{png, pdf}`.
- `intermediate/{signal_metrics, top_feature_membership, identity_specificity_metrics, token_localization_metrics, shared_subspace_scores, feature_metric_table_pre_roles}.csv`.
- `triage_config.json` — captures every CLI threshold so the rule is reproducible.

## Key implementation details

### Per-feature aggregation (lines ~200-565)

For each layer the script computes five families of metrics, each emitted as an `intermediate/*.csv`:

1. **Signal metrics** (`aggregate_signal_metrics`, from `feature_selectivity_alignment_joined.csv` + `decoder_direction_alignment.csv` + `intervention_candidate_features.csv`):
   `max_abs_cohens_d`, `max_auc_distance_from_0_5`, `max_abs_decoder_cosine`, `mean_abs_decoder_cosine`, `max_combined_score`, `top_contrast_by_selectivity`, `top_contrast_by_decoder_alignment`, `n_contrasts_seen`, `n_axes_seen`, `n_identities_seen`, `signal_top_axis`.

2. **Top-feature membership** (`add_top_membership` + `aggregate_membership`): for each (contrast, axis) pair, take the top-`top_n_per_contrast` features by `combined_score` from `intervention_candidate_features.csv`. Counts per feature: `n_contrasts_where_top_feature`, `n_axes_where_top_feature`.

3. **Identity specificity** (`aggregate_identity`, from `feature_identity_selectivity.csv`): per feature, look at the top-10 identities ranked by `cohens_d`. `top_identity`, `top_axis`, `top_axis_fraction` (fraction in same axis), `axis_entropy`, `identity_entropy`, `top_identities_by_activation`.

4. **Token-level localization** (`aggregate_token_metrics`, from `token_feature_activations.csv`): per feature, median across exemplar prompts of (max-identity-span-activation / max-token-activation) → `identity_span_localization_score`; same for final-token → `final_token_integration_score`. Plus `fraction_top_tokens_template_words`, `family_entropy`, `template_entropy`, `token_entropy`, `cross_axis_activation_score`, and a categorical `feature_localization_type ∈ {identity_span_local, final_token_integrated, template_context, diffuse_or_unclear}`.

5. **Shared-subspace loading** (`aggregate_shared_loadings`, optional): a normalized 0-1 `shared_pc_loading_score` per feature, derived from the maximum |loading| / |projection| / |variance| / |score| numeric column across the shared-subspace metrics CSVs.

These are merged into one row per (layer, feature_id) by `complete_feature_table` + `compute_scores`.

### Derived scores (`compute_scores`, lines 642-689)

Each is hand-weighted with hardcoded coefficients:

- `contrast_specificity_score = 0.6 · (1 − min(1, (n_axes_top − 1) / 4)) + 0.2 · top_axis_fraction + 0.2 · min(1, max|d| / 2)`
- `sharedness_score = 0.5 · min(1, n_axes_top / 5) + 0.3 · min(1, n_contrasts_top / 10) + 0.2 · shared_pc_loading_score`
- `template_artifact_score = 0.4 · fraction_top_template_words + 0.3 · (1 − family_entropy) + 0.2 · (1 − template_entropy) + 0.1 · (1 − identity_span_localization_score)`
- `polysemanticity_score = 0.35 · axis_entropy + 0.35 · identity_entropy + 0.20 · token_entropy + 0.10 · (1 − top_axis_fraction)`

The `entropy()` helper used by axis/identity/token entropy treats activation magnitudes as a probability distribution after L1-normalizing, which is heuristic (not a true Shannon entropy over an actual probability model).

### The role-assignment cascade (`assign_roles`, lines 582-639)

Applied **in order** per feature; first match wins:

1. If `max|d| < min_abs_cohens_d` **and** `max|cos| < min_abs_decoder_cosine` → **`low_signal`**, `keep=False`.
2. Else if `template_artifact_score ≥ max_template_artifact_score_keep` (default 0.5) → **`template_or_syntax_artifact`**, `keep=False`.
3. Else if `identity_span_localization_score ≥ identity_span_local_threshold` (default 0.7) **and** `max|d| ≥ min_abs_cohens_d` → **`identity_token_local`**, `keep=True`.
4. Else if `final_token_integration_score ≥ final_token_integrated_threshold` (default 0.7) **and** `max|d| ≥ min_abs_cohens_d` → **`sentence_final_integrated`**, `keep=True`.
5. Else if `sharedness_score ≥ min_sharedness_score_shared` (default 0.5) **and** `n_axes_where_top_feature ≥ 3` → **`shared_social_feature`**, `keep = (|d| ≥ thresh) AND (artifact < thresh)`.
6. Else if `contrast_specificity_score ≥ min_contrast_specificity_keep` (default 0.5) **and** `max|d| ≥ min_abs_cohens_d` **and** `max|cos| ≥ min_abs_decoder_cosine` → **`contrast_specific_identity`**, `keep=True`.
7. Else → **`polysemantic_or_unclear`**, `keep=False`.

After the cascade, `keep` is **anded** with `(max|d| ≥ min_abs_cohens_d) AND (template_artifact_score < max_template_artifact_score_keep)` (line 623), enforcing the floor even on the keep branches. `intervention_priority` is `"high"` if `keep AND role_confidence ≥ 0.7 AND max|d| ≥ 1.5 × min_abs_cohens_d`, else `"medium"` if `keep`, else `"low"`.

A free-text `reason` string is built per feature recording every score and threshold that fired — this is the single most useful column for understanding why a given feature was kept or dropped.

### HTML index (`write_html`)

Filterable table of the top-100 kept features sorted by priority → confidence → `max|d|`. Links to `feature_cards/layer_XX/feature_XXXXX.html` when present.

## Issues & Opportunities

> **Upstream callout — issue 1.4 (FIX LANDED; regenerate inputs).** The encoder fix in [Step 5](05_encode_identity_saes.md#14-blocker--sae-preprocessing-convention-fix-landed-2026-05-26) landed in commit `4b8851a`. This script joins per-feature metrics from `feature_selectivity_alignment_joined.csv`, `intervention_candidate_features.csv`, `feature_identity_selectivity.csv`, and the token-level activations — every one of those was produced by the broken encoder. After re-running [Step 5](05_encode_identity_saes.md) → [Step 6](06_validate_sae_hook_alignment.md) (confirm `reconstruction_fvu <= 0.15`) → [Step 13](13_analyze_identity_sae_features.md), the prior `feature_triage.csv` and `intervention_candidate_features_triaged.csv` are stale and the feature pool that drove every BBQ steering result must be rebuilt from scratch. The triage *logic* (issue 5.2) is independent.

### 5.2 [MAJOR] — Triage roles are heuristic definitions, not validated findings

**What's wrong:** The four derived scores (`contrast_specificity_score`, `sharedness_score`, `template_artifact_score`, `polysemanticity_score`) are linear combinations with hand-picked weights (0.6/0.2/0.2; 0.5/0.3/0.2; 0.4/0.3/0.2/0.1; 0.35/0.35/0.20/0.10). The seven role branches use hand-picked thresholds (0.5, 0.7) on those scores. The entropy components additionally treat L1-normalized activation magnitudes as probability distributions, which is heuristic. None of these weights or thresholds were validated against a behavioral or human-labelled criterion. As an engineering filter to pick which features get steered this is fine, but presenting "we identified N `identity_token_local` features and M `shared_social_feature` features" as a **finding** mistakes a definition for a discovery.

**Why it matters:** The role labels propagate into the steering manifest (`run_bbq_sae_steering.py:load_feature_sets` uses `role` as a column), into the per-role aggregate analyses (`analyze_bbq_feature_level_causal_effects.py:summarize_effects` groups by role), and into the published feature taxonomy in `final_intervention_candidates_table.html`. If a reviewer asks "why these thresholds and not others?" there is no answer in the code.

**Targeted fix (in priority order):**
1. **Reframe triage strictly as feature selection**, and pre-register the cascade and thresholds in `docs/` *before* looking at BBQ results, so the rule cannot be tuned to the causal outcome. The kept-feature *count* is fine to report; the role *taxonomy* is not a finding.
2. **Sensitivity analysis.** Sweep the four weighting tuples and the seven thresholds (e.g. ±20% per coefficient) and report the fraction of features whose role changes. If conclusions on the BBQ side are stable to reasonable perturbations, say so; if not, that is itself important to know.
3. **If the taxonomy is a paper contribution**, validate it: (a) human inter-rater agreement on a sample of feature cards (`identity_token_local` vs `sentence_final_integrated` vs `template_or_syntax_artifact` — a labeller using the feature cards from step 15 should agree with the cascade's label), (b) a falsifiable behavioral criterion: `identity_token_local` features should show their causal effect specifically at `target_identity_last_token` position and **not** at `final_prompt_token`; `sentence_final_integrated` features should show the opposite. `analyze_bbq_feature_level_causal_effects.py` already groups by `intervention_position`, so this prediction is testable from the existing steering data once the 3.1 fix lands.
4. **Replace `entropy()` over L1-normalized activations** with either a properly motivated probability model (e.g. activation as Bernoulli "feature fired" rate and compute Shannon entropy of that), or use a simpler concentration index (e.g. Gini, or top-k share). Document the choice.
5. **Replace the cascade with a soft scoring head** (e.g. each feature gets a vector of "role-fit" scores; the keep decision is a separate, single-threshold rule on `selectivity ∧ not-artifact`). The cascade's first-match-wins ordering is brittle: a feature that scores 0.71 on identity-span localization and 0.85 on sharedness is permanently labeled `identity_token_local`, never `shared_social_feature`, even though sharedness is the stronger signal.

### Inherited issues that flow through this script

These are upstream root causes whose effects are visible in this file's outputs; fix them at the source.

- **2.5 (post-selection bias)**: `max_abs_cohens_d` is the maximum over inflated, post-selection `cohens_d` values from `feature_selectivity.csv`. The cascade's `min_abs_cohens_d` threshold (default 0.5) is therefore operating on a biased scale. After the 2.5 fix, the same threshold value will be **more conservative** in real terms; consider re-tuning.
- **5.1 (reconstruction projection)**: The triage does not currently use `direction_reconstruction.csv`, so this script is not directly affected — but the "fraction norm captured" column is referenced indirectly through the `signal_metrics` aggregation. Audit `aggregate_signal_metrics` after the 5.1 fix.
- **5.3 (`combined_score` double-weighting)**: `max_combined_score` and the top-N-by-combined-score logic in `add_top_membership` propagate the upstream double-weighting. After the 5.3 fix the membership counts (`n_contrasts_where_top_feature`, `n_axes_where_top_feature`) and the derived `sharedness_score` will shift.
- **5.4 (residualized vs raw inconsistency)**: Upstream issue; this script consumes whichever values are in the CSV.
- **4.1 (silent skip of missing identity IDs)**: Because `analyze_identity_sae_features.py` silently drops contrasts with missing identity IDs, the triage's per-feature `n_contrasts_seen` and `n_axes_where_top_feature` undercount the SES axis. The cascade's branch 5 condition `n_axes_where_top_feature ≥ 3` is therefore harder to hit for SES features.

### Optional caveat — Feature card linking is fragile

`feature_card_link` (line 768) hunts in three relative paths to find a card. If the output directory layout changes, the index quietly emits no link. Consider passing the feature-card root as an explicit CLI argument.

## Rebuild checklist

Do these in order:

- [ ] **Pre-register** the cascade and thresholds: write the current rule into `docs/triage_rule.md`, commit, then do not change the rule after looking at BBQ results. (5.2 → main fix)
- [ ] Add a `--sensitivity_analysis` flag that sweeps each coefficient ±20% and each threshold ±0.1, and writes `triage_sensitivity.csv` with role-stability counts. (5.2)
- [ ] Replace the heuristic `entropy()` over L1-normalized activations with a Bernoulli-firing-rate Shannon entropy or a Gini concentration; document the choice. (5.2)
- [ ] Re-run after upstream fixes (2.5, 5.1, 5.3, 5.4, 4.1) land in `analyze_identity_sae_features.py`. The downstream `keep_for_intervention` set will change; the BBQ steering pool changes with it.
- [ ] Validate the role taxonomy: after the steering 3.1 fix, check whether `identity_token_local` features actually have stronger effects at `target_identity_last_token` than at `final_prompt_token`, and vice versa for `sentence_final_integrated`. Add a `validation/` subdir to capture these checks. (5.2)
- [ ] (Optional, larger refactor) Move the cascade out of `assign_roles` into a soft "role-fit-score" matrix + a single keep rule. Roles become continuous; the cascade's order-of-arms artefact disappears. (5.2)
- [ ] Add an explicit `--feature_card_dir` arg to remove the fragile multi-path search in `feature_card_link`.

## Notes from the doc audit

- `aggregate_shared_loadings` (line 520) is heuristically picking which column to treat as the "loading" by looking for substring matches in the column name (`loading`, `shared`, `projection`, `variance`, `score`). If the upstream shared-subspace CSV schema ever changes, this could silently start scoring features off the wrong column. Worth either (a) requiring an explicit column name via CLI, or (b) hard-coding the expected file/column pair (e.g. `contrast_pc_loadings.csv:abs_loading`) and failing loudly when absent.
- `read_token_table` (line 114) searches both `token_level_dir/layer_XX` and the bare `token_level_dir`, which is convenient for legacy layouts but masks the case where the wrong layer's file is read. Worth asserting `layer == requested_layer` from the dataframe after filtering.
- `coalesce_columns` and `normalize_columns` (lines 104, 158) implement an alias map for column-name variations (`decoder_cosine_with_direction → cosine_with_direction`, etc.). This silently tolerates schema drift between upstream runs. Useful in practice but means the triage will produce output even when an upstream script was using a stale column name; consider logging when an alias actually fires.
- `--html_only` (line 67) regenerates the HTML index from `feature_triage.csv` without recomputing metrics. Good for iterating on the HTML view; do not confuse with a full rerun.
- The cascade's branch ordering means a feature with `template_artifact_score = 0.51` (just over default 0.5) cannot ever be kept, regardless of how clean its identity-span localization is. With sensitivity sweeps (above) this becomes visible.

## Cross-references

- The `combined_score` weighting issue is fixed in [step 13](./13_analyze_identity_sae_features.md).
- The post-selection bias issue is fixed in [step 13](./13_analyze_identity_sae_features.md); after that fix, re-tune `min_abs_cohens_d` here.
- The token-level localization data comes from [step 14](./14_extract_token_level_sae_activations.md). The 0.7 thresholds for `feature_localization_type` are set there; this script's `identity_span_local_threshold` / `final_token_integrated_threshold` must stay in sync.
- The kept-feature CSV is consumed by the BBQ pipeline; see `20_run_bbq_sae_steering.md` (forthcoming) and the 3.1 fix that will rewrite the steering hook to actually use these features as features (not as directions).
