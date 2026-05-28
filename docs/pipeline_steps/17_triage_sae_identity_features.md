# Step 17 — `triage_sae_identity_features.py`

**Stage:** 3 — Identity-selective SAE feature analysis (glue)
**Runs after:** `analyze_identity_sae_features.py`, `extract_token_level_sae_activations.py`, and optionally `analyze_shared_social_subspace.py`
**Feeds into:** `prepare_bbq_for_steering.py` (contrast→axis map), `run_bbq_sae_steering.py` (the feature pool, via `keep_for_intervention`), `analyze_bbq_feature_level_causal_effects.py` (metadata merge)

> This is the single most consequential glue script in the project. It is the only place where per-feature signal, membership, identity, token-localization, and shared-subspace metrics are joined into one table, scored, and converted into role-fit scores and a `keep_for_intervention` flag. Every causal claim downstream rests on the features this script keeps. After the audit 5.2 fix (2026-05-27), the **selection** is the load-bearing finding and the role labels are descriptive only; the cascade was replaced by a soft scoring head with a single-threshold keep rule. See [`docs/triage_preregistration_2026-05-27.md`](../triage_preregistration_2026-05-27.md) for the frozen weights and rule.

## Purpose

For each feature × layer, build a per-feature aggregate table that combines every upstream signal source, derive four hand-weighted summary scores (`contrast_specificity_score`, `sharedness_score`, `template_artifact_score`, `polysemanticity_score`), compute a 4-vector of soft role-fit scores, and decide `keep_for_intervention` via a single-threshold rule (audit 5.2, commit `235b5f5`). Emits the catalog (`intervention_candidate_features_triaged.csv`) that the BBQ steering pipeline reads as its feature pool. The descriptive `provisional_role` (argmax of the role-fit scores) is reported but **is not load-bearing** — see [`docs/triage_preregistration_2026-05-27.md`](../triage_preregistration_2026-05-27.md).

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

### Derived scores (`compute_scores`)

Four scores are weighted sums of upstream signals. As of audit 5.2 (commit `f306869`), the weights live in `DEFAULT_SCORE_WEIGHTS` at module level (parameterizable for the sensitivity sweep) and are pre-registered in [`docs/triage_preregistration_2026-05-27.md`](../triage_preregistration_2026-05-27.md):

- `contrast_specificity_score = 0.6 · (1 − min(1, (n_axes_top − 1) / 4)) + 0.2 · top_axis_fraction + 0.2 · min(1, max|d| / 2)`
- `sharedness_score = 0.5 · min(1, n_axes_top / 5) + 0.3 · min(1, n_contrasts_top / 10) + 0.2 · shared_pc_loading_score`
- `template_artifact_score = 0.4 · fraction_top_template_words + 0.3 · (1 − family_entropy) + 0.2 · (1 − template_entropy) + 0.1 · (1 − identity_span_localization_score)`
- `polysemanticity_score = 0.35 · axis_entropy + 0.35 · identity_entropy + 0.20 · token_entropy + 0.10 · (1 − top_axis_fraction)`

`axis_entropy`, `identity_entropy`, `token_entropy` are now Shannon entropy of **firing-count** categorical distributions (audit 5.2 part 1, commit `7f2c302`) — the prior implementation treated activation magnitudes as a probability mass, which is not motivated by any probability model.

### Soft scoring head and keep rule (`assign_roles`)

Audit 5.2 part 2 (commit `235b5f5`) replaced the 7-branch first-match cascade with a soft scoring head plus a single-threshold keep rule. Each feature now gets a 4-vector of role-fit scores in `[0, 1]`:

```
role_fit_identity_token_local       = mean(span_score, norm_d, 1 − template_artifact_score)
role_fit_sentence_final_integrated  = mean(final_score, norm_d, 1 − template_artifact_score)
role_fit_shared_social_feature      = mean(sharedness_score, min(n_axes_top / 3, 1), 1 − template_artifact_score)
role_fit_contrast_specific_identity = mean(contrast_specificity_score, norm_d, norm_cos, 1 − template_artifact_score)
```

with `norm_d = clip01(max|d| / (2 · min_abs_cohens_d))` and `norm_cos = clip01(max|cos| / (4 · min_abs_decoder_cosine))`.

The keep rule is independent of the role label:

```
keep_for_intervention =
        (not is_low_signal)
    AND (not is_template_artifact)
    AND (max(role_fit_*) >= --min_role_fit_keep)         # default 0.5
    AND (max_abs_cohens_d >= --min_abs_cohens_d)         # default 0.5
```

`is_low_signal = (max|d| < --min_abs_cohens_d) AND (max|cos| < --min_abs_decoder_cosine)`. `is_template_artifact = template_artifact_score >= --max_template_artifact_score_keep`.

The descriptive `provisional_role` is `argmax(role_fit_*)`, with overrides to `low_signal`, `template_or_syntax_artifact`, or `polysemantic_or_unclear` when the corresponding flag fires or no role-fit reaches `--min_role_fit_keep`. The audit's pathological case (span=0.71 vs shared=0.85 → permanently `identity_token_local`) now correctly picks `shared_social_feature` because the soft head's argmax is honest about which signal is stronger. `intervention_priority` is `"high"` if `keep AND role_confidence ≥ 0.7 AND max|d| ≥ 1.5 × min_abs_cohens_d`, else `"medium"` if `keep`, else `"low"`.

### Sensitivity sweep (`--sensitivity_sweep`)

Audit 5.2 part 3 (commit `f306869`). When the flag is set, the script re-runs scoring + role assignment with each threshold and each score-weight tuple element perturbed one-at-a-time by `--sensitivity_perturb_fractions` (default `0.8,0.9,1.1,1.2`). Outputs:

- `triage_sensitivity_per_feature.csv`: one row per (perturbation, feature) with baseline and perturbed labels.
- `triage_sensitivity_summary.csv`: one row per perturbation with `role_change_fraction`, `best_role_change_fraction`, `keep_change_fraction`, `delta_n_keep`, sorted descending so the most-disruptive perturbations are at the top.

The BBQ-side stability check (does the kept-feature set's BBQ effect distribution change across these perturbations?) is RunPod-deferred — see the pre-registration doc.

A free-text `reason` string is built per feature recording every score and threshold that fired — this is the single most useful column for understanding why a given feature was kept or dropped.

### HTML index (`write_html`)

Filterable table of the top-100 kept features sorted by priority → confidence → `max|d|`. Links to `feature_cards/layer_XX/feature_XXXXX.html` when present.

## Issues & Opportunities

> **Upstream callout — issue 1.4 (FIX LANDED; regenerate inputs).** The encoder fix in [Step 5](05_encode_identity_saes.md#14-blocker--sae-preprocessing-convention-fix-landed-2026-05-26) landed in commit `4b8851a`. This script joins per-feature metrics from `feature_selectivity_alignment_joined.csv`, `intervention_candidate_features.csv`, `feature_identity_selectivity.csv`, and the token-level activations — every one of those was produced by the broken encoder. After re-running [Step 5](05_encode_identity_saes.md) → [Step 6](06_validate_sae_hook_alignment.md) (confirm `reconstruction_fvu <= 0.15`) → [Step 13](13_analyze_identity_sae_features.md), the prior `feature_triage.csv` and `intervention_candidate_features_triaged.csv` are stale and the feature pool that drove every BBQ steering result must be rebuilt from scratch. The triage *logic* (issue 5.2) is independent.

### 5.2 [MAJOR] — Triage roles are heuristic definitions, not validated findings (PARTIAL FIX LANDED 2026-05-27)

**Status:** Four-part fix landed across commits `7f2c302` + `235b5f5` + `f306869` + the pre-registration doc. The two **validations** of the taxonomy (behavioral criterion + inter-rater agreement) are deferred to RunPod / human labeling and recorded in the pre-registration doc as outstanding work.

**What landed (parts 1-4 of the audit's targeted-fix list):**
1. **Pre-register the rule:** [`docs/triage_preregistration_2026-05-27.md`](../triage_preregistration_2026-05-27.md) pins the score weights, role-fit definitions, and keep-rule thresholds to the three code commits. It also reframes the kept-feature count as the only load-bearing finding and the taxonomy as descriptive unless validated.
2. **Sensitivity analysis:** `--sensitivity_sweep` runs the full triage with each threshold and each score-weight tuple element perturbed one-at-a-time. Writes `triage_sensitivity_summary.csv` with `role_change_fraction`, `keep_change_fraction`, `delta_n_keep` per perturbation. (Commit `f306869`.)
3. **Validation paths for the taxonomy** are recorded in the pre-registration doc:
   - *Behavioral criterion* — `identity_token_local` features must show larger absolute `bias_margin_delta` at `target_identity_last_token` than at `final_prompt_token`, and `sentence_final_integrated` features the opposite. Paired signed-rank test on `keep_for_intervention = True` features. Requires the BBQ steering run with `--intervention_modes ablate` at multiple positions.
   - *Inter-rater criterion* — two human labelers, stratified sample of 80 features (20 per role), Cohen's κ ≥ 0.6 against the cascade label.
4. **Entropy probability model** rewritten as categorical entropy over firing counts (audit's "Bernoulli feature-fired rate" option). The implicit probability model is "given the feature fired somewhere, what is the probability it fired in identity / token i." (Commit `7f2c302`.)
5. **Soft scoring head replaces the first-match cascade.** Each feature gets a 4-vector of role-fit scores; `keep_for_intervention` is a single-threshold rule on `max(role_fits) ≥ --min_role_fit_keep AND not low_signal AND not template_artifact AND max|d| ≥ --min_abs_cohens_d`. The audit's pathological case (span=0.71 vs shared=0.85 → permanently `identity_token_local`) now correctly picks `shared_social_feature`. (Commit `235b5f5`.)

**Why it matters (preserved):** The role labels propagate into the steering manifest (`run_bbq_sae_steering.py:load_feature_sets` uses `role` as a column), into the per-role aggregate analyses (`analyze_bbq_feature_level_causal_effects.py:summarize_effects` groups by role), and into the published feature taxonomy. Under the original cascade, "why these thresholds and not others?" had no answer in the code. Under the soft head + sensitivity sweep + pre-registration, the answer is: the thresholds are pre-registered constants, their stability is measured, and the role labels are descriptive only unless a validation passes.

### Inherited issues that flow through this script

These are upstream root causes whose effects are visible in this file's outputs; fix them at the source.

- **2.5 (post-selection bias)**: `max_abs_cohens_d` is the maximum over inflated, post-selection `cohens_d` values from `feature_selectivity.csv`. The cascade's `min_abs_cohens_d` threshold (default 0.5) is therefore operating on a biased scale. After the 2.5 fix, the same threshold value will be **more conservative** in real terms; consider re-tuning.
- **5.1 (reconstruction projection)**: *FIX LANDED 2026-05-27 upstream (commit `1a569c3`).* `direction_reconstruction.csv` now contains the true orthogonal projection — `fraction_norm_captured` is bounded in `[0, 1]` and satisfies `fraction = cosine²` by the projection identity. The triage doesn't read this CSV directly, but `aggregate_signal_metrics` should be re-validated against the now-correctly-bounded values: prior thresholds may have been set against values that could exceed 1.
- **5.3 (`combined_score` double-weighting)**: *FIX LANDED 2026-05-27 upstream (commit `3b48e5b`).* Upstream formula is now `0.5·z(|cohens_d|) + 0.5·z(|cosine_with_direction|)`. `max_combined_score` and the top-N-by-combined-score logic in `add_top_membership` propagate the rebalanced ranking; membership counts (`n_contrasts_where_top_feature`, `n_axes_where_top_feature`) and the derived `sharedness_score` will shift on regeneration toward features that are alignment-strong rather than selectivity-strong.
- **5.4 (residualized vs raw inconsistency)**: Upstream issue; this script consumes whichever values are in the CSV.
- **4.1 (silent skip of missing identity IDs)**: Because `analyze_identity_sae_features.py` silently drops contrasts with missing identity IDs, the triage's per-feature `n_contrasts_seen` and `n_axes_where_top_feature` undercount the SES axis. The cascade's branch 5 condition `n_axes_where_top_feature ≥ 3` is therefore harder to hit for SES features.

### Optional caveat — Feature card linking is fragile

`feature_card_link` (line 768) hunts in three relative paths to find a card. If the output directory layout changes, the index quietly emits no link. Consider passing the feature-card root as an explicit CLI argument.

## Rebuild checklist

Do these in order:

- [x] **Pre-register** the cascade and thresholds: [`docs/triage_preregistration_2026-05-27.md`](../triage_preregistration_2026-05-27.md). (5.2 → main fix, 2026-05-27)
- [x] Add a `--sensitivity_sweep` flag that sweeps each coefficient and each threshold and writes `triage_sensitivity_per_feature.csv` + `triage_sensitivity_summary.csv` with role-stability counts. (5.2, commit `f306869`)
- [x] Replace the heuristic `entropy()` over L1-normalized activations with categorical entropy over firing counts; documented in [`docs/triage_preregistration_2026-05-27.md`](../triage_preregistration_2026-05-27.md) §2.4. (5.2, commit `7f2c302`)
- [x] Move the cascade out of `assign_roles` into a soft `role_fit_*` matrix + a single keep rule; roles become continuous and the order-of-arms artefact disappears. (5.2, commit `235b5f5`)
- [ ] Validate the role taxonomy: after the steering 3.1 fix, check whether `identity_token_local` features actually have stronger effects at `target_identity_last_token` than at `final_prompt_token`, and vice versa for `sentence_final_integrated`. Add a `validation/` subdir to capture these checks. (5.2 RunPod follow-up; criteria in [pre-registration §5.1](../triage_preregistration_2026-05-27.md#51-behavioral-criterion--position-conditional-causal-effect))
- [ ] Inter-rater validation of role labels: 80 stratified-sample features × 2 human labelers, Cohen's κ ≥ 0.6 against the cascade label. (5.2 follow-up; criteria in [pre-registration §5.2](../triage_preregistration_2026-05-27.md#52-inter-rater-criterion--human-labelers))
- [ ] Re-run after upstream fixes (2.5, 5.1, 5.3, 5.4, 4.1) land in `analyze_identity_sae_features.py`. The downstream `keep_for_intervention` set will change; the BBQ steering pool changes with it.
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
