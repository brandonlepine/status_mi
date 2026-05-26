# Step 23 — `scripts/analyze_bbq_feature_level_causal_effects.py`

**Stage:** 4 — BBQ steering and feature-level causal analysis
**Runs after:** [Step 20 — `run_bbq_sae_steering.py`](20_run_bbq_sae_steering.md), [Step 18 — `prepare_bbq_for_steering.py`](18_prepare_bbq_for_steering.md), [Step 17 — `triage_sae_identity_features.py`](17_triage_sae_identity_features.md). Optionally merges in summary metadata from [Step 19 — `extract_bbq_token_level_sae_activations.py`](19_extract_bbq_token_level_sae_activations.md).
**Feeds into:** Human inspection — `final_intervention_candidates_table.html`, the axis/contrast/identity report folders, and the final ranking CSVs.

This is the substantive feature-level analyzer that supersedes Step 22.

## Purpose
Take every steering result row, compute per-row identity-aware deltas (including identity-specific bias deltas relative to the unknown answer), expand bundle rows into one row per `feature_id`, build identity records per BBQ example (`target`, `nontarget`, `stereotyped_identity`, `nonstereotyped_identity`), aggregate at four grouping levels (feature, subgroup, identity, axis), apply bootstrap CIs and sign-flip permutation tests with Benjamini-Hochberg FDR, label each significant effect with an identity-aware behavioral taxonomy, and emit the final intervention-candidates table plus per-axis / per-contrast / per-identity reports.

## Inputs
- `steering/results_parts/*.parquet` (or `.csv`) — from Step 20.
- `prepared/bbq_prepared_examples.parquet` — joins implicit via `bbq_uid`; metadata already on Step 20's rows.
- `results/.../triage/intervention_candidate_features_triaged.csv` — merged in for feature metadata (`provisional_role`, `contrast_name`, `top_axis`, `top_identity`, `role_confidence`, `max_abs_cohens_d`, `template_artifact_score`, `sharedness_score`, `contrast_specificity_score`, `intervention_priority`).
- `token_level_sae/bbq_token_level_sae_summary.csv` — optional, merged in for `mean_target_identity_activation`, etc.

## Outputs
```
feature_level_causal_analysis/
  merged_results.csv + .parquet
  deltas_long.csv + .parquet            # one row per (steering row × feature_id) after bundle expansion
  identity_records.csv
  identity_deltas_long.csv              # deltas joined to identity records
  feature_level_pre_fdr.csv             # group-level means + p-values before metadata merge
  feature_level_effects.csv + .parquet  # post-FDR + metadata-merged
  subgroup_level_effects.csv            # also groups by contrast × target_identity
  identity_level_effects.csv            # per (feature × identity_id)
  feature_x_subgroup_matrix.csv         # pivot: feature_id × mapped_contrast_name
  feature_effect_rankings.csv           # top-100 by 6 ranking dimensions
  validation_summary.csv
  final_intervention_candidates_table.html
  feature_card_links_table.html
  analysis/axis_reports/<axis>/...
  analysis/contrast_reports/<contrast>/...
  analysis/identity_reports/<identity>/...
  figures/...
  README.md                              # sign convention + statistics description
  logs/, feature_level_causal_config.json
```

## Key implementation details

### Per-row enrichment (`enrich_results`)
For each Step 20 output row, computes:
- `stereotype_preference_delta = Δ[log p(stereotyped) − log p(unknown)]` — the headline bias quantity.
- `nontarget_preference_delta = Δ[log p(nonstereotyped) − log p(unknown)]`.
- `identity_substitution_delta = Δ[log p(nonstereotyped) − log p(stereotyped)]`.
- `correct_margin_delta = Δ[log p(correct) − max log p(others)]` (disambig only).
- `stereotype_error_delta = Δ[log p(stereotyped) − log p(correct)]`.
- `accuracy_delta = correct_intervened − correct_base` (float, ±1, 0).
- `steering_direction_label ∈ {feature_amplification, feature_suppression}` from sign of `alpha`.
- `feature_estimate_type ∈ {individual_feature, feature_bundle_membership}` from len(`feature_ids_json`) == 1.

### Bundle expansion (`expand_feature_rows`)
For each bundle row, emit one row per `feature_id` in `feature_ids_json` with `feature_estimate_type = feature_bundle_membership`. For per-feature rows, `individual_feature`.

### Identity records (`build_identity_records`, `merge_identity_records`)
For each BBQ example, emit up to four identity-role records (`target`, `nontarget`, `stereotyped_identity`, `nonstereotyped_identity`), each with an `answer_idx_for_identity`. After expansion, `identity_answer_delta = Δ log p(answer_idx_for_identity)` and `identity_specific_bias_delta = identity_answer_delta − unknown_delta`.

### Summary statistics (`summarize_effects`)
For each group-col tuple:
1. Aggregate to per-example means (one row per `bbq_uid`) over the grouped rows.
2. `bootstrap_ci`: percentile bootstrap over the per-example deltas (`--bootstrap_samples`, default 1000; `--smoke` caps to 500).
3. `sign_flip_pvalue`: paired sign-flip permutation test on absolute mean. `null = abs((signs * values).mean())`, p = `(sum(null >= observed) + 1) / (n + 1)` (`--permutation_samples`, default 1000; `--smoke` caps to 500).
4. FDR (Benjamini-Hochberg) is applied **within `(axis_mapped, context_condition, alpha, intervention_position)` strata** — see issue 2.6.
5. Significance: `q_value_fdr < args.q_threshold` (default 0.1).

### Effect labels (`effect_label`)
Heuristic identity-aware taxonomy applied per group-row:
- `capability_degrading` (disambig + reliable + correct margin/correct/accuracy drop) — distinct from bias.
- `identity_only` (identity_delta ≥ threshold but bias_delta < threshold or not reliable) — signal but not bias.
- `bias_amplifying` (reliable + bias_delta > threshold).
- `bias_reducing_uncertainty` (reliable + bias_delta < −threshold + `mean_unknown_delta > 0`): mass moves to "unknown".
- `bias_reducing_substitution` (reliable + bias_delta < −threshold + `mean_nonstereotyped_delta > mean_unknown_delta`): mass moves to the *other identity*. This is **not** debiasing — the model just swaps stereotypes.
- `general_answer_suppression` (all three deltas negative).
- `mixed_or_unclear` / `no_reliable_effect` (fallthroughs).

### Rankings (`make_rankings`)
Six top-100 rankings: `strongest_bias_reducing_features`, `strongest_bias_amplifying_features`, `strongest_unknown_increasing_features`, `strongest_substitution_effect_features`, `least_capability_degrading_features`, `strongest_capability_degrading_features`. Sorted by the appropriate `mean_*_delta` columns.

### Reports
Per-axis, per-contrast, per-identity HTML/figure folders gated by `--make_*_reports` flags. The final candidates table `final_intervention_candidates_table.html` is sorted by `beneficial_score` (descending) and `mean_stereotype_preference_delta` (ascending).

## Strengths to preserve

These are explicit positives that should survive any refactor (Section 6 of the audit):

- **The bias taxonomy** distinguishing `bias_reducing_uncertainty` (probability mass moves to "unknown" → genuine debiasing into uncertainty) from `bias_reducing_substitution` (mass moves to the *other identity* → the model just swapped one stereotype for another) is conceptually sharp. A steering result that just swaps stereotypes is **not** debiasing, and the code already encodes that distinction. Keep this label structure.
- **The `individual_feature` vs `feature_bundle_membership` distinction** is tracked explicitly via `feature_estimate_type` and the analyzer warns when only bundle rows exist (`logger.warning("No individual-feature steering rows found. ...")`). This prevents bundle effects from being read as single-feature causal claims and should be preserved verbatim.
- **The sign-flip permutation test** for p-values, rather than naive t-tests, is the right discipline for paired-delta data. The same discipline should be brought to the geometry pipeline (issue 2.2).
- **The `final_intervention_candidates_table.html` cross-links** to both the original identity-prompt feature card and the BBQ feature card per feature. The dual-card framing is correct: an "interesting" feature should make sense on both substrates.
- **Identity-aware metrics** (`identity_answer_delta`, `identity_specific_bias_delta`) let the analyzer report effects in terms of "this identity's answer probability shifted by X" rather than abstract stereotype/unknown deltas — useful for fairness framing.

## Issues & Opportunities

### 4.3 [BLOCKER] — `stereotype_preference_delta` has no polarity sign — `effect_label` and rankings are polarity-confounded

**What's wrong:** BBQ has negative-polarity questions ("Who was bad at X?") and non-negative ones ("Who was good at X?"). `stereotyped_groups` is the group the *negative* stereotype targets. For a **negative** question, choosing the stereotyped group is the *bias-consistent* answer. For a **non-negative** question, choosing the stereotyped group is the *anti-bias* answer.

`enrich_results` defines `stereotype_preference_delta = Δ[log p(stereotyped) − log p(unknown)]` **with no polarity sign**. The grouping keys include `question_polarity`, so individual rows are separable, but `effect_label`, `beneficial_score`, `harmful_score`, `make_rankings`, and `final_candidates_html` do **not** condition on or sign-correct for polarity. A feature that raises `log p(stereotyped group)` is labeled `bias_amplifying` even on non-negative items where that movement is the *unbiased* direction. `final_intervention_candidates_table.html` is sorted by a polarity-confounded `beneficial_score`.

**Why it matters:** Any "this feature reduces bias" / "this feature amplifies bias" claim is partially wrong wherever polarity matters. The candidates table — which is the headline artifact of this script — currently mixes (i) features that genuinely reduce stereotype preference on negative-polarity questions with (ii) features that *amplify* counter-stereotype preference on non-negative questions. The rankings cannot be trusted until polarity is signed.

**Targeted fix:**
- Define `signed_bias_delta = stereotype_preference_delta * bias_polarity_sign`, where `bias_polarity_sign = +1` if `question_polarity == "neg"` else `-1`. (Have Step 18 emit this column — see Step 18 issue 4.3 — or compute it inline here from `question_polarity`.)
- Use `signed_bias_delta` everywhere a "bias direction" is asserted: `effect_label` thresholds, `beneficial_score`, `harmful_score`, `make_rankings`, the per-row `final_candidates_html` sort key.
- Keep `stereotype_preference_delta` available as a diagnostic so polarity-stratified plots are still possible, but make the *signed* version the headline.
- Update `README.md`'s sign-convention section accordingly. Re-state: "Negative `signed_bias_delta` means the feature reduces bias relative to the polarity-correct anti-bias direction."

### 2.4 [MAJOR] — Length bias contaminates argmax/accuracy metrics inherited from Step 20

**What's wrong:** `accuracy_delta`, `correct_base`/`correct_intervened`, and `prediction_changed` are computed in Step 20 via `argmax` over raw summed `answer_logprob` scores (or first-token scores). The summed `answer_logprob` is length-biased; `"Cannot be determined"` is systematically penalized. This analyzer inherits the contamination wholesale: `mean_accuracy_delta`, the `capability_degrading` label condition, and the `least_capability_degrading_features` / `strongest_capability_degrading_features` rankings all rest on contaminated argmaxes.

**Why it matters:** Any disambig-context conclusion about "this feature degrades capability" is biased toward short-correct-answer examples and against unknown-correct-answer examples. Within-example *deltas* on a fixed answer are still unbiased — so `stereotype_preference_delta` etc. are OK — but the accuracy/argmax row is not.

**Targeted fix:**
- Push the fix upstream: have Step 20 emit length-normalized scores (or score the letter A/B/C). Once Step 20 fixes this, this analyzer benefits automatically.
- As a defensive measure here, recompute argmax on length-normalized scores when this script loads steering rows: divide each `ansK_logprob_*` by the token-length of `ansK` (cache token lengths in `bbq_prepared_examples.parquet`).

### 2.5 [MAJOR] — Selection-induced bias (winner's curse) on feature rankings

**What's wrong:** `make_rankings` ranks the top-100 features by `mean_stereotype_preference_delta` and `final_intervention_candidates_table.html` sorts by `beneficial_score` — but the CIs and q-values reported alongside those rankings were computed on the *same* BBQ examples used to rank the features. Post-selection effect sizes and significance are biased upward; the top-ranked feature's "effect" is the maximum of many noisy estimates.

**Why it matters:** "Feature 12345 reduces stereotype preference by Δ, q=0.04" with the CI and q computed on the data that selected the feature is a biased estimate and a misleading inference. A reviewer will catch this immediately.

**Targeted fix:**
- Split BBQ examples into a **selection set** (e.g. 50% stratified by axis × context_condition × question_polarity) and a **confirmation set**. Rank/select features on the selection set; report effect sizes, CIs, and q-values **only** from the confirmation set in `final_intervention_candidates_table.html`, `feature_effect_rankings.csv`, and the per-feature `feature_level_effects.csv`.
- Add a `--selection_fraction` argument (default 0.5) and a `bbq_uid` → split assignment derived deterministically (`hashlib.sha1(bbq_uid) % 100 < 100 * selection_fraction`).
- Report selection-set vs confirmation-set effects side by side so the magnitude of the winner's curse is visible.

### 2.6 [MAJOR] — Multiplicity is inflated by the alpha × position grid

**What's wrong:** The feature-level group cols include `feature_id, layer, alpha, intervention_position, feature_role, feature_contrast_name, mapped_contrast_name, axis_mapped, context_condition, question_polarity, target_identity_id, nontarget_identity_id, feature_estimate_type, steering_direction_label` — a fully crossed grid. A single feature is tested at 6 alphas × 3 positions × 2 contexts × 2 polarities × 2 direction labels = up to 144 highly-correlated hypothesis tests. FDR is applied **within** `(axis_mapped, context_condition, alpha, intervention_position)` strata, so it does not pool over the highly-correlated alpha and position dimensions — and treating the within-stratum tests as independent both inflates the count and mis-estimates FDR.

**Why it matters:** Reported q-values are too liberal where correlation is high (across alphas of the same feature/position) and too conservative for genuinely independent comparisons. Any claim of "N features survived FDR" depends sensitively on this choice.

**Targeted fix:**
- Decide the **unit of inference** explicitly: it should be the *feature* (optionally feature × intervention_position), **not** feature × alpha. Summarize the dose-response into one statistic per feature:
  - Option A (simple, pre-registerable): test the effect at a single pre-registered alpha (e.g. `|alpha| = 2`), once per feature.
  - Option B (uses the grid): fit a sign-consistent monotone-slope statistic per (feature, position) — e.g. the Theil-Sen slope of `mean_stereotype_preference_delta` vs `alpha` — and permutation-test that one statistic per feature.
- Keep the alpha grid in the dose-response *plots* (`dose_response_report`), but do not treat each alpha as a separate hypothesis test.
- Apply FDR across features (and positions, if kept separate), not within-stratum.

### 2.7 [MINOR] — `--smoke` caps bootstrap/permutation; `min_examples = 10` is underpowered

**What's wrong:** `--smoke` caps `--bootstrap_samples` and `--permutation_samples` at 500 (min p ≈ 1/501 ≈ 0.002), and lowers `--min_examples` to 10. A sign-flip permutation test on 10 paired deltas has only 2¹⁰ = 1024 distinct sign assignments; after FDR almost nothing can reach significance. The documented production command in `docs/bbq_steering_pipeline.md` passes `--smoke`, which means **no full-budget run of this analyzer has been executed**.

**Why it matters:** Smoke-budget statistics are not publishable. Any conclusion drawn from the smoke-budget `feature_level_effects.csv` should be treated as preliminary.

**Targeted fix:**
- Drop `--smoke` from the production command. Use `--bootstrap_samples 10000 --permutation_samples 10000`.
- Raise `--min_examples` to a value (≥30, ideally 50) where a permutation test on paired deltas has nontrivial power. If raising `min_examples` thins the table too aggressively, coarsen the grouping (per issue 2.6) so each unit has more examples.
- Consider BCa instead of percentile bootstrap for small n.

### 3.4 / 3.3 / 3.1 / 3.2 [BLOCKER/MAJOR, inherited] — Upstream interventions

The headline numbers `mean_stereotype_preference_delta` etc. are computed on Step 20 outputs that inherit:
- **3.1 [BLOCKER]**: decoder-vector addition is not a feature intervention; the per-feature causal claims here rest on direction interventions, not feature interventions. Until the encode → modify-latent → decode → patch fix lands in Step 20, every row labeled `individual_feature` is actually an individual-decoder-row intervention. (See [Step 20 issue 3.1](20_run_bbq_sae_steering.md) and [Step 13 — `analyze_identity_sae_features.py`](13_analyze_identity_sae_features.md) for the unused helpers.)
- **3.3 [MAJOR]**: `target_identity_last_token` and `stereotype_language_last_token` rows are tagged with positions that may have landed inside answer-option spans. The README claims these positions "answer different causal questions" — only honest if positions land where the names imply.
- **3.4 [MAJOR]**: `mapped_contrast_name` may be a fallback-axis mapping. Restrict headline results here to `mapped_contrast_confidence == exact` rows (after Step 20 carries the column through; see Step 20 issue 3.4).
- **3.2 [MAJOR]**: cross-feature effect-size comparison is on a contaminated scale (uniform alpha vs feature-specific natural scale).

**Targeted fix here, after upstream:** stratify every effect table by `mapped_contrast_confidence` and by `intervention_section` (the section-resolved position name once Step 20 emits it). Make `exact + context-section` the headline subset.

## Rebuild checklist
- [ ] Add `bias_polarity_sign` (`+1` for `neg`, `-1` for `nonneg`) and define `signed_bias_delta = stereotype_preference_delta * bias_polarity_sign`. Refactor `effect_label`, `beneficial_score`, `harmful_score`, `make_rankings`, and `final_candidates_html` to use the signed version. Update `README.md`.
- [ ] Implement a selection / confirmation split keyed by `bbq_uid`. Rank features on the selection set; report effect sizes and q-values on the confirmation set in the final tables.
- [ ] Coarsen the inference grid: collapse alphas into one summary statistic per (feature, position). Apply FDR across features, not within-stratum.
- [ ] Drop `--smoke` and raise bootstrap/permutation budgets to ≥10000; raise `min_examples` accordingly.
- [ ] After Step 20 carries `mapped_contrast_confidence` and `intervention_section` into output rows, stratify every effect table by both and make `exact + context-section` the headline.
- [ ] Defensively recompute argmax-based metrics on length-normalized scores until Step 20 fixes 1.3/2.4 upstream.
- [ ] Preserve the bias taxonomy and the individual_feature/bundle_membership distinction during any refactor — these are paper-grade contributions.
- [ ] Once Step 20 supports `feature_ablate`/`feature_clamp` (Step 20 issue 3.1), add a `feature_estimate_type` value `individual_feature_ablation` so the analyzer can distinguish direction-addition rows from genuine feature interventions and the README can mark headline results as "feature-ablation."

## Notes from the doc audit
- `summarize_effects` builds the per-example aggregate via `groupby("bbq_uid").agg(...)` using mean for every column. If a single `bbq_uid` appears with mixed `alpha`/`position`/`feature_id` rows (which the upstream group-col tuple should prevent), the mean would silently average across them. Add an assertion that within each `(group_cols)` cell, every `bbq_uid` appears at most once before the per-example aggregation.
- The bootstrap rng is `np.random.default_rng(0)` and is reused across every group; this is reproducible but produces slightly correlated bootstrap samples across groups. Pass a fresh rng per group, or pre-generate bootstrap indices once and reuse them across all groups (the better choice for paired statistics).
- `effect_label` thresholds (`small`/`moderate`/`large` defaulting to 0.002 / 0.005 / 0.01 log-prob) are calibrated for `stereotype_preference_delta` magnitudes seen on first-token scoring. Once length-normalized `answer_logprob` or letter scoring lands, recalibrate these thresholds on a baseline pass and document the new values.
- `final_candidates_html` exposes `data-axis` and `data-label` attributes on each `<tr>` for client-side filtering but ships no JavaScript that uses them. A 50-line filter UI on top of those attributes would make the candidates table genuinely usable.
- `validation_summary.csv` is helpful but underpowered — extend it to include the count of features that survived (i) FDR, (ii) FDR + confirmation split, (iii) FDR + confirmation split + `exact`-only mapping. Each filter level should shrink the count, and the table should make that visible.
