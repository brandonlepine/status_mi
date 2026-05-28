# Step 18 — `scripts/prepare_bbq_for_steering.py`

**Stage:** 4 — BBQ steering and feature-level causal analysis
**Runs after:** [Step 17 — `triage_sae_identity_features.py`](17_triage_sae_identity_features.md) (consumes `intervention_candidate_features_triaged.csv`); optionally [Step 18a — `build_few_shot_pool.py`](18a_build_few_shot_pool.md) (consumed via `--few_shot_pool`).
**Feeds into:** [Step 18b — `diagnose_bbq_baseline.py`](18b_diagnose_bbq_baseline.md), [Step 19 — `extract_bbq_token_level_sae_activations.py`](19_extract_bbq_token_level_sae_activations.md), [Step 20 — `run_bbq_sae_steering.py`](20_run_bbq_sae_steering.md), [Step 21 — `build_bbq_sae_feature_cards.py`](21_build_bbq_sae_feature_cards.md), [Step 22 — `analyze_bbq_steering_results.py`](22_analyze_bbq_steering_results.md), [Step 23 — `analyze_bbq_feature_level_causal_effects.py`](23_analyze_bbq_feature_level_causal_effects.md).

## Purpose
Convert raw BBQ JSONL benchmark rows into a steering-ready dataset by (a) normalizing each BBQ example's `category` to a project identity axis, (b) mapping the three `ans0/1/2` group labels to project `identity_id`s, (c) identifying the `unknown`, `stereotyped`, and `nonstereotyped` answer indices, (d) aligning the resulting `(target, nontarget, axis)` triple to an SAE contrast name from the triage CSV, and (e) emitting a fully-formed model prompt. Everything downstream — token-level extraction, steering, and feature-level causal analysis — is keyed off the `bbq_uid` and the `axis_mapped` / `mapped_contrast_name` / `mapped_contrast_confidence` columns this script writes.

## Inputs
- `data/bbq/data/*.jsonl` — one file per BBQ category. Per-row fields used: `example_id`, `question_index`, `question_polarity` (`neg`/`nonneg`), `context_condition` (`ambig`/`disambig`), `category`, `answer_info` (mapping `ans0/1/2` → `[text, group_label]`), `additional_metadata.stereotyped_groups`, `context`, `question`, `ans0/1/2`, `label`.
- `data/bbq_identity_normalized_forms.csv` — identity alias source. The script reads canonical forms (`canonical_label`, `adj_form`, `noun_form`, ..., `has_form`) and the semicolon-delimited `aliases` column to build a `text → identity_id` table; `MANUAL_ALIASES` is layered on top.
- `results/.../triage/intervention_candidate_features_triaged.csv` — used only to derive the `contrast_name → axis` map so BBQ pairs can be matched to available SAE contrast directions.
- *Optional:* `data/bbq/few_shot_pool.json` from [Step 18a](18a_build_few_shot_pool.md), passed via `--few_shot_pool`. When set, the pool's example_ids are excluded from output and the formatted prefix is prepended to every prompt.

## Outputs
- `prepared/bbq_prepared_examples.csv` and `.parquet` — one row per BBQ example with `bbq_uid`, `axis_mapped`, `context_condition`, `question_polarity`, `prompt`, `few_shot_prefix` (empty when `--few_shot_pool` is not set, otherwise the formatted prefix that was prepended to `prompt`), `unknown_answer_idx`, `stereotyped_answer_idx`, `nonstereotyped_answer_idx`, `target_identity_id`, `nontarget_identity_id`, `mapped_contrast_name`, `mapped_contrast_confidence` (∈ `{exact, alias, fallback_axis, unmapped}`), `polarity_role`, and a `notes` semicolon list.
- `prepared/bbq_mapping_diagnostics.csv` — rows with `notes` (missing-unknown, missing-stereotype, unmapped-contrast).
- `prepared/bbq_contrast_mapping.csv` — the SAE-contrast registry derived from triage.
- `prepared/bbq_prepare_summary.csv` — mapping-confidence counts and per-category coverage; logs a warning when overall mapping rate < 70%.
- `prepared/bbq_prepare_config.json`, `prepared/logs/prepare_bbq.log`.

## Key implementation details
- **AXIS_MAP** identity-maps single-axis categories (e.g. `ses` → `socioeconomic_status`). The intersectional categories `race_x_gender` and `race_x_ses` are NOT in `AXIS_MAP` — they're handled separately by `--intersectional_handling` (default `drop`; see audit 4.2 below, closed 2026-05-28 in commit `b189aef`).
- **Identity alias table** is the union of (i) every non-empty surface form column for each identity row, (ii) each entry in the semicolon-delimited `aliases` column, (iii) the hardcoded `MANUAL_ALIASES` dict (49 entries grouped by axis as of audit 4.4 / 4.1 BBQ fix in commit `26998ec`; every target is validated against the identity-forms CSV at startup via `validate_manual_aliases`, which raises `ValueError` on any missing target). Matching uses `norm_text` (lowercase, underscores/hyphens to spaces, alphanumerics only). If exact lookup fails, the script tries (a) whitespace-stripped matching, then (b) decomposing compound labels like `F-Black` via `identity_components` and looking each piece up individually.
- **Role assignment** (`find_answer_indices`): an answer is `is_unknown` if it normalizes to a member of `UNKNOWN_ALIASES` (`"cannot be determined"`, `"not answerable"`, etc.). It is `is_stereo` if its mapped `identity_id`, normalized label, or any component identity is in the BBQ row's `stereotyped_groups`. Target = first stereotyped non-unknown; nontarget = first non-stereotyped non-unknown.
- **Contrast mapping** (`map_contrast`) tries exact identity-pair match against the triage's contrast registry first (returns confidence `exact`), then any contrast on the same axis that touches either identity (`fallback_axis`), then any same-axis contrast at all (still `fallback_axis`), then `""/unmapped`. Note: `confidence == "alias"` is never produced by this code — the column is defined as `exact | fallback_axis | unmapped` despite downstream readers expecting four values.
- **Prompt format** (`prompt_for`):
  ```
  Context:
  {context}

  Question:
  {question}

  Answer choices:
  A. {ans0}
  B. {ans1}
  C. {ans2}

  Answer:
  ```
  This ends with `"Answer:"` — a QA cue intended to elicit the answer letter, but downstream scoring (`run_bbq_sae_steering.py:score_first_token`) targets the first token of the *answer text*, not the letter (see issue 1.3 in Step 20).
- `question_polarity` is preserved in the output but is **never folded into the bias polarity** here or downstream (see issue 4.3 in Step 23).

## Issues & Opportunities

### 1.2 [MAJOR] — Base model vs. a multiple-choice QA benchmark (PARTIAL FIX LANDED 2026-05-26)

**Status:** Code landed in two commits. What remains is to run both prompt modes on RunPod and decide based on the diff.

- [Step 18a — `build_few_shot_pool.py`](18a_build_few_shot_pool.md) writes `data/bbq/few_shot_pool.json` (K=4, seeded, stratified across (ambig/disambig × neg/nonneg), distinct categories).
- This script accepts `--few_shot_pool data/bbq/few_shot_pool.json`. When set, the four exemplar `(source_file, example_id)` keys are excluded from the prepared rows and a ~1.5kB formatted prefix is prepended to every remaining prompt. The prefix is also recorded in the new `few_shot_prefix` column so it can be audited or stripped.
- [Step 18b — `diagnose_bbq_baseline.py`](18b_diagnose_bbq_baseline.md) consumes the prepared parquet and measures all three audit-required diagnostics: (i) total mass on the three options (both letters and answer-text first tokens), (ii) BBQ accuracy + polarity-signed bias score, (iii) argmax-vs-greedy agreement rate. Run it on both zero-shot and few-shot outputs and diff the JSONs.

**What's wrong (original audit):** This script builds a zero-shot, `"Answer:"`-terminated prompt that is then scored against the `meta-llama/Llama-3.1-8B` *base* model (chosen so the OpenMOSS LlamaScope SAEs apply). Base models are weak at, and largely off-distribution for, BBQ-style multiple-choice QA. There is a real risk that the model places ~1–2% total probability mass on `{ans0, ans1, ans2}` and ~98% on free-form continuation text, in which case the entire steering signal is measured in a degenerate regime.

**Why it matters:** Every downstream "bias margin", "stereotype preference delta", and "feature is causally implicated" claim presumes the model actually treats the three options as the answer space. If it does not, the deltas are noise on a small slice of probability.

**Remaining work:**
- Run [Step 18b](18b_diagnose_bbq_baseline.md) on the zero-shot and few-shot prepared parquets on RunPod (see the [side-by-side invocation order](18b_diagnose_bbq_baseline.md#suggested-invocation-order-zero-shot-vs-few-shot-side-by-side)).
- Decide which prompt mode to use for steering based on the diff (decision rule and metric table are in [Step 18b](18b_diagnose_bbq_baseline.md)).
- Re-prepare with the chosen mode, then rerun [Step 19](19_extract_bbq_token_level_sae_activations.md) and [Step 20](20_run_bbq_sae_steering.md).
- Record the chosen mode and headline numbers in the methods writeup as the audit-required "baseline precondition" section.

### 3.4 [MAJOR] — BBQ→SAE contrast mapping silently uses axis-fallback (FIX LANDED 2026-05-27)

**Status:** Closed in commit `56a5f7e` ([Step 20](20_run_bbq_sae_steering.md)). The mapping logic in this script (`map_contrast`) is unchanged — it still emits the three labels `{exact, fallback_axis, unmapped}`. The downstream steering runner is what flipped:

- `--include_unmapped` boolean replaced with `--mapping_confidence_filter {exact, exact_and_fallback, all}` (default `exact`). Headline runs silently drop `fallback_axis` rows.
- `mapped_contrast_confidence` is stamped on every steering output row, so `analyze_bbq_feature_level_causal_effects.py` can stratify any effect table by mapping confidence with a `groupby`.

**Optional follow-up (still open, lower priority):** emit a `bbq_prepare_confidence_breakdown.csv` from this script listing each `(category_raw, axis_mapped) → {exact: N, fallback_axis: M, unmapped: K}` so the operator can audit coverage at prepare-time without running the downstream steering. The runner already prints the breakdown to stdout under the new filter, so this is convenience, not correctness.

**Original audit (preserved):** `map_contrast` falls back from exact-pair matching to "any same-axis contrast that touches either identity", labeling the result `fallback_axis`. A BBQ item about `race_arab vs race_white` could therefore be associated with features selected for `race_black vs race_white`. The prior `run_bbq_sae_steering.py` default kept confidence ∈ `{exact, alias, fallback_axis}` and `analyze_bbq_feature_level_causal_effects.py` treated `mapped_contrast_name` as the relevant contrast — feature-to-example specificity broken, data looked clean. "Feature X is implicated in bias for *this* identity contrast" is the main causal claim, and the fallback path broke that specificity.

### 4.1 [MAJOR] — Contrast lists reference identities that do not exist — silently skipped (BBQ SIDE FIX LANDED 2026-05-28)

**Status:** BBQ side closed alongside audit 4.4 in commit `26998ec`. Geometry side (the `CONTRASTS` literal across the geometry scripts) was closed earlier in commit `1e242c9`.

**What landed (BBQ side):**
- Broken `MANUAL_ALIASES` targets repointed to canonical identity IDs that exist in `bbq_identity_normalized_forms.csv`: `ses_low_income → ses_low` (for "low ses" / "low socioeconomic status" / "low income" / "lowses"), `ses_low_income → ses_poor` (for "poor", which has its own identity), `ses_high_socioeconomic_status → ses_high` (for "high socioeconomic status" / "highses").
- Aliases pointing at identities that simply don't exist were removed entirely: `age_old` / `age_nonold` / `non old` (no age axis in the dataset), `nationality_asia_pacific` / `nationality_african` / `nationality_european` (aggregate continents, not per-country IDs). BBQ rows that previously mentioned these now fall to `mapped_contrast_confidence=unmapped` and are filtered at Step 20 under the audit-3.4 default (`--mapping_confidence_filter exact`).
- New `validate_manual_aliases(identity_meta, logger)` called immediately after `load_identity_aliases()` in `main()`. **Raises `ValueError`** on any missing target, with a per-target `ERROR`-level log line listing the aliases that point at it. Silent drops were the original bug; raising loudly means a future regression (a new alias pointing at a typo'd identity_id) can't slip through.
- `bbq_prepare_summary.csv` now records `manual_aliases_n_total`, `manual_aliases_n_distinct_targets`, and `manual_aliases_n_missing_targets` so the durability of the alias table is visible without parsing source.

**Original audit (preserved):** `MANUAL_ALIASES` previously mapped to multiple identity IDs that did not exist in `bbq_identity_normalized_forms.csv` — `ses_low_income`, `ses_high_socioeconomic_status`, `age_old`, `age_nonold`, `nationality_asia_pacific`, `nationality_african`, `nationality_european`. There was also no `age` axis in the identity CSV at all, so any Age-axis BBQ row that resolved to `age_old`/`age_nonold` received an ID that failed the triage contrast registry lookup and silently mapped to `unmapped`. Paper claims like "we cover N contrasts" were out of sync with what actually ran.

### 4.2 [MAJOR] — Intersectional BBQ categories are flattened to a single axis (FIX LANDED 2026-05-28)

**Status:** Closed in commit `b189aef` with the audit's path (b) — explicit exclusion. Path (a) — first-class compound contrasts with their own directions and SAE features — needs templated compound prompts that the geometry pipeline doesn't produce, plus a compound-contrast registry; that's a separate research workstream.

**What landed:**
- `race_x_gender` and `race_x_ses` removed from `AXIS_MAP` so the silent flatten path is gone. New module-level constants `INTERSECTIONAL_CATEGORIES = {"race_x_gender", "race_x_ses"}` + `INTERSECTIONAL_AXIS_FLATTEN_TO` record the prior flatten target for the opt-in legacy mode.
- New CLI `--intersectional_handling {drop, axis_flatten}` (default `drop`). Default behavior: intersectional rows are excluded; per-category dropped counts are logged to stdout AND added to `bbq_prepare_summary.csv` as `n_intersectional_dropped_*` metrics. `axis_flatten` preserves the legacy "collapse to race_ethnicity" behavior, but every flattened row is stamped `is_intersectional=True` so downstream analyzers can stratify or drop them.
- New `is_intersectional` column on `bbq_prepared_examples.parquet` (always present; `False` for non-intersectional rows). Under the audit-3.4 default mapping filter (`--mapping_confidence_filter exact`), intersectional rows passing through under `axis_flatten` are still filtered at Step 20 because no contrast in the registry matches an intersectional pair — but the `is_intersectional` flag makes the path explicit rather than relying on the unmapped-fallback to silently drop them.
- New helper `resolve_intersectional(category, handling) -> (axis_or_None, is_intersectional)` exported alongside the constants for downstream consumers.
- Synthetic validation: routing correct across both intersectional category names and a sample of non-intersectional ones; CLI default = `drop`; AXIS_MAP no longer contains `race_x_*`.

**Path (a) — future work:** Genuine intersectional handling would require:
1. Adding templated identity prompts that vary BOTH axes simultaneously (e.g. `"This person is {gender_form} and {race_form}."`) so the geometry pipeline can fit compound contrast directions.
2. Extending `contrast_registry.py` with compound contrasts like `(race_black_x_gender_female, race_white_x_gender_male)`.
3. Extending Step 7's `run_contrasts` / `run_contrast_probes` to compute compound directions (the math is unchanged — `compute_direction` is identity-agnostic — but the contrast pair iteration would need a compound case).
4. Extending Step 18's `find_answer_indices` to map compound BBQ labels (`F-Black`) to compound identity IDs rather than picking a single component.
5. Threading compound contrasts through `--mapping_confidence_filter`. This is a substantial paper-extension and is recorded as a follow-up, not the current commit.

**Original audit (preserved):** `AXIS_MAP` collapsed `race_x_gender` and `race_x_ses` to `race_ethnicity`. `identity_components` splits compound labels like `F-Black` and `lowSES-Hispanic`, and `choose_identity_for_role` picks a single component as the target identity. The intersectional structure was discarded before steering ever ran. Intersectionality is central to the marginalized-identities literature this paper claims to address; flattening Race×Gender to "race" loses the most interesting BBQ cases and can mislabel the stereotype-target answer when the BBQ stereotype actually targets the compound identity.

### 4.4 [MINOR] — `MANUAL_ALIASES` has dozens of duplicate `"nondisabled"` keys (FIX LANDED 2026-05-28)

**Status:** Closed in commit `26998ec` (paired with the BBQ side of audit 4.1; see above).

**What landed:**
- `MANUAL_ALIASES` rewritten from 91 literal entries (56 distinct — the runtime dict was silently deduplicating ~35 duplicate `"nondisabled": "disability_nondisabled"` lines, plus a couple of `"non disabled"` duplicates) to 49 literal entries, all distinct, grouped by axis with comments.
- The "unit test" the audit recommended is implemented as the runtime invariant in `validate_manual_aliases()` — described in the 4.1 section above — rather than a separate test file. That validator runs at every `prepare_bbq_for_steering.py` startup, so any future regression of either kind (re-introduced duplicate keys or a broken target) raises immediately rather than slipping through.
- Original audit (preserved): Lines 47-85 of the prior script literally repeated `"nondisabled": "disability_nondisabled"` ~30 times — a copy-paste artifact. The dict deduplicated at runtime, but the file was clearly unreviewed.

### 4.3 [BLOCKER, heads-up for downstream] — `question_polarity` is preserved but never used as a bias-polarity sign

**What's wrong (in this script's role):** This script faithfully records `question_polarity` ∈ `{neg, nonneg}` and exposes it in `bbq_prepared_examples.parquet`. The actual bug is downstream in `analyze_bbq_feature_level_causal_effects.py`, which defines `stereotype_preference_delta` and the `effect_label` taxonomy without folding the polarity sign in. The full discussion lives in [Step 23](23_analyze_bbq_feature_level_causal_effects.md).

**Why it matters here:** Anything *this* script could pre-compute to make the polarity sign easier to use downstream — e.g. an additional `bias_polarity_sign ∈ {+1, -1}` column derived as `+1 if question_polarity == "neg" else -1` — would let the downstream analyzer multiply through without rewriting its delta definitions.

**Targeted fix:** Add a `bias_polarity_sign` column to `bbq_prepared_examples.parquet`: `+1` for `neg`, `-1` for `nonneg`, `0`/`NaN` otherwise. Document in the column dictionary that downstream `stereotype_preference_delta` must be multiplied by `bias_polarity_sign` to obtain a polarity-correct "bias-direction" quantity.

## Rebuild checklist
- [x] Validate every value in `MANUAL_ALIASES` against `bbq_identity_normalized_forms.csv` at startup; promote missing-ID skips from silent drops to `ERROR`-level log lines that are counted in `bbq_prepare_summary.csv`. *(Done 2026-05-28: commit `26998ec` — `validate_manual_aliases()` raises `ValueError` on any missing target with per-target ERROR log lines; `manual_aliases_n_total` / `manual_aliases_n_distinct_targets` / `manual_aliases_n_missing_targets` recorded in `bbq_prepare_summary.csv`.)*
- [x] Remove the duplicate `"nondisabled"` entries and either delete or fix the entries that point to non-existent IDs. *(Done 2026-05-28: commit `26998ec` — 91→49 literal entries (all distinct); ses_low_income / ses_high_socioeconomic_status repointed; age_* and continent-aggregate nationality aliases removed.)*
- [x] Decide intersectional policy. (4.2 FIX LANDED 2026-05-28 in commit `b189aef`: path (b) — `--intersectional_handling drop` is the new default, `axis_flatten` opt-in preserves legacy behavior; `is_intersectional` column added. Path (a) — compound contrasts as first-class objects — recorded as future work in the 4.2 section above.)
- [ ] Add `bias_polarity_sign` column derived from `question_polarity` so downstream can sign the bias delta.
- [ ] Add `bbq_prepare_confidence_breakdown.csv` listing per `(category_raw, axis_mapped)` the count of rows at each `mapped_contrast_confidence` level.
- [ ] Add a one-page baseline diagnostic (answer-option mass, BBQ accuracy/bias, argmax-matches-greedy) computed on a stratified sample of the prepared prompts.
- [ ] Document the discrepancy between scoring target (answer text vs answer letter) somewhere visible to downstream consumers — or change the prompt format to make the letter the natural continuation (see Step 20, issue 1.3).
- [ ] Replace the redundant local `MANUAL_ALIASES`/identity-axis logic with the shared `status_mi/common.py` contrast registry once that module is built (5.10).

## Notes from the doc audit
- `mapped_contrast_confidence` advertises four values (`exact`, `alias`, `fallback_axis`, `unmapped`) per the operational doc and per `run_bbq_sae_steering.py`'s default keep-list, but `map_contrast` in this script only ever returns three (`exact`, `fallback_axis`, `unmapped`). Either delete `alias` from the downstream filter or implement the alias path explicitly (e.g. exact match after applying `MANUAL_ALIASES` could be marked `alias` to distinguish from canonical-form `exact`).
- `identity_axis` has an `age` branch (returns `"age"`) even though the identity CSV has no age axis. This guarantees that any successful alias hit on `age_old`/`age_nonold` produces an `axis_mapped == "age"` row that no SAE feature set can match (no kept features will have `axis == "age"`), so those rows are always dropped by axis matching downstream — silently. Worth either adding the axis or pruning the branch.
- `bbq_prepared_examples.partial.csv` checkpoints are written but never cleaned up at the end of a successful run; consider deleting on completion.
