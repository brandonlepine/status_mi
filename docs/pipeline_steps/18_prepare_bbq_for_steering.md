# Step 18 — `scripts/prepare_bbq_for_steering.py`

**Stage:** 4 — BBQ steering and feature-level causal analysis
**Runs after:** [Step 17 — `triage_sae_identity_features.py`](17_triage_sae_identity_features.md) (consumes `intervention_candidate_features_triaged.csv`)
**Feeds into:** [Step 19 — `extract_bbq_token_level_sae_activations.py`](19_extract_bbq_token_level_sae_activations.md), [Step 20 — `run_bbq_sae_steering.py`](20_run_bbq_sae_steering.md), [Step 21 — `build_bbq_sae_feature_cards.py`](21_build_bbq_sae_feature_cards.md), [Step 22 — `analyze_bbq_steering_results.py`](22_analyze_bbq_steering_results.md), [Step 23 — `analyze_bbq_feature_level_causal_effects.py`](23_analyze_bbq_feature_level_causal_effects.md).

## Purpose
Convert raw BBQ JSONL benchmark rows into a steering-ready dataset by (a) normalizing each BBQ example's `category` to a project identity axis, (b) mapping the three `ans0/1/2` group labels to project `identity_id`s, (c) identifying the `unknown`, `stereotyped`, and `nonstereotyped` answer indices, (d) aligning the resulting `(target, nontarget, axis)` triple to an SAE contrast name from the triage CSV, and (e) emitting a fully-formed model prompt. Everything downstream — token-level extraction, steering, and feature-level causal analysis — is keyed off the `bbq_uid` and the `axis_mapped` / `mapped_contrast_name` / `mapped_contrast_confidence` columns this script writes.

## Inputs
- `data/bbq/data/*.jsonl` — one file per BBQ category. Per-row fields used: `example_id`, `question_index`, `question_polarity` (`neg`/`nonneg`), `context_condition` (`ambig`/`disambig`), `category`, `answer_info` (mapping `ans0/1/2` → `[text, group_label]`), `additional_metadata.stereotyped_groups`, `context`, `question`, `ans0/1/2`, `label`.
- `data/bbq_identity_normalized_forms.csv` — identity alias source. The script reads canonical forms (`canonical_label`, `adj_form`, `noun_form`, ..., `has_form`) and the semicolon-delimited `aliases` column to build a `text → identity_id` table; `MANUAL_ALIASES` is layered on top.
- `results/.../triage/intervention_candidate_features_triaged.csv` — used only to derive the `contrast_name → axis` map so BBQ pairs can be matched to available SAE contrast directions.

## Outputs
- `prepared/bbq_prepared_examples.csv` and `.parquet` — one row per BBQ example with `bbq_uid`, `axis_mapped`, `context_condition`, `question_polarity`, `prompt`, `unknown_answer_idx`, `stereotyped_answer_idx`, `nonstereotyped_answer_idx`, `target_identity_id`, `nontarget_identity_id`, `mapped_contrast_name`, `mapped_contrast_confidence` (∈ `{exact, alias, fallback_axis, unmapped}`), `polarity_role`, and a `notes` semicolon list.
- `prepared/bbq_mapping_diagnostics.csv` — rows with `notes` (missing-unknown, missing-stereotype, unmapped-contrast).
- `prepared/bbq_contrast_mapping.csv` — the SAE-contrast registry derived from triage.
- `prepared/bbq_prepare_summary.csv` — mapping-confidence counts and per-category coverage; logs a warning when overall mapping rate < 70%.
- `prepared/bbq_prepare_config.json`, `prepared/logs/prepare_bbq.log`.

## Key implementation details
- **AXIS_MAP** (lines 26–38) flattens `race_x_gender` and `race_x_ses` to `race_ethnicity`; everything else is identity-mapped, including `ses` → `socioeconomic_status`.
- **Identity alias table** is the union of (i) every non-empty surface form column for each identity row, (ii) each entry in the semicolon-delimited `aliases` column, (iii) the hardcoded `MANUAL_ALIASES` dict. Matching uses `norm_text` (lowercase, underscores/hyphens to spaces, alphanumerics only). If exact lookup fails, the script tries (a) whitespace-stripped matching, then (b) decomposing compound labels like `F-Black` via `identity_components` and looking each piece up individually.
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

### 1.2 [MAJOR] — Base model vs. a multiple-choice QA benchmark

**What's wrong:** This script builds a zero-shot, `"Answer:"`-terminated prompt that is then scored against the `meta-llama/Llama-3.1-8B` *base* model (chosen so the OpenMOSS LlamaScope SAEs apply). Base models are weak at, and largely off-distribution for, BBQ-style multiple-choice QA. There is a real risk that the model places ~1–2% total probability mass on `{ans0, ans1, ans2}` and ~98% on free-form continuation text, in which case the entire steering signal is measured in a degenerate regime.

**Why it matters:** Every downstream "bias margin", "stereotype preference delta", and "feature is causally implicated" claim presumes the model actually treats the three options as the answer space. If it does not, the deltas are noise on a small slice of probability.

**Targeted fix:** Add a baseline diagnostic to `bbq_prepare_summary.csv` (or a sibling script) that, on a sample of the prepared prompts, reports (i) total `p(ans0) + p(ans1) + p(ans2)` mass, (ii) standard BBQ accuracy + bias score in this format, (iii) how often the per-option argmax matches the model's greedy continuation. Consider adding an alternative `prompt_for_fewshot` mode that prepends 3–5 BBQ exemplars and storing it in a parallel column so a few-shot run is possible without rewriting the dataset.

### 3.4 [MAJOR] — BBQ→SAE contrast mapping silently uses axis-fallback

**What's wrong:** `map_contrast` falls back from exact-pair matching to "any same-axis contrast that touches either identity", labeling the result `fallback_axis`. A BBQ item about `race_arab vs race_white` can therefore be associated with features selected for `race_black vs race_white`. `run_bbq_sae_steering.py` keeps confidence ∈ `{exact, alias, fallback_axis}` by default and `analyze_bbq_feature_level_causal_effects.py` treats `mapped_contrast_name` as the relevant contrast.

**Why it matters:** "Feature X is implicated in bias for *this* identity contrast" is the main causal claim. `fallback_axis` breaks the feature-to-example specificity needed to support that claim while leaving the data looking clean.

**Targeted fix:** Make `mapped_contrast_confidence` a first-class headline filter. In this script, also emit a `bbq_prepare_confidence_breakdown.csv` listing each `(category_raw, axis_mapped) → {exact: N, fallback_axis: M, unmapped: K}` so the user can audit coverage at prepare-time. Downstream (Step 20, Step 23), restrict headline results to `exact` and stratify everything else by mapping confidence.

### 4.1 [MAJOR] — Contrast lists reference identities that do not exist — silently skipped

**What's wrong:** `MANUAL_ALIASES` maps to multiple identity IDs that do not exist in `bbq_identity_normalized_forms.csv`: `ses_low_income`, `ses_high_socioeconomic_status`, `age_old`, `age_nonold`, `nationality_asia_pacific`, `nationality_african`, `nationality_european`. Audit `sexuality_*` and `appearance_obese` as well. There is also **no `age` axis** in the identity CSV at all, so any Age-axis BBQ row that resolves to `age_old`/`age_nonold` will receive an ID that fails the triage contrast registry lookup and silently maps to `unmapped`.

**Why it matters:** BBQ rows whose stereotyped answer was the only way an axis got into the run will be silently dropped from the contrast-mapped set, leaving paper claims like "we cover N contrasts" out of sync with what actually ran. The Age category in particular is dead unless the identity CSV is extended.

**Targeted fix:** Add a startup-time validation step: load `aliases | identity_meta` and assert every value in `MANUAL_ALIASES` exists as `identity_id` in `identity_meta`. Fail (or log `ERROR`-level warning per missing ID and count them at the end) if not. Decide whether to (a) add the missing identities to `bbq_identity_normalized_forms.csv` with proper surface forms or (b) remove the orphan aliases entirely.

### 4.2 [MAJOR] — Intersectional BBQ categories are flattened to a single axis

**What's wrong:** `AXIS_MAP` collapses `race_x_gender` and `race_x_ses` to `race_ethnicity`. `identity_components` splits compound labels like `F-Black` and `lowSES-Hispanic`, and `choose_identity_for_role` picks a *single* component as the target identity. The intersectional structure is discarded before steering ever runs.

**Why it matters:** Intersectionality is central to the marginalized-identities literature this paper claims to address. Flattening Race×Gender to "race" loses the most interesting BBQ cases and can mislabel the stereotype-target answer when the BBQ stereotype actually targets the compound identity.

**Targeted fix:** Choose explicitly: either (a) handle intersectional contrasts as first-class objects (preserve the compound `axis_mapped = "race_x_gender"`, build compound identity directions and intersectional contrast features upstream, and let downstream steering address them as new contrast names), or (b) drop `Race_x_*` from the default `--categories` list and document the exclusion. Do not silently flatten.

### 4.4 [MINOR] — `MANUAL_ALIASES` has dozens of duplicate `"nondisabled"` keys

**What's wrong:** Lines ~47–85 of the script literally repeat `"nondisabled": "disability_nondisabled"` ~30 times — a copy-paste artifact. The dict deduplicates so it is harmless at runtime, but the file is clearly unreviewed.

**Why it matters:** Signals to a reviewer that the alias table was not audited (which is also the substance of 4.1).

**Targeted fix:** Delete the duplicates. Add a unit test (or an inline assertion) that `len(MANUAL_ALIASES) == len(set(MANUAL_ALIASES))`. While there, audit every value against the identity CSV (4.1).

### 4.3 [BLOCKER, heads-up for downstream] — `question_polarity` is preserved but never used as a bias-polarity sign

**What's wrong (in this script's role):** This script faithfully records `question_polarity` ∈ `{neg, nonneg}` and exposes it in `bbq_prepared_examples.parquet`. The actual bug is downstream in `analyze_bbq_feature_level_causal_effects.py`, which defines `stereotype_preference_delta` and the `effect_label` taxonomy without folding the polarity sign in. The full discussion lives in [Step 23](23_analyze_bbq_feature_level_causal_effects.md).

**Why it matters here:** Anything *this* script could pre-compute to make the polarity sign easier to use downstream — e.g. an additional `bias_polarity_sign ∈ {+1, -1}` column derived as `+1 if question_polarity == "neg" else -1` — would let the downstream analyzer multiply through without rewriting its delta definitions.

**Targeted fix:** Add a `bias_polarity_sign` column to `bbq_prepared_examples.parquet`: `+1` for `neg`, `-1` for `nonneg`, `0`/`NaN` otherwise. Document in the column dictionary that downstream `stereotype_preference_delta` must be multiplied by `bias_polarity_sign` to obtain a polarity-correct "bias-direction" quantity.

## Rebuild checklist
- [ ] Validate every value in `MANUAL_ALIASES` against `bbq_identity_normalized_forms.csv` at startup; promote missing-ID skips from silent drops to `ERROR`-level log lines that are counted in `bbq_prepare_summary.csv`.
- [ ] Remove the duplicate `"nondisabled"` entries and either delete or fix the entries that point to non-existent IDs (`ses_low_income`, `ses_high_socioeconomic_status`, `age_old`, `age_nonold`, `nationality_asia_pacific`, `nationality_african`, `nationality_european`).
- [ ] Decide intersectional policy. If keeping `race_x_*` as first-class, stop collapsing them in `AXIS_MAP`, propagate the compound axis through the schema, and add compound identity directions upstream. If excluding, default `--categories` to omit them and assert exclusion.
- [ ] Add `bias_polarity_sign` column derived from `question_polarity` so downstream can sign the bias delta.
- [ ] Add `bbq_prepare_confidence_breakdown.csv` listing per `(category_raw, axis_mapped)` the count of rows at each `mapped_contrast_confidence` level.
- [ ] Add a one-page baseline diagnostic (answer-option mass, BBQ accuracy/bias, argmax-matches-greedy) computed on a stratified sample of the prepared prompts.
- [ ] Document the discrepancy between scoring target (answer text vs answer letter) somewhere visible to downstream consumers — or change the prompt format to make the letter the natural continuation (see Step 20, issue 1.3).
- [ ] Replace the redundant local `MANUAL_ALIASES`/identity-axis logic with the shared `status_mi/common.py` contrast registry once that module is built (5.10).

## Notes from the doc audit
- `mapped_contrast_confidence` advertises four values (`exact`, `alias`, `fallback_axis`, `unmapped`) per the operational doc and per `run_bbq_sae_steering.py`'s default keep-list, but `map_contrast` in this script only ever returns three (`exact`, `fallback_axis`, `unmapped`). Either delete `alias` from the downstream filter or implement the alias path explicitly (e.g. exact match after applying `MANUAL_ALIASES` could be marked `alias` to distinguish from canonical-form `exact`).
- `identity_axis` has an `age` branch (returns `"age"`) even though the identity CSV has no age axis. This guarantees that any successful alias hit on `age_old`/`age_nonold` produces an `axis_mapped == "age"` row that no SAE feature set can match (no kept features will have `axis == "age"`), so those rows are always dropped by axis matching downstream — silently. Worth either adding the axis or pruning the branch.
- `bbq_prepared_examples.partial.csv` checkpoints are written but never cleaned up at the end of a successful run; consider deleting on completion.
