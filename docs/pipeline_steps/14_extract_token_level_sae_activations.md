# Step 14 — `extract_token_level_sae_activations.py`

**Stage:** 3 — Identity-selective SAE feature analysis
**Runs after:** `analyze_identity_sae_features.py` (selects which features to extract from)
**Feeds into:** `build_sae_feature_cards.py` (token heat-maps), `triage_sae_identity_features.py` (token-level localization metrics → `feature_localization_type`)

## Purpose

For a curated subset of identity-related SAE features, re-run the model on the prompts where each feature fired most strongly at the final token and record **per-token** SAE activations. Produces the data that lets every downstream artifact answer "where in the prompt does this feature actually activate?" — identity span, sentence-final, template context, or diffuse. This is also the **only** place in the identity pipeline where token character offsets are computed and aligned to identity spans, so the data needed to fix issue 1.1 (final-token vs identity-span geometry) already exists here.

## Inputs

- `results/sae_identity/.../analysis/intervention_candidate_features.csv`, `feature_selectivity_alignment_joined.csv`, `feature_identity_selectivity.csv` — drive feature selection (or `--features` for an explicit list).
- `results/sae_identity/.../layer_XX/feature_indices_top64.npy` and `feature_values_top64.npy` — used to pick the top-activating prompts per feature.
- `results/activations/.../metadata.csv` — `prompt`, `prompt_id`, `form_used`, `identity_id`, `family`, `template_id`, etc.
- Llama-3.1-8B model directory + OpenMOSS SAE checkpoint for the selected layer(s).

## Outputs (under `<output_dir>/token_level/`, default `.../feature_cards/token_level/`)

- `selected_features.json` — `{layer: [feature_ids]}` chosen for extraction.
- `layer_XX/token_feature_activations.csv` — one row per `(prompt, feature, token)` with `token_str`, `token_start_char`, `token_end_char`, `token_feature_activation`, `is_identity_span_token`, `identity_span_match_status`, `feature_localization_type`, `final_token_feature_activation`, `max_token_activation`, `max_identity_span_activation`, `mean_identity_span_activation`, identity/axis/family/template fields.
- `layer_XX/feature_top_tokens.csv` — top-200 non-special tokens per feature, used by triage's token-entropy/template-word metrics.
- `layer_XX/run_config.json`.

## Key implementation details

- Feature selection (`select_features`, line 90): when `--features` is not given, it unions four sources: top-N by `combined_score` per contrast (from `intervention_candidate_features.csv`), top-N by `combined_score`, top-N by signed `cosine_with_direction` (positive and negative), and top-N by `|cohens_d|` per contrast (from `feature_selectivity_alignment_joined.csv`), plus top-N by `|cohens_d|` per identity (from `feature_identity_selectivity.csv`). Note this propagates the `combined_score` double-weighting (5.3) and the post-selection bias (2.5) from the upstream script.
- Prompt selection per feature (`select_prompt_rows`, line 135): top `--max_prompts_per_feature` (default 200) by final-token activation, with a `positive` fallback to "any nonzero" then to "all rows" if no rows have a positive final-token value.
- Identity span localization (`find_identity_span`, line 151): regex search for the realized `form_used` in the prompt; falls back to a whitespace/punctuation-normalized form. Records `identity_span_match_status ∈ {exact, normalized, failed}`. This is the data needed by issue 1.1.
- Per-token SAE encoding (`encode_selected_features`, line 177): `relu((hidden - b_dec) @ w_enc[:, features] + b_enc[features])`. Same generic-loader convention as `encode_identity_saes.py`; inherits the unverified preprocessing convention flagged by issue 1.4.
- Per-prompt `feature_localization_type` is assigned by `common.classify_feature_localization` (audit 5.10 instance closed 2026-05-27 in commit `402731f` — same helper is called from `build_sae_feature_cards.py:exemplar_prompt_table`):
  - `identity_span_local` if `max_span_activation ≥ 0.7 · max_token_activation`,
  - `final_token_integrated` if `final_token_activation ≥ 0.7 · max_token_activation`,
  - `template_context` if max activation exists but didn't land in span or final token,
  - `diffuse_or_unclear` if max activation is zero.
  Threshold default is `common.DEFAULT_LOCALIZATION_THRESHOLD = 0.7` — the same constant the triage's `identity_span_local_threshold` / `final_token_integrated_threshold` are matched against.
- `resume` works at prompt-id granularity by counting how many distinct `feature_id`s have been written per `prompt_id` (`processed_prompt_ids`, line 197) and skipping prompts that already have all selected features.

## Issues & Opportunities

### 1.1 (enabler) [BLOCKER] — Identity-span data already exists here

**What's wrong:** The identity-geometry pipeline characterizes the residual stream at the **final non-padding token**, which is almost always the sentence-final period. The entire claim that "identity is linearly decodable / has a shared subspace / etc." is therefore a claim about period-token geometry. The fix requires re-extracting activations at the **identity-span tokens** (last or mean-pooled) and comparing.

**Why it matters in this file:** This script is the one place in the pipeline that already locates identity spans by regex on the raw prompt and aligns them to tokenizer offsets. Re-using the `find_identity_span` logic in `extract_identity_activations.py` would unblock the 1.1 fix immediately. There is no need to re-derive the span-matching code in another file.

**Targeted fix:** Refactor `find_identity_span` (line 151) into a shared helper (see 5.10 in step 13). Add a `--token_pooling {final, identity_span_last, identity_span_mean}` flag to `extract_identity_activations.py` that consumes the same logic. Re-run extraction at `identity_span_last` and `identity_span_mean`; re-run Stage 2 geometry on those. The token-pooling mode is already scaffolded but `NotImplementedError` in `encode_identity_saes.py`, so this also unblocks span-pooled SAE encoding.

### 4.6 [MINOR] — Top-k SAE truncation may clip true activations (PARTIAL FIX LANDED 2026-05-27)

**Status:** Upstream detection landed in commit `c6dbcfe` ([Step 6](06_validate_sae_hook_alignment.md)): the validator now reports `reconstruction_l0_p50/p95/p99` and gates on `max_l0 > --top_k_save_threshold` (default 64, matching step 5's `--top_k_save`). If empirical max L0 exceeds the cap, the validator fails before any downstream consumer (including this script) sees the encoding. The empirical answer itself still requires the RunPod run.

**Why it matters in this file (preserved):** Even though `encode_selected_features` runs a fresh forward and computes the **dense** per-token activation for the chosen features (so per-token values are not truncated), the **set of prompts** examined per feature is determined by the upstream top-64 sparse encoding. Features whose final-token activation is moderate (rank 65-150) but whose mid-prompt activation is large will appear to have very few prompts and uninformative `feature_localization_type` distributions.

**Remaining (RunPod):** After the audit-1.4 re-encode, [Step 6](06_validate_sae_hook_alignment.md) will report `recon_l0_clipping_risk`. If True, raise `--top_k_save` in step 5 and re-encode before running this script.

## Rebuild checklist

- [ ] Extract `find_identity_span` into the shared common module so `extract_identity_activations.py` can use it for span-pooled extraction. (1.1)
- [ ] Add a `--token_pooling` flag to `extract_identity_activations.py` using the shared helper; rerun Stage 1 with `identity_span_last` and `identity_span_mean`. Rerun Stage 2 geometry on those outputs and report the comparison. (1.1)
- [ ] Audit SAE empirical L0; if needed, raise `--top_k_save` in `encode_identity_saes.py` and rerun `encode_identity_saes.py` and this script. (4.6)
- [ ] After fixing `combined_score` and selection bias upstream (5.3, 2.5), re-run `select_features` here so the per-feature top-prompts reflect the corrected feature pool.
- [ ] Hardcoded 0.7 cutoffs in localization assignment should be exposed as CLI args so the triage's thresholds (`identity_span_local_threshold`, `final_token_integrated_threshold`) can be swept jointly. The threshold now lives as `common.DEFAULT_LOCALIZATION_THRESHOLD` and the helper takes it as a keyword arg, so the change is one `--localization_threshold` CLI flag away.

## Notes from the doc audit

- The `is_top_token_for_feature` flag is computed from the **content-token-masked** argmax (lines 320-327), so special / zero-width tokens cannot win. Good.
- `feature_top_tokens.csv` is rebuilt from the full CSV at the end of each layer (`write_top_tokens_from_token_csv`, line 209). On `--resume` runs after the first batch, this is called even when no new prompts were processed, which is correct but slow on large CSVs — consider gating on "did we append rows this run."
- `processed_prompt_ids` uses `>= len(feature_ids)` distinct features (line 206). If the feature list **changes** between runs (e.g. after re-running `analyze_identity_sae_features.py`), `--resume` will silently keep stale rows for old features. The safe operation is `--overwrite` after any upstream change.
