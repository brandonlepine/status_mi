# Step 19 — `scripts/extract_bbq_token_level_sae_activations.py`

**Stage:** 4 — BBQ steering and feature-level causal analysis
**Runs after:** [Step 18 — `prepare_bbq_for_steering.py`](18_prepare_bbq_for_steering.md), [Step 17 — `triage_sae_identity_features.py`](17_triage_sae_identity_features.md).
**Feeds into:** [Step 21 — `build_bbq_sae_feature_cards.py`](21_build_bbq_sae_feature_cards.md), [Step 23 — `analyze_bbq_feature_level_causal_effects.py`](23_analyze_bbq_feature_level_causal_effects.md) (the latter merges the per-feature activation summary into its causal-effects table).

## Purpose
Re-run Llama on the prepared BBQ prompts and encode each prompt's residual stream at the chosen layer(s) through the SAE encoder for the **kept-for-intervention** features from triage. For every token in every prompt, record the feature activation and a battery of role flags (`is_target_identity_token`, `is_stereotype_language_token`, `is_answer_option_token`, …). This is the data substrate for the BBQ feature cards and for joining "where the feature actually fires" with the steering causal effects.

## Inputs
- `prepared/bbq_prepared_examples.parquet` (or `.csv`) — from Step 18.
- `results/.../triage/intervention_candidate_features_triaged.csv` — selects the feature pool (filtered by `keep_for_intervention == True` and, unless `--include_all_kept_features`, by `provisional_role ∈ {contrast_specific_identity, identity_token_local, shared_social_feature, sentence_final_integrated}`).
- Llama model directory and the OpenMOSS SAE directory.

## Outputs
```
token_level_sae/
  token_activations/layer_XX/
    part_XXXXX.parquet           # one row per (prompt, feature, positive-activation token)
    manifest.csv                 # batch_id, start_row, n_rows, path, elapsed_seconds for --resume
  bbq_token_level_sae_summary.csv  # per (layer, feature_id) means of target / nontarget /
                                   # stereotype-language / final-token activation,
                                   # plus per-context-condition / per-polarity / per-category breakdowns
  token_level_config.json
  logs/token_level_sae.log
```

## Key implementation details
- **SAE encoding** reuses the same `load_sae` / `encode_selected_features` helpers from `encode_identity_saes.py`: `relu((x − b_dec) @ w_enc[:, features] + b_enc[features])`. This inherits the **unverified preprocessing convention** flagged in issue 1.4 — if LlamaScope expects an input normalization step that the generic loader skips, every activation value here is on a mis-scaled input. See [Step 5 — `encode_identity_saes.py`](05_encode_identity_saes.md) and the cross-cutting verification item.
- **Hidden state layer indexing**: `outputs.hidden_states[layer]` where `layer ∈ {8, 16, 24, 32}` by default. This is the post-block-`layer` residual (HF convention), matching the LkR convention of OpenMOSS LlamaScope. Good.
- **Per-prompt loop emits one row per (prompt, feature, token) where `feature_activation > 0`.** Tokens with zero activation are skipped entirely — they are NOT recorded as `feature_activation = 0`, which keeps the parquet small but means absence is implicit.
- **Section spans** (`find_section_spans`): identifies the character ranges of `context`, `question`, and `A. ans0 / B. ans1 / C. ans2` in the prompt — used to set `is_question_token`, `is_context_token`, `is_answer_option_token` and which `answer_option_idx`.
- **Identity / stereotype-language spans** (`find_all_spans`): regex-locates every occurrence of the identity label *and* the corresponding answer-option text *and* the question's content words. Returns **all** matches; downstream uses `overlap` against the full list, so a single token that overlaps any one of these matches is flagged. Audit 3.3 (closed 2026-05-28 in commit `afb3ee3`) added section-aware companion flags so the section where the match occurred is recoverable per-token.
- **Token-role flags** are produced bool-by-bool. The legacy `is_target_identity_token` is True if the token overlaps the identity-label match anywhere in the prompt; the audit-3.3 section-aware variants (`is_target_identity_token_in_{context, question, answer_option}`) attribute the match to its prompt region so downstream "the feature fires at the identity mention" claims can be tied to the **context** identity mention rather than the answer-option identity mention.
- **Per-feature summary** (`bbq_token_level_sae_summary.csv`) computes `mean_target_identity_activation`, `mean_nontarget_identity_activation`, `mean_stereotype_language_activation`, `mean_final_token_activation`, `max_activation_per_prompt`, `fraction_prompts_active`, and breakdowns by `context_condition`, `question_polarity`, and `category_raw` (the last three as JSON-encoded dicts inside CSV cells).

## Issues & Opportunities

> **Upstream callout — issue 1.4 (FIX LANDED in Step 5; this script's encode path still TODO).** The encoder fix in [Step 5](05_encode_identity_saes.md#14-blocker--sae-preprocessing-convention-fix-landed-2026-05-26) landed in commit `4b8851a`. **This script has not been updated** — it still inlines the old `relu((x − b_dec) @ W_enc + b_enc)` formula. Before re-running it, refactor to import `load_sae` + `encode_full` from `encode_identity_saes.py` (single source of truth) so the same JumpReLU + dataset-wise-normalization + no-pre-bias convention is used. Every parquet under `token_level_sae/token_activations/.../layer_XX/` is currently wrong and must be regenerated against the corrected encoder.

### 3.3 [MAJOR] — Intervention positions are located by greedy regex and may hit the wrong span (FIX LANDED 2026-05-28)

**Status:** Closed in commit `afb3ee3` across this script and [Step 20](20_run_bbq_sae_steering.md).

**What landed (this script):**
- New helper `overlap_in_section(start, end, term_spans, section_span)` returns True only when a token overlaps a term match AND lies within the given section span.
- Per-token rows gain nine new boolean columns:
  - `is_target_identity_token_in_{context, question, answer_option}`
  - `is_nontarget_identity_token_in_{context, question, answer_option}`
  - `is_stereotype_language_token_in_{context, question, answer_option}`
- `bbq_token_level_sae_summary.csv` gains nine new mean-activation columns (`mean_*_activation_in_{context, question, answer_option}`), via a local `_mean_where` helper that returns NaN for empty section masks. Downstream consumers can answer "is this feature firing on the **context** identity mention or just the answer-option mention?" without re-deriving from the token table.
- The legacy `is_target_identity_token` / `is_stereotype_language_token` flags are preserved alongside for backward compatibility.

**Companion fix in [Step 20](20_run_bbq_sae_steering.md):** six new section-explicit intervention position names (`target_identity_last_context_token`, `target_identity_last_question_token`, and corresponding `nontarget_*` and `stereotype_language_*` variants); new `intervention_section` column on each output row recording where the chosen token actually landed (`context` / `question` / `answer_option` / `final` / `mixed` / `unknown`).

**Original audit (preserved):** `find_all_spans` collected every regex match of the identity label and answer-option text in the prompt, with no section filtering. The identity label often appears in the context, the question, and the answer-choice list (`B. the Black man`) simultaneously. Tokens were flagged `is_target_identity_token` if they overlapped *any* of these matches. The same logic was reused verbatim in `run_bbq_sae_steering.py:positions_for`, where it determined where the steering hook was applied — so the bug compounded.

#### ⚠️ Pre-RunPod validation caveats (3.3 fix)

The synthetic test that landed with commit `afb3ee3` confirmed the helpers' math on **one** carefully-constructed prompt using a **whitespace-splitting fake tokenizer**. This is enough to verify the geometric logic but does not exercise the cases that are most likely to break in production. Before trusting numbers from these new columns:

**Was tested:**
- `overlap_in_section` on one audit-pathological prompt (Black/White man context + answer options).
- 4 position-name variants on the same prompt, plus no-match fallback to `final_prompt_token`.
- Round-trip of section spans through `find_section_spans`.

**Was NOT tested (likely failure modes):**
- **Real Llama tokenizer offsets.** BPE produces leading-space-included offsets and may split multi-word identity labels (`"Black man"`) across tokens differently than whitespace splitting suggests. A token whose char offsets straddle a section boundary could flip the section assignment.
- **Real BBQ prompt format with few-shot prefix.** `find_section_spans` uses `prompt.find(str(row['context']))` — exact, case-sensitive substring match. If `prepare_bbq_for_steering.py`'s `--few_shot_pool` prepends a prefix, normalizes whitespace, or the prompt template changes, the lookup returns `-1` and **all the new `is_*_in_section` flags silently become `False`**.
- **First-match-wins in `prompt.find`.** If a section's text appears as a substring of another section (e.g. quoted dialogue in the context that repeats the question), `find_section_spans` will pick the wrong span.
- **Multi-token identity labels.** The first token of a span was confirmed to flag correctly; all spanned tokens flipping together was not verified.
- **End-to-end run against real BBQ activations.** The row-dict additions and `_mean_where` summary helper were not exercised in a real `extract_bbq_token_level_sae_activations.py` job.

**Recommended pre-RunPod check:** `scripts/audit_intervention_sections.py` was written for exactly this. It samples ~50 prepared BBQ rows, runs `find_section_spans` / `positions_for` / `position_section_for` against the real Llama tokenizer, and exits non-zero on threshold violations. Four checks:

1. `find_section_spans` returns non-empty `context` / `question` / `ans*` for every row (default threshold: ≤5% failures).
2. For every token with legacy `is_target_identity_token = True`, ≥1 of the three `is_target_identity_token_in_*` flags is True — orphan and boundary-straddle counts reported (default threshold: ≤5% orphans).
3. For each of the six section-explicit position names, the `intervention_section` distribution across the sampled rows (default threshold: ≤20% silent fallback to `final`).
4. The (position × section) cross-tab so the operator can eyeball drift.

```bash
python scripts/audit_intervention_sections.py \
    --prepared_data /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet \
    --tokenizer_path /workspace/status_mi/models/llama-3.1-8b \
    --n_examples 50
```

Run this **before** any steering job that depends on the section-explicit position names. If it exits non-zero, do not cite the new positions in headline causal claims until the failures are diagnosed.

### 4.6 [MINOR] — Top-k SAE truncation (in the encoding upstream) may bias activation summaries

**What's wrong (inherited from `encode_identity_saes.py`):** The identity-side encoding only retains the top-64 feature indices per row; this script does its *own* dense `encode_selected_features` on a pre-filtered feature subset, so it is **not** subject to top-64 truncation here. However, the feature pool itself was selected (via triage) using metrics computed on the truncated identity-side encodings. If the SAE's true L0 at layer 24 exceeds 64 on some identity prompts, mid-ranked features that genuinely fire on identity were not in the kept set and never reach this script.

**Why it matters:** The "kept-for-intervention" feature list could be missing features that are causally relevant but never broke the top-64 at the period token on templated prompts. Those features will never appear in BBQ feature cards or the causal analysis.

**Targeted fix:** Verify SAE L0 at layer 24 on identity prompts (see issue 1.4 / Step 5). If it is comfortably under 50, leave the upstream `--top_k_save 64`. If not, raise `--top_k_save` upstream and re-run triage. Independently, optionally have this script also encode the *full* feature dimension on a stratified subsample of BBQ prompts and report which features have non-zero BBQ activation but were not in the kept set — a "missed candidates" diagnostic.

## Rebuild checklist
- [ ] Verify SAE preprocessing/activation function before trusting any activation in `bbq_token_level_sae_summary.csv` (cross-cutting fix from issue 1.4 — applies to every SAE-touching script).
- [ ] Add section-restricted identity/stereotype-language flags (`*_in_context`, `*_in_question`, `*_in_answer_option`).
- [ ] In the per-feature summary, break out `mean_target_identity_activation` by section so a feature that fires only at the answer-option occurrence is distinguishable from one that fires in the context.
- [ ] Optional: add a one-time diagnostic that encodes the full SAE on a small BBQ sample and lists features with high BBQ activation that were *not* in the kept set.
- [ ] Document that absent rows in `part_*.parquet` mean `feature_activation == 0`; downstream consumers (cards, summary aggregations) should not interpret missing rows as "data not available."

## Notes from the doc audit
- `find_all_spans` collects matches case-insensitively (lowercasing the prompt) but does not enforce word boundaries. A target identity of `"man"` will match inside `"woman"`. Adding `\b...\b` (or token-aware matching using offsets) would prevent these false positives.
- `read_table(path)` is called on every part during the post-loop concatenation that builds `bbq_token_level_sae_summary.csv`. For long runs with many parts, this is the memory bottleneck — consider streaming aggregation instead.
- The `STOPWORDS` set duplicates the one in `run_bbq_sae_steering.py:stereotype_terms` but they are not identical (the latter omits `"did"`/`"does"`). Stereotype-token flags here therefore disagree with the steering runner's `stereotype_language_last_token` position selection in edge cases. Pull this into a shared module (5.10).
