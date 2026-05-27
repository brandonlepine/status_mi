# Step 15 — `build_sae_feature_cards.py`

**Stage:** 3 — Identity-selective SAE feature analysis (presentation layer)
**Runs after:** `analyze_identity_sae_features.py`, `extract_token_level_sae_activations.py`
**Feeds into:** Human inspection only; produces no CSV consumed by later stages. (The BBQ-side analogue is `build_bbq_sae_feature_cards.py` in Stage 4.)

## Purpose

Builds standalone HTML "feature cards" for selected identity-relevant SAE features. Each card consolidates everything the project already knows about a single feature: identity-mean activations, contrast selectivity, decoder alignment, prompt exemplars with per-token activation heatmaps, top tokens, identity-span tokens, localization-type distribution, and (optionally) a raw logit-lens projection. The output is for human triage and feature interpretation — it does not feed downstream computation, but it is the primary artifact used to label features by hand.

## Inputs

- `analysis/intervention_candidate_features.csv`, `feature_selectivity_alignment_joined.csv`, `feature_identity_selectivity.csv`, `feature_selectivity.csv`, `decoder_direction_alignment.csv` — pulled into per-feature stat tables.
- `layer_XX/feature_stats.csv` — per-feature SAE activation stats from `encode_identity_saes.py`.
- `layer_XX/feature_indices_top64.npy`, `feature_values_top64.npy`, `sae_decoder.npy` — sparse encodings and decoder for identity-profile bar plots and optional logit-lens.
- `feature_cards/token_level/layer_XX/token_feature_activations.csv` (or `.parquet`) — per-token activations from `extract_token_level_sae_activations.py`.
- `results/activations/.../metadata.csv` — prompt and identity metadata.
- (Optional) Llama model directory via `--model_path` for `--compute_logit_lens`.

## Outputs (under `<output_dir>/`, default `.../feature_cards/`)

- `layer_XX/feature_XXXXX.html` — the card itself.
- `layer_XX/feature_XXXXX.json` — machine-readable mirror of the card data.
- `layer_XX/feature_XXXXX_identity_profile.png` — identity-mean bar plot.
- `token_exemplars/layer_XX/feature_XXXXX_token_exemplars.{png,pdf}` — token-level matplotlib heatmap.
- `index.html` — when `--make_index`.
- `feature_card_index.csv` — table of all cards built.

## Key implementation details

- Feature selection (`select_features`, line 101): mirrors the union logic in `extract_token_level_sae_activations.py:select_features` — top-N by `combined_score` per contrast, plus top by `|cohens_d|`, plus per-identity tops. Inherits the same upstream selection-bias concerns (issue 2.5) and the `combined_score` double-weighting (5.3).
- `classify_label` (line 168): auto-labels the feature as `"<axis> feature"` if ≥70% of the top-10 identities by mean activation are in the same axis; otherwise as `"<canonical_label>-selective feature"` or `"identity-token-local feature"` (if >50% of prompts have `feature_localization_type == identity_span_local`) or `"mixed/polysemantic feature"`.
- Per-prompt localization recomputation (`exemplar_prompt_table`): re-derives `feature_localization_type` from the token table using the shared `common.classify_feature_localization` helper (audit 5.10 instance closed 2026-05-27 in commit `402731f`); the same helper is used by `extract_token_level_sae_activations.py` so the two scripts cannot drift.
- Token heatmap (`token_heat_html`, line 145): green-intensity coloring scaled to the per-prompt max activation, with yellow outline for identity-span tokens and a black "top-token" border.
- Optional logit-lens (`compute_logit_effects`): applies Llama's final RMSNorm (`model.model.norm`) to the decoder row before projecting through `lm_head.weight` (commit `402731f`). The note string now reads `"Decoder @ final_norm @ lm_head projection."` Absolute logit magnitudes are now on the same scale as the model's actual output logits; cross-feature comparisons are no longer biased by decoder-row norm. Returns top-20 positive and bottom-20 negative tokens.
- The `LOGIT_CACHE` (line 29) keeps the model and tokenizer in module-level memory across calls — single-process only.

## Issues & Opportunities

This script is largely a presentation layer; the substantive issues it inherits from upstream are flagged in the corresponding step docs. The card-specific items are:

### Caveat — Logit-lens projection skips final norm (FIX LANDED 2026-05-27)

**Status:** Closed in commit `402731f`.

**What landed:** `compute_logit_effects` now applies Llama's final RMSNorm (`model.model.norm`) to the decoder row before projecting through `lm_head.weight`. The implementation reads `norm.weight` (gamma) and `norm.variance_epsilon` from the already-loaded HF model and applies `x * rsqrt(mean(x²) + eps) * gamma` in numpy — verified bit-for-bit (~1e-7) against `torch`'s `LlamaRMSNorm`. The note string updated to `"Decoder @ final_norm @ lm_head projection."` If `model.model.norm` is not present (some non-Llama checkpoint), the code falls back to the raw projection with a clarifying note.

**Original audit (preserved):** Llama-3.1-8B applies an RMSNorm before `lm_head`. The prior `compute_logit_effects` projected raw decoder rows through `lm_head.weight` without that norm. Absolute logit scores were not on the same scale as the model's real output logits; cross-feature comparisons were misleading when decoder rows had very different norms. As a discovery / triage tool the rankings were useful; as a calibrated "this feature pushes the logit of token X by Y" claim they were not.

### 5.3 (inherited) [MINOR] — `combined_score` double-weighting flows through `select_features` (RESOLVED UPSTREAM 2026-05-27)

**Status:** Upstream fix landed in commit `3b48e5b` ([Step 13](13_analyze_identity_sae_features.md)). `combined_score` is now `0.5·z(|cohens_d|) + 0.5·z(|cosine_with_direction|)`; the AUC term is gone. No change needed in this script — the feature pool changes on next regeneration as the new ranking propagates through `select_features`.

### 5.10 (instance) [MINOR] — Re-derivation of localization classification (FIX LANDED 2026-05-27)

**Status:** Closed in commit `402731f`. The 4-branch 0.7-threshold logic now lives in `scripts/common.py:classify_feature_localization` (with `DEFAULT_LOCALIZATION_THRESHOLD = 0.7` as a named constant). Both `exemplar_prompt_table` here and the per-prompt classification in `extract_token_level_sae_activations.py` call the same helper, verified against the prior step-15 4-branch on all 9 meaningful (max, span, final) buckets.

Note: this was an instance of audit 5.10 (code duplication) not covered by the original 5.10 commit `e50bbd1` (which extracted `cohens_d`, `cosine`, `normalize`, `compute_direction`, etc.). With this commit, the named-helper coverage now also includes the localization classifier.

## Rebuild checklist

- [x] Apply Llama's final RMSNorm to the decoder row before the `lm_head` projection in `compute_logit_effects`; update the note string. *(Done 2026-05-27: commit `402731f`.)*
- [ ] After the upstream `combined_score` and selection-bias fixes (steps 13, 14), regenerate cards with `--overwrite`; visual sanity-check that the feature pool is roughly the same set of "looks-like-identity" features. (RunPod step.)
- [x] Move `feature_localization_type` to a shared helper and call it from both call sites. (5.10 instance) *(Done 2026-05-27: commit `402731f` — `common.classify_feature_localization`.)*
- [ ] Optional: add an "uncertainty" panel to the card showing where the feature's `combined_score` ranks within its contrast and within all features, so the user can see when a card is on a marginal feature.

## Notes from the doc audit

- The `--make_index` flag must be passed explicitly; otherwise no `index.html` is written. Easy to miss in a fresh run.
- `prepare_output` (line 76) preserves the `token_level/` subdirectory on `--overwrite`, which is correct because that data comes from the upstream script and is expensive to regenerate.
- `feature_token_df[feature_token_df["feature_id"].eq(feature_id)]` (line 393) is called inside `build_card` for every feature; for a large `token_df` and many features this is O(n_features × n_rows). On a normal layer-24 run with ~80 features and ~3M token rows this is non-trivial. Could be faster with a single groupby up front.
