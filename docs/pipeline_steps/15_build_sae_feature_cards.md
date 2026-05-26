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
- Per-prompt localization recomputation (`exemplar_prompt_table`, lines 217-224): re-derives `feature_localization_type` from the token table with the same 0.7 thresholds as the extraction script. Could drift if the constants change in one place and not the other.
- Token heatmap (`token_heat_html`, line 145): green-intensity coloring scaled to the per-prompt max activation, with yellow outline for identity-span tokens and a black "top-token" border.
- Optional logit-lens (`compute_logit_effects`, line 345): `logits = lm_head.weight @ decoder[feature_id]`, top-20 positive and bottom-20 negative tokens. The code itself documents: **`"Raw decoder @ lm_head projection; final norm not applied."`** (line 373). This means the projection skips Llama's final RMSNorm; absolute logit magnitudes are not comparable to what the model actually emits.
- The `LOGIT_CACHE` (line 29) keeps the model and tokenizer in module-level memory across calls — single-process only.

## Issues & Opportunities

This script is largely a presentation layer; the substantive issues it inherits from upstream are flagged in the corresponding step docs. The card-specific items are:

### Caveat — Logit-lens projection skips final norm

**What's wrong:** Llama-3.1-8B applies an RMSNorm before `lm_head`. `compute_logit_effects` projects raw decoder rows through `lm_head.weight` without that norm. The script's own note documents this. Effects:
- Absolute logit scores are not in the same scale as the model's real output logits.
- Token *rankings* among "directions of similar norm" are roughly preserved, but the comparison across features whose decoder rows have very different norms is misleading.

**Why it matters:** If a card user reads "this feature predicts the token `gay`" from the top-positive list, they should know the projection is approximate. As a discovery / triage tool the rankings are useful; as a calibrated "this feature pushes the logit of token X by Y" claim they are not.

**Targeted fix:** Apply Llama's final norm to `decoder[feature_id]` before the `@ lm_head.weight` projection. The norm parameters are accessible at `model.model.norm.weight`; RMSNorm is `x * rsqrt(mean(x²) + eps) * weight`. After the fix, the printed note should be updated to `"Decoder @ final_norm @ lm_head projection."` Keep the raw projection as a hidden diagnostic if needed.

### 5.3 (inherited) [MINOR] — `combined_score` double-weighting flows through `select_features`

**What's wrong:** Same as in step 13 — the feature pool here is partly selected by `combined_score`, which sums z-scored `|d|`, `|cos|`, and `|auc − 0.5|` and over-weights selectivity.

**Why it matters:** The set of features that get cards may be biased toward selectivity-only features.

**Targeted fix:** Fix in `analyze_identity_sae_features.py` (see step 13). No change needed in this script once the upstream `combined_score` definition is corrected.

### 5.10 (inherited) [MINOR] — Re-derivation of localization classification

**What's wrong:** `exemplar_prompt_table` (line 198) recomputes `feature_localization_type` from the token table using the same 0.7 thresholds as `extract_token_level_sae_activations.py:main`. Two independent implementations of the same rule.

**Targeted fix:** Move the classifier into the shared common module so both scripts call the same function. Also the constants (0.7, 0.7) should be a single shared default.

## Rebuild checklist

- [ ] Apply Llama's final RMSNorm to the decoder row before the `lm_head` projection in `compute_logit_effects`; update the note string.
- [ ] After the upstream `combined_score` and selection-bias fixes (steps 13, 14), regenerate cards with `--overwrite`; visual sanity-check that the feature pool is roughly the same set of "looks-like-identity" features.
- [ ] If `feature_localization_type` is refactored into a shared helper (5.10), delete the local recomputation in `exemplar_prompt_table` and call the helper instead.
- [ ] Optional: add an "uncertainty" panel to the card showing where the feature's `combined_score` ranks within its contrast and within all features, so the user can see when a card is on a marginal feature.

## Notes from the doc audit

- The `--make_index` flag must be passed explicitly; otherwise no `index.html` is written. Easy to miss in a fresh run.
- `prepare_output` (line 76) preserves the `token_level/` subdirectory on `--overwrite`, which is correct because that data comes from the upstream script and is expensive to regenerate.
- `feature_token_df[feature_token_df["feature_id"].eq(feature_id)]` (line 393) is called inside `build_card` for every feature; for a large `token_df` and many features this is O(n_features × n_rows). On a normal layer-24 run with ~80 features and ~3M token rows this is non-trivial. Could be faster with a single groupby up front.
