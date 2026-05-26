# Step 21 — `scripts/build_bbq_sae_feature_cards.py`

**Stage:** 4 — BBQ steering and feature-level causal analysis
**Runs after:** [Step 19 — `extract_bbq_token_level_sae_activations.py`](19_extract_bbq_token_level_sae_activations.md), [Step 18 — `prepare_bbq_for_steering.py`](18_prepare_bbq_for_steering.md), [Step 17 — `triage_sae_identity_features.py`](17_triage_sae_identity_features.md).
**Feeds into:** Human inspection. The final causal analyzer (Step 23) links to these cards from `final_intervention_candidates_table.html` and `feature_card_links_table.html`, but does not consume their contents programmatically.

## Purpose
Build standalone HTML "feature cards" for each kept-for-intervention SAE feature, using the BBQ-side token-level activations from Step 19. Each card shows top-activating BBQ prompts, top-activating tokens (with token-role coloring: target / nontarget / stereotype-language / answer-option), per-context-condition / per-polarity activation summaries, and a behavioral classification of where the feature predominantly fires. This is the human-readable lens on what each feature *does* on BBQ, sibling to the identity-prompt feature cards from Stage 3.

## Inputs
- `token_level_sae/token_activations/layer_XX/part_*.parquet` and `bbq_token_level_sae_summary.csv` — from Step 19.
- `prepared/bbq_prepared_examples.parquet` — for prompt text, identity labels, and answer indices used in the prompt-rendering layer.
- `results/.../triage/intervention_candidate_features_triaged.csv` — for `provisional_role`, `top_axis`, `top_identity`, `contrast_name`, scores.

## Outputs
```
feature_cards/
  index.html
  feature_card_index.csv
  layer_XX/
    feature_XXXXX.html        # one card per kept feature
    feature_XXXXX.json        # machine-readable card data
    (optional .png exemplar figures)
  logs/build_cards.log
```

## Key implementation details
- This is the **filtered** card builder that replaces the earlier `feature_cards/` directory. It applies a token filter using `LOW_INFORMATION_TOKENS` (punctuation only: `. , : ; ? ! - — ( ) [ ] / \\ |`) and `LOW_INFORMATION_WORDS` (a hand-curated stoplist `a, an, and, are, as, ..., person, people, someone, somebody`) to drop low-information items from top-token tables.
- Special tokens (`<|begin_of_text|>`, ...) are also filtered before ranking.
- Token-role coloring is driven by the boolean flags from Step 19 (`is_target_identity_token`, `is_nontarget_identity_token`, `is_stereotype_language_token`, `is_answer_option_token`, ...). Activation intensity is mapped to an HSL gradient inline in the HTML.
- Cards are deliberately **independent of steering results**. They use only token-level activations, so they can be rebuilt without re-running the model.

## Issues & Opportunities

### Inherited issues — no new ones specific to this script

This script is a presentation layer over the artifacts of upstream steps. The card content inherits any problems from:

- **Step 19 issue 3.3** (`find_all_spans` greedy regex): if `is_target_identity_token` was set on answer-option-internal tokens that happen to spell the identity, those tokens will be colored as target-identity in the card. The "target identity" exemplars in the card are therefore a mix of true identity mentions and answer-option matches.
- **Step 17 / Step 5 issue 1.4** (SAE preprocessing): if the SAE was loaded with the wrong preprocessing convention, every activation value rendered on the card is on a mis-scaled input.
- **Step 18 issues 4.1 / 4.2** (missing identities, flattened intersectionality): the `top_identity` and contrast metadata on each card come from triage which is itself agnostic to BBQ; the card only inherits what the prepared BBQ data exposes.
- **Step 17 issue 5.2** (heuristic triage roles): `provisional_role` shown on the card is a heuristic label, not a validated finding.

## Rebuild checklist
- [ ] After Step 19 gains section-restricted identity flags (Step 19 issue 3.3 fix), re-render cards using the new `is_target_identity_token_in_context` flag for the headline "target-identity exemplars" panel so the panel stops conflating answer-option mentions with context mentions.
- [ ] Verify the SAE preprocessing convention before publishing any card that lists activation magnitudes (1.4).
- [ ] Optionally, after Step 23 produces `feature_level_effects.csv`, embed a small "causal effect summary" panel on each card linking the feature's BBQ effect labels (`bias_reducing_uncertainty`, etc.) — at the moment cards and the causal analyzer cross-link only via filenames, not content.
- [ ] Confirm `LOW_INFORMATION_WORDS` is consistent with the stopword sets in Step 19 (`STOPWORDS`) and Step 20 (`stereotype_terms`) by moving all three to a shared module (5.10).

## Notes from the doc audit
- The card builder rebuilds the per-feature long-format table by concatenating every `part_*.parquet` in the layer dir; on long BBQ runs this is memory-bound and a candidate for chunked iteration.
- The `feature_card_index.csv` is keyed by `layer, feature_id` but does not record the `mapped_contrast_confidence` distribution of the prompts that activated the feature — adding that column would let the user filter cards by confidence the same way the causal analyzer should be filtered.
