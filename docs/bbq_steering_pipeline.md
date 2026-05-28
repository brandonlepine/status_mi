# BBQ SAE Steering Pipeline

This document records the BBQ steering and SAE feature-card pipeline built for `status_mi`, including what each script does, how to run it on RunPod, where outputs currently live locally, and which existing results should be treated as preliminary or flawed.

## Project Layout

Primary RunPod project root:

```text
/workspace/status_mi
```

Fast local working copy on the pod:

```text
/root/local_status_mi
```

Use `/root/local_status_mi` for model/SAE reads and active compute. Use `/workspace/status_mi` for persistent storage after runs complete. The `/workspace` mount is network storage and has shown very slow metadata and file-read performance.

Local Mac output root:

```text
/Users/brandonlepine/Repositories/Research_Repositories/status_mi/output_data
```

## Scripts

### `scripts/prepare_bbq_for_steering.py`

Loads BBQ JSONL files, normalizes metadata, maps BBQ answer groups to identity IDs and contrasts, and writes a steering-ready dataset.

Main inputs:

- `data/bbq/data/*.jsonl`
- `data/bbq_identity_normalized_forms.csv`
- `results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv`

Main outputs:

- `bbq_prepared_examples.csv`
- `bbq_prepared_examples.parquet`
- `bbq_mapping_diagnostics.csv`
- `bbq_contrast_mapping.csv`
- `bbq_prepare_summary.csv`
- `bbq_prepare_config.json`

Important prepared columns:

- `bbq_uid`: unique example identifier.
- `axis_mapped`: BBQ category mapped to an identity axis.
- `context_condition`: `ambig` or `disambig`.
- `question_polarity`: `neg` or `nonneg`.
- `prompt`: formatted model prompt with answer choices.
- `unknown_answer_idx`: answer index for the unknown/cannot-answer option.
- `stereotyped_answer_idx`: answer index for the stereotype-consistent group.
- `nonstereotyped_answer_idx`: answer index for the contrast identity group.
- `correct_answer_idx`: BBQ gold answer index.
- `target_identity_id`: identity treated as stereotype target.
- `nontarget_identity_id`: contrast identity when identifiable.
- `mapped_contrast_name`: nearest available identity contrast from SAE triage.
- `mapped_contrast_confidence`: `exact`, `alias`, `fallback_axis`, or `unmapped`.

### `scripts/extract_bbq_token_level_sae_activations.py`

Runs the model on prepared BBQ prompts and extracts sparse token-level SAE activations for kept identity features.

Main inputs:

- Prepared BBQ parquet or CSV.
- SAE root: `saes/openmoss/Llama3_1-8B-Base-LXR-32x`.
- Model root: `models/llama-3.1-8b`.
- Triage CSV.

Main outputs:

- `token_activations/layer_XX/part_XXXXX.parquet`
- `token_activations/layer_XX/manifest.csv`
- `bbq_token_level_sae_summary.csv`
- `token_level_config.json`

Important token columns:

- `bbq_uid`
- `layer`
- `feature_id`
- `token_idx`
- `token_str`
- `token_start_char`
- `token_end_char`
- `feature_activation`
- `feature_rank_within_prompt`
- `is_target_identity_token`
- `is_nontarget_identity_token`
- `is_any_identity_token`
- `is_stereotype_language_token`
- `is_question_token`
- `is_context_token`
- `is_answer_option_token`
- `answer_option_idx`
- `is_final_prompt_token`

### `scripts/build_bbq_sae_feature_cards.py`

Builds standalone HTML feature cards from token-level BBQ SAE activations.

Current behavior:

- Filters special tokens such as `<|begin_of_text|>`.
- Filters punctuation-only and low-information tokens from top-token and top-example ranking.
- Keeps meaningful `other_context` tokens.
- Explains token-role coloring and activation intensity.

Main outputs:

- `feature_cards/index.html`
- `feature_cards/feature_card_index.csv`
- `feature_cards/layer_24/feature_XXXXX.html`

These cards are independent of the steering run. They use token-level SAE activations, not causal intervention results.

### `scripts/run_bbq_sae_steering.py`

Runs causal interventions with SAE decoder directions and scores BBQ answer choices.

Important current design:

- `--axis_match_mode matched_only` is the default.
- `per_feature` mode is required for true individual feature causal estimates.
- `--require_per_feature` fails fast if `per_feature` is missing.
- For `first_token` scoring with controls disabled, the runner uses batched forward passes.
- The runner checkpoints with `results_parts/part_XXXXX.parquet` and `completed_jobs.jsonl`.

Important output columns:

- `bbq_uid`
- `layer`
- `alpha` (for `clamp` under audit-3.2 per-feature scaling, this is the clamp *multiplier*, not a raw value)
- `intervention_mode`
- `intervention_position`
- `feature_scale_stat` (audit 3.2: the `feature_stats.csv` column scaling clamp/steer amplitude, or `none` for ablate/uniform rows)
- `feature_scale_value` (audit 3.2: the single-feature scale; NaN for bundles/ablate. clamp target = `alpha × feature_scale_value`; steer increment = `alpha × sign × feature_scale_value`)
- `feature_set_mode`
- `feature_set_id`
- `feature_id`
- `feature_role`
- `feature_axis`
- `feature_estimate_type`
- `feature_ids_json`
- `feature_signs_json`
- `mapped_contrast_name`
- `feature_contrast_name`
- `axis_mapped`
- `context_condition`
- `question_polarity`
- `ans0_logprob_base`
- `ans1_logprob_base`
- `ans2_logprob_base`
- `ans0_logprob_intervened`
- `ans1_logprob_intervened`
- `ans2_logprob_intervened`
- `stereotyped_delta`
- `nonstereotyped_delta`
- `unknown_delta`
- `correct_delta`
- `bias_margin_delta`
- `prediction_changed`
- `correct_base`
- `correct_intervened`

### `scripts/analyze_bbq_steering_results.py`

Aggregate interpretation-first analysis for steering runs. This is useful for quick overview plots, but it is not the final feature-level analysis.

Main outputs:

- `steering_results_merged.csv`
- `coverage_report.csv`
- `README_interpretation.md`
- `SMOKE_TEST_LIMITATIONS.md`
- `interpretation_summary_by_axis.csv`
- `interpretation_summary_by_contrast.csv`
- `interpretation_summary_by_feature_set_alpha.csv`
- overview figures in `figures/`

### `scripts/analyze_bbq_feature_level_causal_effects.py`

Feature-level, subgroup-level, identity-level, contrast-level, and axis-level causal analysis.

Main outputs:

- `merged_results.csv`
- `deltas_long.csv`
- `identity_records.csv`
- `identity_deltas_long.csv`
- `feature_level_pre_fdr.csv`
- `feature_level_effects.csv`
- `subgroup_level_effects.csv`
- `identity_level_effects.csv`
- `feature_x_subgroup_matrix.csv`
- `feature_effect_rankings.csv`
- `validation_summary.csv`
- `final_intervention_candidates_table.html`
- axis reports under `analysis/axis_reports/`
- contrast reports under `analysis/contrast_reports/`
- identity reports under `analysis/identity_reports/`

Important metrics:

- `stereotype_preference_delta`: change in `log p(stereotyped answer) - log p(unknown answer)`.
- `identity_answer_delta`: change in log probability for a specific identity answer option.
- `identity_specific_bias_delta`: identity answer shift relative to unknown answer shift.
- `unknown_delta`: change in log probability for unknown answer.
- `nonstereotyped_delta`: change in log probability for non-stereotyped identity answer.
- `substitution_delta`: change in `log p(nonstereotyped) - log p(stereotyped)`.
- `correct_margin_delta`: change in disambiguated correct-answer margin.
- `accuracy_delta`: change in correctness.
- `q_value_fdr`: Benjamini-Hochberg FDR-corrected p-value.
- `effect_label`: identity-aware behavioral label.

Interpretation:

- Negative `stereotype_preference_delta` means less preference for the stereotyped answer relative to unknown.
- Positive `stereotype_preference_delta` means more preference for the stereotyped answer relative to unknown.
- Positive `unknown_delta` means more model uncertainty.
- Positive `nonstereotyped_delta` with negative stereotyped movement can indicate substitution, not calibrated debiasing.

## Data Files

### BBQ JSONL files

Location:

```text
data/bbq/data/*.jsonl
```

Expected files currently used:

- `Age.jsonl`
- `Disability_status.jsonl`
- `Gender_identity.jsonl`
- `Nationality.jsonl`
- `Physical_appearance.jsonl`
- `Race_ethnicity.jsonl`
- `Race_x_gender.jsonl`
- `Race_x_SES.jsonl`
- `Religion.jsonl`
- `SES.jsonl`
- `Sexual_orientation.jsonl`

Important fields in each JSONL row:

- `example_id`: BBQ example identifier within a category.
- `question_index`: BBQ question template index.
- `question_polarity`: `neg` or `nonneg`.
- `context_condition`: `ambig` or `disambig`.
- `category`: BBQ category.
- `answer_info`: mapping from `ans0`, `ans1`, `ans2` to answer text and group label.
- `additional_metadata`: includes `stereotyped_groups`, `subcategory`, and source metadata.
- `context`: context paragraph.
- `question`: question text.
- `ans0`, `ans1`, `ans2`: answer strings.
- `label`: correct answer index.

### `data/bbq_identity_normalized_forms.csv`

Identity normalization table used to map BBQ labels to project identity IDs.

Important columns:

- `identity_id`: normalized project identity ID.
- `axis`: identity axis.
- `canonical_label`: readable identity label.
- `adj_form`: adjective form.
- `noun_form`: noun form.
- `person_noun_form`: person phrase.
- `plural_noun_form`: plural phrase.
- `group_form`: group phrase.
- `prep_form`: prepositional form.
- `with_form`: “with X” form.
- `has_form`: “has X” form.
- `aliases`: semicolon-delimited aliases.
- `works_*`: template compatibility flags.
- `notes`: identity-specific notes.

### SAE triage CSV

Location:

```text
results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv
```

Important columns:

- `layer`
- `feature_id`
- `contrast_name`
- `provisional_role`
- `keep_for_intervention`
- `intervention_priority`
- `role_confidence`
- `top_axis`
- `top_identity`
- `max_abs_cohens_d`
- `template_artifact_score`
- `sharedness_score`
- `contrast_specificity_score`

## How To Run

### Prepare BBQ

```bash
python scripts/prepare_bbq_for_steering.py \
  --bbq_data_dir /workspace/status_mi/data/bbq/data \
  --triage_csv /workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv \
  --output_dir /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared/ \
  --categories Disability_status,Gender_identity,Physical_appearance,Race_ethnicity,Religion,SES,Sexual_orientation \
  --overwrite
```

### Extract BBQ token-level SAE activations

```bash
python scripts/extract_bbq_token_level_sae_activations.py \
  --model_path /workspace/status_mi/models/llama-3.1-8b \
  --sae_dir /workspace/status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x \
  --prepared_csv_or_parquet /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet \
  --triage_csv /workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv \
  --output_dir /workspace/status_mi/results/bbq_steering/llama-3.1-8b/token_level_sae/ \
  --layers 24 \
  --include_all_kept_features \
  --batch_size 32 \
  --max_length 512 \
  --save_every_batches 25 \
  --resume
```

### Rebuild filtered BBQ feature cards

Use existing token-level activations. No model rerun is needed.

```bash
cd /root/local_status_mi

python scripts/build_bbq_sae_feature_cards.py \
  --token_level_dir /root/local_status_mi/results/bbq_steering/llama-3.1-8b/token_level_sae/ \
  --prepared_data /root/local_status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet \
  --triage_csv /root/local_status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv \
  --output_dir /root/local_status_mi/results/bbq_steering/llama-3.1-8b/feature_cards_filtered/ \
  --layers 24 \
  --top_prompts_per_feature 25 \
  --top_tokens_per_feature 50 \
  --save_every_features 25 \
  --overwrite
```

### Run full per-feature steering asynchronously

Run from the fast local copy on RunPod:

```bash
cd /root/local_status_mi
mkdir -p /root/local_status_mi/run_logs

nohup python scripts/run_bbq_sae_steering.py \
  --model_path /root/local_status_mi/models/llama-3.1-8b \
  --sae_dir /root/local_status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x \
  --prepared_data /root/local_status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet \
  --triage_csv /root/local_status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv \
  --output_dir /root/local_status_mi/results/bbq_steering/llama-3.1-8b/steering_per_feature_matched_full/ \
  --layers 24 \
  --alphas=-8,-4,-2,2,4,8 \
  --feature_set_modes per_feature \
  --require_per_feature \
  --intervention_positions final_prompt_token,target_identity_last_token,stereotype_language_last_token \
  --scoring_mode letter \
  --controls_subsample_frac 0.20 \
  --batch_size 16 \
  --save_every_examples 100 \
  --resume \
  > /root/local_status_mi/run_logs/steering_per_feature_matched_full_2026-05-01.log 2>&1 &
```

Audit 2.3 (closed 2026-05-28): `--disable_controls` removed from the production command and replaced with `--controls_subsample_frac 0.20`. Specificity controls now run alongside the headline in the batched first-token path (no per-example fallback needed) on a deterministic stratified 20% subsample of `(example, feature_set)` pairs. For the audit-3.1 default `--intervention_modes ablate`, the natural control `random_feature_ablate` (ablate K random features) runs automatically; pass `--intervention_modes ablate,add_vector,direction_baseline,probe_baseline` to get the direction-shaped controls too. To skip controls for smoke tests only, pass `--disable_controls` (the runner emits a startup WARNING if it's set on a non-smoke-sized run).

Audit 1.3 (closed 2026-05-28): `--scoring_mode` switched from `first_token` to `letter`. The new `letter` mode scores the answer LETTERS ` A`/` B`/` C` at the final prompt position — single tokens, mutually distinct, matched to the prompt format (`A. {ans0} B. {ans1} C. {ans2} Answer:` makes the letter the natural continuation). Removes the first-token-of-noun-phrase degeneracy where two of three options had identical first-token logprobs ("The grandmother" vs "The boy" → identical "The" logprobs). The legacy `first_token` mode is preserved as a comparison option but should not be the headline; `answer_logprob` remains available as a confirmatory mode (its argmax/accuracy use is still length-biased — audit 2.4 — and headline rankings should not rely on it).

Audit 3.2 (closed 2026-05-28): per-feature amplitude scaling for the `clamp` and `steer` intervention modes. **The headline production command above is unchanged** — it uses the default `--intervention_modes ablate`, which sets the latent to exactly 0 and is unaffected by 3.2. The new flags matter only when you run an *amplification* pass (`clamp`/`steer`):

```bash
# Per-feature amplification headline: clamp each feature to {1,2,4}x its OWN p95.
python scripts/run_bbq_sae_steering.py \
  ... (same flags as above) ... \
  --intervention_modes clamp \
  --feature_scale_stat p95 \
  --feature_stats_dir /root/local_status_mi/results/sae_identity/llama-3.1-8b/final_token \
  --clamp_multipliers 1.0,2.0,4.0
```

Under `--feature_scale_stat p95` (the default), the clamp target is `clamp_multiplier × p95[f]` and the steer increment is `alpha × sign × p95[f]`, so the grid value means "how many p95s of this feature's own activation" for every feature — comparable across features. The chosen multiplier is recorded in the output `alpha` column, and `feature_scale_stat` / `feature_scale_value` columns are added (none removed). `--feature_scale_stat none` reproduces the pre-3.2 uniform behavior (clamp needs scalar `--clamp_value`; steer adds raw `--alphas`).

**Run the gate first.** Before any clamp/steer headline, run `scripts/audit_feature_scale.py --triage_csv <triage> --feature_stats_dir <encode_out> --layers 24 --feature_scale_stat p95`. It exits non-zero if too many kept features are absent from `feature_stats.csv` (→ scale 0) or have a non-positive stat (→ clamp degenerates to ablate, steer to a no-op) — silent-failure modes that would otherwise produce a clean-looking but zero-amplitude run.

Monitor:

```bash
tail -f /root/local_status_mi/run_logs/steering_per_feature_matched_full_2026-05-01.log
```

Stop safely:

```bash
ps aux | grep run_bbq_sae_steering | grep -v grep
kill -TERM <PID>
```

Resume by rerunning the same `nohup` command. The script reads `completed_jobs.jsonl` and skips finished jobs.

### Analyze per-feature steering results

```bash
cd /root/local_status_mi

python scripts/analyze_bbq_feature_level_causal_effects.py \
  --steering_dir /root/local_status_mi/results/bbq_steering/llama-3.1-8b/steering_per_feature_matched_full/ \
  --prepared_data /root/local_status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet \
  --triage_csv /root/local_status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv \
  --token_level_dir /root/local_status_mi/results/bbq_steering/llama-3.1-8b/token_level_sae/ \
  --output_dir /root/local_status_mi/results/bbq_steering/llama-3.1-8b/feature_level_causal_analysis_full/ \
  --layers 24 \
  --selected_alphas=-8,-4,-2,2,4,8 \
  --selected_positions final_prompt_token,target_identity_last_token,stereotype_language_last_token \
  --context_conditions ambig,disambig \
  --grouping_levels axis,contrast,identity,subgroup \
  --make_axis_reports \
  --make_identity_reports \
  --make_contrast_reports \
  --require_complete_alpha_grid \
  --min_examples 10 \
  --min_examples_per_identity 10 \
  --min_examples_per_contrast 20 \
  --top_n_features 25 \
  --bootstrap_samples 500 \
  --permutation_samples 500 \
  --smoke \
  --overwrite
```

## Current Local Results

### `output_data/bbq_steering_2026-04-30/feature_cards`

Downloaded BBQ feature cards from an earlier build.

Status:

- Better than the first card build because special tokens were removed.
- Still had low-information punctuation tokens such as `.` in top-token tables.
- Use the newer filtered card builder and rebuild as `feature_cards_filtered` before relying on cards for interpretation.

### `output_data/bbq_steering_2026-04-30/analysis_smoke`

Aggregate smoke analysis from an early run.

Status:

- Not suitable for substantive interpretation.
- The first smoke run accidentally mixed disability examples with physical-appearance feature sets before axis matching was fixed.
- Figures are retained only as historical/debug artifacts.

### `output_data/bbq_steering_2026-04-30/analysis_medium_matched_balanced`

Aggregate analysis from a balanced, matched-axis run.

Status:

- More meaningful than `analysis_smoke`.
- Still aggregates primarily by feature role/axis rather than individual feature.
- Useful for checking sign conventions and broad axis-level behavior.
- Not sufficient for final claims about individual features.

Key files:

- `README_interpretation.md`
- `coverage_report.csv`
- `figures/ambiguous_bias_margin_by_alpha_and_direction.png`
- `figures/answer_logprob_shifts_ambiguous.png`
- `figures/axis_level_bias_effects.png`
- `figures/baseline_to_intervened_answer_probs_examples.png`

### `output_data/bbq_steering_2026-05-01/steering_per_feature_matched_balanced`

Downloaded per-feature steering outputs from a balanced subset.

Status:

- Contains actual `per_feature` steering rows.
- Covers a subset of features and balanced examples rather than the full prepared dataset.
- Useful for validating feature-level analysis code.

### `output_data/bbq_steering_2026-05-01/feature_level_causal_analysis_per_feature`

First feature-level causal analysis output from per-feature steering.

Status:

- More appropriate for the scientific question than aggregate role plots.
- Early version existed before the latest identity/axis/contrast report extensions.
- Re-run `analyze_bbq_feature_level_causal_effects.py` with the latest script for final identity-aware reports.

Expected important outputs after rerun:

- `feature_level_effects.csv`
- `identity_level_effects.csv`
- `subgroup_level_effects.csv`
- `feature_effect_rankings.csv`
- `final_intervention_candidates_table.html`
- `analysis/axis_reports/`
- `analysis/contrast_reports/`
- `analysis/identity_reports/`

## Known Issues And Lessons

### Network storage on RunPod was the main performance bottleneck

The `/workspace` path is network-mounted storage. Loading the 30GB Llama model from `/workspace` was extremely slow. The fix was to copy active inputs to `/root/local_status_mi` and run from there.

Recommended pattern:

- Read model, SAE, prepared data, and scripts from `/root/local_status_mi`.
- Write active checkpoints to `/root/local_status_mi`.
- Periodically or finally copy results back to `/workspace/status_mi` for persistence.

### Early steering runs were too broad and not axis-matched

The runner originally allowed wrong-axis feature/example combinations by default. This produced uninterpretable smoke outputs. The default is now `--axis_match_mode matched_only`.

### Early smoke figures were not interpretation-ready

The first aggregate analyzer collapsed over alpha signs, feature direction, axis matching, and intervention position. The revised aggregate analyzer now adds sign conventions, coverage reports, and smoke limitations. For feature-level claims, use `analyze_bbq_feature_level_causal_effects.py`.

### Bundle steering is not individual feature causality

Per-contrast top-k runs are useful diagnostics, but they do not isolate individual features. The feature-level analyzer marks these as `feature_bundle_membership`. Individual causal claims require `--feature_set_modes per_feature`.

Audit 3.5 (closed 2026-05-28): when a bundle *is* run, prefer the feature-intervention modes (`ablate`/`clamp`/`steer`) over the legacy averaged-decoder modes. Under the audit-3.1 default `--intervention_modes ablate`, a bundle is ablated as a *set* — every latent in the bundle is zeroed simultaneously, so the effect is the joint causal effect of the set rather than the effect of one re-normalized average direction (which the old `add_vector` path produced and which is hard to interpret). Under audit 3.2, a bundle `clamp`/`steer` scales each member latent by its own `feature_stats`. The runner emits a startup WARNING if a bundle set is about to run under a legacy averaged-vector mode. The production command above sidesteps this entirely with `--feature_set_modes per_feature --require_per_feature`.

### First-token scoring is a speed-oriented approximation

The current long runs use `--scoring_mode first_token` for speed. This scores the first token of each answer option. For final results, `answer_logprob` is more faithful but much slower.

## Download Commands

Download filtered feature cards:

```bash
mkdir -p "/Users/brandonlepine/Repositories/Research_Repositories/status_mi/output_data/bbq_steering_2026-05-01" && \
scp -P 14446 -i ~/.ssh/id_ed25519 -r \
  root@64.247.196.114:/root/local_status_mi/results/bbq_steering/llama-3.1-8b/feature_cards_filtered \
  "/Users/brandonlepine/Repositories/Research_Repositories/status_mi/output_data/bbq_steering_2026-05-01/"
```

Download full per-feature steering output:

```bash
mkdir -p "/Users/brandonlepine/Repositories/Research_Repositories/status_mi/output_data/bbq_steering_2026-05-01" && \
scp -P 14446 -i ~/.ssh/id_ed25519 -r \
  root@64.247.196.114:/root/local_status_mi/results/bbq_steering/llama-3.1-8b/steering_per_feature_matched_full \
  "/Users/brandonlepine/Repositories/Research_Repositories/status_mi/output_data/bbq_steering_2026-05-01/"
```

Download full feature-level causal analysis:

```bash
mkdir -p "/Users/brandonlepine/Repositories/Research_Repositories/status_mi/output_data/bbq_steering_2026-05-01" && \
scp -P 14446 -i ~/.ssh/id_ed25519 -r \
  root@64.247.196.114:/root/local_status_mi/results/bbq_steering/llama-3.1-8b/feature_level_causal_analysis_full \
  "/Users/brandonlepine/Repositories/Research_Repositories/status_mi/output_data/bbq_steering_2026-05-01/"
```

## Recommended Next Steps

1. Let the async full per-feature matched steering run complete.
2. Run the latest feature-level analyzer on `steering_per_feature_matched_full`.
3. Rebuild filtered BBQ feature cards from token-level activations.
4. Download `feature_level_causal_analysis_full` and `feature_cards_filtered`.
5. Inspect `validation_summary.csv`, `coverage_report.csv`, `feature_level_effects.csv`, `identity_level_effects.csv`, and `final_intervention_candidates_table.html`.
6. Only after first-token results look coherent, consider a smaller confirmatory `answer_logprob` run for top features and contrasts.
