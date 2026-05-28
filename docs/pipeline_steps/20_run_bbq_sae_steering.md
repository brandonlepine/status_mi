# Step 20 — `scripts/run_bbq_sae_steering.py`

**Stage:** 4 — BBQ steering and feature-level causal analysis
**Runs after:** [Step 18 — `prepare_bbq_for_steering.py`](18_prepare_bbq_for_steering.md), [Step 17 — `triage_sae_identity_features.py`](17_triage_sae_identity_features.md).
**Feeds into:** [Step 22 — `analyze_bbq_steering_results.py`](22_analyze_bbq_steering_results.md) (legacy), [Step 23 — `analyze_bbq_feature_level_causal_effects.py`](23_analyze_bbq_feature_level_causal_effects.md) (substantive).

This is the central causal-intervention engine of the project. If a single script's correctness determines whether the paper has a feature-level causal claim, it is this one.

## Purpose
Run a grid of steering interventions on the prepared BBQ examples, where each intervention adds (or projects out) a unit-norm direction built from one or more SAE decoder rows at a chosen transformer layer, at a chosen token position, for a chosen magnitude `alpha`, and scores the three answer choices before and after the hook. Aggregated downstream, the per-row deltas (`stereotyped_delta`, `unknown_delta`, `bias_margin_delta`, `correct_delta`, ...) underpin every causal-feature claim.

## Inputs
- Llama model directory and OpenMOSS SAE directory.
- `prepared/bbq_prepared_examples.parquet` — Step 18 output. Filtered by `--mapping_confidence_filter` (default `exact`; choices `exact`, `exact_and_fallback`, `all`). Audit 3.4 closed 2026-05-27 in commit `56a5f7e`; `mapped_contrast_confidence` is stamped on every output row so the analyzer can stratify regardless of which filter was used.
- `results/.../triage/intervention_candidate_features_triaged.csv` — the feature pool. Filtered by `keep_for_intervention == True` and the requested `--layers`.

## Outputs
```
steering/
  steering_config.json
  steering_manifest.csv             # one row per feature set with eligible_examples count
  results_parts/part_XXXXX.parquet  # one row per (example, feature_set, alpha, position, mode)
  completed_jobs.jsonl              # SHA1-prefix-16 job_ids for --resume
  logs/steering.log
```

Each row in `results_parts/*.parquet` carries: `bbq_uid`, `layer`, `alpha`, `intervention_mode`, `intervention_position`, `feature_set_mode`, `feature_set_id`, `feature_id` (or `-1` for bundles), `feature_role`, `feature_axis`, `feature_estimate_type` (`individual_feature` | `feature_bundle`), `feature_ids_json`, `feature_signs_json`, `mapped_contrast_name`, `feature_contrast_name`, `axis_mapped`, `context_condition`, `question_polarity`, per-answer base + intervened logprobs, the derived deltas, `predicted_base`/`predicted_intervened`, `correct_base`/`correct_intervened`, `prediction_changed`, and `control_type ∈ {kept_feature, wrong_axis_features, template_artifact_features, sign_flip, random_direction_norm_matched, random_feature_matched}`.

## Key implementation details

### Feature-set construction (`load_feature_sets`)
- Reads triage, filters to `keep_for_intervention=True`, `priority_sort`s by `intervention_priority` (`high < medium < low`) then `role_confidence`, `max_abs_cohens_d`, `combined_score`.
- Emits feature sets in up to three modes:
  - **`per_feature`**: one `FeatureSet` per `(layer, feature_id)`. Single-element `feature_ids = [f]`, `signs = [feature_sign(row)]`. The only mode that supports clean individual-feature causal estimates.
  - **`per_contrast_topk`**: top-k by priority within each `(layer, contrast_name)`, both per-role and combined-across-roles. Bundles.
  - **`role_bundle`**: all features in a `(layer, top_axis, provisional_role)` bucket. Bundles.
- Always also emits a `template_or_syntax_artifact` "control bundle" (top-20 artifact features per layer) with `control_type = "template_artifact_features"`.
- `feature_sign(row)` reads `direction_side` ∈ `{negative, minus, b, identity_b, -1}` first, then falls back to the sign of `decoder_cosine`/`cohens_d`/`diff_mean`. Default `+1`.

### Axis matching
- `--axis_match_mode matched_only` (default and production): a feature set with axis `A` is only applied to BBQ examples with `axis_mapped == A`. `eligible_prepared_for_feature_set` does this filter per feature set.
- `--axis_match_mode all`: every feature set is applied to every example; the runner stamps `control_type = wrong_axis_features` on any row whose feature axis disagrees with `axis_mapped`. Used historically for wrong-axis sweeps that produced uninterpretable smoke output.

### Vector construction (`make_vector`)
```python
dec = w_dec[feature_ids]               # (n_features_in_set, hidden_dim)
if normalize:
    dec = dec / ||dec||_row            # row-normalize (default; --no_normalize_features disables)
signed = dec * signs[:, None]
vec = signed.mean(dim=0)               # weighted-mean direction
vec = vec / ||vec||                    # unit-norm the final direction
```
The steering "direction" is **always unit-norm**, so the `alpha` grid is the magnitude of the additive perturbation in hidden-state units. For `random_feature_matched`, a random set of decoder rows of the same cardinality is averaged the same way. For `random_direction_norm_matched`, a single `randn(hidden_dim)` is unit-normed.

### Hook installation
- `install_hook` / `install_batched_hook` register a forward hook on `model.model.layers[layer - 1]` — i.e. the transformer block whose *output* is `hidden_states[layer]`. This matches the OpenMOSS LkR convention.
- Two `intervention_mode`s:
  - `add_vector`: `h[:, pos, :] += alpha * vec`
  - `ablate_projection`: `h[:, pos, :] -= alpha * (h[:, pos, :] · unit) * unit` (project out the direction, scaled by `alpha`).
- `install_batched_hook` accepts per-example position lists so a whole prompt-batch can be hooked in one forward pass — used in the fast `--scoring_mode first_token --disable_controls` path.

### Position selection (`positions_for`)
- `final_prompt_token`: the last non-pad token (`attention_mask.sum() - 1` minus offsets that map to empty strings).
- `target_identity_last_token` / `nontarget_identity_last_token` (legacy): regex-locate all spans matching the identity label *and* the relevant answer-option text via `find_spans`, return `max(pos)` — the last token overlapping any match anywhere in the prompt. Frequently lands in the answer-option list rather than the context (the audit's 3.3 failure mode); the new section-explicit variants below should be preferred for any "this feature acts at the identity mention" claim.
- `stereotype_language_last_token` (legacy): the last token overlapping any content word from the BBQ `question` (after removing the local `stop` set).
- **Section-explicit variants (audit 3.3 fix, 2026-05-28, commit `afb3ee3`):** `target_identity_last_context_token` / `target_identity_last_question_token`, plus `nontarget_*` and `stereotype_language_*` counterparts. These clip the term-match spans to the named section via `find_section_spans` + `intersect_spans_with_section` BEFORE taking the last-token argmax. Fall back to `final_prompt_token` (with `intervention_section = "final"` on the output row) when no in-section match exists.
- `all_identity_tokens` / `all_stereotype_language_tokens`: every overlapping token.
- Every `results_parts/*.parquet` row carries an `intervention_section` column (`context` / `question` / `answer_option` / `final` / `mixed` / `unknown`), stamped by `position_section_for()` after the hook runs. Downstream analyzers can `groupby("intervention_section")` to stratify any effect table.

### Scoring modes
- **`first_token`**: `score_first_token` / `score_first_token_batch` take a single forward pass on the prompt (no answer appended), then read `log_softmax(logits[final_pos])` and gather the logprob of `first_token_ids(" " + answer)[0]` for each of `[ans0, ans1, ans2]`. Fast and batchable. **Used in the documented production long run** (`--scoring_mode first_token --disable_controls`).
- **`answer_logprob`**: for each of the three answers, append `" " + answer`, forward, and sum per-token logprobs over the answer span. Three forwards per example × per hook-installation. Slow, but uses true continuation likelihood.

### Job IDs and resume
- `job_id = sha1("|".join([bbq_uid, layer, set_id, alpha, position, mode, scoring_mode]))[:16]`.
- `completed_jobs.jsonl` holds one `{job_id, completed_at}` JSON object per finished job. `--resume` rebuilds the `done` set and skips matching jobs. Malformed lines are backed up to `completed_jobs.jsonl.malformed` and dropped.

### Controls (gated behind not `--disable_controls`)
At `final_prompt_token` only, for each `(example, feature_set, alpha)`:
- `sign_flip`: flip the sign of every feature in the set, rebuild `vec`, hook, score. Tests "is the effect *direction*-specific."
- `random_direction_norm_matched`: `randn(hidden_dim) / ||·||`, scored at the same alpha. Tests "is the effect specific to *this* direction vs. any norm-matched direction."
- `random_feature_matched`: average decoder rows of the same cardinality as the feature set, randomly selected. Tests "is the effect specific to *these* SAE features vs. a random matched set of SAE features."

## Issues & Opportunities

> **Upstream callout — issue 1.4 (FIX LANDED in Step 5; this script's feature pool + future 3.1 fix still TODO).** The encoder fix in [Step 5](05_encode_identity_saes.md#14-blocker--sae-preprocessing-convention-fix-landed-2026-05-26) landed in commit `4b8851a`. Two pieces of this script are downstream:
>
> 1. **The feature pool** — `keep_for_intervention=True` rows come from `intervention_candidate_features_triaged.csv` ([Step 17](17_triage_sae_identity_features.md)), which selects features based on the encoder's outputs. Re-run [Step 5](05_encode_identity_saes.md) → [Step 6](06_validate_sae_hook_alignment.md) → [Step 13](13_analyze_identity_sae_features.md) → [Step 17](17_triage_sae_identity_features.md) before consuming a new pool; the current list of features may not even contain the same `feature_id`s.
> 2. **The 3.1 feature-level fix (LANDED 2026-05-27)** — the encode → modify latent → decode → patch loop now runs through `encode_full` / `decode_full` from `encode_identity_saes.py` via the new `patched_residual_with_intervention(h, sae, intervention)` helper. The change in the residual stream is computed in normalized space and un-scaled by `decode_full` before patching; SAE reconstruction error cancels in the delta. See the 3.1 section below for the full description.
>
> The current decoder-direction-addition path (`make_vector` → `h[:, pos, :] += alpha * vec`) does not itself encode anything, so it is not directly broken by 1.4. But the *feature identity* of the vectors being added is wrong until the upstream chain is regenerated, and the *scale* of `alpha` is in un-normalized residual units while the SAE operates in normalized space — these are entangled with 3.2.

### 3.1 [BLOCKER] — Feature-level intervention (FIX LANDED 2026-05-27)

**Status:** Closed across commits `11d4a4d` (canonical torch primitives) and `84c87b5` (BBQ steering hook dispatch). The script now has both the legacy decoder-direction modes (for the audit 5.5 baseline) and the genuine encode → modify-latent-f → decode → patch loop.

**What landed:**
- New torch primitives in [Step 5 — `encode_identity_saes.py`](05_encode_identity_saes.md): `ablate_features`, `clamp_features`, `steer_features`, plus the wrapper `patched_residual_with_intervention(h, sae, intervention_fn)` that runs the full loop. The wrapper accounts for the audit 1.4 dataset-wise normalization: `decode_full` un-scales the reconstruction back to residual space, so the patch math operates on the model's natural hidden-state scale. SAE reconstruction error cancels in the delta because only the change induced by the intervention is added.
- This script now exposes `install_feature_intervention_hook` + `install_batched_feature_intervention_hook`. The forward hook on `model.model.layers[layer-1]` captures `h`, encodes through the corrected JumpReLU encoder, applies the intervention (ablate / clamp / steer), decodes back to residual space, and patches the residual with the delta.
- `make_batched_hook_fn` / `make_hook_fn` are dispatch factories that take `mode` and return the right hook closure. Legacy modes (`add_vector`, `ablate_projection`) still get the precomputed decoder direction; feature modes pass the full SAE and `feature_ids` / `signs` through.
- `--intervention_modes` default changed from `add_vector` to **`ablate`** (audit's recommended primary causal test; no alpha grid needed). Valid modes: `{add_vector, ablate_projection, ablate, clamp, steer, direction_baseline, probe_baseline}`. Startup validation raises on unknown modes. `direction_baseline` and `probe_baseline` are the audit-5.5 linear-baseline modes — see the 5.5 section below.
- `--clamp_value` flag added (required when `clamp` is in `--intervention_modes`). In normalized latent space units; user looks up the per-feature target in `feature_stats.csv` (p95 / p99 / max).
- `alpha_grid_for_mode` special-cases `ablate` / `clamp` to a single alpha=0 sentinel; the alpha grid only sweeps for `steer` / `add_vector` / `ablate_projection`.
- Vector cache build is skipped when no legacy mode is requested and controls are disabled.

**Synthetic validation (5/5 pass):** hook output identical to direct primitive call (max diff 0.00); batched and non-batched variants agree; dispatch factory routes correctly; alpha-grid special-case fires; unknown modes raise at the factory.

**Original audit (preserved):** The hook did `h[:, pos, :] += alpha * vec` regardless of whether the SAE feature was active on the example. That was a direction intervention, not a feature intervention. The headline claim — "we identified SAE features causally implicated in social bias" — collapsed to "we identified *directions* causally implicated in bias", and a reviewer would reasonably ask why an SAE was needed at all rather than a difference-of-means or probe direction.

**Usage (RunPod):**
```bash
# Primary causal test: ablate the feature(s) on the kept-for-intervention set.
python scripts/run_bbq_sae_steering.py \
    --layers 24 --feature_set_modes per_feature --require_per_feature \
    --intervention_modes ablate \
    --intervention_positions final_prompt_token,target_identity_last_token,stereotype_language_last_token

# Comparison: legacy decoder-direction add_vector run alongside ablate.
python scripts/run_bbq_sae_steering.py \
    --intervention_modes ablate,add_vector ...

# Audit 5.5 head-to-head: SAE feature ablation vs the two linear baselines.
python scripts/run_bbq_sae_steering.py \
    --intervention_modes ablate,direction_baseline,probe_baseline \
    --direction_baselines_path /workspace/status_mi/results/geometry/.../contrasts \
    --probe_baselines_path    /workspace/status_mi/results/geometry/.../contrasts
```

### 5.5 [MAJOR] — Linear-baseline modes for "does the SAE beat a single direction?" (FIX LANDED 2026-05-27 / 2026-05-28)

**Status:** Three-way head-to-head infrastructure landed across four commits — `8f84e5e` (Step 7 persists DoM directions) + `a11cbb8` (Step 20 adds `direction_baseline`) + `7cdb164` (Step 7 persists logistic-probe directions — audit option (c)) + `8c392d7` (Step 20 adds `probe_baseline`). The RunPod head-to-head run is the remainder.

**What landed:**
- Two new intervention modes that share the `h += alpha * vec` plumbing (same as `add_vector`); only the source of the vector differs. Output rows stamp `intervention_mode` so the analyzer can stratify SAE vs DoM vs probe head-to-head.
  - `direction_baseline` — `vec` is the unit-norm difference-of-means contrast direction from [Step 7](07_analyze_identity_geometry.md)'s `contrast_directions_layer_*.npz`.
  - `probe_baseline` — `vec` is the unit-norm logistic-probe weight vector (binary L2 regression on `(identity_a, identity_b)` in raw `d_model` space) from [Step 7](07_analyze_identity_geometry.md)'s `contrast_probe_directions_layer_*.npz`.
- Two parallel CLI flags: `--direction_baselines_path` and `--probe_baselines_path`. Each is required when its mode is requested. Both accept either a single `.npz` file or a directory; when given a directory, the loader globs the matching filename pattern so the two sources stay disjoint even if both files live in the same directory.
- `load_contrast_directions(path, glob_patterns=…)` returns `{(layer, contrast_name): unit-norm direction}`. `make_direction_baseline_vector(directions, fs, device)` does the per-FeatureSet lookup; bundle-mode feature sets (empty `contrast_name`) are skipped with a logged warning — the baseline is defined per-contrast.
- `make_batched_hook_fn` / `make_hook_fn` accept both `baseline_vector` and `probe_vector` kwargs; the helper `_baseline_vector_for_mode(mode, baseline_vector, probe_vector)` picks the right source. Missing-vector errors mention the specific `--*-path` flag that needs to be set.

**Validation (synthetic):**
- DoM loader round-trips multi-layer .npz directories; missing-contrast lookup → None; bundle FeatureSet → None.
- Probe fit recovers the true separating direction with cosine 0.99 on a 60-prompt × `d_model=64` toy (AUC = 1.000, Cohen's d ≈ 16, held-out family AUC = 1.000).
- Hook math: `direction_baseline` adds the DoM vector, `probe_baseline` adds the probe vector, both at the chosen positions with max diff = 0.00 from expected.
- Loader correctly disambiguates DoM vs probe by glob pattern.

**Remaining (RunPod):** Run the three-way command (above) against the (audit-1.4 re-encoded) feature pool. The analyzer can then answer "for each (layer, contrast), does the SAE feature ablation produce a stronger bias-reducing effect than the DoM direction or the logistic probe at the same positions?" If SAE features do not beat both linear baselines, the paper should be reframed around directions instead — the audit's framing note flags this as load-bearing for the SAE story.

**Original audit (preserved):** Throughout the project, the difference-of-means contrast direction was computed *and* SAE features were computed, but they were never put in head-to-head competition as *interventions*. The key scientific question for an SAE-based paper is "does decomposing into SAE features buy anything over a single linear direction?" — and there was no path to answer it.

### 2.3 [BLOCKER] — Steering controls are disabled in the production run

**What's wrong:** The script implements three controls (`sign_flip`, `random_direction_norm_matched`, `random_feature_matched`) but they are gated behind `not args.disable_controls`. The documented production command in `docs/bbq_steering_pipeline.md` explicitly passes `--disable_controls`, so the production results in `steering_per_feature_matched_full/` carry **no** specificity controls.

**Why it matters:** Without controls, "feature X is causally implicated in BBQ bias" cannot be distinguished from "any norm-matched steering vector at this position would shift the logits about this much." Specifically:
- `sign_flip` shows the effect depends on the sign of the direction, not just its magnitude.
- `random_direction_norm_matched` shows the effect is not purely from injecting *any* unit vector at this position.
- `random_feature_matched` shows the effect is specific to *these* SAE features and not a property of decoder rows in general.

The whole "feature X is causally implicated" claim requires effect(X) ≫ effect(random direction) ≫ effect(random feature set), at matched norm.

**Targeted fix:**
- Re-enable controls for the final run by removing `--disable_controls` from the production command.
- If cost is the bottleneck: run controls on a stratified subsample of `(example, feature)` pairs rather than dropping them. Cost dominates with per-example, per-alpha controls; batching `random_feature_matched` across feature sets (since the vector is independent of the kept feature set) can amortize cost.
- Add a fourth control: the **raw difference-of-means contrast direction** from the geometry pipeline. ✅ **Closed 2026-05-27 (audit 5.5)** — the new `direction_baseline` mode loads pre-computed contrast directions from Step 7's `contrast_directions_layer_*.npz` via `--direction_baselines_path` and steers identically. Run `--intervention_modes ablate,direction_baseline` for the head-to-head. If SAE features do not beat the DoM direction, the SAE is not adding causal value over a linear probe and the paper should be reframed around directions.

### 1.3 [MAJOR] — First-token answer scoring is degenerate for BBQ answers

**What's wrong:** `score_first_token` (and the batched variant) scores the log-probability of the first token of `" " + answer`. BBQ answers are noun phrases; many share a leading word (`"The grandmother"`, `"The boy"`, `"Cannot be determined"` all start with `"The"`/`"Cannot"`). When two options share a first token, their first-token logprobs are identical and the metric cannot distinguish them. Separately, the prompt presents labelled choices `A. / B. / C.` ending in `"Answer:"` — the natural model continuation is the *letter*, but scoring targets the answer text. **`--scoring_mode first_token` is the documented production setting.**

**Why it matters:** Every first-token-text result on BBQ items where two options share a leading word is structurally noise. `bias_margin_delta` is computed from these logprobs; `predicted_base`/`predicted_intervened` argmax is computed from them. Downstream `accuracy_delta`, `prediction_changed`, and any base accuracy number inherit the degeneracy.

**Targeted fix:**
- Add a `--scoring_mode letter` (or rename `first_token` and add the letter mode) that tokenizes `" A"`, `" B"`, `" C"` (each is typically a single Llama token, and they are mutually distinct). The prompt ends in `"Answer:"`, so the letter is also the natural continuation — this aligns prompt design with scoring target.
- Keep `answer_logprob` as a confirmatory mode for top features after the letter-based pass selects them, **but length-normalize** it (see issue 2.4).
- Until the rerun, label all first-token-text results "preliminary" in any plot/table.

### 2.4 [MAJOR] — `answer_logprob` summed over different-length answers biases argmax metrics

**What's wrong:** `score_answer_logprob` sums per-token logprobs over the answer span. BBQ options have different token lengths (`"Cannot be determined"` is usually the longest), so the raw summed logprob systematically penalizes the long unknown option. Within-example *deltas* (intervened − base) cancel because length is constant per example — so `stereotyped_delta`, `unknown_delta`, `bias_margin_delta` etc. are unbiased. But `predicted_base`, `predicted_intervened`, `correct_base`, `correct_intervened`, `prediction_changed`, and any `accuracy_delta` derived from them use `argmax` over the raw summed logprobs and **are** length-biased.

**Why it matters:** Baseline accuracy and any accuracy-change number — which are reported in the legacy aggregator (Step 22) and the feature-level analyzer (Step 23) — are systematically biased toward shorter options. The unknown option is disproportionately predicted as the model's least-likely choice.

**Targeted fix:**
- For `argmax`/accuracy computations in `row_metrics`, length-normalize (mean per-token logprob over the answer span). Add `len_normalized_score` and use it for `predicted_base`/`predicted_intervened`/`correct_*`.
- Better: switch the headline scoring to letter (issue 1.3); letters all have constant length and the bias dissolves entirely.

### 3.1 cross-reference and 3.2 [MAJOR] — Steering magnitude is uniform and untethered to feature scale (PARTIAL: 3.1 path unblocked, scale tethering still STILL OPEN)

**What's wrong:** Even within the current direction-addition design, `--alphas=-8,-4,-2,2,4,8` is applied to a unit vector identically for every feature. Features differ in natural activation magnitude (some `feature_stats.p95` are 0.1, others 5+) and decoder norm. A constant `alpha` is neither "amplify feature `f` by X%" nor a fixed fraction of the residual-stream RMS norm at that layer/position (which is itself ~3–10× the unit vector). So one feature's "big effect at α=8" might just reflect a *relatively larger* perturbation than another's "small effect at α=8".

The same problem now applies to the new `--intervention_modes clamp` and `--intervention_modes steer` codepaths: `--clamp_value` is a single scalar applied across all features regardless of each feature's `p95`/`p99`. The `ablate` mode is unaffected (it sets the latent to exactly 0, which is on its own meaningful scale).

**Why it matters:** Cross-feature effect-size comparisons in Step 23's `feature_effect_rankings.csv` and `final_intervention_candidates_table.html` are not on a common scale; rankings reflect a mixture of feature importance and feature-specific scale mismatch.

**Targeted fix:**
- The infrastructure for option 1 below is now available (3.1 landed). Wiring `feature_stats.csv` into the clamp/steer paths is the next step:
  1. **Best (paired with the 3.1 fix — now landed)**: pass a per-feature clamp value derived from each feature's own `p95`/`p99`/`max` in `feature_stats.csv` — then `alpha` becomes "feature ablated" (0) or "feature amplified to k×p95" (k ∈ {1, 2, 4}). Requires extending `--clamp_value` to accept a per-feature map (e.g. `--clamp_values_from_stats feature_stats.csv:p95` and a `--clamp_multiplier` grid).
  2. **Within the current direction-addition design**: scale `alpha` to a fixed fraction (e.g. {2%, 5%, 10%, 20%}) of the median residual-stream RMS norm at that layer/position, computed on a stratified BBQ sample before the run starts. Save the per-layer norm in `steering_config.json` and use it as the scaling constant.

### 3.3 [MAJOR] — Intervention positions are located by greedy regex and may hit the wrong span (FIX LANDED 2026-05-28)

**Status:** Closed in commit `afb3ee3` across this script and [Step 19](19_extract_bbq_token_level_sae_activations.md).

**What landed:**
- Six new section-explicit position names alongside the legacy ones:
  - `target_identity_last_context_token` / `target_identity_last_question_token`
  - `nontarget_identity_last_context_token` / `nontarget_identity_last_question_token`
  - `stereotype_language_last_context_token` / `stereotype_language_last_question_token`
- New helpers: `find_section_spans` (mirrors [Step 19](19_extract_bbq_token_level_sae_activations.md)'s) plus `intersect_spans_with_section(term_spans, section_span)` and `position_section_for(tokenizer, prompt, row, max_length, positions)`. The first two are used by `positions_for` to clip the term-match spans to the named section BEFORE taking the last-token argmax; the third classifies where the chosen position(s) actually landed (`context` / `question` / `answer_option` / `final` / `mixed` / `unknown`).
- New `intervention_section` column on every `results_parts/*.parquet` row, stamped by both call sites (batched first-token path and per-example scoring path). The downstream analyzer can `groupby("intervention_section")` to stratify causal claims by where the hook actually fired.
- When a section-explicit position has no in-section match in a given prompt, `positions_for` falls back to `final_prompt_token` and the output row's `intervention_section = "final"` — operators can filter to detect this regime.
- Legacy `target_identity_last_token` / `nontarget_identity_last_token` / `stereotype_language_last_token` positions are preserved for backward-compatible comparison runs; new code should prefer the `_context_token` variants for any "this feature acts at the identity mention" claim.

**Validation (synthetic, audit's pathological prompt):**

| Position name | Token chosen | `intervention_section` |
| --- | --- | --- |
| `target_identity_last_token` (legacy) | answer-option B | `answer_option` (confirms audit's failure mode) |
| `target_identity_last_context_token` (new) | context | `context` |
| `stereotype_language_last_question_token` (new) | question | `question` |
| `final_prompt_token` | final | `final` |
| `target_identity_last_context_token` on no-context-match prompt | final (fallback) | `final` |

**Original audit (preserved):** `positions_for` called `find_spans(prompt, [target_identity_label, ans{target_answer_idx}])`, which returned **all** regex matches anywhere in the prompt, then returned `max(pos)` for the `*_last_token` variants. The identity label often appears in the context, the question, and the answer option `B. the Black man` — and the *last* occurrence is the answer-choice list. So `target_identity_last_token` was frequently intervening on the identity inside the answer option, not the identity in the context. `stereotype_language_last_token` had the same problem (question content words also recur in context/answers). The downstream Step 23 README treated these positions as answering different causal questions, but only if they landed where the names imply.

#### ⚠️ Pre-RunPod validation caveats (3.3 fix)

The synthetic test that landed with commit `afb3ee3` confirmed the helpers on **one** prompt using a **whitespace-splitting fake tokenizer**. Production behavior with the real Llama tokenizer + the real BBQ prompt format was **not** validated locally. The same gaps the [Step 19 doc](19_extract_bbq_token_level_sae_activations.md#-pre-runpod-validation-caveats-33-fix) lists apply here, plus one steering-specific concern:

**Steering-specific gap:**
- **Silent fallback to `final_prompt_token`.** When a section-explicit position has no in-section match (e.g. because `find_section_spans` couldn't locate the section in the prompt — see step 19 caveats), `positions_for` returns `[final_pos]` and `intervention_section` ends up `"final"`. If the prompt-format mismatch is systematic (e.g. caused by `--few_shot_pool`), **every** `target_identity_last_context_token` job will silently degrade into a `final_prompt_token` intervention — the causal claim "the feature acts at the context identity mention" then rests on rows that actually intervened at the period token. Operators must check the `intervention_section` distribution before reading the results.

**Recommended pre-RunPod check:** run `scripts/audit_intervention_sections.py` before any steering job that uses the new section-explicit positions. The script samples ~50 prepared BBQ rows, runs `positions_for` against the real Llama tokenizer for every section-explicit position name, and tallies the `intervention_section` distribution. Exits non-zero if >20% of any section-explicit position silently falls back to `final` (default threshold). See [Step 19 doc — Pre-RunPod validation caveats](19_extract_bbq_token_level_sae_activations.md#-pre-runpod-validation-caveats-33-fix) for the full check list + usage. After the first steering job lands, also group its `results_parts/*.parquet` rows by `(intervention_position, intervention_section)` and confirm the cross-tab matches what the audit script reported.

### 3.4 [MAJOR] — BBQ→SAE contrast mapping silently uses axis-fallback (FIX LANDED 2026-05-27)

**Status:** Closed in commit `56a5f7e`.

**What landed:**
- `--include_unmapped` boolean replaced with `--mapping_confidence_filter` (default `exact`; choices `exact`, `exact_and_fallback`, `all`). Headline runs use the default and silently drop `fallback_axis` rows. Operators who want fallback rows for a stratified analysis pass `exact_and_fallback`; full inclusive passes `all`. The runner prints the kept-row count and per-confidence breakdown to stdout so the operator sees what was dropped.
- `mapped_contrast_confidence` is stamped on every output row in `steering_output_row` (next to `mapped_contrast_name`). `results_parts/*.parquet` now carries the column natively — stratifying any effect table in [Step 23](23_analyze_bbq_feature_level_causal_effects.md) by mapping confidence is a single `groupby` away.

**Behavior change:** previously, `--include_unmapped` unset kept `{exact, alias, fallback_axis}` (effectively `{exact, fallback_axis}` since `alias` never actually flows out of `map_contrast`). Under the new default the runner keeps `exact` only. Anyone relying on the prior superset for a comparison run passes `--mapping_confidence_filter exact_and_fallback`.

**Original audit (preserved):** The default filter kept `mapped_contrast_confidence ∈ {exact, alias, fallback_axis}`. The `fallback_axis` path meant a BBQ row about `race_arab vs race_white` could be steered with features selected for `race_black vs race_white`, and the downstream analyzer treated `mapped_contrast_name` as the relevant contrast for that example. Once a row was stamped with a fallback-mapped contrast, it was indistinguishable in `results_parts/*.parquet` from an `exact` row unless `mapped_contrast_confidence` was carried through and stratified on — and it wasn't. Full discussion in [Step 18](18_prepare_bbq_for_steering.md) issue 3.4.

### 3.5 [MINOR] — Bundle steering averages decoder rows into one direction

**What's wrong:** `per_contrast_topk` and `role_bundle` modes build `vec = unit-norm mean of signed unit-normed decoder rows`. The downstream analyzer correctly tags these `feature_bundle_membership` and warns. But averaging unit-normed rows and then re-normalizing produces a direction whose relation to any individual feature is weak — bundle effects are hard to interpret even as "membership."

**Why it matters:** As long as bundle modes are run, the temptation to read them as "the role/contrast has effect X" is real. With the 3.1 feature-level fix, bundle interventions become "clamp this *set* of latents simultaneously", which is cleaner and interpretable.

**Targeted fix:**
- Once the feature-level intervention (3.1) lands, redefine bundle interventions as "clamp every latent in the set to 0 (or to p95)" rather than averaged-decoder addition. The bundle effect becomes the joint causal effect of the set.
- Alternatively, deprecate bundle modes for the final paper and run `per_feature` only. The production command already passes `--require_per_feature`.

## Rebuild checklist
- [x] Implement encode → modify-latent → decode → patch as a new intervention mode (`ablate` / `clamp` / `steer`); reuse the canonical torch primitives in `scripts/encode_identity_saes.py`. Ablate is the default primary causal test. *(Done 2026-05-27: commits `11d4a4d`, `84c87b5`.)*
- [ ] Re-enable controls for the final run; if cost is a concern, run them on a stratified subsample, not on no examples. Add a `--controls_subsample_frac` argument.
- [ ] Add a `direction_baselines` control that steers with the difference-of-means contrast direction from `analyze_identity_geometry.py` — same prompts, same alphas, same positions. This is the SAE-vs-linear-direction comparison the paper needs (5.5).
- [ ] Switch default `--scoring_mode` to a new `letter` mode (` A`/` B`/` C`); keep `first_token` for backward compatibility but mark deprecated. Length-normalize `answer_logprob` for argmax/accuracy.
- [ ] Restrict identity- and stereotype-language position search to the prepared `context` (and optionally `question`) span; stamp the actual section onto each output row.
- [x] Change default keep-list to `exact`-only; carry `mapped_contrast_confidence` into every `results_parts` row. *(Done 2026-05-27: commit `56a5f7e` — `--mapping_confidence_filter` default `exact`; column stamped on each output row.)*
- [ ] Scale `alpha` (and the new `--clamp_value`) to either a multiple of the feature's own `p95` (now unblocked by the 3.1 fix) or a fixed fraction of the layer-position residual RMS norm; record the scaling constant in `steering_config.json`.
- [ ] Add `intervention_section` column to the output ∈ `{context, question, answer_option, final}`.
- [ ] Deprecate bundle modes for headline results; keep them only as joint-clamp diagnostics once 3.1 lands.
- [ ] Document the inference-grid problem (alpha × position × feature) and decide on a pre-registered unit-of-inference for the headline statistic (one statistic per feature; see Step 23 issue 2.6).

## Notes from the doc audit
- `result_feature_metadata` always uses `feature_set.feature_ids[0]` and `feature_set.roles[0]` for the single-element case, but for bundles writes `feature_id = -1` while still emitting a `feature_role` of `""`. Downstream `expand_feature_rows` (Step 23) re-derives `feature_id` from `feature_ids_json` and `feature_role` from `feature_roles_json`, so the bundle `feature_id = -1` is overwritten. Still, the placeholder `-1` survives in `feature_level_pre_fdr.csv` for any row that the expander did not touch — worth filtering or marking as bundle explicitly.
- The `random_direction_norm_matched` control uses `torch.manual_seed(seed)` (Python torch RNG) but `make_vector(random_direction=True)` calls `torch.randn` *without* an explicit generator, so the seed-setting is process-global. If the model is mid-forward (it shouldn't be) or any other code touches the torch RNG between seed-set and the `randn` call, the control vector is not reproducibly tied to the job_id. Pass an explicit `torch.Generator` instead.
- `score_answer_logprob` uses `start = max(0, min(prompt_len - 1, labels.shape[1] - 1))` — this is fine when the appended `" " + answer` extends the prompt, but if `truncation=True` kicks in at `max_length=512` and the answer is partially clipped, the summed logprob is over a partial answer. Add a length-check warning.
- *(superseded 2026-05-27)* The `intervention_mode` flag is now `--intervention_modes` (comma-separated), defaults to `ablate`, and accepts `{add_vector, ablate_projection, ablate, clamp, steer}`. See the 3.1 section for the dispatch table; the new feature modes use `install_feature_intervention_hook` / `install_batched_feature_intervention_hook`. Unknown modes raise at startup.
- The control branches (`sign_flip`, `random_*`) only fire under `--scoring_mode answer_logprob` because the fast `--scoring_mode first_token --disable_controls` path is in `run_first_token_batched_feature_set`, which has no control code at all. So even with `--disable_controls` removed, the user must also drop `--scoring_mode first_token` to actually run controls — a footgun worth surfacing in the help text and the operational doc.
- `count_pending_main_jobs` iterates over `fs_prepared.iterrows()` in a triple-nested loop for an O(n_examples × alphas × positions × modes) walk before the model is loaded. On large datasets this is observably slow at startup — acceptable but worth a `--skip_pending_count` flag for resume runs.
