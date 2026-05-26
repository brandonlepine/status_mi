# Step 20 — `scripts/run_bbq_sae_steering.py`

**Stage:** 4 — BBQ steering and feature-level causal analysis
**Runs after:** [Step 18 — `prepare_bbq_for_steering.py`](18_prepare_bbq_for_steering.md), [Step 17 — `triage_sae_identity_features.py`](17_triage_sae_identity_features.md).
**Feeds into:** [Step 22 — `analyze_bbq_steering_results.py`](22_analyze_bbq_steering_results.md) (legacy), [Step 23 — `analyze_bbq_feature_level_causal_effects.py`](23_analyze_bbq_feature_level_causal_effects.md) (substantive).

This is the central causal-intervention engine of the project. If a single script's correctness determines whether the paper has a feature-level causal claim, it is this one.

## Purpose
Run a grid of steering interventions on the prepared BBQ examples, where each intervention adds (or projects out) a unit-norm direction built from one or more SAE decoder rows at a chosen transformer layer, at a chosen token position, for a chosen magnitude `alpha`, and scores the three answer choices before and after the hook. Aggregated downstream, the per-row deltas (`stereotyped_delta`, `unknown_delta`, `bias_margin_delta`, `correct_delta`, ...) underpin every causal-feature claim.

## Inputs
- Llama model directory and OpenMOSS SAE directory.
- `prepared/bbq_prepared_examples.parquet` — Step 18 output. Filtered (unless `--include_unmapped`) to rows with `mapped_contrast_confidence ∈ {exact, alias, fallback_axis}` (`alias` is never actually produced; see Step 18 notes).
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
- `target_identity_last_token` / `nontarget_identity_last_token`: regex-locate all spans matching the identity label *and* the relevant answer-option text via `find_spans`, return `max(pos)` — the last token overlapping any match. **All matches anywhere in the prompt are eligible** (see issue 3.3).
- `stereotype_language_last_token`: the last token overlapping any content word from the BBQ `question` (after removing the local `stop` set).
- `all_identity_tokens` / `all_stereotype_language_tokens`: every overlapping token.

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

> **Upstream callout — issue 1.4 (CONFIRMED).** Two pieces of this script are downstream of the broken encoder in [Step 5](05_encode_identity_saes.md#14-blocker--sae-preprocessing-convention-confirmed-wrong-concrete-fix-below):
>
> 1. **The feature pool** — `keep_for_intervention=True` rows come from `intervention_candidate_features_triaged.csv` ([Step 17](17_triage_sae_identity_features.md)), which selects features based on the broken encoder's outputs. After the Step 5 fix, regenerate Step 13 → Step 17 first; the current feature list may not even be the same features.
> 2. **The 3.1 feature-level fix below** — implementing encode → modify latent → decode → patch requires the *corrected* encoder + decoder. JumpReLU at θ=0.7539, dataset-wise input/output scaling (`scale_in = sqrt(d_model)/29.125`, `scale_out = 29.125/sqrt(d_model)`), and `b_dec` only on the decode side. The change in the residual stream must be computed in normalized space and un-scaled before patching: `h += ((acts_modified − acts) @ W_dec) * scale_out`.
>
> The current decoder-direction-addition path (`make_vector` → `h[:, pos, :] += alpha * vec`) does not itself encode anything, so it is not directly broken by 1.4. But the *feature identity* of the vectors being added is wrong (they're rows of the decoder selected by broken activations), and the *scale* of `alpha` is in un-normalized residual units while the SAE itself operates in normalized space — these are entangled with 3.2.

### 3.1 [BLOCKER] — "Feature steering" is decoder-direction addition, not a feature intervention

**What's wrong:** `make_vector` builds `vec = unit-norm mean of signed decoder rows`, and the hook does `h[:, pos, :] += alpha * vec` — **regardless of whether the SAE feature was active on that example**. That is a *direction* intervention, not a feature intervention. A genuine single-feature causal test of "feature `f` drives BBQ bias" is:
1. Encode the actual hidden state: `a = SAE.encode(h)`.
2. Modify only latent `f`: clamp to `0` (ablate) or to a target value (amplify).
3. Decode and patch: `h' = h + (SAE.decode(a') − SAE.decode(a))`.

The helpers needed for exactly this — `ablate_features_in_sae`, `steer_features_in_sae`, `decode_sae`, `patch_residual_with_sae_reconstruction` — already exist in `scripts/analyze_identity_sae_features.py` but are **never called from its `main()`**, and this steering runner does not import or use them. See [Step 13 — `analyze_identity_sae_features.py`](13_analyze_identity_sae_features.md) for the unused helpers.

**Why it matters:** With decoder-vector addition, the SAE contributes only a *direction*. The headline claim — "we identified SAE features causally implicated in social bias" — collapses to "we identified *directions* causally implicated in bias", and a reviewer will reasonably ask why an SAE was needed at all rather than a difference-of-means or probe direction. The project's stated goal *requires* a feature-level intervention.

**Targeted fix:**
- Add a new `--intervention_mode feature_clamp` (and a corresponding `feature_ablate`) that:
  1. In the hook, encodes the captured residual through `SAE.encode` for the feature(s) in `feature_set.feature_ids` only.
  2. Either clamps the latent to `0` (ablate) or to a target value (amplification: a multiple of `feature_stats.p95`/`p99`/`max` — load `feature_stats.csv` for this).
  3. Decodes the modified latents back through `SAE.decode` and patches the residual: `h' = h + (SAE.decode(a') − SAE.decode(a))`.
- Make `feature_ablate` the *primary* causal test (clean, no `alpha` grid, directly answers "is the feature necessary").
- Keep the current `add_vector` mode as a secondary "direction steering" comparison.
- Import and call the existing helpers from `analyze_identity_sae_features.py` rather than reimplementing.

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
- Add a fourth control: the **raw difference-of-means contrast direction** from the geometry pipeline (`analyze_identity_geometry.py` / `analyze_identity_sae_features.py:decoder_alignment`), steered identically. If SAE features do not beat the difference-of-means direction, the SAE is not adding causal value over a linear probe (see issue 5.5). Pre-compute these directions per contrast and pass them in via a new `--direction_baselines_path` argument.

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

### 3.1 cross-reference and 3.2 [MAJOR] — Steering magnitude is uniform and untethered to feature scale

**What's wrong:** Even within the current direction-addition design, `--alphas=-8,-4,-2,2,4,8` is applied to a unit vector identically for every feature. Features differ in natural activation magnitude (some `feature_stats.p95` are 0.1, others 5+) and decoder norm. A constant `alpha` is neither "amplify feature `f` by X%" nor a fixed fraction of the residual-stream RMS norm at that layer/position (which is itself ~3–10× the unit vector). So one feature's "big effect at α=8" might just reflect a *relatively larger* perturbation than another's "small effect at α=8".

**Why it matters:** Cross-feature effect-size comparisons in Step 23's `feature_effect_rankings.csv` and `final_intervention_candidates_table.html` are not on a common scale; rankings reflect a mixture of feature importance and feature-specific scale mismatch.

**Targeted fix:**
- Express the perturbation relative to a meaningful scale. Two options, ranked by interpretability:
  1. **Best (paired with the 3.1 fix)**: clamp the feature latent to a multiple of its own `p95`/`p99`/`max` from `feature_stats.csv` — then `alpha` becomes "feature ablated" (0) or "feature amplified to k×p95" (k ∈ {1, 2, 4}).
  2. **Within the current direction-addition design**: scale `alpha` to a fixed fraction (e.g. {2%, 5%, 10%, 20%}) of the median residual-stream RMS norm at that layer/position, computed on a stratified BBQ sample before the run starts. Save the per-layer norm in `steering_config.json` and use it as the scaling constant.

### 3.3 [MAJOR] — Intervention positions are located by greedy regex and may hit the wrong span

**What's wrong:** `positions_for` calls `find_spans(prompt, [target_identity_label, ans{target_answer_idx}])`, which returns **all** regex matches anywhere in the prompt, then returns `max(pos)` for the `*_last_token` variants. The identity label often appears in the context, the question, and the answer option `B. the Black man` — and the *last* occurrence is the answer-choice list. So `target_identity_last_token` is frequently intervening on the identity inside the answer option, not the identity in the context. `stereotype_language_last_token` has the same problem (question content words also recur in context/answers). The same code-path is used in Step 19's token-role flagging.

**Why it matters:** The position names imply causal loci (the feature acts at the identity mention; the feature acts at the stereotype-language predicate). If a "target_identity_last_token" intervention is actually editing the residual at the token inside `C. the Black man`, that is a different causal question and arguably uninterpretable — the answer-option text is what is being *predicted from*. The downstream Step 23 README treats these positions as answering different causal questions, but only if they land where the names imply.

**Targeted fix:**
- Restrict identity-token search to the **context** span (and optionally the question span) by reusing the section spans built in Step 19's `find_section_spans` (already in this script as `section_spans`). Spans returned by `find_spans` should be intersected with the context span before `max(pos)` is taken.
- Add an `intervention_section` column to the output (`context`, `question`, `answer_option`, `final`) recording where the intervened token actually fell. Audit the distribution.
- Consider renaming the positions to be section-explicit: `target_identity_last_context_token`, `stereotype_language_last_question_token`, etc.

### 3.4 [MAJOR] — BBQ→SAE contrast mapping silently uses axis-fallback

**What's wrong:** The default filter keeps `mapped_contrast_confidence ∈ {exact, alias, fallback_axis}`. The `fallback_axis` path means a BBQ row about `race_arab vs race_white` can be steered with features selected for `race_black vs race_white`, and the downstream analyzer treats `mapped_contrast_name` as the relevant contrast for that example. Full discussion in [Step 18](18_prepare_bbq_for_steering.md) issue 3.4.

**Why it matters here:** This script is where the contamination becomes part of the causal results. Once a row is stamped with a fallback-mapped contrast, it is indistinguishable in `results_parts/*.parquet` from an `exact` row unless `mapped_contrast_confidence` is carried through and stratified on.

**Targeted fix:**
- For headline results, change the default keep-list to `exact` only. Make `--include_fallback_axis` an explicit opt-in.
- Always carry `mapped_contrast_confidence` into the output row (it is implicitly available through `prepared`-merging but not currently stamped on each `results_parts` row). Add it to `steering_output_row`.

### 3.5 [MINOR] — Bundle steering averages decoder rows into one direction

**What's wrong:** `per_contrast_topk` and `role_bundle` modes build `vec = unit-norm mean of signed unit-normed decoder rows`. The downstream analyzer correctly tags these `feature_bundle_membership` and warns. But averaging unit-normed rows and then re-normalizing produces a direction whose relation to any individual feature is weak — bundle effects are hard to interpret even as "membership."

**Why it matters:** As long as bundle modes are run, the temptation to read them as "the role/contrast has effect X" is real. With the 3.1 feature-level fix, bundle interventions become "clamp this *set* of latents simultaneously", which is cleaner and interpretable.

**Targeted fix:**
- Once the feature-level intervention (3.1) lands, redefine bundle interventions as "clamp every latent in the set to 0 (or to p95)" rather than averaged-decoder addition. The bundle effect becomes the joint causal effect of the set.
- Alternatively, deprecate bundle modes for the final paper and run `per_feature` only. The production command already passes `--require_per_feature`.

## Rebuild checklist
- [ ] Implement encode → modify-latent → decode → patch as a new intervention mode (`feature_ablate` / `feature_clamp`); reuse the helpers in `scripts/analyze_identity_sae_features.py`. Make ablation the primary causal test.
- [ ] Re-enable controls for the final run; if cost is a concern, run them on a stratified subsample, not on no examples. Add a `--controls_subsample_frac` argument.
- [ ] Add a `direction_baselines` control that steers with the difference-of-means contrast direction from `analyze_identity_geometry.py` — same prompts, same alphas, same positions. This is the SAE-vs-linear-direction comparison the paper needs (5.5).
- [ ] Switch default `--scoring_mode` to a new `letter` mode (` A`/` B`/` C`); keep `first_token` for backward compatibility but mark deprecated. Length-normalize `answer_logprob` for argmax/accuracy.
- [ ] Restrict identity- and stereotype-language position search to the prepared `context` (and optionally `question`) span; stamp the actual section onto each output row.
- [ ] Change default keep-list to `exact`-only; carry `mapped_contrast_confidence` into every `results_parts` row.
- [ ] Scale `alpha` to either a multiple of the feature's own `p95` (post-3.1-fix) or a fixed fraction of the layer-position residual RMS norm; record the scaling constant in `steering_config.json`.
- [ ] Add `intervention_section` column to the output ∈ `{context, question, answer_option, final}`.
- [ ] Deprecate bundle modes for headline results; keep them only as joint-clamp diagnostics once 3.1 lands.
- [ ] Document the inference-grid problem (alpha × position × feature) and decide on a pre-registered unit-of-inference for the headline statistic (one statistic per feature; see Step 23 issue 2.6).

## Notes from the doc audit
- `result_feature_metadata` always uses `feature_set.feature_ids[0]` and `feature_set.roles[0]` for the single-element case, but for bundles writes `feature_id = -1` while still emitting a `feature_role` of `""`. Downstream `expand_feature_rows` (Step 23) re-derives `feature_id` from `feature_ids_json` and `feature_role` from `feature_roles_json`, so the bundle `feature_id = -1` is overwritten. Still, the placeholder `-1` survives in `feature_level_pre_fdr.csv` for any row that the expander did not touch — worth filtering or marking as bundle explicitly.
- The `random_direction_norm_matched` control uses `torch.manual_seed(seed)` (Python torch RNG) but `make_vector(random_direction=True)` calls `torch.randn` *without* an explicit generator, so the seed-setting is process-global. If the model is mid-forward (it shouldn't be) or any other code touches the torch RNG between seed-set and the `randn` call, the control vector is not reproducibly tied to the job_id. Pass an explicit `torch.Generator` instead.
- `score_answer_logprob` uses `start = max(0, min(prompt_len - 1, labels.shape[1] - 1))` — this is fine when the appended `" " + answer` extends the prompt, but if `truncation=True` kicks in at `max_length=512` and the answer is partially clipped, the summed logprob is over a partial answer. Add a length-check warning.
- The `intervention_mode` argument default `"add_vector"` only allows the comma-separated string to include `ablate_projection`. There is no documented `feature_ablate` / `feature_clamp` because of issue 3.1.
- The control branches (`sign_flip`, `random_*`) only fire under `--scoring_mode answer_logprob` because the fast `--scoring_mode first_token --disable_controls` path is in `run_first_token_batched_feature_set`, which has no control code at all. So even with `--disable_controls` removed, the user must also drop `--scoring_mode first_token` to actually run controls — a footgun worth surfacing in the help text and the operational doc.
- `count_pending_main_jobs` iterates over `fs_prepared.iterrows()` in a triple-nested loop for an O(n_examples × alphas × positions × modes) walk before the model is loaded. On large datasets this is observably slow at startup — acceptable but worth a `--skip_pending_count` flag for resume runs.
