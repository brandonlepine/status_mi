# Issues, Conceptual Gaps, and Opportunities

A critical audit of the `status_mi` pipeline, written for a return to the project after a gap. The goal is a NeurIPS/ICLR-grade result: *interpretable features for marginalized identities in Llama-3.1-8B, and features/directions causally implicated in biased behavior.* Issues are graded:

- **[BLOCKER]** — threatens a core claim; must be resolved before the result is defensible.
- **[MAJOR]** — materially weakens a claim or biases a number; reviewers will raise it.
- **[MINOR]** — correctness/clarity/maintainability; fix opportunistically.

Each issue says where it lives, why it matters, and what to do. A priority punch list is at the end. Section 6 lists strengths so the audit stays balanced.

> Scope note: code was read, not run. "Silently skipped", "in-sample", etc. are inferred from the source. Where a claim depends on an external fact (SAE preprocessing convention, base-vs-instruct behavior), it is marked as a *verification item*, not a settled defect.

---

## 1. Foundational / measurement issues (the representation substrate)

### 1.1 [BLOCKER] Measurement locus (PARTIAL FIX LANDED 2026-05-27; RunPod three-mode comparison remaining)

**Status:** Code landed in two commits; the three-mode geometry comparison on RunPod is what remains.

- `ca1224e` — `extract_identity_activations.py` gained `--token_mode {final_token, identity_span_last, identity_span_mean}`. Default output dir is now `identity_prompts_{token_mode}` so modes do not overwrite each other. New `find_identity_span` + offset-mapping pre-pass validate that every prompt's `form_used` is locatable and survives tokenizer truncation; failures raise loudly before the GPU run starts. A `span_locations.csv` sidecar records the per-prompt char + token range for the audit trail. `select_layer_activation` reduces per-layer hidden states to (B, D) under each mode (final-token gather, identity-span-last gather, identity-span-mean weighted average).
- `6bd78fc` — `encode_identity_saes.py` widens `--activation_mode` to the same three values and removes the prior `NotImplementedError`. Step 5 is mode-agnostic by design (the input array is the same shape regardless of locus); the label is recorded in `run_config.json` for the audit trail.

**Original audit (preserved for context):**

`extract_identity_activations.py` stored, per layer, the residual stream at `attention_mask.sum(dim=1) - 1` — the last non-padding token. Every template in `mi_identity_templates.csv` ends with `.` (e.g. `A01 = "This person is {form}."`, fragment `F03 = "{form}."`). After tokenization the final token is the sentence-final period in essentially every prompt. So the entire geometry pipeline — PCA, probes, contrast directions, shared-subspace SVD, family-stability cosines — characterized the residual stream **at the period token**, not at the identity token. The implicit assumption is "the final token integrates the identity content of the prompt." For a **base** (non-instruct) model with no `[CLS]`-style aggregation objective and no instruction to summarize, that assumption is untested.

**Remaining work (RunPod):**
- Run all three modes for layers `{0, 8, 16, 24, 32}` (and optionally every layer once the SAE encoder fix from 1.4 is verified):
  ```
  python scripts/extract_identity_activations.py --token_mode final_token        # legacy
  python scripts/extract_identity_activations.py --token_mode identity_span_last
  python scripts/extract_identity_activations.py --token_mode identity_span_mean
  ```
- For each mode: re-encode via Step 5 with the matching `--activation_dir` and `--activation_mode`, then rerun Stage 2 geometry analyses.
- Compare contrast AUC, probe accuracy, and shared-subspace spectrum across the three loci. Report whichever location carries the signal in the methods writeup; if final-token does carry it, that is itself a finding (with the span-pooled comparison as evidence). The audit cycle for this issue closes when the comparison is in the paper.

### 1.2 [MAJOR] Base model vs. a multiple-choice QA benchmark (PARTIAL FIX LANDED 2026-05-26)

**Status:** Code landed; the RunPod measurement remains.

- `scripts/build_few_shot_pool.py` writes `data/bbq/few_shot_pool.json` (K=4, seeded, stratified across (ambig/disambig × neg/nonneg), distinct categories, `Answer: <LETTER>. <text>` format).
- `scripts/prepare_bbq_for_steering.py` accepts `--few_shot_pool`. When set, the pool's `(source_file, example_id)` keys are excluded and the formatted prefix is prepended to every remaining prompt (also recorded in a new `few_shot_prefix` column). Without the flag, behavior is identical to today.
- `scripts/diagnose_bbq_baseline.py` (new) consumes the prepared parquet, runs Llama-3.1-8B-Base, and emits `baseline_diagnostics.{json,csv}` with all three audit-required diagnostics: (i) total mass on the three options measured both on the answer LETTERS and on each answer text's first token, (ii) polarity-signed BBQ accuracy + bias score on ambig and disambig strata, (iii) argmax-over-options vs greedy-continuation agreement rate. `--dry_run` validates data flow without the model.

**Original audit:** `download_llama_3_1_8b.py` pulls `meta-llama/Llama-3.1-8B` — the **base** model (correct for SAE compatibility: OpenMOSS LlamaScope SAEs are trained on `Llama3_1-8B-Base`). But BBQ is a QA benchmark, and `prepare_bbq_for_steering.py` builds a zero-shot prompt ending in `"Answer:"`. Base models are weak at, and often off-distribution for, this format.

Why it matters: the entire BBQ causal story rests on the model placing *meaningful, well-calibrated probability mass on the answer options*. If the base model puts 1–2% total mass on the three options and 98% on continuation text, the logprob deltas you steer are in a degenerate regime and "bias" is barely defined.

Remaining work:
- Run `diagnose_bbq_baseline.py` on RunPod against (a) a zero-shot prepared parquet and (b) a few-shot prepared parquet, then diff the two `baseline_diagnostics.json` files. The decision rule is documented in `docs/pipeline_steps/18b_diagnose_bbq_baseline.md`.
- Pick the prompt mode for steering based on that diff; record the headline numbers in the methods writeup as the precondition the audit requires.
- If even few-shot leaves the model degenerate, fall back to one of the audit's honest options: (a) frame results around logprob *margins* (still defined), or (b) reconsider scope.

### 1.3 [MAJOR] First-token answer scoring is degenerate for BBQ answers

`run_bbq_sae_steering.py:score_first_token` scores the log-probability of the **first token of the answer text** (`first_token_ids` tokenizes `" " + answer`). BBQ answers are noun phrases; many begin with the same word (`"The grandmother"`, `"The boy"`, `"Cannot be determined"`). When two of three options share a first token, their first-token logprobs are *identical* and the metric cannot distinguish them.

Separately, the prompt presents labelled choices `A. / B. / C.` and ends with `"Answer:"`. The natural model continuation is the **letter** (` A`/` B`/` C`), but scoring targets the answer *text*. There is a mismatch between prompt design and scoring target.

The documented production long run uses `--scoring_mode first_token`.

What to do:
- Score the answer **letters** ` A`/` B`/` C` (single tokens, mutually distinct, matched to the prompt format). This is as fast as current first-token scoring and removes the degeneracy.
- Keep `answer_logprob` as a confirmatory mode, but length-normalize it (see 2.4).
- Re-run; first-token-text results should be treated as preliminary.

### 1.4 [BLOCKER] SAE preprocessing convention (FIX LANDED 2026-05-26; RunPod re-encode + validate pending)

**Status:** Three commits closed out the code fix; the verification on RunPod is what remains.

- `1ed1422` — `download_openmoss_saes.py` now selects files by an explicit `L<layer>R-<width>x` marker, requires `hyperparameters.json` per layer, pins `--revision` to an absolute commit SHA via `HfApi`, and asserts (position, width) consistency across requested layers.
- `4b8851a` — `encode_identity_saes.py` reads `hyperparameters.json`, validates `act_fn`/`apply_decoder_bias_to_pre_encoder`/`norm_activation`, computes per-layer `scale_in` / `scale_out` / `theta`, and exposes `encode_full` / `decode_full` with the corrected formula. `sae_config_resolved.json` now records the verified config (the prior version declared `"relu"` and no normalization).
- `efc098c` — `validate_sae_hook_alignment.py` imports the corrected `encode_full` / `decode_full`, samples N rows, computes FVU / mean cosine / mean L0 in fp32, and fails the validator above `--reconstruction_fvu_threshold` (default 0.15).

**Remaining work:** re-download on RunPod with `--revision <commit_hash>`, re-encode every layer (deletes obsolete `feature_*.npy` / `feature_stats.csv`), run the validator and confirm `reconstruction_fvu <= 0.15` and `reconstruction_cosine_mean >= 0.95`. Every downstream Stage-3 and Stage-4 analysis must be rerun against the new encodings.

---

**Original diagnosis** (preserved for context):

Relevant fields:

```json
{
  "d_model": 4096,
  "use_decoder_bias": true,
  "apply_decoder_bias_to_pre_encoder": false,
  "act_fn": "jumprelu",
  "jump_relu_threshold": 0.75390625,
  "norm_activation": "dataset-wise",
  "dataset_average_activation_norm": { "in": 29.125, "out": 29.125 }
}
```

`encode_identity_saes.py:encode_batch` is wrong in three ways:

1. **Activation function is JumpReLU, not ReLU.** `f(x) = x · 1[x > 0.75390625]`. The current `torch.relu(...)` keeps every positive pre-activation, including the `(0, 0.7539]` interval that should be exactly zero. This contaminates the sparse top-64 with spurious low-magnitude features.
2. **Missing dataset-wise input normalization.** OpenMOSS scales input so its average L2 norm equals `sqrt(d_model)`: `x_norm = x · sqrt(d_model) / dataset_average_activation_norm.in`. At L24 that is `x · 64 / 29.125 ≈ x · 2.197`. The current code skips this. Pre-activations are ~2.2× smaller than the network expects, the (training-fixed) JumpReLU threshold (when activated) would kill many true features, and the top-64 ranking is computed in the wrong space.
3. **`b_dec` is subtracted at the wrong stage.** `apply_decoder_bias_to_pre_encoder = false` means `b_dec` is decode-side only. The current code does `(x − b_dec) @ W_enc + b_enc`, applying a spurious shift to the encoder input.

Why it matters: every SAE-based number (`feature_stats.csv`, the top-64 encodings, all `cohens_d`/`auc`/`combined_score` in `analyze_identity_sae_features.py`, the triage roles in `triage_sae_identity_features.py`, the BBQ steering feature pool, every per-feature card, every BBQ token-level activation parquet) was computed on the broken encoding.

What to do:

- Fix `encode_batch` in `encode_identity_saes.py` (correct formula below). Assert `act_fn == "jumprelu"` and `apply_decoder_bias_to_pre_encoder is False` so a future checkpoint with different conventions does not silently mis-encode.
- Add a numerical encode→decode reconstruction check to `validate_sae_hook_alignment.py` (FVU, cosine, mean L0). A correctly-loaded SAE at 32× expansion on Llama-3.1-8B-Base residual streams reconstructs well; if FVU is high after the fix, there is still a bug. Fail validation when FVU exceeds a threshold (~0.15) so the regression test is real.
- Update `sae_config_resolved.json` schema to record the verified `activation_function`, `jump_relu_threshold`, `scale_in`, `scale_out`, and `apply_decoder_bias_to_pre_encoder` (the resolved config currently lies — it claims `"relu"`).
- Mirror the corrected encoder in `extract_bbq_token_level_sae_activations.py` (either import `encode_batch` from `encode_identity_saes.py` or duplicate the corrected formula). Regenerate every BBQ token-level parquet.
- Treat all prior SAE outputs and every downstream artifact derived from them as obsolete.

Correct encode/decode formula:

```python
scale_in  = sqrt(d_model) / dataset_average_activation_norm["in"]   # 64/29.125 at L24
scale_out = dataset_average_activation_norm["out"] / sqrt(d_model)  # 29.125/64 at L24
theta     = jump_relu_threshold                                      # 0.75390625 at L24

# Encode:
x_norm   = x * scale_in
pre_acts = x_norm @ W_enc + b_enc
acts     = pre_acts * (pre_acts > theta)

# Decode:
recon_norm = acts @ W_dec + b_dec
recon      = recon_norm * scale_out
```

For the proper feature-level intervention in `run_bbq_sae_steering.py` (issue 3.1), the perturbation in the *un-normalized* residual stream is `delta_h = ((acts_modified − acts) @ W_dec) * scale_out`.

Per-layer constants must be loaded from each layer's `hyperparameters.json` — `dataset_average_activation_norm` and `jump_relu_threshold` vary by layer; do not hardcode the L24 numbers.

### 1.5 [MINOR] Activations are bf16-precision stored as float32 (PARTIAL FIX LANDED 2026-05-27; GH #1 tracks remainder)

**Status:** Disclosure landed. The default still stores fp32 (no breaking change). `bf16` storage is the right end-state but requires a coordinated multi-script change tracked in [GH issue #1](https://github.com/brandonlepine/status_mi/issues/1).

**What landed in `extract_identity_activations.py`:**
- New `--store_dtype {fp32, fp16}` flag. `fp32` default. `fp16` halves disk vs. fp32 but may lose precision on Llama's outlier residual dimensions; the help text documents the tradeoff.
- `run_config.json` records `forward_dtype` and `storage_dtype` separately so the asymmetry is no longer hidden.

**Also landed alongside (per-batch defensive checks the original audit's Notes section flagged):**
- Right-padding is now asserted per batch (`assert attention_mask[:, 0].all()`), so a future tokenizer override that flipped `padding_side` fails loudly instead of producing silently-wrong `final_idx` gathers.
- The redundant `output_hidden_states=True` on `AutoModelForCausalLM.from_pretrained` is gone; the per-forward call still passes it explicitly.
- A one-time CPU-only length pre-pass (`warn_if_truncation_will_occur`) prints a WARNING with counts + percentiles + offending `prompt_id`s if any prompt exceeds `--max_length`. The summary is also written to `run_config.json["length_pre_pass"]`. The run still proceeds — the audit's "explicit message" requirement is met without forcing a re-run.

**Original audit:** `extract_identity_activations.py` runs the model in bf16, then casts the final hidden state to float32 for storage. Stored values therefore carry ~bf16 precision (~3 significant digits). Mean-difference directions average this away, but per-prompt projections, per-prompt SAE encodings, and individual cosines inherit the noise.

---

## 2. Statistical rigor (numbers that will not survive review as-is)

### 2.1 [MAJOR] Headline contrast AUC / Cohen's d are in-sample (FIX LANDED 2026-05-27 across geometry + subspace; SAE features partial)

**Status:** Two commits closed the headline metric switch. The SAE-features held-out reconstruction is intentionally bundled with 2.5 winner's curse.

- `e15e62f` — `analyze_identity_geometry.py` + `analyze_identity_geometry_diagnostics.py`: `auc_all`/`cohens_d_all` renamed `_in_sample`; new `contrast_holdout_summary.csv` and `contrast_family_holdout_residualized_summary.csv` aggregate the family-holdout rows into headline mean/sd/min/max per (layer, contrast). `plot_identity_geometry.py:plot_contrasts` makes held-out the headline plot (`contrast_auc_by_layer.png`) and demotes in-sample to `_in_sample`-suffixed diagnostic plots.
- `51aa571` — `analyze_shared_social_subspace.py`: in-sample decomposition columns renamed `_in_sample`. New `decomposition_rows_holdout` does leave-one-family-out: re-derives every contrast direction on non-`f` rows, re-SVDs a held-out basis, decomposes each direction onto that basis, evaluates shared / residual / full components on the `f` rows. Writes `decomposition_metrics_holdout.csv` (per fold) and `decomposition_metrics_holdout_summary.csv` (mean / sd / n_folds per contrast × k × component). `aggregate_axis_sharedness` and `plot_axis_summary` updated to read the renamed in-sample columns and mark their plots `DIAGNOSTIC`.
- `51aa571` — `analyze_identity_sae_features.py`: in-sample `auc`/`cohens_d`/`full_direction_*` columns in `direction_reconstruction.csv` renamed `_in_sample`. Held-out reconstruction (which requires held-out feature *selection*, not just held-out direction) is bundled with the 2.5 winner's-curse fix.

**Original audit (preserved):**

`analyze_identity_geometry.py:run_contrasts` and `analyze_identity_geometry_diagnostics.py:run_contrasts` compute the contrast direction from `mean(A) − mean(B)`, then evaluate AUC/Cohen's d of the projection **on the same A and B prompts**. `analyze_shared_social_subspace.py:evaluate_component` and `analyze_identity_sae_features.py` do the same. In-sample separation is optimistically biased — a difference-of-means direction is *defined* to separate the two means.

**Remaining work:**
- Hold-out reconstruction for `analyze_identity_sae_features.py:reconstruction_rows`, folded into the 2.5 fix.
- The followups plotting script (`plot_identity_directional_followups.py`) shows in-sample AUC per residualization in projection-histogram titles; those are genuinely in-sample by design (they describe the projection distribution being plotted), but the title text should explicitly say `in-sample` to avoid confusion. Small label-only tweak.

### 2.2 [BLOCKER] No null model for the central claims (FIX LANDED 2026-05-27; both probe + SVD halves complete)

**Status:** Probe null landed `cafc150` (both geometry scripts). SVD null landed `c4071cd` (`analyze_shared_social_subspace.py`). η² null and contrast-direction AUC null are the only remaining sub-pieces, both noted under "Remaining work" below.

**What landed (probe half — commit `cafc150`):**
- `_run_cv_folds` (geometry) / `_run_cv_folds_diag` (diagnostics) extract the inner CV loop so observed and null share one implementation.
- `crossval_probe` in both scripts accepts `n_permutations` and `null_rng_seed`. When `n_permutations > 0`, `y` is globally shuffled per replicate while the GroupKFold split structure is preserved across all replicates.
- New CLI flags on both scripts: `--n_permutations` (default `20`; bump to `>=100` for the headline number) and `--null_random_seed` (defaults to `--random_seed`).
- All probe output rows gain `null_n_permutations`, `null_accuracy_mean`, `null_accuracy_sd`, `null_macro_f1_mean`, `null_macro_f1_sd`, `accuracy_z`, `macro_f1_z`, `accuracy_p_value`, `macro_f1_p_value`. The p-value uses Phipson-Smyth `(1 + n_above) / (1 + N)` smoothing.
- When `n_permutations == 0` the null fields are `NaN` so downstream readers can tell that no null was computed.
- Synthetic-tested: perfect-feature inputs hit p at the n-perm floor (1/(N+1)); noise inputs sit within the null distribution with z ≈ 0 and high p; same seed reproduces the same null bit-for-bit.

**What landed (SVD half — commit `c4071cd`):**
- Two null methods in `analyze_shared_social_subspace.py`:
  1. `null_directions_shuffle_identities` — shuffle prompts between identity_a and identity_b within each contrast (preserves n_a, n_b).
  2. `null_directions_random_half_split` — random axis-wide partition into halves matching n_a, n_b. Stronger null.
- For each null replicate: stack directions, SVD, record per-PC singular values plus **participation ratio** and **top-k explained variance** (both audit-recommended).
- New CSVs:
  - `shared_subspace_spectrum_null_summary.csv` — per (layer, residualization, null_method, component): observed_singular_value, null mean/sd/p5/p50/p95, `observed_exceeds_p95`. PCs above p95 are the audit-defensible "shared" components.
  - `shared_subspace_concentration_null.csv` — observed participation_ratio and top-k variance vs the null distribution, with `observed_pr_more_concentrated_than_p5` and `observed_top_k_exceeds_p95` flags.
  - `shared_subspace_spectrum_null_replicates.csv` (optional via `--save_null_svd_replicates`) — per-replicate per-component sigmas for downstream plotting.
- New CLI: `--n_nulls_svd` (default 200), `--null_svd_random_seed`, `--null_svd_top_k` (default 5), `--save_null_svd_replicates`.
- Synthetic-tested: a rank-1 shared subspace correctly produces observed PC1 ≫ null p95 (and PR ≪ null p5); random per-axis structure correctly produces PC1 below null p95 (no false positive). Both null methods discriminate signal from noise.

**Original audit (preserved):**

The geometry probes (`crossval_probe`) report accuracy/macro-F1 but never a **label-permutation null**. The shared-subspace SVD reports a singular-value spectrum but never compares it to the spectrum of *random* directions or directions from *shuffled* identity labels. Without a null:

- "Identity is linearly decodable" — high CV accuracy could partly reflect group structure / template leakage rather than identity content.
- "There is a shared social subspace" — *any* set of ~19 unit vectors in 4096-d has *some* SVD spectrum; concentration only means something relative to a null. As written, the "shared subspace" claim is not yet supported.

**Remaining sub-pieces (small follow-ups):**
- η² in [Step 8](pipeline_steps/08_analyze_identity_geometry_diagnostics.md) under shuffled identity labels alongside observed.
- Per-contrast AUC / family-holdout AUC nulls (Step 7 / Step 8): apply the same shuffle to the contrast direction projections.

These are mechanical extensions of the SVD / probe machinery rather than new infrastructure. The two main halves of 2.2 (probes + SVD) are now landed; the BBQ analyzer's sign-flip permutation test is the third leg of the discipline.

### 2.3 [BLOCKER] Steering controls are disabled in the production run

`run_bbq_sae_steering.py` implements three controls — `sign_flip`, `random_direction_norm_matched`, `random_feature_matched` — but they are gated behind *not* `--disable_controls`, and the documented production command in `docs/bbq_steering_pipeline.md` passes `--disable_controls`.

Without these controls you cannot claim a feature's effect is **specific**. A norm-matched random direction added at the same position may shift the bias margin just as much (steering vectors of any kind perturb logits). The whole "feature X is causally implicated in bias" claim needs: effect(feature X) ≫ effect(random direction) and ≫ effect(random feature set), at matched norm.

What to do:
- Re-enable controls for the final run. If cost is the issue, run controls on a stratified subsample of examples × features rather than dropping them.
- Add one more control: the **raw difference-of-means contrast direction** from the geometry pipeline, steered identically. If SAE features do not beat the difference-of-means direction, the SAE is not adding causal value over a linear probe — that comparison must be in the paper (see 5.5).

### 2.4 [MAJOR] `answer_logprob` summed over different-length answers

`score_answer_logprob` sums per-token logprobs over the answer span. BBQ's three options have different token lengths; `"Cannot be determined"` is typically the longest, so summed logprob systematically penalizes the unknown option.

Within-example *deltas* (intervened − base) cancel the length bias because length is constant per example — so `stereotype_preference_delta` etc. are OK. But `predicted_base`, `correct_base`, `prediction_changed`, and `accuracy_delta` use `argmax` over **raw** summed logprobs, which *is* length-biased. Baseline accuracy and any accuracy-change metric are contaminated.

What to do: length-normalize (mean per-token logprob) for any argmax/accuracy metric, or score the answer letter (1.3) which has constant length and dissolves the problem.

### 2.5 [MAJOR] Selection-induced bias ("winner's curse") in feature effect sizes (PARTIAL FIX LANDED 2026-05-27)

**Status:** Identity-prompt selectivity half closed in commit `4481445` (`scripts/analyze_identity_sae_features.py`). The BBQ half (`analyze_bbq_feature_level_causal_effects.py`) and the held-out confirmation-set work are still open.

**What landed (identity-prompt selectivity half):**
- The `|diff_mean|` prefilter is gone in both `feature_selectivity_for_contrast` (was 5·top_n) and `identity_selectivity` (was 3·top_n). Cohen's d and AUC are now computed analytically for every feature, then the top `top_n` is selected by `|d|`.
- New helper `compute_cohens_d_and_auc_for_all_features` does this with closed-form formulas: Cohen's d from sum/sum_sq/count (sample variance, `ddof=1`), and AUC from a sparse 4-bucket decomposition (both-zero / a-only-nonzero / b-only-nonzero / both-nonzero). Vectorized helpers match the per-feature `roc_auc_score` + `common.cohens_d` reference to ~1e-16 across all four buckets on synthetic sparse data. Output schemas of `feature_selectivity.csv` and `feature_identity_selectivity.csv` unchanged.
- Newly visible: features with consistent but small mean differences and tiny pooled SD (large standardized d) — the old prefilter silently discarded these.

**Original audit (two places):**

1. ~~`analyze_identity_sae_features.py:feature_selectivity_for_contrast` filters to the top `5·top_n` features by `|diff_mean|`, *then* computes Cohen's d / AUC only on those, *then* keeps top `top_n` by `|d|`. Because `diff_mean` and `d` are highly correlated, the reported `d`/`auc` are conditioned on having survived a selection screen — inflated.~~ **Closed 2026-05-27.**
2. `analyze_bbq_feature_level_causal_effects.py:make_rankings` ranks the top-100 features by effect and `final_intervention_candidates_table.html` sorts by `beneficial_score`; their CIs and q-values are computed on the *same* BBQ examples used to rank them. **STILL OPEN.**

Why it matters: the top features' effect sizes and significance are over-stated. A paper that says "feature 12345 reduces stereotype preference by Δ" with a CI computed post-selection is reporting a biased estimate.

What to do (remaining):
- Split BBQ examples into a **selection set** and a **confirmation set**. Rank/select features on the selection set; report effect sizes, CIs, and q-values **only** from the confirmation set. (BBQ is large enough; even a 50/50 split per axis works.) Same plumbing should be applied to the identity-prompt screen to add a holdout-set confirmation column to `feature_selectivity.csv`, bundled with audit 2.1's held-out reconstruction math.

### 2.6 [MAJOR] Multiplicity is inflated by the alpha × position grid

`analyze_bbq_feature_level_causal_effects.py` produces one significance test per `(feature, layer, alpha, position, role, contrast, axis, context, polarity, …)` group. A single feature is tested at 6 alphas × 3 positions = 18 highly-correlated tests. FDR (`fdr_bh`) is applied within `(axis, context, alpha, position)` strata, so it does not even pool those 18 — and treating correlated tests as independent both inflates the count and mis-estimates FDR.

What to do:
- Decide the **unit of inference** up front: it should be the *feature* (optionally feature × position), not feature × alpha. Summarize the dose-response across alphas into one statistic per feature (e.g. sign-consistent monotone slope, or the effect at a single pre-registered alpha), test that once, FDR across features.
- Keep the alpha grid for the dose-response *plots*, but not as 6 separate hypothesis tests.

### 2.7 [MINOR] Underpowered cells and small permutation/bootstrap budgets

`min_examples = 10` (and `--smoke` lowers nothing below that). A sign-flip permutation test on 10 paired deltas has only 2¹⁰ = 1024 distinct sign assignments — minimum p ≈ 1/1024 — and after FDR almost nothing can reach significance. The documented `analyze_bbq_feature_level_causal_effects.py` command also passes `--smoke`, which caps bootstrap/permutation at 500 (min p ≈ 0.002) — and the fact that the *production* command still says `--smoke` suggests no full-budget run has been done.

What to do: for final results, drop `--smoke`; use ≥10,000 bootstrap and ≥10,000 permutation samples; raise per-cell minimums (or coarsen grouping, per 2.6) so each tested unit has enough examples for the test to have power. Consider BCa instead of percentile bootstrap for small n.

### 2.8 [MINOR] Probe dimensionality reduction leaks across CV folds (VERIFIER LANDED 2026-05-27; RunPod verification run pending)

**Status:** Audit's "show it does not change conclusions on one layer" path. The design is preserved (global StandardScaler + PCA in `make_probe_features`; refitting per fold is intractable across `n_folds × n_residualizations × n_probe_configs × n_layers`), and a fold-internal verifier is now in place.

**What landed:**
- `analyze_identity_geometry.py` and `analyze_identity_geometry_diagnostics.py` each gain a `crossval_probe_fold_internal_pca[_diag]` function and a `--verify_fold_internal_pca <layer>` CLI flag. When set, every probe configuration on that layer runs a second time with `StandardScaler + PCA` fit inside each fold on train rows only. The output `probes/pca_leakage_verification.csv` has side-by-side `global_pca_*` vs `fold_internal_pca_*` accuracy/macro-F1 means and SDs plus per-row `accuracy_delta` and `macro_f1_delta`.
- `make_probe_features` docstring rewritten to explain the speed tradeoff, the technical leakage, and how to run the verifier.

**Original audit (preserved):** `analyze_identity_geometry.py:make_probe_features` fits `StandardScaler` + PCA once on the *entire* layer, then `crossval_probe` does group-K-fold on the logistic layer only. The PCA basis is fit on data including the test fold. PCA is unsupervised so leakage is mild, but a careful reviewer will still flag it.

**Remaining work:** Run on RunPod once (e.g. `python scripts/analyze_identity_geometry.py --verify_fold_internal_pca 24` and same for diagnostics) and check that `|accuracy_delta|` is smaller than the per-fold `global_pca_accuracy_sd` for every row in the CSV. Record the verification numbers in the methods writeup.

---

## 3. Causal-intervention design (the heart of the second contribution)

### 3.1 [BLOCKER] "Feature steering" is decoder-direction addition, not a feature intervention (FIX LANDED 2026-05-27)

**Status:** Code path closed across commits `11d4a4d` (canonical torch primitives) and `84c87b5` (BBQ steering hook + dispatch). Headline RunPod run with `--intervention_modes ablate` still pending; until then the existing `steering_per_feature_matched_full/` numbers are direction-based and must not be cited as feature-level evidence.

**What landed:**
- Canonical torch primitives now live alongside the corrected SAE encoder in `scripts/encode_identity_saes.py`: `ablate_features`, `clamp_features`, `steer_features`, and the wrapper `patched_residual_with_intervention(h, sae, intervention_fn)`. The wrapper runs encode → modify-latent → decode → patch in normalized space and un-scales by `scale_out` so the residual delta is on the model's natural scale. SAE reconstruction error cancels in the delta because only the intervention-induced change is added back.
- `scripts/run_bbq_sae_steering.py` exposes `install_feature_intervention_hook` and `install_batched_feature_intervention_hook`. The forward hook captures `h` on `model.model.layers[layer-1]`, runs the chosen intervention through the patched-residual wrapper, and writes the modified residual back in place.
- `--intervention_modes` (comma-separated) defaults to `ablate` (no alpha grid: the audit's recommended primary causal test). Valid modes: `{add_vector, ablate_projection, ablate, clamp, steer}`. `--clamp_value` accepts the per-feature clamp target (in normalized latent space; user supplies the lookup from `feature_stats.csv`). Five synthetic hook integration tests pass.
- Legacy `add_vector` / `ablate_projection` decoder-direction modes remain available so the audit-5.5 "is the SAE adding causal value over a linear direction?" comparison can be run side-by-side.

**Remaining:** RunPod headline run with `--intervention_modes ablate` against the (re-encoded under 1.4) feature pool. Pair with an `add_vector` run on the same examples for the linear-direction baseline (5.5).

**Original audit (preserved):**

`run_bbq_sae_steering.py:make_vector` built `vec = unit-norm mean of signed decoder rows` and the hook did `h[:, pos, :] += alpha · vec`. This added a fixed vector **regardless of whether the SAE feature was active** on that example. That was a *direction* intervention, not a *feature* intervention.

A genuine single-feature causal test of "feature f drives bias" is:
1. encode the actual hidden state `a = SAE.encode(h)`,
2. modify only latent `f` (clamp it to 0 to ablate, or to a target value to amplify),
3. decode and patch: `h' = h + (SAE.decode(a') − SAE.decode(a))`.

Why it matters: with decoder-vector addition, the SAE contributes nothing but a *direction*. The claim "we found SAE features causally implicated in bias" collapses to "we found *directions* causally implicated in bias", and a reviewer will ask why an SAE was needed at all rather than a difference-of-means or probe direction. The project's stated goal — *interpretable features causally implicated in biased behavior* — specifically requires a feature-level intervention.

### 3.2 [MAJOR] Steering magnitude is uniform and untethered to feature scale (STILL OPEN; 3.1 unblocks the cleanest fix)

Even within the current direction-addition design, `alpha ∈ {±2, ±4, ±8}` is applied to a **unit** vector, identically for every feature. Features differ enormously in natural activation magnitude and decoder norm. A fixed alpha is neither "amplify feature f by X%" nor a constant fraction of the residual-stream norm at layer 24 (which is itself large and varies by token). So a feature showing a big effect at α=8 may simply have received a *relatively larger* perturbation than another.

The same problem now applies to the post-3.1 `--intervention_modes clamp` / `steer` paths: a single `--clamp_value` (or alpha) is applied across all features regardless of each feature's `p95`/`p99`. The `ablate` mode is unaffected (it sets the latent to exactly 0).

What to do: express the perturbation relative to a meaningful scale — e.g. as a multiple of the residual-stream RMS norm at that layer/position, or (better, now that 3.1 has landed) extend `--clamp_value` to accept a per-feature lookup against `feature_stats.csv` (`p95`/`p99`/`max`) and a multiplier grid. Then "α" is comparable across features.

### 3.3 [MAJOR] Intervention positions are located by greedy regex and may hit the wrong span

`run_bbq_sae_steering.py:positions_for` (and `extract_bbq_token_level_sae_activations.py:find_all_spans`) locate `target_identity_last_token` by searching the prompt for the identity label *and* the answer-option text, collecting **all** matches, and taking `max(pos)` — the last. The identity label often appears in the context, the question, *and* the listed answer option `B. the Black man`; the last occurrence is in the answer-choice list. So `target_identity_last_token` is frequently intervening on the identity token *inside the answer choice*, not the identity mention in the context. `stereotype_language_last_token` has the same problem (question content words recur in context/answers).

Why it matters: the position names imply a causal locus ("the feature acts at the identity mention") that the implementation does not guarantee. The `analyze_bbq_feature_level_causal_effects.py` README explicitly says these positions "answer different causal questions" — but only if they land where the names say.

What to do: use `find_section_spans` (already implemented) to restrict identity-token search to the **context** span specifically, and disambiguate. Record, per job, which section the intervened token fell in, and audit the distribution.

### 3.4 [MAJOR] BBQ→SAE contrast mapping silently uses axis-fallback

`prepare_bbq_for_steering.py:map_contrast` returns `exact` when the BBQ example's `(target, nontarget)` identities match an SAE contrast, otherwise `fallback_axis` (any contrast on the same axis), otherwise `unmapped`. `run_bbq_sae_steering.py` keeps rows with confidence in `{exact, alias, fallback_axis}` by default. So a BBQ item about `race_arab vs race_white` can be steered with features selected for `race_black vs race_white`, and the downstream analyzer treats `mapped_contrast_name` as the relevant contrast.

Why it matters: feature-to-example matching is a load-bearing assumption for "this feature is implicated in *this* identity's bias." `fallback_axis` breaks it while leaving the data looking clean.

What to do: for headline results, restrict to `mapped_contrast_confidence == exact`. Report `fallback_axis` separately, if at all. At minimum, stratify every effect table by mapping confidence so the reader sees which rows rest on a fallback.

### 3.5 [MINOR] Bundle steering averages decoder rows into one direction

`per_contrast_topk` and `role_bundle` modes average signed decoder rows into a single vector. The downstream analyzer correctly tags these `feature_bundle_membership` and warns against single-feature claims — good. But averaging *unit-normed* rows then *re-normalizing* produces a direction whose relation to any individual feature is weak; bundle effects are hard to interpret even as "membership." With the 3.1 feature-level fix landed, bundle interventions can now be implemented as "clamp this *set* of latents simultaneously" via a multi-feature `ablate` / `clamp` call — cleaner and interpretable. Prefer that.

---

## 4. Data construction and conceptual coverage

### 4.1 [MAJOR] Contrast lists reference identities that do not exist (FIX LANDED 2026-05-27 for geometry/subspace/SAE/plots; BBQ side still pending)

**Status:** Four commits closed the geometry/subspace/SAE/plot side. `prepare_bbq_for_steering.py:MANUAL_ALIASES` is separate and still has the same class of problem — it's tracked here as remaining work.

**What landed:**
- `1e242c9` — new `scripts/contrast_registry.py` is the single source of truth. 21 canonical contrasts as 4-tuples (`contrast_name, identity_a, identity_b, axis`). Two typos fixed: `ses_low_income → ses_low`, `ses_high_socioeconomic_status → ses_high`. All 21 entries validate against `data/bbq_identity_normalized_forms.csv`. Module also exposes `KEY_CONTRAST_NAMES`, `SELECTED_CROSS_AXIS_ORDERINGS`, `load_validated_contrasts(...)`, `write_contrasts_skipped(...)`, `get_contrast_pairs(...)`, `filter_to_key_contrasts(...)`. **No startup assertion** — partial-axis runs work.
- `398ffee` — `analyze_identity_geometry.py` + `analyze_identity_geometry_diagnostics.py` source `CONTRASTS` from the registry at `main()` startup and write `contrasts/contrasts_skipped.csv` with per-row `reason` annotations + warning per skipped pair.
- `6eafb4d` — `analyze_shared_social_subspace.py` + `analyze_identity_sae_features.py` import `DEFAULT_CONTRASTS` (+ `KEY_CONTRASTS`, `SELECTED_CROSS_AXIS_ORDERINGS` in subspace). Local `load_contrasts(path, metadata, output_dir=...)` reworked to route through the registry validator and emit the skipped CSV.
- `<plot-commit>` — `plot_identity_directional_followups.py` + `plot_identity_directional_visualizations.py` + `plot_identity_sae_features.py` import their `DEFAULT_CONTRASTS` / `KEY_CONTRASTS` from the registry. The followups script's `RESIDUALIZATION_COMPARISON_CONTRASTS` and `CENTROID_ORDERING_CONTRASTS` get the `ses_low_income_vs_ses_rich → ses_low_vs_ses_rich` rename so the paper-summary panel actually plots a real contrast.

**Headline numerical effect:** SES axis now runs 4 contrasts (`ses_low_vs_ses_rich`, `ses_low_vs_ses_high`, `ses_lower_class_vs_ses_upper_class`, `ses_blue_collar_vs_ses_white_collar`) instead of the silently-2 from before. Any paper claim about "SES" coverage now matches the code's count.

**Original audit (preserved):** The `CONTRASTS` / `DEFAULT_CONTRASTS` lists across six scripts included `ses_low_income` and `ses_high_socioeconomic_status`. The identity-forms CSV had no such IDs. Every analysis did `if identity_a not in identity_set: continue` — so the typo contrasts were dropped with no error, and the SES axis quietly had fewer contrasts than the code implied. `prepare_bbq_for_steering.py:MANUAL_ALIASES` similarly maps to non-existent IDs.

**Remaining work — BBQ side:** `prepare_bbq_for_steering.py:MANUAL_ALIASES` still maps to non-existent IDs (`ses_low_income`, `age_old`, `age_nonold`, `nationality_asia_pacific`, `nationality_african`, `nationality_european`, plus a possible `sexuality_*` / `appearance_obese` audit). There is also no `age` axis in the dataset at all, so any BBQ Age handling via these aliases is dead. Fold these into the BBQ prepare-side fix (separate commit; conceptually distinct from the geometry contrast registry).

### 4.2 [MAJOR] Intersectional BBQ categories are flattened to a single axis

`prepare_bbq_for_steering.py:AXIS_MAP` collapses `race_x_gender` and `race_x_ses` to `race_ethnicity`. Intersectional examples have compound group labels (`F-Black`); `identity_components` splits them and `target_identity_id` becomes a single component. The intersectional structure is discarded.

Why it matters: the project's stated subject is *marginalized identities*, and intersectionality is central to that literature. Flattening Race×Gender to "race" both loses the most interesting cases and risks mislabeling the stereotyped answer. Either (a) handle intersectional contrasts as first-class objects (compound identity directions, intersectional steering), which would be a genuine contribution, or (b) explicitly exclude `race_x_*` and say so. Do not silently flatten.

### 4.3 [BLOCKER] `question_polarity` sign is not folded into the bias metric

BBQ has negative-polarity questions ("Who was bad at X?") and non-negative ones ("Who was good at X?"). `stereotyped_groups` is fixed (it is the group the *negative* stereotype targets). For a **negative** question, choosing the stereotyped group is the *bias-consistent* answer; for a **non-negative** question, choosing the stereotyped group is the *anti*-bias answer.

`analyze_bbq_feature_level_causal_effects.py` defines `stereotype_preference_delta = Δ[log p(stereotyped) − log p(unknown)]` with **no polarity sign**. The grouping keys include `question_polarity`, so individual rows are separable — but `effect_label` (`bias_amplifying` if `bias_delta > threshold`), `beneficial_score`, `harmful_score`, and `make_rankings` do **not** condition on polarity. A feature that raises `log p(stereotyped group)` is labeled "bias-amplifying" even on non-negative items where that is the *unbiased* direction. `final_intervention_candidates_table.html` is sorted by a polarity-confounded `beneficial_score`.

What to do: define a **polarity-signed** bias quantity, e.g. `signed_bias_delta = stereotype_preference_delta · s`, with `s = +1` for `question_polarity == neg` and `s = −1` for `nonneg`. Use the signed quantity everywhere a "bias direction" is asserted (`effect_label`, scores, rankings, the candidates table). Until this is fixed the bias taxonomy is partly wrong.

### 4.4 [MINOR] `MANUAL_ALIASES` has dozens of duplicate `"nondisabled"` keys

`prepare_bbq_for_steering.py:MANUAL_ALIASES` literally repeats `"nondisabled": "disability_nondisabled"` ~30 times (a copy-paste artifact; the dict dedups so it is harmless at runtime). It is a signal the file was not reviewed. Clean it up and add a unit test that the alias table maps only to existing identity IDs (ties into 4.1).

### 4.5 [MINOR] `works_*` template-compatibility flags are dead metadata

`create_dataset.py` decides whether a template×identity pair is realized purely by "is the required form column non-empty", ignoring the `works_is_adj` / `works_group` / … flags in the identity CSV. In practice forms are empty roughly when the flag is 0, so output is mostly correct, but the flags are unused. Either use them as the source of truth or delete them to avoid the impression of a constraint that is not enforced.

### 4.6 [MINOR] Top-64 SAE truncation may clip true activations

`encode_identity_saes.py` keeps only the top-64 features per row (`--top_k_save 64`); everything else is treated as exact zero downstream (`sparse_long` drops non-positive). The SAE is a 32× expansion (~131k features). If the SAE's true L0 (number of active features) at layer 24 exceeds 64 on some prompts, real activations are clipped to zero, which biases `mean_a`/`freq_a` downward for mid-ranked features and slightly inflates apparent contrast selectivity. Check the SAE's reported/empirical L0; if it is comfortably under ~50, 64 is fine — otherwise raise `top_k_save`.

---

## 5. Methodological issues and opportunities

### 5.1 [MAJOR] Direction reconstruction treats decoder rows as an orthonormal basis (FIX LANDED 2026-05-27)

**Status:** Closed in commit `1a569c3` (`scripts/analyze_identity_sae_features.py`). The output schema of `direction_reconstruction.csv` is unchanged; the values are now correct and bounded.

**What landed:** `reconstruct_direction` now solves the least-squares problem `argmin_c ||basis.T @ c − direction||²` via `np.linalg.lstsq`; the minimizer `recon = c @ basis` is the true orthogonal projection of `direction` onto `span(rows of basis)`. With this:
- `fraction_norm_captured = ||recon||² / ||direction||² ∈ [0, 1]` by construction.
- `cosine_with_full_direction = sqrt(fraction_norm_captured)` (projection identity).
- Defensive: `fraction` divides by `||direction||²` rather than assuming unit-norm input.

**Validation:** 200 random trials (`k ∈ [3, 50]`, `d_model = 256`) all gave `fraction ∈ [0, 1]`; the projection identity held to ~1e-15. Pathological case (10 basis rows with avg mutual cosine 0.77, direction along the cluster) — old: `fraction = 74.7` (nonsensical); new: `0.984` (correct).

**Original audit (preserved):**

`analyze_identity_sae_features.py:reconstruct_direction` did `basis = decoder_normed[feature_ids]; coeff = basis @ direction; recon = coeff @ basis` — i.e. `recon = BᵀB d` with `B` having unit-norm but **not orthogonal** rows. The orthogonal projection of `d` onto `span(B)` is `Bᵀ(BBᵀ)⁻¹B d`. The two coincide only when `B` is orthonormal. SAE decoder rows of related identity features are generally *not* orthogonal, so:

- `fraction_norm_captured = ||recon||²` was not a fraction of anything — `BᵀB` is not a projector, and `||BᵀB d||²` can exceed 1.
- `cosine_with_full_direction` was computed against the (re-normalized) `recon`, so it was a real cosine but to a *non-projection* vector, not to "the best k-feature reconstruction."

The reconstruction analysis is meant to answer "how much of the identity direction do k SAE features capture" — a natural and reviewable claim. As written, the numbers were not that.

### 5.2 [MAJOR] Triage roles are heuristic definitions, not validated findings

`triage_sae_identity_features.py` builds `template_artifact_score`, `sharedness_score`, `polysemanticity_score`, `contrast_specificity_score` as linear combinations with hand-picked weights (0.4/0.3/0.2/0.1 …) and hand-picked thresholds (0.5, 0.7), then runs a decision cascade to assign roles (`identity_token_local`, `shared_social_feature`, …) and `keep_for_intervention`.

As an *engineering* filter to choose which features to steer, this is fine. But the roles cannot be presented as *results* ("we identified N identity-token-local features and M shared social features") — they are definitions, and the weights/thresholds are unjustified and unvalidated. The `entropy()`-based scores additionally treat activation magnitudes as if they were a probability distribution, which is heuristic.

What to do:
- Treat triage strictly as feature *selection*, and pre-register the selection rule (so it is not tuned to the outcome).
- If a feature *taxonomy* is a paper contribution, validate it: human inter-rater agreement on a sample of feature cards, and/or a behavioral criterion (e.g. "identity-token-local" features should show their causal effect specifically at `target_identity_last_token` and not at `final_prompt_token` — that is a testable, falsifiable prediction the steering data can check).
- Sensitivity analysis: show conclusions are stable to reasonable changes in the weights/thresholds.

### 5.3 [MINOR] `combined_score` sums three near-duplicate, equally-weighted metrics

`analyze_identity_sae_features.py` sets `combined_score = z(|d|) + z(|cos|) + z(|auc − 0.5|)`. Cohen's d and AUC both measure the same A/B separation and are monotonically related, so the score effectively double-weights selectivity vs. decoder alignment. Use one selectivity metric (d *or* AUC) plus the decoder cosine, or justify the weighting. Minor, but it propagates into `per_contrast_topk` feature selection.

### 5.4 [MINOR] Representation inconsistency: residualized direction vs. raw-encoded SAE features

`analyze_identity_sae_features.py` computes contrast directions from `family_residualized` activations, but the SAE features (`long_df`) were encoded by `encode_identity_saes.py` from **raw** activations. `decoder_alignment` then takes the cosine between a raw-space decoder row and a residualized-space direction, and `combined_score` mixes a residualized-direction cosine with a raw-SAE-activation Cohen's d. The two live in slightly different spaces. Decide on one representation (probably: residualize, then re-encode through the SAE, or do everything raw) and be consistent. Document the choice.

### 5.5 [MAJOR] Missing baseline: does the SAE beat a difference-of-means direction?

Throughout, the difference-of-means contrast direction is computed *and* SAE features are computed, but they are never put in head-to-head competition as *interventions*. The key scientific question for an SAE-based paper is: **does decomposing into SAE features buy anything over a single linear direction?**

What to do: steer with (a) individual SAE feature interventions (after the 3.1 fix), (b) the raw difference-of-means contrast direction, (c) a logistic-probe direction. Compare causal effect on BBQ. If SAE features do not localize or do not beat the linear direction, that is still a publishable (and honest) result — but you must run the comparison.

### 5.6 [Opportunity] Stronger causal methods than steering

Steering (add a vector / clamp a latent) answers "is this direction sufficient to move behavior." It does not establish that the model *uses* this feature on this input. Consider adding:

- **Activation patching / counterfactual patching**: build minimal-pair BBQ contexts (same context, swapped identity), and patch the identity-token residual (or a single SAE latent) from one into the other. This measures the feature's *necessity* on real inputs, not just sufficiency of a synthetic perturbation.
- **Attribution patching / integrated gradients** over SAE latents to *discover* the features implicated in a BBQ answer, instead of importing them from the identity-prompt triage. This would make feature discovery causal end-to-end, rather than: select on templated identity prompts → hope they matter on BBQ.
- **Ablation as the primary test** (clamp latent to 0): cleaner than amplification, no alpha grid, directly answers "is the feature necessary."

### 5.7 [Opportunity] Minimal-pair / counterfactual BBQ instead of (or alongside) steering

The cleanest measurement of "how identity changes the model's answer" is a counterfactual: take a BBQ context, swap only the identity term, and measure the answer-distribution change — no intervention vector needed, fully on-distribution. This also gives a natural, assumption-free *behavioral* target that the feature-level causal analysis can be validated against. The identity-forms CSV already has the surface forms needed to do controlled identity substitution.

### 5.8 [Opportunity] Multi-layer SAE coverage

SAEs are encoded/steered only at layer 24. The geometry diagnostics show identity structure evolving across layers 0/8/16/24/32. A single layer cannot tell you *where* identity-bias features live or whether the causal locus shifts. Download/encode SAEs for at least layers 16 and 32 and run the feature pipeline across them; "the causal layer" is itself a result.

### 5.9 [MINOR] PCA on StandardScaler-ed activations changes the geometry (FIX LANDED 2026-05-27)

**Status:** Both geometry scripts now expose `--scaling {standardize, center_only}` with default `center_only`. The flag is plumbed through `run_pca`, `make_probe_features`, and the fold-internal-PCA verifier from audit 2.8, and recorded in `run_config.json`. A `CenterOnlyScaler` class is a drop-in replacement for `StandardScaler` (just subtracts per-dim mean), routed via a `make_scaler(mode)` factory at every call site.

**Empirical effect:** unit-tested on synthetic data with a 10× rogue dimension — `center_only` PCA gives PC1 = 92.6% explained variance (rogue dim dominates, faithful to activation-space geometry); `standardize` PCA gives PC1 = 15.7% (z-scoring spreads variance evenly and hides the rogue dim). The two modes describe different geometries.

**Original audit (preserved):**

`run_pca`/`make_probe_features` apply `StandardScaler` (per-dimension z-scoring) before PCA. Residual-stream dimensions have meaningful, unequal scale (rogue/high-norm dimensions carry real signal); z-scoring upweights low-variance dimensions, and the resulting explained-variance ratios describe *standardized* space, not activation space. This is defensible for visualization but should be stated, and ideally compared against centered-only (no scaling) PCA. For probes the choice matters less (logistic regression is scale-tolerant) but be consistent and explicit.

**Optional follow-up:** run with both `--scaling center_only` and `--scaling standardize` once on RunPod, confirm the cross-residualization conclusions (η², probe AUC, contrast AUC) are stable under both, and document the stability in the methods writeup.

### 5.10 [MINOR] Heavy code duplication across analysis scripts (FIX LANDED 2026-05-27)

**Status:** All shared helpers now live in `scripts/common.py`. Two commits closed this and the related 4.1 contrast-registry work:

- `1e242c9` — `scripts/contrast_registry.py` consolidated the `CONTRASTS` / `DEFAULT_CONTRASTS` / `KEY_CONTRASTS` / `SELECTED_CROSS_AXIS_ORDERINGS` literals (also fixing the typo identities — audit 4.1).
- `e50bbd1` — `scripts/common.py` consolidated everything else: `cohens_d`, `cosine`, `normalize`, `compute_direction` (low-level + `compute_direction_for_pair` convenience), `evaluate_projection`, `residualize`, `OKABE_ITO` (+ `okabe_ito_palette` helper), `save_fig`, `CenterOnlyScaler` + `make_scaler`. The `ContrastDirection` dataclass carries the canonical 3-tuple of (direction, global_mean, sign_flipped).

**What landed in the 8 consumer scripts:**
- `analyze_identity_geometry.py`: `contrast_direction` and `evaluate_contrast_scores` removed; `run_contrasts` routes through `common.compute_direction` + `common.evaluate_projection`. Sign convention is now consistently mean(A_proj) > mean(B_proj), so in-sample `auc_in_sample` is always ≥ 0.5 (a small but legitimate behavior improvement on a diagnostic-only CSV).
- `analyze_identity_geometry_diagnostics.py`: `cohens_d`, `make_contrast_direction`, `evaluate_projection` (local), `residualize`, `OKABE_ITO`, `save_figure` all removed. Thin adapters preserve each call site's prior 2-tuple shape; the new `center_mean` kwarg on `common.compute_direction` supports the held-out train-mean centering pattern.
- `analyze_shared_social_subspace.py`: 5 local helpers removed; adapters keep the script's `(direction, global_mean, sign_flipped)` return convention.
- `analyze_identity_sae_features.py`: 5 local helpers removed; adapters keep the script's 2-tuple shapes.
- `plot_identity_geometry.py`, `plot_identity_directional_visualizations.py`, `plot_identity_directional_followups.py`, `plot_identity_sae_features.py`: local `save_fig`/`save_figure`/`OKABE_ITO`/`cohens_d`/`normalize`/`residualize`/`compute_direction` removed. Each script keeps a 1-line `save_fig` wrapper that calls `common.save_fig(dpi=220, bbox_inches=None, tight_layout=True)` so prior figure styling is preserved.

**Net change:** 358 lines added (`common.py`), 369 lines removed across the 8 consumers. Single source of truth for every helper that was duplicated. The class of "results differ between scripts because their `cohens_d` drifted" bugs is now impossible by construction.

**Smoke-tested:** 10/10 common.py unit tests pass; all 8 consumer scripts import without runtime error; AST-clean.

---

## 6. What is already done well (keep these)

Stated so the audit is balanced and these are not lost in a refactor:

- **Surface-form residualization diagnostics** (`analyze_identity_geometry_diagnostics.py`): residualizing by `family` / `template_id` / `required_form` and re-running PCA/probes/contrasts directly attacks the "are we measuring identity or template" confound. This is the right instinct and a genuine strength.
- **Family-holdout / family-to-family generalization**: training a direction on some template families and testing on held-out ones is a real generalization test (just make it the headline — see 2.1).
- **Variance decomposition (η²)** by metadata factor is a clean, honest way to show how much variance identity explains relative to surface form.
- **The bias taxonomy** in `analyze_bbq_feature_level_causal_effects.py` distinguishing `bias_reducing_uncertainty` (mass moves to "unknown") from `bias_reducing_substitution` (mass moves to the other identity) is conceptually sharp — a steering result that just swaps one stereotype for another is not debiasing, and the code knows that.
- **`individual_feature` vs `feature_bundle_membership`** is tracked explicitly; the analyzer warns when only bundle rows exist.
- **Controls exist** in the steering code (sign-flip, random direction, random feature) — they just need to be turned on (2.3).
- **Engineering hygiene**: resume/checkpointing everywhere, `run_config.json` / `*_config.json` for every run, the dedicated `validate_sae_hook_alignment.py`, explicit HF `hidden_states[k]` convention notes, axis-matching to prevent wrong-axis contamination. This is well above typical research-code standard and makes the fixes above tractable.

---

## 7. Priority punch list

Ordered by what most threatens a defensible result.

**Tier 1 — do before trusting any current number**

1. **Verify SAE preprocessing** (1.4): confirm LlamaScope normalization + activation function; add an encode→decode reconstruction-quality check to `validate_sae_hook_alignment.py`. If wrong, every SAE number is wrong.
2. **Fix the feature intervention** (3.1) — **FIX LANDED 2026-05-27.** Canonical torch primitives (`ablate_features`, `clamp_features`, `steer_features`, `patched_residual_with_intervention`) live in `scripts/encode_identity_saes.py` alongside `encode_full` / `decode_full`. `scripts/run_bbq_sae_steering.py` dispatches them via `--intervention_modes` (default: `ablate`). Commits `11d4a4d` (primitives) + `84c87b5` (hook + dispatch). RunPod headline run with `--intervention_modes ablate` pending.
3. **Re-enable steering controls** (2.3) and add the difference-of-means direction as a control/baseline (5.5).
4. **Polarity-sign the bias metric** (4.3): the current `effect_label`/rankings/candidates table are polarity-confounded.
5. **Validate the measurement locus** (1.1): compare final-token vs identity-span-pooled geometry; pick and justify one.
6. **Characterize baseline behavior** (1.2): answer-option mass and standard BBQ score for Llama-3.1-8B-Base in this format.

**Tier 2 — required for the numbers to be honest**

7. Held-out split for feature selection vs. effect estimation (2.5 — PARTIAL FIX LANDED 2026-05-27: identity-prompt selectivity half closed in commit `4481445`; BBQ-side rankings + held-out confirmation set still open).
8. Null models for geometry probes and the shared-subspace spectrum (2.2).
9. Make held-out (cross-family/cross-template) AUC the headline; demote in-sample AUC (2.1).
10. Fix answer scoring: score the letter A/B/C, or length-normalize `answer_logprob` (1.3, 2.4).
11. Restrict headline steering to `exact` contrast mapping; stratify by mapping confidence (3.4).
12. Audit every contrast/alias identity ID against the dataset; make missing-ID skips loud (4.1).
13. Reduce the inference grid: one test per feature, not per feature×alpha×position (2.6).
14. Drop `--smoke`; raise bootstrap/permutation budgets and per-cell minimums (2.7).

**Tier 3 — correctness, clarity, strengthening**

15. Fix the reconstruction projection math (least-squares, not `BᵀB`) (5.1 — FIX LANDED 2026-05-27, commit `1a569c3`).
16. Verify intervention positions land in the intended prompt section (3.3).
17. Tie steering magnitude to a meaningful scale (3.2).
18. Decide intersectional BBQ handling — first-class or excluded, not flattened (4.2).
19. Reframe triage as pre-registered *selection*; validate the taxonomy if it is a contribution (5.2).
20. Make representation use consistent (residualized vs raw) across the SAE analysis (5.4).
21. Extract shared code into a common module with a validated contrast registry (5.10, 4.1).
22. Multi-layer SAE coverage (5.8); consider counterfactual/patching methods (5.6, 5.7).

**Framing note.** The repo currently runs two semi-independent investigations — identity *geometry* and BBQ *causal features* — joined only loosely (geometry's contrast list seeds the triage that seeds steering). A NeurIPS/ICLR paper needs one throughline. The strongest version: *templated prompts establish where/how identity is represented (geometry) → SAE features name interpretable components of that representation → feature-level interventions on BBQ show which of those components are causally implicated in bias.* For that arc to hold, the SAE feature interventions must be real feature interventions (3.1) and must beat a plain linear direction (5.5) — otherwise the SAE is decoration on a linear-probe result, and the paper should be reframed around directions instead.
