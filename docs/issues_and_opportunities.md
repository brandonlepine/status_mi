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

### 1.3 [MAJOR] First-token answer scoring is degenerate for BBQ answers (FIX LANDED 2026-05-28)

**Status:** Closed in commit `2829417` (`scripts/run_bbq_sae_steering.py`) + operational doc update in `docs/bbq_steering_pipeline.md`.

**What landed:**
- **New scoring mode `letter`, now the default.** `score_letter` / `score_letter_batch` gather logprobs at the cached token IDs for ` A` / ` B` / ` C` — single tokens, mutually distinct, matched to the prompt format (`A. {ans0} B. {ans1} C. {ans2} Answer:`). Removes the first-token-of-noun-phrase degeneracy entirely.
- **`answer_letter_ids(tokenizer)` helper** caches the three letter IDs per tokenizer and raises a clear `ValueError` (suggesting `--scoring_mode answer_logprob` as the fallback) if any letter tokenizes to more than one token.
- **`--scoring_mode` choices**: `{letter, answer_logprob, first_token}`, default `letter`. Legacy `first_token` preserved for backward-compatible comparison runs.
- **Both scoring paths dispatch by mode.** Per-example `score_fn` and batched `score_batch_fn` each pick the right scorer; the fast batched path runs for `--scoring_mode in {letter, first_token}`.
- **`docs/bbq_steering_pipeline.md` production command** updated from `--scoring_mode first_token` to `--scoring_mode letter`.

**Remaining:** RunPod headline run with the new default. Audit 2.4 (length bias of `answer_logprob` argmax / accuracy) is a separate, still-open analyzer-side fix.

**Original audit (preserved):** `run_bbq_sae_steering.py:score_first_token` scored the log-probability of the first token of the answer text (`first_token_ids` tokenizes `" " + answer`). BBQ answers are noun phrases; many begin with the same word (`"The grandmother"`, `"The boy"`, `"Cannot be determined"`). When two of three options shared a first token, their first-token logprobs were identical and the metric could not distinguish them. The prompt presents labelled choices `A. / B. / C.` and ends with `"Answer:"`; the natural model continuation is the letter, but scoring targeted the answer text. The documented production long run used `--scoring_mode first_token`.

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

### 2.1 [MAJOR] Headline contrast AUC / Cohen's d are in-sample (FIX LANDED 2026-05-27/28 — geometry + subspace + SAE features all closed)

**Status:** Geometry + subspace held-out metrics landed 2026-05-27 (commits `e15e62f` / `51aa571`). The remaining SAE-features held-out reconstruction landed 2026-05-28 in commit `304ddb6`, and the in-sample plot-title label tweak in the same commit. All headline AUC / Cohen's d are now held-out.

- `e15e62f` — `analyze_identity_geometry.py` + `analyze_identity_geometry_diagnostics.py`: `auc_all`/`cohens_d_all` renamed `_in_sample`; new `contrast_holdout_summary.csv` and `contrast_family_holdout_residualized_summary.csv` aggregate the family-holdout rows into headline mean/sd/min/max per (layer, contrast). `plot_identity_geometry.py:plot_contrasts` makes held-out the headline plot (`contrast_auc_by_layer.png`) and demotes in-sample to `_in_sample`-suffixed diagnostic plots.
- `51aa571` — `analyze_shared_social_subspace.py`: in-sample decomposition columns renamed `_in_sample`. New `decomposition_rows_holdout` does leave-one-family-out: re-derives every contrast direction on non-`f` rows, re-SVDs a held-out basis, decomposes each direction onto that basis, evaluates shared / residual / full components on the `f` rows. Writes `decomposition_metrics_holdout.csv` (per fold) and `decomposition_metrics_holdout_summary.csv` (mean / sd / n_folds per contrast × k × component). `aggregate_axis_sharedness` and `plot_axis_summary` updated to read the renamed in-sample columns and mark their plots `DIAGNOSTIC`.
- `51aa571` — `analyze_identity_sae_features.py`: in-sample `auc`/`cohens_d`/`full_direction_*` columns in `direction_reconstruction.csv` renamed `_in_sample`. Held-out reconstruction (which requires held-out feature *selection*, not just held-out direction) is bundled with the 2.5 winner's-curse fix.

**Original audit (preserved):**

`analyze_identity_geometry.py:run_contrasts` and `analyze_identity_geometry_diagnostics.py:run_contrasts` compute the contrast direction from `mean(A) − mean(B)`, then evaluate AUC/Cohen's d of the projection **on the same A and B prompts**. `analyze_shared_social_subspace.py:evaluate_component` and `analyze_identity_sae_features.py` do the same. In-sample separation is optimistically biased — a difference-of-means direction is *defined* to separate the two means.

**SAE-features held-out reconstruction (commit `304ddb6`, 2026-05-28):**
- New `reconstruction_holdout_rows` in `analyze_identity_sae_features.py` does leave-one-family-out: for each held-out prompt `family` it recomputes the contrast direction AND the per-feature selection ranking (selectivity Cohen's d, decoder alignment, combined_score) on the train rows, then evaluates the reconstructed direction's AUC / Cohen's d only on the held-out family. `summarize_reconstruction_holdout` aggregates to one headline row per (layer, contrast, selection_method, k). Writes `direction_reconstruction_holdout.csv` (per fold) + `direction_reconstruction_holdout_summary.csv` (headline); `direction_reconstruction.csv` is retained as the in-sample diagnostic. This is also the identity-screen half of audit 2.5 (held-out feature *selection*).
- `plot_identity_directional_followups.py` projection-histogram titles now say `(in-sample)` so they are not mistaken for held-out numbers.
- Validation (synthetic, 15/15, decoder = identity): held-out AUC ~0.99 with a true signal and ~0.54 (chance) on pure noise — the in-sample optimism is removed; folds == families, each evaluated only on the held-out family; random baseline beaten by selectivity. ⚠️ **NEEDS RUNPOD:** full pass; confirm every contrast has ≥2 usable families so the LOFO summary is non-trivial.

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

### 2.3 [BLOCKER] Steering controls are disabled in the production run (FIX LANDED 2026-05-28)

**Status:** Closed in commit `42b5837` (`scripts/run_bbq_sae_steering.py`) and the operational doc update in `docs/bbq_steering_pipeline.md`. The audit's fourth-control ask (diff-of-means contrast direction) was already closed 2026-05-27 in audit 5.5 via `direction_baseline` (commit `a11cbb8`) + later `probe_baseline` (commit `8c392d7`).

**What landed:**
- **Controls in the batched first-token path.** Previously controls only ran in the slow per-example `answer_logprob` path; the production command (`--scoring_mode first_token --disable_controls`) silently shipped zero controls. The batched path now iterates `build_control_specs()` after the headline scoring and emits control rows with the same `job_id` scheme so resume is stable.
- **Audit-3.1-compatible control: `random_feature_ablate`.** The three existing controls are direction-addition-shaped; under the audit-3.1 default `--intervention_modes ablate`, the natural specificity control is "ablate K random features (matched to the headline feature set size)." Stamped on output rows as `control_type = "random_feature_ablate"`.
- **Mode-coupled control selection** via `build_control_specs(fs, intervention_modes, ...)`:
  - Direction-addition headlines → `sign_flip`, `random_direction_norm_matched`, `random_feature_matched`.
  - Feature-intervention headlines → `random_feature_ablate`.
  - Mixed headlines → both families.
- **`--controls_subsample_frac`** (default `1.0`, production `0.20`): deterministic per `(bbq_uid, fs.set_id)` SHA1 hash → resume-stable. Cuts control cost ~5× at `0.20`.
- **`--controls_positions`** (default `final_prompt_token`, matching prior behavior): pass `same_as_headline` to run controls at every position the headline runs at.
- **`--disable_controls` reframed** as smoke-test-only. The runner emits a startup WARNING if it's set on a run sized like production (`--max_examples > 50` or `--max_feature_sets > 5`).
- **`docs/bbq_steering_pipeline.md`** production command updated: `--disable_controls` removed, replaced with `--controls_subsample_frac 0.20`.

**Validation (synthetic):** `--controls_subsample_frac=0.20` selects 19.91% of 10k synthetic pairs (SHA1-deterministic); `ablate` headline yields only `{random_feature_ablate}`; `add_vector` / `direction_baseline` / `probe_baseline` yield the three direction-shaped controls; mixed headlines yield both families; same `seed_input` → same random feature IDs (resume-stable).

**Original audit (preserved):** `run_bbq_sae_steering.py` implemented three controls — `sign_flip`, `random_direction_norm_matched`, `random_feature_matched` — but they were gated behind *not* `--disable_controls`, and the documented production command in `docs/bbq_steering_pipeline.md` passed `--disable_controls`. Without controls, a feature's effect cannot be claimed specific. A norm-matched random direction added at the same position may shift the bias margin just as much. The whole "feature X is causally implicated in bias" claim needs: effect(feature X) ≫ effect(random direction) ≫ effect(random feature set), at matched norm.

### 2.4 [MAJOR] `answer_logprob` summed over different-length answers (FIX LANDED 2026-05-28)

**Status:** Closed in commit `8ef171c` (`scripts/run_bbq_sae_steering.py`). The complement to audit 1.3 (commit `2829417`, which switched the headline default to `--scoring_mode letter` and dissolves the bias entirely). This commit handles the `--scoring_mode answer_logprob` case used for confirmatory runs on top features.

**What landed:**
- New `answer_lengths(tokenizer, answers)` returns per-answer token-span lengths under the same tokenization the `answer_logprob` scorer sums over.
- `row_metrics` gains optional `base_lengths` / `inter_lengths` / `scoring_mode` kwargs. When lengths are provided, `predicted_*` / `correct_*` / `prediction_changed` are computed on per-token mean logprobs (`base / lengths`), not raw sums. New columns: `ans*_logprob_per_token_*`, `ans*_token_length`, `argmax_length_normalized` (bool), `scoring_mode`.
- `steering_output_row` and `control_output_row` thread the lengths through; the per-example loop computes `answer_lens_for_row` once per example under `answer_logprob` and reuses it across all `(alpha, position, mode, control)` rows.
- Within-example deltas (`stereotyped_delta`, `bias_margin_delta`, etc.) are not normalized — they already canceled length per the audit's analysis.

**Validation (synthetic):** asymmetric case with per-token logprobs `[-2.0, -2.0, -1.6]` and correct = "Cannot be determined" — raw argmax wrongly picks "The boy" (shortest, length-biased); length-normalized argmax correctly picks "Cannot be determined." The audit's exact failure mode reproduced and fixed.

**Original audit (preserved):** `score_answer_logprob` sums per-token logprobs over the answer span. BBQ's three options have different token lengths; `"Cannot be determined"` is typically the longest, so summed logprob systematically penalized the unknown option. Within-example deltas (intervened − base) canceled the length bias because length is constant per example — so `stereotype_preference_delta` etc. were OK. But `predicted_base`, `correct_base`, `prediction_changed`, and `accuracy_delta` used `argmax` over raw summed logprobs and were length-biased. Baseline accuracy and any accuracy-change metric were contaminated.

### 2.5 [MAJOR] Selection-induced bias ("winner's curse") in feature effect sizes (FIX LANDED 2026-05-28 — all halves closed)

**Status:** Three pieces, all closed. (1) Identity-prompt selectivity prefilter — commit `4481445` (compute d/AUC for ALL features, no pre-screen). (2) BBQ winner's-curse on rankings/candidates — commit `b5150ec` 2026-05-28 (held-out selection/confirmation split; part 2/3 of the analyzer inference rework with 2.6/2.7). (3) Identity-screen held-out feature *selection* — commit `304ddb6` 2026-05-28, the leave-one-family-out held-out reconstruction (features re-selected per fold on train rows, reconstruction evaluated on the held-out family); see the 2.1 entry for details. The per-feature Cohen's d in `feature_selectivity.csv` remains an in-sample *descriptive* screen statistic by design; the held-out generalization evidence is `direction_reconstruction_holdout_summary.csv`.

**What landed (BBQ half, commit `b5150ec`):**
- `assign_holdout_split()` deterministically partitions BBQ examples into a **selection** and a **confirmation** set, keyed on `bbq_uid` (an example is in the same half for every feature, so the halves are disjoint example sets) and salted by `--holdout_seed`; ~`--holdout_frac` (default 0.5) per axis, with the realized per-axis balance logged.
- The headline `feature_inference` table (the feature-level unit from 2.6) now **ranks features on the selection half** and **reports effect sizes, CIs, and q-values from the disjoint confirmation half**. `selection_mean_signed_stereotype_preference_delta` + `n_selection` carry the ranking effect; the unprefixed `mean_*`/`ci_*`/`q_value_fdr` + `p_value_confirmation` + `n_confirmation` are the reported confirmation values. `make_rankings` and `final_intervention_candidates_table.html` consume this table.
- Held-out is ON by default; `--disable_holdout` reverts to the pooled (winner's-curse-prone) estimate for diagnostics. The `--min_examples_inference` floor (2.7) applies to **both** halves before FDR.

**Validation (synthetic, part of the 17/17 part-2 suite):** the split is deterministic / ~50/50 / seed-varying / one-half-per-uid; selection and confirmation effects are computed on disjoint data; a **pure-noise winner's-curse demo** shows the top-by-selection units shrinking toward 0 on confirmation (mean |effect| 0.0027 → 0.0020); rankings order by the selection effect. ⚠️ **NEEDS RUNPOD:** full analyzer pass on real steering output, and a check that each axis has enough examples for a 50/50 split at `--min_examples_inference`.

**Original audit (preserved):** Identity-prompt selectivity half closed in commit `4481445` (`scripts/analyze_identity_sae_features.py`). The BBQ half (`analyze_bbq_feature_level_causal_effects.py`) and the held-out confirmation-set work were open at audit time.

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

### 2.6 [MAJOR] Multiplicity is inflated by the alpha × position grid (FIX LANDED 2026-05-28)

**Status:** Closed in commit `6e132d3` (`scripts/analyze_bbq_feature_level_causal_effects.py`, part 1/3 of the analyzer inference rework).

**What landed:**
- New headline `feature_inference` table whose **unit of inference is the feature** (× layer × `intervention_position` × `context_condition`), tested at a single pre-registered `--headline_alpha`. `resolve_headline_alpha` infers the alpha when only one is present (the audit-3.1 `ablate` default has just `alpha=0.0`) and **requires** `--headline_alpha` for a multi-alpha steer/clamp grid — the rest of the grid feeds the dose-response *plots*, not separate tests.
- Question **polarity is pooled** into the unit (valid because audit 4.3 made the signed metric comparable across polarities; also raises power), as are identity pairs. The remaining grouping columns (axis, role, contrast, estimate-type, direction) are feature-constant under matched-axis steering and do not fragment a feature.
- **FDR (BH) is computed across FEATURES** within `(axis × context × position)` — the actual family of hypotheses — instead of within `(axis, context, alpha, position)` strata that never pooled the correlated alpha tests.
- The per-`(alpha, position)` `feature_level_effects` table is retained only as a dose-response diagnostic; its q-values are no longer the headline significance. Rankings + the final-candidates report consume `feature_inference`.

**Validation (synthetic, part of the 13/13 part-1 suite):** single-alpha auto-inference + multi-alpha enforcement; exactly one tested unit per feature with the two polarities pooled; FDR computed within axis strata; controls excluded. ⚠️ **NEEDS RUNPOD:** full analyzer pass on real steering output.

**Original audit (preserved):** `analyze_bbq_feature_level_causal_effects.py` produced one significance test per `(feature, layer, alpha, position, role, contrast, axis, context, polarity, …)` group. A single feature was tested at 6 alphas × 3 positions = 18 highly-correlated tests. FDR (`fdr_bh`) was applied within `(axis, context, alpha, position)` strata, so it did not even pool those 18 — and treating correlated tests as independent both inflates the count and mis-estimates FDR. What to do: decide the unit of inference up front (the feature, optionally feature × position), summarize the dose-response into one statistic per feature, test once, FDR across features; keep the alpha grid for dose-response plots only.

### 2.7 [MINOR] Underpowered cells and small permutation/bootstrap budgets (FIX LANDED 2026-05-28)

**Status:** Closed in commit `6e132d3` (`scripts/analyze_bbq_feature_level_causal_effects.py`, bundled with the 2.6 part-1 change).

**What landed:**
- `--bootstrap_samples` / `--permutation_samples` defaults raised 1000 → **10,000**.
- New `--min_examples_inference` (default 30) is a per-unit power floor on the headline `feature_inference` table; underpowered units are dropped **before** FDR so q-values reflect the actual tested set, not the pre-filter superset. Under the 2.5 held-out split the floor applies to **both** halves (so ≥60 examples/unit total at the default).
- `--smoke` still caps resamples at 500 for fast dev runs but now emits a loud startup WARNING that the run's CIs/q-values are underpowered and must not be cited; the production command drops it (see operational doc).
- The coarser unit of inference from 2.6 (feature, not feature × alpha × polarity × identity-pair) directly raises per-unit n, which is the other half of the power fix.

BCa bootstrap was considered (audit's "consider") and not adopted — with the coarsened unit + held-out split + 10k percentile bootstrap, n per unit is large enough that percentile and BCa converge; recorded as optional future work.

**Validation (synthetic):** the `--min_examples_inference` filter empties the table when the floor exceeds cell size (part-1 suite) and drops units when either held-out half is too small (part-2 suite).

**Original audit (preserved):** `min_examples = 10` (and `--smoke` lowers nothing below that). A sign-flip permutation test on 10 paired deltas has only 2¹⁰ = 1024 distinct sign assignments — minimum p ≈ 1/1024 — and after FDR almost nothing can reach significance. The documented command also passed `--smoke`, which caps bootstrap/permutation at 500 (min p ≈ 0.002) — and the fact that the production command still said `--smoke` suggested no full-budget run had been done. What to do: drop `--smoke`; use ≥10,000 bootstrap/permutation samples; raise per-cell minimums (or coarsen grouping, per 2.6); consider BCa for small n.

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

### 3.2 [MAJOR] Steering magnitude is uniform and untethered to feature scale (FIX LANDED 2026-05-28)

**Status:** Closed for the feature-intervention paths (`clamp`, `steer`) in commit `22e8345` across `scripts/encode_identity_saes.py` (primitives) and `scripts/run_bbq_sae_steering.py` (runner). The audit's recommended approach — per-feature lookup against `feature_stats.csv` × a multiplier grid — was implemented; the residual-RMS-norm alternative was not (the feature-stats approach is the one the audit preferred now that 3.1 has landed). `ablate` (the headline default) is unaffected, so no headline RunPod run changes. A clamp/steer headline is RunPod-pending and gated on the new audit script.

**What landed:**
- Primitives: `clamp_features` accepts a per-feature target (1-D tensor aligned with `feature_ids`) as well as a scalar; `steer_features` gains a per-feature `scale` so its increment becomes `alpha * scale[f] * sign[f]`. Both fall back to the old uniform behavior when no per-feature value/scale is given.
- New CLI: `--feature_scale_stat {none,p95,p99,max,mean_nonzero}` (default `p95`), `--feature_stats_dir` (per-layer `layer_<NN>/feature_stats.csv` from `encode_identity_saes.py`), `--clamp_multipliers` (grid, default `1.0`). Under a stat: **clamp target = `clamp_multiplier * scale[f]`** and **steer increment = `alpha * sign * scale[f]`**, so the grid value means the same thing — "how many p95s" — for every feature. `none` reproduces the pre-3.2 uniform path (clamp uses `--clamp_value`; steer adds raw `alpha`).
- The per-feature scale vector is built once per `FeatureSet` from its layer's `feature_stats.csv` (cached) and threaded through both scoring paths (batched first-token + per-example) and both hook installers. `clamp` now sweeps `--clamp_multipliers` as its grid when scaled (each multiplier a distinct job, recorded in the `alpha` column); job-id / resume / progress counting were made mode-aware so the grid is counted correctly.
- Validation at startup: a stat requires `--feature_stats_dir`; clamp requires `--clamp_value` only under `none`. Missing or non-positive per-feature scales are warned at build time (they collapse the amplitude to ~0).
- Output schema: **two columns added, none removed** — `feature_scale_stat` (the stat used, or `none` for ablate/uniform rows) and `feature_scale_value` (the single-feature scale; NaN for bundles and ablate). For clamp the effective per-feature target = `alpha * feature_scale_value`; for steer the increment = `alpha * sign * feature_scale_value`. Downstream `analyze_bbq_feature_level_causal_effects.py` is unaffected.
- New `scripts/audit_feature_scale.py` (RunPod pre-flight): checks the triage feature pool against the saved per-layer `feature_stats.csv` for the chosen stat and exits non-zero if too many kept features are absent (→ scale 0) or have a non-positive stat (→ clamp degenerates to ablate, steer to a no-op).

**Validation (synthetic only — no GPU / Llama / SAE locally):** 30/30 checks pass, including the central comparability property — with `multiplier=2`, each feature is clamped to 2× its **own** p95, the clamp targets preserve the natural-scale ratio across features (100× p95 → 100× target), and the uniform path collapses both features to one shared value (the bug). Also covered: ablate unaffected, the multiplier-grid wiring, missing/dead-feature handling, the `feature_stats.csv` round-trip. ⚠️ **NEEDS RUNPOD:** run `scripts/audit_feature_scale.py` against the real triage pool + `feature_stats.csv`, then a clamp/steer headline; until then no per-feature-scaled clamp/steer result should be cited.

**Relation to 3.1:** this is the magnitude complement to the 3.1 intervention-locus fix. 3.1 made the intervention a genuine feature edit (encode→modify→decode→patch); 3.2 makes the *amplitude* of that edit comparable across features. `ablate` needs neither (it sets the latent to 0), which is why it remains the audit-preferred primary causal test.

**Original audit (preserved):**

Even within the current direction-addition design, `alpha ∈ {±2, ±4, ±8}` is applied to a **unit** vector, identically for every feature. Features differ enormously in natural activation magnitude and decoder norm. A fixed alpha is neither "amplify feature f by X%" nor a constant fraction of the residual-stream norm at layer 24 (which is itself large and varies by token). So a feature showing a big effect at α=8 may simply have received a *relatively larger* perturbation than another.

The same problem now applies to the post-3.1 `--intervention_modes clamp` / `steer` paths: a single `--clamp_value` (or alpha) is applied across all features regardless of each feature's `p95`/`p99`. The `ablate` mode is unaffected (it sets the latent to exactly 0).

What to do: express the perturbation relative to a meaningful scale — e.g. as a multiple of the residual-stream RMS norm at that layer/position, or (better, now that 3.1 has landed) extend `--clamp_value` to accept a per-feature lookup against `feature_stats.csv` (`p95`/`p99`/`max`) and a multiplier grid. Then "α" is comparable across features.

### 3.3 [MAJOR] Intervention positions are located by greedy regex and may hit the wrong span (FIX LANDED 2026-05-28)

**Status:** Closed in commit `afb3ee3` across `scripts/extract_bbq_token_level_sae_activations.py` and `scripts/run_bbq_sae_steering.py`.

**What landed (Step 19 — extractor):**
- New helper `overlap_in_section(start, end, term_spans, section_span)` returns True only when a token overlaps a term match AND lies within the given section span.
- Per-token rows gain nine new boolean columns: `is_target_identity_token_in_{context, question, answer_option}`, plus `nontarget` and `stereotype_language` counterparts.
- `bbq_token_level_sae_summary.csv` gains nine new mean-activation columns: `mean_*_activation_in_{context, question, answer_option}` (NaN when the section mask is empty for that feature). Downstream consumers can answer "is this feature firing on the **context** identity mention or just the answer-option mention?" without re-deriving from the token table.

**What landed (Step 20 — steering runner):**
- Six new section-explicit position names alongside the legacy ones: `target_identity_last_context_token`, `target_identity_last_question_token`, plus `nontarget_*` and `stereotype_language_*` variants. `positions_for` clips term-match spans to the named section via `find_section_spans` + `intersect_spans_with_section` BEFORE taking the last-token argmax. Falls back to `final_prompt_token` (with `intervention_section = "final"` on the row) when no in-section match exists.
- New `position_section_for(...)` classifies the chosen position into `{context, question, answer_option, final, mixed, unknown}`.
- New `intervention_section` column on every `results_parts/*.parquet` row, stamped by both call sites (batched first-token path and per-example scoring path). The downstream analyzer can `groupby("intervention_section")` to stratify any effect table — exactly what the audit asked for ("Record, per job, which section the intervened token fell in").

**Validation (synthetic, audit's pathological prompt):** legacy `target_identity_last_token` lands in `answer_option`; new `target_identity_last_context_token` lands in `context`; new `stereotype_language_last_question_token` lands in `question`; no-match fallback lands in `final`.

**⚠️ Pre-RunPod validation gap:** the synthetic test confirmed the helpers' math on **one** prompt with a **whitespace-splitting fake tokenizer**, not against the real Llama tokenizer or real BBQ prompt format. Likely silent-failure modes: (i) Llama BPE offsets straddling section boundaries, (ii) `find_section_spans` failing due to `--few_shot_pool` prefix or whitespace normalization (causes every `intervention_section` to fall back to `"final"`), (iii) first-match-wins in `prompt.find` picking a wrong section span when text repeats across sections. **Use `scripts/audit_intervention_sections.py`** (added 2026-05-28) before any steering job that depends on the new section-explicit positions; it runs four threshold-gated checks against the real Llama tokenizer and exits non-zero on failure. Full caveat list + script usage: [Step 19 doc — Pre-RunPod validation caveats (3.3 fix)](pipeline_steps/19_extract_bbq_token_level_sae_activations.md#-pre-runpod-validation-caveats-33-fix). Until that audit script passes, no headline causal claim should reference the new section-explicit positions.

**Original audit (preserved):** `run_bbq_sae_steering.py:positions_for` (and `extract_bbq_token_level_sae_activations.py:find_all_spans`) located `target_identity_last_token` by searching the prompt for the identity label *and* the answer-option text, collecting **all** matches, and taking `max(pos)` — the last. The identity label often appears in the context, the question, *and* the listed answer option `B. the Black man`; the last occurrence was in the answer-choice list. So `target_identity_last_token` was frequently intervening on the identity token *inside the answer choice*, not the identity mention in the context. `stereotype_language_last_token` had the same problem (question content words recur in context/answers). The position names implied a causal locus ("the feature acts at the identity mention") that the implementation did not guarantee.

### 3.4 [MAJOR] BBQ→SAE contrast mapping silently uses axis-fallback (FIX LANDED 2026-05-27)

**Status:** Closed in commit `56a5f7e` (`scripts/run_bbq_sae_steering.py`). The mapping logic in `prepare_bbq_for_steering.py:map_contrast` is unchanged (it still emits `exact` / `fallback_axis` / `unmapped`); the steering runner now defaults to exact-only and stamps the confidence on every output row.

**What landed:**
- `--include_unmapped` (boolean) replaced with `--mapping_confidence_filter` (default `exact`; choices `exact`, `exact_and_fallback`, `all`). Headline runs use the default and silently drop `fallback_axis`. Users who want fallback rows for a separate analysis pass `exact_and_fallback`; full inclusive passes `all`. The runner prints the kept-row count and per-confidence breakdown to stdout.
- `mapped_contrast_confidence` is stamped on every output row in `steering_output_row`. Previously the column was implicitly available by re-merging against the prepared parquet; downstream `analyze_bbq_feature_level_causal_effects.py` rows carried `mapped_contrast_name` but not the confidence. Now stratifying any effect table by mapping confidence is a `groupby` away.

**Behavior change:** under the prior default (`--include_unmapped` unset), the runner kept rows with confidence in `{exact, alias, fallback_axis}` — effectively `{exact, fallback_axis}` since `alias` never actually flows out of `map_contrast`. Under the new default (`--mapping_confidence_filter exact`), only `exact` rows are kept. Production runs that need the prior superset pass `--mapping_confidence_filter exact_and_fallback`.

**Original audit (preserved):** `prepare_bbq_for_steering.py:map_contrast` returns `exact` when the BBQ example's `(target, nontarget)` identities match an SAE contrast, otherwise `fallback_axis` (any contrast on the same axis), otherwise `unmapped`. `run_bbq_sae_steering.py` kept rows with confidence in `{exact, alias, fallback_axis}` by default. So a BBQ item about `race_arab vs race_white` could be steered with features selected for `race_black vs race_white`, and the downstream analyzer treated `mapped_contrast_name` as the relevant contrast. Feature-to-example matching is a load-bearing assumption for "this feature is implicated in *this* identity's bias"; `fallback_axis` broke it while leaving the data looking clean.

### 3.5 [MINOR] Bundle steering averages decoder rows into one direction (FIX LANDED 2026-05-28)

**Status:** Closed in commit `9b0fac8` (`scripts/run_bbq_sae_steering.py`). Structurally **superseded by the 3.1 feature-intervention path**: a bundle feature set run under `ablate`/`clamp`/`steer` already edits all latents in the set simultaneously. This commit adds the operator-facing guard + documentation that makes that the recommended (and, under the `ablate` default, actual) bundle path; the averaged-decoder direction is retained only for the legacy modes and the 5.5 baseline.

**What landed:**
- A bundle `FeatureSet` (`per_contrast_topk` / `role_bundle`, i.e. >1 `feature_id`) flows through `install_feature_intervention_hook` with the **full** `feature_ids` list, so `ablate`/`clamp`/`steer` intervene on every latent in the set **simultaneously** — the audit's recommended "clamp this set of latents simultaneously." Under the audit-3.1 default `--intervention_modes ablate` this is already the behavior; with audit 3.2 a bundle `clamp`/`steer` scales each member latent by its **own** `feature_stats` value (per-feature, not one shared magnitude). The averaged `make_vector` direction is now reached only by the legacy `add_vector`/`ablate_projection` modes and the direction-shaped controls.
- Soft guard in `main()`: warns when bundle feature sets will run under a legacy averaged-decoder-vector mode, pointing to the set-intervention modes (the legacy modes stay available for the audit-5.5 linear-direction baseline). `make_vector`'s docstring is marked legacy/baseline-only and notes the feature-intervention path never calls it.

**No behavioral change** to the default `ablate` path or the output schema: bundles were already stamped `feature_estimate_type="feature_bundle"` (`feature_id` sentinel `-1`, `n_features_in_set` = bundle size) and the downstream analyzer tags them `feature_bundle_membership`.

**Validation (synthetic, 13/13 pass):** bundle `ablate` zeros every member latent and leaves non-members untouched; bundle `clamp` with distinct per-feature 3.2 scales sets each member to `mult × scale[f]` (three **distinct** targets — proving it is not one averaged direction); the feature-intervention dispatch needs no averaged vector while legacy `add_vector` without a vector raises; bundle metadata tags `feature_bundle` / `-1` / `n=3`. No GPU / Llama / SAE exercised (the encode→decode→patch hook is unchanged from 3.1).

**Original audit (preserved):**

`per_contrast_topk` and `role_bundle` modes average signed decoder rows into a single vector. The downstream analyzer correctly tags these `feature_bundle_membership` and warns against single-feature claims — good. But averaging *unit-normed* rows then *re-normalizing* produces a direction whose relation to any individual feature is weak; bundle effects are hard to interpret even as "membership." With the 3.1 feature-level fix landed, bundle interventions can now be implemented as "clamp this *set* of latents simultaneously" via a multi-feature `ablate` / `clamp` call — cleaner and interpretable. Prefer that.

---

## 4. Data construction and conceptual coverage

### 4.1 [MAJOR] Contrast lists reference identities that do not exist (FIX LANDED 2026-05-27/28 — both halves closed)

**Status:** Geometry / subspace / SAE / plot side closed across four commits 2026-05-27. BBQ side closed in commit `26998ec` 2026-05-28 alongside audit 4.4.

**What landed:**
- `1e242c9` — new `scripts/contrast_registry.py` is the single source of truth. 21 canonical contrasts as 4-tuples (`contrast_name, identity_a, identity_b, axis`). Two typos fixed: `ses_low_income → ses_low`, `ses_high_socioeconomic_status → ses_high`. All 21 entries validate against `data/bbq_identity_normalized_forms.csv`. Module also exposes `KEY_CONTRAST_NAMES`, `SELECTED_CROSS_AXIS_ORDERINGS`, `load_validated_contrasts(...)`, `write_contrasts_skipped(...)`, `get_contrast_pairs(...)`, `filter_to_key_contrasts(...)`. **No startup assertion** — partial-axis runs work.
- `398ffee` — `analyze_identity_geometry.py` + `analyze_identity_geometry_diagnostics.py` source `CONTRASTS` from the registry at `main()` startup and write `contrasts/contrasts_skipped.csv` with per-row `reason` annotations + warning per skipped pair.
- `6eafb4d` — `analyze_shared_social_subspace.py` + `analyze_identity_sae_features.py` import `DEFAULT_CONTRASTS` (+ `KEY_CONTRASTS`, `SELECTED_CROSS_AXIS_ORDERINGS` in subspace). Local `load_contrasts(path, metadata, output_dir=...)` reworked to route through the registry validator and emit the skipped CSV.
- `<plot-commit>` — `plot_identity_directional_followups.py` + `plot_identity_directional_visualizations.py` + `plot_identity_sae_features.py` import their `DEFAULT_CONTRASTS` / `KEY_CONTRASTS` from the registry. The followups script's `RESIDUALIZATION_COMPARISON_CONTRASTS` and `CENTROID_ORDERING_CONTRASTS` get the `ses_low_income_vs_ses_rich → ses_low_vs_ses_rich` rename so the paper-summary panel actually plots a real contrast.

**Headline numerical effect:** SES axis now runs 4 contrasts (`ses_low_vs_ses_rich`, `ses_low_vs_ses_high`, `ses_lower_class_vs_ses_upper_class`, `ses_blue_collar_vs_ses_white_collar`) instead of the silently-2 from before. Any paper claim about "SES" coverage now matches the code's count.

**Original audit (preserved):** The `CONTRASTS` / `DEFAULT_CONTRASTS` lists across six scripts included `ses_low_income` and `ses_high_socioeconomic_status`. The identity-forms CSV had no such IDs. Every analysis did `if identity_a not in identity_set: continue` — so the typo contrasts were dropped with no error, and the SES axis quietly had fewer contrasts than the code implied. `prepare_bbq_for_steering.py:MANUAL_ALIASES` similarly maps to non-existent IDs.

**BBQ side closed (2026-05-28, commit `26998ec` — bundled with audit 4.4):**
- Broken `MANUAL_ALIASES` targets repointed: `ses_low_income → ses_low` (canonical low-SES bucket) for "low ses" / "low socioeconomic status" / "low income" / "lowses"; `ses_low_income → ses_poor` for "poor" (own identity exists); `ses_high_socioeconomic_status → ses_high` for "high socioeconomic status" / "highses".
- Aliases removed for identities that don't exist at all: `age_old` / `age_nonold` / `non old` (no age axis), `nationality_asia_pacific` / `nationality_african` / `nationality_european` (aggregate continents). BBQ rows that previously matched these now fall to `mapped_contrast_confidence=unmapped` and are filtered at Step 20 under the audit-3.4 default.
- New `validate_manual_aliases()` raises `ValueError` at startup on any missing target; future regression cannot slip through silently.

### 4.2 [MAJOR] Intersectional BBQ categories are flattened to a single axis (FIX LANDED 2026-05-28)

**Status:** Closed in commit `b189aef` (`scripts/prepare_bbq_for_steering.py`) with the audit's path (b) — explicit exclusion. Path (a) — first-class compound contrasts — is recorded as future work; it would need templated compound prompts that the geometry pipeline doesn't currently produce.

**What landed:**
- `race_x_gender` and `race_x_ses` removed from `AXIS_MAP`. New CLI `--intersectional_handling {drop, axis_flatten}` (default `drop`). Default behavior: intersectional rows are excluded with per-category counts logged to stdout and added to `bbq_prepare_summary.csv` as `n_intersectional_dropped_*` metrics. `axis_flatten` opt-in preserves the legacy "collapse to race_ethnicity" behavior, but every flattened row is stamped `is_intersectional=True` so downstream consumers can stratify.
- New `is_intersectional` column on `bbq_prepared_examples.parquet` (always present; `False` for non-intersectional rows). Under the audit-3.4 default mapping filter (`--mapping_confidence_filter exact`), intersectional rows passing through under `axis_flatten` would also be filtered at Step 20 because no contrast in the registry matches an intersectional pair — but the `is_intersectional` flag makes the exclusion explicit rather than relying on the unmapped-fallback to silently drop them.
- New helper `resolve_intersectional(category, handling) -> (axis_or_None, is_intersectional)`.

**Path (a) — recorded as future work:** Genuine intersectional handling would require (1) templated compound prompts in the identity-geometry corpus, (2) a compound-contrast registry, (3) Step 7 compound direction computation, (4) Step 18 compound BBQ-label mapping, and (5) compound-aware contrast filtering. Substantial paper-extension work.

**Original audit (preserved):** `prepare_bbq_for_steering.py:AXIS_MAP` collapsed `race_x_gender` and `race_x_ses` to `race_ethnicity`. Intersectional examples have compound group labels (`F-Black`); `identity_components` splits them and `target_identity_id` becomes a single component. The intersectional structure was discarded. The project's stated subject is *marginalized identities*, and intersectionality is central to that literature; flattening Race×Gender to "race" both lost the most interesting cases and risked mislabeling the stereotyped answer.

### 4.3 [BLOCKER] `question_polarity` sign is not folded into the bias metric (FIX LANDED 2026-05-27)

**Status:** Closed in commit `a03760f` (`scripts/analyze_bbq_feature_level_causal_effects.py`).

**What landed:**
- `enrich_results` now computes `polarity_sign = +1 if question_polarity == "neg" else -1 if "nonneg" else NaN`, plus three signed delta columns: `signed_stereotype_preference_delta`, `signed_stereotyped_delta`, `signed_nonstereotyped_delta`. Positive signed delta = intervention made the model more biased, regardless of polarity.
- `summarize_effects` aggregates `mean_signed_*` columns alongside the unsigned originals (preserved as diagnostics). CI and the permutation p-value are computed on signed values. `n_polarity_skipped` records rows with unknown/missing polarity.
- `effect_label` branches on `mean_signed_stereotype_preference_delta`. The `bias_amplifying` / `bias_reducing_*` labels now mean the same thing across both polarities. The `general_answer_suppression` branch also uses signed stereotyped/nonstereotyped deltas so it doesn't misfire on nonneg-polarity rows.
- `beneficial_score = -mean_signed_stereotype_preference_delta + mean_unknown_delta + max(0, mean_correct_margin_delta)` — high when intervention reduces bias on both polarities. `harmful_score`, `substitution_score`, and all rankings + plots that ranked by the bias metric (`make_rankings`, `plot_feature_bars`, `plot_top_by_contrast`, `plot_amp_suppression_scatter`, `plot_dose_response`, `plot_answer_role_shift`) now use the signed metric.

**Validation (synthetic, 50 neg + 50 nonneg, both anti-bias by symmetric Δ=1.5):**
- Per-row: signed delta is negative on both polarities; unsigned is -1.5 on neg and +1.5 on nonneg.
- Aggregate: mean unsigned = 0.000 (cancels — the polarity confound). Mean signed = -1.500 (correctly anti-bias).
- effect_label = `bias_reducing_substitution`. beneficial_score = +1.500.
- Missing polarity → NaN signed delta → excluded from aggregates and counted in `n_polarity_skipped`.

**Original audit (preserved):** BBQ has negative-polarity questions ("Who was bad at X?") and non-negative ones ("Who was good at X?"). `stereotyped_groups` is fixed (it is the group the *negative* stereotype targets). For a **negative** question, choosing the stereotyped group is the *bias-consistent* answer; for a **non-negative** question, choosing the stereotyped group is the *anti*-bias answer. `analyze_bbq_feature_level_causal_effects.py` defined `stereotype_preference_delta = Δ[log p(stereotyped) − log p(unknown)]` with no polarity sign. The grouping keys included `question_polarity`, so individual rows were separable — but `effect_label` (`bias_amplifying` if `bias_delta > threshold`), `beneficial_score`, `harmful_score`, and `make_rankings` did not condition on polarity. A feature that raised `log p(stereotyped group)` was labeled "bias-amplifying" even on non-negative items where that is the unbiased direction. `final_intervention_candidates_table.html` was sorted by a polarity-confounded `beneficial_score`.

### 4.4 [MINOR] `MANUAL_ALIASES` has dozens of duplicate `"nondisabled"` keys (FIX LANDED 2026-05-28)

**Status:** Closed in commit `26998ec`, paired with the BBQ side of audit 4.1 (`MANUAL_ALIASES` had targets that didn't exist in the identity-forms CSV — the audit explicitly bundled the two items).

**What landed:**
- `MANUAL_ALIASES` rewritten from 91 literal entries (56 distinct — the runtime dict was silently deduplicating ~35 `"nondisabled": "disability_nondisabled"` repeats plus a couple of `"non disabled"` duplicates) to 49 literal entries, all distinct, grouped by axis with comments.
- Broken targets repointed to canonical identities: `ses_low_income → ses_low` (for "low ses", "low socioeconomic status", "low income", "lowses"), `ses_low_income → ses_poor` (for "poor"), `ses_high_socioeconomic_status → ses_high` (for "high socioeconomic status", "highses").
- Aliases pointing at identities that don't exist at all were removed: `age_old` / `age_nonold` / `non old` (no age axis), `nationality_asia_pacific` / `nationality_african` / `nationality_european` (aggregate continents, not per-country IDs). BBQ rows that previously mentioned these now fall to `mapped_contrast_confidence=unmapped` and are filtered at Step 20 under the audit-3.4 default.
- New `validate_manual_aliases(identity_meta, logger)` is called at every startup. **Raises `ValueError`** on any missing target with a per-target `ERROR`-level log line listing the aliases that point at it — so a future regression (either re-introduced duplicates or a typo'd target) can't slip through silently. `bbq_prepare_summary.csv` records `manual_aliases_n_total` / `manual_aliases_n_distinct_targets` / `manual_aliases_n_missing_targets` for durable record-keeping.

The 4.1 audit explicitly bundled "audit every value against the identity CSV (4.1)" into the 4.4 fix; both items close together. The "unit test" the audit recommended is implemented as the runtime invariant in `validate_manual_aliases()` rather than a separate test file.

### 4.5 [MINOR] `works_*` template-compatibility flags are dead metadata

`create_dataset.py` decides whether a template×identity pair is realized purely by "is the required form column non-empty", ignoring the `works_is_adj` / `works_group` / … flags in the identity CSV. In practice forms are empty roughly when the flag is 0, so output is mostly correct, but the flags are unused. Either use them as the source of truth or delete them to avoid the impression of a constraint that is not enforced.

### 4.6 [MINOR] Top-64 SAE truncation may clip true activations (PARTIAL FIX LANDED 2026-05-27 / 2026-05-28)

**Status:** Two detection paths landed across commits `c6dbcfe` (fresh-encoder gate) and `8b1381b`+ (saved-artifact audit). Remediation — re-encode with larger `--top_k_save` if either flags a problem, then regenerate `analyze_identity_sae_features` + triage — still requires the RunPod run.

**What landed (two complementary gates):**

1. **Fresh-encoder gate, step 6** (commit `c6dbcfe`, 2026-05-27). `scripts/validate_sae_hook_alignment.py` re-encodes a sample of raw activations through the corrected JumpReLU, reports `reconstruction_l0_p50` / `p95` / `p99` / `mean` / `max`, and fails when `max_l0 > --top_k_save_threshold` (default 64). This is the un-truncated truth — the right number to size `--top_k_save` against. Catches "the SAE in principle has L0 > 64."

2. **Saved-artifact audit** (commit `8b1381b`+, 2026-05-28). `scripts/audit_identity_sae_l0.py` reads the SAVED `feature_indices_top{K}.npy` / `feature_values_top{K}.npy` and counts rows "at the cap" (all `top_k` saved values positive → truncation occurred). Tells the operator whether the EXISTING encoded files — and the triaged feature pool derived from them — are biased, without paying for a re-encode. Catches "your existing files actually hit the cap."

```bash
python scripts/validate_sae_hook_alignment.py --layers 0,8,16,24,32  # gate 1
python scripts/audit_identity_sae_l0.py       --layers 0,8,16,24,32  # gate 2
```

**Consequence chain (what to regenerate if either gate fails):**

1. Re-encode the affected layer(s) with `scripts/encode_identity_saes.py --top_k_save <N>`, where `N > max_l0` from gate 1 with headroom (~2× p99).
2. Re-run `scripts/analyze_identity_sae_features.py` against the regenerated encodings.
3. Re-run `scripts/triage_sae_identity_features.py` to rebuild `intervention_candidate_features_triaged.csv` — this is the feature pool that drives every BBQ steering result.
4. Step 19 (`extract_bbq_token_level_sae_activations.py`) reads the regenerated pool; the dense `encode_selected_features` it does is per-feature unchanged, but it now sees the corrected set.

**Still open (audit's optional "missed candidates" diagnostic):** the audit also suggested encoding the FULL SAE feature dimension on a stratified BBQ subsample and reporting which features have non-trivial BBQ activation but were filtered out by the identity-side triage. Requires the full SAE encoder (not just the kept-feature columns) and is recorded as future work; the gates above cover the immediate consequence (was the identity-side pool itself truncation-biased?), not the broader "are there BBQ-relevant features the identity-side triage missed even at the correct L0?" question.

**Original audit (preserved):** `encode_identity_saes.py` keeps only the top-64 features per row (`--top_k_save 64`); everything else is treated as exact zero downstream (`sparse_long` drops non-positive). The SAE is a 32× expansion (~131k features). If the SAE's true L0 at layer 24 exceeds 64 on some prompts, real activations are clipped to zero, which biases `mean_a` / `freq_a` downward for mid-ranked features and slightly inflates apparent contrast selectivity. If empirical max L0 is comfortably under ~50, 64 is fine — otherwise raise `top_k_save`.

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

### 5.2 [MAJOR] Triage roles are heuristic definitions, not validated findings (PARTIAL FIX LANDED 2026-05-27)

**Status:** Four commits landed the structural and methodological fixes (parts 1-4 of the audit's targeted-fix list). The two **validations** (behavioral criterion + inter-rater agreement) are deferred to RunPod / human labeling and tracked as outstanding work in the pre-registration doc.

**What landed:**
- **Part 1 — firing-count entropy (commit `7f2c302`):** `identity_entropy` and `token_entropy` previously called `entropy()` on raw activation magnitudes — not a motivated probability mass. Replaced with categorical entropy over firing counts: per-identity firing count = `freq_identity × n_identity`; per-token firing count = number of token-rows with `token_feature_activation > 0`. The implicit probability model is "given the feature fired somewhere, what is the probability it fired in identity / token i."
- **Part 2 — soft scoring head (commit `235b5f5`):** the 7-branch first-match cascade is gone. Each feature now has a 4-vector of soft role-fit scores (`role_fit_identity_token_local`, `role_fit_sentence_final_integrated`, `role_fit_shared_social_feature`, `role_fit_contrast_specific_identity`), and `keep_for_intervention` is a single-threshold rule on `max(role_fit_*) >= --min_role_fit_keep AND not low_signal AND not template_artifact AND max|d| >= --min_abs_cohens_d`. The audit's pathological case (span=0.71 vs shared=0.85 → permanently `identity_token_local`) now correctly picks `shared_social_feature`. Legacy `provisional_role` survives as `argmax(role_fits)` and is documented as descriptive only.
- **Part 3 — sensitivity sweep (commit `f306869`):** `--sensitivity_sweep` runs the full triage with each threshold and each score-weight tuple element perturbed one-at-a-time by `--sensitivity_perturb_fractions` (default `0.8,0.9,1.1,1.2`). Outputs `triage_sensitivity_per_feature.csv` and `triage_sensitivity_summary.csv` with role-change fraction, keep-change fraction, and `delta_n_keep` per perturbation. On synthetic 100-feature data the sweep ran 88 perturbations cleanly; top disruptor was `max_template_artifact_score_keep` flipping ~18% of role labels.
- **Part 4 — pre-registration (commit pending; docs only):** `docs/triage_preregistration_2026-05-27.md` pins the score weights, role-fit definitions, and keep-rule thresholds to the three commits above. It also frames the kept-feature count as the only load-bearing finding and the taxonomy as descriptive unless one of two validations passes (see below). Any future modification of those constants after BBQ data is materialized must be recorded as a changelog entry on the pre-registration doc with date and rationale.

**Remaining (validations of the taxonomy):**
- *Behavioral criterion* — under the audit-3.1 feature-level intervention, `identity_token_local` features must show a larger absolute `bias_margin_delta` at `target_identity_last_token` than at `final_prompt_token`, and `sentence_final_integrated` features the opposite. Paired-contrast signed-rank test on `keep_for_intervention = True` features, stratified by `provisional_role`. Requires the BBQ steering run with `--intervention_modes ablate` at multiple positions.
- *Inter-rater criterion* — two human labelers, stratified sample of 80 features (20 per role), Cohen's κ ≥ 0.6 against the cascade label. Rubric and sample list to be pre-registered as a sub-document of the triage pre-registration when the feature cards are regenerated post-1.4.

**Original audit (preserved):** `triage_sae_identity_features.py` built `template_artifact_score`, `sharedness_score`, `polysemanticity_score`, `contrast_specificity_score` as linear combinations with hand-picked weights and hand-picked thresholds, then ran a decision cascade to assign roles. As an engineering filter to choose which features to steer, this was fine; but the roles could not be presented as results — they were definitions, and the weights/thresholds were unjustified and unvalidated. The `entropy()`-based scores additionally treated activation magnitudes as if they were a probability distribution, which was heuristic.

### 5.3 [MINOR] `combined_score` sums three near-duplicate, equally-weighted metrics (FIX LANDED 2026-05-27)

**Status:** Closed in commit `3b48e5b` (`scripts/analyze_identity_sae_features.py`). The new formula is `combined_score = 0.5·z(|cohens_d|) + 0.5·z(|cosine_with_direction|)`. Selectivity uses Cohen's d only; alignment uses `|cosine|` at equal weight. Weights are surfaced in `run_config.json` (`combined_score_weights`, `combined_score_formula`, `combined_score_audit_note`). Schema of `feature_selectivity_alignment_joined.csv` unchanged.

**Validation:** Pathological pair (Feature A high d+auc / low cos, Feature B low d+auc / high cos) — old formula favored A by +2.00 z-units; new formula puts them at parity (50/50). Realistic synthetic sweep with d↔auc correlated: Spearman ρ(old, new) = 0.92, so ~8% of the ranking shifts toward alignment-strong features.

**Original audit (preserved):** `combined_score = z(|d|) + z(|cos|) + z(|auc − 0.5|)`. Cohen's d and AUC both measure the same A/B separation and are monotonically related, so the score effectively double-weighted selectivity vs. decoder alignment. Propagated into `per_contrast_topk` feature selection (Step 19), the triage's `max_combined_score` aggregator (Step 17), and the BBQ steering pool (Step 20).

### 5.4 [MINOR] Representation inconsistency: residualized direction vs. raw-encoded SAE features (FIX LANDED 2026-05-27)

**Status:** Closed in commit `ebfdff7` (`scripts/analyze_identity_sae_features.py`). The script is now raw end-to-end.

**What landed:** `--residualization` flag and the `residualize(x, …)` call removed. `decoder_alignment` and `reconstruction_rows` no longer take a `residualization` parameter, and the `residualization` column is gone from `decoder_direction_alignment.csv` and `direction_reconstruction.csv`. `run_config.json` records `representation: "raw"` with an audit note. No downstream consumer (`plot_identity_sae_features.py`, `triage_sae_identity_features.py`, `build_sae_feature_cards.py`, `extract_token_level_sae_activations.py`, `run_bbq_sae_steering.py`) reads the dropped column.

The alternative consistent choice (re-encode residualized activations through the SAE) would require importing `encode_full` from Step 5 and regenerating `long_df`. Left as a future option if template/family variance contamination turns out to matter on the corrected (audit 1.4) encodings.

**Original audit (preserved):** `analyze_identity_sae_features.py` computed contrast directions from `family_residualized` activations, but the SAE features (`long_df`) were encoded by `encode_identity_saes.py` from **raw** activations. `decoder_alignment` then took the cosine between a raw-space decoder row and a residualized-space direction, and `combined_score` mixed a residualized-direction cosine with a raw-SAE-activation Cohen's d. The two lived in slightly different spaces.

### 5.5 [MAJOR] Missing baseline: does the SAE beat a difference-of-means direction? (PARTIAL FIX LANDED 2026-05-27 / 2026-05-28)

**Status:** Code path for all three audit baselines is closed across four commits — `8f84e5e` (Step 7 persists DoM directions) + `a11cbb8` (Step 20 adds `direction_baseline`) + `7cdb164` (Step 7 persists logistic-probe directions — audit option (c)) + `8c392d7` (Step 20 adds `probe_baseline`). The RunPod three-way head-to-head run is the remainder.

**What landed:**
- [Step 7](pipeline_steps/07_analyze_identity_geometry.md) (`analyze_identity_geometry.py`) now persists:
  - `contrasts/contrast_directions_layer_{LL}.npz` — unit-norm diff-of-means direction per `(layer, identity_a, identity_b)`. Already computed in `run_contrasts`; previously thrown away.
  - `contrasts/contrast_probe_directions_layer_{LL}.npz` — unit-norm weight vector from a binary L2-regularized `LogisticRegression` fit per contrast in **raw `d_model` space**, by the new `run_contrast_probes` function (audit 5.5 option (c)). This is deliberately separate from the existing axis/identity-within-axis probes, which run in PCA-reduced 256-D space — useful for measuring decodability, but the weight vector lives in PCA coordinates and cannot steer.
  - `contrasts/contrast_probe_scores.csv` — diagnostic per `(layer, contrast)` row with in-sample AUC / Cohen's d / held-out-family AUC, so the probe quality is sanity-checkable before it's used as a steering baseline.
  - New CLI: `--skip_contrast_probes` (default off), `--contrast_probe_C` (default 1.0).
- [Step 20](pipeline_steps/20_run_bbq_sae_steering.md) (`run_bbq_sae_steering.py`) has two new baseline intervention modes:
  - `direction_baseline` — `h += alpha * vec` where `vec` is the DoM direction.
  - `probe_baseline` — `h += alpha * vec` where `vec` is the logistic-probe direction.
  Both share `add_vector`'s hook plumbing; the mode string differs so output rows can be stratified.
- Two parallel CLI flags: `--direction_baselines_path` and `--probe_baselines_path`. Each accepts either a single `.npz` or a directory; when given a directory, the loader globs the matching filename pattern so the two sources stay disjoint. The single helper `_baseline_vector_for_mode` picks the right vector per mode. Missing-vector errors mention the specific `--*-path` flag.
- Bundle-mode feature sets (empty `contrast_name`) are skipped with a logged warning — the baselines are defined per-contrast.

**Validation:** synthetic 60-prompt × `d_model=64` toy with a known separator + noise: the probe recovers the true direction at cosine 0.988, AUC = 1.000, Cohen's d ≈ 16, held-out family AUC = 1.000. End-to-end hook math: `direction_baseline` adds the DoM vec and `probe_baseline` adds the probe vec at the chosen positions with max diff = 0.00 from expected. Loader correctly disambiguates DoM vs probe by glob pattern.

**Remaining (RunPod):** Run the three-way command against the (audit-1.4 re-encoded) feature pool:
```
python scripts/run_bbq_sae_steering.py \
    --intervention_modes ablate,direction_baseline,probe_baseline \
    --direction_baselines_path .../geometry/.../contrasts \
    --probe_baselines_path    .../geometry/.../contrasts
```
Output rows are pre-stratified by `intervention_mode`, so the analyzer just groups by it. If SAE features do not beat **both** linear baselines on the same prompts/positions, the paper should be reframed around directions — the audit's framing note flags this as load-bearing for the SAE story.

**Original audit (preserved):** Throughout, the difference-of-means contrast direction is computed *and* SAE features are computed, but they are never put in head-to-head competition as *interventions*. The key scientific question for an SAE-based paper is "does decomposing into SAE features buy anything over a single linear direction?" The audit's prescription: steer with (a) individual SAE feature interventions, (b) the raw difference-of-means contrast direction, (c) a logistic-probe direction; compare causal effect on BBQ. If SAE features do not localize or do not beat the linear direction, that is still publishable (and honest) — but the comparison must be run.

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

**Follow-up extension (2026-05-27, commit `402731f`):** the `feature_localization_type` 4-branch classifier was a 5.10 instance not covered by `e50bbd1`. It is now `common.classify_feature_localization(max_token_activation, max_identity_span_activation, final_token_activation, threshold=0.7)`, called by both `extract_token_level_sae_activations.py` and `build_sae_feature_cards.py`. Constants surfaced as `DEFAULT_LOCALIZATION_THRESHOLD = 0.7` and `LOCALIZATION_TYPES` tuple in common.

---

## 6. What is already done well (keep these)

Stated so the audit is balanced and these are not lost in a refactor:

- **Surface-form residualization diagnostics** (`analyze_identity_geometry_diagnostics.py`): residualizing by `family` / `template_id` / `required_form` and re-running PCA/probes/contrasts directly attacks the "are we measuring identity or template" confound. This is the right instinct and a genuine strength.
- **Family-holdout / family-to-family generalization**: training a direction on some template families and testing on held-out ones is a real generalization test (just make it the headline — see 2.1).
- **Variance decomposition (η²)** by metadata factor is a clean, honest way to show how much variance identity explains relative to surface form.
- **The bias taxonomy** in `analyze_bbq_feature_level_causal_effects.py` distinguishing `bias_reducing_uncertainty` (mass moves to "unknown") from `bias_reducing_substitution` (mass moves to the other identity) is conceptually sharp — a steering result that just swaps one stereotype for another is not debiasing, and the code knows that.
- **`individual_feature` vs `feature_bundle_membership`** is tracked explicitly; the analyzer warns when only bundle rows exist.
- **Controls are now on by default** (2.3 closed 2026-05-28). The batched first-token path runs them too; `--controls_subsample_frac` amortizes cost; `random_feature_ablate` is the audit-3.1-compatible specificity control.
- **Engineering hygiene**: resume/checkpointing everywhere, `run_config.json` / `*_config.json` for every run, the dedicated `validate_sae_hook_alignment.py`, explicit HF `hidden_states[k]` convention notes, axis-matching to prevent wrong-axis contamination. This is well above typical research-code standard and makes the fixes above tractable.

---

## 7. Priority punch list

Ordered by what most threatens a defensible result.

**Tier 1 — do before trusting any current number**

1. **Verify SAE preprocessing** (1.4): confirm LlamaScope normalization + activation function; add an encode→decode reconstruction-quality check to `validate_sae_hook_alignment.py`. If wrong, every SAE number is wrong.
2. **Fix the feature intervention** (3.1) — **FIX LANDED 2026-05-27.** Canonical torch primitives (`ablate_features`, `clamp_features`, `steer_features`, `patched_residual_with_intervention`) live in `scripts/encode_identity_saes.py` alongside `encode_full` / `decode_full`. `scripts/run_bbq_sae_steering.py` dispatches them via `--intervention_modes` (default: `ablate`). Commits `11d4a4d` (primitives) + `84c87b5` (hook + dispatch). RunPod headline run with `--intervention_modes ablate` pending.
3. **Re-enable steering controls** (2.3 — FIX LANDED 2026-05-28, commit `42b5837`: controls plumbed into the fast batched first-token path; new `random_feature_ablate` for the audit-3.1 ablate headline; `--controls_subsample_frac` cost knob; `--disable_controls` reframed smoke-test-only; production command updated). Diff-of-means + probe direction baselines already done in 5.5.
4. **Polarity-sign the bias metric** (4.3 — FIX LANDED 2026-05-27, commit `a03760f`): signed_stereotype_preference_delta + signed_stereotyped/nonstereotyped_delta computed in `enrich_results`; `effect_label`, `beneficial_score`, `harmful_score`, all rankings, and all plots now use the signed metric. Polarity-confounded `mean_stereotype_preference_delta` preserved alongside as a diagnostic.
5. **Validate the measurement locus** (1.1): compare final-token vs identity-span-pooled geometry; pick and justify one.
6. **Characterize baseline behavior** (1.2): answer-option mass and standard BBQ score for Llama-3.1-8B-Base in this format.

**Tier 2 — required for the numbers to be honest**

7. Held-out split for feature selection vs. effect estimation (2.5 — FIX LANDED 2026-05-28, all halves: BBQ winner's-curse via held-out selection/confirmation split `b5150ec`; identity-prompt prefilter `4481445`; identity-screen held-out feature selection via leave-one-family-out reconstruction `304ddb6`).
8. Null models for geometry probes and the shared-subspace spectrum (2.2).
9. Make held-out (cross-family/cross-template) AUC the headline; demote in-sample AUC (2.1 — FIX LANDED 2026-05-27/28: geometry + subspace `e15e62f`/`51aa571`; SAE-features held-out reconstruction `304ddb6`).
10. Fix answer scoring: score the letter A/B/C, or length-normalize `answer_logprob` (1.3 — FIX LANDED 2026-05-28 commit `2829417`: `--scoring_mode letter` is the new default; 2.4 — FIX LANDED 2026-05-28 commit `8ef171c`: `row_metrics` length-normalizes argmax under `answer_logprob`; both close).
11. Restrict headline steering to `exact` contrast mapping; stratify by mapping confidence (3.4 — FIX LANDED 2026-05-27, commit `56a5f7e`).
12. Audit every contrast/alias identity ID against the dataset; make missing-ID skips loud (4.1).
13. Reduce the inference grid: one test per feature, not per feature×alpha×position (2.6 — FIX LANDED 2026-05-28, commit `6e132d3`: feature-level `feature_inference` table at a single `--headline_alpha`, polarity pooled, FDR across features).
14. Drop `--smoke`; raise bootstrap/permutation budgets and per-cell minimums (2.7 — FIX LANDED 2026-05-28, commit `6e132d3`: defaults 10k bootstrap/permutation, `--min_examples_inference` floor before FDR, `--smoke` now warns).

**Tier 3 — correctness, clarity, strengthening**

15. Fix the reconstruction projection math (least-squares, not `BᵀB`) (5.1 — FIX LANDED 2026-05-27, commit `1a569c3`).
16. Verify intervention positions land in the intended prompt section (3.3 — FIX LANDED 2026-05-28, commit `afb3ee3`: section-aware token flags + section-explicit position names + `intervention_section` output column). **Pre-RunPod validation gap:** synthetic test only — real Llama tokenizer + real BBQ prompt format unverified. See the section 3.3 entry for the recommended audit before trusting numbers.
17. Tie steering magnitude to a meaningful scale (3.2 — FIX LANDED 2026-05-28, commit `22e8345`: per-feature `clamp`/`steer` amplitude scaled by each feature's own `feature_stats.csv` stat × a multiplier grid; `--feature_scale_stat p95` default; `ablate` unaffected). **Pre-RunPod validation gap:** synthetic only — run `scripts/audit_feature_scale.py` against the real triage pool + `feature_stats.csv` before any clamp/steer headline.
18. Decide intersectional BBQ handling — first-class or excluded, not flattened (4.2 — FIX LANDED 2026-05-28, commit `b189aef`: path (b) chosen; `--intersectional_handling drop` default; `is_intersectional` column added; first-class compound contrasts recorded as future work).
19. Reframe triage as pre-registered *selection*; validate the taxonomy if it is a contribution (5.2 — PARTIAL FIX LANDED 2026-05-27: firing-count entropy `7f2c302`, soft scoring head `235b5f5`, sensitivity sweep `f306869`, pre-registration doc landed; behavioral criterion + inter-rater validation deferred to RunPod).
20. Make representation use consistent (residualized vs raw) across the SAE analysis (5.4 — FIX LANDED 2026-05-27, commit `ebfdff7`: raw end-to-end).
21. Extract shared code into a common module with a validated contrast registry (5.10, 4.1).
22. Multi-layer SAE coverage (5.8); consider counterfactual/patching methods (5.6, 5.7).

**Framing note.** The repo currently runs two semi-independent investigations — identity *geometry* and BBQ *causal features* — joined only loosely (geometry's contrast list seeds the triage that seeds steering). A NeurIPS/ICLR paper needs one throughline. The strongest version: *templated prompts establish where/how identity is represented (geometry) → SAE features name interpretable components of that representation → feature-level interventions on BBQ show which of those components are causally implicated in bias.* For that arc to hold, the SAE feature interventions must be real feature interventions (3.1) and must beat a plain linear direction (5.5) — otherwise the SAE is decoration on a linear-probe result, and the paper should be reframed around directions instead.
