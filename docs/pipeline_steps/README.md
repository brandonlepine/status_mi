# Pipeline Steps — Index

A sequence-ordered guide to every script in the `status_mi` repository, with each step's issues and opportunities integrated inline so you can make targeted adjustments and rebuild.

- **Companion docs:**
  - [`../conceptual_workflow.md`](../conceptual_workflow.md) — descriptive pipeline overview (what each script does, in execution order).
  - [`../issues_and_opportunities.md`](../issues_and_opportunities.md) — full audit with severity tags (`[BLOCKER]`, `[MAJOR]`, `[MINOR]`, `[Opportunity]`) and the Section 7 priority punch list.
  - [`../addressing_issues_and_opportunities.md`](../addressing_issues_and_opportunities.md) — fix-tracking doc (currently a stub).
  - [`../bbq_steering_pipeline.md`](../bbq_steering_pipeline.md) — operational doc with the long-run commands for Stage 4.

Every per-step file uses the same template: **Purpose → Inputs → Outputs → Key implementation details → Issues & Opportunities (with severity tags) → Rebuild checklist → Notes from the doc audit** (the last section flags issues that surfaced during this audit but are NOT in `issues_and_opportunities.md`).

---

## Execution sequence at a glance

```
Stage 0  Prompt + identity datasets
   01  create_dataset.py                            data/mi_identity_prompts.csv

Stage 1  Identity activations + SAE encoding
   02  download_llama_3_1_8b.py                     models/llama-3.1-8b/
   03  download_openmoss_saes.py                    saes/openmoss/...
   04  extract_identity_activations.py              results/activations/.../layer_XX.npy
   05  encode_identity_saes.py                      results/sae_identity/.../layer_XX/...
   06  validate_sae_hook_alignment.py               hook_alignment_validation.{json,csv}

Stage 2  Identity-geometry analyses + plotting
   07  analyze_identity_geometry.py                 results/geometry/.../{pca,probes,contrasts,...}
   08  analyze_identity_geometry_diagnostics.py     diagnostics/ (η², residualization, contrasts)
   09  analyze_shared_social_subspace.py            shared_subspace_decomposition/
   10  plot_identity_geometry.py                    figures over Step 07 CSVs
   11  plot_identity_directional_visualizations.py  re-computes directions + figures
   12  plot_identity_directional_followups.py       paper-ready summary panel

Stage 3  SAE feature analysis, cards, triage
   13  analyze_identity_sae_features.py             analysis/feature_*.csv, decoder alignment, reconstructions
   14  extract_token_level_sae_activations.py       feature_cards/token_level/...
   15  build_sae_feature_cards.py                   feature_cards/layer_XX/*.html
   16  plot_identity_sae_features.py                feature-effect summary plots
   17  triage_sae_identity_features.py              triage/intervention_candidate_features_triaged.csv  ← single most consequential output

Stage 4  BBQ steering + causal analysis
   18a build_few_shot_pool.py     (optional prereq)  data/bbq/few_shot_pool.json
   18  prepare_bbq_for_steering.py                  prepared/bbq_prepared_examples.{csv,parquet}
   18b diagnose_bbq_baseline.py   (diagnostic)      prepared/baseline_diagnostics.{json,csv}
   19  extract_bbq_token_level_sae_activations.py   token_level_sae/token_activations/...
   20  run_bbq_sae_steering.py                      steering/results_parts/*.parquet   ← causal-intervention engine
   21  build_bbq_sae_feature_cards.py               feature_cards_filtered/...
   22  analyze_bbq_steering_results.py              analysis/ (legacy aggregator)
   23  analyze_bbq_feature_level_causal_effects.py  feature_level_causal_analysis/...
```

---

## Per-step docs (with issue severity)

Severity tags are aggregated per script from `issues_and_opportunities.md`. **B** = `[BLOCKER]`, **M** = `[MAJOR]`, **m** = `[MINOR]`, **O** = `[Opportunity]`.

| # | Script | Stage | Severities | Doc |
| --- | --- | --- | --- | --- |
| 01 | `create_dataset.py` | 0 | m | [01_create_dataset.md](01_create_dataset.md) |
| 02 | `download_llama_3_1_8b.py` | 1 | M (1.2 lives across 02/18/20) | [02_download_llama_3_1_8b.md](02_download_llama_3_1_8b.md) |
| 03 | `download_openmoss_saes.py` | 1 | — (1.4 dependency, FIX LANDED) | [03_download_openmoss_saes.md](03_download_openmoss_saes.md) |
| 04 | `extract_identity_activations.py` | 1 | **B** (1.1 — PARTIAL FIX LANDED) · m (1.5) | [04_extract_identity_activations.md](04_extract_identity_activations.md) |
| 05 | `encode_identity_saes.py` | 1 | **B** (1.4 — FIX LANDED) · m (4.6) · **B** (1.1 — FIX LANDED) | [05_encode_identity_saes.md](05_encode_identity_saes.md) |
| 06 | `validate_sae_hook_alignment.py` | 1 | **B** (1.4 — FIX LANDED with recon check) | [06_validate_sae_hook_alignment.md](06_validate_sae_hook_alignment.md) |
| 07 | `analyze_identity_geometry.py` | 2 | **B** (2.2 — PROBE NULL LANDED) · M (2.1 — FIX LANDED, 4.1 — FIX LANDED) · m (2.8 — VERIFIER LANDED, 5.9 — FIX LANDED, 5.10 — FIX LANDED) | [07_analyze_identity_geometry.md](07_analyze_identity_geometry.md) |
| 08 | `analyze_identity_geometry_diagnostics.py` | 2 | **B** (2.2 — PROBE NULL LANDED) · M (2.1 — FIX LANDED, 4.1 — FIX LANDED) · m (2.8 — VERIFIER LANDED, 5.9 — FIX LANDED, 5.10 — FIX LANDED) — strengths to keep | [08_analyze_identity_geometry_diagnostics.md](08_analyze_identity_geometry_diagnostics.md) |
| 09 | `analyze_shared_social_subspace.py` | 2 | **B** (2.2 — FIX LANDED) · M (2.1 — FIX LANDED, 4.1 — FIX LANDED) · m (5.10 — FIX LANDED) | [09_analyze_shared_social_subspace.md](09_analyze_shared_social_subspace.md) |
| 10 | `plot_identity_geometry.py` | 2 | m (2.1 visualization, 5.10 — FIX LANDED) | [10_plot_identity_geometry.md](10_plot_identity_geometry.md) |
| 11 | `plot_identity_directional_visualizations.py` | 2 | m (2.1 visualization — FIX LANDED, 5.10 — FIX LANDED) | [11_plot_identity_directional_visualizations.md](11_plot_identity_directional_visualizations.md) |
| 12 | `plot_identity_directional_followups.py` | 2 | m (2.1 visualization — PARTIAL FIX LANDED, 4.1 — FIX LANDED, 5.10 — FIX LANDED) | [12_plot_identity_directional_followups.md](12_plot_identity_directional_followups.md) |
| 13 | `analyze_identity_sae_features.py` | 3 | M (2.1 — PARTIAL FIX LANDED, 2.5 — PARTIAL FIX LANDED, 4.1 — FIX LANDED, 5.1 — FIX LANDED) · m (5.3 — FIX LANDED, 5.4, 5.10 — FIX LANDED) · enabler (3.1 — FIX LANDED) | [13_analyze_identity_sae_features.md](13_analyze_identity_sae_features.md) |
| 14 | `extract_token_level_sae_activations.py` | 3 | m (4.6) · enabler (1.1) | [14_extract_token_level_sae_activations.md](14_extract_token_level_sae_activations.md) |
| 15 | `build_sae_feature_cards.py` | 3 | m (logit-lens caveat) | [15_build_sae_feature_cards.md](15_build_sae_feature_cards.md) |
| 16 | `plot_identity_sae_features.py` | 3 | m (5.10 — FIX LANDED) | [16_plot_identity_sae_features.md](16_plot_identity_sae_features.md) |
| 17 | `triage_sae_identity_features.py` | 3 | M (5.2) | [17_triage_sae_identity_features.md](17_triage_sae_identity_features.md) |
| 18a | `build_few_shot_pool.py` (prereq) | 4 | enabler (1.2 partial fix) | [18a_build_few_shot_pool.md](18a_build_few_shot_pool.md) |
| 18 | `prepare_bbq_for_steering.py` | 4 | M (1.2 — partial fix, 3.4, 4.1, 4.2) · m (4.4) | [18_prepare_bbq_for_steering.md](18_prepare_bbq_for_steering.md) |
| 18b | `diagnose_bbq_baseline.py` (diagnostic) | 4 | enabler (1.2, also touches 1.3, 2.4, 4.3) | [18b_diagnose_bbq_baseline.md](18b_diagnose_bbq_baseline.md) |
| 19 | `extract_bbq_token_level_sae_activations.py` | 4 | M (3.3) · m (4.6) | [19_extract_bbq_token_level_sae_activations.md](19_extract_bbq_token_level_sae_activations.md) |
| 20 | `run_bbq_sae_steering.py` | 4 | **B** (2.3, 3.1 — FIX LANDED) · M (1.3, 2.4, 3.2, 3.3, 3.4) · m (3.5) | [20_run_bbq_sae_steering.md](20_run_bbq_sae_steering.md) |
| 21 | `build_bbq_sae_feature_cards.py` | 4 | m (downstream inheritance) | [21_build_bbq_sae_feature_cards.md](21_build_bbq_sae_feature_cards.md) |
| 22 | `analyze_bbq_steering_results.py` | 4 | m (5.10) — marked legacy | [22_analyze_bbq_steering_results.md](22_analyze_bbq_steering_results.md) |
| 23 | `analyze_bbq_feature_level_causal_effects.py` | 4 | **B** (4.3) · M (2.4, 2.5, 2.6) · m (2.7) — strengths to keep | [23_analyze_bbq_feature_level_causal_effects.md](23_analyze_bbq_feature_level_causal_effects.md) |

---

## Tier-1 punch list (the six things that gate every current number)

From Section 7 of `issues_and_opportunities.md`. These are the fixes to make before trusting any output the pipeline produced.

1. **Fix the SAE encoder (1.4 — FIX LANDED 2026-05-26, RunPod re-encode + recon-check verification pending).** Three commits closed out the encoder convention fix:
   - `1ed1422` — [Step 3](03_download_openmoss_saes.md) now selects files by explicit `L<layer>R-<width>x` marker, requires `hyperparameters.json` per layer, and pins `--revision` to an absolute commit SHA.
   - `4b8851a` — [Step 5](05_encode_identity_saes.md) reads `hyperparameters.json`, validates `act_fn=="jumprelu"`, `apply_decoder_bias_to_pre_encoder is False`, `norm_activation=="dataset-wise"`, computes per-layer `scale_in` / `scale_out` / `theta`, and implements `encode_full` / `decode_full` with the corrected formula.
   - `efc098c` — [Step 6](06_validate_sae_hook_alignment.md) adds the encode→decode regression test (FVU / cosine / mean L0) and fails the validator above `--reconstruction_fvu_threshold` (default 0.15).
   
   **Remaining:** re-download SAEs on RunPod under the new selector, re-encode every layer (deleting prior `feature_*.npy` / `feature_stats.csv` / downstream CSVs), run the validator, and confirm `reconstruction_fvu <= 0.15` before consuming any new artifact downstream. Every Stage-3 and Stage-4 analysis must be rerun against the new encodings.
2. **Fix the feature intervention (3.1 — FIX LANDED 2026-05-27, RunPod headline run pending).** Two commits closed out the wiring:
   - `11d4a4d` — canonical torch primitives (`ablate_features`, `clamp_features`, `steer_features`, `patched_residual_with_intervention`) live in [Step 5](05_encode_identity_saes.md) alongside `encode_full` / `decode_full`. The wrapper does encode → modify-latent-f → decode → patch in normalized space, then un-scales by `scale_out` so the residual-stream delta is on the model's natural scale; SAE reconstruction error cancels in the delta.
   - `84c87b5` — [Step 20](20_run_bbq_sae_steering.md) registers `install_feature_intervention_hook` / `install_batched_feature_intervention_hook` and dispatches via `--intervention_modes` ∈ `{add_vector, ablate_projection, ablate, clamp, steer}`, defaulting to `ablate`. Five synthetic hook integration tests pass.

   **Remaining:** run `--intervention_modes ablate` on RunPod against the (re-encoded under 1.4) feature pool, then compare against a parallel `add_vector` run to quantify how much of the prior headline effect was direction vs. feature.
3. **Re-enable steering controls** (2.3) — Step [20](20_run_bbq_sae_steering.md) ships with `sign_flip`, `random_direction_norm_matched`, `random_feature_matched`, but the production command uses `--disable_controls`. Add the difference-of-means direction as a parallel control (per 5.5).
4. **Polarity-sign the bias metric** (4.3) — Step [23](23_analyze_bbq_feature_level_causal_effects.md)'s `stereotype_preference_delta` ignores `question_polarity`, so `effect_label`, `beneficial_score`, and the final candidates table are polarity-confounded.
5. **Validate the measurement locus (1.1 — PARTIAL FIX LANDED 2026-05-27).** Step [04](04_extract_identity_activations.md) now supports `--token_mode {final_token, identity_span_last, identity_span_mean}` (commit `ca1224e`), and Step [05](05_encode_identity_saes.md) mirrors the choices and drops the `NotImplementedError` (commit `6bd78fc`). Span pre-pass writes `span_locations.csv` per run for the audit trail. **Remaining:** run all three modes on RunPod for layers `{0, 8, 16, 24, 32}`, re-run Stage 2 geometry on each, compare contrast AUC / probe accuracy / shared-subspace spectrum across loci, and record the chosen locus + comparison in the methods writeup. Decision rule and invocation order are in [Step 4 — 1.1](04_extract_identity_activations.md#11-blocker--measurement-locus-partial-fix-landed-2026-05-27).
6. **Characterize baseline behavior (1.2 — PARTIAL FIX LANDED).** Few-shot pool + prepare integration + a dedicated diagnostic ([Step 18a](18a_build_few_shot_pool.md) / [Step 18](18_prepare_bbq_for_steering.md) `--few_shot_pool` / [Step 18b](18b_diagnose_bbq_baseline.md)) all landed 2026-05-26. **Remaining work:** run both zero-shot and few-shot variants on RunPod, diff `baseline_diagnostics.json` (decision table in [Step 18b](18b_diagnose_bbq_baseline.md#suggested-invocation-order-zero-shot-vs-few-shot-side-by-side)), and pick a prompt mode for steering. Until the diagnostic is run, the precondition is unmet — code is in place, numbers are not.

Tier 2 and Tier 3 are in `issues_and_opportunities.md` Section 7. The per-step docs reference them inline where they apply.

---

## How to use this index

- **Want the big picture?** Read [`../conceptual_workflow.md`](../conceptual_workflow.md) for the descriptive pipeline overview, then [`../issues_and_opportunities.md`](../issues_and_opportunities.md) for the audit.
- **About to rebuild?** Work down this table: open each step's `.md`, work the **Rebuild checklist** at the bottom, then tick boxes as you go.
- **Tracking new findings?** The `Notes from the doc audit` section in each file lists issues observed during this audit that are not yet in `issues_and_opportunities.md`. Pull them into the main audit doc as you triage them.
- **Cross-references** between steps use relative links (e.g. `[Step 13](13_analyze_identity_sae_features.md)`).

---

## New issues surfaced by this audit

Across the per-step files there are ~25 small issues observed in code that are not in the main audit. They are listed inline in each file's `Notes from the doc audit` section, but the most load-bearing ones:

- **Step 5** — `compute_feature_stats` derives `p95`/`p99` only from top-64-surviving activations, so they are not true per-feature percentiles; with the 3.1 fix now wiring `--intervention_modes clamp`, anyone using these percentiles as the clamp target should be aware the scale is silently biased high. The `ablate` mode is unaffected (clamps to 0).
- **Step 5** — `choose_matrix` includes `"gate"` in its encoder-keyword list, which would mis-select a gated-SAE `w_gate` tensor.
- **Step 6** — `position_marker_is_R = parsed["sae_position"] in {None, "R"}` treats parse-failed filenames as a pass, masking wrong-checkpoint mounts.
- **Step 13** — `intervention_candidate_features.csv` writes only the top-N-per-rank per contrast; features mid-ranked everywhere never appear, and the triage in Step 17 falls back to `cosine = 0` for them.
- **Step 17** — first-match-wins cascade: a feature with `template_artifact_score = 0.51` is irrecoverable even with strong identity-span localization. A score-matrix would be more robust than the cascade.
- **Step 18** — `mapped_contrast_confidence == "alias"` is documented but never produced (`map_contrast` only ever returns `exact` / `fallback_axis` / `unmapped`); the Step 20 keep-list contains `alias`, which therefore matches no rows.
- **Step 18** — the `age` axis branch in `identity_axis` produces `axis_mapped == "age"` rows that no kept SAE feature set can match (no `age` axis exists in `bbq_identity_normalized_forms.csv`), so they are silently dropped at Step 20.
- **Step 20** — `--scoring_mode first_token --disable_controls` takes a separate batched code path with **no control code at all**; even with controls re-enabled (2.3 fix), the user must also drop `first_token` mode to get them.
- **Step 20** — `find_all_spans` does no word-boundary matching, so an identity surface form of `"man"` matches inside `"woman"`.
- **Step 23** — see the Notes section for additional polarity/multiplicity nuances.

See the per-step `Notes from the doc audit` for the full list. None of these change the Tier-1 ordering above.
