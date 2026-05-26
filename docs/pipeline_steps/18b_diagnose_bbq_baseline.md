# Step 18b — `scripts/diagnose_bbq_baseline.py`

**Stage:** 4 — BBQ steering (post-prepare diagnostic)
**Runs after:** [Step 18 — `prepare_bbq_for_steering.py`](18_prepare_bbq_for_steering.md), [Step 2 — `download_llama_3_1_8b.py`](02_download_llama_3_1_8b.md).
**Feeds into:** No downstream script consumes its outputs — it is a precondition diagnostic that gates whether the steering claim is well-defined. The companion to the audit's [1.2 — base model vs MCQ benchmark](../issues_and_opportunities.md) fix in [Step 18a](18a_build_few_shot_pool.md).

## Purpose
Characterize how Llama-3.1-8B-Base behaves on the prepared BBQ prompts *before* any steering. The audit calls out three preconditions ([1.2](../issues_and_opportunities.md)) that must be reported in the paper, because if the base model puts ~no probability mass on the answer options the entire "bias steering" framing degenerates. This script measures all three.

## Inputs
- `prepared/bbq_prepared_examples.parquet` from [Step 18](18_prepare_bbq_for_steering.md) (zero-shot or few-shot — same code path for both).
- Llama-3.1-8B-Base from [Step 2](02_download_llama_3_1_8b.md).

## Outputs (under `--output_dir`)
- `baseline_diagnostics.json` — aggregate report with overall, per-`context_condition`, per-`axis_mapped` summaries plus polarity-signed bias scores.
- `baseline_diagnostics_per_example.csv` — one row per BBQ example with per-letter logprobs, per-text-first-token logprobs, the greedy next token, and derived flags (`correct`, `picked_unknown`, `bias_consistency`).
- `baseline_diagnostics_config.json` — invocation snapshot.
- `logs/diagnose_bbq_baseline.log`.

## Key implementation details
- **Final-token logprobs.** For each prompt, a single forward pass returns `log_softmax(logits[:, final_idx, :])`. The `final_idx` is `attention_mask.sum(dim=1) - 1` under right-padding (same convention as [Step 4 — `extract_identity_activations.py`](04_extract_identity_activations.md)).
- **Two mass measurements:**
  - `mass_letters` = `exp(logprob[' A']) + exp(logprob[' B']) + exp(logprob[' C'])` — relevant under the [1.3 letter-scoring fix](18_prepare_bbq_for_steering.md).
  - `mass_first_token_text` = sum of first-token logprobs of ` {ans0}` / ` {ans1}` / ` {ans2}` — what `run_bbq_sae_steering.py:score_first_token` currently scores.
- **Polarity-signed bias score (BBQ paper).** Per-example `bias_consistency` is +1 / -1 / 0:
  - `polarity=neg` and predicted=stereotyped → +1 (bias-consistent)
  - `polarity=nonneg` and predicted=stereotyped → −1 (anti-bias)
  - predicted=unknown → 0 (excluded from the denominator)
  - Stratum-level: `P_bias = mean(bias_consistency == 1) over non-unknown predictions`; `s_amb = (1 − accuracy) · (2·P_bias − 1)`; `s_dis = 2·P_bias − 1`.
- **Greedy-vs-argmax agreement.** A single greedy next-token id is taken from the same final logits. The flag `agree_letter_argmax_vs_greedy` is true iff the greedy token is one of `{ A, B, C}` AND the matching letter index equals the letter-logprob argmax. This is the audit's "argmax-over-options vs greedy continuation agreement rate."
- **`--dry_run`** skips model loading and writes a schema-correct but zero-filled output so the data flow (parquet read, per-example schema, output paths) can be validated locally without the 16GB model.

## Suggested invocation order (zero-shot vs few-shot side-by-side)

The diagnostic is only meaningful when run on both prepared variants and compared. Suggested order on RunPod:

```bash
# (1) Zero-shot baseline
python scripts/prepare_bbq_for_steering.py \
    --output_dir /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared_zeroshot

python scripts/diagnose_bbq_baseline.py \
    --prepared_parquet /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared_zeroshot/bbq_prepared_examples.parquet \
    --output_dir /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared_zeroshot \
    --output_stem baseline_diagnostics

# (2) Few-shot baseline (same model, different prompts)
python scripts/prepare_bbq_for_steering.py \
    --output_dir /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared_fewshot \
    --few_shot_pool /workspace/status_mi/data/bbq/few_shot_pool.json

python scripts/diagnose_bbq_baseline.py \
    --prepared_parquet /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared_fewshot/bbq_prepared_examples.parquet \
    --output_dir /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared_fewshot \
    --output_stem baseline_diagnostics
```

Then diff the two `baseline_diagnostics.json` files. The decision rule:

| Metric | Zero-shot | Few-shot | Implication |
| --- | --- | --- | --- |
| `mean_mass_on_letters` | low (e.g., < 0.1) | meaningfully higher | Few-shot pulled the model onto distribution. Use few-shot for steering. |
| `accuracy_letter_argmax` | near chance (1/3) | meaningfully above chance | Letter scoring is well-defined under few-shot. |
| `frac_greedy_matches_a_letter` | low | high | The model would naturally emit a letter — argmax-over-letters is meaningful. |
| `bias_scores.{ambig,disambig}.bias_score` | small magnitude (numerical noise) | clear signed value | Bias is measurable; steering can move it. |

If few-shot does NOT raise `mean_mass_on_letters` substantially, the honest interpretation is that the base model is too off-distribution for letter-scored MCQ. The audit's three options remain: (a) frame steering results around logprob *margins* rather than argmax accuracy; (b) try larger K or a different exemplar selection; (c) reconsider scope.

## Issues & Opportunities

### 1.2 [MAJOR] — Base model vs MCQ benchmark (PARTIAL FIX LANDED)

**Status:** Code landed 2026-05-26 in two commits ([few-shot pool + prepare integration](18a_build_few_shot_pool.md); this diagnostic). What remains is to run both invocations on RunPod and decide based on the diff. See the suggested order above.

**What this script measures:** all three audit-required diagnostics — total mass on options, BBQ accuracy + bias score, argmax-vs-greedy agreement.

**What it does NOT decide:** the diagnostic is descriptive. It tells you whether the prompt format produces well-calibrated probabilities; it does not pick zero-shot vs few-shot for you. That decision belongs in the methods writeup informed by the numbers.

### Connections to other audit items

- **[1.3](../issues_and_opportunities.md)** — The diagnostic measures both letter-mass and text-first-token-mass so the 1.3 letter-scoring fix can be justified empirically (if mass-on-letters > mass-on-text-first-tokens, the case is straightforward).
- **[2.4](../issues_and_opportunities.md)** — The diagnostic reports raw `accuracy_letter_argmax` (length-independent) rather than the length-biased `answer_logprob` accuracy. Length-normalized answer_logprob accuracy could be added if desired.
- **[4.3](../issues_and_opportunities.md)** — The polarity-signed `bias_consistency` here is the same convention 4.3 calls for in `analyze_bbq_feature_level_causal_effects.py`. The diagnostic is a chance to validate the sign convention on a known stratum before propagating into the causal analyzer.

## Rebuild checklist
- [ ] Run zero-shot baseline on RunPod (full BBQ, default batch size).
- [ ] Run few-shot baseline on RunPod with `--few_shot_pool data/bbq/few_shot_pool.json`.
- [ ] Diff the two `baseline_diagnostics.json` files. Decide which prompt mode to use for steering.
- [ ] Record the decision and the headline numbers (mass-on-options, accuracy, bias score) in the methods writeup as the "baseline precondition" section the audit ([1.2](../issues_and_opportunities.md)) requires.
- [ ] If the chosen prompt mode is zero-shot, re-prepare without `--few_shot_pool` for the steering runs. If few-shot, re-prepare with the flag.
- [ ] Re-run [Step 19](19_extract_bbq_token_level_sae_activations.md) and [Step 20](20_run_bbq_sae_steering.md) against the chosen prepared parquet.

## Notes from the doc audit
- The per-example CSV is large (~58k rows on full BBQ). Sufficient for a one-shot diagnostic; if it is run repeatedly during iteration on the few-shot pool, consider switching to parquet output to keep load times down.
- The diagnostic does not currently measure full-answer-text logprob (would require K=3 extra forward passes per example). If 1.3 is fixed to letter-scoring this is moot; if `answer_logprob` is kept as a confirmatory mode under 2.4, this script is the right place to add it.
- The score depends on the same `final_idx = attention_mask.sum(dim=1) - 1` invariant that the geometry pipeline depends on. The audit's [1.1](../issues_and_opportunities.md) "final-token vs identity-span" question does NOT apply here: BBQ's final token IS the natural scoring position (after `"Answer:"`).
