# Step 18a — `scripts/build_few_shot_pool.py`

**Stage:** 4 — BBQ steering (pre-prepare prerequisite)
**Runs after:** —
**Feeds into:** [Step 18 — `prepare_bbq_for_steering.py`](18_prepare_bbq_for_steering.md) via `--few_shot_pool`.

## Purpose
Build a stratified, seeded few-shot exemplar pool from BBQ so that Llama-3.1-8B-Base can be pulled onto MCQ task distribution before any steering. Addresses audit issue [1.2 — base model vs MCQ benchmark](../issues_and_opportunities.md). The script picks K=4 exemplars covering every (context_condition × question_polarity) cell, drawn from distinct categories, and writes a JSON the prepare script can consume.

## Inputs
- `data/bbq/data/*.jsonl` — the same BBQ source files [Step 18](18_prepare_bbq_for_steering.md) reads.

## Outputs
- `data/bbq/few_shot_pool.json` — committed artifact (gitignored by default; `.gitignore` has an exception for this path). Schema:
  - `seed`, `stratification`, `answer_format`, `k`
  - `exemplars: [...]` — each entry has `example_id`, `question_index`, `category`, `context_condition`, `question_polarity`, `source_file`, `context`, `question`, `ans0..ans2`, `label`, and a pre-rendered `exemplar_block` string in the exact format the prepared prompts use.

## Key implementation details
- **Stratification:** K=4, one exemplar per cell of `(context_condition ∈ {ambig, disambig}) × (question_polarity ∈ {neg, nonneg})`. Each cell's pool is shuffled with the seeded RNG, then the first row from a previously-unused category is picked (so all four exemplars come from distinct categories when possible).
- **Answer format:** `Answer: <LETTER>. <answer_text>` — uses the letter convention (forward-compatible with the audit's [1.3 letter-scoring fix](18_prepare_bbq_for_steering.md)) but also includes the answer text so it works under either scoring mode.
- **Determinism:** `--seed` (default 0) controls the RNG that shuffles each stratum. Same seed + same BBQ data → same exemplars.
- **Exclusion key:** `(source_file, example_id)` — the prepare script uses these tuples to drop the chosen exemplars from the steering test set so there is no contamination.

## Issues & Opportunities

This script does not own an audit issue — it is the *enabler* for the [1.2 — base model vs MCQ benchmark](../issues_and_opportunities.md) fix in [Step 18](18_prepare_bbq_for_steering.md). One indirect consideration:

### Forward-compatibility with 1.3 (letter scoring)

**Context:** Audit issue 1.3 recommends scoring answer letters (` A` / ` B` / ` C`) instead of the first token of each answer text in `run_bbq_sae_steering.py`. The pool format `Answer: <LETTER>. <answer_text>` is intentionally hybrid: it primes both letter-conditioning and text-conditioning, so the same pool works whether 1.3 has landed yet or not.

**If 1.3 changes the format convention later:** re-run this script with the same seed and the pool is reproduced byte-identically; if you want to change the answer format, edit `render_exemplar_block` and regenerate.

## Rebuild checklist
- [ ] Verify the committed pool at `data/bbq/few_shot_pool.json` covers all four (condition × polarity) cells. The script's stdout summary prints the four rows; check the categories are distinct.
- [ ] If you want a different exemplar set, re-run with a different `--seed` and inspect; commit the new JSON.
- [ ] If you want a different K (e.g., 5 with an extra axis-balancing row), edit `STRATA` and `select_pool`.
- [ ] If BBQ source data changes (new categories, edited rows), regenerate the pool and commit.

## Notes from the doc audit
- The pool selection is one-shot, not idempotent against `--max_examples` in [Step 18](18_prepare_bbq_for_steering.md). If you ever pass `--max_examples N` to prepare and `N` is small enough to exclude the exemplar rows, the `Excluded N few-shot exemplar rows from output` log line will count zero. The prepare script warns when the actual exclusion count diverges from the pool size.
- The seed=0 selection happens to draw two of four exemplars from race-related categories (`Race_x_SES`, `Race_x_gender`). Acceptable for demonstrating both stereotype-consistent (Omar) and stereotype-defying (Robert) correct disambig answers, but worth knowing if reviewers ask whether the few-shot prefix itself biases the model. A future "balanced-axis" mode could enforce no two exemplars from the same parent axis.
