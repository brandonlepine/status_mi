# Step 2 — `scripts/download_llama_3_1_8b.py`

**Stage:** 1 — Identity activations and SAE encodings (one-time setup)
**Runs after:** — (entry point; only requires HF auth)
**Feeds into:** [Step 4 — `extract_identity_activations.py`](04_extract_identity_activations.md), [Step 6 — `validate_sae_hook_alignment.py`](06_validate_sae_hook_alignment.md), and every downstream script that loads the model (BBQ steering, token-level extraction, feature cards' logit lens).

## Purpose
Snapshot-download `meta-llama/Llama-3.1-8B` (the **base** model) from Hugging Face into `/workspace/status_mi/models/llama-3.1-8b/`. This is a thin wrapper around `huggingface_hub.snapshot_download` with a fixed default destination, supporting `--revision`, `--token`, and `--local_files_only`.

The base model is the correct choice for SAE compatibility — OpenMOSS LlamaScope SAEs are trained on `Llama3_1-8B-Base`, not the Instruct variant. That choice is load-bearing for [Step 5](05_encode_identity_saes.md) and the Stage 4 BBQ steering pipeline.

## Inputs
- HF auth (env `HF_TOKEN` or `huggingface-cli login`).
- Optional `--revision` to pin a commit hash.

## Outputs
- `/workspace/status_mi/models/llama-3.1-8b/` containing the full HF snapshot (`config.json`, tokenizer files, sharded weights).

## Key implementation details
- `local_dir_use_symlinks=False` — files are copied into the destination, not symlinked back into the HF cache. Disk usage doubles vs. the cached default.
- `resume_download=True` — interrupted runs continue.
- No verification beyond what `snapshot_download` does internally (no SHA check, no smoke forward pass).

## Issues & Opportunities

### 1.2 [MAJOR] — Base model vs. a multiple-choice QA benchmark

**What's wrong:** The base Llama-3.1-8B is correct for SAE compatibility (LlamaScope SAEs were trained on it), but the downstream BBQ pipeline ([Stage 4](../conceptual_workflow.md)) wraps each item as a zero-shot multiple-choice prompt ending in `"Answer:"`. Base LMs are weak and often off-distribution for that format: they may put 1–2% total mass on the three answer options and 98% on continuation text. The pipeline assumes the model places meaningful, well-calibrated probability mass on the answer tokens — that assumption is untested for this base model in this prompt format.

**Why it matters:** Every BBQ logprob delta downstream is computed in whatever regime the base model lands in. If that regime is degenerate (almost all mass off the answer options), "bias" as measured by logprob shifts on the answer set is barely defined, and steering effect sizes mean little even if controls are clean.

**Targeted fix:** Before publishing any steering result, characterize baseline behavior in `scripts/prepare_bbq_for_steering.py` or a new diagnostic:
- Total probability mass on the three answer-option tokens / answer texts.
- Standard BBQ accuracy and bias score for Llama-3.1-8B-Base in this exact prompt format.
- Argmax-over-options vs. greedy-continuation agreement rate.

Then either (a) add a few-shot prompt prefix (3–5 BBQ exemplars) to pull the base model onto distribution before steering, or (b) accept the degenerate regime and frame results around logprob *margins* (which remain defined). Do not paper over it.

This script is not where the fix lives, but it is where the choice of base model is locked in and where a `README` note belongs explaining the tradeoff (base-for-SAEs vs. instruct-for-BBQ-format).

## Rebuild checklist
- [ ] Pin `--revision` to a specific commit hash and record it in `run_config.json`-style provenance — base-model snapshots can shift if Meta re-uploads.
- [ ] Add a post-download smoke check: load the model and tokenizer, run one forward on a known string, log perplexity and a sample completion.
- [ ] Document in the project `README` why the base model (not Instruct) is used despite the BBQ MCQ format, and link to the baseline-behavior diagnostic from 1.2.
- [ ] After fixing 1.2 downstream, link this step's docs to the resulting baseline-mass report.

## Notes from the doc audit
- The default `output_dir` is hard-coded to the RunPod path `/workspace/status_mi/models/llama-3.1-8b`; local Mac runs must override `--output_dir`. Not a bug, but worth flagging in the project setup notes.
