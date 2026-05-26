# Step 4 — `scripts/extract_identity_activations.py`

**Stage:** 1 — Identity activations and SAE encodings
**Runs after:** [Step 1 — `create_dataset.py`](01_create_dataset.md), [Step 2 — `download_llama_3_1_8b.py`](02_download_llama_3_1_8b.md).
**Feeds into:** [Step 5 — `encode_identity_saes.py`](05_encode_identity_saes.md), [Step 6 — `validate_sae_hook_alignment.py`](06_validate_sae_hook_alignment.md), every Stage 2 geometry analysis, and (transitively) the SAE feature triage that drives Stage 4 BBQ steering.

## Purpose
Run Llama-3.1-8B forward over every prompt in `mi_identity_prompts.csv`, take `outputs.hidden_states` (the tuple of per-layer residual streams), and store the **final-non-padding-token** activation at every layer as a row-aligned memmapped `.npy`. This is the single load-bearing measurement of "identity in the residual stream" that the entire geometry pipeline (PCA, probes, contrast directions, shared-subspace SVD, SAE encoding) sits on top of.

## Inputs
- `data/mi_identity_prompts.csv` (must contain `prompt` and `prompt_id`; non-empty `prompt` cells are asserted).
- Llama model directory from [Step 2](02_download_llama_3_1_8b.md).

## Outputs
- `results/activations/llama-3.1-8b/identity_prompts_final_token/`
  - `layer_00.npy ... layer_32.npy` — `(n_prompts, hidden_dim)` float32 memmapped arrays. `layer_00` is the embedding output; `layer_k` (k ≥ 1) is post-block-k residual (HF `output_hidden_states` convention).
  - `metadata.csv` — row-aligned copy of the prompt CSV.
  - `checkpoint.json` — supports `--resume` after interruption.
  - `run_config.json` — model path, batch_size, max_length, dtype, hidden_dim, num_layers_saved, timestamp.

## Key implementation details
- Tokenizer is configured `padding_side="right"`, `pad_token = eos_token` if unset. Right-padding is the invariant the final-token index depends on: `final_idx = attention_mask.sum(dim=1) - 1`.
- Model dtype picked by `choose_torch_dtype`: bf16 on bf16-capable CUDA, else fp16 on CUDA, else fp32 CPU. The default RunPod run is bf16.
- Activations are cast to `float32` before writing to the `.npy` memmap (`final_hidden.detach().float().cpu().numpy()`) — so the *storage* dtype is fp32 but the *precision* of the underlying values is whatever the forward produced (bf16 on GPU).
- Per-batch: encode → forward → for each layer, index `(batch_arange, final_idx)` to pull the final token → write into the layer's memmap slice. Then `flush()` and update the checkpoint atomically. Resume reads `checkpoint["rows_written"]` and reopens memmaps in `r+`.
- `--max_length` defaults to 128; templated identity prompts are short so truncation is unlikely to bite, but it is silently allowed.
- All hidden states are kept, so memory cost scales with `(33 layers × n_prompts × hidden_dim × 4 bytes)` — at 12,567 prompts × 4096 dim × 33 layers that is about 6.6 GB on disk.

## Issues & Opportunities

### 1.1 [BLOCKER] — The "identity representation" is measured at the final token, which is almost always a period

**What's wrong:** Every template in `mi_identity_templates.csv` ends with a period (e.g. `A01 = "This person is {form}."`, `F03 = "{form}."`). After tokenization the final non-padding token is the sentence-final period in essentially every prompt. So `extract_final_token_activations` stores, per layer, the residual stream **at the period token**, not at the identity token. The implicit claim is that "the final token integrates the identity content of the prompt," but Llama-3.1-8B is a *base* LM — its final-token residual is optimized to predict the next token, not to summarize the sentence; no `[CLS]`-style aggregation objective exists.

**Why it matters:** The entire downstream geometry pipeline (PCA, contrast directions, shared-subspace SVD, family-stability cosines, SAE feature selectivity, decoder alignment, triage roles, the BBQ feature pool) is computed on these arrays. If period-token geometry is not faithful to identity geometry, every geometric and causal claim is downstream of an untested assumption a reviewer will challenge on first read.

**Targeted fix:**
- Add `--token_mode {final, identity_span_last, identity_span_mean}` to this script. The first is current behavior; the other two require locating the identity surface form in each prompt (regex against `form_used` column from `metadata.csv` plus offset mapping) and pooling residuals over those token positions.
- Run the geometry analysis on all three modes and compare contrast AUC, probe accuracy, and shared-subspace spectrum across them.
- Pick the locus that carries signal, justify it, and report the comparison itself as a finding.
- The companion `token_span` mode is already scaffolded but `NotImplementedError` in [Step 5 — `encode_identity_saes.py`](05_encode_identity_saes.md); both ends need to be filled in for the span-based SAE encoding to work.

### 1.5 [MINOR] — Activations are bf16-precision stored as float32

**What's wrong:** The model runs in bf16 on GPU, and `final_hidden.detach().float()` upcasts to fp32 *before* the memmap write. So the storage dtype is fp32 but the actual significant precision is bf16 (≈3 decimal digits). The fp32 size is paid on disk and in downstream RAM without any precision benefit.

**Why it matters:** Mean-difference contrast directions average the noise away, but per-prompt projections, per-prompt SAE top-k encodings, and individual cosines inherit bf16 noise. Reproducibility tables in the paper should state this; small numerical differences across runs/hardware will follow from the bf16 forward.

**Targeted fix:**
- Disclose in the reproducibility section: "Activations computed in bf16, stored as fp32." Add the dtype to `run_config.json` (already present) and reference it from the methods text.
- Optional: add `--store_dtype {fp32, fp16, bf16}` and use fp16 for storage to halve disk usage with no precision loss vs. the current setup.
- For the final headline run, if VRAM allows, run the forward in fp32 and store fp32 — eliminates the asymmetry entirely. Otherwise note it and move on.

## Rebuild checklist
- [ ] Implement `--token_mode` with `final`, `identity_span_last`, `identity_span_mean` options. Reuse the regex span-finding logic from `extract_token_level_sae_activations.py`.
- [ ] When running the new modes, change the output directory name to reflect the locus (e.g. `identity_prompts_identity_span_last/`) so artifacts are not confused with the period-token run.
- [ ] Run all three modes for at least layers `{0, 8, 16, 24, 32}` so geometry comparisons are tractable without re-running every layer.
- [ ] Plumb the chosen `token_mode` into [Step 5 — `encode_identity_saes.py`](05_encode_identity_saes.md) so the `token_span` `NotImplementedError` is actually filled in.
- [ ] Record the chosen final locus and the comparison results in the methods doc.
- [ ] Disclose bf16 forward / fp32 storage in `run_config.json` (already there) and propagate to the methods write-up.

## Notes from the doc audit
- The script asserts right-padding implicitly via `tokenizer.padding_side = "right"` but never re-asserts that the eventual `encoded["attention_mask"]` rows are right-padded. If a custom tokenizer subclass overrode that, the final-token index would silently be wrong. A defensive check (e.g. `assert attention_mask[:, 0].all()` for every batch) would catch a future regression.
- `outputs.hidden_states` is requested twice: once via `from_pretrained(..., output_hidden_states=True)` and again on the forward call. Harmless but redundant.
- The `--max_length 128` default is fine for the current template corpus but should be re-checked if longer templates are added in the future.
