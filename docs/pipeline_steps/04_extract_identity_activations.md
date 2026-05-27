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

### 1.1 [BLOCKER] — Measurement locus (PARTIAL FIX LANDED 2026-05-27)

**Status:** Code landed in commits `ca1224e` (this script) and `6bd78fc` ([Step 5](05_encode_identity_saes.md)). The three-mode comparison run on RunPod is what remains.

**What landed:**
- New `--token_mode {final_token, identity_span_last, identity_span_mean}` flag (default `final_token` preserves legacy behavior bit-for-bit). The default output directory is now derived from the mode (`identity_prompts_{token_mode}`), so the three runs land in sibling directories and cannot overwrite each other.
- `find_identity_span(prompt, form)` (lifted from `extract_token_level_sae_activations.py`) plus `span_token_indices(...)` convert the regex char-match to token positions via the tokenizer's offset_mapping. Both scripts now locate spans identically.
- New `precompute_span_locations` pre-pass runs CPU-only before the model loads. It tokenizes every prompt with `return_offsets_mapping=True`, validates that every prompt's `form_used` is locatable AND survives `--max_length` truncation, then writes a sidecar `span_locations.csv` (`prompt_id, form_used, span_status, span_start_char, span_end_char, span_token_first, span_token_last, span_n_tokens, span_token_indices`). Failures raise loudly so a long GPU run cannot be started against broken inputs.
- `select_layer_activation` reduces per-layer hidden states (B, T, D) → (B, D) under the chosen mode:
  - `final_token`: `hidden[b, attention_mask.sum-1, :]` (legacy gather, byte-identical).
  - `identity_span_last`: `hidden[b, span_token_last, :]`.
  - `identity_span_mean`: `sum(hidden * span_mask) / span_mask.sum`, computed once per batch and reused across layers.
- `run_config.json` records `token_mode`; [Step 5](05_encode_identity_saes.md) widens its `--activation_mode` choices to mirror and writes the label into its own run config too.

**Why it matters (original audit):** Every template in `mi_identity_templates.csv` ends with a period (e.g. `A01 = "This person is {form}."`, `F03 = "{form}."`). After tokenization the final non-padding token is the sentence-final period in essentially every prompt. So the prior loader stored, per layer, the residual stream at the period token, not at the identity token. The entire downstream geometry pipeline (PCA, contrast directions, shared-subspace SVD, family-stability cosines, SAE feature selectivity, decoder alignment, triage roles, the BBQ feature pool) was computed on these arrays. If period-token geometry is not faithful to identity geometry, every geometric and causal claim is downstream of an untested assumption a reviewer will challenge on first read.

**Verifying span-finding on the production CSV:**

`scripts/audit_identity_spans.py` runs four integrity checks against any prompt CSV (defaults to `data/mi_identity_prompts.csv`) and exits non-zero on any failure. Pre-run on 2026-05-27 against the current 12,567-prompt dataset:

| Check | Result |
| --- | --- |
| Span found (status != `failed`) | 12,567 / 12,567 PASS |
| Matched text equals form (case-insensitive) | 12,567 / 12,567 PASS |
| No ambiguity (form does not appear multiple times in prompt) | 12,567 / 12,567 PASS |
| Clean word boundary (no substring overlap like `man` inside `woman`) | 12,567 / 12,567 PASS |

```bash
python scripts/audit_identity_spans.py  # exits 0 on full pass, 1 on any failure
```

Rerun this whenever `data/mi_identity_prompts.csv` regenerates (e.g., after a template/identity edit), or whenever `find_identity_span` is touched.

**Remaining work (RunPod):**
- Run all three modes for at least layers `{0, 8, 16, 24, 32}`:
  ```bash
  python scripts/extract_identity_activations.py --token_mode final_token        # legacy
  python scripts/extract_identity_activations.py --token_mode identity_span_last
  python scripts/extract_identity_activations.py --token_mode identity_span_mean
  ```
- Re-encode each through [Step 5](05_encode_identity_saes.md) with the matching `--activation_dir` and `--activation_mode`.
- Run Stage 2 (geometry) on each and compare contrast AUC, probe accuracy, and shared-subspace spectrum across modes.
- Pick the locus that carries signal, justify it, and report the comparison itself as a finding in the methods writeup.

### 1.5 [MINOR] — Activations are bf16-precision stored as float32 (PARTIAL FIX LANDED 2026-05-27)

**What landed:**
- New `--store_dtype {fp32, fp16}` flag. `fp32` is the default (zero behavior change). `fp16` halves disk vs. fp32 but may lose precision on Llama's outlier "rogue" residual dimensions; the help text documents the tradeoff.
- `run_config.json` now records `forward_dtype` and `storage_dtype` separately so the asymmetry is explicit. The legacy `dtype` field is preserved for any downstream readers that already consume it.

**Remaining work:** flip the default to `bf16` storage once the necessary multi-script coordination is in. Tracked in [GH issue #1](https://github.com/brandonlepine/status_mi/issues/1) — that issue covers:

1. Adding `bf16` as a third `--store_dtype` choice via a `uint16`-view save format (numpy has no native bf16 dtype).
2. Coordinating [Step 5](05_encode_identity_saes.md) to decode the bf16-via-uint16 format.
3. Flipping the default to `bf16` once the round-trip is verified on RunPod.

**Original audit:** The model runs in bf16 on GPU, and `final_hidden.detach().float()` upcasts to fp32 before the memmap write. So the storage dtype was fp32 but the actual significant precision was bf16 (≈3 decimal digits). The fp32 size was paid on disk and in downstream RAM without any precision benefit. Mean-difference contrast directions average the noise away, but per-prompt projections, per-prompt SAE top-k encodings, and individual cosines inherit bf16 noise — worth disclosing in the reproducibility section regardless.

## Rebuild checklist
- [x] Implement `--token_mode` with three options and the span pre-pass. (Done.)
- [x] Plumb the chosen `token_mode` into [Step 5](05_encode_identity_saes.md). (Done.)
- [ ] Run all three modes on RunPod for at least layers `{0, 8, 16, 24, 32}` so geometry comparisons are tractable without re-running every layer.
- [ ] Verify each run's `span_locations.csv` shows `span_status == "exact"` for ≥99% of prompts and `span_n_tokens` distribution is reasonable (no single-token spans for multi-word forms).
- [ ] Re-encode each through [Step 5](05_encode_identity_saes.md); rerun [Step 7](07_analyze_identity_geometry.md) / [Step 8](08_analyze_identity_geometry_diagnostics.md) / [Step 9](09_analyze_shared_social_subspace.md) for each mode.
- [ ] Compare contrast AUC, probe accuracy, and shared-subspace spectrum across modes; record the chosen locus and the comparison in the methods doc as the audit-required justification.
- [ ] Disclose bf16 forward / fp32 storage in `run_config.json` (already there) and propagate to the methods write-up.

## Notes from the doc audit
- ~~Right-padding never re-asserted in the loop.~~ **Fixed:** the inference loop now `assert attention_mask[:, 0].all()` per batch, so a future tokenizer override that flipped padding_side fails loudly.
- ~~`outputs.hidden_states` requested twice (`from_pretrained` and per-call).~~ **Fixed:** removed the redundant `output_hidden_states=True` from `from_pretrained`; the per-forward call still passes it explicitly.
- ~~`--max_length 128` truncates silently.~~ **Fixed:** `warn_if_truncation_will_occur` runs a one-time CPU-only length pre-pass at script start and prints a WARNING with counts, percentiles, and the longest-truncated `prompt_id`s if any prompt exceeds `--max_length`. The pre-pass summary (`n_truncated`, `token_length_p99`, etc.) is also written to `run_config.json["length_pre_pass"]` for the audit trail.
