# Step 3 — `scripts/download_openmoss_saes.py`

**Stage:** 1 — Identity activations and SAE encodings (one-time setup)
**Runs after:** — (independent of model download; needs HF auth only)
**Feeds into:** [Step 5 — `encode_identity_saes.py`](05_encode_identity_saes.md), [Step 6 — `validate_sae_hook_alignment.py`](06_validate_sae_hook_alignment.md), and every BBQ steering script that loads SAE weights.

## Purpose
Download per-layer SAE checkpoints from the OpenMOSS/LlamaScope repository (`OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x` by default) for the requested layers. The repo contains SAEs for many layers and submodules; this script scores filenames against layer/residual-stream regexes and downloads the highest-scoring weight files plus sibling configs, then writes a `download_manifest.json`.

## Inputs
- `--repo_id` (default `OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x`).
- `--layers` (comma-separated layer numbers, default `24`).
- HF auth.

## Outputs
- `/workspace/status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x/` containing one or more weight files per requested layer plus configs.
- `download_manifest.json` recording which `repo_path` files were downloaded for each layer.

## Key implementation details
- `layer_score` heuristic: layer regex matches contribute 11–20 (longest pattern wins), `resid|residual|res_stream|res-stream|lxr|blocks` markers add 5, weight-suffix add 3. A file with `score >= 10` is considered a match; ties pull the file's directory siblings as configs.
- Five layer regexes attempted in priority order: `layer_<N>`, `layers_<N>`, `l<N>`, `blocks_<N>`, and a fall-back bare-`<N>` pattern (the last is permissive and can mis-match in cluttered filename trees).
- `RESIDUAL_MARKERS` is the only check for "is this the residual-stream SAE?" — no inspection of internal config keys is performed at download time. Hook/position verification is deferred to [Step 6](06_validate_sae_hook_alignment.md).
- `local_dir_use_symlinks=False` — full copy into the destination.

## Issues & Opportunities

This step has **no direct issue** in `issues_and_opportunities.md`. It does, however, sit immediately upstream of issue **1.4 [MAJOR] SAE preprocessing convention is not verified** (handled in [Step 5](05_encode_identity_saes.md) and [Step 6](06_validate_sae_hook_alignment.md)), and the integrity of every SAE artifact in the project depends on the right files being pulled here.

**Dependency flag:**
The downloader saves whatever `.json`/`.yaml` configs live next to the chosen weight files (`sae_config_resolved.json` is written downstream in [Step 5](05_encode_identity_saes.md)). Whether the LlamaScope-specific preprocessing (input normalization, JumpReLU/TopK activation, threshold) is recorded there determines whether issue 1.4 can be resolved without re-reading the HF repo card. If the config files saved by this script are sparse, 1.4 cannot be answered from local artifacts alone.

**Targeted fix:** When downloading, also download the top-level `README.md` / model card and any `hyperparams.json` / `config.json` from the repo root (not just per-layer siblings) so the activation-function and normalization conventions are recoverable locally without an HF round-trip. Today `is_config_file` matches `.json|.yaml|.yml|.txt|.md` but the `find_layer_matches` selector only pulls configs in the *same parent directory* as the chosen weights (plus `config.json`/`cfg.json` anywhere); a top-level architecture note will be missed if it lives elsewhere.

## Rebuild checklist
- [ ] Add an explicit "always pull these top-level files" list (`README.md`, `hyperparams.json`, `config.json`, `model_card.md` if present) to guarantee the activation-function/normalization convention is captured locally.
- [ ] Log to the manifest the full `layer_score` for each downloaded file so post-hoc audits can confirm the right SAE variant was chosen.
- [ ] When downloading multiple layers (planned multi-layer expansion per issue 5.8), assert that the resolved width (`32x`) and position marker (`R`) are identical across layers — a mismatch would mean the regex picked a different SAE variant for one of the layers.
- [ ] Pin `--revision` to a commit hash and store it in the manifest; LlamaScope checkpoints can be re-released.

## Notes from the doc audit
- The fall-back bare-number regex `(^|[^0-9])0?{layer}([^0-9]|$)` will match almost any path containing the layer number (including, e.g., a config field name with `24` in it). The `score >= 10` cutoff prevents that fall-back alone from triggering a download (it contributes 16, but only on weight files which also need the residual markers to add up to a confident match), but the heuristic is brittle if the upstream repo layout changes — worth replacing with an explicit allowlist per SAE family if/when multi-layer SAEs are added.
