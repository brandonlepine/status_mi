# Step 3 — `scripts/download_openmoss_saes.py`

**Stage:** 1 — Identity activations and SAE encodings (one-time setup)
**Runs after:** — (independent of model download; needs HF auth only)
**Feeds into:** [Step 5 — `encode_identity_saes.py`](05_encode_identity_saes.md), [Step 6 — `validate_sae_hook_alignment.py`](06_validate_sae_hook_alignment.md), and every BBQ steering script that loads SAE weights.

## Purpose
Download per-layer SAE checkpoints from the OpenMOSS/LlamaScope repository (`OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x` by default) for the requested layers. Files are selected by an explicit per-layer marker (`L<layer>R-<width>x`), the HF revision is resolved to an absolute commit SHA and pinned, and the resulting `download_manifest.json` records both the resolved revision and a per-file audit trail (layer marker + legacy heuristic score) so it is possible to verify post-hoc that the right SAE variant was pulled.

## Inputs
- `--repo_id` (default `OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x`).
- `--layers` (comma-separated layer numbers, default `24`).
- `--width` (SAE expansion factor; default `32`).
- `--revision` (optional HF revision spec; resolved to an absolute commit SHA via the Hub API and used for every download in this invocation).
- `--dry_run` (lists what would be downloaded without fetching; manifest is still written).
- HF auth.

## Outputs
- `/workspace/status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x/` — one weight file plus `hyperparameters.json` per requested layer (other configs as present).
- `download_manifest.json` recording per layer:
  - `repo_path` and `local_path` for each file
  - `parsed_position`, `parsed_width` (extracted from the LlamaScope marker on the path)
  - `layer_score` (legacy heuristic; audit-only)
- Manifest top-level: `revision_requested`, `revision_resolved` (absolute SHA), `width`, `resolved_position`, `resolved_width`, `downloaded_at`, `dry_run`.

## Key implementation details
- **Explicit file selection.** `select_layer_files(repo_files, layer, width)` filters the repo listing to paths containing the marker `L<layer>R-<width>x` (word-bounded regex; case-insensitive on the trailing `x`). Only files with weight or config suffixes are kept. The heuristic `layer_score` no longer drives selection but is recorded per file for audit.
- **Mandatory `hyperparameters.json`.** Each layer's matched file list must include a `hyperparameters.json`; otherwise `select_layer_files` raises `FileNotFoundError`. This file is the source of truth for the encoder convention (audit issue 1.4): `act_fn`, `jump_relu_threshold`, `dataset_average_activation_norm`, `apply_decoder_bias_to_pre_encoder`.
- **Revision pinning.** `resolve_revision(api, repo_id, requested)` calls `HfApi.model_info` to convert any branch/tag/None into an absolute commit SHA before any download. Every `hf_hub_download` call then uses that SHA so the local snapshot is atomic even if the branch moves mid-download.
- **Cross-layer consistency.** After all layers are downloaded the script parses `(position, width)` from each layer's filenames. Each layer must agree on a single marker internally, and all requested layers must share the same marker. Mismatches raise rather than write a mixed manifest.
- **Marker regex.** `parse_marker` matches `L(\d+)([A-Za-z])-(\d+)[Xx]` with word boundaries; that prevents `L24` from matching `L240`, and the position character is preserved (so `L24R-32X` and a hypothetical `L24A-32X` parse to `('R', 32)` and `('A', 32)` respectively).

## Issues & Opportunities

This step has **no direct issue** in `issues_and_opportunities.md`. It is the upstream dependency for **1.4 [BLOCKER] SAE preprocessing convention is wrong** in [Step 5](05_encode_identity_saes.md) and [Step 6](06_validate_sae_hook_alignment.md): without `hyperparameters.json` reaching disk, the encoder fix has nothing to read.

### 1.4 dependency — FIX LANDED 2026-05-26

The audit's three concerns about this script — heuristic selection, missing top-level configs, no revision pin — were addressed in commit `1ed1422`:

- Explicit per-layer marker selection replaces the heuristic; `hyperparameters.json` is mandatory; layer_score is logged for audit.
- HF revision always resolved to an absolute commit SHA via the Hub API; both `revision_requested` and `revision_resolved` recorded in the manifest.
- Cross-layer (position, width) consistency assertion runs before the manifest is written.

What remains: when you re-download the SAEs on RunPod (which you must, to verify the new selector pulls `hyperparameters.json` for every layer you want to use), pass a `--revision` to lock to a specific OpenMOSS commit so the local snapshot is reproducible.

## Rebuild checklist
- [ ] Re-run on RunPod for the layers you want to use: `python scripts/download_openmoss_saes.py --layers 24 --revision <commit_hash>`. With `--revision` omitted, the script will still record the resolved SHA in the manifest, but pinning is preferable.
- [ ] Verify the manifest's `resolved_position` is `R` and `resolved_width` matches your `--width`.
- [ ] Verify `hyperparameters.json` is present for every requested layer under the resolved SAE directory.
- [ ] Re-run [Step 5](05_encode_identity_saes.md) and [Step 6](06_validate_sae_hook_alignment.md) against the freshly downloaded checkpoints.

## Notes from the doc audit
- `layer_score` is now an audit signal only: a high score on a path that did NOT match the explicit marker (or vice versa) is worth investigating, since it would indicate the marker regex missed a file the legacy heuristic would have caught.
- The script no longer pulls repo-root configs (`README.md`, etc.) by default — the necessary preprocessing info now lives in per-layer `hyperparameters.json` and is hard-required. If you want top-level docs locally for reference, fetch them manually from the HF UI.
- `--dry_run` lets you verify selection before committing to a multi-GB download. Worth running first whenever `--width`, `--layers`, or the repo changes.
