#!/usr/bin/env python3
"""Download selected OpenMOSS/LlamaScope SAE checkpoints from Hugging Face.

For each requested layer, this script:

1. Filters the repo's full file list to files under the LlamaScope per-layer
   marker `L<layer>R-<width>x` (default expansion factor 32).
2. Hard-requires `hyperparameters.json` to be among the matched files. That
   file is the source of truth for the encoder activation function, the
   dataset-wise input normalization scale, and the decoder-bias-placement
   flag (audit issue 1.4); without it the encoder cannot be loaded.
3. Resolves the HF revision to an absolute commit SHA via the Hub API
   (defaulting to the head commit of the repo's default branch) and uses
   that SHA for every file download so the local snapshot is atomic and
   reproducible regardless of when the script is run.
4. Logs, per downloaded file, the resolved local path, the (position,
   width) marker parsed from its repo path, and the legacy layer_score
   heuristic value so post-hoc audits can confirm the right SAE variant
   was chosen.
5. After all layers are downloaded, asserts that the parsed (position,
   width) markers are identical across the requested layers; raises
   otherwise so a mix of variants cannot be pulled silently.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x"
DEFAULT_LOCAL_DIR = Path("/workspace/status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x")
DEFAULT_WIDTH = 32
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin", ".ckpt")
CONFIG_SUFFIXES = (".json", ".yaml", ".yml", ".txt", ".md")
RESIDUAL_MARKERS = ("resid", "residual", "res_stream", "res-stream", "lxr", "blocks")
REQUIRED_PER_LAYER = ("hyperparameters.json",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--layers", default="24", help="Comma-separated layer numbers, e.g. 16,24,32.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="SAE expansion factor (Llamascope uses 32). Used to build the explicit per-layer file marker.")
    parser.add_argument("--local_dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch / tag / commit hash). If omitted, the script resolves the repo's default branch to an explicit commit SHA via the Hub API and uses that for every download.",
    )
    parser.add_argument("--token", default=None)
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist locally.")
    parser.add_argument("--dry_run", action="store_true", help="List the files that would be downloaded but skip the actual download; the manifest is still written.")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def layer_patterns(layer: int) -> list[re.Pattern[str]]:
    """Legacy heuristic patterns; kept for the audit-only layer_score field."""
    return [
        re.compile(rf"(^|[^0-9])layer[_\-.]?0?{layer}([^0-9]|$)", re.I),
        re.compile(rf"(^|[^0-9])layers?[_\-.]?0?{layer}([^0-9]|$)", re.I),
        re.compile(rf"(^|[^0-9])l[_\-.]?0?{layer}([^0-9]|$)", re.I),
        re.compile(rf"(^|[^0-9])blocks?[_\-.]?0?{layer}([^0-9]|$)", re.I),
        re.compile(rf"(^|[^0-9])0?{layer}([^0-9]|$)", re.I),
    ]


def layer_score(path: str, layer: int) -> int:
    """Legacy heuristic score used as an audit signal only; selection is now
    driven by `layer_marker_pattern`. A high score on a path that did NOT
    match the explicit marker, or a low score on one that did, is worth
    investigating."""
    lower = path.lower()
    score = 0
    for i, pattern in enumerate(layer_patterns(layer)):
        if pattern.search(lower):
            score = max(score, 20 - i)
    if any(marker in lower for marker in RESIDUAL_MARKERS):
        score += 5
    if path.lower().endswith(WEIGHT_SUFFIXES):
        score += 3
    return score


def layer_marker_pattern(layer: int, width: int) -> re.Pattern[str]:
    """Match files whose path contains the LlamaScope per-layer marker, e.g.
    `L24R-32X` or `L24R-32x`. Case-insensitive on the width's trailing 'x'.
    Word boundaries prevent accidental sub-matches inside other numeric tokens.
    """
    return re.compile(rf"(^|[^A-Za-z0-9])L{layer}R-{width}[Xx]([^A-Za-z0-9]|$)")


def parse_marker(path: str) -> tuple[int, str, int] | None:
    """Extract (layer, position, width) from a LlamaScope-style path token."""
    m = re.search(r"(?:^|[^A-Za-z0-9])L(?P<layer>\d+)(?P<position>[A-Za-z])-(?P<width>\d+)[Xx](?:[^A-Za-z0-9]|$)", path)
    if not m:
        return None
    return int(m.group("layer")), m.group("position").upper(), int(m.group("width"))


def is_weight_file(path: str) -> bool:
    return path.lower().endswith(WEIGHT_SUFFIXES)


def is_config_file(path: str) -> bool:
    return path.lower().endswith(CONFIG_SUFFIXES)


def select_layer_files(repo_files: list[str], layer: int, width: int) -> list[str]:
    """Explicit per-layer file selection: any repo file path containing the
    LlamaScope marker `L<layer>R-<width>x`, restricted to weight + config
    suffixes. Hard-requires `hyperparameters.json` to be present; if it is
    not, raise so the encoder fix (audit 1.4) cannot silently miss its
    source of truth.
    """
    marker = layer_marker_pattern(layer, width)
    matched = sorted({path for path in repo_files if marker.search(path)})
    keep = [path for path in matched if is_weight_file(path) or is_config_file(path)]

    if not keep:
        candidates = "\n".join(matched[:80]) or "  (no marker matches)"
        raise FileNotFoundError(
            f"No weight/config files matched the LlamaScope marker L{layer}R-{width}x. "
            f"Matched paths (any suffix):\n{candidates}"
        )

    missing_required = [name for name in REQUIRED_PER_LAYER if not any(Path(f).name == name for f in keep)]
    if missing_required:
        listing = "\n".join(keep)
        raise FileNotFoundError(
            f"Required per-layer files {missing_required} not found among matched files for layer {layer}. "
            f"hyperparameters.json is mandatory because encode_identity_saes.py reads "
            f"act_fn / jump_relu_threshold / dataset_average_activation_norm / "
            f"apply_decoder_bias_to_pre_encoder from it (audit issue 1.4). "
            f"Matched files were:\n{listing}"
        )

    return keep


def resolve_revision(api: HfApi, repo_id: str, requested: str | None) -> str:
    """Resolve any HF revision spec (branch / tag / commit / None) to its
    absolute commit SHA via the Hub API. Storing the resolved SHA makes
    every manifest reproducible even when the requested revision is `main`
    and the branch advances later.
    """
    info = api.model_info(repo_id=repo_id, revision=requested)
    sha = getattr(info, "sha", None)
    if not sha:
        raise RuntimeError(f"Could not resolve revision {requested!r} for {repo_id!r} to a commit SHA.")
    return sha


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    args.local_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=args.token)

    print(f"Listing files from {args.repo_id}...")
    resolved_revision = resolve_revision(api, args.repo_id, args.revision)
    print(f"Resolved revision {args.revision!r} to commit SHA {resolved_revision}")
    files = api.list_repo_files(repo_id=args.repo_id, revision=resolved_revision)

    manifest: dict[str, object] = {
        "repo_id": args.repo_id,
        "revision_requested": args.revision,
        "revision_resolved": resolved_revision,
        "width": args.width,
        "layers_requested": layers,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "layers": {},
    }

    per_layer_markers: dict[int, set[tuple[str, int]]] = {}

    for layer in layers:
        matches = select_layer_files(files, layer, args.width)
        print(f"\nLayer {layer}: {len(matches)} matched files (explicit marker selection)")
        layer_entries = []
        markers_for_layer: set[tuple[str, int]] = set()
        for repo_path in matches:
            parsed = parse_marker(repo_path)
            position = parsed[1] if parsed else None
            width = parsed[2] if parsed else None
            if position is not None and width is not None:
                markers_for_layer.add((position, width))
            entry: dict[str, object] = {
                "repo_path": repo_path,
                "layer_score": int(layer_score(repo_path, layer)),
                "parsed_position": position,
                "parsed_width": width,
            }
            if args.dry_run:
                entry["local_path"] = None
                print(f"  [dry-run] would download {repo_path}")
            else:
                local_path = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=repo_path,
                    revision=resolved_revision,
                    token=args.token,
                    local_dir=args.local_dir,
                    local_dir_use_symlinks=False,
                    force_download=args.force,
                )
                rel_path = str(Path(local_path).relative_to(args.local_dir))
                entry["local_path"] = rel_path
                print(f"  {repo_path} -> {rel_path}")
            layer_entries.append(entry)

        per_layer_markers[layer] = markers_for_layer
        manifest["layers"][str(layer)] = layer_entries

    # Cross-layer (position, width) consistency check.
    # Each layer's matched files must agree on a single (position, width), and
    # all layers must share the same (position, width).
    per_layer_resolved: dict[int, tuple[str, int]] = {}
    for layer, markers in per_layer_markers.items():
        if not markers:
            raise ValueError(
                f"Layer {layer} matched files but no path contained a parseable "
                f"L<layer><position>-<width>x marker; cannot verify the variant."
            )
        if len(markers) > 1:
            raise ValueError(f"Layer {layer} has inconsistent (position, width) markers across files: {markers}")
        per_layer_resolved[layer] = next(iter(markers))

    unique = set(per_layer_resolved.values())
    if len(unique) > 1:
        raise ValueError(
            f"Requested layers downloaded different SAE variants: {per_layer_resolved}. "
            "All layers must share the same (position, width). Refusing to write a mixed manifest."
        )

    if per_layer_resolved:
        position, width = next(iter(unique))
        manifest["resolved_position"] = position
        manifest["resolved_width"] = width
        print(f"\nCross-layer marker check OK: position={position}, width={width}x for layers {sorted(per_layer_resolved)}.")
    else:
        manifest["resolved_position"] = None
        manifest["resolved_width"] = None

    manifest_path = args.local_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nDownload manifest saved: {manifest_path}")
    if args.dry_run:
        print("Dry-run mode: no files were actually downloaded.")


if __name__ == "__main__":
    main()
