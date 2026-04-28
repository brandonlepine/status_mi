#!/usr/bin/env python3
"""Download selected OpenMOSS/LlamaScope SAE checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x"
DEFAULT_LOCAL_DIR = Path("/workspace/status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x")
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin", ".ckpt")
CONFIG_SUFFIXES = (".json", ".yaml", ".yml", ".txt", ".md")
RESIDUAL_MARKERS = ("resid", "residual", "res_stream", "res-stream", "lxr", "blocks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenMOSS/LlamaScope SAEs for selected layers.")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--layers", default="24", help="Comma-separated layer numbers, e.g. 16,24,32.")
    parser.add_argument("--local_dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist locally.")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def layer_patterns(layer: int) -> list[re.Pattern[str]]:
    return [
        re.compile(rf"(^|[^0-9])layer[_\-.]?0?{layer}([^0-9]|$)", re.I),
        re.compile(rf"(^|[^0-9])layers?[_\-.]?0?{layer}([^0-9]|$)", re.I),
        re.compile(rf"(^|[^0-9])l[_\-.]?0?{layer}([^0-9]|$)", re.I),
        re.compile(rf"(^|[^0-9])blocks?[_\-.]?0?{layer}([^0-9]|$)", re.I),
        re.compile(rf"(^|[^0-9])0?{layer}([^0-9]|$)", re.I),
    ]


def is_weight_file(path: str) -> bool:
    return path.lower().endswith(WEIGHT_SUFFIXES)


def is_config_file(path: str) -> bool:
    return path.lower().endswith(CONFIG_SUFFIXES)


def layer_score(path: str, layer: int) -> int:
    lower = path.lower()
    score = 0
    for i, pattern in enumerate(layer_patterns(layer)):
        if pattern.search(lower):
            score = max(score, 20 - i)
    if any(marker in lower for marker in RESIDUAL_MARKERS):
        score += 5
    if is_weight_file(path):
        score += 3
    return score


def find_layer_matches(files: list[str], layer: int) -> list[str]:
    scored = [(layer_score(path, layer), path) for path in files if is_weight_file(path)]
    matches = [path for score, path in scored if score >= 10]
    if not matches:
        return []

    best_score = max(score for score, path in scored if path in matches)
    primary = [path for score, path in scored if score == best_score and path in matches]
    parent_dirs = {str(Path(path).parent) for path in primary}
    related_configs = [
        path for path in files
        if is_config_file(path) and (str(Path(path).parent) in parent_dirs or Path(path).name.lower() in {"config.json", "cfg.json"})
    ]
    return sorted(set(primary + related_configs))


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    args.local_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=args.token)
    print(f"Listing files from {args.repo_id}...")
    files = api.list_repo_files(repo_id=args.repo_id, revision=args.revision)
    manifest: dict[str, object] = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "layers_requested": layers,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "layers": {},
    }

    for layer in layers:
        matches = find_layer_matches(files, layer)
        if not matches:
            examples = "\n".join(files[:80])
            raise FileNotFoundError(
                f"No likely SAE checkpoint files found for layer {layer} in {args.repo_id}.\n"
                "Inspect the repo file names and pass a narrower manual download if needed.\n"
                f"First repo files:\n{examples}"
            )
        print(f"\nLayer {layer}: downloading {len(matches)} matched files")
        layer_entries = []
        for repo_path in matches:
            local_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=repo_path,
                revision=args.revision,
                token=args.token,
                local_dir=args.local_dir,
                local_dir_use_symlinks=False,
                force_download=args.force,
            )
            rel_path = str(Path(local_path).relative_to(args.local_dir))
            print(f"  {repo_path} -> {rel_path}")
            layer_entries.append({"repo_path": repo_path, "local_path": rel_path})
        manifest["layers"][str(layer)] = layer_entries

    manifest_path = args.local_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nDownload manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
