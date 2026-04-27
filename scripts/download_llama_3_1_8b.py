#!/usr/bin/env python3
"""Download Llama-3.1-8B weights from Hugging Face.

The Meta Llama repositories are gated. Before running this script, accept the
model license on Hugging Face and authenticate with `huggingface-cli login` or
set `HF_TOKEN`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "meta-llama/Llama-3.1-8B"
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/models/llama-3.1-8b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Meta Llama-3.1-8B from Hugging Face."
    )
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, tag, or commit hash.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token. If omitted, uses HF_TOKEN or local HF login.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not download; only use files already present in the HF cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id}")
    print(f"Destination: {args.output_dir}")

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
        revision=args.revision,
        token=args.token,
        local_files_only=args.local_files_only,
        resume_download=True,
    )

    print(f"Download complete: {path}")


if __name__ == "__main__":
    main()
