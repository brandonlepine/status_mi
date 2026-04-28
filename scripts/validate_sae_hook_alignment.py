#!/usr/bin/env python3
"""Validate that OpenMOSS/LlamaScope SAE checkpoints match activation hooks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from encode_identity_saes import find_sae_files, load_configs, load_weight_tensors  # noqa: E402


DEFAULT_ACTIVATION_DIR = Path("/workspace/status_mi/results/activations/llama-3.1-8b/identity_prompts_final_token")
DEFAULT_SAE_DIR = Path("/workspace/status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x")
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/hook_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAE checkpoint hook/layer alignment against extracted activations.")
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--sae_dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--allow_mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_llamascope_name(paths: list[Path]) -> dict[str, Any]:
    joined = " ".join(str(path) for path in paths)
    match = re.search(r"L(?P<layer>\d+)(?P<position>[A-Za-z])-(?P<width>\d+)x", joined)
    if not match:
        match = re.search(r"layer[_\-.]?(?P<layer>\d+).*?(?P<position>R|resid|residual).*?(?P<width>\d+)x", joined, re.I)
    if not match:
        return {"sae_layer": None, "sae_position": None, "sae_width": None, "parse_status": "failed"}
    position = match.group("position")
    if position.lower() in {"resid", "residual"}:
        position = "R"
    return {
        "sae_layer": int(match.group("layer")),
        "sae_position": position.upper(),
        "sae_width": int(match.group("width")),
        "parse_status": "parsed",
    }


def flatten_config(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_config(value, child))
    else:
        out[prefix] = obj
    return out


def extract_hook_metadata(configs: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_config(configs)
    wanted = {}
    for key, value in flat.items():
        lower = key.lower()
        if any(token in lower for token in ["hook", "layer", "site", "point", "position", "submodule"]):
            if isinstance(value, (str, int, float, bool)) or value is None:
                wanted[key] = value
    return wanted


def infer_sae_dims(tensors: dict[str, torch.Tensor], hidden_dim: int) -> tuple[int | None, int | None]:
    candidate_mats = [tensor for tensor in tensors.values() if tensor.ndim == 2 and hidden_dim in tensor.shape]
    if not candidate_mats:
        return None, None
    feature_dim = max(max(tensor.shape) for tensor in candidate_mats)
    return hidden_dim, int(feature_dim)


def validate_row(args: argparse.Namespace, layer: int) -> dict[str, Any]:
    weight_files, config_files = find_sae_files(args.sae_dir, layer)
    parsed = parse_llamascope_name(weight_files + config_files)
    configs = load_configs(config_files)
    hook_metadata = extract_hook_metadata(configs)
    activation_path = args.activation_dir / f"layer_{layer:02d}.npy"
    if not activation_path.exists():
        raise FileNotFoundError(f"Missing activation file: {activation_path}")
    x = np.load(activation_path, mmap_mode="r")
    if x.ndim != 2:
        raise ValueError(f"{activation_path} must be 2D, got shape {x.shape}")
    tensors = load_weight_tensors(weight_files)
    sae_input_dim, sae_feature_dim = infer_sae_dims(tensors, x.shape[1])
    checkpoint_layer_match = parsed["sae_layer"] in {None, layer}
    position_marker_is_r = parsed["sae_position"] in {None, "R"}
    hidden_dim_match = sae_input_dim == x.shape[1]
    ok = checkpoint_layer_match and position_marker_is_r and hidden_dim_match
    note = (
        "HF output_hidden_states convention: hidden_states[0] is embeddings and hidden_states[k] "
        "is the post-block-k residual stream for Llama-style decoder layers. Therefore layer_24.npy "
        "matches L24R only if the extraction script saved hidden_states[24] and did not apply final norm/lm_head."
    )
    row = {
        "requested_layer": layer,
        "activation_file": str(activation_path),
        "activation_shape": list(x.shape),
        "activation_hidden_dim": int(x.shape[1]),
        "sae_files": [str(path) for path in weight_files],
        "config_files": [str(path) for path in config_files],
        **parsed,
        "sae_input_dim": sae_input_dim,
        "sae_feature_dim": sae_feature_dim,
        "checkpoint_layer_match": checkpoint_layer_match,
        "position_marker_is_R": position_marker_is_r,
        "hidden_dim_match": hidden_dim_match,
        "validation_passed": ok,
        "hook_metadata": hook_metadata,
        "hf_hidden_states_alignment_note": note,
    }
    if not ok and not args.allow_mismatch:
        raise ValueError(f"Hook alignment validation failed for layer {layer}: {json.dumps(row, indent=2, default=str)}")
    return row


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [validate_row(args, layer) for layer in parse_layers(args.layers)]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "activation_dir": str(args.activation_dir),
        "sae_dir": str(args.sae_dir),
        "model_path": str(args.model_path) if args.model_path else None,
        "rows": rows,
    }
    (args.output_dir / "hook_alignment_validation.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["sae_files"] = ";".join(row["sae_files"])
        flat["config_files"] = ";".join(row["config_files"])
        flat["hook_metadata"] = json.dumps(row["hook_metadata"], default=str)
        csv_rows.append(flat)
    pd.DataFrame(csv_rows).to_csv(args.output_dir / "hook_alignment_validation.csv", index=False)
    print(pd.DataFrame(csv_rows)[["requested_layer", "sae_layer", "sae_position", "sae_width", "activation_hidden_dim", "sae_input_dim", "sae_feature_dim", "validation_passed"]])
    print(f"Validation outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
