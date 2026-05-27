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

from encode_identity_saes import (  # noqa: E402
    find_sae_files,
    load_configs,
    load_weight_tensors,
    load_sae,
    encode_full,
    decode_full,
)


DEFAULT_ACTIVATION_DIR = Path("/workspace/status_mi/results/activations/llama-3.1-8b/identity_prompts_final_token")
DEFAULT_SAE_DIR = Path("/workspace/status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x")
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/hook_validation")
DEFAULT_RECON_FVU_THRESHOLD = 0.15
DEFAULT_RECON_SAMPLE_N = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAE checkpoint hook/layer alignment against extracted activations.")
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--sae_dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--allow_mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check_reconstruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode a sample of real activations through the SAE and decode them; "
             "fail validation if FVU exceeds --reconstruction_fvu_threshold. This is "
             "the standing regression test for audit issue 1.4 (encoder fix).",
    )
    parser.add_argument(
        "--reconstruction_fvu_threshold",
        type=float,
        default=DEFAULT_RECON_FVU_THRESHOLD,
        help="Maximum acceptable FVU (fraction of variance unexplained) for the recon check.",
    )
    parser.add_argument(
        "--reconstruction_sample_n",
        type=int,
        default=DEFAULT_RECON_SAMPLE_N,
        help="Number of activation rows to sample for the recon check.",
    )
    parser.add_argument(
        "--recon_device",
        default=None,
        help="Device for the recon check (cuda / cuda:0 / cpu). Default: auto-detect.",
    )
    parser.add_argument(
        "--top_k_save_threshold",
        type=int,
        default=64,
        help="Audit 4.6 gate: top_k_save in encode_identity_saes.py truncates "
             "each row's encoding to this many features (default 64). If the "
             "SAE's empirical max L0 on real prompts exceeds this threshold, "
             "real activations are silently clipped to zero downstream. The "
             "validator records `recon_l0_clipping_risk = True` and fails the "
             "row when l0_max > threshold (unless --allow_mismatch). Set to "
             "match the --top_k_save you intend to use at encoding time.",
    )
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


def resolve_recon_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reconstruction_metrics(
    sae_dir: Path, layer: int, activation_path: Path, sample_n: int, device: torch.device,
) -> dict[str, Any]:
    """Encode a sample of real activations and decode them; report FVU,
    mean cosine, and mean L0. This is the standing regression test for
    audit issue 1.4 (encoder convention) and the natural place to catch
    a future regression in encode_full / decode_full.
    """
    x_all = np.load(activation_path, mmap_mode="r")
    n_rows = x_all.shape[0]
    n = min(sample_n, n_rows)
    rng = np.random.default_rng(seed=0)
    idx = rng.choice(n_rows, size=n, replace=False)
    idx.sort()
    x = np.asarray(x_all[idx], dtype=np.float32)

    # fp32 for the recon measurement so bf16 noise does not pollute the metric.
    sae = load_sae(sae_dir, layer, x.shape[1], device=device, dtype=torch.float32)
    with torch.inference_mode():
        x_t = torch.as_tensor(x, device=device, dtype=torch.float32)
        acts = encode_full(x_t, sae)
        recon = decode_full(acts, sae)
        diff = x_t - recon
        # FVU = ||x - x_hat||^2 / ||x - mean(x)||^2
        x_mean = x_t.mean(dim=0, keepdim=True)
        denom = ((x_t - x_mean) ** 2).sum().clamp(min=1e-12)
        fvu = (diff ** 2).sum() / denom
        # Mean per-row cosine
        cos = torch.nn.functional.cosine_similarity(x_t, recon, dim=-1).mean()
        # Empirical L0 — distribution stats are the audit-4.6 gate for top_k_save
        # truncation in encode_identity_saes.py. Reporting percentiles in addition
        # to mean/max lets the operator see how close the tail is to the cap.
        l0_per_row = (acts > 0).float().sum(dim=-1)
        mean_l0 = l0_per_row.mean()
        max_l0 = l0_per_row.max()
        l0_p50 = l0_per_row.quantile(0.50)
        l0_p95 = l0_per_row.quantile(0.95)
        l0_p99 = l0_per_row.quantile(0.99)
    return {
        "reconstruction_n_rows_sampled": int(n),
        "reconstruction_fvu": float(fvu.detach().cpu().item()),
        "reconstruction_cosine_mean": float(cos.detach().cpu().item()),
        "reconstruction_mean_l0": float(mean_l0.detach().cpu().item()),
        "reconstruction_max_l0": float(max_l0.detach().cpu().item()),
        "reconstruction_l0_p50": float(l0_p50.detach().cpu().item()),
        "reconstruction_l0_p95": float(l0_p95.detach().cpu().item()),
        "reconstruction_l0_p99": float(l0_p99.detach().cpu().item()),
        "sae_scale_in": float(sae.scale_in),
        "sae_scale_out": float(sae.scale_out),
        "sae_jump_relu_threshold": float(sae.theta),
        "sae_act_fn": sae.act_fn,
        "sae_d_model": int(sae.d_model),
        "sae_d_sae": int(sae.d_sae),
        "sae_hyperparameters_path": str(sae.hyperparameters_path) if sae.hyperparameters_path else None,
    }


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
        "hook_metadata": hook_metadata,
        "hf_hidden_states_alignment_note": note,
    }

    recon_ok = True
    if args.check_reconstruction:
        device = resolve_recon_device(args.recon_device)
        try:
            recon = reconstruction_metrics(
                args.sae_dir, layer, activation_path, args.reconstruction_sample_n, device,
            )
        except (FileNotFoundError, ValueError) as exc:
            row["reconstruction_error"] = f"{type(exc).__name__}: {exc}"
            recon_ok = False
        else:
            row.update(recon)
            recon_ok = recon["reconstruction_fvu"] <= args.reconstruction_fvu_threshold
            row["reconstruction_fvu_threshold"] = args.reconstruction_fvu_threshold
            # Audit 4.6 gate: empirical max L0 must not exceed top_k_save, else
            # encode_identity_saes.py silently truncates real activations to zero.
            l0_clipping_risk = recon["reconstruction_max_l0"] > args.top_k_save_threshold
            row["top_k_save_threshold"] = args.top_k_save_threshold
            row["recon_l0_clipping_risk"] = l0_clipping_risk
            recon_ok = recon_ok and (not l0_clipping_risk)
            row["reconstruction_passed"] = recon_ok
    else:
        row["reconstruction_passed"] = None  # not checked

    ok = ok and recon_ok
    row["validation_passed"] = ok
    if not ok and not args.allow_mismatch:
        raise ValueError(
            f"Validation failed for layer {layer} (reconstruction_passed="
            f"{row.get('reconstruction_passed')}, "
            f"checkpoint_layer_match={checkpoint_layer_match}, "
            f"position_marker_is_R={position_marker_is_r}, "
            f"hidden_dim_match={hidden_dim_match}, "
            f"recon_l0_clipping_risk={row.get('recon_l0_clipping_risk')}). "
            f"Row:\n{json.dumps(row, indent=2, default=str)}"
        )
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
