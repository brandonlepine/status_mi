#!/usr/bin/env python3
"""Encode identity-prompt residual activations with OpenMOSS/LlamaScope SAEs."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover
    load_safetensors = None


DEFAULT_ACTIVATION_DIR = Path(
    "/workspace/status_mi/results/activations/llama-3.1-8b/identity_prompts_final_token"
)
DEFAULT_SAE_DIR = Path("/workspace/status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x")
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token")
WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin", ".ckpt")
CONFIG_SUFFIXES = (".json", ".yaml", ".yml")


@dataclass
class LoadedSAE:
    w_enc: torch.Tensor
    b_enc: torch.Tensor | None
    w_dec: torch.Tensor
    b_dec: torch.Tensor | None
    config: dict[str, Any]
    source_files: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode identity activations with selected OpenMOSS/LlamaScope SAEs.")
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--sae_dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--activation_mode", default="final_token", choices=["final_token", "token_span"])
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--top_k_save", type=int, default=64)
    parser.add_argument("--save_dense_top_features", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[name]


def prepare_layer_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} exists and is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def layer_file_score(path: Path, layer: int) -> int:
    name = str(path).lower()
    layer_tokens = [
        f"layer_{layer}", f"layer-{layer}", f"layer{layer}",
        f"layers_{layer}", f"l{layer}", f"l_{layer}", f"blocks.{layer}", f"block_{layer}",
        f"{layer:02d}",
    ]
    score = 0
    if any(token in name for token in layer_tokens):
        score += 10
    if any(marker in name for marker in ["resid", "residual", "res_stream", "lxr"]):
        score += 4
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        score += 3
    return score


def find_sae_files(sae_dir: Path, layer: int) -> tuple[list[Path], list[Path]]:
    weights = [path for path in sae_dir.rglob("*") if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES]
    scored = [(layer_file_score(path, layer), path) for path in weights]
    matches = [path for score, path in scored if score >= 10]
    if not matches and len(weights) == 1:
        matches = weights
    if not matches:
        examples = "\n".join(str(path.relative_to(sae_dir)) for path in weights[:80])
        raise FileNotFoundError(
            f"No likely SAE weight files found for layer {layer} under {sae_dir}.\n"
            f"Available weight files:\n{examples}"
        )
    best_score = max(layer_file_score(path, layer) for path in matches)
    selected_weights = sorted([path for path in matches if layer_file_score(path, layer) == best_score])
    parent_dirs = {path.parent for path in selected_weights}
    configs = [
        path for path in sae_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in CONFIG_SUFFIXES and (path.parent in parent_dirs or path.name == "download_manifest.json")
    ]
    return selected_weights, sorted(configs)


def collect_tensor_items(obj: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        tensors[prefix or "tensor"] = obj.detach().cpu()
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            tensors.update(collect_tensor_items(value, child_prefix))
    elif hasattr(obj, "state_dict"):
        tensors.update(collect_tensor_items(obj.state_dict(), prefix))
    return tensors


def load_weight_tensors(files: list[Path]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            if load_safetensors is None:
                raise ImportError("safetensors is required to load .safetensors SAE checkpoints.")
            loaded = load_safetensors(str(path), device="cpu")
        else:
            loaded = torch.load(path, map_location="cpu")
        for key, value in collect_tensor_items(loaded, path.name).items():
            if value.ndim >= 1 and torch.is_floating_point(value):
                tensors[key] = value
    if not tensors:
        raise ValueError(f"No floating-point tensors found in files: {[str(path) for path in files]}")
    return tensors


def load_configs(files: list[Path]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        try:
            config[path.name] = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover
            config[path.name] = {"load_error": str(exc)}
    return config


def choose_matrix(tensors: dict[str, torch.Tensor], hidden_dim: int, kind: str) -> tuple[str, torch.Tensor]:
    candidates = []
    keywords = {
        "encoder": ["w_enc", "w_in", "encoder", "enc", "gate"],
        "decoder": ["w_dec", "w_out", "decoder", "dec"],
    }[kind]
    for key, tensor in tensors.items():
        if tensor.ndim != 2 or hidden_dim not in tensor.shape:
            continue
        key_lower = key.lower()
        keyword_score = max([5 for kw in keywords if kw in key_lower], default=0)
        orientation_score = 2 if (kind == "encoder" and tensor.shape[0] == hidden_dim) or (kind == "decoder" and tensor.shape[1] == hidden_dim) else 0
        size_score = max(tensor.shape)
        candidates.append((keyword_score + orientation_score, size_score, key, tensor))
    if not candidates:
        raise ValueError(f"Could not find a {kind} matrix with hidden dimension {hidden_dim}.")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, key, tensor = candidates[0]
    if kind == "encoder":
        matrix = tensor if tensor.shape[0] == hidden_dim else tensor.T
    else:
        matrix = tensor if tensor.shape[1] == hidden_dim else tensor.T
    return key, matrix.contiguous().float()


def choose_bias(tensors: dict[str, torch.Tensor], dim: int, keywords: list[str]) -> tuple[str | None, torch.Tensor | None]:
    candidates = []
    for key, tensor in tensors.items():
        if tensor.ndim != 1 or tensor.numel() != dim:
            continue
        key_lower = key.lower()
        score = max([5 for kw in keywords if kw in key_lower], default=0)
        candidates.append((score, key, tensor))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, key, tensor = candidates[0]
    return key, tensor.contiguous().float()


def load_sae(sae_dir: Path, layer: int, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> LoadedSAE:
    weight_files, config_files = find_sae_files(sae_dir, layer)
    tensors = load_weight_tensors(weight_files)
    config = load_configs(config_files)
    enc_key, w_enc = choose_matrix(tensors, hidden_dim, "encoder")
    dec_key, w_dec = choose_matrix(tensors, hidden_dim, "decoder")
    if w_dec.shape[1] != hidden_dim:
        raise ValueError(f"Resolved decoder shape should be [n_features, hidden_dim], got {tuple(w_dec.shape)}")
    if w_enc.shape[0] != hidden_dim:
        raise ValueError(f"Resolved encoder shape should be [hidden_dim, n_features], got {tuple(w_enc.shape)}")
    n_features = w_dec.shape[0]
    if w_enc.shape[1] != n_features:
        raise ValueError(f"Encoder feature dim {w_enc.shape[1]} does not match decoder feature dim {n_features}.")
    b_enc_key, b_enc = choose_bias(tensors, n_features, ["b_enc", "b_in", "encoder.bias", "enc.bias", "bias"])
    b_dec_key, b_dec = choose_bias(tensors, hidden_dim, ["b_dec", "b_out", "decoder.bias", "dec.bias", "pre_bias", "bias"])
    config["resolved"] = {
        "layer": layer,
        "hidden_dim": hidden_dim,
        "n_features": n_features,
        "encoder_key": enc_key,
        "decoder_key": dec_key,
        "encoder_bias_key": b_enc_key,
        "decoder_bias_key": b_dec_key,
        "source_weight_files": [str(path) for path in weight_files],
        "source_config_files": [str(path) for path in config_files],
        "activation_function": "relu",
        "note": "Generic loader applies ReLU((x - decoder_bias) @ W_enc + b_enc). Check OpenMOSS config if exact preprocessing differs.",
    }
    return LoadedSAE(
        w_enc=w_enc.to(device=device, dtype=dtype),
        b_enc=b_enc.to(device=device, dtype=dtype) if b_enc is not None else None,
        w_dec=w_dec.float(),
        b_dec=b_dec.to(device=device, dtype=dtype) if b_dec is not None else None,
        config=config,
        source_files=[str(path) for path in weight_files + config_files],
    )


def encode_batch(batch: np.ndarray, sae: LoadedSAE, device: torch.device, dtype: torch.dtype, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        x = torch.as_tensor(batch, device=device, dtype=dtype)
        if sae.b_dec is not None:
            x = x - sae.b_dec
        acts = x @ sae.w_enc
        if sae.b_enc is not None:
            acts = acts + sae.b_enc
        acts = torch.relu(acts)
        k = min(top_k, acts.shape[1])
        values, indices = torch.topk(acts, k=k, dim=1)
    return indices.cpu().numpy().astype(np.int32), values.float().cpu().numpy().astype(np.float32)


def compute_feature_stats(indices: np.ndarray, values: np.ndarray, n_features: int) -> pd.DataFrame:
    flat_idx = indices.reshape(-1)
    flat_val = values.reshape(-1)
    positive = flat_val > 0
    flat_idx = flat_idx[positive]
    flat_val = flat_val[positive]
    n_rows = indices.shape[0]
    counts = np.bincount(flat_idx, minlength=n_features).astype(np.int64)
    sums = np.bincount(flat_idx, weights=flat_val, minlength=n_features).astype(np.float64)
    max_vals = np.zeros(n_features, dtype=np.float32)
    if len(flat_idx):
        np.maximum.at(max_vals, flat_idx, flat_val)
    stats = pd.DataFrame({
        "feature_id": np.arange(n_features, dtype=np.int64),
        "activation_count": counts,
        "activation_frequency": counts / max(1, n_rows),
        "mean_activation_nonzero": np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0),
        "mean_activation_all": sums / max(1, n_rows),
        "max_activation": max_vals,
    })
    if len(flat_idx):
        sparse_df = pd.DataFrame({"feature_id": flat_idx, "value": flat_val})
        quantiles = sparse_df.groupby("feature_id")["value"].quantile([0.95, 0.99]).unstack()
        quantiles.columns = ["p95_activation", "p99_activation"]
        stats = stats.merge(quantiles.reset_index(), on="feature_id", how="left")
    else:
        stats["p95_activation"] = np.nan
        stats["p99_activation"] = np.nan
    stats[["p95_activation", "p99_activation"]] = stats[["p95_activation", "p99_activation"]].fillna(0.0)
    return stats


def maybe_save_dense_top_features(layer_dir: Path, indices: np.ndarray, values: np.ndarray, feature_stats: pd.DataFrame, top_n: int = 512) -> None:
    top_features = feature_stats.sort_values("activation_count", ascending=False).head(top_n)["feature_id"].to_numpy()
    feature_to_col = {feature_id: i for i, feature_id in enumerate(top_features)}
    dense = np.zeros((indices.shape[0], len(top_features)), dtype=np.float32)
    for row in range(indices.shape[0]):
        for feature_id, value in zip(indices[row], values[row]):
            col = feature_to_col.get(int(feature_id))
            if col is not None:
                dense[row, col] = value
    np.save(layer_dir / f"dense_top{top_n}_feature_ids.npy", top_features.astype(np.int32))
    np.save(layer_dir / f"dense_top{top_n}_feature_values.npy", dense)


def main() -> None:
    args = parse_args()
    if args.activation_mode != "final_token":
        raise NotImplementedError("token_span mode is scaffolded for later work; final_token mode is implemented first.")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    metadata_path = args.activation_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "activation_dir": str(args.activation_dir),
        "sae_dir": str(args.sae_dir),
        "output_dir": str(args.output_dir),
        "layers": parse_layers(args.layers),
        "activation_mode": args.activation_mode,
        "batch_size": args.batch_size,
        "top_k_save": args.top_k_save,
        "save_dense_top_features": args.save_dense_top_features,
        "device": args.device,
        "dtype": args.dtype,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")

    for layer in tqdm(parse_layers(args.layers), desc="layers"):
        start = time.perf_counter()
        layer_dir = args.output_dir / f"layer_{layer:02d}"
        prepare_layer_dir(layer_dir, args.overwrite)
        x_path = args.activation_dir / f"layer_{layer:02d}.npy"
        if not x_path.exists():
            raise FileNotFoundError(f"Missing activations: {x_path}")
        x = np.load(x_path, mmap_mode="r")
        if x.ndim != 2 or x.shape[0] != len(metadata):
            raise ValueError(f"{x_path} shape {x.shape} does not align with metadata rows {len(metadata)}")
        sae = load_sae(args.sae_dir, layer, x.shape[1], device, dtype)
        n_features = int(sae.w_dec.shape[0])
        top_k = min(args.top_k_save, n_features)
        feature_indices = np.empty((x.shape[0], top_k), dtype=np.int32)
        feature_values = np.empty((x.shape[0], top_k), dtype=np.float32)
        print(f"\nLayer {layer:02d}: encoding {x.shape[0]} rows, hidden_dim={x.shape[1]}, n_features={n_features}")
        for start_idx in tqdm(range(0, x.shape[0], args.batch_size), desc=f"layer {layer:02d} batches", leave=False):
            end_idx = min(start_idx + args.batch_size, x.shape[0])
            idx, val = encode_batch(np.asarray(x[start_idx:end_idx], dtype=np.float32), sae, device, dtype, top_k)
            feature_indices[start_idx:end_idx] = idx
            feature_values[start_idx:end_idx] = val
        np.save(layer_dir / f"feature_indices_top{top_k}.npy", feature_indices)
        np.save(layer_dir / f"feature_values_top{top_k}.npy", feature_values)
        metadata.to_csv(layer_dir / "metadata.csv", index=False)
        np.save(layer_dir / "sae_decoder.npy", sae.w_dec.numpy().astype(np.float32))
        if sae.b_dec is not None:
            np.save(layer_dir / "sae_decoder_bias.npy", sae.b_dec.float().cpu().numpy())
        (layer_dir / "sae_config_resolved.json").write_text(json.dumps(sae.config, indent=2, default=str) + "\n")
        stats = compute_feature_stats(feature_indices, feature_values, n_features)
        stats.to_csv(layer_dir / "feature_stats.csv", index=False)
        if args.save_dense_top_features:
            maybe_save_dense_top_features(layer_dir, feature_indices, feature_values, stats)
        print(f"Layer {layer:02d}: complete in {elapsed(start)}; outputs saved to {layer_dir}")


if __name__ == "__main__":
    main()
