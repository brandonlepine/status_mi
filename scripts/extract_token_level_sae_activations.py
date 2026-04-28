#!/usr/bin/env python3
"""Extract token-level SAE activations for selected identity-related features."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from encode_identity_saes import load_sae, torch_dtype  # noqa: E402


DEFAULT_MODEL_PATH = Path("/workspace/status_mi/models/llama-3.1-8b")
DEFAULT_SAE_DIR = Path("/workspace/status_mi/saes/openmoss/Llama3_1-8B-Base-LXR-32x")
DEFAULT_IDENTITY_CSV = Path("/workspace/status_mi/results/activations/llama-3.1-8b/identity_prompts_final_token/metadata.csv")
DEFAULT_ANALYSIS_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/analysis")
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/feature_cards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract token-level SAE activations for selected identity features.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sae_dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--identity_csv", type=Path, default=DEFAULT_IDENTITY_CSV)
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--features", default=None)
    parser.add_argument("--top_features_per_contrast", type=int, default=20)
    parser.add_argument("--top_features_per_identity", type=int, default=20)
    parser.add_argument("--max_prompts_per_feature", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def find_final_token_root(analysis_dir: Path) -> Path:
    if analysis_dir.name == "analysis":
        return analysis_dir.parent
    return analysis_dir


def find_topk_arrays(layer_dir: Path) -> tuple[Path, Path, int]:
    index_files = sorted(layer_dir.glob("feature_indices_top*.npy"))
    if not index_files:
        raise FileNotFoundError(f"No feature_indices_top*.npy found in {layer_dir}")
    index_path = index_files[-1]
    top_k = int(index_path.stem.split("top")[-1])
    value_path = layer_dir / f"feature_values_top{top_k}.npy"
    if not value_path.exists():
        raise FileNotFoundError(f"Missing {value_path}")
    return index_path, value_path, top_k


def select_features(analysis_dir: Path, layers: list[int], explicit: list[int], top_contrast: int, top_identity: int) -> dict[int, list[int]]:
    if explicit:
        return {layer: sorted(set(explicit)) for layer in layers}
    selected: dict[int, set[int]] = {layer: set() for layer in layers}
    candidates = safe_read(analysis_dir / "intervention_candidate_features.csv")
    joined = safe_read(analysis_dir / "feature_selectivity_alignment_joined.csv")
    identity = safe_read(analysis_dir / "feature_identity_selectivity.csv")
    for layer in layers:
        if not candidates.empty:
            layer_df = candidates[candidates["layer"].eq(layer)]
            for _, group in layer_df.groupby("contrast_name", sort=True):
                selected[layer].update(group.sort_values("combined_score", ascending=False).head(top_contrast)["feature_id"].astype(int))
        if not joined.empty:
            layer_df = joined[joined["layer"].eq(layer)]
            for _, group in layer_df.groupby("contrast_name", sort=True):
                selected[layer].update(group.sort_values("combined_score", ascending=False).head(top_contrast)["feature_id"].astype(int))
                selected[layer].update(group.sort_values("cosine_with_direction", ascending=False).head(top_contrast)["feature_id"].astype(int))
                selected[layer].update(group.sort_values("cosine_with_direction", ascending=True).head(top_contrast)["feature_id"].astype(int))
                selected[layer].update(group.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).head(top_contrast)["feature_id"].astype(int))
        if not identity.empty:
            layer_df = identity[identity["layer"].eq(layer)]
            for _, group in layer_df.groupby("identity_id", sort=True):
                selected[layer].update(group.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).head(top_identity)["feature_id"].astype(int))
    return {layer: sorted(values) for layer, values in selected.items()}


def final_token_feature_values(final_token_root: Path, layer: int, feature_ids: list[int]) -> pd.DataFrame:
    layer_dir = final_token_root / f"layer_{layer:02d}"
    index_path, value_path, _ = find_topk_arrays(layer_dir)
    indices = np.load(index_path, mmap_mode="r")
    values = np.load(value_path, mmap_mode="r")
    rows = []
    for feature_id in tqdm(feature_ids, desc=f"layer {layer:02d} final-token features", leave=False):
        mask = indices == feature_id
        row_has = mask.any(axis=1)
        prompt_rows = np.flatnonzero(row_has)
        if len(prompt_rows) == 0:
            continue
        cols = mask[row_has].argmax(axis=1)
        acts = values[prompt_rows, cols]
        for row_idx, value in zip(prompt_rows, acts):
            rows.append({"layer": layer, "feature_id": feature_id, "row_idx": int(row_idx), "final_token_feature_activation": float(value)})
    return pd.DataFrame(rows)


def select_prompt_rows(final_values: pd.DataFrame, metadata: pd.DataFrame, max_prompts_per_feature: int) -> pd.DataFrame:
    selected = []
    for (layer, feature_id), group in final_values.groupby(["layer", "feature_id"], sort=True):
        top = group.sort_values("final_token_feature_activation", ascending=False).head(max_prompts_per_feature)
        selected.append(top)
    if not selected:
        return pd.DataFrame(columns=["layer", "feature_id", "row_idx", "final_token_feature_activation"])
    rows = pd.concat(selected, ignore_index=True)
    prompt_rows = rows["row_idx"].drop_duplicates().to_numpy()
    prompt_df = metadata.iloc[prompt_rows].copy()
    prompt_df["row_idx"] = prompt_rows
    return rows.merge(prompt_df, on="row_idx", how="left")


def find_identity_span(prompt: str, form: str) -> tuple[int | None, int | None, str]:
    if not isinstance(form, str) or not form.strip():
        return None, None, "failed"
    match = re.search(re.escape(form), prompt, flags=re.I)
    if match:
        return match.start(), match.end(), "exact"
    norm_prompt = re.sub(r"[\W_]+", " ", prompt.lower())
    norm_form = re.sub(r"[\W_]+", " ", form.lower()).strip()
    norm_match = re.search(re.escape(norm_form), norm_prompt)
    if not norm_match:
        return None, None, "failed"
    prefix = norm_prompt[: norm_match.start()]
    compact_prefix = re.sub(r"\s+", "", prefix)
    compact_form = re.sub(r"\s+", "", norm_form)
    compact_prompt = re.sub(r"\s+", "", prompt.lower())
    compact_start = compact_prompt.find(compact_prefix + compact_form)
    if compact_start < 0:
        return None, None, "normalized"
    start = len(compact_prefix)
    return start, start + len(compact_form), "normalized"


def token_str(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def encode_selected_features(hidden: torch.Tensor, sae, selected_features: list[int]) -> torch.Tensor:
    features = torch.as_tensor(selected_features, device=hidden.device, dtype=torch.long)
    x = hidden
    if sae.b_dec is not None:
        x = x - sae.b_dec
    w = sae.w_enc[:, features]
    acts = x @ w
    if sae.b_enc is not None:
        acts = acts + sae.b_enc[features]
    return torch.relu(acts)


def append_layer_outputs(layer_dir: Path, token_rows: list[dict[str, object]]) -> None:
    layer_dir.mkdir(parents=True, exist_ok=True)
    if not token_rows:
        return
    token_path = layer_dir / "token_feature_activations.csv"
    pd.DataFrame(token_rows).to_csv(token_path, mode="a", header=not token_path.exists(), index=False)


def main() -> None:
    args = parse_args()
    start_all = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    layers = parse_int_list(args.layers)
    explicit_features = parse_int_list(args.features)
    metadata = pd.read_csv(args.identity_csv, keep_default_na=False).reset_index(drop=True)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "token_level").mkdir(parents=True, exist_ok=True)
    final_token_root = find_final_token_root(args.analysis_dir)
    features_by_layer = select_features(args.analysis_dir, layers, explicit_features, args.top_features_per_contrast, args.top_features_per_identity)
    (args.output_dir / "token_level" / "selected_features.json").write_text(json.dumps(features_by_layer, indent=2) + "\n")

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map="auto")
    model.eval()

    for layer in tqdm(layers, desc="layers"):
        layer_start = time.perf_counter()
        feature_ids = features_by_layer.get(layer, [])
        if not feature_ids:
            print(f"Layer {layer:02d}: no selected features; skipping.")
            continue
        layer_dir = args.output_dir / "token_level" / f"layer_{layer:02d}"
        if layer_dir.exists() and any(layer_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"{layer_dir} exists. Pass --overwrite to replace it.")
        if layer_dir.exists() and args.overwrite:
            shutil.rmtree(layer_dir)
        layer_dir.mkdir(parents=True, exist_ok=True)
        final_values = final_token_feature_values(final_token_root, layer, feature_ids)
        selected_prompts = select_prompt_rows(final_values, metadata, args.max_prompts_per_feature)
        prompt_df = selected_prompts.drop_duplicates("row_idx").sort_values("row_idx").reset_index(drop=True)
        if prompt_df.empty:
            print(f"Layer {layer:02d}: no prompts selected; skipping.")
            continue
        hidden_dim = model.config.hidden_size
        sae = load_sae(args.sae_dir, layer, hidden_dim, device, dtype)
        feature_to_col = {feature_id: i for i, feature_id in enumerate(feature_ids)}
        final_lookup = final_values.set_index(["feature_id", "row_idx"])["final_token_feature_activation"].to_dict()
        token_rows: list[dict[str, object]] = []
        top_token_rows: list[dict[str, object]] = []
        print(f"Layer {layer:02d}: extracting {len(feature_ids)} features over {len(prompt_df)} prompts")
        for start_idx in tqdm(range(0, len(prompt_df), args.batch_size), desc=f"layer {layer:02d} batches", leave=False):
            batch_df = prompt_df.iloc[start_idx:start_idx + args.batch_size].copy()
            prompts = batch_df["prompt"].astype(str).tolist()
            encoded = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_length, return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping").cpu().numpy()
            input_ids = encoded["input_ids"].cpu().numpy()
            attention_mask = encoded["attention_mask"].cpu().numpy()
            model_inputs = {key: val.to(model.device) for key, val in encoded.items()}
            with torch.inference_mode():
                outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
                hidden = outputs.hidden_states[layer].to(device=device, dtype=dtype)
                acts = encode_selected_features(hidden, sae, feature_ids).float().cpu().numpy()
            for batch_pos, meta in enumerate(batch_df.itertuples(index=False)):
                prompt = str(meta.prompt)
                span_start, span_end, span_status = find_identity_span(prompt, str(meta.form_used))
                valid_len = int(attention_mask[batch_pos].sum())
                for feature_id in feature_ids:
                    col = feature_to_col[feature_id]
                    vals = acts[batch_pos, :valid_len, col]
                    if len(vals) == 0:
                        continue
                    max_idx = int(vals.argmax())
                    max_val = float(vals[max_idx])
                    token_span_flags = []
                    for token_idx in range(valid_len):
                        start_char, end_char = offsets[batch_pos, token_idx]
                        is_span = bool(span_start is not None and end_char > span_start and start_char < span_end)
                        token_span_flags.append(is_span)
                    span_vals = vals[np.array(token_span_flags, dtype=bool)] if any(token_span_flags) else np.array([], dtype=np.float32)
                    final_value = float(final_lookup.get((feature_id, int(meta.row_idx)), 0.0))
                    max_span = float(span_vals.max()) if len(span_vals) else 0.0
                    mean_span = float(span_vals.mean()) if len(span_vals) else 0.0
                    if max_val > 0 and max_span >= 0.7 * max_val:
                        localization = "identity_span_local"
                    elif max_val > 0 and final_value >= 0.7 * max_val:
                        localization = "final_token_integrated"
                    elif max_val > 0 and not token_span_flags[max_idx]:
                        localization = "template_context"
                    else:
                        localization = "diffuse_or_unclear"
                    for token_idx in range(valid_len):
                        start_char, end_char = offsets[batch_pos, token_idx]
                        row = {
                            "layer": layer,
                            "feature_id": feature_id,
                            "prompt_id": meta.prompt_id,
                            "prompt": prompt,
                            "token_idx": token_idx,
                            "token_str": token_str(tokenizer, int(input_ids[batch_pos, token_idx])),
                            "token_start_char": int(start_char),
                            "token_end_char": int(end_char),
                            "token_feature_activation": float(vals[token_idx]),
                            "token_rank_within_prompt": int((-vals).argsort().tolist().index(token_idx) + 1),
                            "is_top_token_for_feature": token_idx == max_idx,
                            "is_identity_span_token": token_span_flags[token_idx],
                            "identity_span_match_status": span_status,
                            "identity_id": meta.identity_id,
                            "canonical_label": meta.canonical_label,
                            "axis": meta.axis,
                            "family": meta.family,
                            "template_id": meta.template_id,
                            "required_form": meta.required_form,
                            "form_used": meta.form_used,
                            "final_token_feature_activation": final_value,
                            "max_token_activation": max_val,
                            "max_identity_span_activation": max_span,
                            "mean_identity_span_activation": mean_span,
                            "token_idx_of_max_activation": max_idx,
                            "token_str_of_max_activation": token_str(tokenizer, int(input_ids[batch_pos, max_idx])),
                            "feature_localization_type": localization,
                        }
                        token_rows.append(row)
                    top_token_rows.append({
                        "layer": layer,
                        "feature_id": feature_id,
                        "prompt_id": meta.prompt_id,
                        "token_idx": max_idx,
                        "token_str": token_str(tokenizer, int(input_ids[batch_pos, max_idx])),
                        "activation": max_val,
                        "prompt_snippet": prompt[:240],
                        "identity_id": meta.identity_id,
                        "axis": meta.axis,
                        "is_identity_span_token": token_span_flags[max_idx],
                    })
            append_layer_outputs(layer_dir, token_rows)
            token_rows.clear()
        top_token_rows = sorted(top_token_rows, key=lambda row: (row["feature_id"], -row["activation"]))
        pd.DataFrame(top_token_rows).groupby("feature_id", group_keys=False).head(200).to_csv(layer_dir / "feature_top_tokens.csv", index=False)
        (layer_dir / "run_config.json").write_text(json.dumps({
            "layer": layer,
            "features": feature_ids,
            "n_prompts": len(prompt_df),
            "hf_hidden_states_note": "Uses outputs.hidden_states[layer]; hidden_states[24] is post-block-24 residual under HF Llama convention.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        print(f"Layer {layer:02d}: token extraction complete in {elapsed(layer_start)}")
    print(f"Token-level SAE extraction complete in {elapsed(start_all)}")


if __name__ == "__main__":
    main()
