#!/usr/bin/env python3
"""Extract token-level SAE activations for prepared BBQ prompts."""

from __future__ import annotations

import argparse
import json
import logging
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
DEFAULT_PREPARED = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet")
DEFAULT_TRIAGE = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv")
DEFAULT_OUTPUT = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/token_level_sae")

STOPWORDS = {
    "who", "what", "when", "where", "why", "how", "was", "were", "is", "are", "the", "a", "an",
    "to", "of", "in", "on", "for", "and", "or", "likely", "had", "has", "have", "did", "does",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract BBQ token-level SAE activations.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sae_dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--prepared_csv_or_parquet", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--triage_csv", type=Path, default=DEFAULT_TRIAGE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default="8,16,24,32")
    parser.add_argument("--feature_roles", default="contrast_specific_identity,identity_token_local,shared_social_feature,sentence_final_integrated")
    parser.add_argument("--include_all_kept_features", action="store_true")
    parser.add_argument("--max_features", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_every_batches", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def setup(output_dir: Path, overwrite: bool, resume: bool) -> logging.Logger:
    if output_dir.exists() and any(output_dir.iterdir()) and overwrite and not resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "token_activations").mkdir(exist_ok=True)
    logger = logging.getLogger("bbq_token_sae")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(output_dir / "logs" / "token_level_sae.log")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_features(path: Path, layers: list[int], roles: set[str], include_all_kept: bool, max_features: int | None) -> dict[int, pd.DataFrame]:
    triage = pd.read_csv(path, low_memory=False)
    if "layer" not in triage.columns or "feature_id" not in triage.columns:
        raise ValueError("triage_csv must contain layer and feature_id columns")
    if "keep_for_intervention" in triage.columns:
        triage = triage[bool_series(triage["keep_for_intervention"])].copy()
    if not include_all_kept and "provisional_role" in triage.columns:
        triage = triage[triage["provisional_role"].astype(str).isin(roles)].copy()
    sort_cols = [col for col in ["intervention_priority", "role_confidence", "max_abs_cohens_d", "combined_score"] if col in triage.columns]
    if sort_cols:
        ascending = [True if col == "intervention_priority" else False for col in sort_cols]
        priority = {"high": 0, "medium": 1, "low": 2}
        if "intervention_priority" in triage.columns:
            triage["_priority_order"] = triage["intervention_priority"].map(priority).fillna(9)
            sort_cols = ["_priority_order" if c == "intervention_priority" else c for c in sort_cols]
        triage = triage.sort_values(sort_cols, ascending=ascending)
    out = {}
    keep_cols = [
        "layer", "feature_id", "contrast_name", "mapped_contrast_name", "provisional_role",
        "top_axis", "top_identity", "direction_side", "max_abs_cohens_d", "template_artifact_score",
        "sharedness_score", "contrast_specificity_score", "role_confidence", "combined_score",
    ]
    for layer, group in triage[triage["layer"].isin(layers)].groupby("layer", sort=True):
        group = group.drop_duplicates(["layer", "feature_id"]).copy()
        if max_features:
            group = group.head(max_features)
        out[int(layer)] = group.reindex(columns=[c for c in keep_cols if c in group.columns])
    return out


def find_section_spans(prompt: str, row: pd.Series) -> dict[str, tuple[int, int]]:
    spans = {}
    for name, text in [("context", row.get("context", "")), ("question", row.get("question", ""))]:
        idx = prompt.find(str(text))
        if idx >= 0:
            spans[name] = (idx, idx + len(str(text)))
    for i in range(3):
        marker = f"{chr(65 + i)}. {row.get(f'ans{i}', '')}"
        idx = prompt.find(marker)
        if idx >= 0:
            spans[f"ans{i}"] = (idx, idx + len(marker))
    return spans


def find_all_spans(prompt: str, terms: list[str]) -> list[tuple[int, int]]:
    spans = []
    lower = prompt.lower()
    for term in terms:
        term = str(term or "").strip()
        if not term:
            continue
        for match in re.finditer(re.escape(term.lower()), lower):
            spans.append((match.start(), match.end()))
    return spans


def overlap(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(end > s and start < e for s, e in spans)


def stereotype_terms(question: str) -> list[str]:
    return [tok for tok in re.findall(r"[A-Za-z][A-Za-z'-]+", str(question).lower()) if tok not in STOPWORDS and len(tok) > 2]


def write_part(df: pd.DataFrame, path_no_suffix: Path) -> Path:
    try:
        path = path_no_suffix.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = path_no_suffix.with_suffix(".csv")
        df.to_csv(path, index=False)
        return path


def encode_selected_features(hidden: torch.Tensor, sae, feature_ids: list[int]) -> torch.Tensor:
    features = torch.as_tensor(feature_ids, device=hidden.device, dtype=torch.long)
    x = hidden
    if sae.b_dec is not None:
        x = x - sae.b_dec
    acts = x @ sae.w_enc[:, features]
    if sae.b_enc is not None:
        acts = acts + sae.b_enc[features]
    return torch.relu(acts)


def completed_batches(layer_dir: Path) -> set[int]:
    manifest = layer_dir / "manifest.csv"
    if not manifest.exists():
        return set()
    try:
        return set(pd.read_csv(manifest)["batch_id"].astype(int))
    except Exception:
        return set()


def append_manifest(layer_dir: Path, row: dict[str, object]) -> None:
    pd.DataFrame([row]).to_csv(layer_dir / "manifest.csv", mode="a", header=not (layer_dir / "manifest.csv").exists(), index=False)


def main() -> None:
    args = parse_args()
    logger = setup(args.output_dir, args.overwrite, args.resume)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    layers = parse_ints(args.layers)
    roles = {part.strip() for part in args.feature_roles.split(",") if part.strip()}
    config = vars(args).copy()
    config.update({k: str(v) for k, v in config.items() if isinstance(v, Path)})
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "token_level_config.json").write_text(json.dumps(config, indent=2) + "\n")

    prepared = read_table(args.prepared_csv_or_parquet)
    if args.max_examples:
        prepared = prepared.head(args.max_examples).copy()
    features_by_layer = load_features(args.triage_csv, layers, roles, args.include_all_kept_features, args.max_features)
    feature_count = sum(len(df) for df in features_by_layer.values())
    logger.info("Prepared examples: %d", len(prepared))
    logger.info("Selected kept SAE features: %d across layers %s", feature_count, sorted(features_by_layer))
    if feature_count == 0:
        raise ValueError("No kept features selected from triage_csv.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    hidden_dim = model.config.hidden_size

    summary_rows = []
    for layer in tqdm(layers, desc="layers"):
        feature_df = features_by_layer.get(layer, pd.DataFrame())
        if feature_df.empty:
            logger.warning("Layer %02d has no selected features; skipping.", layer)
            continue
        feature_ids = feature_df["feature_id"].astype(int).tolist()
        feature_meta = feature_df.set_index("feature_id").to_dict("index")
        layer_dir = args.output_dir / "token_activations" / f"layer_{layer:02d}"
        if layer_dir.exists() and args.overwrite and not args.resume:
            shutil.rmtree(layer_dir)
        layer_dir.mkdir(parents=True, exist_ok=True)
        done = completed_batches(layer_dir) if args.resume else set()
        sae = load_sae(args.sae_dir, layer, hidden_dim, device, dtype)
        start_time = time.perf_counter()
        for batch_id, start in enumerate(tqdm(range(0, len(prepared), args.batch_size), desc=f"layer {layer:02d} batches", leave=False)):
            if batch_id in done:
                continue
            batch = prepared.iloc[start:start + args.batch_size].copy()
            prompts = batch["prompt"].astype(str).tolist()
            encoded = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_length, return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping").cpu().numpy()
            input_ids = encoded["input_ids"].cpu().numpy()
            attention_mask = encoded["attention_mask"].cpu().numpy()
            model_inputs = {key: val.to(model.device) for key, val in encoded.items()}
            with torch.inference_mode():
                outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
                hidden = outputs.hidden_states[layer].to(device=device, dtype=dtype)
                acts = encode_selected_features(hidden, sae, feature_ids).float().cpu().numpy()
            rows = []
            for batch_pos, row in enumerate(batch.itertuples(index=False)):
                row_s = pd.Series(row._asdict())
                prompt = str(row_s["prompt"])
                section_spans = find_section_spans(prompt, row_s)
                target_spans = find_all_spans(prompt, [row_s.get("target_identity_label", ""), row_s.get(f"ans{int(row_s.get('target_answer_idx'))}", "")] if pd.notna(row_s.get("target_answer_idx")) else [row_s.get("target_identity_label", "")])
                nontarget_spans = find_all_spans(prompt, [row_s.get("nontarget_identity_label", ""), row_s.get(f"ans{int(row_s.get('nontarget_answer_idx'))}", "")] if pd.notna(row_s.get("nontarget_answer_idx")) else [row_s.get("nontarget_identity_label", "")])
                unknown_spans = find_all_spans(prompt, [row_s.get(f"ans{int(row_s.get('unknown_answer_idx'))}", "")] if pd.notna(row_s.get("unknown_answer_idx")) else [])
                stereo_terms = stereotype_terms(str(row_s.get("question", "")))
                stereotype_spans = find_all_spans(prompt, stereo_terms)
                valid_len = int(attention_mask[batch_pos].sum())
                final_idx = valid_len - 1
                for feature_col, feature_id in enumerate(feature_ids):
                    vals = acts[batch_pos, :valid_len, feature_col]
                    if not np.any(vals > 0):
                        continue
                    positive_positions = np.flatnonzero(vals > 0)
                    ranks = pd.Series(vals).rank(method="first", ascending=False).astype(int).to_numpy()
                    meta = feature_meta.get(feature_id, {})
                    for token_idx in positive_positions:
                        start_char, end_char = map(int, offsets[batch_pos, token_idx])
                        option_idx = next((i for i in range(3) if f"ans{i}" in section_spans and overlap(start_char, end_char, [section_spans[f"ans{i}"]])), None)
                        rows.append({
                            "bbq_uid": row_s["bbq_uid"],
                            "layer": layer,
                            "feature_id": int(feature_id),
                            "token_idx": int(token_idx),
                            "token_str": tokenizer.decode([int(input_ids[batch_pos, token_idx])], clean_up_tokenization_spaces=False),
                            "token_start_char": start_char,
                            "token_end_char": end_char,
                            "feature_activation": float(vals[token_idx]),
                            "feature_rank_within_prompt": int(ranks[token_idx]),
                            "is_target_identity_token": overlap(start_char, end_char, target_spans),
                            "is_nontarget_identity_token": overlap(start_char, end_char, nontarget_spans),
                            "is_any_identity_token": overlap(start_char, end_char, target_spans + nontarget_spans),
                            "is_stereotype_language_token": overlap(start_char, end_char, stereotype_spans),
                            "is_question_token": "question" in section_spans and overlap(start_char, end_char, [section_spans["question"]]),
                            "is_context_token": "context" in section_spans and overlap(start_char, end_char, [section_spans["context"]]),
                            "is_answer_option_token": option_idx is not None,
                            "answer_option_idx": option_idx,
                            "is_unknown_answer_token": overlap(start_char, end_char, unknown_spans),
                            "is_final_prompt_token": int(token_idx) == final_idx,
                            "stereotype_terms": ";".join(stereo_terms),
                            "context_condition": row_s.get("context_condition", ""),
                            "question_polarity": row_s.get("question_polarity", ""),
                            "axis_mapped": row_s.get("axis_mapped", ""),
                            "category_raw": row_s.get("category_raw", ""),
                            "mapped_contrast_name": row_s.get("mapped_contrast_name", ""),
                            "feature_contrast_name": meta.get("contrast_name", meta.get("mapped_contrast_name", "")),
                            "feature_role": meta.get("provisional_role", ""),
                            "feature_top_axis": meta.get("top_axis", ""),
                            "feature_top_identity": meta.get("top_identity", ""),
                            "target_identity_id": row_s.get("target_identity_id", ""),
                            "nontarget_identity_id": row_s.get("nontarget_identity_id", ""),
                            "stereotyped_answer_idx": row_s.get("stereotyped_answer_idx", np.nan),
                            "nonstereotyped_answer_idx": row_s.get("nonstereotyped_answer_idx", np.nan),
                            "unknown_answer_idx": row_s.get("unknown_answer_idx", np.nan),
                            "correct_answer_idx": row_s.get("correct_answer_idx", np.nan),
                        })
            part_path = write_part(pd.DataFrame(rows), layer_dir / f"part_{batch_id:05d}")
            append_manifest(layer_dir, {"batch_id": batch_id, "start_row": start, "n_examples": len(batch), "n_rows": len(rows), "path": str(part_path), "elapsed_seconds": time.perf_counter() - start_time})

        part_files = sorted(layer_dir.glob("part_*.parquet")) + sorted(layer_dir.glob("part_*.csv"))
        frames = [read_table(path) for path in part_files if path.stat().st_size > 0]
        if frames:
            layer_long = pd.concat(frames, ignore_index=True)
            for (feature_id, group) in layer_long.groupby("feature_id", sort=True):
                summary_rows.append({
                    "layer": layer,
                    "feature_id": feature_id,
                    "mean_target_identity_activation": group.loc[group["is_target_identity_token"].astype(bool), "feature_activation"].mean(),
                    "mean_nontarget_identity_activation": group.loc[group["is_nontarget_identity_token"].astype(bool), "feature_activation"].mean(),
                    "mean_stereotype_language_activation": group.loc[group["is_stereotype_language_token"].astype(bool), "feature_activation"].mean(),
                    "mean_final_token_activation": group.loc[group["is_final_prompt_token"].astype(bool), "feature_activation"].mean(),
                    "max_activation_per_prompt": group.groupby("bbq_uid")["feature_activation"].max().mean(),
                    "fraction_prompts_active": group["bbq_uid"].nunique() / max(1, len(prepared)),
                    "activation_by_context_condition": json.dumps(group.groupby("context_condition")["feature_activation"].mean().to_dict()),
                    "activation_by_question_polarity": json.dumps(group.groupby("question_polarity")["feature_activation"].mean().to_dict()),
                    "activation_by_category": json.dumps(group.groupby("category_raw")["feature_activation"].mean().to_dict()),
                })
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "bbq_token_level_sae_summary.csv", index=False)
    logger.info("Token-level SAE extraction complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
