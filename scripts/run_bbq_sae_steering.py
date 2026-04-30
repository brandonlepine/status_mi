#!/usr/bin/env python3
"""Run BBQ causal steering experiments with SAE decoder directions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
DEFAULT_OUTPUT = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/steering")


@dataclass
class FeatureSet:
    layer: int
    mode: str
    set_id: str
    feature_ids: list[int]
    signs: list[float]
    roles: list[str]
    contrast_name: str
    axis: str
    control_type: str = "kept_feature"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BBQ SAE steering.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sae_dir", type=Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--prepared_data", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--triage_csv", type=Path, default=DEFAULT_TRIAGE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default="8,16,24,32")
    parser.add_argument("--alphas", default="-3,-2,-1,-0.5,0.5,1,2,3")
    parser.add_argument("--feature_set_modes", default="per_feature,per_contrast_topk,role_bundle")
    parser.add_argument("--top_k_per_contrast", default="5,10,20,50")
    parser.add_argument("--scoring_mode", default="answer_logprob", choices=["answer_logprob", "first_token"])
    parser.add_argument("--intervention_positions", default="final_prompt_token,target_identity_last_token,nontarget_identity_last_token,stereotype_language_last_token")
    parser.add_argument("--intervention_modes", default="add_vector", help="Comma-separated add_vector,ablate_projection.")
    parser.add_argument("--include_unmapped", action="store_true")
    parser.add_argument("--no_normalize_features", action="store_true")
    parser.add_argument("--max_feature_sets", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--disable_controls", action="store_true", help="Skip sign-flip/random controls for quick smoke tests.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_every_examples", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def setup(output_dir: Path, overwrite: bool, resume: bool) -> logging.Logger:
    if output_dir.exists() and any(output_dir.iterdir()) and overwrite and not resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "results_parts").mkdir(exist_ok=True)
    logger = logging.getLogger("bbq_steering")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(output_dir / "logs" / "steering.log")]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def bool_series(series: pd.Series) -> pd.Series:
    return series if series.dtype == bool else series.astype(str).str.lower().isin(["true", "1", "yes"])


def priority_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "intervention_priority" in out.columns:
        out["_priority"] = out["intervention_priority"].map({"high": 0, "medium": 1, "low": 2}).fillna(9)
    else:
        out["_priority"] = 9
    for col in ["role_confidence", "max_abs_cohens_d", "combined_score"]:
        if col not in out.columns:
            out[col] = 0.0
    return out.sort_values(["_priority", "role_confidence", "max_abs_cohens_d", "combined_score"], ascending=[True, False, False, False])


def feature_sign(row: pd.Series) -> float:
    side = str(row.get("direction_side", "")).lower()
    if side in {"negative", "minus", "b", "identity_b", "-1"}:
        return -1.0
    for col in ["decoder_cosine", "cosine_with_direction", "cohens_d", "diff_mean"]:
        if col in row and pd.notna(row[col]) and float(row[col]) < 0:
            return -1.0
    return 1.0


def load_feature_sets(triage_path: Path, layers: list[int], modes: set[str], top_ks: list[int]) -> list[FeatureSet]:
    triage = pd.read_csv(triage_path, low_memory=False)
    if "keep_for_intervention" in triage.columns:
        kept = triage[bool_series(triage["keep_for_intervention"])].copy()
    else:
        kept = triage.copy()
    kept = kept[kept["layer"].isin(layers)].copy()
    if "contrast_name" not in kept.columns and "mapped_contrast_name" in kept.columns:
        kept["contrast_name"] = kept["mapped_contrast_name"]
    if "provisional_role" not in kept.columns:
        kept["provisional_role"] = ""
    if "top_axis" not in kept.columns:
        kept["top_axis"] = kept.get("axis", "")
    sets: list[FeatureSet] = []
    sorted_kept = priority_sort(kept)
    if "per_feature" in modes:
        for row in sorted_kept.drop_duplicates(["layer", "feature_id"]).itertuples(index=False):
            row_s = pd.Series(row._asdict())
            sets.append(FeatureSet(
                layer=int(row_s.layer),
                mode="per_feature",
                set_id=f"L{int(row_s.layer):02d}_feature_{int(row_s.feature_id):05d}",
                feature_ids=[int(row_s.feature_id)],
                signs=[feature_sign(row_s)],
                roles=[str(row_s.get("provisional_role", ""))],
                contrast_name=str(row_s.get("contrast_name", "")),
                axis=str(row_s.get("top_axis", "")),
            ))
    if "per_contrast_topk" in modes:
        for (layer, contrast), group in sorted_kept.dropna(subset=["contrast_name"]).groupby(["layer", "contrast_name"], sort=True):
            for role, role_group in group.groupby("provisional_role", sort=True):
                for k in top_ks:
                    top = role_group.head(k)
                    if not top.empty:
                        sets.append(FeatureSet(int(layer), "per_contrast_topk", f"L{int(layer):02d}_{contrast}_{role}_top{k}", top["feature_id"].astype(int).tolist(), [feature_sign(r) for _, r in top.iterrows()], top["provisional_role"].astype(str).tolist(), str(contrast), str(top["top_axis"].iloc[0])))
            for k in top_ks:
                top = group.head(k)
                if not top.empty:
                    sets.append(FeatureSet(int(layer), "per_contrast_topk", f"L{int(layer):02d}_{contrast}_combined_top{k}", top["feature_id"].astype(int).tolist(), [feature_sign(r) for _, r in top.iterrows()], top["provisional_role"].astype(str).tolist(), str(contrast), str(top["top_axis"].iloc[0])))
    if "role_bundle" in modes:
        for (layer, axis, role), group in sorted_kept.groupby(["layer", "top_axis", "provisional_role"], sort=True):
            if str(axis):
                sets.append(FeatureSet(int(layer), "role_bundle", f"L{int(layer):02d}_{axis}_{role}_bundle", group["feature_id"].astype(int).tolist(), [feature_sign(r) for _, r in group.iterrows()], group["provisional_role"].astype(str).tolist(), "", str(axis)))
    if "provisional_role" in triage.columns:
        artifacts = triage[(triage["layer"].isin(layers)) & (triage["provisional_role"].astype(str).eq("template_or_syntax_artifact"))].copy()
        if not artifacts.empty:
            artifacts = priority_sort(artifacts)
            for layer, group in artifacts.groupby("layer", sort=True):
                top = group.head(20)
                sets.append(FeatureSet(
                    int(layer),
                    "control_bundle",
                    f"L{int(layer):02d}_template_artifact_control_top20",
                    top["feature_id"].astype(int).tolist(),
                    [feature_sign(r) for _, r in top.iterrows()],
                    top["provisional_role"].astype(str).tolist(),
                    "",
                    str(top["top_axis"].iloc[0]) if "top_axis" in top.columns and not top.empty else "",
                    "template_artifact_features",
                ))
    return sets


def section_spans(prompt: str, row: pd.Series) -> dict[str, tuple[int, int]]:
    spans = {}
    for key in ["context", "question"]:
        idx = prompt.find(str(row.get(key, "")))
        if idx >= 0:
            spans[key] = (idx, idx + len(str(row.get(key, ""))))
    for i in range(3):
        marker = f"{chr(65 + i)}. {row.get(f'ans{i}', '')}"
        idx = prompt.find(marker)
        if idx >= 0:
            spans[f"ans{i}"] = (idx, idx + len(marker))
    return spans


def find_spans(prompt: str, terms: list[str]) -> list[tuple[int, int]]:
    lower = prompt.lower()
    spans = []
    for term in terms:
        term = str(term or "").strip().lower()
        if not term:
            continue
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(term), lower))
    return spans


def stereotype_terms(question: str) -> list[str]:
    stop = {"who", "what", "when", "where", "why", "how", "was", "were", "is", "are", "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "likely", "had", "has", "have"}
    return [tok for tok in re.findall(r"[A-Za-z][A-Za-z'-]+", str(question).lower()) if tok not in stop and len(tok) > 2]


def positions_for(tokenizer, prompt: str, row: pd.Series, max_length: int, position_name: str) -> list[int]:
    encoded = tokenizer(prompt, truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    content = [i for i, (s, e) in enumerate(offsets) if e > s]
    if not content:
        return [len(offsets) - 1]
    final_pos = content[-1]
    spans = []
    if position_name == "final_prompt_token":
        return [final_pos]
    if position_name in {"target_identity_last_token", "all_identity_tokens"}:
        target_idx = row.get("target_answer_idx", np.nan)
        terms = [row.get("target_identity_label", "")]
        if pd.notna(target_idx):
            terms.append(row.get(f"ans{int(target_idx)}", ""))
        spans.extend(find_spans(prompt, terms))
    if position_name in {"nontarget_identity_last_token", "all_identity_tokens"}:
        nt_idx = row.get("nontarget_answer_idx", np.nan)
        terms = [row.get("nontarget_identity_label", "")]
        if pd.notna(nt_idx):
            terms.append(row.get(f"ans{int(nt_idx)}", ""))
        spans.extend(find_spans(prompt, terms))
    if position_name in {"stereotype_language_last_token", "all_stereotype_language_tokens"}:
        spans.extend(find_spans(prompt, stereotype_terms(str(row.get("question", "")))))
    pos = [i for i, (s, e) in enumerate(offsets) if e > s and any(e > ss and s < ee for ss, ee in spans)]
    if position_name.endswith("last_token") and pos:
        return [max(pos)]
    return pos or [final_pos]


def first_token_ids(tokenizer, answers: list[str]) -> list[int]:
    ids = []
    for answer in answers:
        toks = tokenizer(" " + str(answer), add_special_tokens=False)["input_ids"]
        ids.append(int(toks[0]) if toks else tokenizer.eos_token_id)
    return ids


def install_hook(model, layer: int, vector: torch.Tensor, positions: list[int], alpha: float, mode: str):
    if layer <= 0:
        raise ValueError("Steering hooks require layer >= 1 because LkR maps to post-block k.")
    module = model.model.layers[layer - 1]
    pos = torch.as_tensor(positions, dtype=torch.long, device=vector.device)
    vec = vector.to(device=vector.device, dtype=next(model.parameters()).dtype)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edit = hidden.clone()
        selected = edit[:, pos, :]
        if mode == "ablate_projection":
            unit = vec / vec.norm().clamp_min(1e-9)
            selected = selected - alpha * (selected @ unit).unsqueeze(-1) * unit
        else:
            selected = selected + alpha * vec
        edit[:, pos, :] = selected
        if isinstance(output, tuple):
            return (edit, *output[1:])
        return edit

    return module.register_forward_hook(hook)


def score_first_token(model, tokenizer, prompt: str, answers: list[str], max_length: int, hook_fn: Callable[[], object] | None = None) -> np.ndarray:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    handle = hook_fn() if hook_fn else None
    try:
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits[0, -1]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            return np.array([float(log_probs[token_id]) for token_id in first_token_ids(tokenizer, answers)], dtype=np.float32)
    finally:
        if handle is not None:
            handle.remove()


def score_answer_logprob(model, tokenizer, prompt: str, answers: list[str], max_length: int, hook_fn: Callable[[], object] | None = None) -> np.ndarray:
    scores = []
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    prompt_len = len(prompt_ids)
    handle = hook_fn() if hook_fn else None
    try:
        for answer in answers:
            text = prompt + " " + str(answer)
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            input_ids = encoded["input_ids"]
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False).logits[:, :-1, :].float()
            labels = input_ids[:, 1:]
            start = max(0, min(prompt_len - 1, labels.shape[1] - 1))
            token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            scores.append(float(token_logprobs[0, start:].sum()))
    finally:
        if handle is not None:
            handle.remove()
    return np.array(scores, dtype=np.float32)


def make_vector(sae, feature_set: FeatureSet, normalize: bool, device: torch.device, random_direction: bool = False) -> torch.Tensor:
    if random_direction:
        vec = torch.randn(sae.w_dec.shape[1], dtype=torch.float32)
        return (vec / vec.norm().clamp_min(1e-9)).to(device)
    dec = sae.w_dec[feature_set.feature_ids].float()
    if normalize:
        dec = dec / dec.norm(dim=1, keepdim=True).clamp_min(1e-9)
    signs = torch.as_tensor(feature_set.signs, dtype=torch.float32).unsqueeze(1)
    vec = (dec * signs).mean(dim=0)
    return vec.to(device) / vec.norm().clamp_min(1e-9)


def make_random_feature_vector(sae, n_features: int, normalize: bool, device: torch.device, seed: int) -> tuple[torch.Tensor, list[int]]:
    rng = np.random.default_rng(seed)
    random_ids = rng.choice(np.arange(sae.w_dec.shape[0]), size=max(1, n_features), replace=False).astype(int).tolist()
    dec = sae.w_dec[random_ids].float()
    if normalize:
        dec = dec / dec.norm(dim=1, keepdim=True).clamp_min(1e-9)
    vec = dec.mean(dim=0)
    return (vec.to(device) / vec.norm().clamp_min(1e-9), random_ids)


def completed_jobs(path: Path, logger: logging.Logger | None = None) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    bad_lines = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            job = payload.get("job_id")
            if job:
                done.add(str(job))
            else:
                bad_lines.append((line_no, line))
        except json.JSONDecodeError:
            bad_lines.append((line_no, line))
    if bad_lines:
        backup = path.with_suffix(path.suffix + ".malformed")
        backup.write_text("\n".join(f"{line_no}: {line}" for line_no, line in bad_lines) + "\n")
        clean_lines = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if payload.get("job_id"):
                    clean_lines.append(json.dumps(payload))
            except json.JSONDecodeError:
                continue
        path.write_text("\n".join(clean_lines) + ("\n" if clean_lines else ""))
        if logger:
            logger.warning("Ignored %d malformed completed-job lines; backed them up to %s", len(bad_lines), backup)
    return done


def job_id(parts: list[object]) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def choice_value(values: np.ndarray, idx: object) -> float:
    if pd.isna(idx):
        return float("nan")
    i = int(idx)
    return float(values[i]) if 0 <= i < len(values) else float("nan")


def row_metrics(base: np.ndarray, inter: np.ndarray, row: pd.Series) -> dict[str, object]:
    deltas = inter - base
    pred_base = int(base.argmax())
    pred_inter = int(inter.argmax())
    stereo_idx = row.get("stereotyped_answer_idx", np.nan)
    unknown_idx = row.get("unknown_answer_idx", np.nan)
    correct_idx = row.get("correct_answer_idx", np.nan)
    if str(row.get("context_condition", "")) == "ambig":
        margin_base = choice_value(base, stereo_idx) - choice_value(base, unknown_idx)
        margin_inter = choice_value(inter, stereo_idx) - choice_value(inter, unknown_idx)
    else:
        correct = choice_value(base, correct_idx)
        others = [base[i] for i in range(3) if i != int(correct_idx)] if pd.notna(correct_idx) else [float("nan")]
        margin_base = correct - float(np.nanmax(others))
        correct_i = choice_value(inter, correct_idx)
        others_i = [inter[i] for i in range(3) if i != int(correct_idx)] if pd.notna(correct_idx) else [float("nan")]
        margin_inter = correct_i - float(np.nanmax(others_i))
    return {
        "ans0_logprob_base": float(base[0]), "ans1_logprob_base": float(base[1]), "ans2_logprob_base": float(base[2]),
        "ans0_logprob_intervened": float(inter[0]), "ans1_logprob_intervened": float(inter[1]), "ans2_logprob_intervened": float(inter[2]),
        "ans0_delta": float(deltas[0]), "ans1_delta": float(deltas[1]), "ans2_delta": float(deltas[2]),
        "stereotyped_logprob_base": choice_value(base, stereo_idx),
        "stereotyped_logprob_intervened": choice_value(inter, stereo_idx),
        "stereotyped_delta": choice_value(deltas, stereo_idx),
        "nonstereotyped_delta": choice_value(deltas, row.get("nonstereotyped_answer_idx", np.nan)),
        "unknown_delta": choice_value(deltas, unknown_idx),
        "correct_delta": choice_value(deltas, correct_idx),
        "bias_margin_base": float(margin_base),
        "bias_margin_intervened": float(margin_inter),
        "bias_margin_delta": float(margin_inter - margin_base),
        "predicted_base": pred_base,
        "predicted_intervened": pred_inter,
        "prediction_changed": pred_base != pred_inter,
        "correct_base": bool(pd.notna(correct_idx) and pred_base == int(correct_idx)),
        "correct_intervened": bool(pd.notna(correct_idx) and pred_inter == int(correct_idx)),
    }


def write_part(rows: list[dict[str, object]], output_dir: Path, part_idx: int) -> Path:
    df = pd.DataFrame(rows)
    path = output_dir / "results_parts" / f"part_{part_idx:05d}.parquet"
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)
        return path


def main() -> None:
    args = parse_args()
    logger = setup(args.output_dir, args.overwrite, args.resume)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    layers = parse_ints(args.layers)
    alphas = parse_floats(args.alphas)
    modes = {part.strip() for part in args.feature_set_modes.split(",") if part.strip()}
    top_ks = parse_ints(args.top_k_per_contrast)
    positions = [part.strip() for part in args.intervention_positions.split(",") if part.strip()]
    intervention_modes = [part.strip() for part in args.intervention_modes.split(",") if part.strip()]
    config = vars(args).copy()
    config.update({k: str(v) for k, v in config.items() if isinstance(v, Path)})
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "steering_config.json").write_text(json.dumps(config, indent=2) + "\n")

    prepared = read_table(args.prepared_data)
    if not args.include_unmapped and "mapped_contrast_confidence" in prepared.columns:
        prepared = prepared[prepared["mapped_contrast_confidence"].isin(["exact", "alias", "fallback_axis"])].copy()
    if args.max_examples:
        prepared = prepared.head(args.max_examples).copy()
    feature_sets = load_feature_sets(args.triage_csv, layers, modes, top_ks)
    feature_sets = [fs for fs in feature_sets if fs.layer in layers]
    if args.max_feature_sets:
        feature_sets = feature_sets[: args.max_feature_sets]
    expected = len(prepared) * len(feature_sets) * len(alphas) * len(positions) * len(intervention_modes)
    logger.info("Prepared examples: %d; feature sets: %d; expected steering jobs: %d", len(prepared), len(feature_sets), expected)
    if expected == 0:
        raise ValueError("No steering jobs to run.")
    pd.DataFrame([fs.__dict__ for fs in feature_sets]).to_csv(args.output_dir / "steering_manifest.csv", index=False)
    done_path = args.output_dir / "completed_jobs.jsonl"
    done = completed_jobs(done_path, logger) if args.resume else set()
    logger.info("Resume metadata loaded before model weights: %d completed jobs", len(done))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    hidden_dim = model.config.hidden_size
    score_fn = score_answer_logprob if args.scoring_mode == "answer_logprob" else score_first_token
    part_rows: list[dict[str, object]] = []
    part_idx = len(list((args.output_dir / "results_parts").glob("part_*")))
    sae_cache = {}
    vector_cache = {}
    n_done = 0

    for fs in tqdm(feature_sets, desc="feature sets"):
        if fs.layer not in sae_cache:
            sae_cache[fs.layer] = load_sae(args.sae_dir, fs.layer, hidden_dim, device, dtype)
        sae = sae_cache[fs.layer]
        for _, row in tqdm(prepared.iterrows(), total=len(prepared), desc=fs.set_id[:40], leave=False):
            row_s = pd.Series(row)
            answers = [str(row_s.get(f"ans{i}", "")) for i in range(3)]
            prompt = str(row_s["prompt"])
            base = score_fn(model, tokenizer, prompt, answers, 512)
            for alpha in alphas:
                for pos_name in positions:
                    pos = positions_for(tokenizer, prompt, row_s, 512, pos_name)
                    for mode in intervention_modes:
                        jid = job_id([row_s["bbq_uid"], fs.layer, fs.set_id, alpha, pos_name, mode, args.scoring_mode])
                        if jid in done:
                            continue
                        start = time.perf_counter()
                        vector_key = (fs.set_id, fs.layer, "kept")
                        if vector_key not in vector_cache:
                            vector_cache[vector_key] = make_vector(sae, fs, not args.no_normalize_features, device)
                        vector = vector_cache[vector_key]
                        hook_fn = lambda v=vector, p=pos, a=alpha, m=mode: install_hook(model, fs.layer, v, p, a, m)
                        inter = score_fn(model, tokenizer, prompt, answers, 512, hook_fn=hook_fn)
                        out = {
                            "bbq_uid": row_s["bbq_uid"],
                            "layer": fs.layer,
                            "alpha": alpha,
                            "intervention_mode": mode,
                            "intervention_position": pos_name,
                            "feature_set_mode": fs.mode,
                            "feature_set_id": fs.set_id,
                            "feature_ids_json": json.dumps(fs.feature_ids),
                            "feature_signs_json": json.dumps(fs.signs),
                            "feature_roles_json": json.dumps(fs.roles),
                            "mapped_contrast_name": row_s.get("mapped_contrast_name", ""),
                            "feature_contrast_name": fs.contrast_name,
                            "axis_mapped": row_s.get("axis_mapped", ""),
                            "category_raw": row_s.get("category_raw", ""),
                            "context_condition": row_s.get("context_condition", ""),
                            "question_polarity": row_s.get("question_polarity", ""),
                            "stereotyped_answer_idx": row_s.get("stereotyped_answer_idx", np.nan),
                            "nonstereotyped_answer_idx": row_s.get("nonstereotyped_answer_idx", np.nan),
                            "unknown_answer_idx": row_s.get("unknown_answer_idx", np.nan),
                            "correct_answer_idx": row_s.get("correct_answer_idx", np.nan),
                            "control_type": "wrong_axis_features" if fs.control_type == "kept_feature" and fs.axis and str(row_s.get("axis_mapped", "")) and fs.axis != str(row_s.get("axis_mapped", "")) else fs.control_type,
                            "runtime_seconds": time.perf_counter() - start,
                        }
                        out.update(row_metrics(base, inter, row_s))
                        part_rows.append(out)
                        done_path.open("a").write(json.dumps({"job_id": jid, "completed_at": datetime.now(timezone.utc).isoformat()}) + "\n")
                        done.add(jid)
                        n_done += 1
                        if len(part_rows) >= min(args.save_every_examples, 500):
                            write_part(part_rows, args.output_dir, part_idx)
                            part_idx += 1
                            part_rows = []
                # Sign-flip control for the same feature set at final token only.
                if not args.disable_controls and "final_prompt_token" in positions:
                    jid = job_id([row_s["bbq_uid"], fs.layer, fs.set_id, alpha, "final_prompt_token", "sign_flip", args.scoring_mode])
                    if jid not in done:
                        pos = positions_for(tokenizer, prompt, row_s, 512, "final_prompt_token")
                        vector = -make_vector(sae, fs, not args.no_normalize_features, device)
                        inter = score_fn(model, tokenizer, prompt, answers, 512, hook_fn=lambda v=vector, p=pos, a=alpha: install_hook(model, fs.layer, v, p, a, "add_vector"))
                        out = {"bbq_uid": row_s["bbq_uid"], "layer": fs.layer, "alpha": alpha, "intervention_mode": "add_vector", "intervention_position": "final_prompt_token", "feature_set_mode": fs.mode, "feature_set_id": fs.set_id, "feature_ids_json": json.dumps(fs.feature_ids), "feature_signs_json": json.dumps([-s for s in fs.signs]), "feature_roles_json": json.dumps(fs.roles), "mapped_contrast_name": row_s.get("mapped_contrast_name", ""), "feature_contrast_name": fs.contrast_name, "axis_mapped": row_s.get("axis_mapped", ""), "category_raw": row_s.get("category_raw", ""), "context_condition": row_s.get("context_condition", ""), "question_polarity": row_s.get("question_polarity", ""), "stereotyped_answer_idx": row_s.get("stereotyped_answer_idx", np.nan), "nonstereotyped_answer_idx": row_s.get("nonstereotyped_answer_idx", np.nan), "unknown_answer_idx": row_s.get("unknown_answer_idx", np.nan), "correct_answer_idx": row_s.get("correct_answer_idx", np.nan), "control_type": "sign_flip", "runtime_seconds": 0.0}
                        out.update(row_metrics(base, inter, row_s))
                        part_rows.append(out)
                        done_path.open("a").write(json.dumps({"job_id": jid, "completed_at": datetime.now(timezone.utc).isoformat()}) + "\n")
                    for control_name in ["random_direction_norm_matched", "random_feature_matched"]:
                        jid = job_id([row_s["bbq_uid"], fs.layer, fs.set_id, alpha, "final_prompt_token", control_name, args.scoring_mode])
                        if jid in done:
                            continue
                        pos = positions_for(tokenizer, prompt, row_s, 512, "final_prompt_token")
                        if control_name == "random_direction_norm_matched":
                            seed = int(hashlib.sha1(jid.encode()).hexdigest()[:8], 16)
                            torch.manual_seed(seed)
                            vector = make_vector(sae, fs, not args.no_normalize_features, device, random_direction=True)
                            control_feature_ids = []
                        else:
                            seed = int(hashlib.sha1(jid.encode()).hexdigest()[:8], 16)
                            vector, control_feature_ids = make_random_feature_vector(sae, len(fs.feature_ids), not args.no_normalize_features, device, seed)
                        start_control = time.perf_counter()
                        inter = score_fn(model, tokenizer, prompt, answers, 512, hook_fn=lambda v=vector, p=pos, a=alpha: install_hook(model, fs.layer, v, p, a, "add_vector"))
                        out = {"bbq_uid": row_s["bbq_uid"], "layer": fs.layer, "alpha": alpha, "intervention_mode": "add_vector", "intervention_position": "final_prompt_token", "feature_set_mode": fs.mode, "feature_set_id": fs.set_id, "feature_ids_json": json.dumps(control_feature_ids or fs.feature_ids), "feature_signs_json": json.dumps(fs.signs), "feature_roles_json": json.dumps(fs.roles), "mapped_contrast_name": row_s.get("mapped_contrast_name", ""), "feature_contrast_name": fs.contrast_name, "axis_mapped": row_s.get("axis_mapped", ""), "category_raw": row_s.get("category_raw", ""), "context_condition": row_s.get("context_condition", ""), "question_polarity": row_s.get("question_polarity", ""), "stereotyped_answer_idx": row_s.get("stereotyped_answer_idx", np.nan), "nonstereotyped_answer_idx": row_s.get("nonstereotyped_answer_idx", np.nan), "unknown_answer_idx": row_s.get("unknown_answer_idx", np.nan), "correct_answer_idx": row_s.get("correct_answer_idx", np.nan), "control_type": control_name, "runtime_seconds": time.perf_counter() - start_control}
                        out.update(row_metrics(base, inter, row_s))
                        part_rows.append(out)
                        done_path.open("a").write(json.dumps({"job_id": jid, "completed_at": datetime.now(timezone.utc).isoformat()}) + "\n")
    if part_rows:
        write_part(part_rows, args.output_dir, part_idx)
    logger.info("Completed new jobs: %d", n_done)
    logger.info("Stereotype residual direction scaffold: not run unless prior residual directions are supplied in a future extension.")


if __name__ == "__main__":
    main()
