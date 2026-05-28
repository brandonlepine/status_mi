#!/usr/bin/env python3
"""Estimate feature-level and subgroup-level BBQ SAE steering effects."""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


DEFAULT_STEERING_DIR = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/steering")
DEFAULT_PREPARED = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet")
DEFAULT_TRIAGE = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv")
DEFAULT_TOKEN_LEVEL = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/token_level_sae")
DEFAULT_OUTPUT = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/feature_level_causal_analysis")

BIAS_LABEL = "Delta stereotype preference: Delta[log p(stereotyped) - log p(unknown)]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-level BBQ SAE causal effect analysis.")
    parser.add_argument("--steering_dir", type=Path, default=DEFAULT_STEERING_DIR)
    parser.add_argument("--results_csv", type=Path, default=None)
    parser.add_argument("--prepared_data", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--triage_csv", type=Path, default=DEFAULT_TRIAGE)
    parser.add_argument("--token_level_dir", type=Path, default=DEFAULT_TOKEN_LEVEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--alphas", default="-2,-1,1,2")
    parser.add_argument("--positions", default="final_prompt_token")
    parser.add_argument("--selected_positions", default=None, help="Alias for --positions.")
    parser.add_argument("--context_conditions", default="ambig")
    parser.add_argument("--grouping_levels", default="axis,contrast,identity,subgroup")
    parser.add_argument("--make_axis_reports", action="store_true")
    parser.add_argument("--make_identity_reports", action="store_true")
    parser.add_argument("--make_contrast_reports", action="store_true")
    parser.add_argument("--require_complete_alpha_grid", action="store_true")
    parser.add_argument("--selected_alphas", default=None, help="Alias for --alphas.")
    parser.add_argument("--min_examples", type=int, default=30)
    parser.add_argument("--min_examples_per_identity", type=int, default=10)
    parser.add_argument("--min_examples_per_contrast", type=int, default=20)
    parser.add_argument(
        "--min_examples_inference", type=int, default=30,
        help="Audit 2.7: minimum BBQ examples a feature-level inference unit must "
             "have (per held-out half, audit 2.5) to be tested. Units below this are "
             "dropped from the headline feature_inference table so underpowered cells "
             "do not enter FDR. Coarser than the per-(alpha,position) grid (audit 2.6).",
    )
    parser.add_argument("--top_n_features", type=int, default=25)
    # Audit 2.7: final results need >=10k resamples (a sign-flip test on n=10 has
    # min p ~= 1/1024; small bootstrap budgets give unstable CIs). --smoke caps
    # these for fast dev runs only and must not be used for headline numbers.
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--permutation_samples", type=int, default=10000)
    parser.add_argument(
        "--headline_alpha", type=float, default=None,
        help="Audit 2.6: the single pre-registered intervention amplitude at which "
             "the headline feature_inference hypothesis test is run (the unit of "
             "inference is the FEATURE, not feature x alpha). For the audit-3.1 "
             "default ablate run there is only one alpha (0.0) and this is inferred "
             "automatically; for multi-alpha steer/clamp grids you MUST pass it (the "
             "rest of the grid feeds the dose-response plots, not separate tests).",
    )
    parser.add_argument("--small_effect_threshold", type=float, default=0.002)
    parser.add_argument("--moderate_effect_threshold", type=float, default=0.005)
    parser.add_argument("--large_effect_threshold", type=float, default=0.01)
    parser.add_argument("--q_threshold", type=float, default=0.1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_strings(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def setup(output_dir: Path, overwrite: bool) -> logging.Logger:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    logger = logging.getLogger("bbq_feature_effects")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(output_dir / "logs" / "feature_level_analysis.log")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def load_results(steering_dir: Path, results_csv: Path | None) -> pd.DataFrame:
    if results_csv:
        return read_table(results_csv)
    files = sorted((steering_dir / "results_parts").glob("part_*.parquet")) + sorted((steering_dir / "results_parts").glob("part_*.csv"))
    frames = [read_table(path) for path in tqdm(files, desc="load steering parts") if path.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_table(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    try:
        df.to_parquet(output_dir / f"{stem}.parquet", index=False)
    except Exception:
        pass


def safe_name(value: object) -> str:
    text = str(value or "unknown").strip()
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    return text.strip("_") or "unknown"


def parse_json_list(value: object) -> list[object]:
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def first_role(value: object) -> str:
    values = parse_json_list(value)
    return str(values[0]) if values else ""


def idx_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def choice_value(row: pd.Series, idx_col: str, suffix: str) -> float:
    idx = idx_int(row.get(idx_col))
    if idx is None or idx not in {0, 1, 2}:
        return float("nan")
    return float(row.get(f"ans{idx}_logprob_{suffix}", np.nan))


def correct_margin(row: pd.Series, suffix: str) -> float:
    correct_idx = idx_int(row.get("correct_answer_idx"))
    vals = np.array([row.get(f"ans{i}_logprob_{suffix}", np.nan) for i in range(3)], dtype=float)
    if correct_idx is None or not np.isfinite(vals).all():
        return float("nan")
    return float(vals[correct_idx] - np.max(np.delete(vals, correct_idx)))


def _polarity_sign_value(polarity: object) -> float:
    """Audit 4.3: BBQ question_polarity ∈ {"neg", "nonneg"}.
    For neg-polarity questions the stereotyped answer is the bias-aligned one
    (e.g., "who looked nervous?" → stereotype-confirming answer is bias-aligned);
    for nonneg-polarity questions the framing flips (e.g., "who looked calm?"
    → the NON-stereotyped answer is bias-aligned). To keep
    "positive = intervention made the model more biased" consistent across both
    polarities, multiply the raw stereotype_preference_delta by +1 for neg and
    -1 for nonneg. Unknown / missing polarity → NaN so the row is skipped from
    polarity-sensitive aggregates instead of silently confounding them."""
    s = str(polarity).strip().lower() if polarity is not None else ""
    if s == "neg":
        return 1.0
    if s == "nonneg":
        return -1.0
    return float("nan")


def enrich_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feature_ids_list"] = out["feature_ids_json"].map(lambda v: [int(x) for x in parse_json_list(v)]) if "feature_ids_json" in out.columns else [[] for _ in range(len(out))]
    out["feature_roles_list"] = out["feature_roles_json"].map(lambda v: [str(x) for x in parse_json_list(v)]) if "feature_roles_json" in out.columns else [[] for _ in range(len(out))]
    if "feature_role" not in out.columns:
        out["feature_role"] = out["feature_roles_json"].map(first_role) if "feature_roles_json" in out.columns else ""
    inferred_estimate_type = np.where(out["feature_ids_list"].map(len).eq(1), "individual_feature", "feature_bundle_membership")
    if "feature_estimate_type" not in out.columns:
        out["feature_estimate_type"] = inferred_estimate_type
    else:
        out["feature_estimate_type"] = out["feature_estimate_type"].fillna(pd.Series(inferred_estimate_type, index=out.index))
    out["alpha_abs"] = pd.to_numeric(out["alpha"], errors="coerce").abs()
    out["steering_direction_label"] = np.where(pd.to_numeric(out["alpha"], errors="coerce") > 0, "feature_amplification", "feature_suppression")
    out["stereotype_preference_base"] = out.apply(lambda r: choice_value(r, "stereotyped_answer_idx", "base") - choice_value(r, "unknown_answer_idx", "base"), axis=1)
    out["stereotype_preference_intervened"] = out.apply(lambda r: choice_value(r, "stereotyped_answer_idx", "intervened") - choice_value(r, "unknown_answer_idx", "intervened"), axis=1)
    out["stereotype_preference_delta"] = out["stereotype_preference_intervened"] - out["stereotype_preference_base"]
    out["nontarget_preference_base"] = out.apply(lambda r: choice_value(r, "nonstereotyped_answer_idx", "base") - choice_value(r, "unknown_answer_idx", "base"), axis=1)
    out["nontarget_preference_intervened"] = out.apply(lambda r: choice_value(r, "nonstereotyped_answer_idx", "intervened") - choice_value(r, "unknown_answer_idx", "intervened"), axis=1)
    out["nontarget_preference_delta"] = out["nontarget_preference_intervened"] - out["nontarget_preference_base"]
    out["identity_substitution_base"] = out.apply(lambda r: choice_value(r, "nonstereotyped_answer_idx", "base") - choice_value(r, "stereotyped_answer_idx", "base"), axis=1)
    out["identity_substitution_intervened"] = out.apply(lambda r: choice_value(r, "nonstereotyped_answer_idx", "intervened") - choice_value(r, "stereotyped_answer_idx", "intervened"), axis=1)
    out["identity_substitution_delta"] = out["identity_substitution_intervened"] - out["identity_substitution_base"]
    out["correct_margin_base"] = out.apply(lambda r: correct_margin(r, "base"), axis=1)
    out["correct_margin_intervened"] = out.apply(lambda r: correct_margin(r, "intervened"), axis=1)
    out["correct_margin_delta"] = out["correct_margin_intervened"] - out["correct_margin_base"]
    out["stereotype_error_margin_base"] = out.apply(lambda r: choice_value(r, "stereotyped_answer_idx", "base") - choice_value(r, "correct_answer_idx", "base"), axis=1)
    out["stereotype_error_margin_intervened"] = out.apply(lambda r: choice_value(r, "stereotyped_answer_idx", "intervened") - choice_value(r, "correct_answer_idx", "intervened"), axis=1)
    out["stereotype_error_delta"] = out["stereotype_error_margin_intervened"] - out["stereotype_error_margin_base"]
    out["accuracy_delta"] = out["correct_intervened"].astype(float) - out["correct_base"].astype(float) if {"correct_intervened", "correct_base"}.issubset(out.columns) else np.nan
    if "stereotyped_delta" not in out.columns:
        out["stereotyped_delta"] = out.apply(lambda r: choice_value(r, "stereotyped_answer_idx", "intervened") - choice_value(r, "stereotyped_answer_idx", "base"), axis=1)
    if "unknown_delta" not in out.columns:
        out["unknown_delta"] = out.apply(lambda r: choice_value(r, "unknown_answer_idx", "intervened") - choice_value(r, "unknown_answer_idx", "base"), axis=1)
    if "nonstereotyped_delta" not in out.columns:
        out["nonstereotyped_delta"] = out.apply(lambda r: choice_value(r, "nonstereotyped_answer_idx", "intervened") - choice_value(r, "nonstereotyped_answer_idx", "base"), axis=1)
    if "correct_delta" not in out.columns:
        out["correct_delta"] = out.apply(lambda r: choice_value(r, "correct_answer_idx", "intervened") - choice_value(r, "correct_answer_idx", "base"), axis=1)
    # Audit 4.3: polarity-signed bias metrics. Positive = intervention made the
    # model more biased regardless of question_polarity; negative = less biased.
    # The unsigned `stereotype_preference_delta` / `stereotyped_delta` /
    # `nonstereotyped_delta` are preserved for diagnostic comparison, but every
    # downstream label, score, and ranking uses the signed versions.
    out["polarity_sign"] = out["question_polarity"].map(_polarity_sign_value) if "question_polarity" in out.columns else float("nan")
    out["signed_stereotype_preference_delta"] = out["polarity_sign"] * out["stereotype_preference_delta"]
    out["signed_stereotyped_delta"] = out["polarity_sign"] * out["stereotyped_delta"]
    out["signed_nonstereotyped_delta"] = out["polarity_sign"] * out["nonstereotyped_delta"]
    return out


def answer_delta_for_idx(row: pd.Series, idx: object) -> float:
    i = idx_int(idx)
    if i is None or i not in {0, 1, 2}:
        return float("nan")
    return float(row.get(f"ans{i}_logprob_intervened", np.nan) - row.get(f"ans{i}_logprob_base", np.nan))


def build_identity_records(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="identity records"):
        data = row._asdict()
        roles = [
            ("target", data.get("target_identity_id", ""), data.get("target_identity_label", data.get("target_identity_id", "")), data.get("target_answer_idx", np.nan)),
            ("nontarget", data.get("nontarget_identity_id", ""), data.get("nontarget_identity_label", data.get("nontarget_identity_id", "")), data.get("nontarget_answer_idx", np.nan)),
            ("stereotyped_identity", data.get("target_identity_id", ""), data.get("target_identity_label", data.get("target_identity_id", "")), data.get("stereotyped_answer_idx", np.nan)),
            ("nonstereotyped_identity", data.get("nontarget_identity_id", ""), data.get("nontarget_identity_label", data.get("nontarget_identity_id", "")), data.get("nonstereotyped_answer_idx", np.nan)),
        ]
        for identity_role, identity_id, identity_label, answer_idx in roles:
            if not identity_id or str(identity_id) == "nan":
                continue
            rows.append({
                "bbq_uid": data.get("bbq_uid", ""),
                "identity_id": identity_id,
                "identity_label": identity_label,
                "identity_role": identity_role,
                "answer_idx_for_identity": answer_idx,
                "is_stereotyped_identity": identity_role in {"target", "stereotyped_identity"},
                "is_correct_identity_answer": idx_int(answer_idx) == idx_int(data.get("correct_answer_idx")),
                "context_condition": data.get("context_condition", ""),
                "question_polarity": data.get("question_polarity", ""),
                "mapped_contrast_name": data.get("mapped_contrast_name", ""),
                "axis_mapped": data.get("axis_mapped", ""),
            })
    return pd.DataFrame(rows).drop_duplicates()


def merge_identity_records(expanded: pd.DataFrame, identity_records: pd.DataFrame) -> pd.DataFrame:
    if identity_records.empty:
        return pd.DataFrame()
    key_cols = ["bbq_uid", "context_condition", "question_polarity", "mapped_contrast_name", "axis_mapped"]
    merged = expanded.merge(identity_records, on=key_cols, how="inner", suffixes=("", "_identity"))
    merged["identity_answer_delta"] = merged.apply(lambda row: answer_delta_for_idx(row, row.get("answer_idx_for_identity")), axis=1)
    merged["identity_specific_bias_delta"] = merged.apply(
        lambda row: answer_delta_for_idx(row, row.get("answer_idx_for_identity")) - row.get("unknown_delta", np.nan),
        axis=1,
    )
    merged["substitution_delta"] = merged["identity_substitution_delta"]
    return merged


def expand_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="expand feature rows"):
        data = row._asdict()
        feature_ids = data.pop("feature_ids_list", [])
        roles = data.pop("feature_roles_list", [])
        estimate_type = str(data.get("feature_estimate_type", "")) or ("individual_feature" if len(feature_ids) == 1 else "feature_bundle_membership")
        for i, feature_id in enumerate(feature_ids):
            expanded = data.copy()
            expanded["feature_id"] = int(feature_id)
            expanded["feature_role"] = roles[i] if i < len(roles) else data.get("feature_role", "")
            expanded["feature_estimate_type"] = estimate_type
            expanded["n_features_in_set"] = len(feature_ids)
            rows.append(expanded)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_pvalue(values: np.ndarray, n: int, rng: np.random.Generator) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    if observed == 0:
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(n, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (n + 1))


def fdr_bh(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    ranks = np.arange(1, m + 1)
    q = (valid.to_numpy() * m / ranks)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[valid.index] = np.clip(q, 0, 1)
    return out


def effect_size_label(delta: float, args: argparse.Namespace) -> str:
    ad = abs(delta)
    if ad >= args.large_effect_threshold:
        return "large"
    if ad >= args.moderate_effect_threshold:
        return "moderate"
    if ad >= args.small_effect_threshold:
        return "small"
    return "tiny"


def effect_label(row: pd.Series, args: argparse.Namespace) -> str:
    """Audit 4.3 (closed 2026-05-27): the bias direction tests use the
    polarity-signed metric. `bias_amplifying` and `bias_reducing_*` now mean
    the same thing across both neg-polarity and nonneg-polarity questions:
    the intervention shifted the model's preference toward the stereotype-
    aligned answer (or away from it). Prior code branched on the unsigned
    `mean_stereotype_preference_delta`, which inverts under polarity flip.
    """
    q = row.get("q_value_fdr", np.nan)
    reliable = pd.notna(q) and float(q) < args.q_threshold
    threshold = args.small_effect_threshold
    identity_delta = abs(float(row.get("mean_identity_answer_delta", 0) or 0))
    bias_delta = float(row.get("mean_signed_stereotype_preference_delta", 0) or 0)
    if str(row.get("context_condition", "")) == "disambig":
        if reliable and (row.get("mean_correct_margin_delta", 0) < -threshold or row.get("mean_correct_delta", 0) < -threshold or row.get("mean_accuracy_delta", 0) < 0):
            return "capability_degrading"
        if identity_delta >= threshold and (abs(bias_delta) < threshold or not reliable):
            return "identity_only"
        return "no_reliable_effect" if not reliable or abs(bias_delta) < threshold else "mixed_or_unclear"
    if not reliable or abs(bias_delta) < threshold:
        if identity_delta >= threshold:
            return "identity_only"
        return "no_reliable_effect"
    if bias_delta > threshold:
        return "bias_amplifying"
    if bias_delta < -threshold and row.get("mean_unknown_delta", 0) > 0:
        return "bias_reducing_uncertainty"
    if bias_delta < -threshold and row.get("mean_signed_nonstereotyped_delta", 0) > row.get("mean_unknown_delta", 0):
        return "bias_reducing_substitution"
    if row.get("mean_signed_stereotyped_delta", 0) < 0 and row.get("mean_signed_nonstereotyped_delta", 0) < 0 and row.get("mean_unknown_delta", 0) < 0:
        return "general_answer_suppression"
    return "mixed_or_unclear"


def summarize_effects(df: pd.DataFrame, group_cols: list[str], args: argparse.Namespace, value_col: str = "stereotype_preference_delta") -> pd.DataFrame:
    """Aggregate effect-size deltas per group.

    Audit 4.3 (closed 2026-05-27): the headline label, ranking, and beneficial/
    harmful scores now use the polarity-signed bias metric
    `signed_stereotype_preference_delta` so that positive = intervention made
    the model more biased regardless of `question_polarity`. The unsigned
    `mean_stereotype_preference_delta` is preserved for diagnostic comparison.
    """
    signed_value_col = "signed_stereotype_preference_delta" if "signed_stereotype_preference_delta" in df.columns else value_col
    rows = []
    rng = np.random.default_rng(0)
    for keys, group in tqdm(df.groupby(group_cols, dropna=False), desc=f"summarize {len(group_cols)} keys", leave=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        agg_spec = {
            "stereotype_preference_delta": (value_col, "mean"),
            "signed_stereotype_preference_delta": (signed_value_col, "mean"),
            "signed_stereotyped_delta": ("signed_stereotyped_delta", "mean") if "signed_stereotyped_delta" in group.columns else ("stereotyped_delta", "mean"),
            "signed_nonstereotyped_delta": ("signed_nonstereotyped_delta", "mean") if "signed_nonstereotyped_delta" in group.columns else ("nonstereotyped_delta", "mean"),
            "identity_answer_delta": ("identity_answer_delta", "mean") if "identity_answer_delta" in group.columns else (value_col, "mean"),
            "identity_specific_bias_delta": ("identity_specific_bias_delta", "mean") if "identity_specific_bias_delta" in group.columns else (value_col, "mean"),
            "unknown_delta": ("unknown_delta", "mean"),
            "stereotyped_delta": ("stereotyped_delta", "mean"),
            "nonstereotyped_delta": ("nonstereotyped_delta", "mean"),
            "identity_substitution_delta": ("identity_substitution_delta", "mean"),
            "substitution_delta": ("substitution_delta", "mean") if "substitution_delta" in group.columns else ("identity_substitution_delta", "mean"),
            "correct_delta": ("correct_delta", "mean"),
            "accuracy_delta": ("accuracy_delta", "mean"),
            "correct_margin_delta": ("correct_margin_delta", "mean"),
            "stereotype_error_delta": ("stereotype_error_delta", "mean"),
        }
        by_example = group.groupby("bbq_uid", dropna=False).agg(**agg_spec).reset_index()
        # Significance and CI test the signed metric — that's the load-bearing
        # "did the intervention reduce bias" question.
        signed_vals = by_example["signed_stereotype_preference_delta"].to_numpy(dtype=float)
        signed_vals_no_nan = signed_vals[np.isfinite(signed_vals)]
        ci_low, ci_high = bootstrap_ci(signed_vals_no_nan, args.bootstrap_samples, rng) if len(signed_vals_no_nan) else (float("nan"), float("nan"))
        p_value = sign_flip_pvalue(signed_vals_no_nan, args.permutation_samples, rng) if len(signed_vals_no_nan) else float("nan")
        n_polarity_skipped = int(np.isnan(signed_vals).sum())
        vals_unsigned = by_example["stereotype_preference_delta"].to_numpy(dtype=float)
        row.update({
            "n_examples": int(by_example["bbq_uid"].nunique()),
            "n_polarity_skipped": n_polarity_skipped,
            "mean_signed_stereotype_preference_delta": float(np.nanmean(signed_vals)) if len(signed_vals) else float("nan"),
            "median_signed_stereotype_preference_delta": float(np.nanmedian(signed_vals)) if len(signed_vals) else float("nan"),
            "std_signed_stereotype_preference_delta": float(np.nanstd(signed_vals_no_nan, ddof=1)) if len(signed_vals_no_nan) > 1 else 0.0,
            "mean_stereotype_preference_delta": float(np.nanmean(vals_unsigned)) if len(vals_unsigned) else float("nan"),  # diagnostic, polarity-confounded
            "median_stereotype_preference_delta": float(np.nanmedian(vals_unsigned)) if len(vals_unsigned) else float("nan"),
            "std_stereotype_preference_delta": float(np.nanstd(vals_unsigned, ddof=1)) if len(vals_unsigned) > 1 else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value_bootstrap_or_permutation": p_value,
            "mean_identity_answer_delta": float(by_example["identity_answer_delta"].mean()),
            "mean_identity_specific_bias_delta": float(by_example["identity_specific_bias_delta"].mean()),
            "mean_unknown_delta": float(by_example["unknown_delta"].mean()),
            "mean_signed_stereotyped_delta": float(by_example["signed_stereotyped_delta"].mean()),
            "mean_signed_nonstereotyped_delta": float(by_example["signed_nonstereotyped_delta"].mean()),
            "mean_stereotyped_delta": float(by_example["stereotyped_delta"].mean()),  # diagnostic, polarity-confounded
            "mean_nonstereotyped_delta": float(by_example["nonstereotyped_delta"].mean()),
            "mean_identity_substitution_delta": float(by_example["identity_substitution_delta"].mean()),
            "mean_substitution_delta": float(by_example["substitution_delta"].mean()),
            "mean_correct_delta": float(by_example["correct_delta"].mean()),
            "mean_accuracy_delta": float(by_example["accuracy_delta"].mean()),
            "mean_correct_margin_delta": float(by_example["correct_margin_delta"].mean()),
            "mean_stereotype_error_delta": float(by_example["stereotype_error_delta"].mean()),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    fdr_cols = [col for col in ["axis_mapped", "context_condition", "alpha", "intervention_position"] if col in out.columns]
    if fdr_cols:
        out["q_value_fdr"] = out.groupby(fdr_cols, dropna=False)["p_value_bootstrap_or_permutation"].transform(fdr_bh)
    else:
        out["q_value_fdr"] = fdr_bh(out["p_value_bootstrap_or_permutation"])
    out["significant"] = out["q_value_fdr"] < args.q_threshold
    out["effect_size_label"] = out["mean_signed_stereotype_preference_delta"].map(lambda v: effect_size_label(v, args))
    out["effect_label"] = out.apply(lambda row: effect_label(row, args), axis=1)
    # Audit 4.3: scores use signed metrics. beneficial_score is high when the
    # intervention reduces bias (negative signed_stereotype_preference_delta),
    # shifts the model toward "unknown" in ambiguous contexts, and preserves
    # correctness. harmful_score is the inverse for the bias direction.
    out["beneficial_score"] = -out["mean_signed_stereotype_preference_delta"].fillna(0) + out["mean_unknown_delta"].fillna(0) + np.maximum(0, out["mean_correct_margin_delta"].fillna(0))
    out["harmful_score"] = out["mean_signed_stereotype_preference_delta"].fillna(0) - out["mean_correct_margin_delta"].fillna(0)
    out["substitution_score"] = out["mean_signed_nonstereotyped_delta"].fillna(0) - out["mean_unknown_delta"].fillna(0)
    return out


# Audit 2.6: the headline hypothesis test's unit of inference is the FEATURE
# (× layer × intervention_position × context_condition), NOT feature × alpha ×
# polarity × identity-pair. All other columns here are feature-constant under
# matched-axis steering, so including them does not fragment a feature into
# multiple tested units — they just ride along for stratification/metadata.
# Polarity is POOLED (the audit-4.3 signed metric makes neg/nonneg comparable,
# so pooling is valid and raises power); identity pairs are pooled; alpha is
# fixed to the single pre-registered headline amplitude.
HEADLINE_UNIT_COLS = [
    "feature_id", "layer", "intervention_position", "context_condition",
    "axis_mapped", "feature_role", "feature_contrast_name",
    "feature_estimate_type", "steering_direction_label",
]


def resolve_headline_alpha(expanded: pd.DataFrame, args: argparse.Namespace, logger: logging.Logger) -> float | None:
    """Audit 2.6: pick the single pre-registered amplitude for the headline test."""
    present = sorted({
        float(a) for a in pd.to_numeric(expanded.get("alpha", pd.Series(dtype=float)), errors="coerce").dropna().unique()
    })
    if args.headline_alpha is not None:
        ha = float(args.headline_alpha)
        if present and ha not in present:
            logger.warning(
                "Audit 2.6: --headline_alpha=%s is not present in the results "
                "(present alphas: %s). feature_inference will be empty; for the "
                "audit-3.1 ablate headline pass --alphas 0 --headline_alpha 0.",
                ha, present,
            )
        return ha
    if len(present) == 1:
        logger.info("Audit 2.6: single alpha %s present; using it as the headline inference amplitude.", present[0])
        return present[0]
    if not present:
        logger.warning("Audit 2.6: no alpha values present in results; feature_inference will be empty.")
        return None
    raise ValueError(
        f"Audit 2.6: multiple alphas present ({present}) but --headline_alpha is not set. "
        "The unit of inference is the feature, tested at ONE pre-registered amplitude. "
        "Pass --headline_alpha to choose it; the rest of the alpha grid feeds the "
        "dose-response plots, not separate hypothesis tests."
    )


def feature_inference_table(expanded: pd.DataFrame, args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    """Audit 2.6/2.7: the headline feature-level inference table.

    One row per (feature, layer, position, context) at the single pre-registered
    headline alpha, pooling polarity and identity pairs. CIs/p-values use the
    audit-4.3 signed metric. Underpowered units (audit 2.7) are dropped before
    FDR, and FDR (BH) is computed across FEATURES within (axis, context,
    position) strata — not across the correlated alpha grid (audit 2.6). This is
    the table the rankings and final-candidates report should consume; the
    per-(alpha, position) `feature_level_effects` table is retained only as a
    dose-response diagnostic.

    Audit 2.5 (held-out confirmation) is layered on top of this in
    `feature_inference_with_holdout`; this function computes the pooled-data
    estimate used when no split is requested.
    """
    if expanded.empty:
        return expanded.copy()
    ha = resolve_headline_alpha(expanded, args, logger)
    if ha is None:
        return expanded.iloc[0:0].copy()
    sub = expanded[pd.to_numeric(expanded["alpha"], errors="coerce") == ha].copy()
    # Headline = the kept features themselves; controls live in their own tables.
    if "control_type" in sub.columns:
        sub = sub[sub["control_type"].astype(str).eq("kept_feature")].copy()
    if sub.empty:
        logger.warning("Audit 2.6: no kept_feature rows at headline_alpha=%s; feature_inference empty.", ha)
        return sub
    unit_cols = [c for c in HEADLINE_UNIT_COLS if c in sub.columns]
    table = summarize_effects(sub, unit_cols, args)
    if table.empty:
        return table
    table["headline_alpha"] = ha
    table = _apply_power_filter_and_fdr(table, args, logger, n_col="n_examples")
    return table


def _apply_power_filter_and_fdr(
    table: pd.DataFrame, args: argparse.Namespace, logger: logging.Logger, n_col: str = "n_examples",
) -> pd.DataFrame:
    """Audit 2.7 + 2.6: drop units below --min_examples_inference, then recompute
    FDR across features within (axis, context, position) on the survivors so
    q-values reflect the actual tested set (not the pre-filter superset)."""
    before = len(table)
    table = table[pd.to_numeric(table[n_col], errors="coerce").fillna(0) >= args.min_examples_inference].copy()
    dropped = before - len(table)
    if dropped:
        logger.info(
            "Audit 2.7: dropped %d/%d feature-inference units below "
            "--min_examples_inference=%d (underpowered).",
            dropped, before, args.min_examples_inference,
        )
    if table.empty:
        return table
    fdr_cols = [c for c in ["axis_mapped", "context_condition", "intervention_position"] if c in table.columns]
    if "p_value_confirmation" in table.columns:
        p_col = "p_value_confirmation"
    else:
        p_col = "p_value_bootstrap_or_permutation"
    if fdr_cols:
        table["q_value_fdr"] = table.groupby(fdr_cols, dropna=False)[p_col].transform(fdr_bh)
    else:
        table["q_value_fdr"] = fdr_bh(table[p_col])
    table["significant"] = table["q_value_fdr"] < args.q_threshold
    table["effect_label"] = table.apply(lambda r: effect_label(r, args), axis=1)
    return table


def load_feature_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    triage = pd.read_csv(path, low_memory=False)
    cols = [
        "layer", "feature_id", "provisional_role", "contrast_name", "top_axis", "top_identity",
        "role_confidence", "max_abs_cohens_d", "template_artifact_score", "sharedness_score",
        "contrast_specificity_score", "intervention_priority",
    ]
    return triage.reindex(columns=[c for c in cols if c in triage.columns]).drop_duplicates(["layer", "feature_id"])


def load_token_summary(path: Path) -> pd.DataFrame:
    summary = path / "bbq_token_level_sae_summary.csv"
    if not summary.exists():
        return pd.DataFrame()
    return pd.read_csv(summary, low_memory=False)


def merge_metadata(effects: pd.DataFrame, triage: pd.DataFrame, token_summary: pd.DataFrame) -> pd.DataFrame:
    out = effects.copy()
    if not triage.empty and {"layer", "feature_id"}.issubset(out.columns):
        out = out.merge(triage, on=["layer", "feature_id"], how="left", suffixes=("", "_triage"))
        if "provisional_role" in out.columns:
            out["feature_role"] = out["feature_role"].fillna(out["provisional_role"])
        if "contrast_name" in out.columns:
            out["original_feature_contrast_name"] = out["contrast_name"]
    if not token_summary.empty and {"layer", "feature_id"}.issubset(out.columns):
        out = out.merge(token_summary, on=["layer", "feature_id"], how="left", suffixes=("", "_token_summary"))
    return out


def make_rankings(effects: pd.DataFrame) -> pd.DataFrame:
    # Audit 4.3: rankings sort by the polarity-signed bias metric so
    # "strongest_bias_reducing" and "strongest_bias_amplifying" mean the same
    # thing across neg- and nonneg-polarity questions.
    bias_col = "mean_signed_stereotype_preference_delta" if "mean_signed_stereotype_preference_delta" in effects.columns else "mean_stereotype_preference_delta"
    frames = []
    specs = [
        ("strongest_bias_reducing_features", bias_col, True),
        ("strongest_bias_amplifying_features", bias_col, False),
        ("strongest_unknown_increasing_features", "mean_unknown_delta", False),
        ("strongest_substitution_effect_features", "mean_identity_substitution_delta", False),
        ("least_capability_degrading_features", "mean_accuracy_delta", False),
        ("strongest_capability_degrading_features", "mean_accuracy_delta", True),
    ]
    for ranking, col, ascending in specs:
        if col not in effects.columns:
            continue
        ranked = effects.sort_values(col, ascending=ascending).head(100).copy()
        ranked["ranking_type"] = ranking
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / f"{name}.png", dpi=220)
    fig.savefig(output_dir / "figures" / f"{name}.pdf")
    plt.close(fig)


def plot_feature_bars(effects: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    # Audit 4.3: bars show the polarity-signed bias metric (CI in
    # summarize_effects is also on the signed metric).
    bias_col = "mean_signed_stereotype_preference_delta" if "mean_signed_stereotype_preference_delta" in effects.columns else "mean_stereotype_preference_delta"
    data = effects[(effects["context_condition"].eq("ambig")) & (effects["n_examples"] >= args.min_examples)].copy()
    data = data.sort_values(bias_col, key=lambda s: s.abs(), ascending=False).head(args.top_n_features)
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(data)), 6))
    x = np.arange(len(data))
    colors = data["effect_label"].astype("category").cat.codes
    ax.bar(x, data[bias_col], color=plt.cm.tab20(colors % 20))
    ax.errorbar(x, data[bias_col], yerr=[data[bias_col] - data["ci_low"], data["ci_high"] - data[bias_col]], fmt="none", color="black", lw=1)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, [str(int(v)) for v in data["feature_id"]], rotation=60, ha="right")
    ax.set_ylabel(BIAS_LABEL)
    ax.set_title("Feature-level ambiguous bias effects (top absolute effects, polarity-signed)")
    save_fig(fig, output_dir, "feature_level_bias_effect_bars_by_axis_alpha")


def plot_top_by_contrast(effects: pd.DataFrame, output_dir: Path, args: argparse.Namespace, kind: str) -> None:
    bias_col = "mean_signed_stereotype_preference_delta" if "mean_signed_stereotype_preference_delta" in effects.columns else "mean_stereotype_preference_delta"
    data = effects[(effects["context_condition"].eq("ambig")) & (effects["n_examples"] >= args.min_examples)].copy()
    if data.empty:
        return
    pieces = []
    for _, group in data.groupby("mapped_contrast_name", dropna=False):
        pieces.append(group.nsmallest(10, bias_col) if kind == "reducing" else group.nlargest(10, bias_col))
    plot_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if plot_df.empty:
        return
    plot_df["label"] = plot_df["mapped_contrast_name"].astype(str) + "\nF" + plot_df["feature_id"].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(plot_df))))
    ax.barh(np.arange(len(plot_df)), plot_df[bias_col])
    ax.errorbar(plot_df[bias_col], np.arange(len(plot_df)), xerr=[plot_df[bias_col] - plot_df["ci_low"], plot_df["ci_high"] - plot_df[bias_col]], fmt="none", color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(np.arange(len(plot_df)), plot_df["label"], fontsize=8)
    ax.set_xlabel(BIAS_LABEL)
    ax.set_title(f"Top bias-{kind} features by contrast (polarity-signed)")
    save_fig(fig, output_dir, f"top_bias_{kind}_features_by_contrast")


def plot_amp_suppression_scatter(effects: pd.DataFrame, output_dir: Path) -> None:
    data = effects[effects["context_condition"].eq("ambig")].copy()
    if data.empty:
        return
    data["alpha_abs"] = pd.to_numeric(data["alpha"], errors="coerce").abs()
    target_abs = 1.0 if (data["alpha_abs"] == 1.0).any() else data["alpha_abs"].dropna().min()
    sub = data[data["alpha_abs"].eq(target_abs)]
    bias_col = "mean_signed_stereotype_preference_delta" if "mean_signed_stereotype_preference_delta" in sub.columns else "mean_stereotype_preference_delta"
    pivot = sub.pivot_table(index=["layer", "feature_id", "feature_role", "axis_mapped"], columns="steering_direction_label", values=bias_col, aggfunc="mean").reset_index()
    if not {"feature_amplification", "feature_suppression"}.issubset(pivot.columns):
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    if sns:
        sns.scatterplot(data=pivot, x="feature_amplification", y="feature_suppression", hue="feature_role", style="axis_mapped", ax=ax)
    else:
        ax.scatter(pivot["feature_amplification"], pivot["feature_suppression"])
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(f"Effect at alpha +{target_abs:g}")
    ax.set_ylabel(f"Effect at alpha -{target_abs:g}")
    ax.set_title("Amplification vs suppression feature effects")
    save_fig(fig, output_dir, "amplification_vs_suppression_feature_scatter")


def plot_dose_response(effects: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    bias_col = "mean_signed_stereotype_preference_delta" if "mean_signed_stereotype_preference_delta" in effects.columns else "mean_stereotype_preference_delta"
    data = effects[effects["context_condition"].eq("ambig")].copy()
    top = data.sort_values(bias_col, key=lambda s: s.abs(), ascending=False)["feature_id"].drop_duplicates().head(min(12, args.top_n_features))
    data = data[data["feature_id"].isin(top)]
    if data.empty:
        return
    g = sns.relplot(data=data, x="alpha", y=bias_col, hue="feature_id", col="mapped_contrast_name", kind="line", col_wrap=3, facet_kws={"sharey": False}) if sns else None
    if g is not None:
        g.set_axis_labels("alpha", BIAS_LABEL)
        g.figure.suptitle("Feature effect dose response", y=1.02)
        g.figure.savefig(output_dir / "figures" / "feature_effect_dose_response_grid.png", dpi=220, bbox_inches="tight")
        g.figure.savefig(output_dir / "figures" / "feature_effect_dose_response_grid.pdf", bbox_inches="tight")
        plt.close(g.figure)


def plot_answer_role_shift(effects: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    bias_col = "mean_signed_stereotype_preference_delta" if "mean_signed_stereotype_preference_delta" in effects.columns else "mean_stereotype_preference_delta"
    # Audit 4.3: also melt the signed stereotyped/nonstereotyped deltas so the
    # bars show "did the intervention shift weight onto the bias-aligned vs
    # anti-bias vs unknown answer" coherently across polarities.
    signed_stereo = "mean_signed_stereotyped_delta" if "mean_signed_stereotyped_delta" in effects.columns else "mean_stereotyped_delta"
    signed_nonstereo = "mean_signed_nonstereotyped_delta" if "mean_signed_nonstereotyped_delta" in effects.columns else "mean_nonstereotyped_delta"
    data = effects[effects["context_condition"].eq("ambig")].sort_values(bias_col, key=lambda s: s.abs(), ascending=False).head(args.top_n_features)
    if data.empty:
        return
    melt = data.melt(id_vars=["feature_id", "feature_role"], value_vars=[signed_stereo, signed_nonstereo, "mean_unknown_delta"], var_name="answer_role", value_name="mean_delta")
    fig, ax = plt.subplots(figsize=(max(12, 0.5 * data["feature_id"].nunique()), 6))
    if sns:
        sns.barplot(data=melt, x="feature_id", y="mean_delta", hue="answer_role", ax=ax)
    else:
        melt.groupby(["feature_id", "answer_role"])["mean_delta"].mean().unstack().plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Answer-role shifts by feature")
    ax.set_ylabel("Mean delta log probability")
    ax.tick_params(axis="x", rotation=60)
    save_fig(fig, output_dir, "answer_role_shift_by_feature")


def plot_subgroup_heatmap(subgroup: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    data = subgroup[subgroup["context_condition"].eq("ambig")].copy()
    if data.empty:
        return
    top_features = data.groupby("feature_id")["mean_stereotype_preference_delta"].mean().abs().sort_values(ascending=False).head(args.top_n_features).index
    data = data[data["feature_id"].isin(top_features)]
    matrix = data.pivot_table(index="feature_id", columns="mapped_contrast_name", values="mean_stereotype_preference_delta", aggfunc="mean")
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * matrix.shape[1]), max(6, 0.35 * matrix.shape[0])))
    if sns:
        sns.heatmap(matrix, center=0, cmap="vlag", ax=ax)
    else:
        im = ax.imshow(matrix.fillna(0).to_numpy(), cmap="coolwarm")
        fig.colorbar(im, ax=ax)
    ax.set_title("Feature effects by identity subgroup / contrast")
    ax.set_xlabel("BBQ mapped contrast")
    ax.set_ylabel("Feature ID")
    save_fig(fig, output_dir, "subgroup_heatmap_feature_effects")


def plot_disambig_tradeoff(feature_effects: pd.DataFrame, output_dir: Path) -> None:
    ambig = feature_effects[feature_effects["context_condition"].eq("ambig")]
    dis = feature_effects[feature_effects["context_condition"].eq("disambig")]
    if ambig.empty or dis.empty:
        return
    left = ambig.groupby(["layer", "feature_id", "alpha", "intervention_position"]).agg(ambiguous_delta=("mean_stereotype_preference_delta", "mean"), feature_role=("feature_role", "first"), axis_mapped=("axis_mapped", "first")).reset_index()
    right = dis.groupby(["layer", "feature_id", "alpha", "intervention_position"]).agg(disambig_correct_margin_delta=("mean_correct_margin_delta", "mean"), accuracy_delta=("mean_accuracy_delta", "mean")).reset_index()
    data = left.merge(right, on=["layer", "feature_id", "alpha", "intervention_position"], how="inner")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    if sns:
        sns.scatterplot(data=data, x="ambiguous_delta", y="disambig_correct_margin_delta", hue="axis_mapped", style="feature_role", ax=ax)
    else:
        ax.scatter(data["ambiguous_delta"], data["disambig_correct_margin_delta"])
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(BIAS_LABEL + "\n(negative = less stereotype preference)")
    ax.set_ylabel("Delta disambiguated correct margin")
    ax.set_title("Feature-level ambiguous bias vs disambiguated correctness tradeoff")
    save_fig(fig, output_dir, "disambig_tradeoff_feature_level")


def plot_token_activation_vs_effect(effects: pd.DataFrame, output_dir: Path) -> None:
    needed = {"mean_stereotype_preference_delta", "mean_target_identity_activation", "mean_stereotype_language_activation"}
    if not needed & set(effects.columns):
        return
    data = effects[effects["context_condition"].eq("ambig")].copy()
    if data.empty:
        return
    x_col = "mean_stereotype_language_activation" if "mean_stereotype_language_activation" in data.columns else "mean_target_identity_activation"
    if x_col not in data.columns or data[x_col].notna().sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    if sns:
        sns.scatterplot(data=data, x=x_col, y="mean_stereotype_preference_delta", hue="feature_role", ax=ax)
    else:
        ax.scatter(data[x_col], data["mean_stereotype_preference_delta"])
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(BIAS_LABEL)
    ax.set_title("Token activation vs causal effect")
    save_fig(fig, output_dir, "token_activation_vs_causal_effect")


def feature_card_links_html(effects: pd.DataFrame, output_dir: Path) -> None:
    top = effects.sort_values("mean_stereotype_preference_delta").head(200)
    rows = []
    for row in top.itertuples(index=False):
        layer = int(getattr(row, "layer"))
        feature_id = int(getattr(row, "feature_id"))
        original = f"../../../sae_identity/llama-3.1-8b/feature_cards/layer_{layer:02d}/feature_{feature_id:05d}.html"
        bbq = f"../feature_cards/layer_{layer:02d}/feature_{feature_id:05d}.html"
        rows.append(
            f"<tr><td>{layer}</td><td>{feature_id}</td><td>{html.escape(str(getattr(row, 'feature_role', '')))}</td>"
            f"<td>{html.escape(str(getattr(row, 'mapped_contrast_name', '')))}</td><td>{getattr(row, 'mean_stereotype_preference_delta', float('nan')):.4g}</td>"
            f"<td>{html.escape(str(getattr(row, 'effect_label', '')))}</td><td><a href='{original}'>original</a></td><td><a href='{bbq}'>BBQ card</a></td></tr>"
        )
    (output_dir / "feature_card_links_table.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>BBQ Feature Causal Effects</title>"
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px}th{background:#f5f5f5}</style>"
        "</head><body><h1>Feature-level BBQ causal effects</h1><p>Negative delta stereotype preference means less preference for the stereotyped answer relative to unknown.</p>"
        "<table><tr><th>Layer</th><th>Feature</th><th>Role</th><th>BBQ contrast</th><th>Mean delta</th><th>Effect label</th><th>Original card</th><th>BBQ card</th></tr>"
        + "\n".join(rows)
        + "</table></body></html>"
    )


def filtered_top(data: pd.DataFrame, value_col: str, n: int, ascending: bool = True) -> pd.DataFrame:
    if data.empty or value_col not in data.columns:
        return pd.DataFrame()
    return data.sort_values(value_col, ascending=ascending).head(n).copy()


def identity_profile_heatmap(data: pd.DataFrame, output_path: Path, title: str, args: argparse.Namespace) -> None:
    if data.empty:
        return
    metrics = [
        "mean_identity_answer_delta",
        "mean_stereotyped_delta",
        "mean_nonstereotyped_delta",
        "mean_unknown_delta",
        "mean_stereotype_preference_delta",
        "mean_correct_margin_delta",
    ]
    available = [m for m in metrics if m in data.columns]
    if not available:
        return
    top_features = data.groupby("feature_id")["mean_stereotype_preference_delta"].mean().abs().sort_values(ascending=False).head(args.top_n_features).index
    plot_df = data[data["feature_id"].isin(top_features)].groupby("feature_id")[available].mean()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(available)), max(5, 0.35 * len(plot_df))))
    if sns:
        sns.heatmap(plot_df, center=0, cmap="vlag", annot=False, ax=ax)
    else:
        im = ax.imshow(plot_df.fillna(0).to_numpy(), cmap="coolwarm")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(plot_df.columns)), plot_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(plot_df.index)), plot_df.index)
    ax.set_title(title)
    save_fig(fig, output_path.parent, output_path.stem)


def report_barh(data: pd.DataFrame, output_path: Path, title: str, value_col: str, args: argparse.Namespace, ascending: bool) -> None:
    plot_df = filtered_top(data, value_col, args.top_n_features, ascending=ascending)
    if plot_df.empty:
        return
    plot_df["label"] = "F" + plot_df["feature_id"].astype(int).astype(str) + " | " + plot_df.get("feature_role", "").astype(str)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    ax.barh(np.arange(len(plot_df)), plot_df[value_col])
    if {"ci_low", "ci_high"}.issubset(plot_df.columns):
        ax.errorbar(plot_df[value_col], np.arange(len(plot_df)), xerr=[plot_df[value_col] - plot_df["ci_low"], plot_df["ci_high"] - plot_df[value_col]], fmt="none", color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(np.arange(len(plot_df)), plot_df["label"], fontsize=8)
    ax.set_xlabel(BIAS_LABEL)
    ax.set_title(title)
    save_fig(fig, output_path.parent, output_path.stem)


def axis_overview_plot(data: pd.DataFrame, output_path: Path, title: str, args: argparse.Namespace) -> None:
    if data.empty:
        return
    top_features = data.groupby("feature_id")["mean_stereotype_preference_delta"].mean().abs().sort_values(ascending=False).head(args.top_n_features).index
    plot_df = data[data["feature_id"].isin(top_features)]
    matrix = plot_df.pivot_table(index="feature_id", columns="identity_id", values="mean_identity_answer_delta", aggfunc="mean")
    if matrix.empty:
        matrix = plot_df.pivot_table(index="feature_id", columns="mapped_contrast_name", values="mean_stereotype_preference_delta", aggfunc="mean")
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * matrix.shape[1]), max(5, 0.35 * matrix.shape[0])))
    if sns:
        sns.heatmap(matrix, center=0, cmap="vlag", ax=ax)
    else:
        im = ax.imshow(matrix.fillna(0).to_numpy(), cmap="coolwarm")
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Identity / subgroup")
    ax.set_ylabel("Feature ID")
    save_fig(fig, output_path.parent, output_path.stem)


def answer_role_decomposition(data: pd.DataFrame, output_path: Path, title: str, args: argparse.Namespace) -> None:
    data = data.sort_values("mean_stereotype_preference_delta", key=lambda s: s.abs(), ascending=False).head(args.top_n_features)
    if data.empty:
        return
    melt = data.melt(
        id_vars=["feature_id"],
        value_vars=["mean_stereotyped_delta", "mean_nonstereotyped_delta", "mean_unknown_delta"],
        var_name="answer_role",
        value_name="delta",
    )
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * data["feature_id"].nunique()), 5))
    if sns:
        sns.barplot(data=melt, x="feature_id", y="delta", hue="answer_role", ax=ax)
    else:
        melt.groupby(["feature_id", "answer_role"])["delta"].mean().unstack().plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title(title)
    ax.set_ylabel("Mean delta log probability")
    ax.tick_params(axis="x", rotation=60)
    save_fig(fig, output_path.parent, output_path.stem)


def role_scatter(data: pd.DataFrame, output_path: Path, title: str) -> None:
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    if sns:
        sns.scatterplot(data=data, x="mean_identity_answer_delta", y="mean_stereotype_preference_delta", hue="feature_role", style="intervention_position", ax=ax)
    else:
        ax.scatter(data["mean_identity_answer_delta"], data["mean_stereotype_preference_delta"])
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Delta identity answer salience")
    ax.set_ylabel(BIAS_LABEL)
    ax.set_title(title)
    save_fig(fig, output_path.parent, output_path.stem)


def dose_response_report(data: pd.DataFrame, output_path: Path, title: str, args: argparse.Namespace) -> None:
    if data.empty:
        return
    if args.require_complete_alpha_grid:
        expected = set(parse_floats(args.selected_alphas or args.alphas))
        keep = []
        for feature_id, group in data.groupby("feature_id"):
            if expected.issubset(set(pd.to_numeric(group["alpha"], errors="coerce"))):
                keep.append(feature_id)
        data = data[data["feature_id"].isin(keep)]
    top = data.groupby("feature_id")["mean_stereotype_preference_delta"].mean().abs().sort_values(ascending=False).head(min(12, args.top_n_features)).index
    data = data[data["feature_id"].isin(top)]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    if sns:
        sns.lineplot(data=data, x="alpha", y="mean_stereotype_preference_delta", hue="feature_id", marker="o", ax=ax)
    else:
        data.groupby(["alpha", "feature_id"])["mean_stereotype_preference_delta"].mean().unstack().plot(ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title(title)
    ax.set_ylabel(BIAS_LABEL)
    save_fig(fig, output_path.parent, output_path.stem)


def taxonomy_plot(data: pd.DataFrame, group_col: str, output_path: Path, title: str) -> None:
    if data.empty or group_col not in data.columns:
        return
    sig = data[data["significant"].astype(bool)] if "significant" in data.columns else data
    counts = sig.groupby([group_col, "effect_label"])["feature_id"].nunique().reset_index(name="n_features")
    if counts.empty:
        return
    pivot = counts.pivot(index=group_col, columns="effect_label", values="n_features").fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(pivot))))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Significant feature count")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, output_path.parent, output_path.stem)


def final_candidates_html(data: pd.DataFrame, output_dir: Path) -> None:
    top = data.sort_values(["beneficial_score", "mean_stereotype_preference_delta"], ascending=[False, True]).head(500)
    rows = []
    for row in top.itertuples(index=False):
        layer = int(getattr(row, "layer"))
        feature_id = int(getattr(row, "feature_id"))
        original = f"../../../sae_identity/llama-3.1-8b/feature_cards/layer_{layer:02d}/feature_{feature_id:05d}.html"
        bbq = f"../feature_cards/layer_{layer:02d}/feature_{feature_id:05d}.html"
        rows.append(
            f"<tr data-axis='{html.escape(str(getattr(row, 'axis_mapped', '')))}' data-label='{html.escape(str(getattr(row, 'effect_label', '')))}'>"
            f"<td>{feature_id}</td><td>{layer}</td><td>{html.escape(str(getattr(row, 'axis_mapped', '')))}</td><td>{html.escape(str(getattr(row, 'identity_id', '')))}</td>"
            f"<td>{html.escape(str(getattr(row, 'mapped_contrast_name', '')))}</td><td>{html.escape(str(getattr(row, 'feature_role', '')))}</td>"
            f"<td>{html.escape(str(getattr(row, 'intervention_position', '')))}</td><td>{getattr(row, 'alpha', '')}</td>"
            f"<td>{getattr(row, 'mean_stereotype_preference_delta', float('nan')):.4g}</td><td>{getattr(row, 'mean_identity_answer_delta', float('nan')):.4g}</td>"
            f"<td>{getattr(row, 'mean_unknown_delta', float('nan')):.4g}</td><td>{getattr(row, 'mean_nonstereotyped_delta', float('nan')):.4g}</td>"
            f"<td>{getattr(row, 'mean_correct_margin_delta', float('nan')):.4g}</td><td>{getattr(row, 'q_value_fdr', float('nan')):.4g}</td>"
            f"<td>{html.escape(str(getattr(row, 'effect_label', '')))}</td><td><a href='{original}'>original</a></td><td><a href='{bbq}'>BBQ</a></td></tr>"
        )
    (output_dir / "final_intervention_candidates_table.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Final intervention candidates</title>"
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px}th{background:#f5f5f5;position:sticky;top:0}</style>"
        "</head><body><h1>Final intervention candidates</h1><p>Negative stereotype-preference delta means reduced preference for stereotyped answer relative to unknown.</p>"
        "<table><tr><th>Feature</th><th>Layer</th><th>Axis</th><th>Identity</th><th>Contrast</th><th>Role</th><th>Position</th><th>Alpha</th><th>Δ stereotype pref.</th><th>Δ identity answer</th><th>Δ unknown</th><th>Δ nonstereo</th><th>Δ correct margin</th><th>q</th><th>Label</th><th>Original card</th><th>BBQ card</th></tr>"
        + "\n".join(rows)
        + "</table></body></html>"
    )


def make_reports(identity_effects: pd.DataFrame, feature_effects: pd.DataFrame, output_dir: Path, args: argparse.Namespace, candidates_table: pd.DataFrame | None = None) -> None:
    analysis_dir = output_dir / "analysis"
    axis_root = analysis_dir / "axis_reports"
    contrast_root = analysis_dir / "contrast_reports"
    identity_root = analysis_dir / "identity_reports"
    for root in [axis_root, contrast_root, identity_root]:
        root.mkdir(parents=True, exist_ok=True)

    if args.make_axis_reports:
        for axis, group in identity_effects.groupby("axis_mapped", dropna=False):
            if group["n_examples"].max() < args.min_examples_per_identity:
                continue
            out = axis_root / safe_name(axis)
            out.mkdir(exist_ok=True)
            axis_overview_plot(group, out / "axis_feature_effect_overview.png", f"{axis}: feature effects by identity", args)
            answer_role_decomposition(group, out / "axis_answer_role_decomposition.png", f"{axis}: answer-role decomposition", args)

    if args.make_identity_reports:
        for identity, group in identity_effects.groupby("identity_id", dropna=False):
            if group["n_examples"].max() < args.min_examples_per_identity:
                continue
            out = identity_root / safe_name(identity)
            out.mkdir(exist_ok=True)
            identity_profile_heatmap(group, out / "identity_feature_causal_profile.png", f"{identity}: feature causal profile", args)
            role_scatter(group, out / "identity_feature_role_scatter.png", f"{identity}: identity salience vs stereotype preference")
            report_barh(group, out / "identity_specific_top_bias_reducing_features.png", f"{identity}: top bias-reducing features", "mean_stereotype_preference_delta", args, True)
            report_barh(group, out / "identity_specific_top_bias_amplifying_features.png", f"{identity}: top bias-amplifying features", "mean_stereotype_preference_delta", args, False)

    if args.make_contrast_reports:
        for contrast, group in feature_effects.groupby("mapped_contrast_name", dropna=False):
            if group["n_examples"].max() < args.min_examples_per_contrast:
                continue
            out = contrast_root / safe_name(contrast)
            out.mkdir(exist_ok=True)
            dose_response_report(group, out / "contrast_feature_dose_response.png", f"{contrast}: feature dose response", args)

    taxonomy_plot(identity_effects, "identity_id", analysis_dir / "feature_behavior_taxonomy_by_identity.png", "Feature behavior taxonomy by identity")
    taxonomy_plot(feature_effects, "axis_mapped", analysis_dir / "axis_behavior_taxonomy_summary.png", "Feature behavior taxonomy by axis")
    # Audit 2.5/2.6: the final-candidates table is the headline deliverable, so it
    # uses the feature-level inference table (held-out CIs/q, one pre-registered
    # alpha) when available; the per-identity/per-alpha tables remain a fallback.
    candidates = candidates_table if (candidates_table is not None and not candidates_table.empty) else (identity_effects if not identity_effects.empty else feature_effects)
    final_candidates_html(candidates, output_dir)


def write_validation_summaries(identity_effects: pd.DataFrame, feature_effects: pd.DataFrame, identity_records: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    if not feature_effects.empty:
        rows.append({"metric": "n_features_by_axis_role", "value": feature_effects.groupby(["axis_mapped", "feature_role"])["feature_id"].nunique().to_dict()})
        rows.append({"metric": "n_complete_feature_alpha_grids", "value": int(feature_effects.groupby(["feature_id", "intervention_position"])["alpha"].nunique().ge(2).sum())})
        rows.append({"metric": "n_significant_by_axis_identity", "value": identity_effects[identity_effects["significant"].astype(bool)].groupby(["axis_mapped", "identity_id"])["feature_id"].nunique().to_dict() if not identity_effects.empty else {}})
    if not identity_records.empty:
        counts = identity_records.groupby(["axis_mapped", "identity_id"])["bbq_uid"].nunique()
        rows.append({"metric": "n_identities_with_enough_examples_10", "value": int(counts.ge(10).sum())})
        rows.append({"metric": "top_underpowered_identities", "value": counts[counts < 10].sort_values().head(20).to_dict()})
    pd.DataFrame(rows).to_csv(output_dir / "validation_summary.csv", index=False)


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        f"""# Feature-Level BBQ SAE Causal Analysis

## Sign convention

For ambiguous BBQ items:

```text
stereotype_preference = log p(stereotyped answer) - log p(unknown answer)
Delta stereotype preference = intervened - baseline
```

- Negative Delta stereotype preference means less preference for the stereotyped answer relative to unknown.
- Positive Delta stereotype preference means more preference for the stereotyped answer relative to unknown.
- Positive `unknown_delta` means more probability/logprob mass on the unknown answer.
- Positive `nonstereotyped_delta` with negative `stereotyped_delta` can indicate substitution rather than calibrated uncertainty.

## Identity-aware metrics

- `identity_answer_delta` is the log-probability shift for a specific identity's answer option.
- `identity_specific_bias_delta` compares that identity answer shift to the unknown answer shift.
- `substitution_delta` is Delta[log p(nonstereotyped) - log p(stereotyped)].
- `beneficial_score` rewards lower stereotype preference, higher unknown probability, and non-negative disambiguated correct margin.
- `harmful_score` rewards higher stereotype preference and penalizes correct-margin degradation.

## Intervention positions

Final-token interventions test prompt-level integrated effects at the decision point.
Identity-token interventions test whether the feature acts locally at the identity mention.
Stereotype-language interventions test whether the feature acts through predicate/stereotype terms.
Analyze these separately; they answer different causal questions.

## Taxonomy

The identity-aware labels distinguish `identity_only`, `bias_amplifying`, `bias_reducing_uncertainty`, `bias_reducing_substitution`, `general_answer_suppression`, `capability_degrading`, `mixed_or_unclear`, and `no_reliable_effect`.

## Estimate types

`individual_feature` rows come from steering runs where exactly one SAE feature was intervened on.
`feature_bundle_membership` rows come from bundle steering runs and should be interpreted as membership diagnostics, not isolated single-feature causal estimates.

## Statistics

Confidence intervals bootstrap over BBQ examples. P-values use paired sign-flip permutation tests over per-example deltas.

### Headline inference table: `feature_inference.csv` (audits 2.5 / 2.6 / 2.7)

The headline hypothesis test's **unit of inference is the feature** (× layer × intervention position × context condition), tested at a single pre-registered `--headline_alpha` — NOT one test per feature × alpha × position × polarity × identity-pair. Question polarity is pooled (the audit-4.3 signed metric makes neg/nonneg comparable, which is also more powerful); the rest of the alpha grid feeds the dose-response *plots* only, not separate tests (audit 2.6). FDR (Benjamini–Hochberg) is applied **across features** within (axis × context × intervention position). Underpowered units (fewer than `--min_examples_inference` examples, per held-out half) are dropped before FDR (audit 2.7); final runs use ≥10,000 bootstrap/permutation resamples (`--smoke` is a dev-only cap and emits a warning).

Held-out confirmation (audit 2.5): features are ranked/selected on a per-axis **selection** half of the BBQ examples, and the reported effect sizes, CIs, and q-values come from the disjoint **confirmation** half — so the headline numbers are not the post-selection (winner's-curse) estimates. `selection_*` columns carry the selection-half effect used for ranking; the unprefixed `mean_*`/`ci_*`/`q_value_fdr` are the confirmation-half reported values.

`feature_effect_rankings.csv` and `final_intervention_candidates_table.html` are derived from `feature_inference.csv`. The per-(alpha, position) `feature_level_effects.csv` is retained only as a dose-response diagnostic and its q-values should not be treated as headline significance.
"""
    )


def validation_report(df: pd.DataFrame, args: argparse.Namespace, logger: logging.Logger) -> None:
    required = ["bbq_uid", "feature_ids_json", "alpha", "intervention_position", "context_condition", "unknown_answer_idx", "mapped_contrast_name"]
    missing = [col for col in required if col not in df.columns]
    logger.info("Rows loaded: %d", len(df))
    logger.info("Unique BBQ examples: %d", df["bbq_uid"].nunique() if "bbq_uid" in df.columns else 0)
    logger.info("Rows by alpha:\n%s", df["alpha"].value_counts(dropna=False).sort_index() if "alpha" in df.columns else "missing")
    logger.info("Rows by intervention_position:\n%s", df["intervention_position"].value_counts(dropna=False) if "intervention_position" in df.columns else "missing")
    logger.info("Rows by axis/context/polarity:\n%s", df.groupby(["axis_mapped", "context_condition", "question_polarity"]).size() if {"axis_mapped", "context_condition", "question_polarity"}.issubset(df.columns) else "missing")
    if missing:
        logger.warning("Missing required columns: %s", missing)
    if "feature_ids_json" in df.columns and df["feature_ids_json"].isna().all():
        logger.warning("feature_id information missing; feature-level estimates cannot be produced.")
    if "alpha" in df.columns and not (pd.to_numeric(df["alpha"], errors="coerce") > 0).any():
        logger.warning("No positive-alpha rows found.")
    if "alpha" in df.columns and not (pd.to_numeric(df["alpha"], errors="coerce") < 0).any():
        logger.warning("No negative-alpha rows found.")
    if "unknown_answer_idx" in df.columns:
        logger.info("Rows dropped if unknown answer missing: %d", int(df["unknown_answer_idx"].isna().sum()))


def main() -> None:
    args = parse_args()
    if args.selected_alphas:
        args.alphas = args.selected_alphas
    if args.selected_positions:
        args.positions = args.selected_positions
    if args.smoke:
        args.bootstrap_samples = min(args.bootstrap_samples, 500)
        args.permutation_samples = min(args.permutation_samples, 500)
        args.min_examples = min(args.min_examples, 10)
        args.min_examples_inference = min(args.min_examples_inference, 10)
    logger = setup(args.output_dir, args.overwrite)
    if args.smoke:
        # Audit 2.7: --smoke caps resamples at 500 (min p ~= 0.002) and lowers the
        # per-cell minimum; it is a fast-dev convenience, NOT a headline setting.
        logger.warning(
            "Audit 2.7: --smoke is set — bootstrap/permutation capped at 500 and "
            "min-examples lowered. This run's CIs and q-values are UNDERPOWERED and "
            "must not be cited. Drop --smoke for headline numbers (defaults are now "
            "%d bootstrap / %d permutation resamples).",
            10000, 10000,
        )
    config = vars(args).copy()
    config.update({k: str(v) for k, v in config.items() if isinstance(v, Path)})
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "feature_level_causal_config.json").write_text(json.dumps(config, indent=2) + "\n")

    layers = parse_ints(args.layers)
    alphas = parse_floats(args.alphas)
    positions = parse_strings(args.positions)
    contexts = parse_strings(args.context_conditions)

    results = load_results(args.steering_dir, args.results_csv)
    if results.empty:
        raise FileNotFoundError("No steering results found.")
    validation_report(results, args, logger)
    results = enrich_results(results)
    if layers:
        results = results[results["layer"].isin(layers)].copy()
    if alphas:
        results = results[pd.to_numeric(results["alpha"], errors="coerce").isin(alphas)].copy()
    if positions:
        results = results[results["intervention_position"].astype(str).isin(positions)].copy()
    if contexts:
        results = results[results["context_condition"].astype(str).isin(contexts)].copy()
    results = results[results["unknown_answer_idx"].notna()].copy()
    write_table(results, args.output_dir, "merged_results")

    expanded = expand_feature_rows(results)
    write_table(expanded, args.output_dir, "deltas_long")
    identity_records = build_identity_records(results)
    write_table(identity_records, args.output_dir, "identity_records")
    identity_long = merge_identity_records(expanded, identity_records)
    write_table(identity_long, args.output_dir, "identity_deltas_long")
    individual_count = expanded["feature_estimate_type"].eq("individual_feature").sum() if not expanded.empty else 0
    if individual_count == 0:
        logger.warning("No individual-feature steering rows found. Outputs will reflect feature bundle membership, not isolated feature causal effects.")

    group_cols = [
        "feature_id", "layer", "alpha", "intervention_position", "feature_role",
        "feature_contrast_name", "mapped_contrast_name", "axis_mapped",
        "context_condition", "question_polarity", "target_identity_id",
        "nontarget_identity_id", "feature_estimate_type", "steering_direction_label",
    ]
    group_cols = [col for col in group_cols if col in expanded.columns]
    feature_pre = summarize_effects(expanded, group_cols, args)
    write_table(feature_pre, args.output_dir, "feature_level_pre_fdr")
    triage = load_feature_metadata(args.triage_csv)
    token_summary = load_token_summary(args.token_level_dir)
    feature_effects = merge_metadata(feature_pre, triage, token_summary)
    write_table(feature_effects, args.output_dir, "feature_level_effects")

    # Audit 2.6/2.7: the HEADLINE inference table. Unit of inference = feature
    # (× position × context) at the single pre-registered --headline_alpha;
    # polarity pooled via the signed metric; underpowered units dropped; FDR
    # across features within (axis, context, position). Rankings + the
    # final-candidates report consume THIS table. feature_level_effects above
    # stays as the per-(alpha, position) dose-response diagnostic.
    feature_inference = feature_inference_table(expanded, args, logger)
    if not feature_inference.empty:
        feature_inference = merge_metadata(feature_inference, triage, token_summary)
    write_table(feature_inference, args.output_dir, "feature_inference")
    logger.info(
        "Audit 2.6: feature_inference has %d feature-level units (headline_alpha=%s); "
        "%d significant at q<%.3g.",
        len(feature_inference),
        feature_inference["headline_alpha"].iloc[0] if not feature_inference.empty else "n/a",
        int(feature_inference["significant"].sum()) if not feature_inference.empty else 0,
        args.q_threshold,
    )

    subgroup_cols = [
        "axis_mapped", "mapped_contrast_name", "target_identity_id", "nontarget_identity_id",
        "question_polarity", "context_condition", "feature_id", "layer", "alpha",
        "intervention_position", "feature_role", "feature_estimate_type", "steering_direction_label",
    ]
    subgroup_cols = [col for col in subgroup_cols if col in expanded.columns]
    subgroup = summarize_effects(expanded, subgroup_cols, args)
    write_table(subgroup, args.output_dir, "subgroup_level_effects")

    identity_cols = [
        "feature_id", "layer", "alpha", "intervention_position", "feature_role",
        "feature_contrast_name", "axis_mapped", "mapped_contrast_name", "identity_id",
        "identity_label", "identity_role", "context_condition", "question_polarity",
        "feature_estimate_type", "steering_direction_label",
    ]
    identity_cols = [col for col in identity_cols if col in identity_long.columns]
    identity_effects = summarize_effects(identity_long, identity_cols, args) if not identity_long.empty else pd.DataFrame()
    identity_effects = merge_metadata(identity_effects, triage, token_summary) if not identity_effects.empty else identity_effects
    write_table(identity_effects, args.output_dir, "identity_level_effects")

    matrix = subgroup[subgroup["context_condition"].eq("ambig")].pivot_table(
        index="feature_id",
        columns="mapped_contrast_name",
        values="mean_stereotype_preference_delta",
        aggfunc="mean",
    ).reset_index() if not subgroup.empty else pd.DataFrame()
    matrix.to_csv(args.output_dir / "feature_x_subgroup_matrix.csv", index=False)

    # Audit 2.5/2.6: rank on the feature-level inference table (one pre-registered
    # alpha, FDR across features), not the per-(alpha, position) grid, so the
    # rankings are not inflated by the correlated alpha tests. Falls back to the
    # diagnostic table only if the inference table is empty (e.g. wrong --alphas).
    ranking_source = feature_inference if not feature_inference.empty else feature_effects
    rankings = make_rankings(ranking_source)
    rankings.to_csv(args.output_dir / "feature_effect_rankings.csv", index=False)

    plot_feature_bars(feature_effects, args.output_dir, args)
    plot_top_by_contrast(feature_effects, args.output_dir, args, "reducing")
    plot_top_by_contrast(feature_effects, args.output_dir, args, "amplifying")
    plot_amp_suppression_scatter(feature_effects, args.output_dir)
    plot_dose_response(feature_effects, args.output_dir, args)
    plot_answer_role_shift(feature_effects, args.output_dir, args)
    plot_subgroup_heatmap(subgroup, args.output_dir, args)
    plot_disambig_tradeoff(feature_effects, args.output_dir)
    plot_token_activation_vs_effect(feature_effects, args.output_dir)
    feature_card_links_html(feature_effects, args.output_dir)
    make_reports(identity_effects, feature_effects, args.output_dir, args, candidates_table=feature_inference)
    write_validation_summaries(identity_effects, feature_effects, identity_records, args.output_dir)
    write_readme(args.output_dir)
    logger.info("Feature-level causal analysis written to %s", args.output_dir)


if __name__ == "__main__":
    main()
