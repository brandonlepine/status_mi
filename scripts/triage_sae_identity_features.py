#!/usr/bin/env python3
"""Programmatically triage identity-related SAE features into provisional roles."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


DEFAULT_ANALYSIS_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/analysis")
DEFAULT_TOKEN_LEVEL_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/feature_cards/token_level")
DEFAULT_SHARED_SUBSPACE_DIR = Path("/workspace/status_mi/results/geometry/llama-3.1-8b/identity_prompts_final_token/shared_subspace_decomposition")
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/triage")
TEMPLATE_WORDS = {
    "person", "individual", "someone", "somebody", "people", "group", "community",
    "identity", "descriptor", "demographic", "attribute", "label", "category",
    "described", "describes", "mentioned", "refers", "example", "sentence",
    "this", "that", "these", "those", "the", "a", "an", "is", "are", "was",
    "were", "will", "be", "has", "have", "had", "from", "with", "belongs",
    "associated", "applies", ".", ",", ":", ";",
}
ROLE_ORDER = [
    "identity_token_local",
    "sentence_final_integrated",
    "contrast_specific_identity",
    "shared_social_feature",
    "template_or_syntax_artifact",
    "polysemantic_or_unclear",
    "low_signal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triage SAE identity features into provisional interpretability roles.")
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--token_level_dir", type=Path, default=DEFAULT_TOKEN_LEVEL_DIR)
    parser.add_argument("--shared_subspace_dir", type=Path, default=DEFAULT_SHARED_SUBSPACE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--min_abs_cohens_d", type=float, default=0.5)
    parser.add_argument("--min_abs_decoder_cosine", type=float, default=0.03)
    parser.add_argument("--identity_span_local_threshold", type=float, default=0.7)
    parser.add_argument("--final_token_integrated_threshold", type=float, default=0.7)
    parser.add_argument("--max_template_artifact_score_keep", type=float, default=0.5)
    parser.add_argument("--min_contrast_specificity_keep", type=float, default=0.5)
    parser.add_argument("--min_sharedness_score_shared", type=float, default=0.5)
    parser.add_argument("--top_n_per_contrast", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists and is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "intermediate").mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        msg = f"Missing input: {path}"
        if required:
            raise FileNotFoundError(msg)
        warnings.warn(msg)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        if required:
            raise
        warnings.warn(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def coalesce_columns(df: pd.DataFrame, canonical: str, aliases: list[str]) -> pd.DataFrame:
    """Rename the first available alias to a canonical column name."""
    if df.empty or canonical in df.columns:
        return df
    for alias in aliases:
        if alias in df.columns:
            return df.rename(columns={alias: canonical})
    return df


def read_token_table(token_level_dir: Path, layer: int) -> pd.DataFrame:
    search_dirs = [token_level_dir / f"layer_{layer:02d}", token_level_dir]
    frames = []
    for directory in search_dirs:
        parquet_path = directory / "token_feature_activations.parquet"
        csv_path = directory / "token_feature_activations.csv"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                if "layer" not in df.columns:
                    df["layer"] = layer
                frames.append(df)
                break
            except Exception as exc:
                warnings.warn(f"Could not read parquet {parquet_path}; falling back to CSV if present: {exc}")
        if csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
            if "layer" not in df.columns:
                df["layer"] = layer
            frames.append(df)
            break
    if not frames:
        warnings.warn(f"No token-level activation table found for layer {layer} under {token_level_dir}")
        return pd.DataFrame()
    token_df = pd.concat(frames, ignore_index=True, sort=False)
    token_df = normalize_token_columns(token_df)
    if "layer" in token_df.columns:
        token_df = token_df[pd.to_numeric(token_df["layer"], errors="coerce").eq(layer)].copy()
    return token_df


def read_feature_top_tokens(token_level_dir: Path, layer: int) -> pd.DataFrame:
    search_dirs = [token_level_dir / f"layer_{layer:02d}", token_level_dir]
    for directory in search_dirs:
        path = directory / "feature_top_tokens.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            if "layer" not in df.columns:
                df["layer"] = layer
            return normalize_token_columns(df)
    warnings.warn(f"No optional feature_top_tokens.csv found for layer {layer} under {token_level_dir}")
    return pd.DataFrame()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    alias_map = {
        "cosine_with_direction": ["decoder_cosine_with_direction", "decoder_cosine", "cosine", "decoder_alignment"],
        "feature_id": ["feature", "sae_feature", "sae_feature_id"],
        "contrast_name": ["contrast", "contrast_id"],
        "combined_score": ["score", "candidate_score"],
        "auc": ["auc_identity_vs_other_same_axis", "roc_auc"],
    }
    for canonical, aliases in alias_map.items():
        out = coalesce_columns(out, canonical, aliases)
    for col in ["layer", "feature_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    numeric_cols = ["cohens_d", "auc", "diff_mean", "cosine_with_direction", "combined_score", "decoder_cosine"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    missing = [col for col in ["layer", "feature_id"] if col not in out.columns]
    if missing:
        warnings.warn(f"Table is missing expected columns {missing}; available columns: {list(out.columns)}")
    return out


def normalize_token_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    alias_map = {
        "feature_id": ["feature", "sae_feature", "sae_feature_id"],
        "prompt_id": ["row_idx", "prompt_index", "example_id"],
        "token_str": ["token", "token_string", "str_token"],
        "token_feature_activation": ["activation", "feature_activation", "value"],
        "is_identity_span_token": ["inside_identity_span", "identity_span_token", "is_identity_token"],
        "canonical_label": ["identity_label", "label"],
        "template_id": ["template", "template_name"],
        "final_token_feature_activation": ["final_activation", "final_token_activation"],
    }
    for canonical, aliases in alias_map.items():
        out = coalesce_columns(out, canonical, aliases)
    for col in ["layer", "feature_id", "token_idx", "token_feature_activation", "final_token_feature_activation"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "is_identity_span_token" in out.columns:
        out["is_identity_span_token"] = out["is_identity_span_token"].fillna(False).astype(bool)
    return out


def entropy(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if len(arr) <= 1:
        return 0.0
    probs = arr / arr.sum()
    h = float(-(probs * np.log(probs)).sum())
    return h / math.log(len(probs)) if len(probs) > 1 else 0.0


def categorical_entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    return entropy(counts.to_numpy(dtype=float))


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def ratio(num: float, den: float) -> float:
    return float(num / max(float(den), 1e-9))


def add_top_membership(joined: pd.DataFrame, candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frames = []
    sources = []
    if not joined.empty:
        j = joined.copy()
        if "cosine_with_direction" not in j.columns and "decoder_cosine" in j.columns:
            j["cosine_with_direction"] = j["decoder_cosine"]
        sources.append(j)
    if not candidates.empty:
        c = candidates.copy()
        if "cosine_with_direction" not in c.columns and "decoder_cosine" in c.columns:
            c["cosine_with_direction"] = c["decoder_cosine"]
        sources.append(c)
    if not sources:
        return pd.DataFrame(columns=["layer", "feature_id", "contrast_name", "axis", "membership_type"])
    source = pd.concat(sources, ignore_index=True, sort=False)
    source = source.dropna(subset=["layer", "feature_id", "contrast_name"])
    for (layer, contrast), group in source.groupby(["layer", "contrast_name"], sort=True):
        axis = group["axis"].dropna().astype(str).iloc[0] if "axis" in group.columns and group["axis"].notna().any() else ""
        rankings = {}
        if "cohens_d" in group.columns:
            rankings["abs_cohens_d"] = group.assign(_score=group["cohens_d"].abs()).sort_values("_score", ascending=False)
        if "cosine_with_direction" in group.columns:
            rankings["abs_decoder_cosine"] = group.assign(_score=group["cosine_with_direction"].abs()).sort_values("_score", ascending=False)
        if "combined_score" in group.columns:
            rankings["combined_score"] = group.sort_values("combined_score", ascending=False)
        for membership_type, ranked in rankings.items():
            for feature_id in ranked["feature_id"].drop_duplicates().head(top_n):
                frames.append({
                    "layer": int(layer),
                    "feature_id": int(feature_id),
                    "contrast_name": contrast,
                    "axis": axis,
                    "membership_type": membership_type,
                })
    return pd.DataFrame(frames).drop_duplicates()


def aggregate_signal_metrics(joined: pd.DataFrame, alignment: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    sources = []
    for df in [joined, candidates, alignment]:
        if not df.empty:
            sources.append(normalize_columns(df).copy())
    if not sources:
        return pd.DataFrame()
    source = pd.concat(sources, ignore_index=True, sort=False)
    source = source.dropna(subset=["layer", "feature_id"])
    rows = []
    for (layer, feature_id), group in source.groupby(["layer", "feature_id"], sort=True):
        cohens = group["cohens_d"] if "cohens_d" in group.columns else pd.Series(dtype=float)
        auc = group["auc"] if "auc" in group.columns else pd.Series(dtype=float)
        cos = group["cosine_with_direction"] if "cosine_with_direction" in group.columns else pd.Series(dtype=float)
        combined = group["combined_score"] if "combined_score" in group.columns else pd.Series(dtype=float)
        contrast_names = group["contrast_name"].dropna().astype(str) if "contrast_name" in group.columns else pd.Series(dtype=str)
        axes = group["axis"].dropna().astype(str) if "axis" in group.columns else pd.Series(dtype=str)
        identity_cols = [col for col in ["identity_a", "identity_b", "identity_id"] if col in group.columns]
        identities_seen: set[str] = set()
        for col in identity_cols:
            identities_seen.update(group[col].dropna().astype(str).tolist())
        top_select = ""
        if "cohens_d" in group.columns and group["cohens_d"].notna().any() and "contrast_name" in group.columns:
            top_select = str(group.loc[group["cohens_d"].abs().idxmax(), "contrast_name"])
        top_align = ""
        if "cosine_with_direction" in group.columns and group["cosine_with_direction"].notna().any() and "contrast_name" in group.columns:
            top_align = str(group.loc[group["cosine_with_direction"].abs().idxmax(), "contrast_name"])
        axis = axes.mode()
        if "cohens_d" in group.columns and group["cohens_d"].notna().any() and "axis" in group.columns:
            axis = pd.Series([str(group.loc[group["cohens_d"].abs().idxmax(), "axis"])])
        rows.append({
            "layer": int(layer),
            "feature_id": int(feature_id),
            "n_contrasts_seen": int(contrast_names.nunique()),
            "n_axes_seen": int(axes.nunique()),
            "n_identities_seen": int(len(identities_seen)),
            "max_abs_cohens_d": float(cohens.abs().max()) if len(cohens) else 0.0,
            "max_auc_distance_from_0_5": float((auc - 0.5).abs().max()) if len(auc) else 0.0,
            "max_abs_decoder_cosine": float(cos.abs().max()) if len(cos) else 0.0,
            "mean_abs_decoder_cosine": float(cos.abs().mean()) if len(cos) else 0.0,
            "max_combined_score": float(combined.max()) if len(combined) else 0.0,
            "top_contrast_by_selectivity": top_select,
            "top_contrast_by_decoder_alignment": top_align,
            "signal_top_axis": axis.iloc[0] if not axis.empty else "",
            "top_contrasts_by_selectivity": ";".join(group.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)["contrast_name"].dropna().astype(str).drop_duplicates().head(10)) if "cohens_d" in group.columns and "contrast_name" in group.columns else "",
            "top_contrasts_by_alignment": ";".join(group.sort_values("cosine_with_direction", key=lambda s: s.abs(), ascending=False)["contrast_name"].dropna().astype(str).drop_duplicates().head(10)) if "cosine_with_direction" in group.columns and "contrast_name" in group.columns else "",
        })
    return pd.DataFrame(rows)


def aggregate_membership(membership: pd.DataFrame) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame(columns=["layer", "feature_id", "n_contrasts_where_top_feature", "n_axes_where_top_feature", "membership_axes"])
    return (
        membership.groupby(["layer", "feature_id"], sort=True)
        .agg(
            n_contrasts_where_top_feature=("contrast_name", "nunique"),
            n_axes_where_top_feature=("axis", "nunique"),
            membership_axes=("axis", lambda s: ";".join(sorted(set(s.dropna().astype(str))))),
        )
        .reset_index()
    )


def aggregate_identity(identity_df: pd.DataFrame) -> pd.DataFrame:
    if identity_df.empty:
        return pd.DataFrame()
    identity_df = normalize_columns(identity_df)
    rows = []
    for (layer, feature_id), group in identity_df.groupby(["layer", "feature_id"], sort=True):
        group = group.copy()
        value_col = "mean_identity" if "mean_identity" in group.columns else "diff_mean"
        group[value_col] = pd.to_numeric(group[value_col], errors="coerce").fillna(0.0).clip(lower=0)
        sorted_group = group.sort_values(value_col, ascending=False)
        top = sorted_group.iloc[0]
        second = sorted_group.iloc[1] if len(sorted_group) > 1 else top
        positives = sorted_group[value_col].to_numpy()
        positive_sum = float(positives[positives > 0].sum())
        top_axis = str(top.get("axis", ""))
        top_axis_fraction = float((sorted_group.head(10)["axis"].astype(str).eq(top_axis)).mean()) if top_axis else 0.0
        rows.append({
            "layer": int(layer),
            "feature_id": int(feature_id),
            "top_identity": str(top.get("identity_id", "")),
            "top_identity_label": str(top.get("canonical_label", "")),
            "top_axis": top_axis,
            "top_identity_mean_activation": float(top[value_col]),
            "second_identity_mean_activation": float(second[value_col]),
            "identity_specificity_ratio": ratio(float(top[value_col]), float(second[value_col])),
            "top_identity_share": ratio(float(top[value_col]), positive_sum),
            "top_axis_fraction": top_axis_fraction,
            "same_axis_selectivity_score": top_axis_fraction,
            "axis_entropy": categorical_entropy(sorted_group.loc[sorted_group[value_col] > 0, "axis"]) if "axis" in sorted_group.columns else 0.0,
            "identity_entropy": entropy(positives),
            "top_identities_by_activation": ";".join(
                f"{row.identity_id}:{getattr(row, 'canonical_label', '')}:{getattr(row, value_col):.4g}"
                for row in sorted_group.head(8).itertuples(index=False)
            ),
        })
    return pd.DataFrame(rows)


def token_string_norm(token: object) -> str:
    return str(token).strip().lower().replace("Ġ", "").replace("▁", "")


def aggregate_token_metrics(token_df: pd.DataFrame) -> pd.DataFrame:
    if token_df.empty:
        return pd.DataFrame()
    df = normalize_token_columns(token_df)
    required = {"layer", "feature_id", "token_feature_activation"}
    missing = required - set(df.columns)
    if missing:
        warnings.warn(f"Skipping token metrics because token table is missing {sorted(missing)}")
        return pd.DataFrame()
    if "prompt_id" not in df.columns:
        warnings.warn("Token table has no prompt_id column; treating each feature as one aggregate prompt.")
        df["prompt_id"] = 0
    if "token_str" not in df.columns:
        df["token_str"] = ""
    if "is_identity_span_token" not in df.columns:
        warnings.warn("Token table has no is_identity_span_token column; identity-span localization will be zero.")
        df["is_identity_span_token"] = False
    for col in ["token_feature_activation", "final_token_feature_activation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "is_special_token" in df.columns:
        df = df[~df["is_special_token"].astype(bool)]
    elif {"token_start_char", "token_end_char"}.issubset(df.columns):
        df = df[pd.to_numeric(df["token_end_char"], errors="coerce").fillna(0) > pd.to_numeric(df["token_start_char"], errors="coerce").fillna(0)]
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (layer, feature_id), group in tqdm(list(df.groupby(["layer", "feature_id"], sort=True)), desc="token metrics", leave=False):
        prompt_rows = []
        for prompt_id, pg in group.groupby("prompt_id", sort=False):
            max_token = float(pg["token_feature_activation"].max())
            if "final_token_feature_activation" in pg.columns and pg["final_token_feature_activation"].notna().any():
                final_token = float(pg["final_token_feature_activation"].max())
            elif "token_idx" in pg.columns and pg["token_idx"].notna().any():
                final_token = float(pg.sort_values("token_idx").iloc[-1]["token_feature_activation"])
            else:
                final_token = 0.0
            span = pg[pg["is_identity_span_token"].astype(bool)] if "is_identity_span_token" in pg.columns else pd.DataFrame()
            max_span = float(span["token_feature_activation"].max()) if not span.empty else 0.0
            mean_span = float(span["token_feature_activation"].mean()) if not span.empty else 0.0
            max_row = pg.loc[pg["token_feature_activation"].idxmax()]
            prompt_rows.append({
                "prompt_id": prompt_id,
                "max_token_activation": max_token,
                "final_token_activation": final_token,
                "max_identity_span_activation": max_span,
                "mean_identity_span_activation": mean_span,
                "identity_span_ratio": ratio(max_span, max_token),
                "final_ratio": ratio(final_token, max_token),
                "max_inside_span": bool(max_row.get("is_identity_span_token", False)),
                "token_idx_of_max_activation": int(max_row.get("token_idx", -1)) if pd.notna(max_row.get("token_idx", np.nan)) else -1,
                "token_str_of_max_activation": str(max_row.get("token_str", "")),
                "family": str(max_row.get("family", "")),
                "template_id": str(max_row.get("template_id", "")),
                "axis": str(max_row.get("axis", "")),
                "identity_id": str(max_row.get("identity_id", "")),
            })
        prompt_df = pd.DataFrame(prompt_rows)
        top_tokens = (
            group.assign(token_norm=group["token_str"].map(token_string_norm))
            .groupby("token_norm")["token_feature_activation"].sum()
            .sort_values(ascending=False)
            .head(30)
        )
        top_token_strings = ";".join(top_tokens.index.astype(str))
        top_tokens_with_span = (
            group.assign(token_norm=group["token_str"].map(token_string_norm))
            .groupby("token_norm")
            .agg(
                activation=("token_feature_activation", "sum"),
                identity_span_fraction=("is_identity_span_token", "mean"),
            )
            .sort_values("activation", ascending=False)
            .head(30)
        )
        template_fraction = float(sum(token in TEMPLATE_WORDS for token in top_tokens.index[:20]) / max(1, min(20, len(top_tokens))))
        top_token_identity_span_fraction = float(top_tokens_with_span["identity_span_fraction"].head(20).mean()) if not top_tokens_with_span.empty else 0.0
        family_entropy = categorical_entropy(prompt_df["family"])
        template_entropy = categorical_entropy(prompt_df["template_id"])
        identity_span_score = clip01(float(prompt_df["identity_span_ratio"].median()))
        final_score = clip01(float(prompt_df["final_ratio"].median()))
        if identity_span_score >= 0.7:
            localization_type = "identity_span_local"
        elif final_score >= 0.7:
            localization_type = "final_token_integrated"
        elif template_fraction >= 0.5:
            localization_type = "template_context"
        else:
            localization_type = "diffuse_or_unclear"
        rows.append({
            "layer": int(layer),
            "feature_id": int(feature_id),
            "max_token_activation": float(prompt_df["max_token_activation"].max()),
            "final_token_activation": float(prompt_df["final_token_activation"].median()),
            "max_identity_span_activation": float(prompt_df["max_identity_span_activation"].median()),
            "mean_identity_span_activation": float(prompt_df["mean_identity_span_activation"].median()),
            "identity_span_localization_score": identity_span_score,
            "identity_span_localization_score_mean": clip01(float(prompt_df["identity_span_ratio"].mean())),
            "final_token_integration_score": final_score,
            "fraction_top_tokens_inside_identity_span": float(prompt_df["max_inside_span"].mean()),
            "top_token_identity_span_fraction": top_token_identity_span_fraction,
            "top_token_strings": top_token_strings,
            "fraction_top_tokens_template_words": template_fraction,
            "fraction_top_prompts_single_family": float(prompt_df["family"].value_counts(normalize=True).iloc[0]) if not prompt_df.empty else 0.0,
            "family_entropy": family_entropy,
            "template_entropy": template_entropy,
            "token_entropy": entropy(top_tokens.to_numpy(dtype=float)),
            "cross_axis_activation_score": categorical_entropy(prompt_df["axis"]) if "axis" in prompt_df.columns else 0.0,
            "feature_localization_type": localization_type,
        })
    return pd.DataFrame(rows)


def aggregate_feature_top_tokens(top_tokens_df: pd.DataFrame) -> pd.DataFrame:
    if top_tokens_df.empty:
        return pd.DataFrame()
    df = normalize_token_columns(top_tokens_df)
    if not {"layer", "feature_id", "token_str"}.issubset(df.columns):
        warnings.warn("feature_top_tokens.csv is missing layer/feature_id/token_str; skipping fallback top-token metrics.")
        return pd.DataFrame()
    if "token_feature_activation" not in df.columns:
        df["token_feature_activation"] = 1.0
    rows = []
    for (layer, feature_id), group in df.groupby(["layer", "feature_id"], sort=True):
        top_tokens = (
            group.assign(token_norm=group["token_str"].map(token_string_norm))
            .groupby("token_norm")["token_feature_activation"].sum()
            .sort_values(ascending=False)
            .head(30)
        )
        rows.append({
            "layer": int(layer),
            "feature_id": int(feature_id),
            "top_token_strings": ";".join(top_tokens.index.astype(str)),
            "fraction_top_tokens_template_words": float(sum(token in TEMPLATE_WORDS for token in top_tokens.index[:20]) / max(1, min(20, len(top_tokens)))),
            "token_entropy": entropy(top_tokens.to_numpy(dtype=float)),
        })
    return pd.DataFrame(rows)


def aggregate_shared_loadings(shared_dir: Path, layers: list[int]) -> pd.DataFrame:
    if not shared_dir or not shared_dir.exists():
        warnings.warn(f"Optional shared-subspace directory not found: {shared_dir}")
        return pd.DataFrame()
    metric_paths = [
        shared_dir / "metrics" / "contrast_pc_loadings.csv",
        shared_dir / "metrics" / "shared_pc_identity_rankings.csv",
        shared_dir / "metrics" / "cross_axis_projection_summary.csv",
        shared_dir / "metrics" / "decomposition_metrics.csv",
    ]
    frames = []
    for path in metric_paths:
        df = safe_read_csv(path)
        if df.empty:
            continue
        df = normalize_columns(df)
        df["_shared_metric_source"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    source = pd.concat(frames, ignore_index=True, sort=False)
    if "layer" in source.columns:
        source = source[source["layer"].isin(layers)].copy()
    numeric_candidates = [
        col for col in source.columns
        if col not in {"layer", "feature_id"}
        and pd.api.types.is_numeric_dtype(source[col])
        and any(key in col.lower() for key in ["loading", "shared", "projection", "variance", "score"])
    ]
    if not numeric_candidates:
        return pd.DataFrame()
    score_col = "abs_loading" if "abs_loading" in numeric_candidates else numeric_candidates[0]
    source["_shared_score"] = pd.to_numeric(source[score_col], errors="coerce").abs()
    max_score = float(source["_shared_score"].max()) if source["_shared_score"].notna().any() else 0.0
    if max_score > 0:
        source["_shared_score"] = source["_shared_score"] / max_score
    if {"layer", "feature_id"}.issubset(source.columns) and source["feature_id"].notna().any():
        return (
            source.dropna(subset=["layer", "feature_id"])
            .groupby(["layer", "feature_id"], sort=True)["_shared_score"]
            .max()
            .reset_index(name="shared_pc_loading_score")
        )
    if "layer" in source.columns:
        return source.groupby("layer", sort=True)["_shared_score"].mean().reset_index(name="shared_pc_loading_score")
    return pd.DataFrame({"layer": layers, "shared_pc_loading_score": float(source["_shared_score"].mean())})


def complete_feature_table(layers: list[int], *dfs: pd.DataFrame) -> pd.DataFrame:
    keys = []
    for df in dfs:
        if not df.empty and {"layer", "feature_id"}.issubset(df.columns):
            keys.append(df[["layer", "feature_id"]])
    if not keys:
        return pd.DataFrame(columns=["layer", "feature_id"])
    out = pd.concat(keys, ignore_index=True).drop_duplicates()
    out = out[out["layer"].isin(layers)].copy()
    out["layer"] = out["layer"].astype(int)
    out["feature_id"] = out["feature_id"].astype(int)
    return out


def assign_roles(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out_rows = []
    for row in df.fillna(0).itertuples(index=False):
        max_d = float(getattr(row, "max_abs_cohens_d", 0.0))
        max_cos = float(getattr(row, "max_abs_decoder_cosine", 0.0))
        artifact = float(getattr(row, "template_artifact_score", 0.0))
        span_score = float(getattr(row, "identity_span_localization_score", 0.0))
        final_score = float(getattr(row, "final_token_integration_score", 0.0))
        shared = float(getattr(row, "sharedness_score", 0.0))
        specificity = float(getattr(row, "contrast_specificity_score", 0.0))
        axes = int(getattr(row, "n_axes_where_top_feature", 0))
        norm_d = clip01(max_d / max(args.min_abs_cohens_d * 2, 1e-9))
        norm_cos = clip01(max_cos / max(args.min_abs_decoder_cosine * 4, 1e-9))
        if max_d < args.min_abs_cohens_d and max_cos < args.min_abs_decoder_cosine:
            role = "low_signal"
            confidence = clip01(1 - max(norm_d, norm_cos))
            keep = False
        elif artifact >= args.max_template_artifact_score_keep:
            role = "template_or_syntax_artifact"
            confidence = artifact
            keep = False
        elif span_score >= args.identity_span_local_threshold and max_d >= args.min_abs_cohens_d:
            role = "identity_token_local"
            confidence = float(np.mean([span_score, norm_d, 1 - artifact]))
            keep = True
        elif final_score >= args.final_token_integrated_threshold and max_d >= args.min_abs_cohens_d:
            role = "sentence_final_integrated"
            confidence = float(np.mean([final_score, norm_d, 1 - artifact]))
            keep = True
        elif shared >= args.min_sharedness_score_shared and axes >= 3:
            role = "shared_social_feature"
            confidence = shared
            keep = max_d >= args.min_abs_cohens_d and artifact < args.max_template_artifact_score_keep
        elif specificity >= args.min_contrast_specificity_keep and max_d >= args.min_abs_cohens_d and max_cos >= args.min_abs_decoder_cosine:
            role = "contrast_specific_identity"
            confidence = float(np.mean([specificity, norm_d, norm_cos, 1 - artifact]))
            keep = True
        else:
            role = "polysemantic_or_unclear"
            confidence = float(getattr(row, "polysemanticity_score", 0.0))
            keep = False
        keep = bool(keep and max_d >= args.min_abs_cohens_d and artifact < args.max_template_artifact_score_keep)
        priority = "high" if keep and confidence >= 0.7 and max_d >= args.min_abs_cohens_d * 1.5 else ("medium" if keep else "low")
        reason = (
            f"{role}: span={span_score:.2f}, final={final_score:.2f}, artifact={artifact:.2f}, "
            f"shared={shared:.2f}, specificity={specificity:.2f}, max|d|={max_d:.2f}, max|cos|={max_cos:.3f}, "
            f"top_axis={getattr(row, 'top_axis', '')}."
        )
        values = row._asdict()
        values.update({
            "provisional_role": role,
            "role_confidence": clip01(confidence),
            "keep_for_intervention": keep,
            "intervention_priority": priority,
            "reason": reason,
        })
        out_rows.append(values)
    return pd.DataFrame(out_rows)


def compute_scores(df: pd.DataFrame, shared_layer_scores: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "n_contrasts_where_top_feature", "n_axes_where_top_feature", "top_axis_fraction",
        "max_abs_cohens_d", "max_abs_decoder_cosine", "identity_span_localization_score",
        "final_token_integration_score", "fraction_top_tokens_template_words", "family_entropy",
        "template_entropy", "axis_entropy", "identity_entropy", "token_entropy",
        "cross_axis_activation_score", "same_axis_selectivity_score",
    ]:
        if col not in out.columns:
            out[col] = 0.0
    for col in ["n_contrasts_seen", "n_axes_seen", "n_identities_seen"]:
        if col not in out.columns:
            out[col] = 0
    if "feature_localization_type" not in out.columns:
        out["feature_localization_type"] = "diffuse_or_unclear"
    out["contrast_specificity_score"] = (
        1 - np.minimum(1, (out["n_axes_where_top_feature"].clip(lower=1) - 1) / 4)
    ) * 0.6 + out["top_axis_fraction"].fillna(0) * 0.2 + np.minimum(1, out["max_abs_cohens_d"] / 2) * 0.2
    out["contrast_specificity_score"] = out["contrast_specificity_score"].map(clip01)
    out["shared_pc_loading_score"] = 0.0
    if not shared_layer_scores.empty:
        shared_keys = ["layer", "feature_id"] if "feature_id" in shared_layer_scores.columns else ["layer"]
        out = out.merge(shared_layer_scores, on=shared_keys, how="left", suffixes=("", "_shared_proxy"))
        if "shared_pc_loading_score_shared_proxy" in out.columns:
            out["shared_pc_loading_score"] = out["shared_pc_loading_score_shared_proxy"].fillna(0)
            out = out.drop(columns=["shared_pc_loading_score_shared_proxy"])
        elif "shared_pc_loading_score" in out.columns:
            out["shared_pc_loading_score"] = out["shared_pc_loading_score"].fillna(0)
    out["sharedness_score"] = (
        0.5 * np.minimum(1, out["n_axes_where_top_feature"] / 5)
        + 0.3 * np.minimum(1, out["n_contrasts_where_top_feature"] / 10)
        + 0.2 * out["shared_pc_loading_score"].fillna(0).clip(0, 1)
    ).map(clip01)
    out["template_artifact_score"] = (
        0.4 * out["fraction_top_tokens_template_words"].fillna(0)
        + 0.3 * (1 - out["family_entropy"].fillna(1))
        + 0.2 * (1 - out["template_entropy"].fillna(1))
        + 0.1 * (1 - out["identity_span_localization_score"].fillna(0))
    ).map(clip01)
    out["polysemanticity_score"] = (
        0.35 * out["axis_entropy"].fillna(0)
        + 0.35 * out["identity_entropy"].fillna(0)
        + 0.2 * out["token_entropy"].fillna(0)
        + 0.1 * (1 - out["top_axis_fraction"].fillna(0))
    ).map(clip01)
    out["cross_axis_activation_score"] = out["cross_axis_activation_score"].fillna(out["axis_entropy"]).map(clip01)
    return out


def save_fig(fig: plt.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=220)
    fig.savefig(path_no_suffix.with_suffix(".pdf"))
    plt.close(fig)


def make_figures(triage: pd.DataFrame, candidates_triaged: pd.DataFrame, output_dir: Path) -> None:
    if triage.empty:
        return
    role_counts = triage["provisional_role"].value_counts().reindex(ROLE_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    role_counts.plot(kind="bar", ax=ax, color="#2E86AB")
    ax.set_title("SAE feature triage role counts")
    ax.set_ylabel("Features")
    ax.tick_params(axis="x", rotation=35)
    save_fig(fig, output_dir / "figures" / "role_counts")

    score_cols = [
        "identity_span_localization_score", "final_token_integration_score",
        "template_artifact_score", "sharedness_score", "contrast_specificity_score",
    ]
    score_df = triage.melt(id_vars=["provisional_role"], value_vars=[c for c in score_cols if c in triage.columns], var_name="score", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 6))
    if sns is not None:
        sns.violinplot(data=score_df, x="score", y="value", ax=ax, cut=0)
    else:
        score_df.boxplot(column="value", by="score", ax=ax)
    ax.set_ylim(0, 1.02)
    ax.set_title("Triage score distributions")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, output_dir / "figures" / "score_distributions")

    heat = triage.pivot_table(index="top_axis", columns="provisional_role", values="feature_id", aggfunc="count", fill_value=0)
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(heat))))
        if sns is not None:
            sns.heatmap(heat, annot=True, fmt="d", cmap="Blues", ax=ax)
        else:
            im = ax.imshow(heat.to_numpy(), cmap="Blues")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(heat.index)), heat.index)
        ax.set_title("Feature roles by top axis")
        save_fig(fig, output_dir / "figures" / "role_by_axis_heatmap")

    if not candidates_triaged.empty and "contrast_name" in candidates_triaged.columns:
        kept = candidates_triaged[candidates_triaged["keep_for_intervention"].astype(bool)]
        counts = kept.groupby("contrast_name")["feature_id"].nunique().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 5))
        counts.plot(kind="bar", ax=ax, color="#4E79A7")
        ax.set_title("Kept intervention candidate features by contrast")
        ax.set_ylabel("Kept features")
        ax.tick_params(axis="x", rotation=75)
        save_fig(fig, output_dir / "figures" / "keep_for_intervention_by_contrast")

    fig, ax = plt.subplots(figsize=(8, 6))
    if sns is not None:
        sns.scatterplot(data=triage, x="template_artifact_score", y="max_abs_cohens_d", hue="provisional_role", ax=ax, alpha=0.75)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        ax.scatter(triage["template_artifact_score"], triage["max_abs_cohens_d"], alpha=0.7)
    ax.set_title("Selectivity vs template artifact score")
    save_fig(fig, output_dir / "figures" / "scatter_selectivity_vs_artifact")

    fig, ax = plt.subplots(figsize=(8, 6))
    if sns is not None:
        sns.scatterplot(data=triage, x="sharedness_score", y="contrast_specificity_score", hue="provisional_role", ax=ax, alpha=0.75)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        ax.scatter(triage["sharedness_score"], triage["contrast_specificity_score"], alpha=0.7)
    ax.set_title("Sharedness vs contrast specificity")
    save_fig(fig, output_dir / "figures" / "scatter_sharedness_vs_specificity")


def feature_card_link(output_dir: Path, layer: int, feature_id: int) -> str:
    candidates = [
        output_dir.parent / "feature_cards" / f"layer_{layer:02d}" / f"feature_{feature_id:05d}.html",
        output_dir.parent.parent / "feature_cards" / f"layer_{layer:02d}" / f"feature_{feature_id:05d}.html",
        output_dir.parent.parent / "feature_cards" / f"layer_{layer:02d}" / f"feature_{feature_id}.html",
    ]
    for path in candidates:
        if path.exists():
            return html.escape(str(path.relative_to(output_dir)))
    return ""


def write_html(triage: pd.DataFrame, output_dir: Path) -> None:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    kept = triage[triage["keep_for_intervention"].astype(bool)].copy()
    kept["_priority_order"] = kept["intervention_priority"].map(priority_order).fillna(9)
    kept = kept.sort_values(["_priority_order", "role_confidence", "max_abs_cohens_d"], ascending=[True, False, False]).head(100)
    summary = triage.groupby("provisional_role").agg(n_features=("feature_id", "count"), n_keep=("keep_for_intervention", "sum"), mean_confidence=("role_confidence", "mean")).reset_index()
    role_options = "\n".join(f"<option value='{html.escape(str(role))}'>{html.escape(str(role))}</option>" for role in sorted(triage["provisional_role"].dropna().astype(str).unique()))
    axis_options = "\n".join(f"<option value='{html.escape(str(axis))}'>{html.escape(str(axis))}</option>" for axis in sorted(triage["top_axis"].dropna().astype(str).unique()) if axis)
    priority_options = "\n".join(f"<option value='{html.escape(str(priority))}'>{html.escape(str(priority))}</option>" for priority in ["high", "medium", "low"])
    summary_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.provisional_role))}</td><td>{int(row.n_features)}</td><td>{int(row.n_keep)}</td><td>{row.mean_confidence:.3f}</td></tr>"
        for row in summary.itertuples(index=False)
    )
    kept_html_rows = []
    for row in kept.itertuples(index=False):
        card = feature_card_link(output_dir, int(row.layer), int(row.feature_id))
        card_cell = f"<a href='{card}'>card</a>" if card else ""
        kept_html_rows.append(
            f"<tr data-role='{html.escape(str(row.provisional_role))}' data-axis='{html.escape(str(row.top_axis))}' data-priority='{html.escape(str(row.intervention_priority))}'>"
            f"<td>{int(row.layer)}</td><td>{int(row.feature_id)}</td><td>{html.escape(str(row.provisional_role))}</td>"
            f"<td>{html.escape(str(row.intervention_priority))}</td><td>{row.role_confidence:.3f}</td>"
            f"<td>{html.escape(str(row.top_axis))}</td><td>{html.escape(str(row.top_identity_label))}</td>"
            f"<td>{row.max_abs_cohens_d:.3f}</td><td>{row.template_artifact_score:.3f}</td><td>{card_cell}</td>"
            f"<td>{html.escape(str(row.reason))}</td></tr>"
        )
    kept_rows = "\n".join(kept_html_rows)
    (output_dir / "triage_index.html").write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SAE Identity Feature Triage</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}}th{{background:#f5f5f5;position:sticky;top:0}}code{{background:#f3f4f6;padding:1px 3px}}.filters{{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin:12px 0}}label{{font-size:13px;color:#475569}}select{{display:block;margin-top:4px;padding:4px}}</style>
</head><body>
<h1>SAE Identity Feature Triage</h1>
<p>Generated {datetime.now(timezone.utc).isoformat()}</p>
<h2>Role Summary</h2>
<table><tr><th>Role</th><th>Features</th><th>Kept</th><th>Mean confidence</th></tr>{summary_rows}</table>
<h2>Top Kept Features</h2>
<div class="filters">
<label>Role<select id="roleFilter"><option value="">All</option>{role_options}</select></label>
<label>Axis<select id="axisFilter"><option value="">All</option>{axis_options}</select></label>
<label>Priority<select id="priorityFilter"><option value="">All</option>{priority_options}</select></label>
</div>
<table id="keptTable"><tr><th>Layer</th><th>Feature</th><th>Role</th><th>Priority</th><th>Confidence</th><th>Axis</th><th>Identity</th><th>max |d|</th><th>Artifact</th><th>Card</th><th>Reason</th></tr>{kept_rows}</table>
<script>
function applyFilters() {{
  const role = document.getElementById('roleFilter').value;
  const axis = document.getElementById('axisFilter').value;
  const priority = document.getElementById('priorityFilter').value;
  document.querySelectorAll('#keptTable tr[data-role]').forEach(row => {{
    const keep = (!role || row.dataset.role === role) && (!axis || row.dataset.axis === axis) && (!priority || row.dataset.priority === priority);
    row.style.display = keep ? '' : 'none';
  }});
}}
['roleFilter', 'axisFilter', 'priorityFilter'].forEach(id => document.getElementById(id).addEventListener('change', applyFilters));
</script>
</body></html>""")


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    prepare_output(args.output_dir, args.overwrite)
    config = vars(args).copy()
    config["analysis_dir"] = str(args.analysis_dir)
    config["token_level_dir"] = str(args.token_level_dir)
    config["shared_subspace_dir"] = str(args.shared_subspace_dir) if args.shared_subspace_dir else None
    config["output_dir"] = str(args.output_dir)
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "triage_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    joined = normalize_columns(safe_read_csv(args.analysis_dir / "feature_selectivity_alignment_joined.csv"))
    selectivity = normalize_columns(safe_read_csv(args.analysis_dir / "feature_selectivity.csv"))
    identity = safe_read_csv(args.analysis_dir / "feature_identity_selectivity.csv")
    alignment = normalize_columns(safe_read_csv(args.analysis_dir / "decoder_direction_alignment.csv"))
    candidates = normalize_columns(safe_read_csv(args.analysis_dir / "intervention_candidate_features.csv"))
    recon = safe_read_csv(args.analysis_dir / "direction_reconstruction.csv")
    if joined.empty and not selectivity.empty:
        joined = selectivity.copy()
    membership = add_top_membership(joined, candidates, args.top_n_per_contrast)
    signal = aggregate_signal_metrics(joined, alignment, candidates)
    signal.to_csv(args.output_dir / "intermediate" / "signal_metrics.csv", index=False)
    member_agg = aggregate_membership(membership)
    member_agg.to_csv(args.output_dir / "intermediate" / "top_feature_membership.csv", index=False)
    identity_agg = aggregate_identity(identity)
    identity_agg.to_csv(args.output_dir / "intermediate" / "identity_specificity_metrics.csv", index=False)
    token_aggs = []
    for layer in layers:
        token_df = read_token_table(args.token_level_dir, layer)
        token_metrics = aggregate_token_metrics(token_df)
        top_token_metrics = aggregate_feature_top_tokens(read_feature_top_tokens(args.token_level_dir, layer))
        if token_metrics.empty:
            token_metrics = top_token_metrics
        elif not top_token_metrics.empty:
            fill_cols = [col for col in top_token_metrics.columns if col not in {"layer", "feature_id"} and col not in token_metrics.columns]
            if fill_cols:
                token_metrics = token_metrics.merge(top_token_metrics[["layer", "feature_id", *fill_cols]], on=["layer", "feature_id"], how="outer")
        token_aggs.append(token_metrics)
    token_agg = pd.concat([df for df in token_aggs if not df.empty], ignore_index=True) if any(not df.empty for df in token_aggs) else pd.DataFrame()
    token_agg.to_csv(args.output_dir / "intermediate" / "token_localization_metrics.csv", index=False)
    shared_scores = aggregate_shared_loadings(args.shared_subspace_dir, layers) if args.shared_subspace_dir else pd.DataFrame()
    shared_scores.to_csv(args.output_dir / "intermediate" / "shared_subspace_scores.csv", index=False)
    base = complete_feature_table(layers, signal, member_agg, identity_agg, token_agg)
    for df in [signal, member_agg, identity_agg, token_agg]:
        if not df.empty:
            base = base.merge(df, on=["layer", "feature_id"], how="left")
    if "top_axis" not in base.columns:
        base["top_axis"] = ""
    if "signal_top_axis" in base.columns:
        base["top_axis"] = base["top_axis"].fillna("")
        base.loc[base["top_axis"].astype(str).eq(""), "top_axis"] = base.loc[base["top_axis"].astype(str).eq(""), "signal_top_axis"].fillna("")
    triage_input = compute_scores(base, shared_scores)
    triage_input.to_csv(args.output_dir / "intermediate" / "feature_metric_table_pre_roles.csv", index=False)
    triage = assign_roles(triage_input, args)

    columns_first = [
        "layer", "feature_id", "provisional_role", "role_confidence", "keep_for_intervention",
        "intervention_priority", "reason", "top_axis", "top_identity", "top_identity_label",
        "n_contrasts_seen", "n_axes_seen", "n_identities_seen",
        "top_contrast_by_selectivity", "top_contrast_by_decoder_alignment", "max_abs_cohens_d",
        "max_auc_distance_from_0_5", "max_abs_decoder_cosine", "mean_abs_decoder_cosine", "max_combined_score",
        "top_identity_mean_activation", "second_identity_mean_activation", "identity_specificity_ratio",
        "top_axis_fraction", "same_axis_selectivity_score",
        "n_contrasts_where_top_feature", "n_axes_where_top_feature", "identity_span_localization_score",
        "identity_span_localization_score_mean", "final_token_integration_score", "fraction_top_tokens_inside_identity_span",
        "top_token_identity_span_fraction", "feature_localization_type",
        "fraction_top_tokens_template_words", "template_artifact_score", "contrast_specificity_score",
        "sharedness_score", "cross_axis_activation_score", "shared_pc_loading_score",
        "polysemanticity_score", "axis_entropy", "identity_entropy",
        "family_entropy", "template_entropy", "token_entropy", "top_token_strings", "top_identities_by_activation",
        "top_contrasts_by_selectivity", "top_contrasts_by_alignment",
    ]
    for col in columns_first:
        if col not in triage.columns:
            triage[col] = "" if col.startswith("top_") or col == "reason" else 0
    triage = triage[columns_first + [col for col in triage.columns if col not in columns_first]]
    priority_order = {"high": 0, "medium": 1, "low": 2}
    triage["_priority_order"] = triage["intervention_priority"].map(priority_order).fillna(9)
    triage.sort_values(["keep_for_intervention", "_priority_order", "role_confidence", "max_abs_cohens_d"], ascending=[False, True, False, False]).drop(columns=["_priority_order"]).to_csv(args.output_dir / "feature_triage.csv", index=False)

    summary = triage.groupby("provisional_role").agg(
        n_features=("feature_id", "count"),
        mean_confidence=("role_confidence", "mean"),
        n_keep_for_intervention=("keep_for_intervention", "sum"),
        mean_max_abs_d=("max_abs_cohens_d", "mean"),
        mean_max_abs_decoder_cosine=("max_abs_decoder_cosine", "mean"),
        mean_template_artifact_score=("template_artifact_score", "mean"),
    ).reset_index().rename(columns={"provisional_role": "role"})
    summary.to_csv(args.output_dir / "feature_triage_summary.csv", index=False)
    role_counts = triage.groupby(["layer", "provisional_role"]).agg(n_features=("feature_id", "count"), n_keep_for_intervention=("keep_for_intervention", "sum")).reset_index()
    role_counts.to_csv(args.output_dir / "role_counts.csv", index=False)

    if not candidates.empty:
        cand = candidates.merge(triage[["layer", "feature_id", "provisional_role", "role_confidence", "keep_for_intervention", "intervention_priority", "reason", "template_artifact_score", "sharedness_score", "contrast_specificity_score"]], on=["layer", "feature_id"], how="left")
        priority_order = {"high": 0, "medium": 1, "low": 2}
        cand["_priority_order"] = cand["intervention_priority"].map(priority_order).fillna(9)
        for col in ["keep_for_intervention", "combined_score", "role_confidence"]:
            if col not in cand.columns:
                cand[col] = False if col == "keep_for_intervention" else 0.0
        cand["keep_for_intervention"] = cand["keep_for_intervention"].fillna(False).astype(bool)
        cand.sort_values(["keep_for_intervention", "_priority_order", "combined_score", "role_confidence"], ascending=[False, True, False, False]).drop(columns=["_priority_order"]).to_csv(args.output_dir / "intervention_candidate_features_triaged.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "intervention_candidate_features_triaged.csv", index=False)

    make_figures(triage, candidates.merge(triage[["layer", "feature_id", "keep_for_intervention"]], on=["layer", "feature_id"], how="left") if not candidates.empty else pd.DataFrame(), args.output_dir)
    write_html(triage, args.output_dir)
    print(f"Triage complete: {args.output_dir}")
    print(summary)


if __name__ == "__main__":
    main()
