#!/usr/bin/env python3
"""Build standalone Neuronpedia-style cards for identity-selective SAE features."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


DEFAULT_ANALYSIS_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/analysis")
DEFAULT_TOKEN_LEVEL_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/feature_cards/token_level")
DEFAULT_SAE_ENCODED_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token")
DEFAULT_ACTIVATION_DIR = Path("/workspace/status_mi/results/activations/llama-3.1-8b/identity_prompts_final_token")
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/feature_cards")
LOGIT_CACHE: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standalone HTML feature cards for identity-related SAE features.")
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--token_level_dir", type=Path, default=DEFAULT_TOKEN_LEVEL_DIR)
    parser.add_argument("--sae_encoded_dir", type=Path, default=DEFAULT_SAE_ENCODED_DIR)
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--features", default=None)
    parser.add_argument("--top_n_features", type=int, default=100)
    parser.add_argument("--top_prompts_per_feature", type=int, default=20)
    parser.add_argument("--top_tokens_per_feature", type=int, default=30)
    parser.add_argument("--make_index", action="store_true")
    parser.add_argument("--compute_logit_lens", action="store_true")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_token_table(layer_dir: Path) -> pd.DataFrame:
    parquet_path = layer_dir / "token_feature_activations.parquet"
    csv_path = layer_dir / "token_feature_activations.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    preserved = output_dir / "token_level"
    if output_dir.exists() and any(output_dir.iterdir()) and overwrite:
        for child in output_dir.iterdir():
            if child == preserved:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


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


def select_features(args: argparse.Namespace, layer: int) -> list[int]:
    explicit = parse_int_list(args.features)
    if explicit:
        return sorted(set(explicit))
    candidates = safe_read(args.analysis_dir / "intervention_candidate_features.csv")
    joined = safe_read(args.analysis_dir / "feature_selectivity_alignment_joined.csv")
    identity = safe_read(args.analysis_dir / "feature_identity_selectivity.csv")
    features: set[int] = set()
    if not candidates.empty:
        layer_df = candidates[candidates["layer"].eq(layer)]
        for _, group in layer_df.groupby("contrast_name", sort=True):
            features.update(group.sort_values("combined_score", ascending=False).head(args.top_n_features)["feature_id"].astype(int))
    if not joined.empty:
        layer_df = joined[joined["layer"].eq(layer)]
        for _, group in layer_df.groupby("contrast_name", sort=True):
            features.update(group.sort_values("combined_score", ascending=False).head(args.top_n_features)["feature_id"].astype(int))
            features.update(group.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).head(args.top_n_features)["feature_id"].astype(int))
    if not identity.empty:
        layer_df = identity[identity["layer"].eq(layer)]
        for _, group in layer_df.groupby("identity_id", sort=True):
            features.update(group.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).head(max(5, args.top_n_features // 5))["feature_id"].astype(int))
    return sorted(features)[: args.top_n_features]


def feature_values_for_id(indices: np.ndarray, values: np.ndarray, feature_id: int) -> np.ndarray:
    out = np.zeros(indices.shape[0], dtype=np.float32)
    mask = indices == feature_id
    row_has = mask.any(axis=1)
    if row_has.any():
        rows = np.flatnonzero(row_has)
        cols = mask[row_has].argmax(axis=1)
        out[rows] = values[rows, cols]
    return out


def highlight_identity(prompt: str, form: str) -> str:
    text = html.escape(str(prompt))
    form_escaped = html.escape(str(form))
    if form_escaped and form_escaped.lower() in text.lower():
        idx = text.lower().find(form_escaped.lower())
        return text[:idx] + f"<mark class='identity'>{text[idx:idx + len(form_escaped)]}</mark>" + text[idx + len(form_escaped):]
    return text


def token_heat_html(prompt_token_df: pd.DataFrame) -> str:
    if prompt_token_df.empty:
        return ""
    if "is_special_token" in prompt_token_df.columns:
        prompt_token_df = prompt_token_df[~prompt_token_df["is_special_token"].astype(bool)].copy()
    else:
        prompt_token_df = prompt_token_df[prompt_token_df["token_end_char"] > prompt_token_df["token_start_char"]].copy()
    if prompt_token_df.empty:
        return ""
    max_val = max(float(prompt_token_df["token_feature_activation"].max()), 1e-9)
    spans = []
    for row in prompt_token_df.sort_values("token_idx").itertuples(index=False):
        token = html.escape(str(row.token_str))
        alpha = min(0.85, max(0.05, float(row.token_feature_activation) / max_val))
        classes = "token"
        if bool(row.is_identity_span_token):
            classes += " identity-token"
        if bool(row.is_top_token_for_feature):
            classes += " top-token"
        spans.append(f"<span class='{classes}' style='background-color: rgba(46, 204, 113, {alpha:.2f});' title='act={row.token_feature_activation:.3f}'>{token}</span>")
    return "".join(spans)


def classify_label(identity_means: pd.DataFrame, token_df: pd.DataFrame) -> tuple[str, str, str]:
    if identity_means.empty:
        return "mixed/polysemantic feature", "", ""
    top = identity_means.sort_values("mean_activation", ascending=False).head(10)
    top_axis = str(top["axis"].mode().iloc[0]) if not top["axis"].mode().empty else ""
    top_identity = str(top.iloc[0]["canonical_label"])
    axis_share = (top["axis"].eq(top_axis).mean()) if top_axis else 0
    if float(top.iloc[0]["mean_activation"]) > 0 and axis_share >= 0.7:
        return f"{top_axis.replace('_', ' ')} feature", top_axis, top_identity
    if float(top.iloc[0]["mean_activation"]) > 0:
        return f"{top_identity}-selective feature", top_axis, top_identity
    if not token_df.empty and token_df["feature_localization_type"].eq("identity_span_local").mean() > 0.5:
        return "identity-token-local feature", top_axis, top_identity
    return "mixed/polysemantic feature", top_axis, top_identity


def localization_summary(feature_token_df: pd.DataFrame) -> pd.DataFrame:
    if feature_token_df.empty or "prompt_id" not in feature_token_df.columns:
        return pd.DataFrame()
    prompt_level = feature_token_df.drop_duplicates("prompt_id")
    if prompt_level.empty or "feature_localization_type" not in prompt_level.columns:
        return pd.DataFrame()
    return (
        prompt_level["feature_localization_type"]
        .value_counts(normalize=True)
        .rename_axis("localization_type")
        .reset_index(name="share")
    )


def exemplar_prompt_table(feature_token_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if feature_token_df.empty:
        return pd.DataFrame()
    token_df = feature_token_df.copy()
    if "is_special_token" in token_df.columns:
        token_df = token_df[~token_df["is_special_token"].astype(bool)]
    else:
        token_df = token_df[token_df["token_end_char"] > token_df["token_start_char"]]
    if token_df.empty:
        return pd.DataFrame()
    rows = []
    for prompt_id, group in token_df.groupby("prompt_id", sort=False):
        max_idx = group["token_feature_activation"].idxmax()
        max_row = group.loc[max_idx].to_dict()
        span = group[group["is_identity_span_token"].astype(bool)]
        max_token = float(group["token_feature_activation"].max())
        span_max = float(span["token_feature_activation"].max()) if not span.empty else 0.0
        span_mean = float(span["token_feature_activation"].mean()) if not span.empty else 0.0
        final_token = float(group["final_token_feature_activation"].iloc[0]) if "final_token_feature_activation" in group.columns else 0.0
        if max_token > 0 and span_max >= 0.7 * max_token:
            localization = "identity_span_local"
        elif max_token > 0 and final_token >= 0.7 * max_token:
            localization = "final_token_integrated"
        elif max_token > 0:
            localization = "template_context"
        else:
            localization = "diffuse_or_unclear"
        max_row.update({
            "prompt_id": prompt_id,
            "final_token_feature_activation": final_token,
            "max_token_activation": max_token,
            "max_identity_span_activation": span_max,
            "mean_identity_span_activation": span_mean,
            "feature_localization_type": localization,
        })
        rows.append(max_row)
    prompts = pd.DataFrame(rows)
    prompts = prompts.sort_values(["token_feature_activation", "final_token_feature_activation"], ascending=False)
    return prompts.head(top_n)


def identity_span_token_table(feature_token_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if feature_token_df.empty:
        return pd.DataFrame()
    span_df = feature_token_df[feature_token_df["is_identity_span_token"].astype(bool)].copy()
    if "is_special_token" in span_df.columns:
        span_df = span_df[~span_df["is_special_token"].astype(bool)]
    if span_df.empty:
        return pd.DataFrame()
    return span_df.sort_values("token_feature_activation", ascending=False).head(top_n)


def save_identity_profile(card_dir: Path, layer: int, feature_id: int, identity_means: pd.DataFrame) -> str | None:
    if identity_means.empty:
        return None
    plot_df = identity_means.sort_values("mean_activation", ascending=False).head(20).sort_values("mean_activation")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(plot_df))))
    labels = plot_df["canonical_label"] + " (" + plot_df["axis"].str.replace("_", " ") + ")"
    ax.barh(labels, plot_df["mean_activation"], color="#2E86AB")
    ax.set_title(f"Feature {feature_id} mean activation by identity")
    ax.set_xlabel("Mean final-token activation")
    fig.tight_layout()
    filename = f"feature_{feature_id:05d}_identity_profile.png"
    fig.savefig(card_dir / filename, dpi=180)
    plt.close(fig)
    return filename


def save_token_exemplar_figure(output_dir: Path, layer: int, feature_id: int, feature_token_df: pd.DataFrame, exemplar_df: pd.DataFrame) -> str | None:
    if feature_token_df.empty or exemplar_df.empty:
        return None
    figure_dir = output_dir / "token_exemplars" / f"layer_{layer:02d}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    prompts = exemplar_df["prompt_id"].astype(str).tolist()
    plot_df = feature_token_df[feature_token_df["prompt_id"].astype(str).isin(prompts)].copy()
    if plot_df.empty:
        return None
    if "is_special_token" in plot_df.columns:
        plot_df = plot_df[~plot_df["is_special_token"].astype(bool)]
    else:
        plot_df = plot_df[plot_df["token_end_char"] > plot_df["token_start_char"]]
    if plot_df.empty:
        return None

    max_act = max(float(plot_df["token_feature_activation"].max()), 1e-9)
    n_rows = min(len(prompts), 12)
    fig, ax = plt.subplots(figsize=(18, max(5.5, n_rows * 0.9)))
    ax.axis("off")
    y = 1.0
    row_gap = 0.085
    for prompt_id in prompts[:n_rows]:
        row_tokens = plot_df[plot_df["prompt_id"].astype(str).eq(prompt_id)].sort_values("token_idx")
        if row_tokens.empty:
            continue
        meta = row_tokens.iloc[0]
        label = f"{meta.identity_id} | final={float(meta.final_token_feature_activation):.2f} | max={float(row_tokens['token_feature_activation'].max()):.2f}"
        ax.text(0.01, y, label, transform=ax.transAxes, fontsize=8, color="#475569", va="top")
        x = 0.01
        token_y = y - 0.032
        for token in row_tokens.itertuples(index=False):
            text = str(token.token_str).replace("\n", "\\n")
            if text == "":
                continue
            alpha = min(0.92, max(0.06, float(token.token_feature_activation) / max_act))
            edge = "#facc15" if bool(token.is_identity_span_token) else "#d1d5db"
            linewidth = 1.8 if bool(token.is_identity_span_token) else 0.5
            if bool(token.is_top_token_for_feature):
                edge = "#111827"
                linewidth = 2.0
            ax.text(
                x,
                token_y,
                text,
                transform=ax.transAxes,
                fontsize=9,
                family="monospace",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": (0.18, 0.80, 0.44, alpha),
                    "edgecolor": edge,
                    "linewidth": linewidth,
                },
            )
            x += min(0.12, max(0.024, len(text) * 0.012))
            if x > 0.94:
                x = 0.01
                token_y -= 0.032
        y -= row_gap
        if token_y < y:
            y = token_y - 0.035
        if y < 0.04:
            break
    ax.set_title(
        f"Layer {layer} SAE feature {feature_id}: token-level activation exemplars\n"
        "Green intensity = token activation; yellow outline = identity span; black outline = max token in prompt",
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    stem = figure_dir / f"feature_{feature_id:05d}_token_exemplars"
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return str(stem.with_suffix(".png").relative_to(output_dir))


def compute_logit_effects(args: argparse.Namespace, layer_dir: Path, feature_id: int) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    if not args.compute_logit_lens:
        return [], [], "Not computed."
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        return [], [], "Not computed because transformers is unavailable."
    if args.model_path is None:
        return [], [], "Not computed because --model_path was not provided."
    decoder_path = layer_dir / "sae_decoder.npy"
    if not decoder_path.exists():
        return [], [], "Not computed because sae_decoder.npy is missing."
    cache_key = str(args.model_path)
    if cache_key not in LOGIT_CACHE:
        LOGIT_CACHE[cache_key] = {
            "tokenizer": AutoTokenizer.from_pretrained(args.model_path),
            "model": AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="cpu"),
        }
    tokenizer = LOGIT_CACHE[cache_key]["tokenizer"]
    model = LOGIT_CACHE[cache_key]["model"]
    decoder = np.load(decoder_path, mmap_mode="r")
    if feature_id >= decoder.shape[0]:
        return [], [], "Not computed because feature_id is outside decoder shape."
    w = np.asarray(decoder[feature_id], dtype=np.float32)
    unembed = model.lm_head.weight.detach().float().cpu().numpy()
    logits = unembed @ w
    top_idx = np.argsort(logits)[-20:][::-1]
    bottom_idx = np.argsort(logits)[:20]
    positive = [{"token": tokenizer.decode([int(i)]), "token_id": int(i), "score": float(logits[i])} for i in top_idx]
    negative = [{"token": tokenizer.decode([int(i)]), "token_id": int(i), "score": float(logits[i])} for i in bottom_idx]
    return positive, negative, "Raw decoder @ lm_head projection; final norm not applied."


def build_card(args: argparse.Namespace, layer: int, feature_id: int, layer_dir: Path, metadata: pd.DataFrame, feature_values: np.ndarray, token_df: pd.DataFrame, hook_summary: dict[str, object]) -> dict[str, object]:
    card_dir = args.output_dir / f"layer_{layer:02d}"
    card_dir.mkdir(parents=True, exist_ok=True)
    stats = safe_read(layer_dir / "feature_stats.csv")
    stats_row = stats[stats["feature_id"].eq(feature_id)].iloc[0].to_dict() if not stats.empty and stats["feature_id"].eq(feature_id).any() else {}
    select = safe_read(args.analysis_dir / "feature_selectivity.csv")
    align = safe_read(args.analysis_dir / "decoder_direction_alignment.csv")
    identity = safe_read(args.analysis_dir / "feature_identity_selectivity.csv")
    select_f = select[select["layer"].eq(layer) & select["feature_id"].eq(feature_id)].copy() if not select.empty else pd.DataFrame()
    align_f = align[align["layer"].eq(layer) & align["feature_id"].eq(feature_id)].copy() if not align.empty else pd.DataFrame()
    identity_f = identity[identity["layer"].eq(layer) & identity["feature_id"].eq(feature_id)].copy() if not identity.empty else pd.DataFrame()
    if identity_f.empty:
        temp = metadata[["identity_id", "canonical_label", "axis"]].copy()
        temp["activation"] = feature_values
        identity_means = temp.groupby(["identity_id", "canonical_label", "axis"], sort=True)["activation"].agg(mean_activation="mean", freq=lambda s: float((s > 0).mean())).reset_index()
    else:
        identity_means = identity_f.rename(columns={"mean_identity": "mean_activation", "freq_identity": "freq"})
    feature_token_df = token_df[token_df["feature_id"].eq(feature_id)].copy() if not token_df.empty else pd.DataFrame()
    auto_label, top_axis, top_identity = classify_label(identity_means, feature_token_df)
    loc_summary = localization_summary(feature_token_df)
    exemplar_df = exemplar_prompt_table(feature_token_df, args.top_prompts_per_feature)
    span_token_df = identity_span_token_table(feature_token_df, args.top_tokens_per_feature)
    profile_img = save_identity_profile(card_dir, layer, feature_id, identity_means)
    exemplar_img = save_token_exemplar_figure(args.output_dir, layer, feature_id, feature_token_df, exemplar_df)
    prompt_df = metadata.copy()
    prompt_df["final_token_activation"] = feature_values
    top_prompts = prompt_df.sort_values("final_token_activation", ascending=False).head(args.top_prompts_per_feature)
    positive_logits, negative_logits, logit_note = compute_logit_effects(args, layer_dir, feature_id)

    exemplar_sections = []
    prompt_metrics = []
    for row in exemplar_df.itertuples(index=False):
        prompt_tokens = feature_token_df[feature_token_df["prompt_id"].eq(row.prompt_id)]
        snippet = token_heat_html(prompt_tokens)
        exemplar_sections.append(
            "<div class='exemplar'>"
            f"<div class='exemplar-meta'>max token <strong>{row.token_feature_activation:.3f}</strong> on <code>{html.escape(str(row.token_str))}</code> | "
            f"final <strong>{row.final_token_feature_activation:.3f}</strong> | span max <strong>{row.max_identity_span_activation:.3f}</strong> | "
            f"{html.escape(str(row.feature_localization_type))} | {html.escape(str(row.identity_id))} / {html.escape(str(row.family))}</div>"
            f"<div class='tokens'>{snippet}</div>"
            "</div>"
        )
    for row in exemplar_df.itertuples(index=False):
        prompt_metrics.append({
            "prompt_id": row.prompt_id,
            "final_token_activation": float(row.final_token_feature_activation),
            "max_token_activation": float(row.max_token_activation),
            "max_identity_span_activation": float(row.max_identity_span_activation),
            "mean_identity_span_activation": float(row.mean_identity_span_activation),
            "token_str_of_max_activation": str(row.token_str),
            "feature_localization_type": str(row.feature_localization_type),
        })

    top_tokens = feature_token_df.copy() if not feature_token_df.empty else pd.DataFrame()
    if not top_tokens.empty:
        if "is_special_token" in top_tokens.columns:
            top_tokens = top_tokens[~top_tokens["is_special_token"].astype(bool)]
        else:
            top_tokens = top_tokens[top_tokens["token_end_char"] > top_tokens["token_start_char"]]
        top_tokens = top_tokens.sort_values("token_feature_activation", ascending=False).head(args.top_tokens_per_feature)
    token_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.token_str))}</td><td>{row.token_feature_activation:.3f}</td><td>{html.escape(str(row.identity_id))}</td><td>{html.escape(str(row.axis))}</td><td>{bool(row.is_identity_span_token)}</td><td>{html.escape(str(row.prompt))[:180]}</td></tr>"
        for row in top_tokens.itertuples(index=False)
    ) or "<tr><td colspan='6'>Token-level data missing.</td></tr>"
    span_token_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.token_str))}</td><td>{row.token_feature_activation:.3f}</td><td>{html.escape(str(row.identity_id))}</td><td>{html.escape(str(row.axis))}</td><td>{html.escape(str(row.prompt))[:180]}</td></tr>"
        for row in span_token_df.itertuples(index=False)
    ) or "<tr><td colspan='5'>No identity-span token activations found.</td></tr>"
    loc_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.localization_type))}</td><td>{row.share:.2%}</td></tr>"
        for row in loc_summary.itertuples(index=False)
    ) or "<tr><td colspan='2'>No localization summary available.</td></tr>"
    identity_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.identity_id))}</td><td>{html.escape(str(row.canonical_label))}</td><td>{html.escape(str(row.axis))}</td><td>{getattr(row, 'mean_activation', 0):.4f}</td><td>{getattr(row, 'freq', 0):.3f}</td></tr>"
        for row in identity_means.sort_values("mean_activation", ascending=False).head(15).itertuples(index=False)
    )
    select_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.contrast_name))}</td><td>{row.cohens_d:.3f}</td><td>{row.auc:.3f}</td><td>{row.mean_a:.4f}</td><td>{row.mean_b:.4f}</td></tr>"
        for row in select_f.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).head(15).itertuples(index=False)
    ) or "<tr><td colspan='5'>No selectivity rows found.</td></tr>"
    align_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.contrast_name))}</td><td>{row.cosine_with_direction:.4f}</td><td>{row.signed_dot:.4f}</td></tr>"
        for row in align_f.sort_values("cosine_with_direction", key=lambda s: s.abs(), ascending=False).head(15).itertuples(index=False)
    ) or "<tr><td colspan='3'>No alignment rows found.</td></tr>"
    pos_rows = "\n".join(f"<span class='logit-pos'>{html.escape(item['token'])} ({item['score']:.2f})</span>" for item in positive_logits) or "Not computed."
    neg_rows = "\n".join(f"<span class='logit-neg'>{html.escape(item['token'])} ({item['score']:.2f})</span>" for item in negative_logits) or "Not computed."
    hook_html = html.escape(json.dumps(hook_summary, default=str)[:1200])
    profile_html = f"<img src='{profile_img}' class='profile-img'>" if profile_img else "<p>No profile image.</p>"
    exemplar_img_html = (
        f"<a href='../{exemplar_img}'><img src='../{exemplar_img}' class='exemplar-img'></a>"
        if exemplar_img else
        "<p>No token-exemplar figure available.</p>"
    )
    card_html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Layer {layer} residual SAE feature {feature_id}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f6f7f9; color: #1f2933; }}
.page {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
.grid {{ display: grid; grid-template-columns: minmax(420px, 1fr) minmax(420px, 1fr); gap: 16px; align-items: start; }}
.card {{ background: white; border: 1px solid #d8dee4; border-radius: 10px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
.hero {{ margin: 16px 0; }}
h1 {{ margin: 0 0 4px 0; }}
h2 {{ font-size: 1.05rem; margin-top: 18px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 5px 6px; vertical-align: top; }}
th {{ text-align: left; background: #f3f4f6; }}
mark.identity, .identity-token {{ outline: 2px solid #facc15; }}
.top-token {{ outline: 2px solid #16a34a; }}
.token {{ border-radius: 3px; padding: 1px 2px; white-space: pre-wrap; }}
.exemplar {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 9px; margin: 9px 0; background: #fbfdff; }}
.exemplar-meta {{ font-size: .84rem; color: #475569; margin-bottom: 6px; }}
.tokens {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.8; font-size: .95rem; }}
.snippet-table td:last-child {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.profile-img {{ max-width: 100%; }}
.exemplar-img {{ max-width: 100%; border: 1px solid #d8dee4; border-radius: 8px; background: white; }}
.logit-pos {{ display: inline-block; margin: 2px; padding: 2px 5px; background: #dbeafe; color: #1e40af; border-radius: 4px; }}
.logit-neg {{ display: inline-block; margin: 2px; padding: 2px 5px; background: #fee2e2; color: #991b1b; border-radius: 4px; }}
.note {{ color: #64748b; font-size: .9rem; }}
</style>
</head>
<body><div class="page">
<h1>Layer {layer} residual SAE feature {feature_id}</h1>
<p><strong>{html.escape(auto_label)}</strong></p>
<section class="card hero">
<h2>Token-Level Exemplars</h2>
<p class="note">These are generated from model <code>outputs.hidden_states[{layer}]</code>, encoded through the layer-{layer} residual SAE. Rows are ranked by max non-special token activation. Green intensity = token activation; yellow outline = identity span.</p>
{exemplar_img_html}
{''.join(exemplar_sections) if exemplar_sections else '<p>No token-level exemplars available. The token-level CSV was missing, empty, or did not include this feature.</p>'}
</section>
<div class="grid">
<section class="card">
<h2>Hook Validation</h2>
<p class="note">Expected L{layer}R residual stream: HF <code>hidden_states[{layer}]</code> if extraction used post-block hidden states.</p>
<pre>{hook_html}</pre>
<h2>Summary Stats</h2>
<pre>{html.escape(json.dumps(stats_row, indent=2, default=str))}</pre>
<h2>Identity Profile</h2>{profile_html}
<table><tr><th>Identity ID</th><th>Label</th><th>Axis</th><th>Mean</th><th>Freq</th></tr>{identity_rows}</table>
</section>
<section class="card">
<h2>Localization Summary</h2>
<table><tr><th>Localization type</th><th>Share of selected prompts</th></tr>{loc_rows}</table>
<h2>Top Non-Special Activating Tokens</h2>
<table><tr><th>Token</th><th>Activation</th><th>Identity</th><th>Axis</th><th>In identity span?</th><th>Prompt</th></tr>{token_rows}</table>
<h2>Top Identity-Span Tokens</h2>
<table><tr><th>Token</th><th>Activation</th><th>Identity</th><th>Axis</th><th>Prompt</th></tr>{span_token_rows}</table>
</section>
<section class="card">
<h2>Contrast Selectivity</h2>
<table><tr><th>Contrast</th><th>d</th><th>AUC</th><th>Mean A</th><th>Mean B</th></tr>{select_rows}</table>
<h2>Decoder Alignment</h2>
<table><tr><th>Contrast</th><th>Cosine</th><th>Signed dot</th></tr>{align_rows}</table>
<h2>Raw Logit Effects</h2>
<p class="note">{html.escape(logit_note)}</p>
<h3>Positive</h3><p>{pos_rows}</p>
<h3>Negative</h3><p>{neg_rows}</p>
</section>
</div></div></body></html>"""
    html_path = card_dir / f"feature_{feature_id:05d}.html"
    json_path = card_dir / f"feature_{feature_id:05d}.json"
    html_path.write_text(card_html)
    payload = {
        "layer": layer,
        "feature_id": feature_id,
        "auto_label": auto_label,
        "top_axis": top_axis,
        "top_identity": top_identity,
        "summary_stats": stats_row,
        "hook_summary": hook_summary,
        "prompt_metrics": prompt_metrics,
        "token_exemplar_figure": exemplar_img,
        "top_identities": identity_means.sort_values("mean_activation", ascending=False).head(20).to_dict("records"),
        "top_tokens": top_tokens.head(args.top_tokens_per_feature).to_dict("records") if not top_tokens.empty else [],
        "positive_logits": positive_logits,
        "negative_logits": negative_logits,
        "logit_note": logit_note,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return {"layer": layer, "feature_id": feature_id, "auto_label": auto_label, "top_axis": top_axis, "top_identity": top_identity, "card": str(html_path.relative_to(args.output_dir)), "token_exemplar_figure": exemplar_img, "max_activation": stats_row.get("max_activation", np.nan), "activation_frequency": stats_row.get("activation_frequency", np.nan)}


def load_hook_summary(output_dir: Path, layer: int) -> dict[str, object]:
    candidates = [
        output_dir.parent / "hook_validation" / "hook_alignment_validation.json",
        output_dir / "hook_alignment_validation.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            for row in data.get("rows", []):
                if int(row.get("requested_layer", -1)) == layer:
                    return row
    return {"validation_status": "not_found", "note": "Run validate_sae_hook_alignment.py for explicit hook validation."}


def write_index(output_dir: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "feature_card_index.csv", index=False)
    table_rows = "\n".join(
        f"<tr><td>{row.layer}</td><td>{row.feature_id}</td><td>{html.escape(str(row.auto_label))}</td><td>{html.escape(str(row.top_axis))}</td><td>{html.escape(str(row.top_identity))}</td><td>{row.activation_frequency:.4f}</td><td>{row.max_activation:.3f}</td><td><a href='{html.escape(str(row.card))}'>card</a></td></tr>"
        for row in df.itertuples(index=False)
    )
    (output_dir / "index.html").write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SAE Identity Feature Cards</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:6px}}th{{background:#f5f5f5;cursor:pointer}}</style>
<script>
function sortTable(n){{var t=document.getElementById('cards'),r=true,d='asc',s=0;while(r){{r=false;var rows=t.rows;for(var i=1;i<rows.length-1;i++){{var x=rows[i].getElementsByTagName('TD')[n],y=rows[i+1].getElementsByTagName('TD')[n],sw=false;var xv=x.innerText,yv=y.innerText;var xn=parseFloat(xv),yn=parseFloat(yv);if(!isNaN(xn)&&!isNaN(yn)){{xv=xn;yv=yn}}if((d=='asc'&&xv>yv)||(d=='desc'&&xv<yv)){{sw=true;break}}}}if(sw){{rows[i].parentNode.insertBefore(rows[i+1],rows[i]);r=true;s++}}else if(s==0&&d=='asc'){{d='desc';r=true}}}}}}
</script></head><body>
<h1>SAE Identity Feature Cards</h1><p>Generated {datetime.now(timezone.utc).isoformat()}</p>
<table id="cards"><tr><th onclick="sortTable(0)">Layer</th><th onclick="sortTable(1)">Feature</th><th onclick="sortTable(2)">Auto label</th><th onclick="sortTable(3)">Top axis</th><th onclick="sortTable(4)">Top identity</th><th onclick="sortTable(5)">Frequency</th><th onclick="sortTable(6)">Max activation</th><th>Link</th></tr>{table_rows}</table>
</body></html>""")


def main() -> None:
    args = parse_args()
    prepare_output(args.output_dir, args.overwrite)
    metadata = pd.read_csv(args.activation_dir / "metadata.csv", keep_default_na=False)
    index_rows = []
    for layer in parse_int_list(args.layers):
        layer_dir = args.sae_encoded_dir / f"layer_{layer:02d}"
        index_path, value_path, _ = find_topk_arrays(layer_dir)
        indices = np.load(index_path, mmap_mode="r")
        values = np.load(value_path, mmap_mode="r")
        token_df = read_token_table(args.token_level_dir / f"layer_{layer:02d}")
        hook_summary = load_hook_summary(args.output_dir, layer)
        for feature_id in select_features(args, layer):
            feature_values = feature_values_for_id(np.asarray(indices), np.asarray(values), feature_id)
            index_rows.append(build_card(args, layer, feature_id, layer_dir, metadata, feature_values, token_df, hook_summary))
    if args.make_index:
        write_index(args.output_dir, index_rows)
    print(f"Feature cards saved to {args.output_dir}")


if __name__ == "__main__":
    main()
