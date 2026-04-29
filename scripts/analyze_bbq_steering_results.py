#!/usr/bin/env python3
"""Analyze BBQ SAE steering result parts."""

from __future__ import annotations

import argparse
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


DEFAULT_STEERING = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/steering")
DEFAULT_PREPARED = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet")
DEFAULT_OUTPUT = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BBQ SAE steering results.")
    parser.add_argument("--steering_dir", type=Path, default=DEFAULT_STEERING)
    parser.add_argument("--prepared_data", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap_samples", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup(output_dir: Path, overwrite: bool) -> logging.Logger:
    if output_dir.exists() and any(output_dir.iterdir()) and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    logger = logging.getLogger("bbq_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(output_dir / "logs" / "analysis.log")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def load_results(steering_dir: Path) -> pd.DataFrame:
    files = sorted((steering_dir / "results_parts").glob("part_*.parquet")) + sorted((steering_dir / "results_parts").glob("part_*.csv"))
    frames = [read_table(path) for path in tqdm(files, desc="load result parts") if path.stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_parquet_csv(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    try:
        df.to_parquet(output_dir / f"{stem}.parquet", index=False)
    except Exception:
        pass


def first_role(value: object) -> str:
    try:
        roles = json.loads(str(value))
        return str(roles[0]) if roles else ""
    except Exception:
        return ""


def bootstrap_ci(group: pd.DataFrame, col: str, n: int) -> tuple[float, float]:
    vals = group[["bbq_uid", col]].dropna()
    if vals.empty:
        return float("nan"), float("nan")
    by_example = vals.groupby("bbq_uid")[col].mean().to_numpy()
    if len(by_example) <= 1:
        mean = float(np.mean(by_example))
        return mean, mean
    rng = np.random.default_rng(0)
    samples = [float(rng.choice(by_example, size=len(by_example), replace=True).mean()) for _ in range(n)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize(df: pd.DataFrame, group_cols: list[str], bootstrap_samples: int) -> pd.DataFrame:
    rows = []
    for keys, group in tqdm(df.groupby(group_cols, dropna=False), desc=f"summarize {group_cols}", leave=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in ["stereotyped_delta", "unknown_delta", "nonstereotyped_delta", "bias_margin_delta", "correct_delta"]:
            if col in group.columns:
                row[f"mean_{col}"] = float(group[col].mean())
        lo, hi = bootstrap_ci(group, "bias_margin_delta", bootstrap_samples)
        row["bias_margin_delta_ci_low"] = lo
        row["bias_margin_delta_ci_high"] = hi
        row["n_examples"] = int(group["bbq_uid"].nunique())
        row["n_rows"] = int(len(group))
        row["n_featuresets"] = int(group["feature_set_id"].nunique()) if "feature_set_id" in group.columns else 0
        row["prediction_change_rate"] = float(group["prediction_changed"].astype(bool).mean()) if "prediction_changed" in group.columns else float("nan")
        disambig = group[group["context_condition"].eq("disambig")] if "context_condition" in group.columns else pd.DataFrame()
        row["accuracy_change_disambig"] = float((disambig["correct_intervened"].astype(float) - disambig["correct_base"].astype(float)).mean()) if not disambig.empty and "correct_intervened" in disambig.columns else float("nan")
        row["interpretation_label"] = interpretation_label(row)
        rows.append(row)
    return pd.DataFrame(rows)


def interpretation_label(row: dict[str, object]) -> str:
    bias = float(row.get("mean_bias_margin_delta", 0) or 0)
    unknown = float(row.get("mean_unknown_delta", 0) or 0)
    nontarget = float(row.get("mean_nonstereotyped_delta", 0) or 0)
    correct = float(row.get("mean_correct_delta", 0) or 0)
    acc = float(row.get("accuracy_change_disambig", 0) or 0)
    if correct < -0.02 or acc < -0.02:
        return "capability_degrading"
    if bias < 0 and unknown > 0:
        return "bias_reducing_uncertainty_increasing"
    if bias < 0 and nontarget > unknown:
        return "substitution_effect"
    if abs(bias) < 1e-4:
        return "no_effect"
    return "identity_salience_only"


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / f"{name}.png", dpi=220)
    fig.savefig(output_dir / "figures" / f"{name}.pdf")
    plt.close(fig)


def make_figures(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    plot_df["feature_role"] = plot_df["feature_roles_json"].map(first_role) if "feature_roles_json" in plot_df.columns else ""
    ambig = plot_df[plot_df["context_condition"].eq("ambig")].copy()
    if not ambig.empty:
        melt = ambig.melt(id_vars=["feature_role", "control_type"], value_vars=["stereotyped_delta", "nonstereotyped_delta", "unknown_delta"], var_name="answer_role", value_name="delta")
        fig, ax = plt.subplots(figsize=(12, 6))
        if sns:
            sns.barplot(data=melt, x="feature_role", y="delta", hue="answer_role", ax=ax)
        else:
            melt.groupby(["feature_role", "answer_role"])["delta"].mean().unstack().plot(kind="bar", ax=ax)
        ax.set_title("Ambiguous BBQ logprob shifts by feature role")
        ax.tick_params(axis="x", rotation=35)
        save_fig(fig, output_dir, "ambiguous_logit_shift_by_intervention_type")

    fig, ax = plt.subplots(figsize=(10, 6))
    if sns:
        sns.lineplot(data=plot_df, x="alpha", y="bias_margin_delta", hue="feature_role", errorbar="se", ax=ax)
    else:
        plot_df.groupby(["alpha", "feature_role"])["bias_margin_delta"].mean().unstack().plot(ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Dose response: delta bias margin")
    save_fig(fig, output_dir, "dose_response_bias_margin")

    fig, ax = plt.subplots(figsize=(10, 6))
    compare = plot_df[plot_df["control_type"].isin(["kept_feature", "sign_flip", "random_direction_norm_matched", "wrong_axis_features"])]
    if sns:
        sns.barplot(data=compare, x="control_type", y="bias_margin_delta", hue="feature_role", ax=ax)
    else:
        compare.groupby("control_type")["bias_margin_delta"].mean().plot(kind="bar", ax=ax)
    ax.set_title("Identity vs stereotype residual comparison")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, output_dir, "identity_vs_stereotype_residual_comparison")

    fig, ax = plt.subplots(figsize=(10, 6))
    if sns:
        sns.lineplot(data=plot_df, x="layer", y="bias_margin_delta", hue="feature_role", errorbar="se", ax=ax)
    else:
        plot_df.groupby(["layer", "feature_role"])["bias_margin_delta"].mean().unstack().plot(ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Layer sweep causal effect")
    save_fig(fig, output_dir, "layer_sweep_causal_effect")

    fig, ax = plt.subplots(figsize=(10, 6))
    if sns:
        sns.barplot(data=plot_df, x="question_polarity", y="bias_margin_delta", hue="feature_role", ax=ax)
    else:
        plot_df.groupby(["question_polarity", "feature_role"])["bias_margin_delta"].mean().unstack().plot(kind="bar", ax=ax)
    ax.set_title("Polarity split effects")
    save_fig(fig, output_dir, "polarity_split_effects")

    trade = plot_df.groupby("feature_set_id").agg(
        ambiguous_bias_delta=("bias_margin_delta", lambda s: s[plot_df.loc[s.index, "context_condition"].eq("ambig")].mean()),
        disambig_accuracy_delta=("correct_intervened", lambda s: (s[plot_df.loc[s.index, "context_condition"].eq("disambig")].astype(float) - plot_df.loc[s.index][plot_df.loc[s.index, "context_condition"].eq("disambig")]["correct_base"].astype(float)).mean()),
        feature_role=("feature_role", "first"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    if sns:
        sns.scatterplot(data=trade, x="ambiguous_bias_delta", y="disambig_accuracy_delta", hue="feature_role", ax=ax)
    else:
        ax.scatter(trade["ambiguous_bias_delta"], trade["disambig_accuracy_delta"])
    ax.axvline(0, color="black", lw=1)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Ambiguous bias vs disambiguated accuracy tradeoff")
    save_fig(fig, output_dir, "ambiguous_vs_disambig_tradeoff")

    heat = plot_df.pivot_table(index="mapped_contrast_name", columns="feature_role", values="bias_margin_delta", aggfunc="mean")
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(heat))))
        if sns:
            sns.heatmap(heat, center=0, cmap="vlag", ax=ax)
        else:
            im = ax.imshow(heat.fillna(0).to_numpy(), cmap="coolwarm")
            fig.colorbar(im, ax=ax)
        ax.set_title("Per-contrast best effects heatmap")
        save_fig(fig, output_dir, "per_contrast_best_effects_heatmap")

    per_feature = plot_df[plot_df["feature_set_mode"].eq("per_feature")].groupby(["feature_set_id", "feature_role"]).agg(mean_bias_margin_delta=("bias_margin_delta", "mean"), n=("bbq_uid", "nunique"), sd=("bias_margin_delta", "std")).reset_index()
    if not per_feature.empty:
        per_feature["se"] = per_feature["sd"] / np.sqrt(per_feature["n"].clip(lower=1))
        per_feature["volcano_y"] = -np.log10(np.maximum(1e-12, 2 * (1 - np.minimum(0.999999, np.abs(per_feature["mean_bias_margin_delta"] / per_feature["se"].replace(0, np.nan)).fillna(0) / 10))))
        fig, ax = plt.subplots(figsize=(9, 6))
        if sns:
            sns.scatterplot(data=per_feature, x="mean_bias_margin_delta", y="volcano_y", hue="feature_role", ax=ax)
        else:
            ax.scatter(per_feature["mean_bias_margin_delta"], per_feature["volcano_y"])
        ax.axvline(0, color="black", lw=1)
        ax.set_title("Feature effect volcano")
        save_fig(fig, output_dir, "feature_effect_volcano")


def main() -> None:
    args = parse_args()
    logger = setup(args.output_dir, args.overwrite)
    config = vars(args).copy()
    config.update({k: str(v) for k, v in config.items() if isinstance(v, Path)})
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "causal_effects_config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = load_results(args.steering_dir)
    if results.empty:
        raise FileNotFoundError(f"No result parts found under {args.steering_dir / 'results_parts'}")
    logger.info("Loaded %d steering rows", len(results))
    write_parquet_csv(results, args.output_dir, "steering_results_merged")
    results["feature_role"] = results["feature_roles_json"].map(first_role) if "feature_roles_json" in results.columns else ""
    summaries = {
        "steering_summary_by_feature": ["feature_set_id", "feature_role", "layer", "intervention_position", "alpha"],
        "steering_summary_by_contrast": ["mapped_contrast_name", "feature_role", "intervention_position", "alpha"],
        "steering_summary_by_role": ["feature_role", "intervention_position", "alpha", "context_condition", "question_polarity"],
        "steering_summary_by_layer": ["layer", "feature_role", "intervention_position", "alpha"],
    }
    for stem, cols in summaries.items():
        available = [col for col in cols if col in results.columns]
        summarize(results, available, args.bootstrap_samples).to_csv(args.output_dir / f"{stem}.csv", index=False)
    feature_summary = summarize(results, ["feature_set_id", "feature_role"], args.bootstrap_samples)
    best = feature_summary.sort_values("mean_bias_margin_delta", ascending=True).head(100)
    harmful = feature_summary.sort_values("mean_bias_margin_delta", ascending=False).head(100)
    best.to_csv(args.output_dir / "best_interventions.csv", index=False)
    harmful.to_csv(args.output_dir / "harmful_interventions.csv", index=False)
    make_figures(results, args.output_dir)
    logger.info("Analysis written to %s", args.output_dir)


if __name__ == "__main__":
    main()
