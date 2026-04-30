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
    parser.add_argument("--results_csv", type=Path, default=None, help="Optional merged steering CSV to re-analyze without raw part files.")
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


def parse_json_list(value: object) -> list[object]:
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


AXIS_PREFIX_MAP = {
    "disability": "disability_status",
    "gender": "gender_identity",
    "sex": "gender_identity",
    "appearance": "physical_appearance",
    "race": "race_ethnicity",
    "religion": "religion",
    "sexuality": "sexual_orientation",
    "ses": "socioeconomic_status",
    "nationality": "nationality",
    "age": "age",
}


def axis_from_identity(identity: object) -> str:
    text = str(identity or "")
    prefix = text.split("_", 1)[0]
    return AXIS_PREFIX_MAP.get(prefix, "")


def feature_axis(row: pd.Series) -> str:
    contrast = str(row.get("feature_contrast_name", ""))
    if "_vs_" in contrast:
        left = contrast.split("_vs_", 1)[0]
        axis = axis_from_identity(left)
        if axis:
            return axis
    set_id = str(row.get("feature_set_id", ""))
    for prefix, axis in AXIS_PREFIX_MAP.items():
        if f"_{prefix}_" in f"_{set_id}_":
            return axis
    return ""


def idx_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def row_choice_value(row: pd.Series, idx_col: str, suffix: str) -> float:
    idx = idx_int(row.get(idx_col))
    if idx is None or idx not in {0, 1, 2}:
        return float("nan")
    return float(row.get(f"ans{idx}_logprob_{suffix}", np.nan))


def choice_probabilities(row: pd.Series, suffix: str) -> np.ndarray:
    vals = np.array([row.get(f"ans{i}_logprob_{suffix}", np.nan) for i in range(3)], dtype=float)
    if not np.isfinite(vals).all():
        return np.full(3, np.nan)
    shifted = vals - vals.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def enrich_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feature_role"] = out["feature_roles_json"].map(first_role) if "feature_roles_json" in out.columns else ""
    out["feature_roles_readable"] = out["feature_roles_json"].map(lambda v: ", ".join(sorted(set(map(str, parse_json_list(v)))))) if "feature_roles_json" in out.columns else ""
    out["feature_axis"] = out.apply(feature_axis, axis=1)
    out["steering_direction_label"] = np.where(
        pd.to_numeric(out["alpha"], errors="coerce") > 0,
        "amplify_feature_direction",
        "suppress_feature_direction",
    )
    out["alpha_label"] = out["alpha"].map(lambda x: f"{float(x):+g}")
    out["axis_match_type"] = "unknown_axis_match"
    if {"axis_mapped", "control_type"}.issubset(out.columns):
        same_axis = out["axis_mapped"].astype(str).eq(out["feature_axis"].astype(str)) & out["feature_axis"].astype(str).ne("")
        out.loc[same_axis, "axis_match_type"] = "matched_axis"
        out.loc[~same_axis & out["feature_axis"].astype(str).ne(""), "axis_match_type"] = "wrong_axis_control"
        out.loc[out["control_type"].astype(str).str.contains("wrong_axis", na=False), "axis_match_type"] = "wrong_axis_control"
    out["ambiguous_bias_effect_label"] = "not_ambiguous"
    ambig = out["context_condition"].astype(str).eq("ambig") if "context_condition" in out.columns else pd.Series(False, index=out.index)
    delta = pd.to_numeric(out.get("bias_margin_delta", pd.Series(np.nan, index=out.index)), errors="coerce")
    out.loc[ambig & (delta < -1e-6), "ambiguous_bias_effect_label"] = "reduces_stereotype_preference"
    out.loc[ambig & (delta > 1e-6), "ambiguous_bias_effect_label"] = "increases_stereotype_preference"
    out.loc[ambig & (delta.abs() <= 1e-6), "ambiguous_bias_effect_label"] = "near_zero"

    for role, idx_col in [
        ("stereotyped", "stereotyped_answer_idx"),
        ("unknown", "unknown_answer_idx"),
        ("nonstereotyped", "nonstereotyped_answer_idx"),
        ("correct", "correct_answer_idx"),
    ]:
        out[f"{role}_logprob_base_readable"] = out.apply(lambda row, col=idx_col: row_choice_value(row, col, "base"), axis=1)
        out[f"{role}_logprob_intervened_readable"] = out.apply(lambda row, col=idx_col: row_choice_value(row, col, "intervened"), axis=1)
        out[f"{role}_logprob_delta_readable"] = out[f"{role}_logprob_intervened_readable"] - out[f"{role}_logprob_base_readable"]

    base_probs = np.vstack(out.apply(lambda row: choice_probabilities(row, "base"), axis=1).to_numpy())
    inter_probs = np.vstack(out.apply(lambda row: choice_probabilities(row, "intervened"), axis=1).to_numpy())
    for i in range(3):
        out[f"ans{i}_choice_prob_base"] = base_probs[:, i]
        out[f"ans{i}_choice_prob_intervened"] = inter_probs[:, i]
        out[f"ans{i}_choice_prob_delta"] = inter_probs[:, i] - base_probs[:, i]
    out["stereotype_preference_delta"] = out["bias_margin_delta"]
    return out


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


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "layer", "alpha", "steering_direction_label", "intervention_position", "feature_set_mode",
        "feature_role", "axis_mapped", "feature_axis", "axis_match_type", "mapped_contrast_name",
        "control_type", "context_condition", "question_polarity",
    ]:
        if col in df.columns:
            values = sorted(df[col].dropna().astype(str).unique())
            rows.append({
                "field": col,
                "n_unique": len(values),
                "values": "; ".join(values[:50]) + ("; ..." if len(values) > 50 else ""),
            })
    rows.extend([
        {"field": "n_rows", "n_unique": int(len(df)), "values": ""},
        {"field": "n_examples", "n_unique": int(df["bbq_uid"].nunique()) if "bbq_uid" in df.columns else 0, "values": ""},
        {"field": "n_feature_sets", "n_unique": int(df["feature_set_id"].nunique()) if "feature_set_id" in df.columns else 0, "values": ""},
    ])
    return pd.DataFrame(rows)


def write_interpretation_docs(df: pd.DataFrame, output_dir: Path) -> None:
    coverage = coverage_report(df)
    is_smoke = (
        df["bbq_uid"].nunique() <= 200
        or df["layer"].nunique() <= 1
        or df["feature_set_id"].nunique() <= 20
        or df["alpha"].nunique() <= 2
    )
    readme = f"""# BBQ SAE Steering Analysis Interpretation

## Sign Convention

For ambiguous BBQ items, this analysis defines:

```text
stereotype_preference = log p(stereotyped answer) - log p(unknown answer)
Delta stereotype preference = intervened stereotype_preference - baseline stereotype_preference
```

Interpretation:

- Negative `Delta stereotype preference` means the intervention reduced stereotype preference relative to the unknown answer.
- Positive `Delta stereotype preference` means the intervention increased stereotype preference relative to the unknown answer.
- `alpha > 0` is labeled `amplify_feature_direction`.
- `alpha < 0` is labeled `suppress_feature_direction`.

For disambiguated BBQ items, accuracy/correct-answer changes are tracked separately so bias reduction can be checked against capability degradation.

## Axis Matching

`matched_axis` means the SAE feature set comes from the same identity axis as the BBQ example. `wrong_axis_control` means the feature set comes from a different axis and should be treated as a control, not as the primary causal estimate.

## Run Scope

Rows: {len(df)}

Examples: {df['bbq_uid'].nunique() if 'bbq_uid' in df.columns else 'n/a'}

Feature sets: {df['feature_set_id'].nunique() if 'feature_set_id' in df.columns else 'n/a'}

Layers: {', '.join(map(str, sorted(df['layer'].dropna().unique()))) if 'layer' in df.columns else 'n/a'}

Alphas: {', '.join(map(str, sorted(df['alpha'].dropna().unique()))) if 'alpha' in df.columns else 'n/a'}

## Smoke-Test Warning

{'This appears to be a smoke/limited run. Treat figures as pipeline sanity checks, not substantive research claims.' if is_smoke else 'This does not appear to be a minimal smoke run, but coverage should still be checked before interpretation.'}
"""
    (output_dir / "README_interpretation.md").write_text(readme)
    if is_smoke:
        (output_dir / "SMOKE_TEST_LIMITATIONS.md").write_text(
            """# Smoke-Test Limitations

This run is limited and should not be interpreted as final evidence.

Use it to check:

- The pipeline runs end-to-end.
- Interventions change answer log probabilities by nonzero amounts.
- Alpha signs produce separable effects.
- Matched-axis and wrong-axis control labels are available.

Do not use it to make broad claims about identity axes, layers, or feature roles unless those dimensions are actually covered in `coverage_report.csv`.
"""
        )
    coverage.to_csv(output_dir / "coverage_report.csv", index=False)


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


BIAS_LABEL = "Delta stereotype preference: Delta[log p(stereotyped) - log p(unknown)]"


def write_caption(output_dir: Path, name: str, text: str) -> None:
    (output_dir / "figures" / f"{name}.txt").write_text(text.strip() + "\n")


def nice_name(value: object) -> str:
    return str(value).replace("_", " ")


def make_figures(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    for col in ["bias_margin_delta", "stereotyped_delta", "unknown_delta", "nonstereotyped_delta", "correct_delta", "alpha"]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    ambig = plot_df[plot_df["context_condition"].eq("ambig")].copy()

    if not ambig.empty:
        positions = list(ambig["intervention_position"].dropna().astype(str).unique())
        if len(positions) > 4:
            positions = positions[:4]
        fig, axes = plt.subplots(1, max(1, len(positions)), figsize=(6 * max(1, len(positions)), 5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, position in zip(axes, positions or ["all"]):
            data = ambig[ambig["intervention_position"].eq(position)] if position != "all" else ambig
            if sns:
                sns.barplot(data=data, x="alpha_label", y="bias_margin_delta", hue="feature_role", ax=ax, errorbar=("ci", 95))
            else:
                data.groupby(["alpha_label", "feature_role"])["bias_margin_delta"].mean().unstack().plot(kind="bar", ax=ax)
            ax.axhline(0, color="black", lw=1)
            ax.set_title(f"Position: {nice_name(position)}")
            ax.set_xlabel("Alpha / steering direction")
            ax.set_ylabel(BIAS_LABEL)
        name = "ambiguous_bias_margin_by_alpha_and_direction"
        save_fig(fig, output_dir, name)
        write_caption(
            output_dir,
            name,
            "Ambiguous BBQ items only. Negative values mean the intervention reduced stereotype preference "
            "relative to the unknown answer. Positive values mean the intervention increased stereotype preference. "
            "Alpha > 0 amplifies the signed SAE feature direction; alpha < 0 suppresses/reverses it.",
        )

        melt = ambig.melt(
            id_vars=["feature_role", "alpha_label", "steering_direction_label", "intervention_position"],
            value_vars=["stereotyped_delta", "nonstereotyped_delta", "unknown_delta"],
            var_name="answer_role",
            value_name="delta_logprob",
        )
        melt["answer_role"] = melt["answer_role"].str.replace("_delta", "", regex=False).str.replace("_", " ")
        fig, ax = plt.subplots(figsize=(12, 6))
        if sns:
            sns.barplot(data=melt, x="alpha_label", y="delta_logprob", hue="answer_role", ax=ax, errorbar=("ci", 95))
        else:
            melt.groupby(["alpha_label", "answer_role"])["delta_logprob"].mean().unstack().plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", lw=1)
        ax.set_title("Ambiguous items: answer-choice logprob shifts by alpha")
        ax.set_xlabel("Alpha / steering direction")
        ax.set_ylabel("Mean Delta answer log probability")
        ax.tick_params(axis="x", rotation=35)
        name = "answer_logprob_shifts_ambiguous"
        save_fig(fig, output_dir, name)
        write_caption(
            output_dir,
            name,
            "Shows which answer roles move under steering on ambiguous BBQ items. A bias-reducing intervention "
            "should generally lower the stereotyped answer and/or raise the unknown answer.",
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    matched = plot_df[plot_df["context_condition"].eq("ambig")].copy()
    if sns:
        sns.barplot(data=matched, x="axis_match_type", y="bias_margin_delta", hue="alpha_label", ax=ax, errorbar=("ci", 95))
    else:
        matched.groupby(["axis_match_type", "alpha_label"])["bias_margin_delta"].mean().unstack().plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Matched-axis steering vs wrong-axis controls")
    ax.set_xlabel("Feature-set relationship to BBQ example axis")
    ax.set_ylabel(BIAS_LABEL)
    ax.tick_params(axis="x", rotation=20)
    name = "matched_vs_wrong_axis_effects"
    save_fig(fig, output_dir, name)
    write_caption(
        output_dir,
        name,
        "Ambiguous BBQ items. Matched-axis feature sets come from the same identity axis as the BBQ example; "
        "wrong-axis controls use identity features from another axis. Negative values are bias-reducing.",
    )

    axis_summary = ambig.pivot_table(
        index="axis_mapped",
        columns="steering_direction_label",
        values="bias_margin_delta",
        aggfunc="mean",
    ) if not ambig.empty else pd.DataFrame()
    if not axis_summary.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(axis_summary))))
        if sns:
            sns.heatmap(axis_summary, center=0, cmap="vlag", annot=True, fmt=".4f", ax=ax)
        else:
            im = ax.imshow(axis_summary.fillna(0).to_numpy(), cmap="coolwarm")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(axis_summary.columns)), axis_summary.columns, rotation=25, ha="right")
            ax.set_yticks(range(len(axis_summary.index)), axis_summary.index)
        ax.set_title("Axis-level ambiguous stereotype-preference effect")
        ax.set_xlabel("Steering direction")
        ax.set_ylabel("BBQ identity axis")
        name = "axis_level_bias_effects"
        save_fig(fig, output_dir, name)
        write_caption(output_dir, name, f"Values are mean {BIAS_LABEL}. Negative is bias-reducing; positive increases stereotype preference.")

    trade = plot_df.groupby("feature_set_id").agg(
        ambiguous_bias_delta=("bias_margin_delta", lambda s: s[plot_df.loc[s.index, "context_condition"].eq("ambig")].mean()),
        disambig_accuracy_delta=("correct_intervened", lambda s: (s[plot_df.loc[s.index, "context_condition"].eq("disambig")].astype(float) - plot_df.loc[s.index][plot_df.loc[s.index, "context_condition"].eq("disambig")]["correct_base"].astype(float)).mean()),
        feature_role=("feature_role", "first"),
        axis_mapped=("axis_mapped", "first"),
        steering_direction_label=("steering_direction_label", "first"),
    ).reset_index()
    if trade[["ambiguous_bias_delta", "disambig_accuracy_delta"]].notna().any().all():
        fig, ax = plt.subplots(figsize=(8, 6))
        if sns:
            sns.scatterplot(data=trade, x="ambiguous_bias_delta", y="disambig_accuracy_delta", hue="axis_mapped", style="steering_direction_label", ax=ax)
        else:
            ax.scatter(trade["ambiguous_bias_delta"], trade["disambig_accuracy_delta"])
        ax.axvline(0, color="black", lw=1)
        ax.axhline(0, color="black", lw=1)
        ax.set_title("Ambiguous bias reduction vs disambiguated accuracy tradeoff")
        ax.set_xlabel(BIAS_LABEL + "\n(left/negative = less stereotype preference)")
        ax.set_ylabel("Delta disambiguated accuracy")
        name = "disambig_accuracy_tradeoff_by_axis"
        save_fig(fig, output_dir, name)
        write_caption(output_dir, name, "Each point is a feature set. Left is better for ambiguous bias; up is better for disambiguated accuracy.")

    heat = ambig.pivot_table(index="mapped_contrast_name", columns="alpha_label", values="bias_margin_delta", aggfunc="mean") if not ambig.empty else pd.DataFrame()
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(heat))))
        if sns:
            sns.heatmap(heat, center=0, cmap="vlag", annot=True, fmt=".4f", ax=ax)
        else:
            im = ax.imshow(heat.fillna(0).to_numpy(), cmap="coolwarm")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=25, ha="right")
            ax.set_yticks(range(len(heat.index)), heat.index)
        title = "Contrast-level ambiguous stereotype-preference effect"
        if heat.shape[0] == 1:
            title += " (smoke-limited: one contrast)"
        ax.set_title(title)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Mapped BBQ contrast")
        name = "contrast_level_bias_effects"
        save_fig(fig, output_dir, name)
        write_caption(output_dir, name, f"Values are mean {BIAS_LABEL}. Negative is bias-reducing. If only one contrast appears, this is a smoke-test coverage artifact.")

    if plot_df["layer"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        if sns:
            sns.lineplot(data=plot_df, x="layer", y="bias_margin_delta", hue="feature_role", errorbar="se", ax=ax)
        else:
            plot_df.groupby(["layer", "feature_role"])["bias_margin_delta"].mean().unstack().plot(ax=ax)
        ax.axhline(0, color="black", lw=1)
        ax.set_title("Layer sweep: stereotype-preference effect")
        ax.set_ylabel(BIAS_LABEL)
        name = "layer_sweep_causal_effect"
        save_fig(fig, output_dir, name)
        write_caption(output_dir, name, "Generated only when multiple layers are present. Negative values reduce stereotype preference.")

    example_cols = [
        "bbq_uid", "axis_mapped", "mapped_contrast_name", "feature_set_id", "alpha_label",
        "intervention_position", "stereotyped_logprob_base", "stereotyped_logprob_intervened",
        "unknown_logprob_base_readable", "unknown_logprob_intervened_readable", "bias_margin_delta",
    ]
    example_table = ambig.reindex(columns=[c for c in example_cols if c in ambig.columns]).sort_values("bias_margin_delta").head(20)
    if not example_table.empty:
        fig, ax = plt.subplots(figsize=(14, max(3, 0.35 * len(example_table))))
        ax.axis("off")
        display = example_table.copy()
        for col in display.select_dtypes(include=[float]).columns:
            display[col] = display[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="left")
        ax.set_title("Top smoke examples: strongest bias-reducing ambiguous interventions")
        name = "baseline_to_intervened_answer_probs_examples"
        save_fig(fig, output_dir, name)
        write_caption(output_dir, name, "Rows with most negative stereotype-preference deltas. This table is intended for smoke-test inspection, not final claims.")


def main() -> None:
    args = parse_args()
    logger = setup(args.output_dir, args.overwrite)
    config = vars(args).copy()
    config.update({k: str(v) for k, v in config.items() if isinstance(v, Path)})
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "causal_effects_config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = read_table(args.results_csv) if args.results_csv else load_results(args.steering_dir)
    if results.empty:
        source = args.results_csv if args.results_csv else args.steering_dir / "results_parts"
        raise FileNotFoundError(f"No steering results found at {source}")
    logger.info("Loaded %d steering rows", len(results))
    results = enrich_results(results)
    write_parquet_csv(results, args.output_dir, "steering_results_merged")
    write_interpretation_docs(results, args.output_dir)
    summaries = {
        "steering_summary_by_feature": ["feature_set_id", "feature_role", "layer", "intervention_position", "alpha"],
        "steering_summary_by_contrast": ["mapped_contrast_name", "feature_role", "intervention_position", "alpha"],
        "steering_summary_by_role": ["feature_role", "intervention_position", "alpha", "context_condition", "question_polarity"],
        "steering_summary_by_layer": ["layer", "feature_role", "intervention_position", "alpha"],
    }
    for stem, cols in summaries.items():
        available = [col for col in cols if col in results.columns]
        summarize(results, available, args.bootstrap_samples).to_csv(args.output_dir / f"{stem}.csv", index=False)
    interpretation_tables = {
        "interpretation_summary_by_axis": ["axis_mapped", "axis_match_type", "steering_direction_label", "intervention_position", "feature_role"],
        "interpretation_summary_by_contrast": ["mapped_contrast_name", "axis_match_type", "steering_direction_label", "intervention_position", "feature_role"],
        "interpretation_summary_by_feature_set_alpha": ["feature_set_id", "feature_role", "feature_axis", "axis_match_type", "alpha", "steering_direction_label", "intervention_position"],
    }
    for stem, cols in interpretation_tables.items():
        available = [col for col in cols if col in results.columns]
        summarize(results, available, args.bootstrap_samples).to_csv(args.output_dir / f"{stem}.csv", index=False)
    feature_summary = summarize(results, ["feature_set_id", "feature_role"], args.bootstrap_samples)
    detailed_summary = summarize(results, ["feature_set_id", "feature_role", "alpha", "steering_direction_label", "intervention_position", "axis_match_type"], args.bootstrap_samples)
    best = detailed_summary.sort_values("mean_bias_margin_delta", ascending=True).head(100)
    harmful = detailed_summary.sort_values("mean_bias_margin_delta", ascending=False).head(100)
    best.to_csv(args.output_dir / "best_interventions.csv", index=False)
    harmful.to_csv(args.output_dir / "harmful_interventions.csv", index=False)
    best.to_csv(args.output_dir / "top_bias_reducing_interventions.csv", index=False)
    harmful.to_csv(args.output_dir / "top_bias_increasing_interventions.csv", index=False)
    make_figures(results, args.output_dir)
    logger.info("Analysis written to %s", args.output_dir)


if __name__ == "__main__":
    main()
