#!/usr/bin/env python3
"""Directional visualizations for identity contrast representations.

Unlike PCA, these plots use theory-driven contrast directions, e.g.
Black - White or disabled - nondisabled, then visualize prompt-level
projections along those axes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pandas.errors import EmptyDataError
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
except ImportError:  # pragma: no cover
    leaves_list = None
    linkage = None
    squareform = None


DEFAULT_ACTIVATION_DIR = Path(
    "/workspace/status_mi/results/activations/llama-3.1-8b/"
    "identity_prompts_final_token"
)
DEFAULT_OUTPUT_DIR = Path(
    "/workspace/status_mi/results/geometry/llama-3.1-8b/"
    "identity_prompts_final_token/directional_visualizations"
)
METADATA_COLUMNS = [
    "prompt_id",
    "prompt",
    "identity_id",
    "axis",
    "canonical_label",
    "template_id",
    "family",
    "required_form",
    "form_used",
]
DEFAULT_RESIDUALIZATIONS = [
    "raw",
    "family_residualized",
    "template_residualized",
    "required_form_residualized",
]
RESIDUALIZATION_GROUPS = {
    "raw": None,
    "family_residualized": "family",
    "template_residualized": "template_id",
    "required_form_residualized": "required_form",
}
DEFAULT_CONTRASTS = [
    ("race_black_vs_race_white", "race_black", "race_white", "race_ethnicity"),
    ("race_black_vs_race_asian", "race_black", "race_asian", "race_ethnicity"),
    ("race_black_vs_race_caucasian", "race_black", "race_caucasian", "race_ethnicity"),
    ("sexuality_gay_vs_sexuality_straight", "sexuality_gay", "sexuality_straight", "sexual_orientation"),
    ("sexuality_gay_vs_sexuality_heterosexual", "sexuality_gay", "sexuality_heterosexual", "sexual_orientation"),
    ("sexuality_lesbian_vs_sexuality_straight", "sexuality_lesbian", "sexuality_straight", "sexual_orientation"),
    ("sexuality_bisexual_vs_sexuality_straight", "sexuality_bisexual", "sexuality_straight", "sexual_orientation"),
    ("disability_disabled_vs_disability_nondisabled", "disability_disabled", "disability_nondisabled", "disability_status"),
    ("disability_disabled_vs_disability_able_bodied", "disability_disabled", "disability_able_bodied", "disability_status"),
    ("appearance_short_vs_appearance_tall", "appearance_short", "appearance_tall", "physical_appearance"),
    ("appearance_obese_vs_appearance_thin", "appearance_obese", "appearance_thin", "physical_appearance"),
    ("appearance_poorly_dressed_vs_appearance_well_dressed", "appearance_poorly_dressed", "appearance_well_dressed", "physical_appearance"),
    ("ses_low_income_vs_ses_rich", "ses_low_income", "ses_rich", "socioeconomic_status"),
    ("ses_low_income_vs_ses_high_socioeconomic_status", "ses_low_income", "ses_high_socioeconomic_status", "socioeconomic_status"),
    ("ses_lower_class_vs_ses_upper_class", "ses_lower_class", "ses_upper_class", "socioeconomic_status"),
    ("ses_blue_collar_vs_ses_white_collar", "ses_blue_collar", "ses_white_collar", "socioeconomic_status"),
    ("gender_transgender_vs_gender_cisgender", "gender_transgender", "gender_cisgender", "gender_identity"),
    ("gender_transgender_man_vs_gender_cisgender_man", "gender_transgender_man", "gender_cisgender_man", "gender_identity"),
    ("gender_transgender_woman_vs_gender_cisgender_woman", "gender_transgender_woman", "gender_cisgender_woman", "gender_identity"),
    ("religion_muslim_vs_religion_christian", "religion_muslim", "religion_christian", "religion"),
    ("religion_jewish_vs_religion_christian", "religion_jewish", "religion_christian", "religion"),
]
PLANE_SPECS = {
    "race_ethnicity": [
        ("Black - White", "race_black", "race_white"),
        ("Asian - White", "race_asian", "race_white"),
    ],
    "gender_identity": [
        ("transgender - cisgender", "gender_transgender", "gender_cisgender"),
        ("woman - man", "gender_woman", "gender_man"),
    ],
    "sexual_orientation": [
        ("gay - straight", "sexuality_gay", "sexuality_straight"),
        ("bisexual - straight", "sexuality_bisexual", "sexuality_straight"),
    ],
    "socioeconomic_status": [
        ("lower-class - upper-class", "ses_lower_class", "ses_upper_class"),
        ("blue-collar - white-collar", "ses_blue_collar", "ses_white_collar"),
    ],
    "physical_appearance": [
        ("obese - thin", "appearance_obese", "appearance_thin"),
        ("short - tall", "appearance_short", "appearance_tall"),
    ],
}
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
LINESTYLES = ["-", "--", "-.", ":"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot directional identity contrast visualizations.")
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="8,16,24,32")
    parser.add_argument("--residualizations", default=",".join(DEFAULT_RESIDUALIZATIONS))
    parser.add_argument("--contrasts", type=Path, default=None)
    parser.add_argument("--max_points_per_plot", type=int, default=20000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def prepare_dirs(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    dirs = {
        "projections": output_dir / "projections",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
    }
    figure_subdirs = [
        "histograms",
        "kde",
        "stripplots",
        "axis_context",
        "layer_curves",
        "family_holdout",
        "direction_cosines",
        "centroid_distance_heatmaps",
        "directional_planes",
    ]
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for subdir in figure_subdirs:
        (dirs["figures"] / subdir).mkdir(parents=True, exist_ok=True)
    return dirs


def load_metadata(activation_dir: Path) -> pd.DataFrame:
    path = activation_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")
    metadata = pd.read_csv(path, keep_default_na=False)
    missing = [col for col in METADATA_COLUMNS if col not in metadata.columns]
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")
    return metadata.reset_index(drop=True)


def load_layer(activation_dir: Path, layer: int, n_rows: int) -> np.ndarray:
    path = activation_dir / f"layer_{layer:02d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing layer file: {path}")
    x = np.load(path, mmap_mode="r")
    if x.ndim != 2 or x.shape[0] != n_rows:
        raise ValueError(f"{path.name} shape {x.shape} does not align with metadata rows {n_rows}")
    return np.asarray(x, dtype=np.float32)


def load_contrasts(path: Path | None, metadata: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        contrasts = pd.DataFrame(DEFAULT_CONTRASTS, columns=["contrast_name", "identity_a", "identity_b", "axis"])
    else:
        contrasts = pd.read_csv(path, keep_default_na=False)
    required = {"contrast_name", "identity_a", "identity_b", "axis"}
    missing = required - set(contrasts.columns)
    if missing:
        raise ValueError(f"Contrast CSV missing columns: {sorted(missing)}")
    identity_set = set(metadata["identity_id"])
    valid = contrasts[
        contrasts["identity_a"].isin(identity_set) & contrasts["identity_b"].isin(identity_set)
    ].copy()
    skipped = len(contrasts) - len(valid)
    if skipped:
        print(f"Skipping {skipped} contrasts because one or both identities are absent.")
    return valid.reset_index(drop=True)


def residualize(x: np.ndarray, metadata: pd.DataFrame, residualization: str) -> np.ndarray:
    group_col = RESIDUALIZATION_GROUPS[residualization]
    if group_col is None:
        return x
    global_mean = x.mean(axis=0, keepdims=True)
    x_resid = x.copy()
    for _, idx in metadata.groupby(group_col, sort=True).groups.items():
        idx_array = np.fromiter(idx, dtype=int)
        group_mean = x[idx_array].mean(axis=0, keepdims=True)
        x_resid[idx_array] = x[idx_array] - group_mean + global_mean
    return x_resid


def category_colors(values: pd.Series) -> dict[str, str]:
    labels = sorted(values.astype(str).unique())
    return {label: OKABE_ITO[i % len(OKABE_ITO)] for i, label in enumerate(labels)}


def category_markers(values: pd.Series) -> dict[str, str]:
    labels = sorted(values.astype(str).unique())
    return {label: MARKERS[i % len(MARKERS)] for i, label in enumerate(labels)}


def category_linestyles(values: pd.Series) -> dict[str, str]:
    labels = sorted(values.astype(str).unique())
    return {label: LINESTYLES[i % len(LINESTYLES)] for i, label in enumerate(labels)}


def add_legend(ax: plt.Axes, labels: list[str], colors: dict[str, str], markers: dict[str, str] | None = None, linestyles: dict[str, str] | None = None, max_items: int = 60) -> None:
    if len(labels) > max_items:
        ax.text(1.02, 1.0, f"{len(labels)} categories\nlegend omitted", transform=ax.transAxes, va="top", fontsize=8)
        return
    handles = [
        Line2D([0], [0], color=colors[label], marker=markers[label] if markers else None, linestyle=linestyles[label] if linestyles else "", linewidth=2 if linestyles else 0, markersize=6, label=label)
        for label in labels
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8, ncol=1 if len(labels) <= 20 else 2)


def save_fig(fig: plt.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=220)
    fig.savefig(path_no_suffix.with_suffix(".pdf"))
    plt.close(fig)


def sample_plot_rows(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    sampled = (
        df.groupby("identity_id", group_keys=False, sort=False)
        .sample(frac=1, random_state=seed)
        .groupby("identity_id", group_keys=False, sort=False)
        .head(max(1, int(np.ceil(max_points / df["identity_id"].nunique()))))
    )
    if len(sampled) > max_points:
        sampled = sampled.sample(n=max_points, random_state=seed)
    return sampled


def normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        return None
    return vec / norm


def compute_direction(x_centered: np.ndarray, metadata: pd.DataFrame, identity_a: str, identity_b: str) -> tuple[np.ndarray | None, bool]:
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None, False
    direction = x_centered[mask_a].mean(axis=0) - x_centered[mask_b].mean(axis=0)
    direction = normalize(direction)
    if direction is None:
        return None, False
    scores = x_centered @ direction
    flipped = False
    if scores[mask_a].mean() < scores[mask_b].mean():
        direction = -direction
        flipped = True
    return direction, flipped


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if len(scores_a) < 2 or len(scores_b) < 2:
        return float("nan")
    pooled = (((len(scores_a) - 1) * np.var(scores_a, ddof=1)) + ((len(scores_b) - 1) * np.var(scores_b, ddof=1))) / (len(scores_a) + len(scores_b) - 2)
    if pooled <= 0:
        return float("nan")
    return float((scores_a.mean() - scores_b.mean()) / np.sqrt(pooled))


def binary_metrics(scores: np.ndarray, metadata: pd.DataFrame, identity_a: str, identity_b: str) -> dict[str, float | int]:
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
    scores_a = scores[mask_a]
    scores_b = scores[mask_b]
    pair_scores = np.concatenate([scores_a, scores_b])
    y = np.concatenate([np.ones(len(scores_a)), np.zeros(len(scores_b))])
    auc = float(roc_auc_score(y, pair_scores)) if len(np.unique(y)) == 2 else float("nan")
    threshold = (scores_a.mean() + scores_b.mean()) / 2
    pred = pair_scores >= threshold
    accuracy = float((pred == y.astype(bool)).mean())
    return {
        "mean_a": float(scores_a.mean()),
        "mean_b": float(scores_b.mean()),
        "sd_a": float(scores_a.std(ddof=1)) if len(scores_a) > 1 else 0.0,
        "sd_b": float(scores_b.std(ddof=1)) if len(scores_b) > 1 else 0.0,
        "cohens_d": cohens_d(scores_a, scores_b),
        "auc": auc,
        "accuracy_midpoint": accuracy,
        "n_a": int(len(scores_a)),
        "n_b": int(len(scores_b)),
    }


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_projection_distributions(plot_df: pd.DataFrame, metrics: dict[str, object], output_dir: Path, residualization: str, contrast_name: str, layer: int, seed: int, max_points: int) -> None:
    endpoint_df = plot_df[plot_df["endpoint_label"].isin(["identity_a", "identity_b"])].copy()
    if endpoint_df.empty:
        return
    endpoint_df = sample_plot_rows(endpoint_df, max_points, seed)
    title = f"{contrast_name} layer {layer:02d} ({residualization}) AUC={metrics['auc']:.3f}, d={metrics['cohens_d']:.2f}"

    fig, ax = plt.subplots(figsize=(10, 6))
    for identity, group in endpoint_df.groupby("identity_id", sort=True):
        ax.hist(group["projection_score"], bins=35, alpha=0.45, label=identity, density=True)
        ax.axvline(group["projection_score"].mean(), linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Projection score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save_fig(fig, output_dir / "figures" / "histograms" / residualization / f"{contrast_name}_layer_{layer:02d}")

    fig, ax = plt.subplots(figsize=(10, 6))
    if sns is not None:
        sns.kdeplot(data=endpoint_df, x="projection_score", hue="identity_id", common_norm=False, ax=ax)
    else:
        for identity, group in endpoint_df.groupby("identity_id", sort=True):
            ax.hist(group["projection_score"], bins=35, alpha=0.35, label=identity, density=True, histtype="step")
        ax.legend(frameon=False)
    ax.set_title(title)
    save_fig(fig, output_dir / "figures" / "kde" / residualization / f"{contrast_name}_layer_{layer:02d}")

    fig, ax = plt.subplots(figsize=(11, 6))
    if sns is not None:
        sns.stripplot(data=endpoint_df, x="identity_id", y="projection_score", hue="family", jitter=0.25, alpha=0.35, size=3, ax=ax)
        sns.pointplot(data=endpoint_df, x="identity_id", y="projection_score", errorbar=("ci", 95), color="black", ax=ax)
    else:
        ax.scatter(endpoint_df["identity_id"], endpoint_df["projection_score"], alpha=0.35, s=8)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, output_dir / "figures" / "stripplots" / residualization / f"{contrast_name}_layer_{layer:02d}")

    context = sample_plot_rows(plot_df, max_points, seed)
    order = context.groupby("canonical_label")["projection_score"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(max(12, 0.4 * len(order)), 6))
    if sns is not None:
        sns.stripplot(data=context, x="canonical_label", y="projection_score", order=order, jitter=0.25, alpha=0.35, size=3, ax=ax)
    else:
        ax.scatter(context["canonical_label"], context["projection_score"], alpha=0.35, s=8)
    ax.set_title(f"Same-axis context: {title}")
    ax.tick_params(axis="x", rotation=75)
    save_fig(fig, output_dir / "figures" / "axis_context" / residualization / f"{contrast_name}_layer_{layer:02d}")


def run_projection_analysis(x: np.ndarray, metadata: pd.DataFrame, contrasts: pd.DataFrame, layer: int, residualization: str, output_dir: Path, seed: int, max_points: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    global_mean = x.mean(axis=0, keepdims=True)
    x_centered = x - global_mean
    projection_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    directions: dict[str, np.ndarray] = {}

    for row in tqdm(list(contrasts.itertuples(index=False)), desc="contrasts", leave=False):
        contrast_name = row.contrast_name
        identity_a = row.identity_a
        identity_b = row.identity_b
        axis = row.axis
        direction, flipped = compute_direction(x_centered, metadata, identity_a, identity_b)
        if direction is None:
            continue
        directions[contrast_name] = direction
        scores = x_centered @ direction
        metrics = binary_metrics(scores, metadata, identity_a, identity_b)
        metric_row = {
            "layer": layer,
            "residualization": residualization,
            "contrast_name": contrast_name,
            "identity_a": identity_a,
            "identity_b": identity_b,
            "axis": axis,
            "sign_flipped": flipped,
            **metrics,
        }
        metric_rows.append(metric_row)

        axis_mask = metadata["axis"].eq(axis)
        plot_df = metadata.loc[axis_mask, ["prompt_id", "identity_id", "canonical_label", "family", "template_id", "required_form", "form_used", "axis"]].copy()
        plot_df["layer"] = layer
        plot_df["residualization"] = residualization
        plot_df["contrast_name"] = contrast_name
        plot_df["identity_a"] = identity_a
        plot_df["identity_b"] = identity_b
        plot_df["projection_score"] = scores[axis_mask.to_numpy()]
        plot_df["is_endpoint_identity"] = plot_df["identity_id"].isin([identity_a, identity_b])
        plot_df["endpoint_label"] = np.where(plot_df["identity_id"].eq(identity_a), "identity_a", np.where(plot_df["identity_id"].eq(identity_b), "identity_b", "other_same_axis"))
        projection_rows.extend(plot_df[[
            "layer", "residualization", "contrast_name", "identity_a", "identity_b", "axis",
            "prompt_id", "identity_id", "canonical_label", "family", "template_id",
            "required_form", "form_used", "projection_score", "is_endpoint_identity", "endpoint_label"
        ]].to_dict("records"))
        plot_projection_distributions(plot_df, metric_row, output_dir, residualization, contrast_name, layer, seed, max_points)

    return projection_rows, metric_rows, directions


def run_family_holdout(x: np.ndarray, metadata: pd.DataFrame, contrasts: pd.DataFrame, layer: int, residualization: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    families = sorted(metadata["family"].unique())
    for contrast in tqdm(list(contrasts.itertuples(index=False)), desc="family holdout", leave=False):
        identity_a = contrast.identity_a
        identity_b = contrast.identity_b
        axis = contrast.axis
        mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
        mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
        for family in families:
            heldout = metadata["family"].eq(family).to_numpy()
            train = ~heldout
            eval_a = mask_a & heldout
            eval_b = mask_b & heldout
            train_a = mask_a & train
            train_b = mask_b & train
            if min(eval_a.sum(), eval_b.sum(), train_a.sum(), train_b.sum()) == 0:
                continue
            train_mean = x[train].mean(axis=0, keepdims=True)
            x_train_centered = x - train_mean
            direction = x_train_centered[train_a].mean(axis=0) - x_train_centered[train_b].mean(axis=0)
            direction = normalize(direction)
            if direction is None:
                continue
            eval_scores = (x - train_mean) @ direction
            if eval_scores[eval_a].mean() < eval_scores[eval_b].mean():
                eval_scores = -eval_scores
            scores_a = eval_scores[eval_a]
            scores_b = eval_scores[eval_b]
            y = np.concatenate([np.ones(len(scores_a)), np.zeros(len(scores_b))])
            scores = np.concatenate([scores_a, scores_b])
            rows.append({
                "layer": layer,
                "residualization": residualization,
                "contrast_name": contrast.contrast_name,
                "identity_a": identity_a,
                "identity_b": identity_b,
                "axis": axis,
                "heldout_family": family,
                "auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
                "cohens_d": cohens_d(scores_a, scores_b),
                "n_eval_a": int(eval_a.sum()),
                "n_eval_b": int(eval_b.sum()),
            })
    return rows


def plot_layer_curves(metrics_path: Path, holdout_path: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if not metrics.empty:
        for y_col, filename, ylabel in [
            ("auc", "directional_auc_by_layer", "AUC"),
            ("cohens_d", "directional_cohens_d_by_layer", "Cohen's d"),
            ("accuracy_midpoint", "directional_accuracy_by_layer", "Accuracy at midpoint threshold"),
        ]:
            fig, ax = plt.subplots(figsize=(16, 7))
            if sns is not None:
                sns.lineplot(data=metrics, x="layer", y=y_col, hue="contrast_name", style="residualization", ax=ax)
            else:
                for name, group in metrics.groupby("contrast_name"):
                    ax.plot(group["layer"], group[y_col], label=name)
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
            ax.set_title(filename.replace("_", " ").title())
            ax.set_ylabel(ylabel)
            save_fig(fig, output_dir / "figures" / "layer_curves" / filename)

    holdout = pd.read_csv(holdout_path) if holdout_path.exists() else pd.DataFrame()
    if not holdout.empty:
        summary = holdout.groupby(["layer", "residualization"], sort=True)["auc"].agg(mean="mean", sd="std").reset_index()
        fig, ax = plt.subplots(figsize=(14, 6))
        if sns is not None:
            sns.lineplot(data=summary, x="layer", y="mean", hue="residualization", ax=ax)
        else:
            for name, group in summary.groupby("residualization"):
                ax.plot(group["layer"], group["mean"], label=name)
            ax.legend(frameon=False)
        ax.set_title("Mean family-holdout AUC by layer")
        ax.set_ylabel("Mean heldout AUC")
        save_fig(fig, output_dir / "figures" / "family_holdout" / "mean_family_holdout_auc_by_layer")
        for residualization, resid_df in holdout.groupby("residualization", sort=True):
            for contrast_name, contrast_df in resid_df.groupby("contrast_name", sort=True):
                curve = contrast_df.groupby("layer")["auc"].agg(mean="mean", sd="std").reset_index()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.errorbar(curve["layer"], curve["mean"], yerr=curve["sd"].fillna(0), linewidth=2)
                ax.set_title(f"{contrast_name}: family-holdout AUC ({residualization})")
                ax.set_xlabel("Layer")
                ax.set_ylabel("AUC")
                save_fig(fig, output_dir / "figures" / "family_holdout" / residualization / f"{contrast_name}_family_holdout_auc_by_layer")


def direction_cosine_outputs(directions: dict[str, np.ndarray], contrasts: pd.DataFrame, layer: int, residualization: str, output_dir: Path) -> list[dict[str, object]]:
    if not directions:
        return []
    names = sorted(directions)
    matrix = np.eye(len(names), dtype=np.float32)
    axis_lookup = contrasts.set_index("contrast_name")["axis"].to_dict()
    rows = []
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            cosine = float(np.dot(directions[name_i], directions[name_j]))
            matrix[i, j] = cosine
            rows.append({
                "layer": layer,
                "residualization": residualization,
                "contrast_i": name_i,
                "contrast_j": name_j,
                "axis_i": axis_lookup.get(name_i, ""),
                "axis_j": axis_lookup.get(name_j, ""),
                "cosine_similarity": cosine,
            })
    pd.DataFrame(matrix, index=names, columns=names).to_csv(output_dir / "metrics" / f"direction_cosine_matrix_layer_{layer:02d}_{residualization}.csv")
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.35), max(7, len(names) * 0.35)))
    if sns is not None:
        sns.heatmap(pd.DataFrame(matrix, index=names, columns=names), cmap="vlag", center=0, ax=ax)
    else:
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(names)), names, rotation=90)
        ax.set_yticks(np.arange(len(names)), names)
    ax.set_title(f"Contrast direction cosine similarity layer {layer:02d} ({residualization})")
    save_fig(fig, output_dir / "figures" / "direction_cosines" / residualization / f"direction_cosine_heatmap_layer_{layer:02d}")
    return rows


def plot_direction_cosine_summary(long_path: Path, output_dir: Path) -> None:
    if not long_path.exists():
        return
    df = pd.read_csv(long_path)
    df = df[df["contrast_i"] < df["contrast_j"]].copy()
    if df.empty:
        return
    df["pair_type"] = np.where(df["axis_i"].eq(df["axis_j"]), "same_axis", "different_axis")
    df["abs_cosine"] = df["cosine_similarity"].abs()
    summary = df.groupby(["layer", "residualization", "pair_type"], sort=True)["abs_cosine"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    if sns is not None:
        sns.lineplot(data=summary, x="layer", y="abs_cosine", hue="residualization", style="pair_type", ax=ax)
    else:
        for name, group in summary.groupby(["residualization", "pair_type"]):
            ax.plot(group["layer"], group["abs_cosine"], label=str(name))
        ax.legend(frameon=False)
    ax.set_title("Cross-axis entanglement: mean absolute cosine between contrast directions")
    ax.set_ylabel("Mean absolute cosine")
    save_fig(fig, output_dir / "figures" / "direction_cosines" / "cross_axis_entanglement_by_layer")


def compute_centroid_distances(x: np.ndarray, metadata: pd.DataFrame, layer: int, residualization: str, output_dir: Path) -> None:
    for axis, axis_meta in metadata.groupby("axis", sort=True):
        idx = axis_meta.index.to_numpy()
        centroids = []
        identities = []
        for identity, ident_meta in axis_meta.groupby("identity_id", sort=True):
            centroids.append(x[ident_meta.index.to_numpy()].mean(axis=0))
            identities.append(identity)
        if len(centroids) < 2:
            continue
        centroids_arr = np.vstack(centroids).astype(np.float32)
        normed = centroids_arr / np.maximum(np.linalg.norm(centroids_arr, axis=1, keepdims=True), 1e-12)
        cosine_sim = normed @ normed.T
        euclid = np.linalg.norm(centroids_arr[:, None, :] - centroids_arr[None, :, :], axis=2)
        rows = []
        for i, ident_i in enumerate(identities):
            for j, ident_j in enumerate(identities):
                rows.append({
                    "layer": layer,
                    "residualization": residualization,
                    "axis": axis,
                    "identity_i": ident_i,
                    "identity_j": ident_j,
                    "cosine_similarity": float(cosine_sim[i, j]),
                    "cosine_distance": float(1 - cosine_sim[i, j]),
                    "euclidean_distance": float(euclid[i, j]),
                })
        write_csv(output_dir / "metrics" / f"identity_centroid_distances_layer_{layer:02d}_{axis}_{residualization}.csv", rows)
        order = list(range(len(identities)))
        if linkage is not None and leaves_list is not None and squareform is not None and len(identities) > 2:
            try:
                order = leaves_list(linkage(squareform(1 - cosine_sim, checks=False), method="average")).tolist()
            except Exception:
                order = list(range(len(identities)))
        ordered_labels = [identities[i] for i in order]
        ordered = cosine_sim[np.ix_(order, order)]
        fig, ax = plt.subplots(figsize=(max(8, len(identities) * 0.35), max(7, len(identities) * 0.35)))
        if sns is not None:
            sns.heatmap(pd.DataFrame(ordered, index=ordered_labels, columns=ordered_labels), cmap="viridis", vmin=-1, vmax=1, ax=ax)
        else:
            im = ax.imshow(ordered, vmin=-1, vmax=1, cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(len(ordered_labels)), ordered_labels, rotation=90)
            ax.set_yticks(np.arange(len(ordered_labels)), ordered_labels)
        ax.set_title(f"{axis} centroid cosine similarity layer {layer:02d} ({residualization})")
        save_fig(fig, output_dir / "figures" / "centroid_distance_heatmaps" / residualization / f"{axis}_layer_{layer:02d}_cosine")


def plane_direction(x_centered: np.ndarray, metadata: pd.DataFrame, identity_a: str, identity_b: str) -> np.ndarray | None:
    direction, _ = compute_direction(x_centered, metadata, identity_a, identity_b)
    return direction


def plot_directional_planes(x: np.ndarray, metadata: pd.DataFrame, layer: int, residualization: str, output_dir: Path, seed: int, max_points: int) -> None:
    x_centered = x - x.mean(axis=0, keepdims=True)
    for axis, specs in PLANE_SPECS.items():
        if len(specs) < 2:
            continue
        d1 = plane_direction(x_centered, metadata, specs[0][1], specs[0][2])
        d2 = plane_direction(x_centered, metadata, specs[1][1], specs[1][2])
        if d1 is None or d2 is None:
            continue
        e1 = normalize(d1)
        if e1 is None:
            continue
        d2_orth = d2 - np.dot(d2, e1) * e1
        e2 = normalize(d2_orth)
        if e2 is None:
            continue
        axis_mask = metadata["axis"].eq(axis)
        z1 = x_centered[axis_mask.to_numpy()] @ e1
        z2 = x_centered[axis_mask.to_numpy()] @ e2
        plane_df = metadata.loc[axis_mask, ["identity_id", "canonical_label"]].copy()
        plane_df["z1"] = z1
        plane_df["z2"] = z2
        plane_df = sample_plot_rows(plane_df, max_points, seed)
        hue = "canonical_label" if plane_df["canonical_label"].nunique() <= 60 else "identity_id"
        colors = category_colors(plane_df[hue])
        markers = category_markers(plane_df[hue])
        centroids = plane_df.groupby(hue, sort=True)[["z1", "z2"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 7))
        for label, group in plane_df.groupby(hue, sort=True):
            label = str(label)
            ax.scatter(group["z1"], group["z2"], c=colors[label], marker=markers[label], s=9, alpha=0.35, linewidths=0)
        for label, group in centroids.groupby(hue, sort=True):
            label = str(label)
            ax.scatter(group["z1"], group["z2"], c=colors[label], marker=markers[label], s=65, edgecolors="black", linewidths=0.5)
        if centroids[hue].nunique() <= 15:
            for _, row in centroids.iterrows():
                ax.text(row["z1"], row["z2"], str(row[hue]), fontsize=7)
        ax.set_title(f"{axis} directional plane layer {layer:02d} ({residualization})")
        ax.set_xlabel(specs[0][0])
        ax.set_ylabel(specs[1][0])
        add_legend(ax, sorted(plane_df[hue].astype(str).unique()), colors, markers, max_items=60)
        save_fig(fig, output_dir / "figures" / "directional_planes" / residualization / f"{axis}_layer_{layer:02d}")


def plot_plane_progressions(output_dir: Path, layers: list[int], residualizations: list[str]) -> None:
    # Individual layer plots are already saved. Progression assembly is kept
    # light-weight by leaving the per-layer files as the primary output.
    # Users can compare layer_XX files side-by-side in the output directory.
    for residualization in residualizations:
        for axis in PLANE_SPECS:
            summary_path = output_dir / "figures" / "directional_planes" / residualization / f"{axis}_layer_progression.txt"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                "Layer progression is represented by the per-layer PNG/PDF files: "
                + ", ".join(f"{axis}_layer_{layer:02d}" for layer in layers)
                + "\n"
            )


def main() -> None:
    args = parse_args()
    layers = parse_int_list(args.layers)
    residualizations = parse_str_list(args.residualizations)
    invalid = [item for item in residualizations if item not in RESIDUALIZATION_GROUPS]
    if invalid:
        raise ValueError(f"Unknown residualizations: {invalid}")
    dirs = prepare_dirs(args.output_dir, args.overwrite)
    metadata = load_metadata(args.activation_dir)
    contrasts = load_contrasts(args.contrasts, metadata)
    run_config = {
        "activation_dir": str(args.activation_dir),
        "output_dir": str(args.output_dir),
        "layers": layers,
        "residualizations": residualizations,
        "max_points_per_plot": args.max_points_per_plot,
        "random_seed": args.random_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    direction_cosine_rows: list[dict[str, object]] = []
    all_metric_path = dirs["metrics"] / "directional_metrics.csv"
    holdout_path = dirs["metrics"] / "directional_family_holdout_metrics.csv"
    projection_path = dirs["projections"] / "projection_scores.csv"
    for path in [all_metric_path, holdout_path, projection_path, dirs["metrics"] / "direction_cosines_long.csv"]:
        if path.exists():
            path.unlink()

    for layer in tqdm(layers, desc="layers"):
        start_layer = time.perf_counter()
        print(f"\nLayer {layer:02d}")
        x_raw = load_layer(args.activation_dir, layer, len(metadata))
        variants = {}
        for residualization in tqdm(residualizations, desc="residualizing", leave=False):
            start = time.perf_counter()
            variants[residualization] = residualize(x_raw, metadata, residualization)
            print(f"  {residualization} residualization ready in {elapsed(start)}")

        for residualization, x in tqdm(variants.items(), desc="residualization analyses", leave=False):
            start = time.perf_counter()
            projection_rows, metric_rows, directions = run_projection_analysis(
                x, metadata, contrasts, layer, residualization, args.output_dir, args.random_seed, args.max_points_per_plot
            )
            append_csv(projection_path, projection_rows)
            append_csv(all_metric_path, metric_rows)
            print(f"  Projections/metrics {residualization} finished in {elapsed(start)}")

            start = time.perf_counter()
            holdout_rows = run_family_holdout(x, metadata, contrasts, layer, residualization)
            append_csv(holdout_path, holdout_rows)
            print(f"  Family holdout {residualization} finished in {elapsed(start)}")

            start = time.perf_counter()
            direction_cosine_rows.extend(direction_cosine_outputs(directions, contrasts, layer, residualization, args.output_dir))
            print(f"  Direction cosines {residualization} finished in {elapsed(start)}")

            start = time.perf_counter()
            compute_centroid_distances(x, metadata, layer, residualization, args.output_dir)
            print(f"  Centroid distances {residualization} finished in {elapsed(start)}")

            start = time.perf_counter()
            plot_directional_planes(x, metadata, layer, residualization, args.output_dir, args.random_seed, args.max_points_per_plot)
            print(f"  Directional planes {residualization} finished in {elapsed(start)}")

        print(f"Layer {layer:02d} complete in {elapsed(start_layer)}")

    write_csv(dirs["metrics"] / "direction_cosines_long.csv", direction_cosine_rows)
    print("\nPlotting layer summaries")
    plot_layer_curves(all_metric_path, holdout_path, args.output_dir)
    plot_direction_cosine_summary(dirs["metrics"] / "direction_cosines_long.csv", args.output_dir)
    plot_plane_progressions(args.output_dir, layers, residualizations)
    print(f"\nDirectional visualizations complete: {args.output_dir}")


# Fast selected-layer run:
#
# python /workspace/status_mi/scripts/plot_identity_directional_visualizations.py \
#   --activation_dir /workspace/status_mi/results/activations/llama-3.1-8b/identity_prompts_final_token/ \
#   --output_dir /workspace/status_mi/results/geometry/llama-3.1-8b/identity_prompts_final_token/directional_visualizations/ \
#   --layers 8,16,24,32 \
#   --residualizations raw,family_residualized,template_residualized \
#   --overwrite
#
# Template-residualized only, fastest:
#
# python /workspace/status_mi/scripts/plot_identity_directional_visualizations.py \
#   --activation_dir /workspace/status_mi/results/activations/llama-3.1-8b/identity_prompts_final_token/ \
#   --output_dir /workspace/status_mi/results/geometry/llama-3.1-8b/identity_prompts_final_token/directional_visualizations_template_resid/ \
#   --layers 8,16,24,32 \
#   --residualizations template_residualized \
#   --overwrite
if __name__ == "__main__":
    main()
