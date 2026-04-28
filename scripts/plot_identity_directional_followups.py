#!/usr/bin/env python3
"""Higher-level direction-focused analyses for identity geometry.

This script complements the first directional visualization pass by focusing
on interpretable identity contrast directions: emergence across layers,
residualization comparisons, direction cosine structure, centroid orderings,
family-to-family generalization, and direction stability.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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
    "identity_prompts_final_token/directional_followups"
)
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
METADATA_COLUMNS = [
    "prompt_id",
    "identity_id",
    "axis",
    "canonical_label",
    "template_id",
    "family",
    "required_form",
]
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
KEY_CONTRASTS = [
    "sexuality_gay_vs_sexuality_straight",
    "race_black_vs_race_white",
    "gender_transgender_vs_gender_cisgender",
    "appearance_obese_vs_appearance_thin",
    "ses_low_income_vs_ses_rich",
    "disability_disabled_vs_disability_able_bodied",
]
RESIDUALIZATION_COMPARISON_CONTRASTS = [
    "appearance_obese_vs_appearance_thin",
    "gender_transgender_vs_gender_cisgender",
    "sexuality_gay_vs_sexuality_straight",
    "race_black_vs_race_white",
    "ses_low_income_vs_ses_rich",
]
CENTROID_ORDERING_CONTRASTS = [
    "sexuality_gay_vs_sexuality_straight",
    "sexuality_bisexual_vs_sexuality_straight",
    "race_black_vs_race_white",
    "gender_transgender_vs_gender_cisgender",
    "ses_low_income_vs_ses_rich",
    "ses_lower_class_vs_ses_upper_class",
    "appearance_obese_vs_appearance_thin",
    "appearance_short_vs_appearance_tall",
    "disability_disabled_vs_disability_able_bodied",
    "religion_muslim_vs_religion_christian",
    "religion_jewish_vs_religion_christian",
]
PLANE_SPECS = {
    "sexual_orientation": [
        ("gay - straight", "sexuality_gay", "sexuality_straight"),
        ("bisexual - straight", "sexuality_bisexual", "sexuality_straight"),
    ],
    "gender_identity": [
        ("transgender - cisgender", "gender_transgender", "gender_cisgender"),
        ("transgender woman - cisgender woman", "gender_transgender_woman", "gender_cisgender_woman"),
    ],
}
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
LINESTYLES = ["-", "--", "-.", ":"]
LAYERWISE_COLUMNS = [
    "layer",
    "residualization",
    "contrast_name",
    "axis",
    "identity_a",
    "identity_b",
    "auc",
    "cohens_d",
    "accuracy_midpoint",
    "mean_a",
    "mean_b",
    "sd_a",
    "sd_b",
    "n_a",
    "n_b",
    "sign_flipped",
]
COSINE_LONG_COLUMNS = [
    "layer",
    "residualization",
    "contrast_i",
    "contrast_j",
    "axis_i",
    "axis_j",
    "cosine_similarity",
]
CENTROID_COLUMNS = [
    "layer",
    "residualization",
    "contrast_name",
    "axis",
    "identity_id",
    "canonical_label",
    "mean_projection",
    "sd_projection",
    "se_projection",
    "ci95_low",
    "ci95_high",
    "n",
    "is_endpoint",
]
FAMILY_GENERALIZATION_COLUMNS = [
    "layer",
    "residualization",
    "contrast_name",
    "axis",
    "train_family",
    "test_family",
    "auc",
    "cohens_d",
    "n_train_a",
    "n_train_b",
    "n_test_a",
    "n_test_b",
]
STABILITY_COLUMNS = [
    "residualization",
    "contrast_name",
    "axis",
    "layer",
    "next_layer",
    "adjacent_cosine",
    "reference_layer",
    "reference_cosine",
]


@dataclass
class DirectionRecord:
    direction: np.ndarray
    global_mean: np.ndarray
    sign_flipped: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direction-focused identity geometry follow-up analyses.")
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default=None, help="Comma-separated layers. Default: all layer_*.npy, skipping layer_00 if zero variance.")
    parser.add_argument("--selected_layers_for_planes", default="8,16,24,32")
    parser.add_argument("--residualizations", default=",".join(DEFAULT_RESIDUALIZATIONS))
    parser.add_argument("--main_residualization", default="family_residualized")
    parser.add_argument("--contrasts_csv", type=Path, default=None)
    parser.add_argument("--max_points_per_plot", type=int, default=20000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def save_fig(fig: plt.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=220)
    fig.savefig(path_no_suffix.with_suffix(".pdf"))
    plt.close(fig)


def append_rows(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    df = pd.DataFrame(rows).reindex(columns=columns)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def write_csv(path: Path, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = df.reindex(columns=columns)
    df.to_csv(path, index=False)


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists and is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    for subdir in [
        "metrics",
        "metrics/direction_cosines",
        "figures/layerwise",
        "figures/residualization_comparison",
        "figures/direction_cosines",
        "figures/centroid_ordering",
        "figures/family_generalization",
        "figures/direction_stability",
        "figures/paper_panels",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def load_metadata(activation_dir: Path) -> pd.DataFrame:
    path = activation_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")
    metadata = pd.read_csv(path, keep_default_na=False).reset_index(drop=True)
    missing = [col for col in METADATA_COLUMNS if col not in metadata.columns]
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")
    return metadata


def load_layer(activation_dir: Path, layer: int) -> np.ndarray:
    path = activation_dir / f"layer_{layer:02d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing layer activation file: {path}")
    x = np.load(path, mmap_mode="r")
    if x.ndim != 2:
        raise ValueError(f"{path} should be a 2D array, found shape {x.shape}")
    return np.asarray(x, dtype=np.float32)


def discover_layers(activation_dir: Path) -> list[int]:
    layers = sorted(int(path.stem.split("_")[-1]) for path in activation_dir.glob("layer_*.npy"))
    if not layers:
        raise FileNotFoundError(f"No layer_*.npy files found in {activation_dir}")
    if 0 in layers:
        layer0 = np.load(activation_dir / "layer_00.npy", mmap_mode="r")
        stride = max(1, layer0.shape[0] // 1000)
        sample = np.asarray(layer0[::stride], dtype=np.float32)
        if not np.isfinite(sample).all() or float(np.nanvar(sample)) == 0.0:
            print("Skipping layer_00 because sampled activations have zero/non-finite variance.")
            layers = [layer for layer in layers if layer != 0]
    return layers


def residualize(x: np.ndarray, metadata: pd.DataFrame, residualization: str) -> np.ndarray:
    if residualization not in RESIDUALIZATION_GROUPS:
        raise ValueError(f"Unknown residualization '{residualization}'. Expected one of {sorted(RESIDUALIZATION_GROUPS)}")
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


def normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        return None
    return vec / norm


def compute_contrast_direction(
    x: np.ndarray,
    metadata: pd.DataFrame,
    identity_a: str,
    identity_b: str,
    center: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
    global_mean = x.mean(axis=0, keepdims=True) if center else np.zeros((1, x.shape[1]), dtype=np.float32)
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None, global_mean, False
    centered = x - global_mean
    direction = normalize(centered[mask_a].mean(axis=0) - centered[mask_b].mean(axis=0))
    if direction is None:
        return None, global_mean, False
    scores = centered @ direction
    sign_flipped = False
    if scores[mask_a].mean() < scores[mask_b].mean():
        direction = -direction
        sign_flipped = True
    return direction, global_mean, sign_flipped


def compute_masked_contrast_direction(
    x: np.ndarray,
    metadata: pd.DataFrame,
    identity_a: str,
    identity_b: str,
    row_mask: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    global_mean = x.mean(axis=0, keepdims=True)
    centered = x - global_mean
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy() & row_mask
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy() & row_mask
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None, global_mean, False
    direction = normalize(centered[mask_a].mean(axis=0) - centered[mask_b].mean(axis=0))
    if direction is None:
        return None, global_mean, False
    scores = centered @ direction
    sign_flipped = False
    if scores[mask_a].mean() < scores[mask_b].mean():
        direction = -direction
        sign_flipped = True
    return direction, global_mean, sign_flipped


def project_scores(x: np.ndarray, direction: np.ndarray, global_mean: np.ndarray) -> np.ndarray:
    return (x - global_mean) @ direction


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if len(scores_a) < 2 or len(scores_b) < 2:
        return float("nan")
    pooled = (((len(scores_a) - 1) * np.var(scores_a, ddof=1)) + ((len(scores_b) - 1) * np.var(scores_b, ddof=1))) / (len(scores_a) + len(scores_b) - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((scores_a.mean() - scores_b.mean()) / np.sqrt(pooled))


def compute_auc_cohens_d(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels, scores))
    scores_a = scores[labels == 1]
    scores_b = scores[labels == 0]
    return auc, cohens_d(scores_a, scores_b)


def endpoint_metrics(scores: np.ndarray, metadata: pd.DataFrame, identity_a: str, identity_b: str) -> dict[str, float | int]:
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
    scores_a = scores[mask_a]
    scores_b = scores[mask_b]
    labels = np.concatenate([np.ones(len(scores_a)), np.zeros(len(scores_b))])
    pair_scores = np.concatenate([scores_a, scores_b])
    auc, d_value = compute_auc_cohens_d(pair_scores, labels)
    mean_a = float(scores_a.mean()) if len(scores_a) else float("nan")
    mean_b = float(scores_b.mean()) if len(scores_b) else float("nan")
    midpoint = (mean_a + mean_b) / 2
    accuracy = float(np.mean(np.concatenate([scores_a >= midpoint, scores_b < midpoint]))) if len(pair_scores) else float("nan")
    return {
        "auc": auc,
        "cohens_d": d_value,
        "accuracy_midpoint": accuracy,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "sd_a": float(scores_a.std(ddof=1)) if len(scores_a) > 1 else float("nan"),
        "sd_b": float(scores_b.std(ddof=1)) if len(scores_b) > 1 else float("nan"),
        "n_a": int(len(scores_a)),
        "n_b": int(len(scores_b)),
    }


def stratified_sample_for_plot(df: pd.DataFrame, group_col: str, max_n: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_n:
        return df
    n_groups = max(1, df[group_col].nunique())
    per_group = max(1, int(np.ceil(max_n / n_groups)))
    sampled = (
        df.groupby(group_col, group_keys=False, sort=False)
        .sample(frac=1, random_state=seed)
        .groupby(group_col, group_keys=False, sort=False)
        .head(per_group)
    )
    if len(sampled) > max_n:
        sampled = sampled.sample(n=max_n, random_state=seed)
    return sampled


def load_contrasts(path: Path | None, metadata: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        contrasts = pd.DataFrame(DEFAULT_CONTRASTS, columns=["contrast_name", "identity_a", "identity_b", "axis"])
    else:
        contrasts = pd.read_csv(path, keep_default_na=False)
    required = {"contrast_name", "identity_a", "identity_b", "axis"}
    missing = required - set(contrasts.columns)
    if missing:
        raise ValueError(f"Contrast CSV missing columns: {sorted(missing)}")
    identities = set(metadata["identity_id"])
    valid = contrasts[contrasts["identity_a"].isin(identities) & contrasts["identity_b"].isin(identities)].copy()
    skipped = len(contrasts) - len(valid)
    if skipped:
        print(f"Skipping {skipped} contrasts because one or both identity IDs are absent.")
    return valid.reset_index(drop=True)


def color_map(labels: list[str]) -> dict[str, str]:
    return {label: OKABE_ITO[i % len(OKABE_ITO)] for i, label in enumerate(labels)}


def marker_map(labels: list[str]) -> dict[str, str]:
    return {label: MARKERS[i % len(MARKERS)] for i, label in enumerate(labels)}


def add_outside_legend(ax: plt.Axes, max_items: int = 40) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if len(labels) > max_items:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.text(1.01, 1.0, f"{len(labels)} series\nlegend omitted", transform=ax.transAxes, va="top", fontsize=8)
        return
    ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)


def plot_metric_by_axis(metrics: pd.DataFrame, residualization: str, y_col: str, ylabel: str, output_path: Path) -> None:
    df = metrics[metrics["residualization"].eq(residualization)].copy()
    if df.empty:
        return
    axes = sorted(df["axis"].unique())
    ncols = min(2, len(axes))
    nrows = int(np.ceil(len(axes) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(9 * ncols, 4.8 * nrows), squeeze=False, sharex=True)
    for ax, axis in zip(axs.ravel(), axes):
        axis_df = df[df["axis"].eq(axis)]
        if sns is not None:
            sns.lineplot(data=axis_df, x="layer", y=y_col, hue="contrast_name", ax=ax, linewidth=2, marker=None)
        else:
            for name, group in axis_df.groupby("contrast_name", sort=True):
                ax.plot(group["layer"], group[y_col], linewidth=2, label=name)
        ax.set_title(axis.replace("_", " "))
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        if y_col in {"auc", "accuracy_midpoint"}:
            ax.axhline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.2)
        add_outside_legend(ax)
    for ax in axs.ravel()[len(axes):]:
        ax.axis("off")
    fig.suptitle(f"{ylabel} by layer ({residualization})", y=1.01, fontsize=14)
    save_fig(fig, output_path)


def plot_layerwise_figures(metrics_path: Path, output_dir: Path, residualizations: list[str]) -> None:
    if not metrics_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return
    for residualization in residualizations:
        plot_metric_by_axis(
            metrics,
            residualization,
            "auc",
            "AUC",
            output_dir / "figures" / "layerwise" / residualization / "auc_by_layer_all_contrasts",
        )
        plot_metric_by_axis(
            metrics,
            residualization,
            "cohens_d",
            "Cohen's d",
            output_dir / "figures" / "layerwise" / residualization / "cohens_d_by_layer_all_contrasts",
        )
    key = metrics[metrics["contrast_name"].isin(KEY_CONTRASTS) & metrics["residualization"].isin(["raw", "family_residualized", "template_residualized"])].copy()
    if key.empty:
        return
    for y_col, ylabel, filename in [
        ("auc", "AUC", "key_contrasts_auc_by_layer_residualization_comparison"),
        ("cohens_d", "Cohen's d", "key_contrasts_d_by_layer_residualization_comparison"),
    ]:
        fig, ax = plt.subplots(figsize=(18, 8))
        if sns is not None:
            sns.lineplot(data=key, x="layer", y=y_col, hue="contrast_name", style="residualization", ax=ax, linewidth=2.2, marker=None)
        else:
            colors = color_map(sorted(key["contrast_name"].unique()))
            styles = {name: LINESTYLES[i % len(LINESTYLES)] for i, name in enumerate(sorted(key["residualization"].unique()))}
            for (contrast, resid), group in key.groupby(["contrast_name", "residualization"], sort=True):
                ax.plot(group["layer"], group[y_col], color=colors[contrast], linestyle=styles[resid], linewidth=2.2, label=f"{contrast} | {resid}")
        if y_col == "auc":
            ax.axhline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_ylim(0.45, 1.02)
        ax.set_title(f"Key identity contrast {ylabel} by layer and residualization")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        add_outside_legend(ax, max_items=80)
        save_fig(fig, output_dir / "figures" / "layerwise" / filename)


def projection_distribution_rows(
    x: np.ndarray,
    metadata: pd.DataFrame,
    contrast: pd.Series,
    residualization: str,
    layer: int,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]] | None:
    direction, global_mean, sign_flipped = compute_contrast_direction(x, metadata, contrast.identity_a, contrast.identity_b)
    if direction is None:
        return None
    scores = project_scores(x, direction, global_mean)
    mask = metadata["identity_id"].isin([contrast.identity_a, contrast.identity_b])
    plot_df = metadata.loc[mask, ["identity_id", "canonical_label", "family", "template_id", "required_form"]].copy()
    plot_df["projection_score"] = scores[mask.to_numpy()]
    plot_df["residualization"] = residualization
    plot_df["layer"] = layer
    metrics = endpoint_metrics(scores, metadata, contrast.identity_a, contrast.identity_b)
    metrics["sign_flipped"] = sign_flipped
    return plot_df, metrics


def plot_residualization_comparison(
    contrast_name: str,
    comparison_data: dict[str, tuple[pd.DataFrame, dict[str, float | int | bool]]],
    output_dir: Path,
    layer: int,
    max_points: int,
    seed: int,
) -> None:
    if not comparison_data:
        return
    residualizations = [resid for resid in DEFAULT_RESIDUALIZATIONS if resid in comparison_data]
    fig, axs = plt.subplots(1, len(residualizations), figsize=(6 * len(residualizations), 5), squeeze=False, sharey=False)
    for ax, residualization in zip(axs.ravel(), residualizations):
        plot_df, metrics = comparison_data[residualization]
        plot_df = stratified_sample_for_plot(plot_df, "identity_id", max_points, seed)
        if sns is not None:
            sns.histplot(data=plot_df, x="projection_score", hue="canonical_label", stat="density", common_norm=False, bins=35, alpha=0.45, ax=ax)
        else:
            for label, group in plot_df.groupby("canonical_label", sort=True):
                ax.hist(group["projection_score"], bins=35, density=True, alpha=0.45, label=label)
            ax.legend(frameon=False)
        ax.set_title(f"{residualization}\nAUC={metrics['auc']:.3f}, d={metrics['cohens_d']:.2f}")
        ax.set_xlabel("Projection score")
        ax.set_ylabel("Density")
    fig.suptitle(f"{contrast_name}: endpoint projection distributions at layer {layer:02d}", y=1.04)
    save_fig(fig, output_dir / "figures" / "residualization_comparison" / f"{contrast_name}_layer_{layer:02d}_projection_distributions")

    combined = pd.concat([value[0] for value in comparison_data.values()], ignore_index=True)
    combined = stratified_sample_for_plot(combined, "identity_id", max_points, seed)
    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(residualizations)), 6))
    if sns is not None:
        sns.violinplot(data=combined, x="residualization", y="projection_score", hue="canonical_label", inner=None, cut=0, alpha=0.35, ax=ax)
        sns.stripplot(data=combined, x="residualization", y="projection_score", hue="canonical_label", dodge=True, alpha=0.22, size=2, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        half = len(handles) // 2
        ax.legend(handles[:half], labels[:half], bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    else:
        for i, (resid, group) in enumerate(combined.groupby("residualization", sort=False)):
            jitter = np.random.default_rng(seed + i).normal(0, 0.06, size=len(group))
            ax.scatter(np.full(len(group), i) + jitter, group["projection_score"], s=8, alpha=0.25)
        ax.set_xticks(range(len(residualizations)), residualizations, rotation=20)
    ax.set_title(f"{contrast_name}: raw vs residualized endpoint projections at layer {layer:02d}")
    ax.set_xlabel("Residualization")
    ax.set_ylabel("Projection score")
    save_fig(fig, output_dir / "figures" / "residualization_comparison" / f"{contrast_name}_layer_{layer:02d}_strip_residualization_comparison")


def direction_cosines(
    direction_records: dict[tuple[int, str, str], DirectionRecord],
    contrasts: pd.DataFrame,
    layer: int,
    residualization: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    names = [name for name in contrasts["contrast_name"] if (layer, residualization, name) in direction_records]
    if len(names) < 2:
        return []
    axis_lookup = contrasts.set_index("contrast_name")["axis"].to_dict()
    matrix = np.eye(len(names), dtype=np.float32)
    rows: list[dict[str, object]] = []
    for i, name_i in enumerate(names):
        d_i = direction_records[(layer, residualization, name_i)].direction
        for j, name_j in enumerate(names):
            d_j = direction_records[(layer, residualization, name_j)].direction
            cosine = float(np.dot(d_i, d_j))
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
    matrix_df = pd.DataFrame(matrix, index=names, columns=names)
    matrix_df.to_csv(output_dir / "metrics" / "direction_cosines" / f"direction_cosine_matrix_layer_{layer:02d}_{residualization}.csv")
    plot_direction_cosine_heatmap(matrix_df, residualization, layer, output_dir)
    return rows


def order_similarity_matrix(matrix_df: pd.DataFrame) -> pd.DataFrame:
    if leaves_list is None or linkage is None or squareform is None or len(matrix_df) < 3:
        return matrix_df
    distance = 1 - matrix_df.to_numpy()
    np.fill_diagonal(distance, 0)
    try:
        order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
        return matrix_df.iloc[order, order]
    except Exception as exc:  # pragma: no cover
        print(f"Warning: clustering direction cosine heatmap failed: {exc}")
        return matrix_df


def plot_direction_cosine_heatmap(matrix_df: pd.DataFrame, residualization: str, layer: int, output_dir: Path) -> None:
    ordered = order_similarity_matrix(matrix_df)
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(ordered)), max(8, 0.45 * len(ordered))))
    if sns is not None:
        sns.heatmap(ordered, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, square=True)
    else:
        im = ax.imshow(ordered, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(ordered)), ordered.columns, rotation=90)
        ax.set_yticks(range(len(ordered)), ordered.index)
    ax.set_title(f"Contrast direction cosine similarity, layer {layer:02d} ({residualization})")
    save_fig(fig, output_dir / "figures" / "direction_cosines" / residualization / f"direction_cosine_heatmap_layer_{layer:02d}")


def plot_direction_cosine_summaries(cosine_path: Path, output_dir: Path) -> None:
    if not cosine_path.exists():
        return
    df = pd.read_csv(cosine_path)
    df = df[df["contrast_i"] < df["contrast_j"]].copy()
    if df.empty:
        return
    df["same_axis"] = np.where(df["axis_i"].eq(df["axis_j"]), "same axis", "different axes")
    df["abs_cosine_similarity"] = df["cosine_similarity"].abs()
    summary = df.groupby(["layer", "residualization", "same_axis"], sort=True)["abs_cosine_similarity"].mean().reset_index(name="mean_abs_cosine")
    fig, ax = plt.subplots(figsize=(15, 7))
    if sns is not None:
        sns.lineplot(data=summary, x="layer", y="mean_abs_cosine", hue="same_axis", style="residualization", ax=ax, linewidth=2.2, marker=None)
    else:
        for name, group in summary.groupby(["same_axis", "residualization"], sort=True):
            ax.plot(group["layer"], group["mean_abs_cosine"], label=str(name), linewidth=2.2)
        ax.legend(frameon=False)
    ax.set_title("Cross-axis entanglement: mean absolute cosine between contrast directions")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean absolute cosine")
    ax.set_ylim(0, min(1, max(0.1, summary["mean_abs_cosine"].max() * 1.1)))
    ax.grid(alpha=0.2)
    add_outside_legend(ax)
    save_fig(fig, output_dir / "figures" / "direction_cosines" / "cross_axis_entanglement_by_layer")

    top = df.sort_values("abs_cosine_similarity", ascending=False).head(200).copy()
    top.to_csv(output_dir / "metrics" / "direction_cosines" / "top_direction_alignments.csv", index=False)


def centroid_ordering_rows(
    x: np.ndarray,
    metadata: pd.DataFrame,
    contrast: pd.Series,
    layer: int,
    residualization: str,
    direction_record: DirectionRecord,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    scores = project_scores(x, direction_record.direction, direction_record.global_mean)
    axis_mask = metadata["axis"].eq(contrast.axis)
    plot_df = metadata.loc[axis_mask, ["identity_id", "canonical_label"]].copy()
    plot_df["projection_score"] = scores[axis_mask.to_numpy()]
    rows: list[dict[str, object]] = []
    for (identity_id, canonical_label), group in plot_df.groupby(["identity_id", "canonical_label"], sort=True):
        values = group["projection_score"].to_numpy()
        n = len(values)
        mean = float(values.mean()) if n else float("nan")
        sd = float(values.std(ddof=1)) if n > 1 else float("nan")
        se = sd / np.sqrt(n) if n > 1 and np.isfinite(sd) else float("nan")
        ci = 1.96 * se if np.isfinite(se) else float("nan")
        rows.append({
            "layer": layer,
            "residualization": residualization,
            "contrast_name": contrast.contrast_name,
            "axis": contrast.axis,
            "identity_id": identity_id,
            "canonical_label": canonical_label,
            "mean_projection": mean,
            "sd_projection": sd,
            "se_projection": se,
            "ci95_low": mean - ci if np.isfinite(ci) else float("nan"),
            "ci95_high": mean + ci if np.isfinite(ci) else float("nan"),
            "n": int(n),
            "is_endpoint": identity_id in {contrast.identity_a, contrast.identity_b},
        })
    return rows, plot_df


def plot_centroid_ordering(
    centroid_rows: list[dict[str, object]],
    prompt_df: pd.DataFrame,
    contrast_name: str,
    residualization: str,
    layer: int,
    output_dir: Path,
    max_points: int,
    seed: int,
) -> None:
    if not centroid_rows:
        return
    centroid_df = pd.DataFrame(centroid_rows).sort_values("mean_projection")
    y_pos = np.arange(len(centroid_df))
    fig, ax = plt.subplots(figsize=(12, max(6, 0.38 * len(centroid_df))))
    prompt_df = stratified_sample_for_plot(prompt_df, "identity_id", max_points, seed)
    label_to_pos = {row.canonical_label: i for i, row in enumerate(centroid_df.itertuples(index=False))}
    prompt_y = prompt_df["canonical_label"].map(label_to_pos).astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    ax.scatter(prompt_df["projection_score"], prompt_y + rng.normal(0, 0.08, size=len(prompt_y)), s=8, alpha=0.12, color="gray", linewidths=0)
    colors = np.where(centroid_df["is_endpoint"], "#D55E00", "#0072B2")
    markers = np.where(centroid_df["is_endpoint"], "D", "o")
    for i, row in enumerate(centroid_df.itertuples(index=False)):
        xerr = np.array([[row.mean_projection - row.ci95_low], [row.ci95_high - row.mean_projection]])
        ax.errorbar(row.mean_projection, i, xerr=xerr, fmt=markers[i], color=colors[i], ecolor=colors[i], markersize=7 if row.is_endpoint else 5, capsize=3, markeredgecolor="black", markeredgewidth=0.6)
    labels = [f"{label}{' *' if is_endpoint else ''}" for label, is_endpoint in zip(centroid_df["canonical_label"], centroid_df["is_endpoint"])]
    ax.set_yticks(y_pos, labels)
    ax.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title(f"{contrast_name}: same-axis identity ordering along contrast direction\nLayer {layer:02d}, {residualization}; endpoints marked with *")
    ax.set_xlabel("Mean projection score with 95% CI")
    ax.set_ylabel("Identity")
    ax.grid(axis="x", alpha=0.2)
    save_fig(fig, output_dir / "figures" / "centroid_ordering" / residualization / f"{contrast_name}_layer_{layer:02d}_centroid_ordering")


def family_to_family_rows(
    x: np.ndarray,
    metadata: pd.DataFrame,
    contrast: pd.Series,
    layer: int,
    residualization: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    families = sorted(metadata["family"].unique())
    mask_a = metadata["identity_id"].eq(contrast.identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(contrast.identity_b).to_numpy()
    for train_family in families:
        train_mask = metadata["family"].eq(train_family).to_numpy()
        direction, global_mean, _ = compute_masked_contrast_direction(x, metadata, contrast.identity_a, contrast.identity_b, train_mask)
        if direction is None:
            continue
        scores = project_scores(x, direction, global_mean)
        for test_family in families:
            test_mask = metadata["family"].eq(test_family).to_numpy()
            rows.append(make_family_eval_row(scores, mask_a, mask_b, train_mask, test_mask, contrast, layer, residualization, train_family, test_family))
    for heldout_family in families:
        heldout = metadata["family"].eq(heldout_family).to_numpy()
        train_mask = ~heldout
        direction, global_mean, _ = compute_masked_contrast_direction(x, metadata, contrast.identity_a, contrast.identity_b, train_mask)
        if direction is None:
            continue
        scores = project_scores(x, direction, global_mean)
        rows.append(make_family_eval_row(scores, mask_a, mask_b, train_mask, heldout, contrast, layer, residualization, f"all_except_{heldout_family}", heldout_family))
    return rows


def make_family_eval_row(
    scores: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    contrast: pd.Series,
    layer: int,
    residualization: str,
    train_family: str,
    test_family: str,
) -> dict[str, object]:
    train_a = mask_a & train_mask
    train_b = mask_b & train_mask
    test_a = mask_a & test_mask
    test_b = mask_b & test_mask
    scores_a = scores[test_a]
    scores_b = scores[test_b]
    labels = np.concatenate([np.ones(len(scores_a)), np.zeros(len(scores_b))])
    pair_scores = np.concatenate([scores_a, scores_b])
    auc, d_value = compute_auc_cohens_d(pair_scores, labels) if min(len(scores_a), len(scores_b)) else (float("nan"), float("nan"))
    return {
        "layer": layer,
        "residualization": residualization,
        "contrast_name": contrast.contrast_name,
        "axis": contrast.axis,
        "train_family": train_family,
        "test_family": test_family,
        "auc": auc,
        "cohens_d": d_value,
        "n_train_a": int(train_a.sum()),
        "n_train_b": int(train_b.sum()),
        "n_test_a": int(test_a.sum()),
        "n_test_b": int(test_b.sum()),
    }


def plot_family_generalization(family_path: Path, output_dir: Path, selected_layer: int, residualizations: list[str]) -> None:
    if not family_path.exists():
        return
    df = pd.read_csv(family_path)
    if df.empty:
        return
    one_to_one = df[~df["train_family"].str.startswith("all_except_", na=False)].copy()
    if one_to_one.empty:
        return
    for residualization in residualizations:
        subset = one_to_one[(one_to_one["residualization"].eq(residualization)) & (one_to_one["layer"].eq(selected_layer)) & (one_to_one["contrast_name"].isin(KEY_CONTRASTS))]
        for contrast_name, contrast_df in subset.groupby("contrast_name", sort=True):
            pivot = contrast_df.pivot_table(index="train_family", columns="test_family", values="auc", aggfunc="mean")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.columns)), max(6, 0.6 * len(pivot.index))))
            if sns is not None:
                sns.heatmap(pivot, cmap="viridis", vmin=0.5, vmax=1.0, annot=True, fmt=".2f", ax=ax)
            else:
                im = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0.5, vmax=1.0)
                fig.colorbar(im, ax=ax)
                ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(pivot.index)), pivot.index)
            ax.set_title(f"{contrast_name}: train-family to test-family AUC\nLayer {selected_layer:02d}, {residualization}")
            ax.set_xlabel("Test family")
            ax.set_ylabel("Train family")
            save_fig(fig, output_dir / "figures" / "family_generalization" / residualization / f"{contrast_name}_layer_{selected_layer:02d}_family_to_family_auc_heatmap")

    offdiag = one_to_one[one_to_one["train_family"].ne(one_to_one["test_family"])].copy()
    summary = offdiag.groupby(["layer", "residualization", "contrast_name", "axis"], sort=True)["auc"].mean().reset_index(name="mean_offdiag_auc")
    write_csv(output_dir / "metrics" / "family_to_family_summary.csv", summary)
    if summary.empty:
        return
    plot_df = summary[summary["contrast_name"].isin(KEY_CONTRASTS)].copy()
    fig, ax = plt.subplots(figsize=(16, 7))
    if sns is not None:
        sns.lineplot(data=plot_df, x="layer", y="mean_offdiag_auc", hue="contrast_name", style="residualization", ax=ax, linewidth=2.2, marker=None)
    else:
        for name, group in plot_df.groupby(["contrast_name", "residualization"], sort=True):
            ax.plot(group["layer"], group["mean_offdiag_auc"], label=str(name), linewidth=2.2)
        ax.legend(frameon=False)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Mean off-diagonal family-to-family AUC by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean off-diagonal AUC")
    ax.grid(alpha=0.2)
    add_outside_legend(ax, max_items=80)
    save_fig(fig, output_dir / "figures" / "family_generalization" / "mean_offdiag_auc_by_layer")


def compute_direction_stability(
    direction_records: dict[tuple[int, str, str], DirectionRecord],
    layers: list[int],
    residualizations: list[str],
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    contrast_lookup = contrasts.set_index("contrast_name")["axis"].to_dict()
    reference_layer = 24 if 24 in layers else layers[-1]
    for residualization in residualizations:
        for contrast_name in contrasts["contrast_name"]:
            ref_key = (reference_layer, residualization, contrast_name)
            reference_direction = direction_records.get(ref_key)
            for i, layer in enumerate(layers):
                key = (layer, residualization, contrast_name)
                if key not in direction_records:
                    continue
                next_layer = layers[i + 1] if i + 1 < len(layers) else np.nan
                next_key = (next_layer, residualization, contrast_name)
                adjacent = float(np.dot(direction_records[key].direction, direction_records[next_key].direction)) if next_key in direction_records else float("nan")
                reference = float(np.dot(direction_records[key].direction, reference_direction.direction)) if reference_direction is not None else float("nan")
                rows.append({
                    "residualization": residualization,
                    "contrast_name": contrast_name,
                    "axis": contrast_lookup.get(contrast_name, ""),
                    "layer": layer,
                    "next_layer": next_layer,
                    "adjacent_cosine": adjacent,
                    "reference_layer": reference_layer,
                    "reference_cosine": reference,
                })
    return pd.DataFrame(rows).reindex(columns=STABILITY_COLUMNS)


def plot_direction_stability(stability_path: Path, output_dir: Path, residualizations: list[str]) -> None:
    if not stability_path.exists():
        return
    df = pd.read_csv(stability_path)
    if df.empty:
        return
    for residualization in residualizations:
        resid_df = df[df["residualization"].eq(residualization)]
        for y_col, filename, ylabel in [
            ("adjacent_cosine", "adjacent_layer_direction_stability", "Adjacent-layer cosine"),
            ("reference_cosine", "reference_layer_direction_stability", "Reference-layer cosine"),
        ]:
            fig, ax = plt.subplots(figsize=(16, 7))
            if sns is not None:
                sns.lineplot(data=resid_df, x="layer", y=y_col, hue="contrast_name", ax=ax, linewidth=2, marker=None)
            else:
                for name, group in resid_df.groupby("contrast_name", sort=True):
                    ax.plot(group["layer"], group[y_col], label=name, linewidth=2)
                ax.legend(frameon=False)
            ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_ylim(-1.02, 1.02)
            ax.set_title(f"{ylabel} for contrast directions ({residualization})")
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            add_outside_legend(ax)
            save_fig(fig, output_dir / "figures" / "direction_stability" / residualization / filename)
    key = df[df["contrast_name"].isin(KEY_CONTRASTS)].copy()
    fig, ax = plt.subplots(figsize=(16, 7))
    if sns is not None:
        sns.lineplot(data=key, x="layer", y="reference_cosine", hue="contrast_name", style="residualization", ax=ax, linewidth=2.2, marker=None)
    else:
        for name, group in key.groupby(["contrast_name", "residualization"], sort=True):
            ax.plot(group["layer"], group["reference_cosine"], label=str(name), linewidth=2.2)
        ax.legend(frameon=False)
    ax.set_ylim(-1.02, 1.02)
    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title("Key contrast direction stability relative to reference layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Reference-layer cosine")
    ax.grid(alpha=0.2)
    add_outside_legend(ax, max_items=80)
    save_fig(fig, output_dir / "figures" / "direction_stability" / "key_contrasts_direction_stability")


def plane_dataframe(
    x: np.ndarray,
    metadata: pd.DataFrame,
    axis: str,
    specs: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, str, str] | None:
    if len(specs) < 2:
        return None
    d1, global_mean, _ = compute_contrast_direction(x, metadata, specs[0][1], specs[0][2])
    d2, _, _ = compute_contrast_direction(x, metadata, specs[1][1], specs[1][2])
    if d1 is None or d2 is None:
        return None
    d2_orth = d2 - np.dot(d2, d1) * d1
    d2_orth = normalize(d2_orth)
    if d2_orth is None:
        return None
    axis_mask = metadata["axis"].eq(axis)
    centered = x - global_mean
    plane = metadata.loc[axis_mask, ["identity_id", "canonical_label"]].copy()
    plane["z1"] = centered[axis_mask.to_numpy()] @ d1
    plane["z2"] = centered[axis_mask.to_numpy()] @ d2_orth
    return plane, specs[0][0], specs[1][0]


def scatter_plane(ax: plt.Axes, plane_df: pd.DataFrame, xlabel: str, ylabel: str, title: str, max_points: int, seed: int) -> None:
    plane_df = stratified_sample_for_plot(plane_df, "identity_id", max_points, seed)
    labels = sorted(plane_df["canonical_label"].astype(str).unique())
    colors = color_map(labels)
    markers = marker_map(labels)
    for label, group in plane_df.groupby("canonical_label", sort=True):
        label = str(label)
        ax.scatter(group["z1"], group["z2"], color=colors[label], marker=markers[label], s=10, alpha=0.28, linewidths=0, label=label)
    centroids = plane_df.groupby("canonical_label", sort=True)[["z1", "z2"]].mean().reset_index()
    for _, row in centroids.iterrows():
        label = str(row["canonical_label"])
        ax.scatter(row["z1"], row["z2"], color=colors[label], marker=markers[label], s=80, edgecolors="black", linewidths=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.15)


def find_variance_decomposition(output_dir: Path) -> Path | None:
    candidates = [
        output_dir.parent / "diagnostics" / "variance_decomposition.csv",
        output_dir.parent / "diagnostics" / "metrics" / "variance_decomposition.csv",
        output_dir.parent / "identity_geometry_diagnostics" / "variance_decomposition.csv",
        output_dir.parent / "identity_geometry_diagnostics" / "metrics" / "variance_decomposition.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(path for path in output_dir.parent.glob("**/variance_decomposition.csv") if output_dir not in path.parents)
    if matches:
        return matches[0]
    return None


def plot_variance_panel(ax: plt.Axes, output_dir: Path) -> None:
    path = find_variance_decomposition(output_dir)
    if path is None:
        ax.text(0.5, 0.5, "Variance decomposition\nnot found", ha="center", va="center")
        ax.set_title("A. Variance Decomposition")
        ax.axis("off")
        print("Warning: diagnostics variance_decomposition.csv not found; skipping paper panel A.")
        return
    df = pd.read_csv(path)
    value_col = next((col for col in ["eta_squared", "eta2", "variance_explained"] if col in df.columns), None)
    factor_col = next((col for col in ["factor", "variable", "grouping", "term"] if col in df.columns), None)
    if value_col is None or factor_col is None or "layer" not in df.columns:
        ax.text(0.5, 0.5, "Variance file found\nbut columns are not recognized", ha="center", va="center")
        ax.set_title("A. Variance Decomposition")
        ax.axis("off")
        print(f"Warning: unrecognized variance decomposition columns in {path}.")
        return
    keep = df[df[factor_col].astype(str).isin(["identity_id", "axis", "family", "template_id", "required_form"])].copy()
    if keep.empty:
        keep = df.copy()
    if sns is not None:
        sns.lineplot(data=keep, x="layer", y=value_col, hue=factor_col, ax=ax, linewidth=2, marker=None)
    else:
        for name, group in keep.groupby(factor_col, sort=True):
            ax.plot(group["layer"], group[value_col], label=str(name), linewidth=2)
        ax.legend(frameon=False)
    ax.set_title("A. Variance Decomposition")
    ax.set_xlabel("Layer")
    ax.set_ylabel(value_col)
    add_outside_legend(ax, max_items=10)


def create_paper_panel(
    output_dir: Path,
    activation_dir: Path,
    metadata: pd.DataFrame,
    metrics_path: Path,
    centroid_path: Path,
    main_residualization: str,
    max_points: int,
    seed: int,
) -> None:
    layer = 24
    if not (activation_dir / "layer_24.npy").exists():
        print("Warning: layer_24.npy not available; skipping paper-ready summary panel.")
        return
    fig, axs = plt.subplots(2, 3, figsize=(22, 13))
    plot_variance_panel(axs[0, 0], output_dir)

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        key = metrics[metrics["contrast_name"].isin(KEY_CONTRASTS) & metrics["residualization"].isin(["raw", "family_residualized", "template_residualized"])].copy()
        if sns is not None and not key.empty:
            sns.lineplot(data=key, x="layer", y="auc", hue="contrast_name", style="residualization", ax=axs[0, 1], linewidth=2, marker=None)
        elif not key.empty:
            for name, group in key.groupby(["contrast_name", "residualization"], sort=True):
                axs[0, 1].plot(group["layer"], group["auc"], label=str(name), linewidth=2)
        axs[0, 1].axhline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.6)
        axs[0, 1].set_ylim(0.45, 1.02)
        axs[0, 1].set_title("B. Key Contrast AUC")
        axs[0, 1].set_xlabel("Layer")
        axs[0, 1].set_ylabel("AUC")
        add_outside_legend(axs[0, 1], max_items=10)

    x24 = residualize(load_layer(activation_dir, layer), metadata, main_residualization)
    for ax, axis, label in [(axs[0, 2], "sexual_orientation", "C"), (axs[1, 0], "gender_identity", "D")]:
        plane = plane_dataframe(x24, metadata, axis, PLANE_SPECS[axis])
        if plane is None:
            ax.text(0.5, 0.5, f"{axis} plane unavailable", ha="center", va="center")
            ax.axis("off")
        else:
            plane_df, xlabel, ylabel = plane
            scatter_plane(ax, plane_df, xlabel, ylabel, f"{label}. {axis.replace('_', ' ').title()} Plane", max_points, seed)

    if centroid_path.exists():
        centroid = pd.read_csv(centroid_path)
        rows = centroid[
            centroid["contrast_name"].eq("sexuality_gay_vs_sexuality_straight")
            & centroid["layer"].eq(layer)
            & centroid["residualization"].eq(main_residualization)
        ].sort_values("mean_projection")
        if not rows.empty:
            y = np.arange(len(rows))
            axs[1, 1].errorbar(rows["mean_projection"], y, xerr=[rows["mean_projection"] - rows["ci95_low"], rows["ci95_high"] - rows["mean_projection"]], fmt="o", color="#0072B2", ecolor="#0072B2", capsize=3)
            axs[1, 1].set_yticks(y, rows["canonical_label"])
            axs[1, 1].set_title("E. Sexual Orientation Centroid Ordering")
            axs[1, 1].set_xlabel("Projection on gay - straight")
        else:
            axs[1, 1].text(0.5, 0.5, "Centroid ordering\nnot available", ha="center", va="center")
            axs[1, 1].axis("off")

    matrix_path = output_dir / "metrics" / "direction_cosines" / f"direction_cosine_matrix_layer_{layer:02d}_{main_residualization}.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path, index_col=0)
        matrix = order_similarity_matrix(matrix)
        if sns is not None:
            sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, ax=axs[1, 2], cbar=False)
        else:
            axs[1, 2].imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        axs[1, 2].set_title("F. Direction Cosines")
    else:
        axs[1, 2].text(0.5, 0.5, "Direction cosine\nmatrix not available", ha="center", va="center")
        axs[1, 2].axis("off")

    fig.suptitle(f"Identity Geometry Summary, Layer 24 ({main_residualization})", fontsize=18, y=1.01)
    save_fig(fig, output_dir / "figures" / "paper_panels" / "identity_geometry_summary_layer24")


def main() -> None:
    args = parse_args()
    start_all = time.perf_counter()
    metadata = load_metadata(args.activation_dir)
    layers = parse_int_list(args.layers) if args.layers else discover_layers(args.activation_dir)
    selected_layers = parse_int_list(args.selected_layers_for_planes)
    selected_layers = [layer for layer in selected_layers if layer in layers]
    comparison_layer = 24 if 24 in layers else (selected_layers[0] if selected_layers else layers[-1])
    residualizations = parse_str_list(args.residualizations)
    invalid_residualizations = [item for item in residualizations if item not in RESIDUALIZATION_GROUPS]
    if invalid_residualizations:
        raise ValueError(f"Unknown residualizations: {invalid_residualizations}")
    if args.main_residualization not in residualizations:
        print(f"Adding main_residualization '{args.main_residualization}' to residualizations.")
        residualizations.append(args.main_residualization)
    contrasts = load_contrasts(args.contrasts_csv, metadata)
    prepare_output(args.output_dir, args.overwrite)
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "activation_dir": str(args.activation_dir),
        "output_dir": str(args.output_dir),
        "layers": layers,
        "selected_layers_for_planes": selected_layers,
        "comparison_layer": comparison_layer,
        "residualizations": residualizations,
        "main_residualization": args.main_residualization,
        "contrasts_csv": str(args.contrasts_csv) if args.contrasts_csv else None,
        "max_points_per_plot": args.max_points_per_plot,
        "random_seed": args.random_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")

    metrics_path = args.output_dir / "metrics" / "layerwise_contrast_metrics.csv"
    cosine_long_path = args.output_dir / "metrics" / "direction_cosines" / "direction_cosines_long.csv"
    centroid_path = args.output_dir / "metrics" / "centroid_ordering.csv"
    family_path = args.output_dir / "metrics" / "family_to_family_generalization.csv"
    direction_records: dict[tuple[int, str, str], DirectionRecord] = {}

    for layer in tqdm(layers, desc="layers"):
        layer_start = time.perf_counter()
        x_raw = load_layer(args.activation_dir, layer)
        if x_raw.shape[0] != len(metadata):
            raise ValueError(f"layer_{layer:02d}.npy has {x_raw.shape[0]} rows but metadata has {len(metadata)} rows.")
        print(f"\nLayer {layer:02d}: loaded activations {x_raw.shape} in {elapsed(layer_start)}")
        comparison_cache: dict[str, dict[str, tuple[pd.DataFrame, dict[str, float | int | bool]]]] = {
            name: {} for name in RESIDUALIZATION_COMPARISON_CONTRASTS
        }
        for residualization in tqdm(residualizations, desc=f"layer {layer:02d} residualizations", leave=False):
            resid_start = time.perf_counter()
            x = residualize(x_raw, metadata, residualization)
            print(f"Layer {layer:02d} {residualization}: residualized in {elapsed(resid_start)}")
            layer_rows: list[dict[str, object]] = []
            centroid_rows_all: list[dict[str, object]] = []
            family_rows_all: list[dict[str, object]] = []
            contrast_iter = tqdm(list(contrasts.itertuples(index=False)), desc=f"layer {layer:02d} {residualization} contrasts", leave=False)
            for contrast in contrast_iter:
                direction, global_mean, sign_flipped = compute_contrast_direction(x, metadata, contrast.identity_a, contrast.identity_b)
                if direction is None:
                    continue
                direction_records[(layer, residualization, contrast.contrast_name)] = DirectionRecord(direction=direction, global_mean=global_mean, sign_flipped=sign_flipped)
                scores = project_scores(x, direction, global_mean)
                metric_row = {
                    "layer": layer,
                    "residualization": residualization,
                    "contrast_name": contrast.contrast_name,
                    "axis": contrast.axis,
                    "identity_a": contrast.identity_a,
                    "identity_b": contrast.identity_b,
                    **endpoint_metrics(scores, metadata, contrast.identity_a, contrast.identity_b),
                    "sign_flipped": sign_flipped,
                }
                layer_rows.append(metric_row)
                if layer == comparison_layer and contrast.contrast_name in comparison_cache:
                    comparison = projection_distribution_rows(x, metadata, contrast, residualization, layer)
                    if comparison is not None:
                        comparison_cache[contrast.contrast_name][residualization] = comparison
                if layer in selected_layers and contrast.contrast_name in CENTROID_ORDERING_CONTRASTS:
                    rows, prompt_df = centroid_ordering_rows(x, metadata, contrast, layer, residualization, direction_records[(layer, residualization, contrast.contrast_name)])
                    centroid_rows_all.extend(rows)
                    plot_centroid_ordering(rows, prompt_df, contrast.contrast_name, residualization, layer, args.output_dir, args.max_points_per_plot, args.random_seed)
                family_rows_all.extend(family_to_family_rows(x, metadata, contrast, layer, residualization))
            append_rows(metrics_path, layer_rows, LAYERWISE_COLUMNS)
            append_rows(centroid_path, centroid_rows_all, CENTROID_COLUMNS)
            append_rows(family_path, family_rows_all, FAMILY_GENERALIZATION_COLUMNS)
            cosine_rows = direction_cosines(direction_records, contrasts, layer, residualization, args.output_dir)
            append_rows(cosine_long_path, cosine_rows, COSINE_LONG_COLUMNS)
            print(f"Layer {layer:02d} {residualization}: metrics/figures saved in {elapsed(resid_start)}")
        if layer == comparison_layer:
            for contrast_name, data in comparison_cache.items():
                plot_residualization_comparison(contrast_name, data, args.output_dir, layer, args.max_points_per_plot, args.random_seed)
        print(f"Layer {layer:02d}: complete in {elapsed(layer_start)}")

    stability = compute_direction_stability(direction_records, layers, residualizations, contrasts)
    write_csv(args.output_dir / "metrics" / "direction_stability.csv", stability, STABILITY_COLUMNS)
    plot_layerwise_figures(metrics_path, args.output_dir, residualizations)
    plot_direction_cosine_summaries(cosine_long_path, args.output_dir)
    plot_family_generalization(family_path, args.output_dir, comparison_layer, residualizations)
    plot_direction_stability(args.output_dir / "metrics" / "direction_stability.csv", args.output_dir, residualizations)
    create_paper_panel(args.output_dir, args.activation_dir, metadata, metrics_path, centroid_path, args.main_residualization, args.max_points_per_plot, args.random_seed)
    print(f"\nDirectional follow-up analyses complete in {elapsed(start_all)}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
