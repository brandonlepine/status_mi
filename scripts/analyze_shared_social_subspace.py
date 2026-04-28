#!/usr/bin/env python3
"""Decompose identity contrast directions into shared and residual components.

For each layer and residualization, this script computes a set of normalized
identity contrast directions, runs SVD on those directions, treats the top-k
right singular vectors as a shared social subspace, and evaluates how well the
shared and contrast-specific residual components preserve endpoint separation.
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
    "identity_prompts_final_token/shared_subspace_decomposition"
)
DEFAULT_LAYERS = "8,16,24,32"
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
SELECTED_CROSS_AXIS_ORDERINGS = [
    ("appearance_poorly_dressed_vs_appearance_well_dressed", "socioeconomic_status"),
    ("ses_low_income_vs_ses_rich", "race_ethnicity"),
    ("appearance_poorly_dressed_vs_appearance_well_dressed", "gender_identity"),
    ("ses_lower_class_vs_ses_upper_class", "physical_appearance"),
]
SPECTRUM_COLUMNS = [
    "layer",
    "residualization",
    "component",
    "singular_value",
    "explained_variance_ratio",
    "cumulative_explained_variance",
]
DECOMPOSITION_COLUMNS = [
    "layer",
    "residualization",
    "contrast_name",
    "axis",
    "identity_a",
    "identity_b",
    "k",
    "component_type",
    "component_norm",
    "fraction_norm",
    "cosine_with_full",
    "auc",
    "cohens_d",
    "accuracy_midpoint",
    "mean_a",
    "mean_b",
    "sd_a",
    "sd_b",
    "n_a",
    "n_b",
]
AXIS_SUMMARY_COLUMNS = [
    "layer",
    "residualization",
    "axis",
    "k",
    "mean_fraction_shared",
    "median_fraction_shared",
    "mean_full_auc",
    "mean_shared_auc",
    "mean_residual_auc",
    "mean_full_d",
    "mean_shared_d",
    "mean_residual_d",
    "n_contrasts",
]
PC_RANKING_COLUMNS = [
    "layer",
    "residualization",
    "pc",
    "identity_id",
    "canonical_label",
    "axis",
    "projection_score",
    "rank_descending",
]
PC_TOP_BOTTOM_COLUMNS = [
    "layer",
    "residualization",
    "pc",
    "side",
    "rank",
    "identity_id",
    "canonical_label",
    "axis",
    "projection_score",
]
CONTRAST_LOADING_COLUMNS = [
    "layer",
    "residualization",
    "contrast_name",
    "axis",
    "pc",
    "loading",
    "abs_loading",
]
CROSS_AXIS_SUMMARY_COLUMNS = [
    "layer",
    "residualization",
    "contrast_name",
    "contrast_axis",
    "projected_axis",
    "mean_projection",
    "sd_projection",
    "range_projection",
    "max_identity",
    "max_label",
    "max_score",
    "min_identity",
    "min_label",
    "min_score",
]
CROSS_AXIS_LONG_COLUMNS = [
    "layer",
    "residualization",
    "contrast_name",
    "contrast_axis",
    "projected_axis",
    "identity_id",
    "canonical_label",
    "projection_score",
]
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]


@dataclass
class DirectionRecord:
    contrast_name: str
    axis: str
    identity_a: str
    identity_b: str
    direction: np.ndarray
    global_mean: np.ndarray
    sign_flipped: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze shared vs contrast-specific identity direction subspaces.")
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--residualizations", default=",".join(DEFAULT_RESIDUALIZATIONS))
    parser.add_argument("--k_values", default="1,2,3,5,10")
    parser.add_argument("--contrasts_csv", type=Path, default=None)
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


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists and is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    for subdir in [
        "metrics",
        "figures/spectrum",
        "figures/decomposition",
        "figures/axis_summary",
        "figures/pc_interpretation",
        "figures/pc_loadings",
        "figures/cross_axis/selected_orderings",
        "figures/paper_panels",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


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
    pd.DataFrame(rows).reindex(columns=columns).to_csv(path, mode="a", header=not path.exists(), index=False)


def write_csv(path: Path, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = df.reindex(columns=columns)
    df.to_csv(path, index=False)


def add_outside_legend(ax: plt.Axes, max_items: int = 50) -> None:
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
        raise ValueError(f"{path} should be 2D, found shape {x.shape}")
    return np.asarray(x, dtype=np.float32)


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


def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    norm = np.linalg.norm(vec)
    if norm <= eps or not np.isfinite(norm):
        return None
    return vec / norm


def compute_direction(x: np.ndarray, metadata: pd.DataFrame, identity_a: str, identity_b: str) -> tuple[np.ndarray | None, np.ndarray, bool]:
    global_mean = x.mean(axis=0, keepdims=True)
    centered = x - global_mean
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
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
    return direction.astype(np.float32), global_mean.astype(np.float32), sign_flipped


def project_onto_subspace(direction: np.ndarray, basis: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    shared_raw = basis.T @ (basis @ direction)
    residual_raw = direction - shared_raw
    shared_norm = float(np.linalg.norm(shared_raw))
    residual_norm = float(np.linalg.norm(residual_raw))
    shared_unit = shared_raw / shared_norm if shared_norm > eps and np.isfinite(shared_norm) else None
    residual_unit = residual_raw / residual_norm if residual_norm > eps and np.isfinite(residual_norm) else None
    return shared_unit, residual_unit, shared_norm, residual_norm


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if len(scores_a) < 2 or len(scores_b) < 2:
        return float("nan")
    pooled = (((len(scores_a) - 1) * np.var(scores_a, ddof=1)) + ((len(scores_b) - 1) * np.var(scores_b, ddof=1))) / (len(scores_a) + len(scores_b) - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((scores_a.mean() - scores_b.mean()) / np.sqrt(pooled))


def evaluate_component(
    x: np.ndarray,
    metadata: pd.DataFrame,
    identity_a: str,
    identity_b: str,
    component: np.ndarray | None,
    global_mean: np.ndarray,
) -> dict[str, float | int]:
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    if component is None or n_a == 0 or n_b == 0:
        return {
            "auc": float("nan"),
            "cohens_d": float("nan"),
            "accuracy_midpoint": float("nan"),
            "mean_a": float("nan"),
            "mean_b": float("nan"),
            "sd_a": float("nan"),
            "sd_b": float("nan"),
            "n_a": n_a,
            "n_b": n_b,
        }
    scores = (x - global_mean) @ component
    scores_a = scores[mask_a]
    scores_b = scores[mask_b]
    pair_scores = np.concatenate([scores_a, scores_b])
    labels = np.concatenate([np.ones(len(scores_a)), np.zeros(len(scores_b))])
    auc = float(roc_auc_score(labels, pair_scores)) if len(np.unique(labels)) == 2 else float("nan")
    mean_a = float(scores_a.mean())
    mean_b = float(scores_b.mean())
    midpoint = (mean_a + mean_b) / 2
    accuracy = float(np.mean(np.concatenate([scores_a >= midpoint, scores_b < midpoint])))
    return {
        "auc": auc,
        "cohens_d": cohens_d(scores_a, scores_b),
        "accuracy_midpoint": accuracy,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "sd_a": float(scores_a.std(ddof=1)) if len(scores_a) > 1 else float("nan"),
        "sd_b": float(scores_b.std(ddof=1)) if len(scores_b) > 1 else float("nan"),
        "n_a": n_a,
        "n_b": n_b,
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


def compute_contrast_directions(x: np.ndarray, metadata: pd.DataFrame, contrasts: pd.DataFrame) -> list[DirectionRecord]:
    records: list[DirectionRecord] = []
    for contrast in tqdm(list(contrasts.itertuples(index=False)), desc="directions", leave=False):
        direction, global_mean, sign_flipped = compute_direction(x, metadata, contrast.identity_a, contrast.identity_b)
        if direction is None:
            continue
        records.append(DirectionRecord(
            contrast_name=contrast.contrast_name,
            axis=contrast.axis,
            identity_a=contrast.identity_a,
            identity_b=contrast.identity_b,
            direction=direction,
            global_mean=global_mean,
            sign_flipped=sign_flipped,
        ))
    return records


def run_svd(records: list[DirectionRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(records) < 2:
        raise ValueError("Need at least two valid contrast directions for shared subspace SVD.")
    direction_matrix = np.vstack([record.direction for record in records]).astype(np.float32)
    return np.linalg.svd(direction_matrix, full_matrices=False)


def spectrum_rows(singular_values: np.ndarray, layer: int, residualization: str) -> list[dict[str, object]]:
    denom = float(np.sum(singular_values ** 2))
    rows = []
    cumulative = 0.0
    for idx, singular_value in enumerate(singular_values, start=1):
        ratio = float((singular_value ** 2) / denom) if denom > 0 else float("nan")
        cumulative += ratio if np.isfinite(ratio) else 0.0
        rows.append({
            "layer": layer,
            "residualization": residualization,
            "component": idx,
            "singular_value": float(singular_value),
            "explained_variance_ratio": ratio,
            "cumulative_explained_variance": cumulative,
        })
    return rows


def decomposition_rows(
    x: np.ndarray,
    records: list[DirectionRecord],
    basis: np.ndarray,
    metadata: pd.DataFrame,
    k_values: list[int],
    layer: int,
    residualization: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_k = basis.shape[0]
    for record in tqdm(records, desc="decompose", leave=False):
        full_metrics = evaluate_component(x, metadata, record.identity_a, record.identity_b, record.direction, record.global_mean)
        for k in k_values:
            k_eff = min(k, max_k)
            shared, residual, shared_norm, residual_norm = project_onto_subspace(record.direction, basis[:k_eff])
            components = [
                ("full", record.direction, 1.0, 1.0, 1.0, full_metrics),
                ("shared", shared, shared_norm, shared_norm ** 2, float(np.dot(record.direction, shared)) if shared is not None else float("nan"), None),
                ("residual", residual, residual_norm, residual_norm ** 2, float(np.dot(record.direction, residual)) if residual is not None else float("nan"), None),
            ]
            for component_type, component, component_norm, fraction_norm, cosine, metrics in components:
                if metrics is None:
                    metrics = evaluate_component(x, metadata, record.identity_a, record.identity_b, component, record.global_mean)
                rows.append({
                    "layer": layer,
                    "residualization": residualization,
                    "contrast_name": record.contrast_name,
                    "axis": record.axis,
                    "identity_a": record.identity_a,
                    "identity_b": record.identity_b,
                    "k": k,
                    "component_type": component_type,
                    "component_norm": component_norm,
                    "fraction_norm": fraction_norm,
                    "cosine_with_full": cosine,
                    **metrics,
                })
    return rows


def identity_centroids(x: np.ndarray, metadata: pd.DataFrame, global_mean: np.ndarray | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    center = global_mean if global_mean is not None else x.mean(axis=0, keepdims=True)
    centered = x - center
    for (identity_id, canonical_label, axis), group in metadata.groupby(["identity_id", "canonical_label", "axis"], sort=True):
        rows.append({
            "identity_id": identity_id,
            "canonical_label": canonical_label,
            "axis": axis,
            "centroid": centered[group.index.to_numpy()].mean(axis=0),
        })
    return pd.DataFrame(rows)


def pc_interpretation_rows(
    x: np.ndarray,
    metadata: pd.DataFrame,
    basis: np.ndarray,
    layer: int,
    residualization: str,
    n_pcs: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    top_bottom: list[dict[str, object]] = []
    centroids = identity_centroids(x, metadata)
    centroid_matrix = np.vstack(centroids["centroid"].to_numpy())
    for pc_idx in range(min(n_pcs, basis.shape[0])):
        scores = centroid_matrix @ basis[pc_idx]
        pc_df = centroids.drop(columns=["centroid"]).copy()
        pc_df["projection_score"] = scores
        pc_df = pc_df.sort_values("projection_score", ascending=False).reset_index(drop=True)
        pc_df["rank_descending"] = np.arange(1, len(pc_df) + 1)
        for row in pc_df.itertuples(index=False):
            rows.append({
                "layer": layer,
                "residualization": residualization,
                "pc": pc_idx + 1,
                "identity_id": row.identity_id,
                "canonical_label": row.canonical_label,
                "axis": row.axis,
                "projection_score": float(row.projection_score),
                "rank_descending": int(row.rank_descending),
            })
        for side, side_df in [("top", pc_df.head(20)), ("bottom", pc_df.tail(20).sort_values("projection_score", ascending=True))]:
            for rank, row in enumerate(side_df.itertuples(index=False), start=1):
                top_bottom.append({
                    "layer": layer,
                    "residualization": residualization,
                    "pc": pc_idx + 1,
                    "side": side,
                    "rank": rank,
                    "identity_id": row.identity_id,
                    "canonical_label": row.canonical_label,
                    "axis": row.axis,
                    "projection_score": float(row.projection_score),
                })
    return rows, top_bottom


def contrast_loading_rows(records: list[DirectionRecord], basis: np.ndarray, layer: int, residualization: str, n_pcs: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        for pc_idx in range(min(n_pcs, basis.shape[0])):
            loading = float(np.dot(record.direction, basis[pc_idx]))
            rows.append({
                "layer": layer,
                "residualization": residualization,
                "contrast_name": record.contrast_name,
                "axis": record.axis,
                "pc": pc_idx + 1,
                "loading": loading,
                "abs_loading": abs(loading),
            })
    return rows


def cross_axis_projection_rows(
    x: np.ndarray,
    metadata: pd.DataFrame,
    records: list[DirectionRecord],
    layer: int,
    residualization: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    global_mean = x.mean(axis=0, keepdims=True)
    base_centroids = identity_centroids(x, metadata, global_mean)
    centroid_matrix = np.vstack(base_centroids["centroid"].to_numpy())
    for record in records:
        centroids = base_centroids.drop(columns=["centroid"]).copy()
        centroids["projection_score"] = centroid_matrix @ record.direction
        for row in centroids.itertuples(index=False):
            long_rows.append({
                "layer": layer,
                "residualization": residualization,
                "contrast_name": record.contrast_name,
                "contrast_axis": record.axis,
                "projected_axis": row.axis,
                "identity_id": row.identity_id,
                "canonical_label": row.canonical_label,
                "projection_score": float(row.projection_score),
            })
        for projected_axis, axis_df in centroids.groupby("axis", sort=True):
            max_row = axis_df.loc[axis_df["projection_score"].idxmax()]
            min_row = axis_df.loc[axis_df["projection_score"].idxmin()]
            summary_rows.append({
                "layer": layer,
                "residualization": residualization,
                "contrast_name": record.contrast_name,
                "contrast_axis": record.axis,
                "projected_axis": projected_axis,
                "mean_projection": float(axis_df["projection_score"].mean()),
                "sd_projection": float(axis_df["projection_score"].std(ddof=1)) if len(axis_df) > 1 else float("nan"),
                "range_projection": float(axis_df["projection_score"].max() - axis_df["projection_score"].min()),
                "max_identity": max_row["identity_id"],
                "max_label": max_row["canonical_label"],
                "max_score": float(max_row["projection_score"]),
                "min_identity": min_row["identity_id"],
                "min_label": min_row["canonical_label"],
                "min_score": float(min_row["projection_score"]),
            })
    return summary_rows, long_rows


def aggregate_axis_sharedness(decomp_path: Path, output_dir: Path) -> pd.DataFrame:
    if not decomp_path.exists():
        return pd.DataFrame(columns=AXIS_SUMMARY_COLUMNS)
    df = pd.read_csv(decomp_path)
    if df.empty:
        return pd.DataFrame(columns=AXIS_SUMMARY_COLUMNS)
    shared = df[df["component_type"].eq("shared")][["layer", "residualization", "axis", "contrast_name", "k", "fraction_norm", "auc", "cohens_d"]].rename(columns={
        "fraction_norm": "fraction_shared",
        "auc": "shared_auc",
        "cohens_d": "shared_d",
    })
    residual = df[df["component_type"].eq("residual")][["layer", "residualization", "axis", "contrast_name", "k", "auc", "cohens_d"]].rename(columns={
        "auc": "residual_auc",
        "cohens_d": "residual_d",
    })
    full = df[df["component_type"].eq("full")][["layer", "residualization", "axis", "contrast_name", "k", "auc", "cohens_d"]].rename(columns={
        "auc": "full_auc",
        "cohens_d": "full_d",
    })
    merged = shared.merge(residual, on=["layer", "residualization", "axis", "contrast_name", "k"], how="left").merge(full, on=["layer", "residualization", "axis", "contrast_name", "k"], how="left")
    summary = merged.groupby(["layer", "residualization", "axis", "k"], sort=True).agg(
        mean_fraction_shared=("fraction_shared", "mean"),
        median_fraction_shared=("fraction_shared", "median"),
        mean_full_auc=("full_auc", "mean"),
        mean_shared_auc=("shared_auc", "mean"),
        mean_residual_auc=("residual_auc", "mean"),
        mean_full_d=("full_d", "mean"),
        mean_shared_d=("shared_d", "mean"),
        mean_residual_d=("residual_d", "mean"),
        n_contrasts=("contrast_name", "nunique"),
    ).reset_index()
    write_csv(output_dir / "metrics" / "axis_sharedness_summary.csv", summary, AXIS_SUMMARY_COLUMNS)
    return summary


def order_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if leaves_list is None or linkage is None or squareform is None or min(df.shape) < 3:
        return df
    try:
        values = df.fillna(0).to_numpy()
        corr = np.corrcoef(values)
        distance = 1 - np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(distance, 0)
        order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
        return df.iloc[order]
    except Exception as exc:  # pragma: no cover
        print(f"Warning: matrix clustering failed: {exc}")
        return df


def plot_spectrum(output_dir: Path) -> None:
    path = output_dir / "metrics" / "shared_subspace_spectrum.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(15, 7))
    if sns is not None:
        sns.lineplot(data=df, x="component", y="singular_value", hue="residualization", style="layer", ax=ax, linewidth=2, marker="o")
    else:
        for name, group in df.groupby(["residualization", "layer"], sort=True):
            ax.plot(group["component"], group["singular_value"], label=str(name), marker="o")
        ax.legend(frameon=False)
    ax.set_title("Shared-subspace singular value spectrum")
    ax.set_xlabel("Shared PC")
    ax.set_ylabel("Singular value")
    add_outside_legend(ax, max_items=80)
    save_fig(fig, output_dir / "figures" / "spectrum" / "shared_subspace_singular_values")

    fig, ax = plt.subplots(figsize=(15, 7))
    if sns is not None:
        sns.lineplot(data=df, x="component", y="cumulative_explained_variance", hue="residualization", style="layer", ax=ax, linewidth=2, marker="o")
    else:
        for name, group in df.groupby(["residualization", "layer"], sort=True):
            ax.plot(group["component"], group["cumulative_explained_variance"], label=str(name), marker="o")
        ax.legend(frameon=False)
    ax.set_ylim(0, 1.02)
    ax.set_title("Cumulative variance of contrast directions explained by shared PCs")
    ax.set_xlabel("Number of shared PCs")
    ax.set_ylabel("Cumulative explained variance")
    add_outside_legend(ax, max_items=80)
    save_fig(fig, output_dir / "figures" / "spectrum" / "shared_subspace_cumulative_variance")


def plot_decomposition(output_dir: Path, main_layer: int = 24, main_residualization: str = "family_residualized") -> None:
    path = output_dir / "metrics" / "decomposition_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    max_k = int(df["k"].max())
    shared_all = df[df["component_type"].eq("shared")].copy()
    if not shared_all.empty:
        summary = shared_all.groupby(["layer", "residualization", "k"], sort=True)["fraction_norm"].mean().reset_index(name="mean_fraction_shared")
        fig, ax = plt.subplots(figsize=(15, 7))
        if sns is not None:
            sns.lineplot(data=summary, x="layer", y="mean_fraction_shared", hue="residualization", style="k", ax=ax, linewidth=2, marker="o")
        else:
            for name, group in summary.groupby(["residualization", "k"], sort=True):
                ax.plot(group["layer"], group["mean_fraction_shared"], marker="o", label=str(name))
            ax.legend(frameon=False)
        ax.set_ylim(0, 1.02)
        ax.set_title("Mean shared-subspace fraction across layers and residualizations")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean shared fraction of squared direction norm")
        add_outside_legend(ax, max_items=80)
        save_fig(fig, output_dir / "figures" / "decomposition" / "mean_fraction_shared_by_layer_and_residualization")

        key_layer_summary = shared_all[shared_all["k"].eq(max_k) & shared_all["contrast_name"].isin(KEY_CONTRASTS)].copy()
        if not key_layer_summary.empty:
            fig, ax = plt.subplots(figsize=(15, 7))
            if sns is not None:
                sns.lineplot(data=key_layer_summary, x="layer", y="fraction_norm", hue="contrast_name", style="residualization", ax=ax, linewidth=2, marker="o")
            else:
                for name, group in key_layer_summary.groupby(["contrast_name", "residualization"], sort=True):
                    ax.plot(group["layer"], group["fraction_norm"], marker="o", label=str(name))
                ax.legend(frameon=False)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"Key contrast shared-subspace fraction across layers, k={max_k}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Shared fraction of squared direction norm")
            add_outside_legend(ax, max_items=80)
            save_fig(fig, output_dir / "figures" / "decomposition" / f"key_contrast_fraction_shared_by_layer_k{max_k}")

    component_summary = df[df["k"].eq(max_k)].groupby(["layer", "residualization", "component_type"], sort=True)["auc"].mean().reset_index(name="mean_auc")
    if not component_summary.empty:
        fig, ax = plt.subplots(figsize=(15, 7))
        if sns is not None:
            sns.lineplot(data=component_summary, x="layer", y="mean_auc", hue="component_type", style="residualization", ax=ax, linewidth=2, marker="o")
        else:
            for name, group in component_summary.groupby(["component_type", "residualization"], sort=True):
                ax.plot(group["layer"], group["mean_auc"], marker="o", label=str(name))
            ax.legend(frameon=False)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_ylim(0.45, 1.02)
        ax.set_title(f"Mean full/shared/residual AUC across layers, k={max_k}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean AUC")
        add_outside_legend(ax, max_items=80)
        save_fig(fig, output_dir / "figures" / "decomposition" / f"component_auc_by_layer_k{max_k}")

    layer_df = df[df["layer"].eq(main_layer)].copy()
    shared = layer_df[layer_df["component_type"].eq("shared")].copy()
    if not shared.empty:
        fig, ax = plt.subplots(figsize=(16, 8))
        plot_df = shared[shared["contrast_name"].isin(KEY_CONTRASTS)].copy()
        if plot_df.empty:
            plot_df = shared
        if sns is not None:
            sns.lineplot(data=plot_df, x="k", y="fraction_norm", hue="contrast_name", style="residualization", ax=ax, linewidth=2, marker="o")
        else:
            for name, group in plot_df.groupby(["contrast_name", "residualization"], sort=True):
                ax.plot(group["k"], group["fraction_norm"], marker="o", label=str(name))
            ax.legend(frameon=False)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Fraction of direction norm in top-k shared subspace, layer {main_layer:02d}")
        ax.set_xlabel("k shared PCs")
        ax.set_ylabel("Shared fraction of squared direction norm")
        add_outside_legend(ax, max_items=80)
        save_fig(fig, output_dir / "figures" / "decomposition" / f"fraction_shared_by_k_layer{main_layer:02d}")

        heat_df = shared[shared["residualization"].eq(main_residualization)].pivot_table(index="contrast_name", columns="k", values="fraction_norm", aggfunc="mean")
        if not heat_df.empty:
            heat_df = order_matrix(heat_df)
            fig, ax = plt.subplots(figsize=(10, max(7, 0.4 * len(heat_df))))
            if sns is not None:
                sns.heatmap(heat_df, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax)
            else:
                im = ax.imshow(heat_df.to_numpy(), cmap="viridis", vmin=0, vmax=1)
                fig.colorbar(im, ax=ax)
                ax.set_xticks(range(len(heat_df.columns)), heat_df.columns)
                ax.set_yticks(range(len(heat_df.index)), heat_df.index)
            ax.set_title(f"Fraction shared by contrast, layer {main_layer:02d} ({main_residualization})")
            ax.set_xlabel("k shared PCs")
            ax.set_ylabel("Contrast")
            save_fig(fig, output_dir / "figures" / "decomposition" / f"fraction_shared_heatmap_layer{main_layer:02d}_{main_residualization}")

    for metric, ylabel, filename in [
        ("auc", "AUC", f"shared_vs_residual_auc_by_k_layer{main_layer:02d}"),
        ("cohens_d", "Cohen's d", f"shared_vs_residual_d_by_k_layer{main_layer:02d}"),
    ]:
        plot_df = layer_df[layer_df["contrast_name"].isin(KEY_CONTRASTS) & layer_df["component_type"].isin(["full", "shared", "residual"])].copy()
        if plot_df.empty:
            continue
        grid = sns.FacetGrid(plot_df, col="contrast_name", col_wrap=3, hue="component_type", height=4, aspect=1.45, sharey=False) if sns is not None else None
        if grid is not None:
            grid.map_dataframe(sns.lineplot, x="k", y=metric, linewidth=2, marker="o")
            grid.add_legend()
            grid.set_axis_labels("k shared PCs", ylabel)
            grid.fig.suptitle(f"Full vs shared vs residual {ylabel}, layer {main_layer:02d}", y=1.02)
            save_fig(grid.fig, output_dir / "figures" / "decomposition" / filename)
        else:
            fig, ax = plt.subplots(figsize=(16, 8))
            for name, group in plot_df.groupby(["contrast_name", "component_type"], sort=True):
                ax.plot(group["k"], group[metric], marker="o", label=str(name))
            ax.set_title(f"Full vs shared vs residual {ylabel}, layer {main_layer:02d}")
            ax.set_xlabel("k shared PCs")
            ax.set_ylabel(ylabel)
            add_outside_legend(ax, max_items=80)
            save_fig(fig, output_dir / "figures" / "decomposition" / filename)


def plot_axis_summary(output_dir: Path, main_layer: int = 24) -> None:
    path = output_dir / "metrics" / "axis_sharedness_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    layer_df = df[df["layer"].eq(main_layer)].copy()
    if layer_df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    if sns is not None:
        sns.lineplot(data=layer_df, x="k", y="mean_fraction_shared", hue="axis", style="residualization", ax=ax, linewidth=2, marker="o")
    else:
        for name, group in layer_df.groupby(["axis", "residualization"], sort=True):
            ax.plot(group["k"], group["mean_fraction_shared"], marker="o", label=str(name))
        ax.legend(frameon=False)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Mean fraction of contrast direction in shared subspace by axis, layer {main_layer:02d}")
    ax.set_xlabel("k shared PCs")
    ax.set_ylabel("Mean fraction shared")
    add_outside_legend(ax, max_items=80)
    save_fig(fig, output_dir / "figures" / "axis_summary" / f"axis_fraction_shared_layer{main_layer:02d}")

    auc_df = layer_df.melt(
        id_vars=["layer", "residualization", "axis", "k"],
        value_vars=["mean_full_auc", "mean_shared_auc", "mean_residual_auc"],
        var_name="component",
        value_name="mean_auc",
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    if sns is not None:
        sns.lineplot(data=auc_df, x="k", y="mean_auc", hue="axis", style="component", ax=ax, linewidth=2, marker="o")
    else:
        for name, group in auc_df.groupby(["axis", "component"], sort=True):
            ax.plot(group["k"], group["mean_auc"], marker="o", label=str(name))
        ax.legend(frameon=False)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_ylim(0.45, 1.02)
    ax.set_title(f"Shared vs residual AUC by axis, layer {main_layer:02d}")
    ax.set_xlabel("k shared PCs")
    ax.set_ylabel("Mean AUC")
    add_outside_legend(ax, max_items=80)
    save_fig(fig, output_dir / "figures" / "axis_summary" / f"axis_shared_vs_residual_auc_layer{main_layer:02d}")


def signed_bar_colors(values: pd.Series) -> list[str]:
    return ["#0072B2" if value >= 0 else "#D55E00" for value in values]


def plot_pc_interpretation(output_dir: Path, main_layer: int = 24, main_residualization: str = "family_residualized") -> None:
    path = output_dir / "metrics" / "shared_pc_top_bottom.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["layer"].eq(main_layer) & df["residualization"].eq(main_residualization) & df["pc"].isin([1, 2, 3])].copy()
    if df.empty:
        return
    for pc, pc_df in df.groupby("pc", sort=True):
        top = pc_df[pc_df["side"].eq("top")].head(15)
        bottom = pc_df[pc_df["side"].eq("bottom")].head(15)
        plot_df = pd.concat([bottom.sort_values("projection_score"), top.sort_values("projection_score")], ignore_index=True)
        fig, ax = plt.subplots(figsize=(11, 9))
        labels = plot_df["canonical_label"] + " (" + plot_df["axis"].str.replace("_", " ") + ")"
        ax.barh(labels, plot_df["projection_score"], color=signed_bar_colors(plot_df["projection_score"]))
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"Shared PC{pc}: top and bottom identity centroids\nLayer {main_layer:02d}, {main_residualization}")
        ax.set_xlabel("Projection score")
        save_fig(fig, output_dir / "figures" / "pc_interpretation" / f"layer{main_layer:02d}_{main_residualization}_PC{pc}_top_bottom")


def plot_pc_loadings(output_dir: Path, main_layer: int = 24, main_residualization: str = "family_residualized") -> None:
    path = output_dir / "metrics" / "contrast_pc_loadings.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["layer"].eq(main_layer) & df["residualization"].eq(main_residualization) & df["pc"].le(10)].copy()
    if df.empty:
        return
    pivot = df.pivot_table(index="contrast_name", columns="pc", values="loading", aggfunc="mean")
    if not pivot.empty:
        pivot = order_matrix(pivot)
        fig, ax = plt.subplots(figsize=(12, max(8, 0.42 * len(pivot))))
        if sns is not None:
            sns.heatmap(pivot, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
        else:
            im = ax.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(pivot.columns)), [f"PC{col}" for col in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(f"Contrast loadings on shared PCs, layer {main_layer:02d} ({main_residualization})")
        ax.set_xlabel("Shared PC")
        ax.set_ylabel("Contrast")
        save_fig(fig, output_dir / "figures" / "pc_loadings" / f"contrast_pc_loading_heatmap_layer{main_layer:02d}_{main_residualization}")

    axis_pivot = df.groupby(["axis", "pc"], sort=True)["abs_loading"].mean().reset_index().pivot(index="axis", columns="pc", values="abs_loading")
    if not axis_pivot.empty:
        fig, ax = plt.subplots(figsize=(11, max(5, 0.6 * len(axis_pivot))))
        if sns is not None:
            sns.heatmap(axis_pivot, cmap="mako", vmin=0, vmax=min(1, axis_pivot.max().max()), annot=True, fmt=".2f", ax=ax)
        else:
            im = ax.imshow(axis_pivot.to_numpy(), cmap="viridis", vmin=0)
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(axis_pivot.columns)), [f"PC{col}" for col in axis_pivot.columns])
            ax.set_yticks(range(len(axis_pivot.index)), axis_pivot.index)
        ax.set_title(f"Mean absolute loading by axis and shared PC, layer {main_layer:02d} ({main_residualization})")
        ax.set_xlabel("Shared PC")
        ax.set_ylabel("Axis")
        save_fig(fig, output_dir / "figures" / "pc_loadings" / f"axis_pc_abs_loading_heatmap_layer{main_layer:02d}_{main_residualization}")


def plot_cross_axis(output_dir: Path, main_layer: int = 24, main_residualization: str = "family_residualized") -> None:
    summary_path = output_dir / "metrics" / "cross_axis_projection_summary.csv"
    long_path = output_dir / "metrics" / "cross_axis_identity_projections.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    summary = summary[summary["layer"].eq(main_layer) & summary["residualization"].eq(main_residualization)].copy()
    if summary.empty:
        return
    pivot = summary.pivot_table(index="contrast_name", columns="projected_axis", values="range_projection", aggfunc="mean")
    if not pivot.empty:
        pivot = order_matrix(pivot)
        fig, ax = plt.subplots(figsize=(12, max(8, 0.4 * len(pivot))))
        if sns is not None:
            sns.heatmap(pivot, cmap="magma", annot=True, fmt=".2f", ax=ax)
        else:
            im = ax.imshow(pivot.to_numpy(), cmap="magma")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(f"Cross-axis projection range, layer {main_layer:02d} ({main_residualization})")
        ax.set_xlabel("Projected identity axis")
        ax.set_ylabel("Contrast direction")
        save_fig(fig, output_dir / "figures" / "cross_axis" / f"cross_axis_projection_range_heatmap_layer{main_layer:02d}_{main_residualization}")

    if not long_path.exists():
        return
    long_df = pd.read_csv(long_path)
    long_df = long_df[long_df["layer"].eq(main_layer) & long_df["residualization"].eq(main_residualization)].copy()
    for contrast_name, projected_axis in SELECTED_CROSS_AXIS_ORDERINGS:
        plot_df = long_df[long_df["contrast_name"].eq(contrast_name) & long_df["projected_axis"].eq(projected_axis)].copy()
        if plot_df.empty:
            continue
        plot_df = plot_df.sort_values("projection_score")
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_df))))
        ax.barh(plot_df["canonical_label"], plot_df["projection_score"], color=signed_bar_colors(plot_df["projection_score"]))
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"{projected_axis.replace('_', ' ').title()} identities projected on {contrast_name}\nLayer {main_layer:02d}, {main_residualization}")
        ax.set_xlabel("Centroid projection score")
        save_fig(fig, output_dir / "figures" / "cross_axis" / "selected_orderings" / f"{projected_axis}_on_{contrast_name}_layer{main_layer:02d}_{main_residualization}")


def create_paper_panel(output_dir: Path, main_layer: int = 24, main_residualization: str = "family_residualized") -> None:
    fig, axs = plt.subplots(2, 3, figsize=(22, 13))
    try:
        spectrum = pd.read_csv(output_dir / "metrics" / "shared_subspace_spectrum.csv")
        plot_df = spectrum[spectrum["layer"].eq(main_layer)].copy()
        if sns is not None and not plot_df.empty:
            sns.lineplot(data=plot_df, x="component", y="cumulative_explained_variance", hue="residualization", marker="o", ax=axs[0, 0])
        axs[0, 0].set_ylim(0, 1.02)
        axs[0, 0].set_title("A. Cumulative Shared-PC Variance")
    except Exception as exc:
        axs[0, 0].text(0.5, 0.5, f"Spectrum unavailable\n{exc}", ha="center", va="center")
        axs[0, 0].axis("off")

    try:
        decomp = pd.read_csv(output_dir / "metrics" / "decomposition_metrics.csv")
        shared = decomp[decomp["layer"].eq(main_layer) & decomp["residualization"].eq(main_residualization) & decomp["component_type"].eq("shared")]
        pivot = shared.pivot_table(index="contrast_name", columns="k", values="fraction_norm", aggfunc="mean")
        if sns is not None and not pivot.empty:
            sns.heatmap(order_matrix(pivot), cmap="viridis", vmin=0, vmax=1, ax=axs[0, 1], cbar=False)
        axs[0, 1].set_title("B. Fraction Shared by Contrast")
    except Exception as exc:
        axs[0, 1].text(0.5, 0.5, f"Decomposition unavailable\n{exc}", ha="center", va="center")
        axs[0, 1].axis("off")

    try:
        key = decomp[decomp["layer"].eq(main_layer) & decomp["contrast_name"].isin(KEY_CONTRASTS) & decomp["component_type"].isin(["shared", "residual"])]
        if sns is not None and not key.empty:
            sns.lineplot(data=key, x="k", y="auc", hue="component_type", style="contrast_name", ax=axs[0, 2], linewidth=2, marker="o")
        axs[0, 2].axhline(0.5, color="black", linestyle=":", linewidth=1)
        axs[0, 2].set_ylim(0.45, 1.02)
        axs[0, 2].set_title("C. Shared vs Residual AUC")
        add_outside_legend(axs[0, 2], max_items=12)
    except Exception as exc:
        axs[0, 2].text(0.5, 0.5, f"AUC panel unavailable\n{exc}", ha="center", va="center")
        axs[0, 2].axis("off")

    try:
        top = pd.read_csv(output_dir / "metrics" / "shared_pc_top_bottom.csv")
        pc1 = top[top["layer"].eq(main_layer) & top["residualization"].eq(main_residualization) & top["pc"].eq(1)]
        plot_df = pd.concat([pc1[pc1["side"].eq("bottom")].head(8).sort_values("projection_score"), pc1[pc1["side"].eq("top")].head(8).sort_values("projection_score")])
        labels = plot_df["canonical_label"] + " (" + plot_df["axis"].str.replace("_", " ") + ")"
        axs[1, 0].barh(labels, plot_df["projection_score"], color=signed_bar_colors(plot_df["projection_score"]))
        axs[1, 0].axvline(0, color="black", linewidth=1)
        axs[1, 0].set_title("D. Shared PC1 Extremes")
    except Exception as exc:
        axs[1, 0].text(0.5, 0.5, f"PC panel unavailable\n{exc}", ha="center", va="center")
        axs[1, 0].axis("off")

    try:
        loadings = pd.read_csv(output_dir / "metrics" / "contrast_pc_loadings.csv")
        loadings = loadings[loadings["layer"].eq(main_layer) & loadings["residualization"].eq(main_residualization) & loadings["pc"].le(5)]
        pivot = loadings.pivot_table(index="contrast_name", columns="pc", values="loading", aggfunc="mean")
        if sns is not None and not pivot.empty:
            sns.heatmap(order_matrix(pivot), cmap="vlag", center=0, vmin=-1, vmax=1, ax=axs[1, 1], cbar=False)
        axs[1, 1].set_title("E. Contrast x Shared-PC Loadings")
    except Exception as exc:
        axs[1, 1].text(0.5, 0.5, f"Loading panel unavailable\n{exc}", ha="center", va="center")
        axs[1, 1].axis("off")

    try:
        cross = pd.read_csv(output_dir / "metrics" / "cross_axis_projection_summary.csv")
        cross = cross[cross["layer"].eq(main_layer) & cross["residualization"].eq(main_residualization)]
        pivot = cross.pivot_table(index="contrast_name", columns="projected_axis", values="range_projection", aggfunc="mean")
        if sns is not None and not pivot.empty:
            sns.heatmap(order_matrix(pivot), cmap="magma", ax=axs[1, 2], cbar=False)
        axs[1, 2].set_title("F. Cross-Axis Projection Range")
    except Exception as exc:
        axs[1, 2].text(0.5, 0.5, f"Cross-axis panel unavailable\n{exc}", ha="center", va="center")
        axs[1, 2].axis("off")

    fig.suptitle(f"Shared Social Subspace Summary, Layer {main_layer:02d} ({main_residualization})", fontsize=18, y=1.01)
    save_fig(fig, output_dir / "figures" / "paper_panels" / f"shared_subspace_summary_layer{main_layer:02d}")


def plot_all_outputs(output_dir: Path) -> None:
    plot_spectrum(output_dir)
    plot_decomposition(output_dir)
    plot_axis_summary(output_dir)
    plot_pc_interpretation(output_dir)
    plot_pc_loadings(output_dir)
    plot_cross_axis(output_dir)
    create_paper_panel(output_dir)


def main() -> None:
    args = parse_args()
    start_all = time.perf_counter()
    metadata = load_metadata(args.activation_dir)
    layers = parse_int_list(args.layers)
    residualizations = parse_str_list(args.residualizations)
    k_values = sorted(set(parse_int_list(args.k_values)))
    invalid = [item for item in residualizations if item not in RESIDUALIZATION_GROUPS]
    if invalid:
        raise ValueError(f"Unknown residualizations: {invalid}")
    contrasts = load_contrasts(args.contrasts_csv, metadata)
    prepare_output(args.output_dir, args.overwrite)
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "activation_dir": str(args.activation_dir),
        "output_dir": str(args.output_dir),
        "layers": layers,
        "residualizations": residualizations,
        "k_values": k_values,
        "contrasts_csv": str(args.contrasts_csv) if args.contrasts_csv else None,
        "max_points_per_plot": args.max_points_per_plot,
        "random_seed": args.random_seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")

    spectrum_path = args.output_dir / "metrics" / "shared_subspace_spectrum.csv"
    decomp_path = args.output_dir / "metrics" / "decomposition_metrics.csv"
    pc_rankings_path = args.output_dir / "metrics" / "shared_pc_identity_rankings.csv"
    pc_top_bottom_path = args.output_dir / "metrics" / "shared_pc_top_bottom.csv"
    loadings_path = args.output_dir / "metrics" / "contrast_pc_loadings.csv"
    cross_summary_path = args.output_dir / "metrics" / "cross_axis_projection_summary.csv"
    cross_long_path = args.output_dir / "metrics" / "cross_axis_identity_projections.csv"

    max_pc_needed = max(max(k_values), 10)
    for layer in tqdm(layers, desc="layers"):
        layer_start = time.perf_counter()
        x_raw = load_layer(args.activation_dir, layer)
        if x_raw.shape[0] != len(metadata):
            raise ValueError(f"layer_{layer:02d}.npy has {x_raw.shape[0]} rows but metadata has {len(metadata)} rows.")
        print(f"\nLayer {layer:02d}: loaded activations {x_raw.shape} in {elapsed(layer_start)}")
        for residualization in tqdm(residualizations, desc=f"layer {layer:02d} residualizations", leave=False):
            resid_start = time.perf_counter()
            x = residualize(x_raw, metadata, residualization)
            print(f"Layer {layer:02d} {residualization}: prepared representation in {elapsed(resid_start)}")
            records = compute_contrast_directions(x, metadata, contrasts)
            if len(records) < 2:
                print(f"Warning: layer {layer:02d} {residualization} has fewer than 2 valid directions; skipping.")
                continue
            _, singular_values, vt = run_svd(records)
            basis = vt[: min(max_pc_needed, vt.shape[0])]
            append_rows(spectrum_path, spectrum_rows(singular_values, layer, residualization), SPECTRUM_COLUMNS)
            append_rows(decomp_path, decomposition_rows(x, records, basis, metadata, k_values, layer, residualization), DECOMPOSITION_COLUMNS)
            pc_rows, top_bottom_rows = pc_interpretation_rows(x, metadata, basis, layer, residualization, n_pcs=min(10, basis.shape[0]))
            append_rows(pc_rankings_path, pc_rows, PC_RANKING_COLUMNS)
            append_rows(pc_top_bottom_path, top_bottom_rows, PC_TOP_BOTTOM_COLUMNS)
            append_rows(loadings_path, contrast_loading_rows(records, basis, layer, residualization, n_pcs=min(10, basis.shape[0])), CONTRAST_LOADING_COLUMNS)
            cross_summary_rows, cross_long_rows = cross_axis_projection_rows(x, metadata, records, layer, residualization)
            append_rows(cross_summary_path, cross_summary_rows, CROSS_AXIS_SUMMARY_COLUMNS)
            append_rows(cross_long_path, cross_long_rows, CROSS_AXIS_LONG_COLUMNS)
            print(f"Layer {layer:02d} {residualization}: saved metrics in {elapsed(resid_start)}")
        print(f"Layer {layer:02d}: complete in {elapsed(layer_start)}")

    aggregate_axis_sharedness(decomp_path, args.output_dir)
    plot_all_outputs(args.output_dir)
    print(f"\nShared-subspace decomposition complete in {elapsed(start_all)}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
