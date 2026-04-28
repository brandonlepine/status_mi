#!/usr/bin/env python3
"""Plot identity-representation geometry analysis outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from tqdm.auto import tqdm

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency.
    sns = None


DEFAULT_GEOMETRY_DIR = Path(
    "/workspace/status_mi/results/geometry/llama-3.1-8b/"
    "identity_prompts_final_token"
)
DEFAULT_OUTPUT_DIR = DEFAULT_GEOMETRY_DIR / "figures"
DEFAULT_LAYERS = "0,8,16,24,32"
AXES_TO_PLOT = [
    "sexual_orientation",
    "race_ethnicity",
    "disability_status",
    "socioeconomic_status",
    "gender_identity",
    "physical_appearance",
    "religion",
    "nationality",
]
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
LINESTYLES = ["-", "--", "-.", ":"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot identity geometry outputs.")
    parser.add_argument("--geometry_dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--max_points_per_plot", type=int, default=15000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--make_umap",
        action="store_true",
        help="If umap-learn is installed, also make UMAP plots from PCA coordinates.",
    )
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", default="cosine")
    return parser.parse_args()


def parse_layers(layers: str) -> list[int]:
    return [int(part.strip()) for part in layers.split(",") if part.strip()]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains files. Pass --overwrite to add/replace plots."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    subdirs = {
        "root": output_dir,
        "pca_by_identity": output_dir / "pca_by_identity",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def read_pca_layer(geometry_dir: Path, layer: int) -> pd.DataFrame | None:
    path = geometry_dir / "pca" / f"pca_layer_{layer:02d}.csv"
    if not path.exists():
        print(f"Skipping missing PCA file: {path}")
        return None
    return pd.read_csv(path, keep_default_na=False)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        print(f"Skipping empty file: {path}")
        return None


def sample_points(df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def category_colors(values: pd.Series) -> dict[str, str]:
    categories = sorted(values.astype(str).unique())
    return {category: OKABE_ITO[i % len(OKABE_ITO)] for i, category in enumerate(categories)}


def category_markers(values: pd.Series) -> dict[str, str]:
    categories = sorted(values.astype(str).unique())
    return {category: MARKERS[i % len(MARKERS)] for i, category in enumerate(categories)}


def category_linestyles(values: pd.Series) -> dict[str, str]:
    categories = sorted(values.astype(str).unique())
    return {category: LINESTYLES[i % len(LINESTYLES)] for i, category in enumerate(categories)}


def add_category_legend(
    ax: plt.Axes,
    labels: list[str],
    color_map: dict[str, str],
    marker_map: dict[str, str] | None = None,
    linestyle_map: dict[str, str] | None = None,
    max_legend_items: int = 60,
) -> None:
    if len(labels) > max_legend_items:
        ax.text(
            1.02,
            1.0,
            f"{len(labels)} categories\nlegend omitted",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        return

    handles = []
    for label in labels:
        handles.append(
            Line2D(
                [0],
                [0],
                color=color_map[label],
                marker=marker_map[label] if marker_map else None,
                linestyle=linestyle_map[label] if linestyle_map else "",
                linewidth=2 if linestyle_map else 0,
                markersize=6,
                label=label,
            )
        )
    ncol = 1 if len(labels) <= 20 else 2
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncol=ncol,
    )


def save_figure(fig: plt.Figure, path_no_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=220)
    fig.savefig(path_no_suffix.with_suffix(".pdf"))
    plt.close(fig)


def scatter_pca(
    df: pd.DataFrame,
    hue_col: str,
    title: str,
    path_no_suffix: Path,
    max_points: int,
    color_map: dict[str, str] | None = None,
    alpha: float = 0.3,
    point_size: float = 8,
    show_legend: bool = True,
) -> None:
    plot_df = sample_points(df, max_points)
    color_map = color_map or category_colors(df[hue_col])
    marker_map = category_markers(df[hue_col])

    fig, ax = plt.subplots(figsize=(10, 7))
    for label, group in plot_df.groupby(hue_col, sort=True):
        label = str(label)
        ax.scatter(
            group["PC1"],
            group["PC2"],
            c=color_map[label],
            marker=marker_map[label],
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if show_legend:
        add_category_legend(
            ax,
            sorted(df[hue_col].astype(str).unique()),
            color_map,
            marker_map=marker_map,
        )

    save_figure(fig, path_no_suffix)


def progression_plot(
    pca_by_layer: dict[int, pd.DataFrame],
    layers: list[int],
    hue_col: str,
    title: str,
    path_no_suffix: Path,
    max_points: int,
) -> None:
    available_layers = [layer for layer in layers if layer in pca_by_layer]
    if not available_layers:
        return

    combined = pd.concat([pca_by_layer[layer] for layer in available_layers], ignore_index=True)
    color_map = category_colors(combined[hue_col])
    marker_map = category_markers(combined[hue_col])
    n_cols = min(3, len(available_layers))
    n_rows = math.ceil(len(available_layers) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4.6 * n_rows),
        squeeze=False,
    )

    for ax, layer in zip(axes.ravel(), available_layers):
        df = sample_points(pca_by_layer[layer], max_points)
        for label, group in df.groupby(hue_col, sort=True):
            label = str(label)
            ax.scatter(
                group["PC1"],
                group["PC2"],
                c=color_map[label],
                marker=marker_map[label],
                s=6,
                alpha=0.3,
                linewidths=0,
            )
        ax.set_title(f"Layer {layer:02d}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    for ax in axes.ravel()[len(available_layers):]:
        ax.axis("off")

    fig.suptitle(title)
    add_category_legend(
        axes.ravel()[min(len(available_layers), len(axes.ravel())) - 1],
        sorted(combined[hue_col].astype(str).unique()),
        color_map,
        marker_map=marker_map,
    )
    save_figure(fig, path_no_suffix)


def plot_pca_outputs(
    geometry_dir: Path,
    output_dir: Path,
    layers: list[int],
    max_points: int,
) -> dict[int, pd.DataFrame]:
    pca_by_layer = {}
    for layer in tqdm(layers, desc="PCA scatter plots"):
        df = read_pca_layer(geometry_dir, layer)
        if df is None:
            continue
        pca_by_layer[layer] = df
        scatter_pca(
            df,
            "axis",
            f"Prompt-level final-token PCA by identity axis - Layer {layer:02d}",
            output_dir / f"pca_axis_layer_{layer:02d}",
            max_points=max_points,
            alpha=0.3,
        )
        scatter_pca(
            df,
            "family",
            f"Prompt-level final-token PCA by template family - Layer {layer:02d}",
            output_dir / f"pca_family_layer_{layer:02d}",
            max_points=max_points,
            alpha=0.3,
        )
        plot_centroids(
            df,
            output_dir / f"pca_axis_centroids_layer_{layer:02d}",
            f"Final-token PCA by axis with identity centroids - Layer {layer:02d}",
            max_points,
        )

    progression_plot(
        pca_by_layer,
        layers,
        "axis",
        "Prompt-level final-token PCA by identity axis across selected layers",
        output_dir / "pca_axis_layer_progression",
        max_points,
    )
    progression_plot(
        pca_by_layer,
        layers,
        "family",
        "Prompt-level final-token PCA by template family across selected layers",
        output_dir / "pca_family_layer_progression",
        max_points,
    )
    return pca_by_layer


def plot_centroids(
    df: pd.DataFrame, path_no_suffix: Path, title: str, max_points: int
) -> None:
    plot_df = sample_points(df, max_points)
    color_map = category_colors(df["axis"])
    marker_map = category_markers(df["axis"])
    centroids = (
        df.groupby(["identity_id", "axis"], sort=True)[["PC1", "PC2"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    for axis, group in plot_df.groupby("axis", sort=True):
        axis = str(axis)
        ax.scatter(
            group["PC1"],
            group["PC2"],
            c=color_map[axis],
            marker=marker_map[axis],
            s=6,
            alpha=0.18,
            linewidths=0,
        )
    for axis, group in centroids.groupby("axis", sort=True):
        axis = str(axis)
        ax.scatter(
            group["PC1"],
            group["PC2"],
            c=color_map[axis],
            marker=marker_map[axis],
            s=55,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.5,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    add_category_legend(
        ax,
        sorted(df["axis"].astype(str).unique()),
        color_map,
        marker_map=marker_map,
    )
    save_figure(fig, path_no_suffix)


def plot_identity_axis_pcas(
    pca_by_layer: dict[int, pd.DataFrame],
    output_dir: Path,
    layers: list[int],
    max_points: int,
) -> None:
    identity_dir = output_dir / "pca_by_identity"
    identity_dir.mkdir(parents=True, exist_ok=True)

    for axis in tqdm(AXES_TO_PLOT, desc="Identity PCA plots"):
        axis_layers = {}
        for layer in layers:
            if layer not in pca_by_layer:
                continue
            df = pca_by_layer[layer]
            axis_df = df[df["axis"].eq(axis)].copy()
            if axis_df.empty:
                continue
            axis_layers[layer] = axis_df
            hue_col = "canonical_label" if axis_df["canonical_label"].nunique() <= 60 else "identity_id"
            scatter_pca(
                axis_df,
                hue_col,
                f"Prompt-level final-token PCA by identity - {axis} - Layer {layer:02d}",
                identity_dir / f"{axis}_layer_{layer:02d}",
                max_points=max_points,
                alpha=0.4,
                point_size=10,
                show_legend=axis_df[hue_col].nunique() <= 60,
            )
        if axis_layers:
            progression_plot(
                axis_layers,
                layers,
                "identity_id",
                f"Prompt-level final-token PCA by identity across selected layers - {axis}",
                identity_dir / f"{axis}_layer_progression",
                max_points,
            )


def line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    ylabel: str,
    path_no_suffix: Path,
    figsize: tuple[float, float] = (16, 6),
) -> None:
    if df.empty or y_col not in df.columns:
        print(f"Skipping empty plot: {title}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    color_map = category_colors(df[hue_col])
    linestyle_map = category_linestyles(df[hue_col])
    for label, group in df.groupby(hue_col, sort=True):
        label = str(label)
        group = group.sort_values(x_col)
        ax.plot(
            group[x_col],
            group[y_col],
            color=color_map[label],
            linestyle=linestyle_map[label],
            linewidth=2,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)
    add_category_legend(
        ax,
        sorted(df[hue_col].astype(str).unique()),
        color_map,
        linestyle_map=linestyle_map,
        max_legend_items=40,
    )
    save_figure(fig, path_no_suffix)


def plot_probe_scores(geometry_dir: Path, output_dir: Path) -> None:
    axis_path = geometry_dir / "probes" / "axis_probe_scores.csv"
    identity_path = geometry_dir / "probes" / "identity_within_axis_probe_scores.csv"

    axis_scores = safe_read_csv(axis_path)
    if axis_scores is not None:
        line_plot(
            axis_scores,
            "layer",
            "macro_f1_mean",
            "split_type",
            "Axis probe performance by layer: macro F1 for final-token activations",
            "Macro F1",
            output_dir / "probe_axis_macro_f1_by_layer",
        )
        line_plot(
            axis_scores,
            "layer",
            "accuracy_mean",
            "split_type",
            "Axis probe performance by layer: accuracy for final-token activations",
            "Accuracy",
            output_dir / "probe_axis_accuracy_by_layer",
        )

    identity_scores = safe_read_csv(identity_path)
    if identity_scores is not None:
        line_plot(
            identity_scores,
            "layer",
            "macro_f1_mean",
            "axis",
            "Identity-within-axis probe performance by layer: macro F1 for final-token activations",
            "Macro F1",
            output_dir / "probe_identity_within_axis_macro_f1_by_layer",
        )


def plot_family_stability(geometry_dir: Path, output_dir: Path) -> None:
    path = geometry_dir / "family_stability" / "family_cosines_summary.csv"
    df = safe_read_csv(path)
    if df is None:
        return

    line_plot(
        df,
        "layer",
        "mean_cosine",
        "axis",
        "Family stability by layer: mean pairwise cosine between same-identity family means",
        "Mean centered cosine\n(family means, centered by layer global mean)",
        output_dir / "family_stability_mean_cosine_by_layer",
        figsize=(16, 6),
    )
    line_plot(
        df,
        "layer",
        "median_cosine",
        "axis",
        "Family stability by layer: median pairwise cosine between same-identity family means",
        "Median centered cosine\n(family means, centered by layer global mean)",
        output_dir / "family_stability_median_cosine_by_layer",
        figsize=(16, 6),
    )

    pivot = df.pivot(index="axis", columns="layer", values="mean_cosine")
    fig, ax = plt.subplots(figsize=(15, 7))
    if sns is not None:
        sns.heatmap(pivot, cmap="viridis", annot=False, ax=ax)
    else:
        im = ax.imshow(pivot.to_numpy(), aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title(
        "Family stability heatmap: mean pairwise centered cosine between "
        "same-identity family mean vectors"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Identity axis")
    save_figure(fig, output_dir / "family_stability_heatmap")


def plot_contrasts(geometry_dir: Path, output_dir: Path) -> None:
    scores_path = geometry_dir / "contrasts" / "contrast_scores.csv"
    holdout_path = geometry_dir / "contrasts" / "contrast_family_holdout_scores.csv"

    scores = safe_read_csv(scores_path)
    if scores is not None:
        line_plot(
            scores,
            "layer",
            "auc_all",
            "contrast_name",
            "Contrast direction separation by layer: AUC from final-token projections",
            "AUC for identity_a vs identity_b",
            output_dir / "contrast_auc_by_layer",
            figsize=(17, 7),
        )
        line_plot(
            scores,
            "layer",
            "cohens_d_all",
            "contrast_name",
            "Contrast direction separation by layer: Cohen's d of final-token projections",
            "Cohen's d for identity_a minus identity_b",
            output_dir / "contrast_cohens_d_by_layer",
            figsize=(17, 7),
        )

    holdout = safe_read_csv(holdout_path)
    if holdout is not None:
        if not holdout.empty:
            summary = (
                holdout.groupby(["layer", "contrast_name"], sort=True)["auc"]
                .agg(auc_mean="mean", auc_sd="std")
                .reset_index()
            )
            line_plot(
                summary,
                "layer",
                "auc_mean",
                "contrast_name",
                "Contrast direction family holdout by layer: mean AUC on held-out template family",
                "Mean held-out-family AUC",
                output_dir / "contrast_family_holdout_auc_by_layer",
                figsize=(17, 7),
            )


def plot_umap_if_available(
    pca_by_layer: dict[int, pd.DataFrame],
    output_dir: Path,
    max_points: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> None:
    try:
        import umap  # type: ignore
    except ImportError:
        print("Skipping UMAP plots because umap-learn is not installed.")
        return

    umap_dir = output_dir / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)
    for layer, df in tqdm(pca_by_layer.items(), desc="UMAP plots"):
        pc_cols = [col for col in df.columns if col.startswith("PC")]
        if len(pc_cols) < 2:
            continue
        plot_df = sample_points(df, max_points)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=None,
            n_jobs=-1,
        )
        coords = reducer.fit_transform(plot_df[pc_cols].to_numpy())
        umap_df = plot_df.copy()
        umap_df["UMAP1"] = coords[:, 0]
        umap_df["UMAP2"] = coords[:, 1]

        for hue_col, suffix, title_label in [
            ("axis", "axis", "identity axis"),
            ("family", "family", "template family"),
        ]:
            color_map = category_colors(umap_df[hue_col])
            marker_map = category_markers(umap_df[hue_col])
            fig, ax = plt.subplots(figsize=(10, 7))
            for label, group in umap_df.groupby(hue_col, sort=True):
                label = str(label)
                ax.scatter(
                    group["UMAP1"],
                    group["UMAP2"],
                    c=color_map[label],
                    marker=marker_map[label],
                    s=7,
                    alpha=0.35,
                    linewidths=0,
                )
            ax.set_title(
                f"UMAP of prompt-level PCA coordinates by {title_label} - "
                f"Layer {layer:02d}"
            )
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            add_category_legend(
                ax,
                sorted(umap_df[hue_col].astype(str).unique()),
                color_map,
                marker_map=marker_map,
            )
            save_figure(fig, umap_dir / f"umap_{suffix}_layer_{layer:02d}")

        centroids = (
            umap_df.groupby(["identity_id", "axis"], sort=True)[["UMAP1", "UMAP2"]]
            .mean()
            .reset_index()
        )
        color_map = category_colors(umap_df["axis"])
        marker_map = category_markers(umap_df["axis"])
        fig, ax = plt.subplots(figsize=(10, 7))
        for axis, group in umap_df.groupby("axis", sort=True):
            axis = str(axis)
            ax.scatter(
                group["UMAP1"],
                group["UMAP2"],
                c=color_map[axis],
                marker=marker_map[axis],
                s=5,
                alpha=0.15,
                linewidths=0,
            )
        for axis, group in centroids.groupby("axis", sort=True):
            axis = str(axis)
            ax.scatter(
                group["UMAP1"],
                group["UMAP2"],
                c=color_map[axis],
                marker=marker_map[axis],
                s=55,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.5,
            )
        ax.set_title(f"UMAP by axis with identity centroids - Layer {layer:02d}")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        add_category_legend(
            ax,
            sorted(umap_df["axis"].astype(str).unique()),
            color_map,
            marker_map=marker_map,
        )
        save_figure(fig, umap_dir / f"umap_axis_centroids_layer_{layer:02d}")


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    subdirs = prepare_output_dir(args.output_dir, args.overwrite)

    pca_by_layer = plot_pca_outputs(
        args.geometry_dir,
        subdirs["root"],
        layers,
        args.max_points_per_plot,
    )
    plot_identity_axis_pcas(
        pca_by_layer,
        subdirs["root"],
        layers,
        args.max_points_per_plot,
    )
    plot_probe_scores(args.geometry_dir, subdirs["root"])
    plot_family_stability(args.geometry_dir, subdirs["root"])
    plot_contrasts(args.geometry_dir, subdirs["root"])

    if args.make_umap:
        plot_umap_if_available(
            pca_by_layer,
            subdirs["root"],
            args.max_points_per_plot,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
        )
    else:
        print("UMAP plots not requested. Re-run with --make_umap to create figures/umap outputs.")

    print(f"Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
