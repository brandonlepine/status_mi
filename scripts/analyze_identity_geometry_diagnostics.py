#!/usr/bin/env python3
"""Second-pass diagnostics for identity-representation geometry.

This script asks whether identity geometry survives controls for prompt surface
form. It computes variance decomposition, residualized PCA/probes, contrast
generalization, surface-form probes, and identity-specific PCA/UMAP plots.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pandas.errors import EmptyDataError
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler  # noqa: F401 (kept; canonical scaler lives in common)

from common import (
    SCALING_MODES, DEFAULT_SCALING, CenterOnlyScaler, make_scaler,
    cohens_d, cosine, normalize, compute_direction, compute_direction_for_pair,
    evaluate_projection, residualize, OKABE_ITO, save_fig,
)  # noqa: E402, F401  (audit 5.10 — single source of truth)
from tqdm.auto import tqdm

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency.
    sns = None


DEFAULT_ACTIVATION_DIR = Path(
    "/workspace/status_mi/results/activations/llama-3.1-8b/"
    "identity_prompts_final_token"
)
DEFAULT_GEOMETRY_DIR = Path(
    "/workspace/status_mi/results/geometry/llama-3.1-8b/"
    "identity_prompts_final_token"
)
DEFAULT_OUTPUT_DIR = DEFAULT_GEOMETRY_DIR / "diagnostics"
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
FACTORS = ["family", "template_id", "required_form", "axis", "identity_id"]
RESIDUALIZATIONS = {
    "raw": None,
    "family_residualized": "family",
    "template_residualized": "template_id",
    "required_form_residualized": "required_form",
}
PROBE_COLUMNS = [
    "layer",
    "residualization",
    "split_type",
    "task",
    "axis",
    "n_classes",
    "n_samples",
    "accuracy_mean",
    "accuracy_sd",
    "macro_f1_mean",
    "macro_f1_sd",
]
_CONTRAST_HEADER = [
    "layer",
    "residualization",
    "contrast_name",
    "identity_a",
    "identity_b",
    "axis",
    "heldout_family",
]
_CONTRAST_FOOTER = ["n_train_a", "n_train_b", "n_eval_a", "n_eval_b"]
# In-sample rows: direction and evaluation both on the same A and B prompts.
# Kept as a diagnostic; the headline is the family-holdout columns below (2.1).
CONTRAST_IN_SAMPLE_COLUMNS = _CONTRAST_HEADER + ["auc_in_sample", "cohens_d_in_sample"] + _CONTRAST_FOOTER
CONTRAST_HOLDOUT_COLUMNS = _CONTRAST_HEADER + ["auc", "cohens_d"] + _CONTRAST_FOOTER
# Backwards-compatible alias for any caller that imports CONTRAST_COLUMNS.
CONTRAST_COLUMNS = CONTRAST_HOLDOUT_COLUMNS
PLOT_RESIDUALIZATIONS = ["raw", "family_residualized", "template_residualized"]
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
# Canonical contrast pairs are sourced from scripts/contrast_registry.py per
# audit 4.1 (single validated registry; ses_low_income / ses_high_socioeconomic_status
# typos corrected in the registry so the SES axis runs 4 contrasts not silently 2).
# Populated at main() startup; see resolve_contrasts_from_registry.
CONTRASTS: list[tuple[str, str]] = []
# OKABE_ITO sourced from common module (audit 5.10).
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
LINESTYLES = ["-", "--", "-.", ":"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run second-pass identity geometry diagnostics."
    )
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--geometry_dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--selected_layers_for_plots", default="0,8,16,24,32")
    parser.add_argument("--probe_pca_dim", type=int, default=64)
    parser.add_argument("--pca_components", type=int, default=10)
    parser.add_argument("--max_plot_points", type=int, default=15000)
    parser.add_argument("--make_umap", action="store_true")
    parser.add_argument("--skip_probes", action="store_true")
    parser.add_argument("--skip_axis_probes", action="store_true")
    parser.add_argument("--skip_identity_within_axis_probes", action="store_true")
    parser.add_argument("--skip_surface_form_probes", action="store_true")
    parser.add_argument("--skip_template_id_probe", action="store_true")
    parser.add_argument("--run_template_id_probe", action="store_true")
    parser.add_argument("--skip_umap", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--only_variance", action="store_true")
    parser.add_argument("--only_pca", action="store_true")
    parser.add_argument("--only_contrasts", action="store_true")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--max_probe_rows", type=int, default=None)
    parser.add_argument("--solver", default="saga")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-layer outputs when present instead of recomputing them.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=20,
        help=(
            "Number of label-permutation null replicates per probe (audit 2.2). "
            "Default 20 keeps the diagnostics run tractable; bump to >=100 for "
            "the headline number. Set 0 to disable; null fields are NaN."
        ),
    )
    parser.add_argument(
        "--null_random_seed",
        type=int,
        default=None,
        help="RNG seed for the permutation null. Defaults to --random_seed.",
    )
    parser.add_argument(
        "--scaling",
        choices=SCALING_MODES,
        default=DEFAULT_SCALING,
        help=(
            "Pre-PCA scaling (audit 5.9). 'center_only' subtracts per-dim mean "
            "only — preserves the activation-space variance structure including "
            "rogue / high-norm dimensions, so explained_variance_ratio describes "
            "activation space. 'standardize' z-scores per dim (legacy)."
        ),
    )
    parser.add_argument(
        "--verify_fold_internal_pca",
        type=int,
        default=None,
        help=(
            "Optional layer index to verify the global-PCA design choice (audit 2.8). "
            "If set, runs each probe configuration on this layer at residualization='raw' "
            "a second time with StandardScaler + PCA fit INSIDE each CV fold and writes "
            "probes/pca_leakage_verification.csv. Run once; small deltas vindicate the "
            "speed-tradeoff in make_probe_features."
        ),
    )
    return parser.parse_args()


def run_fold_internal_pca_verification_diag(
    *,
    x_raw_for_layer: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    axis_probe_rows: list[dict[str, object]],
    identity_probe_rows: list[dict[str, object]],
    surface_probe_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """For the layer chosen by --verify_fold_internal_pca, run the same
    axis / identity-within-axis / surface-form probes at residualization='raw'
    with fold-internal PCA and pair the numbers with the existing global-PCA
    rows. Returns one row per probe configuration; the caller writes the CSV.
    """
    rows: list[dict[str, object]] = []
    null_seed = args.null_random_seed if args.null_random_seed is not None else args.random_seed

    def _global_row(rows_list: list[dict[str, object]], **filters) -> dict[str, object] | None:
        for row in rows_list:
            if row.get("layer") != layer:
                continue
            if row.get("residualization") != "raw":
                continue
            if all(row.get(k) == v for k, v in filters.items()):
                return row
        return None

    # Axis probes
    for group_col in ["template_id", "family"]:
        split_name = f"group_by_{group_col}"
        global_row = _global_row(axis_probe_rows, split_type=split_name, task="axis_prediction")
        verification = crossval_probe_fold_internal_pca_diag(
            x_raw=x_raw_for_layer,
            y=metadata["axis"],
            groups=metadata[group_col],
            probe_pca_dim=args.probe_pca_dim,
            random_seed=args.random_seed,
            layer=layer,
            task="axis_prediction",
            split_type=split_name,
            n_splits=args.n_splits,
            solver=args.solver,
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
            scaling=args.scaling,
        )
        if global_row and verification:
            rows.append(_pair_global_and_fold_internal_diag(global_row, verification))

    # Identity-within-axis probes
    for axis_name, axis_meta in metadata.groupby("axis", sort=True):
        if axis_meta["identity_id"].nunique() < 2:
            continue
        idx = axis_meta.index.to_numpy()
        global_row = _global_row(
            identity_probe_rows,
            split_type="group_by_template_id",
            task="identity_within_axis_prediction",
            axis=axis_name,
        )
        verification = crossval_probe_fold_internal_pca_diag(
            x_raw=x_raw_for_layer[idx],
            y=axis_meta["identity_id"].reset_index(drop=True),
            groups=axis_meta["template_id"].reset_index(drop=True),
            probe_pca_dim=args.probe_pca_dim,
            random_seed=args.random_seed,
            layer=layer,
            task=f"identity_within_axis_prediction[{axis_name}]",
            split_type="group_by_template_id",
            n_splits=args.n_splits,
            solver=args.solver,
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
            scaling=args.scaling,
        )
        if global_row and verification:
            rows.append(_pair_global_and_fold_internal_diag(global_row, verification, axis=axis_name))

    # Surface-form probes (raw residualization only)
    surface_tasks = ["required_form", "family"]
    if getattr(args, "run_template_id_probe", False) and not args.skip_template_id_probe:
        surface_tasks.append("template_id")
    for task_col in surface_tasks:
        task_name = f"{task_col}_prediction"
        global_row = _global_row(surface_probe_rows, task=task_name)
        verification = crossval_probe_fold_internal_pca_diag(
            x_raw=x_raw_for_layer,
            y=metadata[task_col],
            groups=metadata["identity_id"],
            probe_pca_dim=args.probe_pca_dim,
            random_seed=args.random_seed,
            layer=layer,
            task=task_name,
            split_type="group_by_identity_id",
            n_splits=args.n_splits,
            solver=args.solver,
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
            scaling=args.scaling,
        )
        if global_row and verification:
            rows.append(_pair_global_and_fold_internal_diag(global_row, verification))
    return rows


def _pair_global_and_fold_internal_diag(
    global_row: dict[str, object], verification: dict[str, object], axis: str = "",
) -> dict[str, object]:
    return {
        "layer": global_row["layer"],
        "residualization": "raw",
        "task": global_row["task"],
        "split_type": global_row["split_type"],
        "axis": axis or global_row.get("axis", ""),
        "n_classes": global_row["n_classes"],
        "n_samples": global_row["n_samples"],
        "global_pca_accuracy_mean": global_row["accuracy_mean"],
        "global_pca_accuracy_sd": global_row["accuracy_sd"],
        "global_pca_macro_f1_mean": global_row["macro_f1_mean"],
        "global_pca_macro_f1_sd": global_row["macro_f1_sd"],
        "fold_internal_pca_accuracy_mean": verification["fold_internal_pca_accuracy_mean"],
        "fold_internal_pca_accuracy_sd": verification["fold_internal_pca_accuracy_sd"],
        "fold_internal_pca_macro_f1_mean": verification["fold_internal_pca_macro_f1_mean"],
        "fold_internal_pca_macro_f1_sd": verification["fold_internal_pca_macro_f1_sd"],
        "accuracy_delta": (
            verification["fold_internal_pca_accuracy_mean"] - global_row["accuracy_mean"]
        ),
        "macro_f1_delta": (
            verification["fold_internal_pca_macro_f1_mean"] - global_row["macro_f1_mean"]
        ),
    }


def parse_layers(layer_arg: str | None, activation_dir: Path) -> list[int]:
    available = sorted(
        int(path.stem.split("_")[1]) for path in activation_dir.glob("layer_*.npy")
    )
    if not available:
        raise FileNotFoundError(f"No layer_*.npy files found in {activation_dir}")
    if layer_arg is None:
        return available
    selected = [int(part.strip()) for part in layer_arg.split(",") if part.strip()]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(f"Requested missing layers: {missing}")
    return selected


def parse_selected_layers(layer_arg: str) -> list[int]:
    return [int(part.strip()) for part in layer_arg.split(",") if part.strip()]


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def prepare_output_dir(output_dir: Path, overwrite: bool, resume: bool) -> dict[str, Path]:
    if overwrite and resume:
        raise ValueError("Use only one of --overwrite or --resume.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite and not resume:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to replace it.")
        if overwrite:
            shutil.rmtree(output_dir)

    subdirs = {
        "pca_residualized": output_dir / "pca_residualized",
        "probes": output_dir / "probes",
        "contrasts": output_dir / "contrasts",
        "figures": output_dir / "figures",
        "pca_by_identity": output_dir / "figures" / "pca_by_identity",
        "umap_by_identity": output_dir / "figures" / "umap_by_identity",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for residualization in RESIDUALIZATIONS:
        (subdirs["pca_residualized"] / residualization).mkdir(parents=True, exist_ok=True)
    return subdirs


def load_metadata(activation_dir: Path) -> pd.DataFrame:
    path = activation_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata.csv: {path}")
    metadata = pd.read_csv(path, keep_default_na=False)
    missing = [col for col in METADATA_COLUMNS if col not in metadata.columns]
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")
    if metadata["prompt"].astype(str).str.strip().eq("").any():
        raise ValueError("metadata.csv contains empty prompts.")
    return metadata.reset_index(drop=True)


def load_layer(activation_dir: Path, layer: int, n_rows: int) -> np.ndarray:
    path = activation_dir / f"layer_{layer:02d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing layer file: {path}")
    x = np.load(path, mmap_mode="r")
    if x.ndim != 2:
        raise ValueError(f"{path.name} must be 2D; got shape {x.shape}")
    if x.shape[0] != n_rows:
        raise ValueError(f"{path.name} rows {x.shape[0]} != metadata rows {n_rows}")
    return np.asarray(x, dtype=np.float32)


def category_colors(values: pd.Series) -> dict[str, str]:
    categories = sorted(values.astype(str).unique())
    return {category: OKABE_ITO[i % len(OKABE_ITO)] for i, category in enumerate(categories)}


def category_markers(values: pd.Series) -> dict[str, str]:
    categories = sorted(values.astype(str).unique())
    return {category: MARKERS[i % len(MARKERS)] for i, category in enumerate(categories)}


def category_linestyles(values: pd.Series) -> dict[str, str]:
    categories = sorted(values.astype(str).unique())
    return {category: LINESTYLES[i % len(LINESTYLES)] for i, category in enumerate(categories)}


def add_legend(
    ax: plt.Axes,
    labels: list[str],
    color_map: dict[str, str],
    marker_map: dict[str, str] | None = None,
    linestyle_map: dict[str, str] | None = None,
    max_items: int = 60,
) -> None:
    if len(labels) > max_items:
        ax.text(
            1.02,
            1.0,
            f"{len(labels)} categories\nlegend omitted",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        return
    handles = [
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
        for label in labels
    ]
    ncol = 1 if len(labels) <= 20 else 2
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
        ncol=ncol,
    )


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, keep_default_na=False)
    except EmptyDataError:
        return None


def total_sum_squares(x: np.ndarray, global_mean: np.ndarray) -> float:
    centered = x - global_mean
    return float(np.sum(centered * centered))


def variance_decomposition_layer(
    x: np.ndarray, metadata: pd.DataFrame, layer: int
) -> list[dict[str, object]]:
    global_mean = x.mean(axis=0)
    ss_total = total_sum_squares(x, global_mean)
    rows = []
    for factor in FACTORS:
        ss_factor = 0.0
        for _, idx in metadata.groupby(factor, sort=True).groups.items():
            idx_array = np.fromiter(idx, dtype=int)
            group_mean = x[idx_array].mean(axis=0)
            diff = group_mean - global_mean
            ss_factor += len(idx_array) * float(np.dot(diff, diff))
        rows.append(
            {
                "layer": layer,
                "factor": factor,
                "n_groups": int(metadata[factor].nunique()),
                "ss_factor": ss_factor,
                "ss_total": ss_total,
                "eta_squared": ss_factor / ss_total if ss_total > 0 else np.nan,
            }
        )
    return rows


def run_pca(
    x: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    output_dir: Path,
    pca_components: int,
    random_seed: int,
    scaling: str = DEFAULT_SCALING,
) -> pd.DataFrame:
    n_components = min(pca_components, x.shape[0] - 1, x.shape[1])
    if n_components < 1:
        raise ValueError("Need at least two rows for PCA.")
    x_scaled = make_scaler(scaling).fit_transform(x)
    total_variance = float(np.var(x_scaled, axis=0).sum())
    if total_variance <= 0 or not np.isfinite(total_variance):
        pcs = np.zeros((len(x), n_components), dtype=np.float32)
        evr = np.zeros(n_components, dtype=np.float32)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pca = PCA(
                n_components=n_components,
                random_state=random_seed,
                svd_solver="randomized",
            )
            pcs = pca.fit_transform(x_scaled)
            evr = np.nan_to_num(pca.explained_variance_ratio_)

    pca_df = metadata[METADATA_COLUMNS].copy()
    pca_df.insert(0, "original_row_idx", np.arange(len(metadata)))
    for idx in range(n_components):
        pca_df[f"PC{idx + 1}"] = pcs[:, idx]
    pca_df.to_csv(output_dir / f"pca_layer_{layer:02d}.csv", index=False)

    return pd.DataFrame(
        {
            "layer": layer,
            "pc": np.arange(1, n_components + 1),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": np.cumsum(evr),
        }
    )


def make_probe_features(
    x: np.ndarray,
    probe_pca_dim: int,
    random_seed: int,
    label: str,
    scaling: str = DEFAULT_SCALING,
) -> np.ndarray | None:
    """Build probe features by fitting StandardScaler + randomized PCA once on
    the full layer/residualization.

    Audit issue 2.8: PCA is fit on data that include the held-out CV fold.
    The leakage is mild (PCA is unsupervised) but technically present. The
    design choice is for speed: refitting the randomized SVD inside every
    (fold × residualization × probe configuration × layer) is intractable
    on the full corpus. To empirically vindicate the choice, run with
    `--verify_fold_internal_pca <layer>`: that triggers an extra pass through
    `crossval_probe_fold_internal_pca_diag` which fits the scaler + PCA
    inside each fold on train rows only, and writes
    `probes/pca_leakage_verification.csv` with side-by-side numbers. If the
    accuracy_delta is < the per-fold SD, the global-PCA path is defensible.
    """
    if not np.isfinite(x).all():
        print(f"Skipping probes for {label}: non-finite activations.")
        return None
    if float(np.var(x, axis=0).sum()) <= 0:
        print(f"Skipping probes for {label}: zero variance.")
        return None
    x_scaled = make_scaler(scaling).fit_transform(x)
    n_components = min(probe_pca_dim, x_scaled.shape[0] - 1, x_scaled.shape[1])
    if probe_pca_dim and n_components >= 1 and n_components < x_scaled.shape[1]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_scaled = PCA(
                n_components=n_components,
                random_state=random_seed,
                svd_solver="randomized",
            ).fit_transform(x_scaled)
    return np.asarray(np.nan_to_num(x_scaled), dtype=np.float32)


def crossval_probe_fold_internal_pca_diag(
    x_raw: np.ndarray,
    y: pd.Series,
    groups: pd.Series,
    probe_pca_dim: int,
    random_seed: int,
    layer: int,
    task: str,
    split_type: str,
    n_splits: int,
    solver: str,
    max_iter: int,
    n_jobs: int,
    scaling: str = DEFAULT_SCALING,
) -> dict[str, object] | None:
    """Audit-2.8 verifier (diagnostics flavor). Fits scaler + PCA inside each
    CV fold on the train rows only. The scaler honors --scaling (audit 5.9).
    """
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    if y.nunique() < 2 or groups.nunique() < 2 or not np.isfinite(x_raw).all():
        return None
    n_splits = min(n_splits, groups.nunique())
    if n_splits < 2:
        return None
    try:
        splits = list(GroupKFold(n_splits=n_splits).split(x_raw, y, groups))
    except ValueError as exc:
        print(f"Skipping fold-internal-PCA verifier {task} layer {layer}: {exc}")
        return None
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        for train_idx, test_idx in splits:
            if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
                continue
            x_tr_raw = x_raw[train_idx]
            x_te_raw = x_raw[test_idx]
            scaler = make_scaler(scaling).fit(x_tr_raw)
            x_tr = scaler.transform(x_tr_raw)
            x_te = scaler.transform(x_te_raw)
            if not np.isfinite(x_tr).all() or not np.isfinite(x_te).all():
                continue
            if probe_pca_dim and probe_pca_dim > 0:
                n_components = min(probe_pca_dim, x_tr.shape[0] - 1, x_tr.shape[1])
                if 1 <= n_components < x_tr.shape[1]:
                    pca = PCA(n_components=n_components, random_state=random_seed, svd_solver="randomized")
                    pca.fit(x_tr)
                    x_tr = pca.transform(x_tr)
                    x_te = pca.transform(x_te)
            x_tr = np.nan_to_num(x_tr).astype(np.float32, copy=False)
            x_te = np.nan_to_num(x_te).astype(np.float32, copy=False)
            model = LogisticRegression(max_iter=max_iter, class_weight="balanced", solver=solver, n_jobs=n_jobs)
            try:
                model.fit(x_tr, y.iloc[train_idx])
                pred = model.predict(x_te)
            except Exception as exc:
                print(f"Skipping failed fold-internal-PCA fold {task} layer {layer}: {exc}")
                continue
            accuracies.append(accuracy_score(y.iloc[test_idx], pred))
            macro_f1s.append(f1_score(y.iloc[test_idx], pred, average="macro"))
    if not accuracies:
        return None
    return {
        "layer": layer,
        "task": task,
        "split_type": split_type,
        "fold_internal_pca_accuracy_mean": float(np.mean(accuracies)),
        "fold_internal_pca_accuracy_sd": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "fold_internal_pca_macro_f1_mean": float(np.mean(macro_f1s)),
        "fold_internal_pca_macro_f1_sd": float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0,
    }


def sample_probe_rows(
    x: np.ndarray,
    metadata: pd.DataFrame,
    target: pd.Series,
    max_probe_rows: int | None,
    random_seed: int,
    label: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.Series]:
    if max_probe_rows is None or len(metadata) <= max_probe_rows:
        print(f"Probe sampling {label}: using all {len(metadata):,} rows")
        return x, metadata.reset_index(drop=True), target.reset_index(drop=True)
    if max_probe_rows <= 0:
        raise ValueError("--max_probe_rows must be positive when provided.")

    sample = (
        metadata.assign(_target=target.to_numpy())
        .groupby("_target", group_keys=False, sort=False)
        .sample(frac=1, random_state=random_seed)
        .groupby("_target", group_keys=False, sort=False)
        .head(max(1, int(np.ceil(max_probe_rows / target.nunique()))))
    )
    if len(sample) > max_probe_rows:
        sample = sample.sample(n=max_probe_rows, random_state=random_seed)
    sample = sample.sort_index()
    idx = sample.index.to_numpy()
    print(f"Probe sampling {label}: {len(metadata):,} -> {len(idx):,} rows")
    return x[idx], metadata.iloc[idx].reset_index(drop=True), target.iloc[idx].reset_index(drop=True)


def _run_cv_folds_diag(
    x: np.ndarray,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    layer: int,
    residualization: str,
    task: str,
    solver: str,
    max_iter: int,
    n_jobs: int,
    verbose: bool = True,
) -> tuple[list[float], list[float]]:
    """Inner CV loop. Returns (accuracies, macro_f1s) over the provided folds."""
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        for train_idx, test_idx in splits:
            if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
                continue
            model = LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                solver=solver,
                n_jobs=n_jobs,
            )
            try:
                model.fit(x[train_idx], y.iloc[train_idx])
                pred = model.predict(x[test_idx])
            except Exception as exc:
                if verbose:
                    print(f"Skipping failed probe fold {task} layer {layer}: {exc}")
                continue
            accuracies.append(accuracy_score(y.iloc[test_idx], pred))
            macro_f1s.append(f1_score(y.iloc[test_idx], pred, average="macro"))
    return accuracies, macro_f1s


def _empty_null_fields_diag() -> dict[str, float | int]:
    return {
        "null_n_permutations": 0,
        "null_accuracy_mean": float("nan"),
        "null_accuracy_sd": float("nan"),
        "null_macro_f1_mean": float("nan"),
        "null_macro_f1_sd": float("nan"),
        "accuracy_z": float("nan"),
        "macro_f1_z": float("nan"),
        "accuracy_p_value": float("nan"),
        "macro_f1_p_value": float("nan"),
    }


def crossval_probe(
    x: np.ndarray,
    y: pd.Series,
    groups: pd.Series,
    layer: int,
    residualization: str,
    split_type: str,
    task: str,
    n_splits: int,
    solver: str,
    max_iter: int,
    n_jobs: int,
    axis: str | None = None,
    n_permutations: int = 0,
    null_rng_seed: int = 42,
) -> dict[str, object] | None:
    """Probe with GroupKFold + optional label-permutation null (audit 2.2).

    When n_permutations > 0, y is globally shuffled `n_permutations` times
    against the same GroupKFold splits, and the observed accuracy /
    macro-F1 are compared to that null distribution. Reports z-score and
    Phipson-Smyth smoothed empirical p alongside the existing means/SDs.
    """
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    if y.nunique() < 2 or groups.nunique() < 2:
        return None
    n_splits = min(n_splits, groups.nunique())
    if n_splits < 2:
        return None
    try:
        splits = list(GroupKFold(n_splits=n_splits).split(x, y, groups))
    except ValueError as exc:
        print(f"Skipping probe {task} {residualization} layer {layer}: {exc}")
        return None

    accuracies, macro_f1s = _run_cv_folds_diag(
        x, y, splits, layer, residualization, task, solver, max_iter, n_jobs, verbose=True,
    )

    if not accuracies:
        return None

    obs_acc = float(np.mean(accuracies))
    obs_f1 = float(np.mean(macro_f1s))

    row: dict[str, object] = {
        "layer": layer,
        "residualization": residualization,
        "split_type": split_type,
        "task": task,
        "axis": axis or "",
        "n_classes": int(y.nunique()),
        "n_samples": int(len(y)),
        "accuracy_mean": obs_acc,
        "accuracy_sd": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "macro_f1_mean": obs_f1,
        "macro_f1_sd": float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0,
    }
    row.update(_empty_null_fields_diag())

    if n_permutations > 0:
        rng = np.random.default_rng(null_rng_seed)
        y_arr = y.to_numpy()
        null_accs: list[float] = []
        null_f1s: list[float] = []
        for _ in range(n_permutations):
            y_perm = pd.Series(rng.permutation(y_arr))
            perm_accs, perm_f1s = _run_cv_folds_diag(
                x, y_perm, splits, layer, residualization, task, solver, max_iter, n_jobs,
                verbose=False,
            )
            if perm_accs:
                null_accs.append(float(np.mean(perm_accs)))
                null_f1s.append(float(np.mean(perm_f1s)))
        if null_accs:
            null_acc_arr = np.asarray(null_accs)
            null_f1_arr = np.asarray(null_f1s)
            n_obs = len(null_acc_arr)
            null_acc_mean = float(null_acc_arr.mean())
            null_acc_sd = float(null_acc_arr.std(ddof=1)) if n_obs > 1 else 0.0
            null_f1_mean = float(null_f1_arr.mean())
            null_f1_sd = float(null_f1_arr.std(ddof=1)) if n_obs > 1 else 0.0
            row["null_n_permutations"] = int(n_obs)
            row["null_accuracy_mean"] = null_acc_mean
            row["null_accuracy_sd"] = null_acc_sd
            row["null_macro_f1_mean"] = null_f1_mean
            row["null_macro_f1_sd"] = null_f1_sd
            row["accuracy_z"] = (obs_acc - null_acc_mean) / (null_acc_sd + 1e-12)
            row["macro_f1_z"] = (obs_f1 - null_f1_mean) / (null_f1_sd + 1e-12)
            row["accuracy_p_value"] = float(
                (1 + (null_acc_arr >= obs_acc).sum()) / (1 + n_obs)
            )
            row["macro_f1_p_value"] = float(
                (1 + (null_f1_arr >= obs_f1).sum()) / (1 + n_obs)
            )

    return row


def run_identity_probes(
    x_probe: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    residualization: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    axis_rows = []
    if not args.skip_axis_probes:
        x_axis, meta_axis, y_axis = sample_probe_rows(
            x_probe,
            metadata,
            metadata["axis"],
            args.max_probe_rows,
            args.random_seed,
            f"axis layer={layer} residualization={residualization}",
        )
        for group_col in tqdm(["template_id", "family"], desc="axis probe splits", leave=False):
            print(
                "Starting axis probes: "
                f"layer={layer} residualization={residualization} "
                f"split={group_col} n={len(meta_axis):,} dim={x_axis.shape[1]} "
                f"classes={y_axis.nunique()}"
            )
            start = time.perf_counter()
            result = crossval_probe(
                x_axis,
                y_axis,
                meta_axis[group_col],
                layer,
                residualization,
                f"group_by_{group_col}",
                "axis_prediction",
                n_splits=args.n_splits,
                solver=args.solver,
                max_iter=args.max_iter,
                n_jobs=args.n_jobs,
                n_permutations=args.n_permutations,
                null_rng_seed=(args.null_random_seed if args.null_random_seed is not None else args.random_seed),
            )
            print(f"Finished axis probes in {elapsed(start)}")
            if result:
                axis_rows.append(result)

    identity_rows = []
    if not args.skip_identity_within_axis_probes:
        axis_groups = list(metadata.groupby("axis", sort=True))
        for axis, axis_meta in tqdm(axis_groups, desc="identity probes by axis", leave=False):
            if axis_meta["identity_id"].nunique() < 2:
                continue
            idx = axis_meta.index.to_numpy()
            x_axis = x_probe[idx]
            y_axis = axis_meta["identity_id"].reset_index(drop=True)
            meta_axis = axis_meta.reset_index(drop=True)
            x_axis, meta_axis, y_axis = sample_probe_rows(
                x_axis,
                meta_axis,
                y_axis,
                args.max_probe_rows,
                args.random_seed,
                f"identity axis={axis} layer={layer} residualization={residualization}",
            )
            print(
                "Starting identity-within-axis probes: "
                f"layer={layer} residualization={residualization} axis={axis} "
                f"n={len(meta_axis):,} dim={x_axis.shape[1]} classes={y_axis.nunique()}"
            )
            start = time.perf_counter()
            result = crossval_probe(
                x_axis,
                y_axis,
                meta_axis["template_id"],
                layer,
                residualization,
                "group_by_template_id",
                "identity_within_axis_prediction",
                n_splits=args.n_splits,
                solver=args.solver,
                max_iter=args.max_iter,
                n_jobs=args.n_jobs,
                axis=axis,
                n_permutations=args.n_permutations,
                null_rng_seed=(args.null_random_seed if args.null_random_seed is not None else args.random_seed),
            )
            print(f"Finished identity-within-axis probes in {elapsed(start)}")
            if result:
                identity_rows.append(result)
    return axis_rows, identity_rows


def run_surface_probes(
    x_probe: np.ndarray, metadata: pd.DataFrame, layer: int, args: argparse.Namespace
) -> list[dict[str, object]]:
    rows = []
    task_cols = ["required_form", "family"]
    if args.run_template_id_probe and not args.skip_template_id_probe:
        task_cols.append("template_id")
    for task_col in tqdm(task_cols, desc="surface probes", leave=False):
        x_task, meta_task, y_task = sample_probe_rows(
            x_probe,
            metadata,
            metadata[task_col],
            args.max_probe_rows,
            args.random_seed,
            f"surface task={task_col} layer={layer}",
        )
        print(
            "Starting surface-form probe: "
            f"layer={layer} task={task_col} n={len(meta_task):,} "
            f"dim={x_task.shape[1]} classes={y_task.nunique()}"
        )
        start = time.perf_counter()
        result = crossval_probe(
            x_task,
            y_task,
            meta_task["identity_id"],
            layer,
            "raw",
            "group_by_identity_id",
            f"{task_col}_prediction",
            n_splits=args.n_splits,
            solver=args.solver,
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
            n_permutations=args.n_permutations,
            null_rng_seed=(args.null_random_seed if args.null_random_seed is not None else args.random_seed),
        )
        print(f"Finished surface-form probe in {elapsed(start)}")
        if result:
            rows.append(result)
    return rows


# cohens_d sourced from common. make_contrast_direction and evaluate_projection
# below are thin adapters around common.compute_direction / evaluate_projection
# that preserve this script's prior signatures (audit 5.10).


def make_contrast_direction(
    x: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, center_mean: np.ndarray
) -> np.ndarray | None:
    """Compute the unit-normed difference-of-means direction using the
    supplied `center_mean` (typically a train-subset mean for held-out
    evaluation). Returns the array directly so existing callers' tuple
    contracts are unchanged; common.compute_direction is the underlying
    implementation."""
    cd = compute_direction(x, mask_a, mask_b, center=True, center_mean=center_mean)
    return cd.direction if cd is not None else None


def _evaluate_with_centering(
    x: np.ndarray,
    direction: np.ndarray,
    center_mean: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[float, float]:
    """Thin wrapper around common.evaluate_projection that does the
    (x - center_mean) @ direction projection first; returns (auc, cohens_d)
    in the 2-tuple shape this script's existing call sites use."""
    scores = (x - center_mean) @ direction
    metrics = evaluate_projection(scores, mask_a, mask_b)
    return metrics["auc"], metrics["cohens_d"]


# Preserve the in-file name `evaluate_projection` for back-compat with this
# script's existing call sites. common.evaluate_projection has a different
# (cleaner) signature; we shadow it here only inside this script.
def evaluate_projection_local(  # noqa: D401 - shim
    x: np.ndarray,
    direction: np.ndarray,
    center_mean: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[float, float]:
    return _evaluate_with_centering(x, direction, center_mean, mask_a, mask_b)


# Existing callers use the bare name `evaluate_projection`; route them to the
# local shim that preserves the old 2-tuple signature.
evaluate_projection = evaluate_projection_local  # type: ignore[assignment]


def _write_holdout_summary(
    contrast_holdout_rows: list[dict[str, object]], contrasts_dir: Path
) -> None:
    """Aggregate the family-holdout rows into a single headline row per
    (layer, residualization, contrast): mean / sd / min / max / n_families.
    This is the audit-recommended headline number (2.1) — downstream
    plotting and any methods-doc table should cite this, not the in-sample
    contrast_full_residualized_scores.csv.
    """
    if not contrast_holdout_rows:
        return
    holdout_df = pd.DataFrame(contrast_holdout_rows)
    if holdout_df.empty or "auc" not in holdout_df.columns:
        return
    summary = (
        holdout_df.groupby(
            ["layer", "residualization", "contrast_name", "identity_a", "identity_b", "axis"],
            sort=True,
        )
        .agg(
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),
            auc_min=("auc", "min"),
            auc_max=("auc", "max"),
            cohens_d_mean=("cohens_d", "mean"),
            cohens_d_sd=("cohens_d", "std"),
            n_families=("auc", "size"),
        )
        .reset_index()
    )
    summary.to_csv(
        contrasts_dir / "contrast_family_holdout_residualized_summary.csv", index=False
    )


def run_contrasts(
    x: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    residualization: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    identity_set = set(metadata["identity_id"].unique())
    axis_lookup = metadata.groupby("identity_id")["axis"].first().to_dict()
    full_rows = []
    holdout_rows = []
    global_mean = x.mean(axis=0, keepdims=True)

    for identity_a, identity_b in tqdm(CONTRASTS, desc="contrasts", leave=False):
        if identity_a not in identity_set or identity_b not in identity_set:
            continue
        contrast_name = f"{identity_a}_vs_{identity_b}"
        mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
        mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
        direction = make_contrast_direction(x, mask_a, mask_b, global_mean)
        if direction is None:
            continue
        auc, d_value = evaluate_projection(x, direction, global_mean, mask_a, mask_b)
        # In-sample evaluation: the direction is the difference of means on the
        # same A and B prompts being projected. Reported as a diagnostic only;
        # the headline is the family-holdout below (audit 2.1).
        full_rows.append(
            {
                "layer": layer,
                "residualization": residualization,
                "contrast_name": contrast_name,
                "identity_a": identity_a,
                "identity_b": identity_b,
                "axis": axis_lookup.get(identity_a, ""),
                "heldout_family": "",
                "auc_in_sample": auc,
                "cohens_d_in_sample": d_value,
                "n_train_a": int(mask_a.sum()),
                "n_train_b": int(mask_b.sum()),
                "n_eval_a": int(mask_a.sum()),
                "n_eval_b": int(mask_b.sum()),
            }
        )

        for heldout_family in sorted(metadata["family"].unique()):
            heldout = metadata["family"].eq(heldout_family).to_numpy()
            train_mask_a = mask_a & ~heldout
            train_mask_b = mask_b & ~heldout
            eval_mask_a = mask_a & heldout
            eval_mask_b = mask_b & heldout
            if min(train_mask_a.sum(), train_mask_b.sum(), eval_mask_a.sum(), eval_mask_b.sum()) == 0:
                continue
            train_mean = x[~heldout].mean(axis=0, keepdims=True)
            direction = make_contrast_direction(x, train_mask_a, train_mask_b, train_mean)
            if direction is None:
                continue
            auc, d_value = evaluate_projection(
                x, direction, train_mean, eval_mask_a, eval_mask_b
            )
            holdout_rows.append(
                {
                    "layer": layer,
                    "residualization": residualization,
                    "contrast_name": contrast_name,
                    "identity_a": identity_a,
                    "identity_b": identity_b,
                    "axis": axis_lookup.get(identity_a, ""),
                    "heldout_family": heldout_family,
                    "auc": auc,
                    "cohens_d": d_value,
                    "n_train_a": int(train_mask_a.sum()),
                    "n_train_b": int(train_mask_b.sum()),
                    "n_eval_a": int(eval_mask_a.sum()),
                    "n_eval_b": int(eval_mask_b.sum()),
                }
            )
    return full_rows, holdout_rows


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
    if df.empty:
        return
    color_map = category_colors(df[hue_col])
    linestyle_map = category_linestyles(df[hue_col])
    fig, ax = plt.subplots(figsize=figsize)
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
    add_legend(
        ax,
        sorted(df[hue_col].astype(str).unique()),
        color_map,
        linestyle_map=linestyle_map,
        max_items=40,
    )
    save_fig(fig, path_no_suffix, dpi=220, bbox_inches=None, tight_layout=True)


def plot_variance_decomposition(output_dir: Path) -> None:
    df = pd.read_csv(output_dir / "variance_decomposition.csv")
    line_plot(
        df,
        "layer",
        "eta_squared",
        "factor",
        "Variance decomposition by layer: eta-squared for metadata factors",
        "eta-squared (between-group SS / total SS)",
        output_dir / "figures" / "variance_decomposition_by_layer",
    )


def plot_probe_diagnostics(output_dir: Path) -> None:
    axis_scores = safe_read_csv(output_dir / "probes" / "axis_probe_residualized_scores.csv")
    if axis_scores is not None and not axis_scores.empty:
        axis_grouped = (
            axis_scores.groupby(["layer", "residualization"], sort=True)["macro_f1_mean"]
            .mean()
            .reset_index()
        )
        line_plot(
            axis_grouped,
            "layer",
            "macro_f1_mean",
            "residualization",
            "Axis probe macro F1 before vs after residualization",
            "Macro F1",
            output_dir / "figures" / "probe_axis_macro_f1_residualized_by_layer",
        )

    identity_scores = safe_read_csv(
        output_dir / "probes" / "identity_within_axis_probe_residualized_scores.csv"
    )
    if identity_scores is not None and not identity_scores.empty:
        grouped = (
            identity_scores.groupby(["layer", "axis", "residualization"], sort=True)[
                "macro_f1_mean"
            ]
            .mean()
            .reset_index()
        )
        for residualization, sub_df in grouped.groupby("residualization", sort=True):
            line_plot(
                sub_df,
                "layer",
                "macro_f1_mean",
                "axis",
                f"Identity-within-axis macro F1 by layer ({residualization})",
                "Macro F1",
                output_dir
                / "figures"
                / f"probe_identity_macro_f1_{residualization}_by_layer",
            )
        line_plot(
            grouped,
            "layer",
            "macro_f1_mean",
            "residualization",
            "Identity-within-axis macro F1 by layer, averaged over axes",
            "Macro F1",
            output_dir / "figures" / "probe_identity_macro_f1_residualized_by_layer",
        )

    surface_scores = safe_read_csv(output_dir / "probes" / "surface_form_probe_scores.csv")
    if surface_scores is not None and not surface_scores.empty:
        line_plot(
            surface_scores,
            "layer",
            "macro_f1_mean",
            "task",
            "Surface-form probe macro F1 by layer",
            "Macro F1",
            output_dir / "figures" / "surface_form_probe_macro_f1_by_layer",
        )


def plot_contrast_diagnostics(output_dir: Path) -> None:
    holdout = safe_read_csv(
        output_dir / "contrasts" / "contrast_family_holdout_residualized_scores.csv"
    )
    if holdout is not None and not holdout.empty:
        summary = (
            holdout.groupby(["layer", "residualization"], sort=True)["auc"]
            .mean()
            .reset_index()
        )
        line_plot(
            summary,
            "layer",
            "auc",
            "residualization",
            "Mean family-holdout contrast AUC by layer",
            "Mean held-out-family AUC",
            output_dir / "figures" / "contrast_family_holdout_auc_residualized_by_layer",
        )

    full = safe_read_csv(output_dir / "contrasts" / "contrast_full_residualized_scores.csv")
    if full is not None and not full.empty:
        summary = (
            full.groupby(["layer", "residualization"], sort=True)["auc"]
            .mean()
            .reset_index()
        )
        line_plot(
            summary,
            "layer",
            "auc",
            "residualization",
            "Full-data contrast AUC by layer",
            "Mean full-data AUC",
            output_dir / "figures" / "contrast_full_auc_residualized_by_layer",
        )


def sample_points(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def scatter_identity(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    axis: str,
    title: str,
    path_no_suffix: Path,
    max_points: int,
    random_seed: int,
) -> None:
    plot_df = sample_points(df, max_points, random_seed)
    hue_col = "canonical_label" if df["canonical_label"].nunique() <= 60 else "identity_id"
    color_map = category_colors(df[hue_col])
    marker_map = category_markers(df[hue_col])
    centroids = (
        df.groupby([hue_col], sort=True)[[x_col, y_col]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    for label, group in plot_df.groupby(hue_col, sort=True):
        label = str(label)
        ax.scatter(
            group[x_col],
            group[y_col],
            c=color_map[label],
            marker=marker_map[label],
            s=9,
            alpha=0.35,
            linewidths=0,
        )
    for label, group in centroids.groupby(hue_col, sort=True):
        label = str(label)
        ax.scatter(
            group[x_col],
            group[y_col],
            c=color_map[label],
            marker=marker_map[label],
            s=65,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.5,
        )
    if df[hue_col].nunique() <= 15:
        for _, row in centroids.iterrows():
            ax.text(row[x_col], row[y_col], str(row[hue_col]), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    add_legend(
        ax,
        sorted(df[hue_col].astype(str).unique()),
        color_map,
        marker_map=marker_map,
        max_items=60,
    )
    save_fig(fig, path_no_suffix, dpi=220, bbox_inches=None, tight_layout=True)


def pca_progression_plot(
    pca_by_layer: dict[int, pd.DataFrame],
    layers: list[int],
    residualization: str,
    axis: str,
    out_dir: Path,
    max_points: int,
    random_seed: int,
) -> None:
    available = [layer for layer in layers if layer in pca_by_layer]
    if not available:
        return
    combined = pd.concat([pca_by_layer[layer] for layer in available], ignore_index=True)
    axis_df = combined[combined["axis"].eq(axis)]
    if axis_df.empty:
        return
    hue_col = "canonical_label" if axis_df["canonical_label"].nunique() <= 60 else "identity_id"
    color_map = category_colors(axis_df[hue_col])
    marker_map = category_markers(axis_df[hue_col])
    n_cols = min(3, len(available))
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.6 * n_rows), squeeze=False)
    for ax, layer in zip(axes.ravel(), available):
        df = pca_by_layer[layer]
        df = sample_points(df[df["axis"].eq(axis)], max_points, random_seed)
        for label, group in df.groupby(hue_col, sort=True):
            label = str(label)
            ax.scatter(
                group["PC1"],
                group["PC2"],
                c=color_map[label],
                marker=marker_map[label],
                s=7,
                alpha=0.35,
                linewidths=0,
            )
        ax.set_title(f"Layer {layer:02d}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    fig.suptitle(f"{axis}: PCA by identity across layers ({residualization})")
    add_legend(
        axes.ravel()[min(len(available), len(axes.ravel())) - 1],
        sorted(axis_df[hue_col].astype(str).unique()),
        color_map,
        marker_map=marker_map,
        max_items=60,
    )
    save_fig(fig, out_dir / f"{axis}_layer_progression", dpi=220, bbox_inches=None, tight_layout=True)


def plot_axis_specific_pca(
    output_dir: Path,
    selected_layers: list[int],
    max_points: int,
    random_seed: int,
) -> None:
    for residualization in PLOT_RESIDUALIZATIONS:
        resid_dir = output_dir / "pca_residualized" / residualization
        pca_by_layer = {}
        for layer in selected_layers:
            pca_path = resid_dir / f"pca_layer_{layer:02d}.csv"
            pca_df = safe_read_csv(pca_path)
            if pca_df is not None:
                pca_by_layer[layer] = pca_df
        if not pca_by_layer:
            continue

        for axis in tqdm(AXES_TO_PLOT, desc=f"PCA identity plots {residualization}"):
            axis_out = output_dir / "figures" / "pca_by_identity" / residualization
            axis_out.mkdir(parents=True, exist_ok=True)
            for layer, pca_df in pca_by_layer.items():
                axis_df = pca_df[pca_df["axis"].eq(axis)].copy()
                if axis_df.empty:
                    continue
                scatter_identity(
                    axis_df,
                    "PC1",
                    "PC2",
                    axis,
                    f"{axis}: prompt-level PCA by identity - layer {layer:02d} ({residualization})",
                    axis_out / f"{axis}_layer_{layer:02d}",
                    max_points,
                    random_seed,
                )
            pca_progression_plot(
                pca_by_layer,
                selected_layers,
                residualization,
                axis,
                axis_out,
                max_points,
                random_seed,
            )


def plot_axis_specific_umap(
    output_dir: Path,
    selected_layers: list[int],
    max_points: int,
    random_seed: int,
) -> None:
    try:
        import umap  # type: ignore
    except ImportError:
        print("UMAP skipped: umap-learn is not installed.")
        return

    for residualization in PLOT_RESIDUALIZATIONS:
        resid_dir = output_dir / "pca_residualized" / residualization
        axis_out = output_dir / "figures" / "umap_by_identity" / residualization
        axis_out.mkdir(parents=True, exist_ok=True)
        for layer in tqdm(selected_layers, desc=f"UMAP identity plots {residualization}"):
            pca_df = safe_read_csv(resid_dir / f"pca_layer_{layer:02d}.csv")
            if pca_df is None:
                continue
            pc_cols = [col for col in pca_df.columns if col.startswith("PC")][:50]
            if len(pc_cols) < 2:
                continue
            for axis in AXES_TO_PLOT:
                axis_df = pca_df[pca_df["axis"].eq(axis)].copy()
                if axis_df.empty:
                    continue
                plot_df = sample_points(axis_df, max_points, random_seed).copy()
                reducer = umap.UMAP(
                    n_neighbors=30,
                    min_dist=0.1,
                    metric="euclidean",
                    random_state=random_seed,
                )
                coords = reducer.fit_transform(plot_df[pc_cols].to_numpy())
                plot_df["UMAP1"] = coords[:, 0]
                plot_df["UMAP2"] = coords[:, 1]
                scatter_identity(
                    plot_df,
                    "UMAP1",
                    "UMAP2",
                    axis,
                    f"{axis}: UMAP by identity - layer {layer:02d} ({residualization})",
                    axis_out / f"{axis}_layer_{layer:02d}",
                    max_points,
                    random_seed,
                )


def write_run_config(args: argparse.Namespace, layers: list[int], metadata: pd.DataFrame) -> None:
    run_config = {
        "activation_dir": str(args.activation_dir),
        "geometry_dir": str(args.geometry_dir),
        "output_dir": str(args.output_dir),
        "layers": layers,
        "selected_layers_for_plots": parse_selected_layers(args.selected_layers_for_plots),
        "probe_pca_dim": args.probe_pca_dim,
        "pca_components": args.pca_components,
        "n_splits": args.n_splits,
        "max_probe_rows": args.max_probe_rows,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "n_jobs": args.n_jobs,
        "skip_probes": args.skip_probes,
        "skip_axis_probes": args.skip_axis_probes,
        "skip_identity_within_axis_probes": args.skip_identity_within_axis_probes,
        "skip_surface_form_probes": args.skip_surface_form_probes,
        "skip_template_id_probe": args.skip_template_id_probe,
        "run_template_id_probe": args.run_template_id_probe,
        "skip_umap": args.skip_umap,
        "skip_plots": args.skip_plots,
        "only_variance": args.only_variance,
        "only_pca": args.only_pca,
        "only_contrasts": args.only_contrasts,
        "max_plot_points": args.max_plot_points,
        "make_umap": args.make_umap,
        "resume": args.resume,
        "random_seed": args.random_seed,
        "scaling": args.scaling,
        "n_permutations": args.n_permutations,
        "num_rows": len(metadata),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")


def read_existing_layer_csvs(output_dir: Path) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Load existing incremental outputs for --resume."""
    variance_rows: list[dict[str, object]] = []
    axis_probe_rows: list[dict[str, object]] = []
    identity_probe_rows: list[dict[str, object]] = []
    surface_probe_rows: list[dict[str, object]] = []
    contrast_full_rows: list[dict[str, object]] = []
    contrast_holdout_rows: list[dict[str, object]] = []

    for path, rows in [
        (output_dir / "variance_decomposition.csv", variance_rows),
        (output_dir / "probes" / "axis_probe_residualized_scores.csv", axis_probe_rows),
        (
            output_dir / "probes" / "identity_within_axis_probe_residualized_scores.csv",
            identity_probe_rows,
        ),
        (output_dir / "probes" / "surface_form_probe_scores.csv", surface_probe_rows),
        (output_dir / "contrasts" / "contrast_full_residualized_scores.csv", contrast_full_rows),
        (
            output_dir / "contrasts" / "contrast_family_holdout_residualized_scores.csv",
            contrast_holdout_rows,
        ),
    ]:
        df = safe_read_csv(path)
        if df is not None and not df.empty:
            rows.extend(df.to_dict("records"))

    return (
        variance_rows,
        axis_probe_rows,
        identity_probe_rows,
        surface_probe_rows,
        contrast_full_rows,
        contrast_holdout_rows,
    )


def write_incremental_outputs(
    output_dir: Path,
    variance_rows: list[dict[str, object]],
    axis_probe_rows: list[dict[str, object]],
    identity_probe_rows: list[dict[str, object]],
    surface_probe_rows: list[dict[str, object]],
    contrast_full_rows: list[dict[str, object]],
    contrast_holdout_rows: list[dict[str, object]],
) -> None:
    probes_dir = output_dir / "probes"
    contrasts_dir = output_dir / "contrasts"
    probes_dir.mkdir(parents=True, exist_ok=True)
    contrasts_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(variance_rows).drop_duplicates().to_csv(
        output_dir / "variance_decomposition.csv", index=False
    )
    pd.DataFrame(axis_probe_rows, columns=PROBE_COLUMNS).drop_duplicates().to_csv(
        probes_dir / "axis_probe_residualized_scores.csv", index=False
    )
    pd.DataFrame(identity_probe_rows, columns=PROBE_COLUMNS).drop_duplicates().to_csv(
        probes_dir / "identity_within_axis_probe_residualized_scores.csv", index=False
    )
    pd.DataFrame(surface_probe_rows, columns=PROBE_COLUMNS).drop_duplicates().to_csv(
        probes_dir / "surface_form_probe_scores.csv", index=False
    )
    pd.DataFrame(contrast_holdout_rows, columns=CONTRAST_HOLDOUT_COLUMNS).drop_duplicates().to_csv(
        contrasts_dir / "contrast_family_holdout_residualized_scores.csv", index=False
    )
    pd.DataFrame(contrast_full_rows, columns=CONTRAST_IN_SAMPLE_COLUMNS).drop_duplicates().to_csv(
        contrasts_dir / "contrast_full_residualized_scores.csv", index=False
    )
    _write_holdout_summary(contrast_holdout_rows, contrasts_dir)


def resolve_contrasts_from_registry(
    metadata: pd.DataFrame, output_dir: Path, logger=None,
) -> list[tuple[str, str]]:
    """Audit 4.1: load registry, validate against this run's identity_ids,
    write contrasts_skipped.csv next to the contrast outputs."""
    from contrast_registry import (
        load_validated_contrasts, write_contrasts_skipped, get_contrast_pairs,
    )
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"identity_id": metadata["identity_id"].unique()}).to_csv(f.name, index=False)
        tmp_path = f.name
    try:
        result = load_validated_contrasts(tmp_path)
    finally:
        os.unlink(tmp_path)
    contrasts_dir = output_dir / "contrasts"
    contrasts_dir.mkdir(parents=True, exist_ok=True)
    write_contrasts_skipped(result.skipped, contrasts_dir / "contrasts_skipped.csv", logger=logger)
    return get_contrast_pairs(result.valid)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.activation_dir)
    layers = parse_layers(args.layers, args.activation_dir)
    selected_layers = parse_selected_layers(args.selected_layers_for_plots)
    prepare_output_dir(args.output_dir, args.overwrite, args.resume)

    global CONTRASTS
    CONTRASTS = resolve_contrasts_from_registry(metadata, args.output_dir)
    print(f"Contrast registry: {len(CONTRASTS)} pairs valid for this run.")

    if args.resume:
        (
            variance_rows,
            axis_probe_rows,
            identity_probe_rows,
            surface_probe_rows,
            contrast_full_rows,
            contrast_holdout_rows,
        ) = read_existing_layer_csvs(args.output_dir)
    else:
        variance_rows = []
        axis_probe_rows = []
        identity_probe_rows = []
        surface_probe_rows = []
        contrast_full_rows = []
        contrast_holdout_rows = []
    pca_evr_by_resid: dict[str, list[pd.DataFrame]] = {
        residualization: [] for residualization in RESIDUALIZATIONS
    }
    if args.resume:
        for residualization in RESIDUALIZATIONS:
            evr_path = (
                args.output_dir
                / "pca_residualized"
                / residualization
                / "pca_explained_variance.csv"
            )
            evr_df = safe_read_csv(evr_path)
            if evr_df is not None and not evr_df.empty:
                pca_evr_by_resid[residualization].append(evr_df)

    for layer in tqdm(layers, desc="Diagnostic layers"):
        print(f"\nLayer {layer:02d}")
        layer_start = time.perf_counter()
        start = time.perf_counter()
        x_raw = load_layer(args.activation_dir, layer, len(metadata))
        print(f"Loaded layer {layer:02d} in {elapsed(start)} shape={x_raw.shape}")

        layer_variance = float(np.var(x_raw))
        if np.isclose(layer_variance, 0.0):
            print(
                f"Layer {layer:02d} has approximately zero variance "
                f"(var={layer_variance:.3e}); skipping downstream diagnostics."
            )
            start = time.perf_counter()
            variance_rows.extend(variance_decomposition_layer(x_raw, metadata, layer))
            print(f"Variance decomposition finished in {elapsed(start)}")
            write_incremental_outputs(
                args.output_dir,
                variance_rows,
                axis_probe_rows,
                identity_probe_rows,
                surface_probe_rows,
                contrast_full_rows,
                contrast_holdout_rows,
            )
            continue

        start = time.perf_counter()
        variance_rows.extend(variance_decomposition_layer(x_raw, metadata, layer))
        print(f"Variance decomposition finished in {elapsed(start)}")

        if args.only_variance:
            write_incremental_outputs(
                args.output_dir,
                variance_rows,
                axis_probe_rows,
                identity_probe_rows,
                surface_probe_rows,
                contrast_full_rows,
                contrast_holdout_rows,
            )
            continue

        start = time.perf_counter()
        residualized_variants = {}
        for residualization, group_col in tqdm(
            RESIDUALIZATIONS.items(),
            desc=f"residualizing layer {layer:02d}",
            leave=False,
        ):
            residualized_variants[residualization] = residualize(x_raw, metadata, group_col)
        print(f"Residualization finished in {elapsed(start)}")

        for residualization, x in tqdm(
            residualized_variants.items(),
            desc=f"residualization types layer {layer:02d}",
            leave=False,
        ):
            print(f"  {residualization}")
            if not args.only_contrasts:
                start = time.perf_counter()
                pca_dir = args.output_dir / "pca_residualized" / residualization
                pca_evr_by_resid[residualization].append(
                    run_pca(
                        x,
                        metadata,
                        layer,
                        pca_dir,
                        args.pca_components,
                        args.random_seed,
                        scaling=args.scaling,
                    )
                )
                print(f"  PCA finished in {elapsed(start)} (scaling={args.scaling})")

            if not args.only_pca and not args.only_contrasts and not args.skip_probes:
                start = time.perf_counter()
                probe_features = make_probe_features(
                    x, args.probe_pca_dim, args.random_seed, f"{residualization} layer {layer}",
                    scaling=args.scaling,
                )
                if probe_features is not None:
                    axis_rows, identity_rows = run_identity_probes(
                        probe_features, metadata, layer, residualization, args
                    )
                    axis_probe_rows.extend(axis_rows)
                    identity_probe_rows.extend(identity_rows)
                    if residualization == "raw" and not args.skip_surface_form_probes:
                        surface_probe_rows.extend(
                            run_surface_probes(probe_features, metadata, layer, args)
                        )

                    if (
                        residualization == "raw"
                        and args.verify_fold_internal_pca is not None
                        and layer == args.verify_fold_internal_pca
                    ):
                        print(f"  Fold-internal-PCA verification on layer {layer:02d} (audit 2.8)")
                        verification_rows = run_fold_internal_pca_verification_diag(
                            x_raw_for_layer=x,
                            metadata=metadata,
                            layer=layer,
                            axis_probe_rows=axis_probe_rows,
                            identity_probe_rows=identity_probe_rows,
                            surface_probe_rows=surface_probe_rows,
                            args=args,
                        )
                        probes_dir = args.output_dir / "probes"
                        probes_dir.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(verification_rows).to_csv(
                            probes_dir / "pca_leakage_verification.csv", index=False
                        )
                        print(f"    wrote {probes_dir / 'pca_leakage_verification.csv'}")
                print(f"  Probes finished in {elapsed(start)}")

            if not args.only_pca and not args.only_variance:
                start = time.perf_counter()
                full_rows, holdout_rows = run_contrasts(x, metadata, layer, residualization)
                contrast_full_rows.extend(full_rows)
                contrast_holdout_rows.extend(holdout_rows)
                print(f"  Contrasts finished in {elapsed(start)}")

        for residualization, evr_frames in pca_evr_by_resid.items():
            if evr_frames:
                pd.concat(evr_frames, ignore_index=True).drop_duplicates().to_csv(
                    args.output_dir
                    / "pca_residualized"
                    / residualization
                    / "pca_explained_variance.csv",
                    index=False,
                )
        write_incremental_outputs(
            args.output_dir,
            variance_rows,
            axis_probe_rows,
            identity_probe_rows,
            surface_probe_rows,
            contrast_full_rows,
            contrast_holdout_rows,
        )
        print(f"Layer {layer:02d} complete in {elapsed(layer_start)}")

    pd.DataFrame(variance_rows).to_csv(
        args.output_dir / "variance_decomposition.csv", index=False
    )
    for residualization, evr_frames in pca_evr_by_resid.items():
        pd.concat(evr_frames, ignore_index=True).to_csv(
            args.output_dir
            / "pca_residualized"
            / residualization
            / "pca_explained_variance.csv",
            index=False,
        )

    probes_dir = args.output_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(axis_probe_rows, columns=PROBE_COLUMNS).to_csv(
        probes_dir / "axis_probe_residualized_scores.csv", index=False
    )
    pd.DataFrame(identity_probe_rows, columns=PROBE_COLUMNS).to_csv(
        probes_dir / "identity_within_axis_probe_residualized_scores.csv", index=False
    )
    pd.DataFrame(surface_probe_rows, columns=PROBE_COLUMNS).to_csv(
        probes_dir / "surface_form_probe_scores.csv", index=False
    )

    contrasts_dir = args.output_dir / "contrasts"
    contrasts_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(contrast_holdout_rows, columns=CONTRAST_HOLDOUT_COLUMNS).to_csv(
        contrasts_dir / "contrast_family_holdout_residualized_scores.csv", index=False
    )
    pd.DataFrame(contrast_full_rows, columns=CONTRAST_IN_SAMPLE_COLUMNS).to_csv(
        contrasts_dir / "contrast_full_residualized_scores.csv", index=False
    )
    _write_holdout_summary(contrast_holdout_rows, contrasts_dir)

    if not args.skip_plots:
        start = time.perf_counter()
        print("\nPlotting diagnostics")
        plot_variance_decomposition(args.output_dir)
        plot_probe_diagnostics(args.output_dir)
        plot_contrast_diagnostics(args.output_dir)
        plot_axis_specific_pca(
            args.output_dir,
            selected_layers,
            args.max_plot_points,
            args.random_seed,
        )
        print(f"Static plots finished in {elapsed(start)}")

    if args.make_umap and not args.skip_umap and not args.skip_plots:
        start = time.perf_counter()
        plot_axis_specific_umap(
            args.output_dir,
            selected_layers,
            args.max_plot_points,
            args.random_seed,
        )
        print(f"UMAP plots finished in {elapsed(start)}")
    elif args.skip_umap:
        print("UMAP skipped due to --skip_umap.")

    print("Alias/lexical holdout skipped: no alias-expanded prompts detected.")

    write_run_config(args, layers, metadata)
    print(f"\nDiagnostics complete: {args.output_dir}")


# Recommended fast command:
#
# python /workspace/status_mi/scripts/analyze_identity_geometry_diagnostics.py \
#   --activation_dir /workspace/status_mi/results/activations/llama-3.1-8b/identity_prompts_final_token/ \
#   --geometry_dir /workspace/status_mi/results/geometry/llama-3.1-8b/identity_prompts_final_token/ \
#   --output_dir /workspace/status_mi/results/geometry/llama-3.1-8b/identity_prompts_final_token/diagnostics_fast/ \
#   --layers 8,16,24,32 \
#   --selected_layers_for_plots 8,16,24,32 \
#   --skip_umap \
#   --skip_identity_within_axis_probes \
#   --skip_template_id_probe \
#   --probe_pca_dim 64 \
#   --max_probe_rows 5000 \
#   --n_splits 3 \
#   --overwrite
if __name__ == "__main__":
    main()
