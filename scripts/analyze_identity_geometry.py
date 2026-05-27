#!/usr/bin/env python3
"""Analyze identity-representation geometry from final-token activations."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler  # noqa: F401 (kept for back-compat; canonical scaler lives in common)

from common import (
    SCALING_MODES, DEFAULT_SCALING, CenterOnlyScaler, make_scaler,
    cohens_d, cosine, normalize, compute_direction, compute_direction_for_pair,
    evaluate_projection, residualize, OKABE_ITO, save_fig,
)  # noqa: E402, F401  (audit 5.10 — single source of truth)
from tqdm.auto import tqdm


DEFAULT_ACTIVATION_DIR = Path(
    "/workspace/status_mi/results/activations/llama-3.1-8b/"
    "identity_prompts_final_token"
)
DEFAULT_OUTPUT_DIR = Path(
    "/workspace/status_mi/results/geometry/llama-3.1-8b/"
    "identity_prompts_final_token"
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
AXIS_PROBE_COLUMNS = [
    "layer",
    "split_type",
    "n_classes",
    "n_samples",
    "accuracy_mean",
    "accuracy_sd",
    "macro_f1_mean",
    "macro_f1_sd",
]
IDENTITY_PROBE_COLUMNS = [
    "layer",
    "axis",
    "split_type",
    "n_classes",
    "n_samples",
    "accuracy_mean",
    "accuracy_sd",
    "macro_f1_mean",
    "macro_f1_sd",
]
CONTRAST_COLUMNS = [
    "layer",
    "contrast_name",
    "identity_a",
    "identity_b",
    "axis",
    # In-sample: direction estimated AND evaluated on the same prompts. Kept as
    # a diagnostic of how much the difference-of-means direction overfits its
    # own evaluation; the headline number is the family-holdout below. Audit 2.1.
    "auc_in_sample",
    "cohens_d_in_sample",
    "mean_a",
    "mean_b",
    "n_a",
    "n_b",
]
CONTRAST_HOLDOUT_COLUMNS = [
    "layer",
    "contrast_name",
    "identity_a",
    "identity_b",
    "heldout_family",
    "auc",
    "cohens_d",
    "n_eval_a",
    "n_eval_b",
]
FAMILY_SUMMARY_COLUMNS = [
    "layer",
    "axis",
    "mean_cosine",
    "median_cosine",
    "sd_cosine",
    "n_pairs",
]
PROJECTION_LAYERS = {0, 8, 16, 24, 32}
# Canonical contrast pairs are sourced from scripts/contrast_registry.py per
# audit 4.1 (single validated registry; typos previously in this literal —
# ses_low_income, ses_high_socioeconomic_status — are corrected in the
# registry so the SES axis runs all 4 contrasts instead of silently 2).
# CONTRASTS here is populated at main() startup after running the registry
# validator against the loaded metadata; see resolve_contrasts() below.
CONTRASTS: list[tuple[str, str]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute first-pass identity geometry summaries."
    )
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--layers",
        default=None,
        help='Optional comma-separated layers, e.g. "0,8,16,24,32".',
    )
    parser.add_argument("--max_pca_points", type=int, default=None)
    parser.add_argument("--pca_components", type=int, default=10)
    parser.add_argument("--probe_pca_dim", type=int, default=256)
    parser.add_argument(
        "--skip_probes",
        action="store_true",
        help="Skip logistic probe analyses and compute only PCA/means/stability/contrasts.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=20,
        help=(
            "Number of label-permutation null replicates per probe (audit 2.2). "
            "Default 20 keeps the run tractable; bump to >=100 for the headline number. "
            "Set 0 to disable; null fields are then NaN."
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
            "Pre-PCA scaling (audit 5.9). 'center_only' subtracts per-dim mean only; "
            "preserves the activation-space variance structure including rogue / "
            "high-norm dimensions, so explained_variance_ratio describes activation "
            "space. 'standardize' z-scores per dim — the legacy behavior — and was "
            "shown by the audit to upweight low-variance dimensions. Applies to "
            "both run_pca and make_probe_features so a run is internally consistent."
        ),
    )
    parser.add_argument(
        "--verify_fold_internal_pca",
        type=int,
        default=None,
        help=(
            "Optional layer index to verify the global-PCA design choice (audit 2.8). "
            "If set, runs each probe configuration on this layer a second time with "
            "StandardScaler + PCA fit INSIDE each CV fold (no leakage) and writes "
            "probes/pca_leakage_verification.csv comparing the two. Small "
            "accuracy/macro-F1 deltas vindicate the speed-tradeoff design."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_layer_list(layer_arg: str | None, activation_dir: Path) -> list[int]:
    all_layers = sorted(
        int(path.stem.split("_")[1]) for path in activation_dir.glob("layer_*.npy")
    )
    if not all_layers:
        raise FileNotFoundError(f"No layer_*.npy files found in {activation_dir}")

    if layer_arg is None:
        return all_layers

    selected = [int(part.strip()) for part in layer_arg.split(",") if part.strip()]
    missing = sorted(set(selected) - set(all_layers))
    if missing:
        raise FileNotFoundError(f"Requested layers not found: {missing}")
    return selected


def prepare_output_dir(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains files. Pass --overwrite to replace/add outputs."
        )

    subdirs = {
        "pca": output_dir / "pca",
        "means": output_dir / "means",
        "probes": output_dir / "probes",
        "family_stability": output_dir / "family_stability",
        "contrasts": output_dir / "contrasts",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def load_metadata(activation_dir: Path) -> pd.DataFrame:
    metadata_path = activation_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv: {metadata_path}")

    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    missing = [col for col in METADATA_COLUMNS if col not in metadata.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {missing}")
    if metadata["prompt"].astype(str).str.strip().eq("").any():
        raise ValueError("metadata.csv contains empty prompts.")
    return metadata


def layer_path(activation_dir: Path, layer: int) -> Path:
    return activation_dir / f"layer_{layer:02d}.npy"


def load_layer(activation_dir: Path, layer: int, n_rows: int) -> np.ndarray:
    path = layer_path(activation_dir, layer)
    if not path.exists():
        raise FileNotFoundError(f"Missing activation file: {path}")

    x = np.load(path, mmap_mode="r")
    if x.shape[0] != n_rows:
        raise ValueError(
            f"{path.name} has {x.shape[0]} rows, but metadata has {n_rows} rows."
        )
    if x.ndim != 2:
        raise ValueError(f"{path.name} must be 2D, got shape {x.shape}.")
    return np.asarray(x, dtype=np.float32)


def stratified_sample_indices(
    metadata: pd.DataFrame, max_points: int | None, random_seed: int
) -> np.ndarray:
    if max_points is None or max_points >= len(metadata):
        return np.arange(len(metadata))
    if max_points <= 0:
        raise ValueError("--max_pca_points must be positive when provided.")

    strata = metadata["axis"].astype(str) + "||" + metadata["family"].astype(str)
    sample = (
        metadata.assign(_stratum=strata)
        .groupby("_stratum", group_keys=False, sort=False)
        .sample(frac=1, random_state=random_seed)
        .groupby("_stratum", group_keys=False, sort=False)
        .head(max(1, int(np.ceil(max_points / strata.nunique()))))
    )
    if len(sample) > max_points:
        sample = sample.sample(n=max_points, random_state=random_seed)
    elif len(sample) < max_points:
        remaining = metadata.drop(index=sample.index)
        if not remaining.empty:
            top_up = remaining.sample(
                n=min(max_points - len(sample), len(remaining)),
                random_state=random_seed,
            )
            sample = pd.concat([sample, top_up], axis=0)
    return np.array(sorted(sample.index), dtype=int)


def run_pca(
    x: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    subdirs: dict[str, Path],
    pca_components: int,
    max_pca_points: int | None,
    random_seed: int,
    scaling: str = DEFAULT_SCALING,
) -> pd.DataFrame:
    indices = stratified_sample_indices(metadata, max_pca_points, random_seed)
    x_sample = x[indices]
    n_components = min(pca_components, x_sample.shape[0] - 1, x_sample.shape[1])
    if n_components < 1:
        raise ValueError("Need at least two rows to compute PCA.")

    x_scaled = make_scaler(scaling).fit_transform(x_sample)
    total_variance = float(np.var(x_scaled, axis=0).sum())
    if total_variance <= 0 or not np.isfinite(total_variance):
        pcs = np.zeros((len(x_sample), n_components), dtype=np.float32)
        explained_variance_ratio = np.zeros(n_components, dtype=np.float32)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pca = PCA(n_components=n_components, random_state=random_seed)
            pcs = pca.fit_transform(x_scaled)
            explained_variance_ratio = np.nan_to_num(
                pca.explained_variance_ratio_,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

    pca_df = metadata.iloc[indices][METADATA_COLUMNS].copy()
    pca_df.insert(0, "original_row_idx", indices)
    for pc_idx in range(n_components):
        pca_df[f"PC{pc_idx + 1}"] = pcs[:, pc_idx]

    pca_df.to_csv(subdirs["pca"] / f"pca_layer_{layer:02d}.csv", index=False)

    evr = pd.DataFrame(
        {
            "layer": layer,
            "pc": np.arange(1, n_components + 1),
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance": np.cumsum(
                explained_variance_ratio
            ),
        }
    )
    return evr


def save_group_means(
    x: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    group_cols: list[str],
    array_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    mean_rows = []
    meta_rows = []

    by = group_cols[0] if len(group_cols) == 1 else group_cols
    for group_key, idx in metadata.groupby(by, sort=True).groups.items():
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        idx_array = np.fromiter(idx, dtype=int)
        mean_rows.append(x[idx_array].mean(axis=0))
        meta_rows.append(dict(zip(group_cols, group_key)) | {"n_prompts": len(idx_array)})

    means = np.vstack(mean_rows).astype(np.float32)
    means_metadata = pd.DataFrame(meta_rows)
    np.save(array_path, means)
    means_metadata.to_csv(metadata_path, index=False)
    return means, means_metadata


def run_means(
    x: np.ndarray, metadata: pd.DataFrame, layer: int, subdirs: dict[str, Path]
) -> tuple[np.ndarray, pd.DataFrame]:
    save_group_means(
        x,
        metadata,
        layer,
        ["identity_id"],
        subdirs["means"] / f"identity_means_layer_{layer:02d}.npy",
        subdirs["means"] / f"identity_means_metadata_layer_{layer:02d}.csv",
    )
    identity_family_means, identity_family_meta = save_group_means(
        x,
        metadata,
        layer,
        ["identity_id", "family"],
        subdirs["means"] / f"identity_family_means_layer_{layer:02d}.npy",
        subdirs["means"] / f"identity_family_means_metadata_layer_{layer:02d}.csv",
    )
    save_group_means(
        x,
        metadata,
        layer,
        ["axis"],
        subdirs["means"] / f"axis_means_layer_{layer:02d}.npy",
        subdirs["means"] / f"axis_means_metadata_layer_{layer:02d}.csv",
    )
    return identity_family_means, identity_family_meta


def make_probe_features(
    x: np.ndarray, probe_pca_dim: int, random_seed: int, layer: int,
    scaling: str = DEFAULT_SCALING,
) -> np.ndarray | None:
    """Build fast probe features once per layer.

    This intentionally fits the unsupervised StandardScaler + PCA once on the
    *full* layer so cross-validation only tests supervised label recovery.
    Fitting inside each CV fold would multiply randomized-SVD cost by the
    number of folds × number of probe configurations × residualizations × layers.

    Audit issue 2.8 flags the technical leakage: the PCA basis is computed on
    data that include the held-out fold. PCA is unsupervised, so the leakage
    is mild in principle, but a careful reviewer will ask.

    To verify the leakage is empirically negligible on this dataset, run
    `--verify_fold_internal_pca <layer>` (default off). That triggers an extra
    pass through `crossval_probe_fold_internal_pca` which fits StandardScaler +
    PCA inside each CV fold on the train rows only, then writes
    `probes/pca_leakage_verification.csv` next to the existing probe CSVs
    with side-by-side global-PCA and fold-internal-PCA numbers. If the
    accuracy_delta is small (< the per-fold SD), the design choice is
    defensible to a reviewer.
    """
    if not np.isfinite(x).all():
        print(f"Skipping probes for layer {layer}: activations contain non-finite values.")
        return None

    total_variance = float(np.var(x, axis=0).sum())
    if total_variance <= 0 or not np.isfinite(total_variance):
        print(f"Skipping probes for layer {layer}: activations have zero variance.")
        return None

    x_scaled = make_scaler(scaling).fit_transform(x)
    if not np.isfinite(x_scaled).all():
        print(f"Skipping probes for layer {layer}: scaled activations are non-finite.")
        return None

    if probe_pca_dim and probe_pca_dim > 0:
        n_components = min(probe_pca_dim, x_scaled.shape[0] - 1, x_scaled.shape[1])
        if n_components >= 1 and n_components < x_scaled.shape[1]:
            print(f"  Probe PCA: reducing {x_scaled.shape[1]} -> {n_components} dims")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pca = PCA(
                    n_components=n_components,
                    random_state=random_seed,
                    svd_solver="randomized",
                )
                x_scaled = pca.fit_transform(x_scaled)

    return np.asarray(np.nan_to_num(x_scaled), dtype=np.float32)


def build_probe_model() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )


def _run_cv_folds(
    x: np.ndarray,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    split_type: str,
    layer: int,
    verbose: bool = True,
) -> tuple[list[float], list[float]]:
    """Inner CV loop. Returns (accuracies, macro_f1s) over the provided folds.
    Reused for both the observed probe and each permutation null replicate."""
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for train_idx, test_idx in splits:
            if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
                continue
            x_train = x[train_idx]
            x_test = x[test_idx]
            if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
                if verbose:
                    print(f"Skipping non-finite probe fold ({split_type}, layer {layer}).")
                continue
            if np.all(np.std(x_train, axis=0) == 0):
                if verbose:
                    print(f"Skipping zero-variance probe fold ({split_type}, layer {layer}).")
                continue
            model = build_probe_model()
            try:
                model.fit(x_train, y.iloc[train_idx])
                pred = model.predict(x_test)
                accuracies.append(accuracy_score(y.iloc[test_idx], pred))
                macro_f1s.append(f1_score(y.iloc[test_idx], pred, average="macro"))
            except Exception as exc:
                if verbose:
                    print(f"Skipping failed probe fold ({split_type}, layer {layer}): {exc}")
                continue
    return accuracies, macro_f1s


def crossval_probe_fold_internal_pca(
    x_raw: np.ndarray,
    y: pd.Series,
    groups: pd.Series,
    probe_pca_dim: int,
    random_seed: int,
    split_type: str,
    layer: int,
    scaling: str = DEFAULT_SCALING,
) -> dict[str, float | int | str] | None:
    """Audit-2.8 verifier: same GroupKFold probe as `crossval_probe` but fits
    StandardScaler + PCA inside each fold on the train rows only.

    Used when `--verify_fold_internal_pca <layer>` is set, to confirm the
    fast in-place global-PCA path (`make_probe_features`) produces equivalent
    numbers. No null distribution is computed here — the comparison is
    accuracy / macro-F1 only.
    """
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    n_classes = y.nunique()
    n_groups = groups.nunique()
    if n_classes < 2 or n_groups < 2 or not np.isfinite(x_raw).all():
        return None
    n_splits = min(5, n_groups)
    try:
        splits = list(GroupKFold(n_splits=n_splits).split(x_raw, y, groups))
    except ValueError as exc:
        print(f"Skipping fold-internal-PCA split ({split_type}, layer {layer}): {exc}")
        return None
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
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
            try:
                model = build_probe_model()
                model.fit(x_tr, y.iloc[train_idx])
                pred = model.predict(x_te)
                accuracies.append(accuracy_score(y.iloc[test_idx], pred))
                macro_f1s.append(f1_score(y.iloc[test_idx], pred, average="macro"))
            except Exception as exc:
                print(f"Skipping failed fold-internal-PCA fold ({split_type}, layer {layer}): {exc}")
                continue
    if not accuracies:
        return None
    return {
        "layer": layer,
        "split_type": split_type,
        "n_classes": int(n_classes),
        "n_samples": int(len(y)),
        "fold_internal_pca_accuracy_mean": float(np.mean(accuracies)),
        "fold_internal_pca_accuracy_sd": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "fold_internal_pca_macro_f1_mean": float(np.mean(macro_f1s)),
        "fold_internal_pca_macro_f1_sd": float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0,
    }


def _empty_null_fields() -> dict[str, float | int]:
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
    split_type: str,
    layer: int,
    n_permutations: int = 0,
    null_rng_seed: int = 42,
) -> dict[str, float | int | str] | None:
    """Probe a feature matrix on labels with GroupKFold cross-validation, and
    optionally compute a label-permutation null (audit issue 2.2).

    When n_permutations > 0, y is globally shuffled `n_permutations` times
    (with the GroupKFold structure preserved across all replicates). The
    observed accuracy / macro-F1 are reported alongside null mean / SD,
    a z-score, and a Phipson-Smyth smoothed empirical p-value (the
    fraction of null replicates that meet or exceed the observed value,
    plus a +1 / +1 smoothing so p > 0 even with all-null-below).
    """
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    n_classes = y.nunique()
    n_groups = groups.nunique()
    if n_classes < 2 or n_groups < 2:
        return None

    n_splits = min(5, n_groups)
    splitter = GroupKFold(n_splits=n_splits)
    try:
        splits = list(splitter.split(x, y, groups))
    except ValueError as exc:
        print(f"Skipping probe split ({split_type}, layer {layer}): {exc}")
        return None

    accuracies, macro_f1s = _run_cv_folds(x, y, splits, split_type, layer, verbose=True)
    if not accuracies:
        return None

    obs_acc = float(np.mean(accuracies))
    obs_f1 = float(np.mean(macro_f1s))

    result: dict[str, float | int | str] = {
        "layer": layer,
        "split_type": split_type,
        "n_classes": int(n_classes),
        "n_samples": int(len(y)),
        "accuracy_mean": obs_acc,
        "accuracy_sd": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "macro_f1_mean": obs_f1,
        "macro_f1_sd": float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0,
    }
    result.update(_empty_null_fields())

    if n_permutations > 0:
        rng = np.random.default_rng(null_rng_seed)
        y_arr = y.to_numpy()
        null_accs: list[float] = []
        null_f1s: list[float] = []
        for _ in range(n_permutations):
            y_perm = pd.Series(rng.permutation(y_arr))
            perm_accs, perm_f1s = _run_cv_folds(
                x, y_perm, splits, split_type, layer, verbose=False
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
            result["null_n_permutations"] = int(n_obs)
            result["null_accuracy_mean"] = null_acc_mean
            result["null_accuracy_sd"] = null_acc_sd
            result["null_macro_f1_mean"] = null_f1_mean
            result["null_macro_f1_sd"] = null_f1_sd
            result["accuracy_z"] = (obs_acc - null_acc_mean) / (null_acc_sd + 1e-12)
            result["macro_f1_z"] = (obs_f1 - null_f1_mean) / (null_f1_sd + 1e-12)
            # Phipson-Smyth smoothed empirical p (Phipson & Smyth 2010).
            result["accuracy_p_value"] = float(
                (1 + (null_acc_arr >= obs_acc).sum()) / (1 + n_obs)
            )
            result["macro_f1_p_value"] = float(
                (1 + (null_f1_arr >= obs_f1).sum()) / (1 + n_obs)
            )

    return result


def run_probes(
    x: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    probe_pca_dim: int,
    random_seed: int,
    n_permutations: int = 0,
    null_rng_seed: int = 42,
    scaling: str = DEFAULT_SCALING,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    x_probe = make_probe_features(x, probe_pca_dim, random_seed, layer, scaling=scaling)
    if x_probe is None:
        return [], []

    axis_rows = []
    for group_col in ["template_id", "family"]:
        try:
            result = crossval_probe(
                x=x_probe,
                y=metadata["axis"],
                groups=metadata[group_col],
                split_type=f"group_by_{group_col}",
                layer=layer,
                n_permutations=n_permutations,
                null_rng_seed=null_rng_seed,
            )
        except Exception as exc:
            print(f"Skipping axis probe ({group_col}, layer {layer}): {exc}")
            result = None
        if result:
            axis_rows.append(result)

    identity_rows = []
    for axis, axis_meta in metadata.groupby("axis", sort=True):
        if axis_meta["identity_id"].nunique() < 2:
            continue
        idx = axis_meta.index.to_numpy()
        try:
            result = crossval_probe(
                x=x_probe[idx],
                y=axis_meta["identity_id"].reset_index(drop=True),
                groups=axis_meta["template_id"].reset_index(drop=True),
                split_type="group_by_template_id",
                layer=layer,
                n_permutations=n_permutations,
                null_rng_seed=null_rng_seed,
            )
        except Exception as exc:
            print(f"Skipping identity-within-axis probe ({axis}, layer {layer}): {exc}")
            result = None
        if result:
            result["axis"] = axis
            identity_rows.append(result)

    return axis_rows, identity_rows


def run_fold_internal_pca_verification(
    *,
    x: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    axis_rows: list[dict[str, object]],
    identity_rows: list[dict[str, object]],
    probe_pca_dim: int,
    random_seed: int,
    scaling: str = DEFAULT_SCALING,
) -> list[dict[str, object]]:
    """For the layer chosen by --verify_fold_internal_pca, re-run each probe
    configuration with StandardScaler + PCA fit inside each CV fold and pair
    those numbers up with the corresponding global-PCA rows.

    Returns one row per (split_type [, axis]) with columns:
        layer, split_type, axis, n_classes, n_samples,
        global_pca_accuracy_mean / sd, global_pca_macro_f1_mean / sd,
        fold_internal_pca_accuracy_mean / sd, fold_internal_pca_macro_f1_mean / sd,
        accuracy_delta, macro_f1_delta
    """
    rows: list[dict[str, object]] = []
    by_split = {row["split_type"]: row for row in axis_rows if row.get("layer") == layer}
    for group_col in ["template_id", "family"]:
        split_name = f"group_by_{group_col}"
        global_row = by_split.get(split_name)
        verification = crossval_probe_fold_internal_pca(
            x_raw=x,
            y=metadata["axis"],
            groups=metadata[group_col],
            probe_pca_dim=probe_pca_dim,
            random_seed=random_seed,
            split_type=split_name,
            layer=layer,
            scaling=scaling,
        )
        if global_row is None or verification is None:
            continue
        rows.append(_pair_global_and_fold_internal(global_row, verification, axis=""))

    by_axis = {row.get("axis"): row for row in identity_rows if row.get("layer") == layer}
    for axis_name, axis_meta in metadata.groupby("axis", sort=True):
        if axis_meta["identity_id"].nunique() < 2:
            continue
        idx = axis_meta.index.to_numpy()
        global_row = by_axis.get(axis_name)
        verification = crossval_probe_fold_internal_pca(
            x_raw=x[idx],
            y=axis_meta["identity_id"].reset_index(drop=True),
            groups=axis_meta["template_id"].reset_index(drop=True),
            probe_pca_dim=probe_pca_dim,
            random_seed=random_seed,
            split_type="group_by_template_id",
            layer=layer,
            scaling=scaling,
        )
        if global_row is None or verification is None:
            continue
        rows.append(_pair_global_and_fold_internal(global_row, verification, axis=axis_name))
    return rows


def _pair_global_and_fold_internal(
    global_row: dict[str, object], verification: dict[str, object], axis: str
) -> dict[str, object]:
    return {
        "layer": global_row["layer"],
        "split_type": global_row["split_type"],
        "axis": axis,
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


def run_family_stability(
    x: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    subdirs: dict[str, Path],
) -> pd.DataFrame:
    global_mean = x.mean(axis=0)
    rows = []

    grouped = metadata.groupby(["identity_id", "family"], sort=True).groups
    means = {
        key: x[np.fromiter(idx, dtype=int)].mean(axis=0)
        for key, idx in grouped.items()
    }
    axis_lookup = metadata.groupby("identity_id")["axis"].first().to_dict()

    for identity_id in sorted(metadata["identity_id"].unique()):
        family_means = {
            family: means[(identity_id, family)]
            for (ident, family) in means
            if ident == identity_id
        }
        if len(family_means) < 2:
            continue
        for family_a, family_b in combinations(sorted(family_means), 2):
            vec_a = family_means[family_a]
            vec_b = family_means[family_b]
            rows.append(
                {
                    "layer": layer,
                    "identity_id": identity_id,
                    "axis": axis_lookup.get(identity_id, ""),
                    "family_a": family_a,
                    "family_b": family_b,
                    "raw_cosine": cosine(vec_a, vec_b),
                    "centered_cosine": cosine(vec_a - global_mean, vec_b - global_mean),
                }
            )

    family_df = pd.DataFrame(rows)
    family_df.to_csv(
        subdirs["family_stability"] / f"family_cosines_layer_{layer:02d}.csv",
        index=False,
    )
    return family_df


def resolve_contrasts_from_registry(
    metadata: pd.DataFrame,
    subdirs: dict[str, Path],
    logger=None,
) -> list[tuple[str, str]]:
    """Load the canonical contrast registry, validate against the loaded
    metadata's identity_ids, write contrasts_skipped.csv to the contrasts
    subdir, and return 2-tuple (identity_a, identity_b) pairs ready for
    run_contrasts. Audit 4.1."""
    from contrast_registry import (
        load_validated_contrasts, write_contrasts_skipped, get_contrast_pairs,
    )
    # Use the metadata's identity set as the validation oracle. We write a
    # temp CSV with just the identity_id column for the loader.
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"identity_id": metadata["identity_id"].unique()}).to_csv(f.name, index=False)
        tmp_path = f.name
    try:
        result = load_validated_contrasts(tmp_path)
    finally:
        os.unlink(tmp_path)
    skipped_path = subdirs["contrasts"] / "contrasts_skipped.csv"
    write_contrasts_skipped(result.skipped, skipped_path, logger=logger)
    return get_contrast_pairs(result.valid)


def run_contrasts(
    x: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    subdirs: dict[str, Path],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    x_centered = x - x.mean(axis=0, keepdims=True)
    identity_set = set(metadata["identity_id"].unique())
    score_rows = []
    holdout_rows = []
    projection_rows = []
    axis_lookup = metadata.groupby("identity_id")["axis"].first().to_dict()

    for identity_a, identity_b in CONTRASTS:
        if identity_a not in identity_set or identity_b not in identity_set:
            continue

        mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
        mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
        # common.compute_direction sign-flips so projection of A has higher
        # mean than projection of B. x_centered is already mean-zero, so
        # we pass center=False to avoid double-centering. Audit 5.10.
        cd = compute_direction(x_centered, mask_a, mask_b, center=False)
        if cd is None:
            continue
        direction = cd.direction

        scores = x_centered @ direction
        metrics = evaluate_projection(scores, mask_a, mask_b)
        contrast_name = f"{identity_a}_vs_{identity_b}"
        score_rows.append(
            {
                "layer": layer,
                "contrast_name": contrast_name,
                "identity_a": identity_a,
                "identity_b": identity_b,
                "axis": axis_lookup.get(identity_a, ""),
                "auc_in_sample": metrics["auc"],
                "cohens_d_in_sample": metrics["cohens_d"],
                "mean_a": metrics["mean_a"],
                "mean_b": metrics["mean_b"],
                "n_a": int(mask_a.sum()),
                "n_b": int(mask_b.sum()),
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

            heldout_cd = compute_direction(x_centered, train_mask_a, train_mask_b, center=False)
            if heldout_cd is None:
                continue
            heldout_direction = heldout_cd.direction
            heldout_scores = x_centered @ heldout_direction
            heldout_metrics = evaluate_projection(heldout_scores, eval_mask_a, eval_mask_b)
            heldout_auc = heldout_metrics["auc"]
            heldout_d = heldout_metrics["cohens_d"]
            holdout_rows.append(
                {
                    "layer": layer,
                    "contrast_name": contrast_name,
                    "identity_a": identity_a,
                    "identity_b": identity_b,
                    "heldout_family": heldout_family,
                    "auc": heldout_auc,
                    "cohens_d": heldout_d,
                    "n_eval_a": int(eval_mask_a.sum()),
                    "n_eval_b": int(eval_mask_b.sum()),
                }
            )

        if layer in PROJECTION_LAYERS:
            projection_meta = metadata[
                ["prompt_id", "identity_id", "axis", "family", "template_id"]
            ].copy()
            projection_meta["contrast_name"] = contrast_name
            projection_meta["projection_score"] = scores
            projection_rows.append(projection_meta)

    if projection_rows:
        pd.concat(projection_rows, ignore_index=True).to_csv(
            subdirs["contrasts"] / f"contrast_projection_scores_layer_{layer:02d}.csv",
            index=False,
        )
    return score_rows, holdout_rows


def write_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    metadata: pd.DataFrame,
    layers: list[int],
    hidden_dim: int,
) -> None:
    run_config = {
        "activation_dir": str(args.activation_dir),
        "output_dir": str(output_dir),
        "layers": layers,
        "max_pca_points": args.max_pca_points,
        "pca_components": args.pca_components,
        "probe_pca_dim": args.probe_pca_dim,
        "skip_probes": args.skip_probes,
        "random_seed": args.random_seed,
        "scaling": args.scaling,
        "n_permutations": args.n_permutations,
        "num_rows": len(metadata),
        "hidden_dim": hidden_dim,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")


def write_rows_csv(rows: list[dict[str, object]], columns: list[str], path: Path) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if args.pca_components <= 0:
        raise ValueError("--pca_components must be positive.")

    metadata = load_metadata(args.activation_dir)
    layers = parse_layer_list(args.layers, args.activation_dir)
    subdirs = prepare_output_dir(args.output_dir, args.overwrite)

    # Audit 4.1: load contrast registry, validate against this run's
    # identity_ids, write contrasts_skipped.csv next to the contrast outputs.
    global CONTRASTS
    CONTRASTS = resolve_contrasts_from_registry(metadata, subdirs)
    print(f"Contrast registry: {len(CONTRASTS)} pairs valid for this run.")

    pca_evr_rows = []
    axis_probe_rows = []
    identity_probe_rows = []
    family_summary_inputs = []
    contrast_rows = []
    contrast_holdout_rows = []
    hidden_dim = None

    for layer in tqdm(layers, desc="Analyzing layers"):
        print(f"\nLayer {layer:02d}")
        x = load_layer(args.activation_dir, layer, len(metadata))
        hidden_dim = x.shape[1]

        print(f"  PCA (scaling={args.scaling})")
        pca_evr_rows.append(
            run_pca(
                x=x,
                metadata=metadata,
                layer=layer,
                subdirs=subdirs,
                pca_components=args.pca_components,
                max_pca_points=args.max_pca_points,
                random_seed=args.random_seed,
                scaling=args.scaling,
            )
        )

        print("  Means")
        run_means(x=x, metadata=metadata, layer=layer, subdirs=subdirs)

        if args.skip_probes:
            print("  Probes skipped")
        else:
            print(f"  Probes (null n_permutations={args.n_permutations}, scaling={args.scaling})")
            null_seed = args.null_random_seed if args.null_random_seed is not None else args.random_seed
            axis_rows, identity_rows = run_probes(
                x=x,
                metadata=metadata,
                layer=layer,
                probe_pca_dim=args.probe_pca_dim,
                random_seed=args.random_seed,
                n_permutations=args.n_permutations,
                null_rng_seed=null_seed,
                scaling=args.scaling,
            )
            axis_probe_rows.extend(axis_rows)
            identity_probe_rows.extend(identity_rows)

            if args.verify_fold_internal_pca is not None and layer == args.verify_fold_internal_pca:
                print(f"  Fold-internal PCA verification on layer {layer:02d} (audit 2.8)")
                verification_rows = run_fold_internal_pca_verification(
                    x=x,
                    metadata=metadata,
                    layer=layer,
                    axis_rows=axis_rows,
                    identity_rows=identity_rows,
                    probe_pca_dim=args.probe_pca_dim,
                    random_seed=args.random_seed,
                    scaling=args.scaling,
                )
                pd.DataFrame(verification_rows).to_csv(
                    subdirs["probes"] / "pca_leakage_verification.csv", index=False
                )
                print(f"    wrote {subdirs['probes'] / 'pca_leakage_verification.csv'}")

        print("  Family stability")
        family_df = run_family_stability(
            x=x, metadata=metadata, layer=layer, subdirs=subdirs
        )
        family_summary_inputs.append(family_df)

        print("  Contrasts")
        layer_contrast_rows, layer_holdout_rows = run_contrasts(
            x=x, metadata=metadata, layer=layer, subdirs=subdirs
        )
        contrast_rows.extend(layer_contrast_rows)
        contrast_holdout_rows.extend(layer_holdout_rows)

    pd.concat(pca_evr_rows, ignore_index=True).to_csv(
        subdirs["pca"] / "pca_explained_variance.csv", index=False
    )
    write_rows_csv(
        axis_probe_rows,
        AXIS_PROBE_COLUMNS,
        subdirs["probes"] / "axis_probe_scores.csv",
    )
    write_rows_csv(
        identity_probe_rows,
        IDENTITY_PROBE_COLUMNS,
        subdirs["probes"] / "identity_within_axis_probe_scores.csv",
    )

    if family_summary_inputs:
        family_all = pd.concat(family_summary_inputs, ignore_index=True)
        family_summary = (
            family_all.groupby(["layer", "axis"], sort=True)["centered_cosine"]
            .agg(
                mean_cosine="mean",
                median_cosine="median",
                sd_cosine="std",
                n_pairs="count",
            )
            .reset_index()
        )
    else:
        family_summary = pd.DataFrame(columns=FAMILY_SUMMARY_COLUMNS)
    family_summary.to_csv(
        subdirs["family_stability"] / "family_cosines_summary.csv", index=False
    )

    write_rows_csv(
        contrast_rows,
        CONTRAST_COLUMNS,
        subdirs["contrasts"] / "contrast_scores.csv",
    )
    write_rows_csv(
        contrast_holdout_rows,
        CONTRAST_HOLDOUT_COLUMNS,
        subdirs["contrasts"] / "contrast_family_holdout_scores.csv",
    )

    # Headline held-out summary: one row per (layer, contrast) with mean / sd
    # / min / max / n_families across the held-out-family replicates. This is
    # what downstream plotting and the methods writeup should cite (audit 2.1).
    if contrast_holdout_rows:
        holdout_df = pd.DataFrame(contrast_holdout_rows)
        summary = (
            holdout_df.groupby(["layer", "contrast_name", "identity_a", "identity_b"], sort=True)
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
            subdirs["contrasts"] / "contrast_holdout_summary.csv", index=False
        )

    assert hidden_dim is not None
    write_run_config(args, args.output_dir, metadata, layers, hidden_dim)
    print(f"\nAnalysis complete: {args.output_dir}")


if __name__ == "__main__":
    main()
