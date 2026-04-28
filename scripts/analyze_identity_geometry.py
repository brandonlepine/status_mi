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
from sklearn.preprocessing import StandardScaler
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
    "auc_all",
    "cohens_d_all",
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
CONTRASTS = [
    ("race_black", "race_white"),
    ("sexuality_gay", "sexuality_straight"),
    ("sexuality_gay", "sexuality_heterosexual"),
    ("sexuality_lesbian", "sexuality_straight"),
    ("sexuality_bisexual", "sexuality_straight"),
    ("disability_disabled", "disability_nondisabled"),
    ("disability_disabled", "disability_able_bodied"),
    ("appearance_short", "appearance_tall"),
    ("appearance_obese", "appearance_thin"),
    ("appearance_poorly_dressed", "appearance_well_dressed"),
    ("ses_low_income", "ses_rich"),
    ("ses_low_income", "ses_high_socioeconomic_status"),
    ("ses_lower_class", "ses_upper_class"),
    ("ses_blue_collar", "ses_white_collar"),
    ("gender_transgender", "gender_cisgender"),
    ("gender_transgender_man", "gender_cisgender_man"),
    ("gender_transgender_woman", "gender_cisgender_woman"),
]


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
) -> pd.DataFrame:
    indices = stratified_sample_indices(metadata, max_pca_points, random_seed)
    x_sample = x[indices]
    n_components = min(pca_components, x_sample.shape[0] - 1, x_sample.shape[1])
    if n_components < 1:
        raise ValueError("Need at least two rows to compute PCA.")

    x_scaled = StandardScaler().fit_transform(x_sample)
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
    x: np.ndarray, probe_pca_dim: int, random_seed: int, layer: int
) -> np.ndarray | None:
    """Build fast probe features once per layer.

    This intentionally fits the unsupervised scaler/PCA once on the full layer so
    cross-validation only tests supervised label recovery. It avoids rerunning a
    costly randomized SVD for every probe fold.
    """
    if not np.isfinite(x).all():
        print(f"Skipping probes for layer {layer}: activations contain non-finite values.")
        return None

    total_variance = float(np.var(x, axis=0).sum())
    if total_variance <= 0 or not np.isfinite(total_variance):
        print(f"Skipping probes for layer {layer}: activations have zero variance.")
        return None

    x_scaled = StandardScaler().fit_transform(x)
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


def crossval_probe(
    x: np.ndarray,
    y: pd.Series,
    groups: pd.Series,
    split_type: str,
    layer: int,
) -> dict[str, float | int | str] | None:
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
    accuracies = []
    macro_f1s = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for train_idx, test_idx in splits:
            if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
                continue
            x_train = x[train_idx]
            x_test = x[test_idx]
            if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
                print(
                    f"Skipping non-finite probe fold ({split_type}, layer {layer})."
                )
                continue
            if np.all(np.std(x_train, axis=0) == 0):
                print(f"Skipping zero-variance probe fold ({split_type}, layer {layer}).")
                continue
            model = build_probe_model()
            try:
                model.fit(x_train, y.iloc[train_idx])
                pred = model.predict(x_test)
                accuracies.append(accuracy_score(y.iloc[test_idx], pred))
                macro_f1s.append(f1_score(y.iloc[test_idx], pred, average="macro"))
            except Exception as exc:
                print(f"Skipping failed probe fold ({split_type}, layer {layer}): {exc}")
                continue

    if not accuracies:
        return None

    return {
        "layer": layer,
        "split_type": split_type,
        "n_classes": int(n_classes),
        "n_samples": int(len(y)),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_sd": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_sd": float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0,
    }


def run_probes(
    x: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    probe_pca_dim: int,
    random_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    x_probe = make_probe_features(x, probe_pca_dim, random_seed, layer)
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
            )
        except Exception as exc:
            print(f"Skipping identity-within-axis probe ({axis}, layer {layer}): {exc}")
            result = None
        if result:
            result["axis"] = axis
            identity_rows.append(result)

    return axis_rows, identity_rows


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


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


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if len(scores_a) < 2 or len(scores_b) < 2:
        return float("nan")
    pooled_var = (
        ((len(scores_a) - 1) * np.var(scores_a, ddof=1))
        + ((len(scores_b) - 1) * np.var(scores_b, ddof=1))
    ) / (len(scores_a) + len(scores_b) - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((np.mean(scores_a) - np.mean(scores_b)) / np.sqrt(pooled_var))


def contrast_direction(x_centered: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray | None:
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None
    d = x_centered[mask_a].mean(axis=0) - x_centered[mask_b].mean(axis=0)
    norm = np.linalg.norm(d)
    if norm == 0:
        return None
    return d / norm


def evaluate_contrast_scores(scores: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float, float, float]:
    pair_mask = mask_a | mask_b
    y = mask_a[pair_mask].astype(int)
    pair_scores = scores[pair_mask]
    auc = float(roc_auc_score(y, pair_scores)) if len(np.unique(y)) == 2 else float("nan")
    scores_a = scores[mask_a]
    scores_b = scores[mask_b]
    return auc, cohens_d(scores_a, scores_b), float(np.mean(scores_a)), float(np.mean(scores_b))


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
        direction = contrast_direction(x_centered, mask_a, mask_b)
        if direction is None:
            continue

        scores = x_centered @ direction
        auc, d_all, mean_a, mean_b = evaluate_contrast_scores(scores, mask_a, mask_b)
        contrast_name = f"{identity_a}_vs_{identity_b}"
        score_rows.append(
            {
                "layer": layer,
                "contrast_name": contrast_name,
                "identity_a": identity_a,
                "identity_b": identity_b,
                "axis": axis_lookup.get(identity_a, ""),
                "auc_all": auc,
                "cohens_d_all": d_all,
                "mean_a": mean_a,
                "mean_b": mean_b,
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

            heldout_direction = contrast_direction(x_centered, train_mask_a, train_mask_b)
            if heldout_direction is None:
                continue
            heldout_scores = x_centered @ heldout_direction
            heldout_auc, heldout_d, _, _ = evaluate_contrast_scores(
                heldout_scores, eval_mask_a, eval_mask_b
            )
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

        print("  PCA")
        pca_evr_rows.append(
            run_pca(
                x=x,
                metadata=metadata,
                layer=layer,
                subdirs=subdirs,
                pca_components=args.pca_components,
                max_pca_points=args.max_pca_points,
                random_seed=args.random_seed,
            )
        )

        print("  Means")
        run_means(x=x, metadata=metadata, layer=layer, subdirs=subdirs)

        if args.skip_probes:
            print("  Probes skipped")
        else:
            print("  Probes")
            axis_rows, identity_rows = run_probes(
                x=x,
                metadata=metadata,
                layer=layer,
                probe_pca_dim=args.probe_pca_dim,
                random_seed=args.random_seed,
            )
            axis_probe_rows.extend(axis_rows)
            identity_probe_rows.extend(identity_rows)

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

    assert hidden_dim is not None
    write_run_config(args, args.output_dir, metadata, layers, hidden_dim)
    print(f"\nAnalysis complete: {args.output_dir}")


if __name__ == "__main__":
    main()
