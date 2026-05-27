#!/usr/bin/env python3
"""Analyze SAE features associated with identity directions."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


DEFAULT_SAE_ENCODED_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token")
DEFAULT_ACTIVATION_DIR = Path("/workspace/status_mi/results/activations/llama-3.1-8b/identity_prompts_final_token")
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/analysis")
# Canonical contrast registry — see scripts/contrast_registry.py. Audit 4.1.
from contrast_registry import CONTRASTS as DEFAULT_CONTRASTS  # noqa: E402
RESIDUALIZATION_GROUPS = {
    "raw": None,
    "family_residualized": "family",
    "template_residualized": "template_id",
    "required_form_residualized": "required_form",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze identity-selective SAE features and decoder alignments.")
    parser.add_argument("--sae_encoded_dir", type=Path, default=DEFAULT_SAE_ENCODED_DIR)
    parser.add_argument("--activation_dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--contrasts_csv", type=Path, default=None)
    parser.add_argument("--top_n_features", type=int, default=100)
    parser.add_argument("--top_k_reconstruction_values", default="5,10,20,50,100,200")
    parser.add_argument("--residualization", default="family_residualized")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists and is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not path.exists():
            pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_contrasts(path: Path | None, metadata: pd.DataFrame, output_dir: Path | None = None) -> pd.DataFrame:
    """Load + validate the contrast registry against this run's identity set.

    When `path` is None, the canonical registry (`contrast_registry.CONTRASTS`)
    is used. When `path` is given, the CSV must have the same 4 columns.
    Per-row warnings emit for each skipped pair, and a `contrasts_skipped.csv`
    sidecar is written next to the analysis outputs (audit 4.1; no startup
    assertion so partial-axis runs work).
    """
    from contrast_registry import (
        CONTRASTS as _REGISTRY,
        load_validated_contrasts,
        write_contrasts_skipped,
    )
    if path is None:
        registry_rows = _REGISTRY
    else:
        custom = pd.read_csv(path, keep_default_na=False)
        registry_rows = [tuple(row[c] for c in ("contrast_name", "identity_a", "identity_b", "axis"))
                         for _, row in custom.iterrows()]
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"identity_id": metadata["identity_id"].unique()}).to_csv(f.name, index=False)
        tmp_path = f.name
    try:
        result = load_validated_contrasts(tmp_path, registry=registry_rows)
    finally:
        os.unlink(tmp_path)
    if output_dir is not None:
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        write_contrasts_skipped(result.skipped, analysis_dir / "contrasts_skipped.csv")
    return result.valid.reset_index(drop=True)


# Audit 5.10: shared utilities imported from common; local adapters preserve
# this script's prior return-tuple shapes.
import sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from common import (  # noqa: E402
    cohens_d, cosine, normalize,
    compute_direction as _compute_direction_lowlevel,
    evaluate_projection as _evaluate_projection_canonical,
    residualize as _residualize_by_column,
)


def residualize(x: np.ndarray, metadata: pd.DataFrame, residualization: str) -> np.ndarray:
    """Adapter that maps residualization NAME -> column via RESIDUALIZATION_GROUPS.
    Routes to common.residualize. Audit 5.10."""
    return _residualize_by_column(x, metadata, RESIDUALIZATION_GROUPS[residualization])


def compute_direction(x: np.ndarray, metadata: pd.DataFrame, identity_a: str, identity_b: str) -> tuple[np.ndarray | None, np.ndarray]:
    """Adapter — calls common.compute_direction and returns this script's
    2-tuple (direction, global_mean) shape. Audit 5.10."""
    global_mean = x.mean(axis=0, keepdims=True)
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
    cd = _compute_direction_lowlevel(x, mask_a, mask_b, center=True, center_mean=global_mean)
    if cd is None:
        return None, global_mean.astype(np.float32)
    return cd.direction.astype(np.float32), global_mean.astype(np.float32)


def evaluate_direction(x: np.ndarray, metadata: pd.DataFrame, identity_a: str, identity_b: str, direction: np.ndarray, global_mean: np.ndarray) -> tuple[float, float]:
    """Adapter — projects then routes to common.evaluate_projection; returns
    the 2-tuple (auc, cohens_d) this script's callers use. Audit 5.10."""
    scores = (x - global_mean) @ direction
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
    metrics = _evaluate_projection_canonical(scores, mask_a, mask_b)
    return metrics["auc"], metrics["cohens_d"]


def find_topk_files(layer_dir: Path) -> tuple[Path, Path, int]:
    index_files = sorted(layer_dir.glob("feature_indices_top*.npy"))
    if not index_files:
        raise FileNotFoundError(f"No feature_indices_top*.npy found in {layer_dir}")
    index_path = index_files[-1]
    top_k = int(index_path.stem.split("top")[-1])
    value_path = layer_dir / f"feature_values_top{top_k}.npy"
    if not value_path.exists():
        raise FileNotFoundError(f"Missing matching values file: {value_path}")
    return index_path, value_path, top_k


def sparse_long(indices: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
    df = pd.DataFrame({
        "row": rows.astype(np.int32),
        "feature_id": indices.reshape(-1).astype(np.int64),
        "activation": values.reshape(-1).astype(np.float32),
    })
    return df[df["activation"] > 0].copy()


def dense_feature_values(long_df: pd.DataFrame, feature_id: int, n_rows: int) -> np.ndarray:
    vals = np.zeros(n_rows, dtype=np.float32)
    feature_df = long_df[long_df["feature_id"].eq(feature_id)]
    vals[feature_df["row"].to_numpy()] = feature_df["activation"].to_numpy()
    return vals


def summarize_feature_groups(long_df: pd.DataFrame, mask: np.ndarray, n_features: int, prefix: str) -> pd.DataFrame:
    rows = np.flatnonzero(mask)
    subset = long_df[long_df["row"].isin(rows)]
    activation = subset["activation"].to_numpy()
    grouped = subset.assign(activation_sq=activation * activation).groupby("feature_id").agg(
        **{
            f"sum_{prefix}": ("activation", "sum"),
            f"nonzero_{prefix}": ("activation", "count"),
            f"mean_nonzero_{prefix}": ("activation", "mean"),
            f"sum_sq_{prefix}": ("activation_sq", "sum"),
        }
    ).reset_index()
    out = pd.DataFrame({"feature_id": np.arange(n_features, dtype=np.int64)})
    out = out.merge(grouped, on="feature_id", how="left").fillna(0)
    out[f"mean_{prefix}"] = out[f"sum_{prefix}"] / max(1, int(mask.sum()))
    out[f"freq_{prefix}"] = out[f"nonzero_{prefix}"] / max(1, int(mask.sum()))
    return out


def compute_cohens_d_and_auc_for_all_features(
    long_df: pd.DataFrame,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    df_groups: pd.DataFrame,
    prefix_a: str,
    prefix_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Cohen's d and AUC for every row of df_groups (i.e. every feature).

    Closes audit 2.5 (winner's-curse). The previous code applied a top-(5·k)
    |diff_mean| prefilter before computing per-feature d/AUC, which conditions
    the reported effect sizes on survival of that screen. Here we compute both
    metrics analytically and on a sparse-aware basis for ALL features:

    Cohen's d (sample variance, ddof=1; matches common.cohens_d):
        var = (sum_sq - n·mean²) / (n - 1)
        pooled = ((n_a - 1)·var_a + (n_b - 1)·var_b) / (n_a + n_b - 2)
        d = (mean_a - mean_b) / sqrt(pooled), NaN when pooled <= 0 or n < 2.

    AUC, exploiting the fact that long_df only stores positive activations:
        Buckets over pairs (i ∈ a, j ∈ b):
          B1: both zero  → tie  → 0.5 each
          B2: a > 0, b = 0      → 1   each (positive > 0)
          B3: a = 0, b > 0      → 0   each
          B4: a > 0, b > 0      → direct comparison of nonzero arrays

        For features with B4 empty (k_a = 0 XOR k_b = 0 XOR both 0), AUC has
        a closed form in (k_a, k_b, n_a, n_b). For B4-active features we
        materialize the small nonzero arrays once via groupby and compare.
    """
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    n_features = len(df_groups)

    sum_a = df_groups[f"sum_{prefix_a}"].to_numpy(dtype=np.float64)
    sum_b = df_groups[f"sum_{prefix_b}"].to_numpy(dtype=np.float64)
    sum_sq_a = df_groups[f"sum_sq_{prefix_a}"].to_numpy(dtype=np.float64)
    sum_sq_b = df_groups[f"sum_sq_{prefix_b}"].to_numpy(dtype=np.float64)
    nz_a = df_groups[f"nonzero_{prefix_a}"].to_numpy(dtype=np.int64)
    nz_b = df_groups[f"nonzero_{prefix_b}"].to_numpy(dtype=np.int64)

    mean_a = sum_a / max(n_a, 1)
    mean_b = sum_b / max(n_b, 1)

    if n_a >= 2:
        var_a = np.clip((sum_sq_a - n_a * mean_a ** 2) / (n_a - 1), 0.0, None)
    else:
        var_a = np.zeros_like(sum_a)
    if n_b >= 2:
        var_b = np.clip((sum_sq_b - n_b * mean_b ** 2) / (n_b - 1), 0.0, None)
    else:
        var_b = np.zeros_like(sum_b)

    denom_pooled = max(n_a + n_b - 2, 1)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / denom_pooled
    cohens_d_arr = np.full(n_features, np.nan, dtype=np.float64)
    if n_a >= 2 and n_b >= 2:
        valid = pooled > 0
        cohens_d_arr[valid] = (mean_a[valid] - mean_b[valid]) / np.sqrt(pooled[valid])

    auc_arr = np.full(n_features, 0.5, dtype=np.float64)
    if n_a > 0 and n_b > 0:
        bucket_a_only = (nz_a > 0) & (nz_b == 0)
        auc_arr[bucket_a_only] = (nz_a[bucket_a_only] + 0.5 * (n_a - nz_a[bucket_a_only])) / n_a

        bucket_b_only = (nz_a == 0) & (nz_b > 0)
        auc_arr[bucket_b_only] = 0.5 * (n_b - nz_b[bucket_b_only]) / n_b

        bucket_both = (nz_a > 0) & (nz_b > 0)
        both_feature_ids = df_groups.loc[bucket_both, "feature_id"].astype(np.int64).to_numpy()
        if both_feature_ids.size:
            row_to_group = np.full(len(mask_a), -1, dtype=np.int8)
            row_to_group[mask_a] = 0
            row_to_group[mask_b] = 1
            row_arr = long_df["row"].to_numpy()
            grp_arr = row_to_group[row_arr]
            in_ab = grp_arr >= 0
            in_set = np.isin(long_df["feature_id"].to_numpy(), both_feature_ids)
            keep = in_ab & in_set
            sub = pd.DataFrame({
                "feature_id": long_df["feature_id"].to_numpy()[keep],
                "group": grp_arr[keep],
                "activation": long_df["activation"].to_numpy()[keep].astype(np.float64),
            })
            nonzero_by_feat: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for fid, grp in sub.groupby("feature_id"):
                vals_a = grp.loc[grp["group"] == 0, "activation"].to_numpy()
                vals_b = grp.loc[grp["group"] == 1, "activation"].to_numpy()
                nonzero_by_feat[int(fid)] = (vals_a, vals_b)
            feature_id_col = df_groups["feature_id"].to_numpy(dtype=np.int64)
            both_indices = np.flatnonzero(bucket_both)
            inv_n_ab = 1.0 / (n_a * n_b)
            for idx in both_indices:
                fid = int(feature_id_col[idx])
                vals_a, vals_b = nonzero_by_feat.get(fid, (np.zeros(0), np.zeros(0)))
                k_a = vals_a.size
                k_b = vals_b.size
                if k_a == 0 or k_b == 0:
                    continue  # safety: shouldn't happen given the mask
                diff = vals_a[:, None] - vals_b[None, :]
                wins = float((diff > 0).sum())
                ties = float((diff == 0).sum())
                numerator = (
                    0.5 * (n_a - k_a) * (n_b - k_b)
                    + k_a * (n_b - k_b)
                    + wins
                    + 0.5 * ties
                )
                auc_arr[idx] = numerator * inv_n_ab

    return cohens_d_arr, auc_arr


def feature_selectivity_for_contrast(
    long_df: pd.DataFrame,
    metadata: pd.DataFrame,
    contrast: pd.Series,
    n_features: int,
    top_n: int,
    layer: int,
) -> pd.DataFrame:
    """Compute per-feature contrast selectivity (Cohen's d / AUC / diff_mean)
    for ALL features, then rank and return the top_n. Audit 2.5: prior code
    pre-filtered to top (5·top_n) by |diff_mean| before computing d / AUC,
    which conditioned the reported effect sizes on having survived that
    selection screen and biased them upward. The d / AUC are now computed
    analytically over the full feature set using sufficient-statistic
    aggregations (see compute_cohens_d_and_auc_for_all_features)."""
    mask_a = metadata["identity_id"].eq(contrast.identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(contrast.identity_b).to_numpy()
    stats_a = summarize_feature_groups(long_df, mask_a, n_features, "a")
    stats_b = summarize_feature_groups(long_df, mask_b, n_features, "b")
    df = stats_a.merge(stats_b, on="feature_id", how="outer").fillna(0)
    df["diff_mean"] = df["mean_a"] - df["mean_b"]
    cohens_d_arr, auc_arr = compute_cohens_d_and_auc_for_all_features(
        long_df, mask_a, mask_b, df, prefix_a="a", prefix_b="b"
    )
    df["cohens_d"] = cohens_d_arr
    df["auc"] = auc_arr
    df["auc_distance"] = (df["auc"] - 0.5).abs()
    df["rank_abs_d"] = df["cohens_d"].abs().rank(method="min", ascending=False, na_option="bottom")
    df["rank_auc"] = df["auc_distance"].rank(method="min", ascending=False, na_option="bottom")
    df["rank_abs_diff"] = df["diff_mean"].abs().rank(method="min", ascending=False, na_option="bottom")
    candidates = df.sort_values(["rank_abs_d", "rank_auc", "rank_abs_diff"]).head(top_n).copy()
    candidates["layer"] = layer
    candidates["contrast_name"] = contrast.contrast_name
    candidates["axis"] = contrast.axis
    candidates["identity_a"] = contrast.identity_a
    candidates["identity_b"] = contrast.identity_b
    candidates["n_a"] = int(mask_a.sum())
    candidates["n_b"] = int(mask_b.sum())
    columns = [
        "layer", "contrast_name", "axis", "identity_a", "identity_b", "feature_id",
        "mean_a", "mean_b", "freq_a", "freq_b", "mean_nonzero_a", "mean_nonzero_b",
        "diff_mean", "cohens_d", "auc", "n_a", "n_b", "rank_abs_d", "rank_auc", "rank_abs_diff",
    ]
    return candidates.reindex(columns=columns)


def identity_selectivity(long_df: pd.DataFrame, metadata: pd.DataFrame, n_features: int, top_n: int, layer: int) -> pd.DataFrame:
    """Per-(identity, feature) selectivity vs. other identities on the same axis.
    Audit 2.5: prior code pre-filtered to top (3·top_n) by |diff_mean| before
    computing d / AUC. We now compute both analytically for ALL features and
    rank afterward, removing the selection-induced bias."""
    rows = []
    for (axis, identity_id, canonical_label), _ in tqdm(list(metadata.groupby(["axis", "identity_id", "canonical_label"], sort=True)), desc="identity selectivity", leave=False):
        mask_identity = metadata["identity_id"].eq(identity_id).to_numpy()
        mask_other = metadata["axis"].eq(axis).to_numpy() & ~mask_identity
        if min(mask_identity.sum(), mask_other.sum()) == 0:
            continue
        stats_i = summarize_feature_groups(long_df, mask_identity, n_features, "identity")
        stats_o = summarize_feature_groups(long_df, mask_other, n_features, "other")
        df = stats_i.merge(stats_o, on="feature_id", how="outer").fillna(0)
        df["diff_mean"] = df["mean_identity"] - df["mean_other"]
        cohens_d_arr, auc_arr = compute_cohens_d_and_auc_for_all_features(
            long_df, mask_identity, mask_other, df, prefix_a="identity", prefix_b="other"
        )
        df["cohens_d"] = cohens_d_arr
        df["auc_identity_vs_other_same_axis"] = auc_arr
        candidates = df.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False, na_position="last").head(top_n)
        for row in candidates.itertuples(index=False):
            rows.append({
                "layer": layer,
                "axis": axis,
                "identity_id": identity_id,
                "canonical_label": canonical_label,
                "feature_id": int(row.feature_id),
                "mean_identity": row.mean_identity,
                "mean_other_same_axis": row.mean_other,
                "diff_mean": row.diff_mean,
                "cohens_d": row.cohens_d,
                "auc_identity_vs_other_same_axis": row.auc_identity_vs_other_same_axis,
                "freq_identity": row.freq_identity,
                "freq_other": row.freq_other,
                "n_identity": int(mask_identity.sum()),
                "n_other": int(mask_other.sum()),
            })
    return pd.DataFrame(rows)


def decoder_alignment(decoder: np.ndarray, direction: np.ndarray, contrast: pd.Series, layer: int, residualization: str) -> pd.DataFrame:
    norms = np.linalg.norm(decoder, axis=1)
    safe_norms = np.maximum(norms, 1e-12)
    cosine = (decoder @ direction) / safe_norms
    signed_dot = decoder @ direction
    df = pd.DataFrame({
        "layer": layer,
        "residualization": residualization,
        "contrast_name": contrast.contrast_name,
        "axis": contrast.axis,
        "identity_a": contrast.identity_a,
        "identity_b": contrast.identity_b,
        "feature_id": np.arange(decoder.shape[0], dtype=np.int64),
        "decoder_norm": norms,
        "cosine_with_direction": cosine,
        "signed_dot": signed_dot,
    })
    df["abs_cosine_rank"] = df["cosine_with_direction"].abs().rank(method="min", ascending=False)
    df["positive_rank"] = df["cosine_with_direction"].rank(method="min", ascending=False)
    df["negative_rank"] = df["cosine_with_direction"].rank(method="min", ascending=True)
    return df


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / sd


def reconstruct_direction(decoder_normed: np.ndarray, direction: np.ndarray, feature_ids: np.ndarray) -> tuple[np.ndarray | None, float, float]:
    if len(feature_ids) == 0:
        return None, float("nan"), float("nan")
    basis = decoder_normed[feature_ids]
    coeff = basis @ direction
    recon = coeff @ basis
    norm = np.linalg.norm(recon)
    if norm == 0 or not np.isfinite(norm):
        return None, float("nan"), float("nan")
    recon_unit = recon / norm
    cosine = float(np.dot(direction, recon_unit))
    fraction = float(np.linalg.norm(recon) ** 2)
    return recon_unit.astype(np.float32), cosine, fraction


def reconstruction_rows(
    joined: pd.DataFrame,
    alignment: pd.DataFrame,
    decoder: np.ndarray,
    x: np.ndarray,
    metadata: pd.DataFrame,
    contrast: pd.Series,
    direction: np.ndarray,
    global_mean: np.ndarray,
    k_values: list[int],
    layer: int,
    residualization: str,
) -> list[dict[str, object]]:
    rows = []
    decoder_normed = decoder / np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-12)
    full_auc, full_d = evaluate_direction(x, metadata, contrast.identity_a, contrast.identity_b, direction, global_mean)
    rng = np.random.default_rng(layer + len(contrast.contrast_name))
    selection_sources = {
        "decoder_alignment": alignment.sort_values("cosine_with_direction", key=lambda s: s.abs(), ascending=False)["feature_id"].to_numpy(),
        "selectivity": joined.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)["feature_id"].to_numpy(),
        "combined_score": joined.sort_values("combined_score", ascending=False)["feature_id"].to_numpy(),
        "random_baseline": rng.permutation(decoder.shape[0]),
    }
    for method, ordered_features in selection_sources.items():
        for k in k_values:
            selected = ordered_features[: min(k, len(ordered_features))].astype(int)
            recon, cosine, fraction = reconstruct_direction(decoder_normed, direction, selected)
            if recon is None:
                auc, d_value = float("nan"), float("nan")
            else:
                auc, d_value = evaluate_direction(x, metadata, contrast.identity_a, contrast.identity_b, recon, global_mean)
            # All AUC / Cohen's d here are in-sample — the contrast direction and
            # the feature ranking are both derived from the same data the metrics
            # are evaluated on. Audit 2.1; the held-out reconstruction is a
            # follow-up bundled with the 2.5 winner's-curse fix because the
            # full held-out story also requires held-out feature SELECTION,
            # not just held-out direction estimation.
            rows.append({
                "layer": layer,
                "residualization": residualization,
                "contrast_name": contrast.contrast_name,
                "axis": contrast.axis,
                "identity_a": contrast.identity_a,
                "identity_b": contrast.identity_b,
                "selection_method": method,
                "k": k,
                "cosine_with_full_direction": cosine,
                "auc_in_sample": auc,
                "cohens_d_in_sample": d_value,
                "fraction_norm_captured": fraction,
                "n_features": len(selected),
                "full_direction_auc_in_sample": full_auc,
                "full_direction_cohens_d_in_sample": full_d,
            })
    return rows


def intervention_candidates(joined: pd.DataFrame, contrast: pd.Series, layer: int, top_n: int) -> pd.DataFrame:
    candidates = joined.sort_values("combined_score", ascending=False).head(top_n).copy()
    rows = []
    for row in candidates.itertuples(index=False):
        side = "identity_a_positive" if row.diff_mean >= 0 else "identity_b_positive"
        rows.append({
            "layer": layer,
            "contrast_name": contrast.contrast_name,
            "axis": contrast.axis,
            "identity_a": contrast.identity_a,
            "identity_b": contrast.identity_b,
            "feature_id": int(row.feature_id),
            "direction_side": side,
            "mean_activation_identity_a": row.mean_a,
            "mean_activation_identity_b": row.mean_b,
            "decoder_cosine": row.cosine_with_direction,
            "cohens_d": row.cohens_d,
            "auc": row.auc,
            "combined_score": row.combined_score,
            "recommended_intervention": "ablate",
            "notes": "Candidate only; validate with model-forward BBQ intervention before causal claims.",
        })
    return pd.DataFrame(rows)


# SAE intervention primitives — numpy versions used by this script's analysis
# helpers. The canonical TORCH versions (`ablate_features`, `clamp_features`,
# `steer_features`, `patched_residual_with_intervention`) live in
# scripts/encode_identity_saes.py and are used by run_bbq_sae_steering.py
# (audit 3.1). These numpy versions share semantics; both implementations
# operate on latent activations in NORMALIZED space (the output of the
# JumpReLU encoder) and the decode produces a residual-space reconstruction
# that must be added to (not replaced into) the original residual.


def ablate_features_in_sae(latent_acts: np.ndarray, feature_ids: list[int]) -> np.ndarray:
    """Clamp the specified latent features to 0 (numpy)."""
    modified = latent_acts.copy()
    modified[:, feature_ids] = 0
    return modified


def steer_features_in_sae(latent_acts: np.ndarray, feature_ids: list[int], alpha: float) -> np.ndarray:
    """Add `alpha` to the specified latent features (numpy; alpha is in
    normalized latent units, same caveat as the torch version)."""
    modified = latent_acts.copy()
    modified[:, feature_ids] += alpha
    return modified


def clamp_features_in_sae(latent_acts: np.ndarray, feature_ids: list[int], value: float) -> np.ndarray:
    """Clamp the specified latent features to `value` (numpy). Use with
    feature_stats.csv p95/p99/max for amplification."""
    modified = latent_acts.copy()
    modified[:, feature_ids] = value
    return modified


def decode_sae(latent_acts: np.ndarray, decoder: np.ndarray, decoder_bias: np.ndarray | None = None) -> np.ndarray:
    """Pure decode in NORMALIZED space — does NOT apply the audit-1.4
    scale_out factor. For the corrected residual-space decode use the torch
    `decode_full` in encode_identity_saes.py (which multiplies by scale_out
    after the affine step)."""
    recon = latent_acts @ decoder
    if decoder_bias is not None:
        recon = recon + decoder_bias
    return recon


def patch_residual_with_sae_reconstruction(original_x: np.ndarray, modified_reconstruction: np.ndarray, original_reconstruction: np.ndarray) -> np.ndarray:
    """h + (recon_modified - recon_original). SAE reconstruction error cancels
    in the delta — see encode_identity_saes.patched_residual_with_intervention
    for the canonical torch implementation used by BBQ steering."""
    return original_x + (modified_reconstruction - original_reconstruction)


def main() -> None:
    args = parse_args()
    start_all = time.perf_counter()
    if args.residualization not in RESIDUALIZATION_GROUPS:
        raise ValueError(f"Unknown residualization: {args.residualization}")
    prepare_output(args.output_dir, args.overwrite)
    metadata = pd.read_csv(args.activation_dir / "metadata.csv", keep_default_na=False)
    contrasts = load_contrasts(args.contrasts_csv, metadata, output_dir=args.output_dir)
    k_values = parse_int_list(args.top_k_reconstruction_values)
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "sae_encoded_dir": str(args.sae_encoded_dir),
        "activation_dir": str(args.activation_dir),
        "output_dir": str(args.output_dir),
        "layers": parse_int_list(args.layers),
        "top_n_features": args.top_n_features,
        "top_k_reconstruction_values": k_values,
        "residualization": args.residualization,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")

    for layer in tqdm(parse_int_list(args.layers), desc="layers"):
        layer_start = time.perf_counter()
        layer_dir = args.sae_encoded_dir / f"layer_{layer:02d}"
        index_path, value_path, _ = find_topk_files(layer_dir)
        indices = np.load(index_path, mmap_mode="r")
        values = np.load(value_path, mmap_mode="r")
        decoder = np.load(layer_dir / "sae_decoder.npy", mmap_mode="r")
        n_features = decoder.shape[0]
        x = np.asarray(np.load(args.activation_dir / f"layer_{layer:02d}.npy", mmap_mode="r"), dtype=np.float32)
        x = residualize(x, metadata, args.residualization)
        long_df = sparse_long(np.asarray(indices), np.asarray(values))
        print(f"\nLayer {layer:02d}: sparse rows={len(long_df):,}, n_features={n_features:,}")

        identity_df = identity_selectivity(long_df, metadata, n_features, args.top_n_features, layer)
        append_csv(args.output_dir / "feature_identity_selectivity.csv", identity_df.to_dict("records"))

        for contrast in tqdm(list(contrasts.itertuples(index=False)), desc=f"layer {layer:02d} contrasts", leave=False):
            direction, global_mean = compute_direction(x, metadata, contrast.identity_a, contrast.identity_b)
            if direction is None:
                continue
            selectivity = feature_selectivity_for_contrast(long_df, metadata, contrast, n_features, args.top_n_features, layer)
            alignment = decoder_alignment(np.asarray(decoder, dtype=np.float32), direction, contrast, layer, args.residualization)
            joined = selectivity.merge(alignment, on=["layer", "contrast_name", "axis", "identity_a", "identity_b", "feature_id"], how="left")
            joined["auc_distance"] = (joined["auc"] - 0.5).abs()
            joined["combined_score"] = zscore(joined["cohens_d"].abs()) + zscore(joined["cosine_with_direction"].abs()) + zscore(joined["auc_distance"])
            append_csv(args.output_dir / "feature_selectivity.csv", selectivity.to_dict("records"))
            alignment_top = pd.concat([
                alignment.nsmallest(args.top_n_features, "negative_rank"),
                alignment.nsmallest(args.top_n_features, "positive_rank"),
                alignment.nsmallest(args.top_n_features, "abs_cosine_rank"),
            ]).drop_duplicates("feature_id")
            append_csv(args.output_dir / "decoder_direction_alignment.csv", alignment_top.to_dict("records"))
            append_csv(args.output_dir / "feature_selectivity_alignment_joined.csv", joined.to_dict("records"))
            recon = reconstruction_rows(joined, alignment, np.asarray(decoder, dtype=np.float32), x, metadata, contrast, direction, global_mean, k_values, layer, args.residualization)
            append_csv(args.output_dir / "direction_reconstruction.csv", recon)
            candidates = intervention_candidates(joined, contrast, layer, args.top_n_features)
            append_csv(args.output_dir / "intervention_candidate_features.csv", candidates.to_dict("records"))
        print(f"Layer {layer:02d}: complete in {elapsed(layer_start)}")

    print(f"\nSAE feature analysis complete in {elapsed(start_all)}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
