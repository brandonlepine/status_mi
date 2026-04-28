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


def load_contrasts(path: Path | None, metadata: pd.DataFrame) -> pd.DataFrame:
    contrasts = pd.read_csv(path, keep_default_na=False) if path else pd.DataFrame(DEFAULT_CONTRASTS, columns=["contrast_name", "identity_a", "identity_b", "axis"])
    identities = set(metadata["identity_id"])
    valid = contrasts[contrasts["identity_a"].isin(identities) & contrasts["identity_b"].isin(identities)].copy()
    skipped = len(contrasts) - len(valid)
    if skipped:
        print(f"Skipping {skipped} contrasts because one or both identity IDs are absent.")
    return valid.reset_index(drop=True)


def residualize(x: np.ndarray, metadata: pd.DataFrame, residualization: str) -> np.ndarray:
    group_col = RESIDUALIZATION_GROUPS[residualization]
    if group_col is None:
        return x
    global_mean = x.mean(axis=0, keepdims=True)
    x_resid = x.copy()
    for _, idx in metadata.groupby(group_col, sort=True).groups.items():
        idx_array = np.fromiter(idx, dtype=int)
        x_resid[idx_array] = x[idx_array] - x[idx_array].mean(axis=0, keepdims=True) + global_mean
    return x_resid


def normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        return None
    return vec / norm


def compute_direction(x: np.ndarray, metadata: pd.DataFrame, identity_a: str, identity_b: str) -> tuple[np.ndarray | None, np.ndarray]:
    global_mean = x.mean(axis=0, keepdims=True)
    centered = x - global_mean
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
    if min(mask_a.sum(), mask_b.sum()) == 0:
        return None, global_mean
    direction = normalize(centered[mask_a].mean(axis=0) - centered[mask_b].mean(axis=0))
    if direction is None:
        return None, global_mean
    scores = centered @ direction
    if scores[mask_a].mean() < scores[mask_b].mean():
        direction = -direction
    return direction.astype(np.float32), global_mean.astype(np.float32)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = (((len(a) - 1) * np.var(a, ddof=1)) + ((len(b) - 1) * np.var(b, ddof=1))) / (len(a) + len(b) - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def evaluate_direction(x: np.ndarray, metadata: pd.DataFrame, identity_a: str, identity_b: str, direction: np.ndarray, global_mean: np.ndarray) -> tuple[float, float]:
    scores = (x - global_mean) @ direction
    mask_a = metadata["identity_id"].eq(identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(identity_b).to_numpy()
    a = scores[mask_a]
    b = scores[mask_b]
    labels = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    vals = np.concatenate([a, b])
    auc = float(roc_auc_score(labels, vals)) if len(np.unique(labels)) == 2 else float("nan")
    return auc, cohens_d(a, b)


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
    grouped = subset.groupby("feature_id")["activation"].agg(["sum", "count", "mean"]).reset_index()
    grouped = grouped.rename(columns={"sum": f"sum_{prefix}", "count": f"nonzero_{prefix}", "mean": f"mean_nonzero_{prefix}"})
    out = pd.DataFrame({"feature_id": np.arange(n_features, dtype=np.int64)})
    out = out.merge(grouped, on="feature_id", how="left").fillna(0)
    out[f"mean_{prefix}"] = out[f"sum_{prefix}"] / max(1, int(mask.sum()))
    out[f"freq_{prefix}"] = out[f"nonzero_{prefix}"] / max(1, int(mask.sum()))
    return out


def feature_selectivity_for_contrast(
    long_df: pd.DataFrame,
    metadata: pd.DataFrame,
    contrast: pd.Series,
    n_features: int,
    top_n: int,
    layer: int,
) -> pd.DataFrame:
    mask_a = metadata["identity_id"].eq(contrast.identity_a).to_numpy()
    mask_b = metadata["identity_id"].eq(contrast.identity_b).to_numpy()
    stats_a = summarize_feature_groups(long_df, mask_a, n_features, "a")
    stats_b = summarize_feature_groups(long_df, mask_b, n_features, "b")
    df = stats_a.merge(stats_b, on="feature_id", how="outer").fillna(0)
    df["diff_mean"] = df["mean_a"] - df["mean_b"]
    df["pooled"] = np.nan
    candidates = df.reindex(df["diff_mean"].abs().sort_values(ascending=False).head(max(top_n * 5, top_n)).index).copy()
    aucs = []
    ds = []
    for feature_id in candidates["feature_id"].astype(int):
        vals = dense_feature_values(long_df, feature_id, len(metadata))
        a = vals[mask_a]
        b = vals[mask_b]
        labels = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
        scores = np.concatenate([a, b])
        aucs.append(float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan"))
        ds.append(cohens_d(a, b))
    candidates["auc"] = aucs
    candidates["cohens_d"] = ds
    candidates["auc_distance"] = (candidates["auc"] - 0.5).abs()
    candidates["rank_abs_d"] = candidates["cohens_d"].abs().rank(method="min", ascending=False)
    candidates["rank_auc"] = candidates["auc_distance"].rank(method="min", ascending=False)
    candidates["rank_abs_diff"] = candidates["diff_mean"].abs().rank(method="min", ascending=False)
    candidates = candidates.sort_values(["rank_abs_d", "rank_auc", "rank_abs_diff"]).head(top_n).copy()
    candidates["layer"] = layer
    candidates["contrast_name"] = contrast.contrast_name
    candidates["axis"] = contrast.axis
    candidates["identity_a"] = contrast.identity_a
    candidates["identity_b"] = contrast.identity_b
    candidates["n_a"] = int(mask_a.sum())
    candidates["n_b"] = int(mask_b.sum())
    candidates = candidates.rename(columns={"freq_a": "freq_a", "freq_b": "freq_b", "mean_nonzero_a": "mean_nonzero_a", "mean_nonzero_b": "mean_nonzero_b"})
    columns = [
        "layer", "contrast_name", "axis", "identity_a", "identity_b", "feature_id",
        "mean_a", "mean_b", "freq_a", "freq_b", "mean_nonzero_a", "mean_nonzero_b",
        "diff_mean", "cohens_d", "auc", "n_a", "n_b", "rank_abs_d", "rank_auc", "rank_abs_diff",
    ]
    return candidates.reindex(columns=columns)


def identity_selectivity(long_df: pd.DataFrame, metadata: pd.DataFrame, n_features: int, top_n: int, layer: int) -> pd.DataFrame:
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
        candidates = df.reindex(df["diff_mean"].abs().sort_values(ascending=False).head(max(top_n * 3, top_n)).index).copy()
        aucs = []
        ds = []
        for feature_id in candidates["feature_id"].astype(int):
            vals = dense_feature_values(long_df, feature_id, len(metadata))
            a = vals[mask_identity]
            b = vals[mask_other]
            labels = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
            scores = np.concatenate([a, b])
            aucs.append(float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan"))
            ds.append(cohens_d(a, b))
        candidates["auc_identity_vs_other_same_axis"] = aucs
        candidates["cohens_d"] = ds
        candidates = candidates.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).head(top_n)
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
                "auc": auc,
                "cohens_d": d_value,
                "fraction_norm_captured": fraction,
                "n_features": len(selected),
                "full_direction_auc": full_auc,
                "full_direction_cohens_d": full_d,
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


def ablate_features_in_sae(latent_acts: np.ndarray, feature_ids: list[int]) -> np.ndarray:
    modified = latent_acts.copy()
    modified[:, feature_ids] = 0
    return modified


def steer_features_in_sae(latent_acts: np.ndarray, feature_ids: list[int], alpha: float) -> np.ndarray:
    modified = latent_acts.copy()
    modified[:, feature_ids] += alpha
    return modified


def decode_sae(latent_acts: np.ndarray, decoder: np.ndarray, decoder_bias: np.ndarray | None = None) -> np.ndarray:
    recon = latent_acts @ decoder
    if decoder_bias is not None:
        recon = recon + decoder_bias
    return recon


def patch_residual_with_sae_reconstruction(original_x: np.ndarray, modified_reconstruction: np.ndarray, original_reconstruction: np.ndarray) -> np.ndarray:
    return original_x + (modified_reconstruction - original_reconstruction)


def main() -> None:
    args = parse_args()
    start_all = time.perf_counter()
    if args.residualization not in RESIDUALIZATION_GROUPS:
        raise ValueError(f"Unknown residualization: {args.residualization}")
    prepare_output(args.output_dir, args.overwrite)
    metadata = pd.read_csv(args.activation_dir / "metadata.csv", keep_default_na=False)
    contrasts = load_contrasts(args.contrasts_csv, metadata)
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
