#!/usr/bin/env python3
"""Plot SAE identity-feature analysis outputs and make feature cards."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


DEFAULT_ANALYSIS_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/analysis")
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/figures")
KEY_CONTRASTS = [
    "sexuality_gay_vs_sexuality_straight",
    "race_black_vs_race_white",
    "gender_transgender_vs_gender_cisgender",
    "appearance_obese_vs_appearance_thin",
    "ses_low_income_vs_ses_rich",
    "disability_disabled_vs_disability_able_bodied",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAE identity-feature analyses and generate feature cards.")
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--make_feature_cards", action="store_true")
    parser.add_argument("--max_feature_cards", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists and is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    for subdir in ["selectivity", "alignment", "reconstruction", "feature_profiles", "feature_cards"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=220)
    fig.savefig(path_no_suffix.with_suffix(".pdf"))
    plt.close(fig)


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_selectivity_heatmap(analysis_dir: Path, output_dir: Path, layer: int) -> None:
    df = safe_read(analysis_dir / "feature_selectivity.csv")
    if df.empty:
        return
    df = df[df["layer"].eq(layer)].copy()
    if df.empty:
        return
    top_features = df.assign(abs_d=df["cohens_d"].abs()).sort_values("abs_d", ascending=False).head(80)["feature_id"].unique()
    heat = df[df["feature_id"].isin(top_features)].pivot_table(index="feature_id", columns="contrast_name", values="cohens_d", aggfunc="mean").fillna(0)
    if heat.empty:
        return
    fig, ax = plt.subplots(figsize=(max(12, 0.45 * heat.shape[1]), max(8, 0.18 * heat.shape[0])))
    if sns is not None:
        sns.heatmap(heat, cmap="vlag", center=0, ax=ax)
    else:
        im = ax.imshow(heat.to_numpy(), cmap="coolwarm")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=90)
        ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title(f"Top SAE feature selectivity by contrast, layer {layer:02d}")
    ax.set_xlabel("Contrast")
    ax.set_ylabel("SAE feature ID")
    save_fig(fig, output_dir / "selectivity" / f"top_feature_selectivity_heatmap_layer{layer:02d}")


def plot_alignment_scatters(analysis_dir: Path, output_dir: Path, layer: int) -> None:
    df = safe_read(analysis_dir / "feature_selectivity_alignment_joined.csv")
    if df.empty:
        return
    df = df[df["layer"].eq(layer)].copy()
    for contrast_name, contrast_df in df.groupby("contrast_name", sort=True):
        if contrast_name not in KEY_CONTRASTS and len(contrast_df) > 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        x = contrast_df["cosine_with_direction"]
        y = contrast_df["cohens_d"]
        ax.scatter(x, y, s=18, alpha=0.55)
        label_df = contrast_df.assign(score=contrast_df["combined_score"]).sort_values("score", ascending=False).head(10)
        for row in label_df.itertuples(index=False):
            ax.text(row.cosine_with_direction, row.cohens_d, str(int(row.feature_id)), fontsize=7)
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set_title(f"{contrast_name}: selectivity vs decoder alignment, layer {layer:02d}")
        ax.set_xlabel("Decoder cosine with identity direction")
        ax.set_ylabel("Activation Cohen's d")
        save_fig(fig, output_dir / "alignment" / f"selectivity_vs_decoder_alignment_{contrast_name}_layer{layer:02d}")


def plot_reconstruction(analysis_dir: Path, output_dir: Path) -> None:
    df = safe_read(analysis_dir / "direction_reconstruction.csv")
    if df.empty:
        return
    plot_df = df[df["contrast_name"].isin(KEY_CONTRASTS)].copy()
    if plot_df.empty:
        plot_df = df
    for y_col, ylabel, filename in [
        ("cosine_with_full_direction", "Cosine with full direction", "reconstruction_cosine_by_k"),
        ("auc", "Endpoint AUC", "reconstruction_auc_by_k"),
        ("cohens_d", "Endpoint Cohen's d", "reconstruction_d_by_k"),
    ]:
        fig, ax = plt.subplots(figsize=(15, 7))
        if sns is not None:
            sns.lineplot(data=plot_df, x="k", y=y_col, hue="contrast_name", style="selection_method", ax=ax, linewidth=2, marker="o")
        else:
            for name, group in plot_df.groupby(["contrast_name", "selection_method"], sort=True):
                ax.plot(group["k"], group[y_col], marker="o", label=str(name))
            ax.legend(frameon=False)
        if y_col == "auc":
            ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
            ax.set_ylim(0.45, 1.02)
        ax.set_title(ylabel + " from sparse SAE decoder reconstruction")
        ax.set_xlabel("Number of selected SAE features")
        ax.set_ylabel(ylabel)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)
        save_fig(fig, output_dir / "reconstruction" / filename)


def plot_feature_profiles(analysis_dir: Path, output_dir: Path, layer: int) -> None:
    df = safe_read(analysis_dir / "feature_identity_selectivity.csv")
    if df.empty:
        return
    df = df[df["layer"].eq(layer)].copy()
    top_features = df.assign(abs_d=df["cohens_d"].abs()).sort_values("abs_d", ascending=False).head(30)["feature_id"].unique()
    for feature_id in top_features:
        feature_df = df[df["feature_id"].eq(feature_id)].sort_values("mean_identity")
        fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(feature_df))))
        labels = feature_df["canonical_label"] + " (" + feature_df["axis"].str.replace("_", " ") + ")"
        ax.barh(labels, feature_df["mean_identity"], color="#0072B2")
        ax.set_title(f"SAE feature {feature_id}: mean activation by identity, layer {layer:02d}")
        ax.set_xlabel("Mean activation")
        save_fig(fig, output_dir / "feature_profiles" / f"feature_{int(feature_id)}_identity_profile_layer{layer:02d}")


def find_topk_files(layer_dir: Path) -> tuple[Path, Path, int]:
    index_files = sorted(layer_dir.glob("feature_indices_top*.npy"))
    if not index_files:
        raise FileNotFoundError(f"No feature_indices_top*.npy found in {layer_dir}")
    index_path = index_files[-1]
    top_k = int(index_path.stem.split("top")[-1])
    return index_path, layer_dir / f"feature_values_top{top_k}.npy", top_k


def feature_values_for_id(indices: np.ndarray, values: np.ndarray, feature_id: int) -> np.ndarray:
    out = np.zeros(indices.shape[0], dtype=np.float32)
    mask = indices == feature_id
    row_has = mask.any(axis=1)
    if row_has.any():
        rows = np.flatnonzero(row_has)
        cols = mask[row_has].argmax(axis=1)
        out[rows] = values[rows, cols]
    return out


def highlight_form(prompt: str, form: str) -> str:
    prompt_html = html.escape(str(prompt))
    form_html = html.escape(str(form))
    if form_html and form_html in prompt_html:
        return prompt_html.replace(form_html, f"<mark>{form_html}</mark>", 1)
    return prompt_html


def write_feature_card(
    card_dir: Path,
    layer: int,
    feature_id: int,
    metadata: pd.DataFrame,
    feature_values: np.ndarray,
    analysis_dir: Path,
) -> Path:
    card_dir.mkdir(parents=True, exist_ok=True)
    top_rows = metadata.copy()
    top_rows["activation"] = feature_values
    top_rows = top_rows.sort_values("activation", ascending=False).head(25)
    identity_means = metadata[["identity_id", "canonical_label", "axis"]].copy()
    identity_means["activation"] = feature_values
    identity_means = identity_means.groupby(["identity_id", "canonical_label", "axis"], sort=True)["activation"].mean().reset_index().sort_values("activation", ascending=False)
    align = safe_read(analysis_dir / "decoder_direction_alignment.csv")
    select = safe_read(analysis_dir / "feature_selectivity.csv")
    align = align[(align["layer"].eq(layer)) & (align["feature_id"].eq(feature_id))].copy() if not align.empty else pd.DataFrame()
    select = select[(select["layer"].eq(layer)) & (select["feature_id"].eq(feature_id))].copy() if not select.empty else pd.DataFrame()

    profile_path = card_dir / f"feature_{feature_id:05d}_identity_profile.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_means = identity_means.head(12).sort_values("activation")
    ax.barh(plot_means["canonical_label"] + " (" + plot_means["axis"].str.replace("_", " ") + ")", plot_means["activation"], color="#0072B2")
    ax.set_title(f"Top identities by mean activation, feature {feature_id}")
    ax.set_xlabel("Mean activation")
    fig.tight_layout()
    fig.savefig(profile_path, dpi=180)
    plt.close(fig)

    json_path = card_dir / f"feature_{feature_id:05d}.json"
    json_payload = {
        "layer": layer,
        "feature_id": feature_id,
        "top_identities": identity_means.head(20).to_dict("records"),
        "top_prompts": top_rows[["activation", "prompt", "identity_id", "axis", "family", "form_used"]].to_dict("records"),
        "top_decoder_aligned_contrasts": align.sort_values("cosine_with_direction", key=lambda s: s.abs(), ascending=False).head(15).to_dict("records") if not align.empty else [],
        "top_activation_selective_contrasts": select.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).head(15).to_dict("records") if not select.empty else [],
        "logit_effects": "not computed",
    }
    json_path.write_text(json.dumps(json_payload, indent=2, default=str) + "\n")

    prompt_rows = "\n".join(
        "<tr>"
        f"<td>{row.activation:.3f}</td>"
        f"<td>{html.escape(str(row.identity_id))}</td>"
        f"<td>{html.escape(str(row.axis))}</td>"
        f"<td>{html.escape(str(row.family))}</td>"
        f"<td>{highlight_form(row.prompt, row.form_used)}</td>"
        "</tr>"
        for row in top_rows.itertuples(index=False)
    )
    identity_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.canonical_label))}</td><td>{html.escape(str(row.axis))}</td><td>{row.activation:.4f}</td></tr>"
        for row in identity_means.head(15).itertuples(index=False)
    )
    align_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.contrast_name))}</td><td>{row.cosine_with_direction:.4f}</td><td>{row.signed_dot:.4f}</td></tr>"
        for row in align.sort_values("cosine_with_direction", key=lambda s: s.abs(), ascending=False).head(15).itertuples(index=False)
    ) if not align.empty else "<tr><td colspan='3'>No decoder alignment rows found.</td></tr>"
    select_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.contrast_name))}</td><td>{row.cohens_d:.3f}</td><td>{row.auc:.3f}</td><td>{row.diff_mean:.4f}</td></tr>"
        for row in select.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False).head(15).itertuples(index=False)
    ) if not select.empty else "<tr><td colspan='4'>No selectivity rows found.</td></tr>"
    html_path = card_dir / f"feature_{feature_id:05d}.html"
    html_path.write_text(f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Layer {layer} SAE feature {feature_id}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 28px; line-height: 1.35; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    mark {{ background: #fff59d; padding: 0 2px; }}
    .note {{ color: #555; }}
  </style>
</head>
<body>
  <h1>Layer {layer} SAE feature {feature_id}</h1>
  <p class="note">Auto-generated final-token feature card. Token-span activations and logit effects are not computed in this first version.</p>
  <h2>Identity Activation Profile</h2>
  <img src="{profile_path.name}" style="max-width: 900px; width: 100%;">
  <table><tr><th>Identity</th><th>Axis</th><th>Mean activation</th></tr>{identity_rows}</table>
  <h2>Top Activating Prompts</h2>
  <table><tr><th>Activation</th><th>Identity</th><th>Axis</th><th>Family</th><th>Prompt</th></tr>{prompt_rows}</table>
  <h2>Decoder-Aligned Contrasts</h2>
  <table><tr><th>Contrast</th><th>Decoder cosine</th><th>Signed dot</th></tr>{align_rows}</table>
  <h2>Activation Selective Contrasts</h2>
  <table><tr><th>Contrast</th><th>Cohen's d</th><th>AUC</th><th>Mean diff</th></tr>{select_rows}</table>
  <h2>Logit Effects</h2>
  <p>Logit effects not computed. This requires loading the model unembedding and confirming SAE decoder normalization.</p>
</body>
</html>
""")
    return html_path


def make_feature_cards(analysis_dir: Path, output_dir: Path, layers: list[int], max_cards: int) -> None:
    encoded_root = analysis_dir.parent
    candidates = safe_read(analysis_dir / "intervention_candidate_features.csv")
    joined = safe_read(analysis_dir / "feature_selectivity_alignment_joined.csv")
    if candidates.empty and joined.empty:
        return
    links = []
    for layer in layers:
        layer_dir = encoded_root / f"layer_{layer:02d}"
        metadata_path = layer_dir / "metadata.csv"
        if not metadata_path.exists():
            print(f"Skipping feature cards for layer {layer}: missing {metadata_path}")
            continue
        metadata = pd.read_csv(metadata_path, keep_default_na=False)
        index_path, value_path, _ = find_topk_files(layer_dir)
        indices = np.load(index_path, mmap_mode="r")
        values = np.load(value_path, mmap_mode="r")
        if not candidates.empty:
            feature_ids = candidates[candidates["layer"].eq(layer)].sort_values("combined_score", ascending=False)["feature_id"].drop_duplicates().head(max_cards).astype(int).tolist()
        else:
            feature_ids = joined[joined["layer"].eq(layer)].sort_values("combined_score", ascending=False)["feature_id"].drop_duplicates().head(max_cards).astype(int).tolist()
        card_dir = output_dir / "feature_cards" / f"layer_{layer:02d}"
        for feature_id in feature_ids:
            vals = feature_values_for_id(np.asarray(indices), np.asarray(values), feature_id)
            html_path = write_feature_card(card_dir, layer, feature_id, metadata, vals, analysis_dir)
            links.append((layer, feature_id, html_path.relative_to(output_dir / "feature_cards")))
    index_rows = "\n".join(f"<li>Layer {layer} feature {feature_id}: <a href='{path}'>{path}</a></li>" for layer, feature_id, path in links)
    (output_dir / "feature_cards" / "index.html").write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SAE Identity Feature Cards</title></head>
<body>
<h1>SAE Identity Feature Cards</h1>
<p>Generated {datetime.now(timezone.utc).isoformat()}</p>
<ul>{index_rows}</ul>
</body></html>
""")


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    prepare_output(args.output_dir, args.overwrite)
    for layer in layers:
        plot_selectivity_heatmap(args.analysis_dir, args.output_dir, layer)
        plot_alignment_scatters(args.analysis_dir, args.output_dir, layer)
        plot_feature_profiles(args.analysis_dir, args.output_dir, layer)
    plot_reconstruction(args.analysis_dir, args.output_dir)
    if args.make_feature_cards:
        make_feature_cards(args.analysis_dir, args.output_dir, layers, args.max_feature_cards)
    print(f"SAE identity figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
