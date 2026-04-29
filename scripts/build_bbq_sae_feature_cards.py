#!/usr/bin/env python3
"""Build BBQ-specific SAE feature cards from token-level activations."""

from __future__ import annotations

import argparse
import html
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


DEFAULT_TOKEN_DIR = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/token_level_sae")
DEFAULT_PREPARED = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet")
DEFAULT_TRIAGE = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv")
DEFAULT_OUTPUT = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/feature_cards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BBQ SAE feature cards.")
    parser.add_argument("--token_level_dir", type=Path, default=DEFAULT_TOKEN_DIR)
    parser.add_argument("--prepared_data", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--triage_csv", type=Path, default=DEFAULT_TRIAGE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", default="24")
    parser.add_argument("--top_prompts_per_feature", type=int, default=25)
    parser.add_argument("--top_tokens_per_feature", type=int, default=50)
    parser.add_argument("--save_every_features", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def setup(output_dir: Path, overwrite: bool, resume: bool) -> logging.Logger:
    if output_dir.exists() and any(output_dir.iterdir()) and overwrite and not resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    logger = logging.getLogger("bbq_cards")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(output_dir / "logs" / "build_cards.log")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def load_layer_tokens(token_dir: Path, layer: int) -> pd.DataFrame:
    layer_dir = token_dir / "token_activations" / f"layer_{layer:02d}"
    files = sorted(layer_dir.glob("part_*.parquet")) + sorted(layer_dir.glob("part_*.csv"))
    frames = [read_table(path) for path in files if path.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    if df[col].dtype == bool:
        return df[col]
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def token_role(row: pd.Series) -> str:
    if bool(row.get("is_target_identity_token", False)):
        return "target_identity"
    if bool(row.get("is_nontarget_identity_token", False)):
        return "nontarget_identity"
    if bool(row.get("is_stereotype_language_token", False)):
        return "stereotype_language"
    if bool(row.get("is_answer_option_token", False)):
        return "answer_option"
    if bool(row.get("is_final_prompt_token", False)):
        return "final_token"
    return "other_context"


def classify_behavior(group: pd.DataFrame) -> str:
    means = {
        "target_identity_local": group.loc[bool_col(group, "is_target_identity_token"), "feature_activation"].mean(),
        "nontarget_identity_local": group.loc[bool_col(group, "is_nontarget_identity_token"), "feature_activation"].mean(),
        "stereotype_language_local": group.loc[bool_col(group, "is_stereotype_language_token"), "feature_activation"].mean(),
        "final_token_integrated": group.loc[bool_col(group, "is_final_prompt_token"), "feature_activation"].mean(),
    }
    means = {k: (0 if pd.isna(v) else float(v)) for k, v in means.items()}
    best, value = max(means.items(), key=lambda item: item[1])
    overall = float(group["feature_activation"].mean()) if not group.empty else 0.0
    return best if value > 0 and value >= overall * 1.25 else "mixed_or_unclear"


def highlighted_prompt(prompt: str, group: pd.DataFrame) -> str:
    if group.empty:
        return html.escape(prompt)
    pieces = []
    cursor = 0
    max_act = max(float(group["feature_activation"].max()), 1e-9)
    for row in group.sort_values("token_start_char").itertuples(index=False):
        start = int(row.token_start_char)
        end = int(row.token_end_char)
        if end <= start or start < cursor:
            continue
        pieces.append(html.escape(prompt[cursor:start]))
        role = token_role(pd.Series(row._asdict()))
        alpha = min(0.9, max(0.08, float(row.feature_activation) / max_act))
        pieces.append(
            f"<span class='tok {role}' style='background:rgba(46,204,113,{alpha:.2f})' "
            f"title='{role}; act={float(row.feature_activation):.3f}'>{html.escape(prompt[start:end])}</span>"
        )
        cursor = end
    pieces.append(html.escape(prompt[cursor:]))
    return "".join(pieces)


def card_html(layer: int, feature_id: int, group: pd.DataFrame, prepared: pd.DataFrame, meta: dict[str, object], summary: dict[str, object], top_prompts: int, top_tokens: int) -> str:
    behavior = classify_behavior(group)
    prompt_scores = group.groupby("bbq_uid")["feature_activation"].max().sort_values(ascending=False).head(top_prompts)
    prepared_lookup = prepared.set_index("bbq_uid").to_dict("index")
    examples = []
    for uid, score in prompt_scores.items():
        prompt = str(prepared_lookup.get(uid, {}).get("prompt", ""))
        prompt_group = group[group["bbq_uid"].eq(uid)]
        labels = prepared_lookup.get(uid, {})
        examples.append(
            "<section class='example'>"
            f"<h3>{html.escape(str(uid))} <span>max={score:.3f}</span></h3>"
            f"<p><b>Roles:</b> stereotyped={labels.get('stereotyped_answer_idx', '')}, "
            f"nonstereotyped={labels.get('nonstereotyped_answer_idx', '')}, unknown={labels.get('unknown_answer_idx', '')}, "
            f"correct={labels.get('correct_answer_idx', '')}</p>"
            f"<pre>{highlighted_prompt(prompt, prompt_group)}</pre>"
            "</section>"
        )
    top_token_rows = []
    top_token_df = group.sort_values("feature_activation", ascending=False).head(top_tokens).copy()
    for _, row in top_token_df.iterrows():
        top_token_rows.append(
            f"<tr><td>{html.escape(str(row.get('token_str', '')))}</td><td>{float(row.get('feature_activation', 0)):.4f}</td>"
            f"<td>{html.escape(token_role(row))}</td><td>{html.escape(str(row.get('bbq_uid', '')))}</td></tr>"
        )
    breakdown_context = group.groupby("context_condition")["feature_activation"].mean().to_dict() if "context_condition" in group.columns else {}
    breakdown_polarity = group.groupby("question_polarity")["feature_activation"].mean().to_dict() if "question_polarity" in group.columns else {}
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BBQ SAE feature {feature_id}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color:#17202a}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.card,.example{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0;background:#fff}} pre{{white-space:pre-wrap;line-height:1.65;background:#f8fafc;padding:12px;border-radius:8px}}
.tok{{border-radius:3px;padding:1px 2px}} .target_identity{{outline:2px solid #2563eb}} .nontarget_identity{{outline:2px solid #9333ea}} .stereotype_language{{outline:2px solid #dc2626}} .answer_option{{outline:2px solid #f59e0b}} .final_token{{outline:2px solid #111827}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:6px}} th{{background:#f1f5f9}}
</style></head><body>
<h1>BBQ SAE Feature Card: layer {layer}, feature {feature_id}</h1>
<div class="grid">
<div class="card"><h2>Original triage</h2><p>role={html.escape(str(meta.get('provisional_role','')))}<br>contrast={html.escape(str(meta.get('contrast_name', meta.get('mapped_contrast_name',''))))}<br>axis={html.escape(str(meta.get('top_axis','')))}<br>identity={html.escape(str(meta.get('top_identity','')))}<br>priority={html.escape(str(meta.get('intervention_priority','')))}</p></div>
<div class="card"><h2>BBQ behavior</h2><p>behavior={behavior}<br>active prompts={group['bbq_uid'].nunique()}<br>target mean={summary.get('mean_target_identity_activation', 0):.4g}<br>nontarget mean={summary.get('mean_nontarget_identity_activation', 0):.4g}<br>stereotype mean={summary.get('mean_stereotype_language_activation', 0):.4g}<br>final mean={summary.get('mean_final_token_activation', 0):.4g}</p></div>
<div class="card"><h2>Breakdowns</h2><p>context={html.escape(json.dumps(breakdown_context))}<br>polarity={html.escape(json.dumps(breakdown_polarity))}</p></div>
</div>
<h2>Top Activating Tokens</h2><table><tr><th>Token</th><th>Activation</th><th>Token role</th><th>BBQ UID</th></tr>{''.join(top_token_rows)}</table>
<h2>Top Activating BBQ Examples</h2>{''.join(examples)}
</body></html>"""


def main() -> None:
    args = parse_args()
    logger = setup(args.output_dir, args.overwrite, args.resume)
    config = vars(args).copy()
    config.update({k: str(v) for k, v in config.items() if isinstance(v, Path)})
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "bbq_feature_cards_config.json").write_text(json.dumps(config, indent=2) + "\n")
    prepared = read_table(args.prepared_data)
    triage = pd.read_csv(args.triage_csv, low_memory=False)
    summary_path = args.token_level_dir / "bbq_token_level_sae_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    index_rows = []
    for layer in tqdm(parse_ints(args.layers), desc="layers"):
        tokens = load_layer_tokens(args.token_level_dir, layer)
        if tokens.empty:
            logger.warning("No token activations found for layer %02d", layer)
            continue
        layer_dir = args.output_dir / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for i, (feature_id, group) in enumerate(tqdm(tokens.groupby("feature_id", sort=True), desc=f"layer {layer:02d} features", leave=False), start=1):
            path = layer_dir / f"feature_{int(feature_id):05d}.html"
            if args.resume and path.exists():
                continue
            meta_rows = triage[(triage["layer"].eq(layer)) & (triage["feature_id"].eq(feature_id))]
            meta = meta_rows.iloc[0].to_dict() if not meta_rows.empty else {}
            summary_rows = summary[(summary["layer"].eq(layer)) & (summary["feature_id"].eq(feature_id))]
            summary_meta = summary_rows.iloc[0].to_dict() if not summary_rows.empty else {}
            behavior = classify_behavior(group)
            path.write_text(card_html(layer, int(feature_id), group, prepared, meta, summary_meta, args.top_prompts_per_feature, args.top_tokens_per_feature))
            index_rows.append({
                "feature_id": int(feature_id),
                "layer": layer,
                "role": meta.get("provisional_role", ""),
                "contrast_name": meta.get("contrast_name", meta.get("mapped_contrast_name", "")),
                "top_axis": meta.get("top_axis", ""),
                "mean_target_identity_activation": summary_meta.get("mean_target_identity_activation", float("nan")),
                "mean_nontarget_identity_activation": summary_meta.get("mean_nontarget_identity_activation", float("nan")),
                "mean_stereotype_language_activation": summary_meta.get("mean_stereotype_language_activation", float("nan")),
                "mean_final_token_activation": summary_meta.get("mean_final_token_activation", float("nan")),
                "bbq_behavior_role": behavior,
                "link": str(path.relative_to(args.output_dir)),
            })
            if args.save_every_features and i % args.save_every_features == 0:
                pd.DataFrame(index_rows).to_csv(args.output_dir / "feature_card_index.partial.csv", index=False)
    index = pd.DataFrame(index_rows)
    index.to_csv(args.output_dir / "feature_card_index.csv", index=False)
    rows = "\n".join(
        f"<tr><td>{r.layer}</td><td>{r.feature_id}</td><td>{html.escape(str(r.role))}</td><td>{html.escape(str(r.contrast_name))}</td><td>{html.escape(str(r.top_axis))}</td><td>{html.escape(str(r.bbq_behavior_role))}</td><td><a href='{html.escape(str(r.link))}'>card</a></td></tr>"
        for r in index.itertuples(index=False)
    )
    (args.output_dir / "index.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>BBQ SAE Feature Cards</title><style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:6px}}th{{background:#f5f5f5}}</style></head><body><h1>BBQ SAE Feature Cards</h1><table><tr><th>Layer</th><th>Feature</th><th>Role</th><th>Contrast</th><th>Axis</th><th>BBQ behavior</th><th>Link</th></tr>{rows}</table></body></html>")
    logger.info("Feature cards written to %s", args.output_dir)


if __name__ == "__main__":
    main()
