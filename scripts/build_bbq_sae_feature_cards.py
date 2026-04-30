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


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def content_tokens(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty or not {"token_start_char", "token_end_char"}.issubset(group.columns):
        return group.copy()
    out = group[
        pd.to_numeric(group["token_end_char"], errors="coerce").fillna(0)
        > pd.to_numeric(group["token_start_char"], errors="coerce").fillna(0)
    ].copy()
    special = out["token_str"].astype(str).str.contains(r"<\|.*\|>", regex=True, na=False) if "token_str" in out.columns else pd.Series(False, index=out.index)
    return out[~special].copy()


def token_role(row: pd.Series) -> str:
    if as_bool(row.get("is_target_identity_token", False)):
        return "target_identity"
    if as_bool(row.get("is_nontarget_identity_token", False)):
        return "nontarget_identity"
    if as_bool(row.get("is_stereotype_language_token", False)):
        return "stereotype_language"
    if as_bool(row.get("is_answer_option_token", False)):
        return "answer_option"
    if as_bool(row.get("is_final_prompt_token", False)):
        return "final_token"
    return "other_context"


def classify_behavior(group: pd.DataFrame) -> str:
    group = content_tokens(group)
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


def mean_on(group: pd.DataFrame, flag_col: str) -> float:
    group = content_tokens(group)
    vals = group.loc[bool_col(group, flag_col), "feature_activation"] if flag_col in group.columns else pd.Series(dtype=float)
    return 0.0 if vals.empty or pd.isna(vals.mean()) else float(vals.mean())


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def answer_label(idx: object, labels: dict[str, object]) -> str:
    if pd.isna(idx):
        return "not identified"
    i = int(float(idx))
    letter = chr(65 + i) if 0 <= i <= 2 else str(i)
    return f"{letter}. {labels.get(f'ans{i}', '')}"


def parsed_uid(uid: object, labels: dict[str, object]) -> str:
    return (
        f"Category: {html.escape(str(labels.get('category_raw', '')))} | "
        f"Example ID: {html.escape(str(labels.get('example_id', '')))} | "
        f"Question index: {html.escape(str(labels.get('question_index', '')))} | "
        f"Context: {html.escape(str(labels.get('context_condition', '')))} | "
        f"Polarity: {html.escape(str(labels.get('question_polarity', '')))}"
    )


def metric_table(rows: list[tuple[str, str, str]]) -> str:
    return "<table class='metric-table'>" + "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td><td class='explain'>{html.escape(explanation)}</td></tr>"
        for label, value, explanation in rows
    ) + "</table>"


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
    group = content_tokens(group)
    behavior = classify_behavior(group)
    prompt_scores = group.groupby("bbq_uid")["feature_activation"].max().sort_values(ascending=False).head(top_prompts) if not group.empty else pd.Series(dtype=float)
    prepared_lookup = prepared.set_index("bbq_uid").to_dict("index")
    examples = []
    for uid, score in prompt_scores.items():
        prompt = str(prepared_lookup.get(uid, {}).get("prompt", ""))
        prompt_group = group[group["bbq_uid"].eq(uid)]
        labels = prepared_lookup.get(uid, {})
        answer_roles = metric_table([
            ("Stereotype-consistent answer", answer_label(labels.get("stereotyped_answer_idx"), labels), "The BBQ answer option whose group is listed in stereotyped_groups. This is metadata, not a model prediction."),
            ("Non-stereotyped / contrast answer", answer_label(labels.get("nonstereotyped_answer_idx"), labels), "The other identity answer when one is identifiable."),
            ("Unknown answer", answer_label(labels.get("unknown_answer_idx"), labels), "The normatively correct option for ambiguous BBQ items."),
            ("Gold/correct answer", answer_label(labels.get("correct_answer_idx"), labels), "The dataset label. For ambiguous items this should usually be the unknown answer."),
        ])
        examples.append(
            "<section class='example'>"
            f"<h3>{parsed_uid(uid, labels)}</h3>"
            f"<p><b>Maximum content-token activation in this example:</b> {score:.3f}</p>"
            f"{answer_roles}"
            f"<pre>{highlighted_prompt(prompt, prompt_group)}</pre>"
            "</section>"
        )
    top_token_rows = []
    top_token_df = group.sort_values("feature_activation", ascending=False).head(top_tokens).copy()
    for _, row in top_token_df.iterrows():
        top_token_rows.append(
            f"<tr><td>{html.escape(str(row.get('token_str', '')))}</td><td>{float(row.get('feature_activation', 0)):.4f}</td>"
            f"<td>{html.escape(token_role(row))}</td><td>{parsed_uid(row.get('bbq_uid', ''), prepared_lookup.get(row.get('bbq_uid', ''), {}))}</td></tr>"
        )
    breakdown_context = group.groupby("context_condition")["feature_activation"].mean().to_dict() if "context_condition" in group.columns and not group.empty else {}
    breakdown_polarity = group.groupby("question_polarity")["feature_activation"].mean().to_dict() if "question_polarity" in group.columns and not group.empty else {}
    context_rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{fmt_float(v)}</td></tr>" for k, v in breakdown_context.items())
    polarity_rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{fmt_float(v)}</td></tr>" for k, v in breakdown_polarity.items())
    target_mean = mean_on(group, "is_target_identity_token")
    nontarget_mean = mean_on(group, "is_nontarget_identity_token")
    stereotype_mean = mean_on(group, "is_stereotype_language_token")
    final_mean = mean_on(group, "is_final_prompt_token")
    active_prompts = group["bbq_uid"].nunique() if "bbq_uid" in group.columns else 0
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BBQ SAE feature {feature_id}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color:#17202a}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
.card,.example{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0;background:#fff}} pre{{white-space:pre-wrap;line-height:1.75;background:#f8fafc;padding:12px;border-radius:8px}}
.tok{{border-radius:3px;padding:1px 2px}} .target_identity{{outline:2px solid #2563eb}} .nontarget_identity{{outline:2px solid #9333ea}} .stereotype_language{{outline:2px solid #dc2626}} .answer_option{{outline:2px solid #f59e0b}} .final_token{{outline:2px solid #111827}}
table{{border-collapse:collapse;width:100%;margin:8px 0}} th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}} th{{background:#f1f5f9}} .metric-table th{{width:220px;text-align:left}} .explain{{color:#475569;font-size:13px}} .legend span{{display:inline-block;margin:4px 10px 4px 0;padding:3px 6px;border-radius:4px;background:#f8fafc}} .note{{color:#475569;line-height:1.45}}
</style></head><body>
<h1>BBQ SAE Feature Card: layer {layer}, feature {feature_id}</h1>
<p class="note">These cards summarize SAE activations on BBQ prompts. They do not show steering results or repeated trials. Green background intensity is token activation relative to the maximum content-token activation within that displayed example. Colored outlines indicate token role.</p>
<div class="legend"><b>Token role legend:</b>
<span class="target_identity">target identity</span>
<span class="nontarget_identity">non-target identity</span>
<span class="stereotype_language">stereotype-language/question term</span>
<span class="answer_option">answer option</span>
<span class="final_token">final prompt token</span>
</div>
<div class="grid">
<div class="card"><h2>Original identity-feature triage</h2>{metric_table([
("Provisional role", str(meta.get('provisional_role','')), "Role assigned from the clean identity-prompt SAE triage."),
("Source contrast", str(meta.get('contrast_name', meta.get('mapped_contrast_name',''))), "Identity contrast where this SAE feature was selected."),
("Top axis", str(meta.get('top_axis','')), "Identity axis from the original triage."),
("Top identity", str(meta.get('top_identity','')), "Most associated identity from the original triage, if available."),
("Intervention priority", str(meta.get('intervention_priority','')), "Priority assigned before looking at BBQ."),
])}</div>
<div class="card"><h2>BBQ activation behavior</h2>{metric_table([
("Behavior label", behavior, "Heuristic label based on where this feature most strongly activates in BBQ prompts."),
("Active BBQ prompts", str(active_prompts), "Number of BBQ prompts with at least one positive non-special token activation for this feature."),
("Mean on target identity tokens", fmt_float(target_mean), "Average activation on tokens naming the stereotype-targeted identity/group in the BBQ metadata."),
("Mean on non-target identity tokens", fmt_float(nontarget_mean), "Average activation on tokens naming the contrast identity/group, when identifiable."),
("Mean on stereotype-language tokens", fmt_float(stereotype_mean), "Average activation on content words in the BBQ question, such as 'criminal', 'lazy', or 'secretary'. This is not model choice behavior."),
("Mean on final prompt token", fmt_float(final_mean), "Average activation at the final content token before the model answers. Higher values suggest a more integrated prompt-level feature."),
])}</div>
<div class="card"><h2>Activation breakdowns</h2><p class="note">Values are mean positive feature activations over non-special content tokens, grouped by BBQ metadata.</p><h3>Context condition</h3><table><tr><th>Condition</th><th>Mean activation</th></tr>{context_rows}</table><h3>Question polarity</h3><table><tr><th>Polarity</th><th>Mean activation</th></tr>{polarity_rows}</table></div>
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
            content_group = content_tokens(group)
            index_rows.append({
                "feature_id": int(feature_id),
                "layer": layer,
                "role": meta.get("provisional_role", ""),
                "contrast_name": meta.get("contrast_name", meta.get("mapped_contrast_name", "")),
                "top_axis": meta.get("top_axis", ""),
                "mean_target_identity_activation": mean_on(content_group, "is_target_identity_token"),
                "mean_nontarget_identity_activation": mean_on(content_group, "is_nontarget_identity_token"),
                "mean_stereotype_language_activation": mean_on(content_group, "is_stereotype_language_token"),
                "mean_final_token_activation": mean_on(content_group, "is_final_prompt_token"),
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
