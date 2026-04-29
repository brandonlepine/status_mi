#!/usr/bin/env python3
"""Prepare BBQ examples for SAE steering experiments."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


DEFAULT_BBQ_DIR = Path("/workspace/status_mi/data/bbq/data")
DEFAULT_FORMS_CSV = Path("/workspace/status_mi/data/bbq_identity_normalized_forms.csv")
DEFAULT_TRIAGE_CSV = Path("/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared")

AXIS_MAP = {
    "disability_status": "disability_status",
    "gender_identity": "gender_identity",
    "nationality": "nationality",
    "physical_appearance": "physical_appearance",
    "race_ethnicity": "race_ethnicity",
    "race_x_gender": "race_ethnicity",
    "race_x_ses": "race_ethnicity",
    "religion": "religion",
    "sexual_orientation": "sexual_orientation",
    "ses": "socioeconomic_status",
    "socioeconomic_status": "socioeconomic_status",
}
UNKNOWN_ALIASES = {
    "unknown", "not answerable", "can't answer", "cannot answer", "can't be determined",
    "cannot be determined", "not enough information", "not known", "not applicable",
}
MANUAL_ALIASES = {
    "disabled": "disability_disabled",
    "physically disabled": "disability_physically_disabled",
    "wheelchair": "disability_physically_disabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled person": "disability_nondisabled",
    "non disabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "non disabled": "disability_nondisabled",
    "non-disabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "nondisabled": "disability_nondisabled",
    "able bodied": "disability_able_bodied",
    "able-bodied": "disability_able_bodied",
    "man": "gender_man",
    "m": "gender_man",
    "male": "gender_man",
    "woman": "gender_woman",
    "f": "gender_woman",
    "female": "gender_woman",
    "fat": "appearance_obese",
    "obese": "appearance_obese",
    "nonobese": "appearance_thin",
    "non obese": "appearance_thin",
    "slim": "appearance_thin",
    "thin": "appearance_thin",
    "black": "race_black",
    "african american": "race_black",
    "white": "race_white",
    "caucasian": "race_caucasian",
    "asian": "race_asian",
    "hispanic": "race_hispanic",
    "latino": "race_latino",
    "native american": "race_native_american",
    "muslim": "religion_muslim",
    "jewish": "religion_jewish",
    "christian": "religion_christian",
    "gay": "sexuality_gay",
    "straight": "sexuality_straight",
    "heterosexual": "sexuality_heterosexual",
    "lesbian": "sexuality_lesbian",
    "bisexual": "sexuality_bisexual",
    "low ses": "ses_low_income",
    "low socioeconomic status": "ses_low_income",
    "low income": "ses_low_income",
    "poor": "ses_low_income",
    "lowses": "ses_low_income",
    "rich": "ses_rich",
    "high socioeconomic status": "ses_high_socioeconomic_status",
    "highses": "ses_high_socioeconomic_status",
    "upper class": "ses_upper_class",
    "lower class": "ses_lower_class",
    "old": "age_old",
    "nonold": "age_nonold",
    "non old": "age_nonold",
    "african": "race_black",
    "arab": "race_arab",
    "asia pacific": "nationality_asia_pacific",
    "asiapacific": "nationality_asia_pacific",
    "africa": "nationality_african",
    "europe": "nationality_european",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BBQ examples for SAE steering.")
    parser.add_argument("--bbq_data_dir", type=Path, default=DEFAULT_BBQ_DIR)
    parser.add_argument("--identity_forms_csv", type=Path, default=DEFAULT_FORMS_CSV)
    parser.add_argument("--triage_csv", type=Path, default=DEFAULT_TRIAGE_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", default=None, help="Optional comma-separated BBQ category filenames/stems.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--save_every_examples", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_output(output_dir: Path, overwrite: bool, resume: bool) -> logging.Logger:
    if output_dir.exists() and any(output_dir.iterdir()) and overwrite and not resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    logger = logging.getLogger("prepare_bbq")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(output_dir / "logs" / "prepare_bbq.log")]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def norm_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def axis_from_category(category: str) -> str:
    return AXIS_MAP.get(norm_text(category).replace(" ", "_"), norm_text(category).replace(" ", "_"))


def load_identity_aliases(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    aliases: dict[str, str] = {}
    metadata: dict[str, dict[str, str]] = {}
    if path.exists():
        forms = pd.read_csv(path, keep_default_na=False)
        for row in forms.to_dict("records"):
            identity_id = str(row.get("identity_id", ""))
            metadata[identity_id] = {k: str(v) for k, v in row.items()}
            for col in [
                "identity_id", "canonical_label", "adj_form", "noun_form", "person_noun_form",
                "plural_noun_form", "group_form", "prep_form", "with_form", "has_form",
            ]:
                value = row.get(col, "")
                if str(value).strip():
                    aliases[norm_text(value)] = identity_id
            for alias in str(row.get("aliases", "")).split(";"):
                if alias.strip():
                    aliases[norm_text(alias)] = identity_id
    aliases.update({norm_text(k): v for k, v in MANUAL_ALIASES.items()})
    return aliases, metadata


def parse_contrast_identity_ids(name: str) -> tuple[str, str] | None:
    if not isinstance(name, str) or "_vs_" not in name:
        return None
    left, right = name.split("_vs_", 1)
    return left, right


def load_contrasts(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["mapped_contrast_name", "identity_a", "identity_b", "axis"])
    triage = pd.read_csv(path, low_memory=False)
    contrast_col = "contrast_name" if "contrast_name" in triage.columns else "mapped_contrast_name"
    if contrast_col not in triage.columns:
        return pd.DataFrame(columns=["mapped_contrast_name", "identity_a", "identity_b", "axis"])
    rows = []
    for contrast_name, group in triage.dropna(subset=[contrast_col]).groupby(contrast_col):
        parsed = parse_contrast_identity_ids(str(contrast_name))
        if not parsed:
            continue
        axis = ""
        for col in ["axis", "top_axis", "axis_mapped"]:
            if col in group.columns and group[col].notna().any():
                axis = str(group[col].dropna().iloc[0])
                break
        rows.append({"mapped_contrast_name": str(contrast_name), "identity_a": parsed[0], "identity_b": parsed[1], "axis": axis})
    return pd.DataFrame(rows).drop_duplicates()


def read_bbq_files(bbq_dir: Path, categories: str | None, logger: logging.Logger) -> list[tuple[Path, dict[str, Any]]]:
    wanted = {norm_text(part).replace(" ", "_") for part in categories.split(",")} if categories else None
    files = sorted(bbq_dir.glob("*.jsonl"))
    rows = []
    for path in files:
        stem_norm = norm_text(path.stem).replace(" ", "_")
        if wanted and stem_norm not in wanted:
            continue
        logger.info("Loading %s", path)
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append((path, json.loads(line)))
    return rows


def prompt_for(row: dict[str, Any]) -> str:
    return (
        f"Context:\n{row.get('context', '')}\n\n"
        f"Question:\n{row.get('question', '')}\n\n"
        "Answer choices:\n"
        f"A. {row.get('ans0', '')}\n"
        f"B. {row.get('ans1', '')}\n"
        f"C. {row.get('ans2', '')}\n\n"
        "Answer:"
    )


def identity_for(label: object, aliases: dict[str, str]) -> tuple[str, str, str]:
    label_norm = norm_text(label)
    if label_norm in UNKNOWN_ALIASES or label_norm == "unknown":
        return "unknown", "unknown", "unknown"
    if label_norm in aliases:
        return aliases[label_norm], "alias", label_norm
    compact = label_norm.replace(" ", "")
    for alias, identity_id in aliases.items():
        if alias.replace(" ", "") == compact:
            return identity_id, "alias", label_norm
    for component in identity_components(label):
        if component in aliases:
            return aliases[component], "component_alias", label_norm
    return "", "unmapped", label_norm


def identity_components(label: object) -> list[str]:
    """Split BBQ compound labels such as F-Black and lowSES-Hispanic."""
    raw = str(label or "").strip()
    pieces = [raw]
    pieces.extend(re.split(r"[-_/+]", raw))
    normalized = []
    for piece in pieces:
        norm = norm_text(piece)
        if norm:
            normalized.append(norm)
        compact = norm.replace(" ", "")
        if compact and compact != norm:
            normalized.append(compact)
    component_aliases = {
        "f": "female",
        "m": "male",
        "nonold": "nonold",
        "lowses": "lowses",
        "highses": "highses",
    }
    normalized.extend(component_aliases.get(item, item) for item in list(normalized))
    return list(dict.fromkeys(normalized))


def identity_component_ids(label: object, aliases: dict[str, str]) -> set[str]:
    ids = set()
    for component in identity_components(label):
        identity_id = aliases.get(component)
        if identity_id:
            ids.add(identity_id)
    return ids


def identity_axis(identity_id: str) -> str:
    if identity_id.startswith("disability_"):
        return "disability_status"
    if identity_id.startswith(("gender_", "sex_")):
        return "gender_identity"
    if identity_id.startswith("appearance_"):
        return "physical_appearance"
    if identity_id.startswith("race_"):
        return "race_ethnicity"
    if identity_id.startswith("religion_"):
        return "religion"
    if identity_id.startswith("sexuality_"):
        return "sexual_orientation"
    if identity_id.startswith("ses_"):
        return "socioeconomic_status"
    if identity_id.startswith("nationality_"):
        return "nationality"
    if identity_id.startswith("age_"):
        return "age"
    return identity_id.split("_", 1)[0] if "_" in identity_id else ""


def choose_identity_for_role(answer_row: dict[str, Any], preferred_ids: set[str] | None = None, preferred_axis: str | None = None) -> str:
    component_ids = set(answer_row.get("component_identity_ids", []))
    identity_id = str(answer_row.get("identity_id", ""))
    candidates = [identity_id] + sorted(component_ids)
    candidates = [candidate for candidate in candidates if candidate and candidate != "unknown"]
    if preferred_ids:
        for candidate in candidates:
            if candidate in preferred_ids:
                return candidate
    if preferred_axis:
        for candidate in candidates:
            if identity_axis(candidate) == preferred_axis:
                return candidate
    return candidates[0] if candidates else ""


def find_answer_indices(answer_info: dict[str, Any], stereotyped_groups: list[str], aliases: dict[str, str]) -> dict[str, Any]:
    answer_rows = []
    stereo_norms = {norm_text(item) for item in stereotyped_groups}
    stereo_ids = {identity_for(item, aliases)[0] for item in stereotyped_groups if identity_for(item, aliases)[0]}
    for item in stereotyped_groups:
        stereo_ids.update(identity_component_ids(item, aliases))
    for idx in range(3):
        key = f"ans{idx}"
        info = answer_info.get(key, ["", ""])
        answer_text = info[0] if isinstance(info, list) and info else ""
        group_label = info[1] if isinstance(info, list) and len(info) > 1 else answer_text
        identity_id, map_status, normalized = identity_for(group_label, aliases)
        component_ids = identity_component_ids(group_label, aliases) | identity_component_ids(answer_text, aliases)
        is_unknown = identity_id == "unknown" or norm_text(answer_text) in UNKNOWN_ALIASES
        is_stereo = normalized in stereo_norms or identity_id in stereo_ids or bool(component_ids & stereo_ids)
        if not is_stereo:
            answer_components = set(identity_components(group_label)) | set(identity_components(answer_text))
            is_stereo = bool(answer_components & stereo_norms)
        answer_rows.append({
            "idx": idx,
            "answer_text": answer_text,
            "group_label": group_label,
            "identity_id": identity_id,
            "component_identity_ids": sorted(component_ids),
            "map_status": map_status,
            "is_unknown": is_unknown,
            "is_stereo": is_stereo,
        })
    unknown = next((r["idx"] for r in answer_rows if r["is_unknown"]), None)
    stereo = next((r["idx"] for r in answer_rows if r["is_stereo"] and not r["is_unknown"]), None)
    nonstereo = next((r["idx"] for r in answer_rows if not r["is_stereo"] and not r["is_unknown"]), None)
    target = answer_rows[stereo] if stereo is not None else {}
    nontarget = answer_rows[nonstereo] if nonstereo is not None else {}
    target_identity_id = choose_identity_for_role(target, preferred_ids=stereo_ids)
    target_axis = identity_axis(target_identity_id)
    nontarget_identity_id = choose_identity_for_role(nontarget, preferred_axis=target_axis)
    return {
        "answer_rows": answer_rows,
        "unknown_answer_idx": unknown,
        "stereotyped_answer_idx": stereo,
        "nonstereotyped_answer_idx": nonstereo,
        "target_identity_id": target_identity_id,
        "target_identity_label": target.get("group_label", ""),
        "target_answer_idx": stereo,
        "nontarget_identity_id": nontarget_identity_id,
        "nontarget_identity_label": nontarget.get("group_label", ""),
        "nontarget_answer_idx": nonstereo,
    }


def map_contrast(target_id: str, nontarget_id: str, axis: str, contrasts: pd.DataFrame) -> tuple[str, str]:
    if contrasts.empty or not target_id or not nontarget_id:
        return "", "unmapped"
    exact = contrasts[
        ((contrasts["identity_a"].eq(target_id)) & (contrasts["identity_b"].eq(nontarget_id)))
        | ((contrasts["identity_a"].eq(nontarget_id)) & (contrasts["identity_b"].eq(target_id)))
    ]
    if not exact.empty:
        return str(exact.iloc[0]["mapped_contrast_name"]), "exact"
    axis_matches = contrasts[contrasts["axis"].astype(str).eq(axis)]
    for _, row in axis_matches.iterrows():
        if target_id in {row.identity_a, row.identity_b} or nontarget_id in {row.identity_a, row.identity_b}:
            return str(row.mapped_contrast_name), "fallback_axis"
    if not axis_matches.empty:
        return str(axis_matches.iloc[0]["mapped_contrast_name"]), "fallback_axis"
    return "", "unmapped"


def write_table(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    try:
        df.to_parquet(output_dir / f"{stem}.parquet", index=False)
    except Exception as exc:
        logging.getLogger("prepare_bbq").warning("Could not write %s.parquet: %s", stem, exc)


def main() -> None:
    args = parse_args()
    logger = setup_output(args.output_dir, args.overwrite, args.resume)
    config = vars(args).copy()
    config.update({k: str(v) for k, v in config.items() if isinstance(v, Path)})
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "bbq_prepare_config.json").write_text(json.dumps(config, indent=2) + "\n")

    aliases, identity_meta = load_identity_aliases(args.identity_forms_csv)
    contrasts = load_contrasts(args.triage_csv)
    raw_rows = read_bbq_files(args.bbq_data_dir, args.categories, logger)
    if args.max_examples:
        raw_rows = raw_rows[: args.max_examples]
    logger.info("Loaded %d BBQ rows from %s", len(raw_rows), args.bbq_data_dir)

    out_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    checkpoint_path = args.output_dir / "bbq_prepared_examples.partial.csv"
    for global_idx, (path, row) in enumerate(tqdm(raw_rows, desc="prepare BBQ")):
        answer_info = row.get("answer_info", {})
        additional = row.get("additional_metadata", {}) or {}
        stereotyped_groups = additional.get("stereotyped_groups", []) or []
        axis = axis_from_category(str(row.get("category", path.stem)))
        answer_indices = find_answer_indices(answer_info, stereotyped_groups, aliases)
        mapped_contrast, confidence = map_contrast(
            str(answer_indices["target_identity_id"]),
            str(answer_indices["nontarget_identity_id"]),
            axis,
            contrasts,
        )
        notes = []
        if answer_indices["unknown_answer_idx"] is None:
            notes.append("missing_unknown_answer")
        if answer_indices["stereotyped_answer_idx"] is None:
            notes.append("missing_stereotyped_answer")
        if confidence == "unmapped":
            notes.append("unmapped_contrast")
        uid = f"{path.stem}:{row.get('example_id', global_idx)}:{row.get('question_index', '')}:{row.get('context_condition', '')}:{row.get('question_polarity', '')}:{global_idx}"
        prepared = {
            "bbq_uid": uid,
            "source_file": path.name,
            "example_id": row.get("example_id", global_idx),
            "question_index": row.get("question_index", ""),
            "category_raw": row.get("category", path.stem),
            "axis_mapped": axis,
            "subcategory": additional.get("subcategory", ""),
            "context_condition": row.get("context_condition", ""),
            "question_polarity": row.get("question_polarity", ""),
            "context": row.get("context", ""),
            "question": row.get("question", ""),
            "ans0": row.get("ans0", ""),
            "ans1": row.get("ans1", ""),
            "ans2": row.get("ans2", ""),
            "label": row.get("label", math.nan),
            "answer_info_json": json.dumps(answer_info, ensure_ascii=False),
            "stereotyped_groups_json": json.dumps(stereotyped_groups, ensure_ascii=False),
            "prompt": prompt_for(row),
            "unknown_answer_idx": answer_indices["unknown_answer_idx"],
            "stereotyped_answer_idx": answer_indices["stereotyped_answer_idx"],
            "nonstereotyped_answer_idx": answer_indices["nonstereotyped_answer_idx"],
            "correct_answer_idx": row.get("label", math.nan),
            "target_identity_label": answer_indices["target_identity_label"],
            "target_identity_id": answer_indices["target_identity_id"],
            "target_answer_idx": answer_indices["target_answer_idx"],
            "nontarget_identity_label": answer_indices["nontarget_identity_label"],
            "nontarget_identity_id": answer_indices["nontarget_identity_id"],
            "nontarget_answer_idx": answer_indices["nontarget_answer_idx"],
            "mapped_contrast_name": mapped_contrast,
            "mapped_contrast_confidence": confidence,
            "polarity_role": f"{row.get('context_condition', '')}_{row.get('question_polarity', '')}",
            "notes": ";".join(notes),
        }
        out_rows.append(prepared)
        if notes:
            diagnostics.append({
                "bbq_uid": uid,
                "category_raw": prepared["category_raw"],
                "axis_mapped": axis,
                "target_identity_label": prepared["target_identity_label"],
                "target_identity_id": prepared["target_identity_id"],
                "nontarget_identity_label": prepared["nontarget_identity_label"],
                "nontarget_identity_id": prepared["nontarget_identity_id"],
                "mapped_contrast_confidence": confidence,
                "notes": ";".join(notes),
                "answer_info_json": prepared["answer_info_json"],
                "stereotyped_groups_json": prepared["stereotyped_groups_json"],
            })
        if args.save_every_examples and len(out_rows) % args.save_every_examples == 0:
            pd.DataFrame(out_rows).to_csv(checkpoint_path, index=False)

    prepared_df = pd.DataFrame(out_rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    write_table(prepared_df, args.output_dir, "bbq_prepared_examples")
    diagnostics_df.to_csv(args.output_dir / "bbq_mapping_diagnostics.csv", index=False)
    contrasts.to_csv(args.output_dir / "bbq_contrast_mapping.csv", index=False)

    summary_rows = []
    if not prepared_df.empty:
        summary_rows.extend(
            {"metric": f"n_{k}", "value": int(v)}
            for k, v in prepared_df.groupby(["category_raw", "context_condition", "question_polarity"]).size().items()
        )
        for metric, col in [
            ("fraction_unknown_answer_found", "unknown_answer_idx"),
            ("fraction_stereotyped_answer_found", "stereotyped_answer_idx"),
        ]:
            summary_rows.append({"metric": metric, "value": float(prepared_df[col].notna().mean())})
        mapped_rate = float(prepared_df["mapped_contrast_confidence"].isin(["exact", "alias", "fallback_axis"]).mean())
        summary_rows.append({"metric": "fraction_mapped_to_contrast", "value": mapped_rate})
        confidence_counts = prepared_df["mapped_contrast_confidence"].value_counts(dropna=False).to_dict()
        for key, value in confidence_counts.items():
            summary_rows.append({"metric": f"mapping_confidence_{key}", "value": int(value)})
        if mapped_rate < 0.7:
            logger.warning("Only %.1f%% of BBQ examples mapped to a contrast. Inspect bbq_mapping_diagnostics.csv.", mapped_rate * 100)
        logger.info("Loaded examples by category/context/polarity:\n%s", prepared_df.groupby(["category_raw", "context_condition", "question_polarity"]).size())
        logger.info("Unknown-answer found: %.1f%%", prepared_df["unknown_answer_idx"].notna().mean() * 100)
        logger.info("Stereotyped-answer found: %.1f%%", prepared_df["stereotyped_answer_idx"].notna().mean() * 100)
        logger.info("Mapped to contrast: %.1f%%", mapped_rate * 100)
        failed = Counter(
            (row.get("target_identity_label", ""), row.get("nontarget_identity_label", ""))
            for row in diagnostics
            if "unmapped_contrast" in row.get("notes", "")
        )
        logger.info("Top unmapped identity pairs: %s", failed.most_common(20))
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "bbq_prepare_summary.csv", index=False)
    logger.info("Prepared data written to %s", args.output_dir)


if __name__ == "__main__":
    main()
