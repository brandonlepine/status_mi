"""Canonical identity-contrast registry.

Single source of truth for the contrast pairs used by every geometry / SAE
feature analysis and plot script. Audit issue 4.1: prior copies of `CONTRASTS`
/ `DEFAULT_CONTRASTS` were duplicated across six scripts and contained
identity IDs that did not exist in `data/bbq_identity_normalized_forms.csv`
(`ses_low_income`, `ses_high_socioeconomic_status`); those contrasts were
silently filtered out by per-script `if id not in identity_set: continue`
loops, so the SES axis quietly ran with two contrasts when the code implied
four.

This module fixes the typos against the actual dataset (`ses_low_income` ->
`ses_low`, `ses_high_socioeconomic_status` -> `ses_high`) and exposes a
single loader (`load_validated_contrasts`) that returns both the contrasts
that validate against a given identity CSV and a separate DataFrame of
contrasts that were skipped (with a `reason` column). Callers should write
the skipped DataFrame to disk so partial-axis runs leave an audit trail.

There is no startup assertion that every contrast's identities are present
in the input CSV. By design — running the pipeline on a single axis (e.g.,
to validate end-to-end behavior before a full run) should not fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# Canonical contrasts. The 4-tuple form (name, identity_a, identity_b, axis)
# is the superset; callers that want the 2-tuple (identity_a, identity_b) form
# can use `get_contrast_pairs(...)`.
#
# All identity IDs in this list MUST appear in
# `data/bbq_identity_normalized_forms.csv`. The two former typos
# (ses_low_income, ses_high_socioeconomic_status) are corrected here.
CONTRASTS: list[tuple[str, str, str, str]] = [
    # race_ethnicity
    ("race_black_vs_race_white",      "race_black",      "race_white",      "race_ethnicity"),
    ("race_black_vs_race_asian",      "race_black",      "race_asian",      "race_ethnicity"),
    ("race_black_vs_race_caucasian",  "race_black",      "race_caucasian",  "race_ethnicity"),
    # sexual_orientation
    ("sexuality_gay_vs_sexuality_straight",        "sexuality_gay",       "sexuality_straight",     "sexual_orientation"),
    ("sexuality_gay_vs_sexuality_heterosexual",    "sexuality_gay",       "sexuality_heterosexual", "sexual_orientation"),
    ("sexuality_lesbian_vs_sexuality_straight",    "sexuality_lesbian",   "sexuality_straight",     "sexual_orientation"),
    ("sexuality_bisexual_vs_sexuality_straight",   "sexuality_bisexual",  "sexuality_straight",     "sexual_orientation"),
    # disability_status
    ("disability_disabled_vs_disability_nondisabled", "disability_disabled", "disability_nondisabled", "disability_status"),
    ("disability_disabled_vs_disability_able_bodied", "disability_disabled", "disability_able_bodied", "disability_status"),
    # physical_appearance
    ("appearance_short_vs_appearance_tall",                       "appearance_short",          "appearance_tall",          "physical_appearance"),
    ("appearance_obese_vs_appearance_thin",                       "appearance_obese",          "appearance_thin",          "physical_appearance"),
    ("appearance_poorly_dressed_vs_appearance_well_dressed",      "appearance_poorly_dressed", "appearance_well_dressed",  "physical_appearance"),
    # socioeconomic_status — typos fixed: ses_low_income -> ses_low, ses_high_socioeconomic_status -> ses_high
    ("ses_low_vs_ses_rich",                "ses_low",          "ses_rich",         "socioeconomic_status"),
    ("ses_low_vs_ses_high",                "ses_low",          "ses_high",         "socioeconomic_status"),
    ("ses_lower_class_vs_ses_upper_class", "ses_lower_class",  "ses_upper_class",  "socioeconomic_status"),
    ("ses_blue_collar_vs_ses_white_collar","ses_blue_collar",  "ses_white_collar", "socioeconomic_status"),
    # gender_identity
    ("gender_transgender_vs_gender_cisgender",             "gender_transgender",        "gender_cisgender",        "gender_identity"),
    ("gender_transgender_man_vs_gender_cisgender_man",     "gender_transgender_man",    "gender_cisgender_man",    "gender_identity"),
    ("gender_transgender_woman_vs_gender_cisgender_woman", "gender_transgender_woman",  "gender_cisgender_woman",  "gender_identity"),
    # religion
    ("religion_muslim_vs_religion_christian", "religion_muslim", "religion_christian", "religion"),
    ("religion_jewish_vs_religion_christian", "religion_jewish", "religion_christian", "religion"),
]

# Contrasts that should always anchor headline plots (subset of CONTRASTS by name).
# Used by the plotting scripts; the registry's loader will only return the rows
# of CONTRASTS that match these names (after the validity check), in order.
KEY_CONTRAST_NAMES: list[str] = [
    "sexuality_gay_vs_sexuality_straight",
    "race_black_vs_race_white",
    "gender_transgender_vs_gender_cisgender",
    "appearance_obese_vs_appearance_thin",
    "ses_low_vs_ses_rich",
    "disability_disabled_vs_disability_able_bodied",
]

# Cross-axis ordering pairs (contrast_name, target_axis_for_projection) used by
# analyze_shared_social_subspace.py:cross_axis_projection_rows.
SELECTED_CROSS_AXIS_ORDERINGS: list[tuple[str, str]] = [
    ("appearance_poorly_dressed_vs_appearance_well_dressed", "socioeconomic_status"),
    ("ses_low_vs_ses_rich",                                  "race_ethnicity"),
    ("appearance_poorly_dressed_vs_appearance_well_dressed", "gender_identity"),
    ("ses_lower_class_vs_ses_upper_class",                   "physical_appearance"),
]


@dataclass
class ContrastValidationResult:
    """Result of validating the registry against a particular identity dataset.

    Both DataFrames have columns `contrast_name, identity_a, identity_b, axis`.
    `skipped` additionally has a `reason` column describing why the contrast
    was excluded ("missing_identity_a", "missing_identity_b", or both).
    """
    valid: pd.DataFrame
    skipped: pd.DataFrame


def _registry_dataframe(registry: Iterable[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        list(registry),
        columns=["contrast_name", "identity_a", "identity_b", "axis"],
    )


def load_validated_contrasts(
    identity_csv_path: Path | str,
    *,
    registry: Iterable[tuple[str, str, str, str]] = CONTRASTS,
    restrict_axes: Iterable[str] | None = None,
) -> ContrastValidationResult:
    """Load the canonical registry and split into (valid, skipped) against the
    identity CSV.

    Parameters
    ----------
    identity_csv_path : Path
        Path to `bbq_identity_normalized_forms.csv` (or compatible — must have
        an `identity_id` column).
    registry : iterable of 4-tuples
        Defaults to the module-level CONTRASTS. Override to test a custom set
        without mutating the module global.
    restrict_axes : iterable of str, optional
        If given, only return contrasts on these axes (after validation). The
        skipped DataFrame still includes rows on other axes that fail
        validation, so the audit trail is preserved.

    Returns
    -------
    ContrastValidationResult
        `valid` rows have both identities present in the CSV; `skipped` rows
        do not, with the per-row `reason` column saying which side(s) were
        missing.
    """
    identities = pd.read_csv(identity_csv_path, keep_default_na=False)
    if "identity_id" not in identities.columns:
        raise ValueError(
            f"Identity CSV {identity_csv_path} is missing the 'identity_id' column."
        )
    identity_set = set(identities["identity_id"].astype(str))

    df = _registry_dataframe(registry)
    reasons = []
    keeps = []
    for _, row in df.iterrows():
        a_ok = row["identity_a"] in identity_set
        b_ok = row["identity_b"] in identity_set
        if a_ok and b_ok:
            reasons.append("")
            keeps.append(True)
        else:
            missing = []
            if not a_ok:
                missing.append("missing_identity_a")
            if not b_ok:
                missing.append("missing_identity_b")
            reasons.append(";".join(missing))
            keeps.append(False)
    df["validation_reason"] = reasons

    valid = df[df.apply(lambda r: r["validation_reason"] == "", axis=1)].drop(columns=["validation_reason"]).reset_index(drop=True)
    skipped = df[df.apply(lambda r: r["validation_reason"] != "", axis=1)].rename(columns={"validation_reason": "reason"}).reset_index(drop=True)

    if restrict_axes is not None:
        axes = set(restrict_axes)
        valid = valid[valid["axis"].isin(axes)].reset_index(drop=True)

    return ContrastValidationResult(valid=valid, skipped=skipped)


def write_contrasts_skipped(
    skipped: pd.DataFrame, output_path: Path, *, logger=None
) -> None:
    """Write the skipped contrasts to a CSV and log a warning per row.

    No-op when `skipped` is empty (and writes an empty header-only CSV anyway
    so downstream tooling can rely on the artifact existing).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["contrast_name", "identity_a", "identity_b", "axis", "reason"]
    skipped = skipped.copy()
    for col in columns:
        if col not in skipped.columns:
            skipped[col] = ""
    skipped[columns].to_csv(output_path, index=False)
    if logger is None:
        emit = print
    else:
        emit = logger.warning if hasattr(logger, "warning") else logger
    if len(skipped) == 0:
        emit(f"Contrast validation: 0 skipped against {output_path.name} (all registry pairs validated).")
        return
    for _, row in skipped.iterrows():
        emit(
            f"Contrast validation: skipping {row['contrast_name']!r} "
            f"({row['identity_a']!r} vs {row['identity_b']!r}, axis={row['axis']!r}): "
            f"reason={row['reason']}"
        )
    emit(f"Contrast validation: {len(skipped)} contrasts skipped; full list written to {output_path}.")


def get_contrast_pairs(
    contrasts: pd.DataFrame, *, axis: str | None = None
) -> list[tuple[str, str]]:
    """Return (identity_a, identity_b) 2-tuples for callers that don't need
    the contrast_name or axis. Optional `axis` filters the rows."""
    if axis is not None:
        contrasts = contrasts[contrasts["axis"] == axis]
    return [(str(row.identity_a), str(row.identity_b)) for row in contrasts.itertuples(index=False)]


def filter_to_key_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Filter a contrasts DataFrame down to the KEY_CONTRAST_NAMES subset,
    preserving the registry's ordering. Skips key contrasts that aren't in
    the input (they may have been validated out)."""
    name_to_idx = {n: i for i, n in enumerate(KEY_CONTRAST_NAMES)}
    have = contrasts[contrasts["contrast_name"].isin(KEY_CONTRAST_NAMES)].copy()
    have["_key_order"] = have["contrast_name"].map(name_to_idx)
    return have.sort_values("_key_order").drop(columns=["_key_order"]).reset_index(drop=True)
