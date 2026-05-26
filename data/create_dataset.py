from pathlib import Path
import re

import pandas as pd

DATA_DIR = Path("/Users/brandonlepine/Repositories/Research_Repositories/status_mi/data")

TEMPLATES_PATH = DATA_DIR / "templates" / "mi_identity_templates.csv"
IDENTITIES_PATH = DATA_DIR / "bbq_identity_normalized_forms.csv"
OUTPUT_PATH = DATA_DIR / "mi_identity_prompts.csv"
AUDIT_PATH = DATA_DIR / "mi_identity_prompts_audit.csv"

# Single source of truth: which `works_*` flag governs each `required_form`.
# When works_<form> == 1 the identity is expected to realize that template form;
# the corresponding form cell must be non-empty (a missing cell is now a hard error).
REQUIRED_FORM_TO_WORKS_FLAG = {
    "adj_form": "works_is_adj",
    "noun_form": "works_is_noun",
    "person_noun_form": "works_person_noun",
    "plural_noun_form": "works_plural",
    "group_form": "works_group",
    "prep_form": "works_prep",
    "with_form": "works_with",
    "has_form": "works_has",
}

BAD_PATTERNS = [
    "has is ",
    "has has ",
    "is is ",
    "A a ",
    "A an ",
    "One a ",
    "One an ",
    "The a ",
    "The an ",
    "people are people",
]


def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def sentence_case(s: str) -> str:
    s = str(s).strip()
    return s[:1].upper() + s[1:] if s else s


def works_flag_true(identity_row: pd.Series, works_col: str) -> bool:
    return clean_str(identity_row.get(works_col, "")) == "1"


def main():
    templates = pd.read_csv(TEMPLATES_PATH, keep_default_na=False).fillna("")
    identities = pd.read_csv(IDENTITIES_PATH, keep_default_na=False).fillna("")

    for required_form, works_col in REQUIRED_FORM_TO_WORKS_FLAG.items():
        if required_form not in identities.columns:
            raise ValueError(
                f"Identities CSV is missing form column '{required_form}' "
                f"required by REQUIRED_FORM_TO_WORKS_FLAG."
            )
        if works_col not in identities.columns:
            raise ValueError(
                f"Identities CSV is missing works flag '{works_col}' "
                f"required by REQUIRED_FORM_TO_WORKS_FLAG."
            )

    template_required_forms = templates["required_form"].map(clean_str)
    unknown_required_forms = sorted(
        set(template_required_forms) - set(REQUIRED_FORM_TO_WORKS_FLAG)
    )
    if unknown_required_forms:
        raise ValueError(
            "Templates reference required_form values without a works_* mapping: "
            f"{unknown_required_forms}. Update REQUIRED_FORM_TO_WORKS_FLAG."
        )

    works_realizable_counts = {
        required_form: int((identities[works_col].map(clean_str) == "1").sum())
        for required_form, works_col in REQUIRED_FORM_TO_WORKS_FLAG.items()
    }
    expected_row_count = int(
        template_required_forms.map(works_realizable_counts).sum()
    )

    rows = []

    for _, template in templates.iterrows():
        template_id = clean_str(template["template_id"])
        family = clean_str(template["family"])
        template_text = clean_str(template["template_text"])
        required_form = clean_str(template["required_form"])
        number = clean_str(template.get("number", ""))
        template_notes = clean_str(template.get("notes", ""))

        works_col = REQUIRED_FORM_TO_WORKS_FLAG[required_form]

        for _, identity in identities.iterrows():
            if not works_flag_true(identity, works_col):
                continue

            form = clean_str(identity[required_form])
            if not form:
                raise ValueError(
                    f"Identity '{identity['identity_id']}' has {works_col}=1 "
                    f"but empty '{required_form}'. Fix the identities CSV: either "
                    f"populate the form cell or set {works_col}=0."
                )

            prompt = sentence_case(template_text.format(form=form))

            rows.append({
                "prompt_id": f"{template_id}__{identity['identity_id']}",
                "template_id": template_id,
                "family": family,
                "template_text": template_text,
                "required_form": required_form,
                "form_used": form,
                "number": number,
                "identity_id": clean_str(identity.get("identity_id", "")),
                "axis": clean_str(identity.get("axis", "")),
                "canonical_label": clean_str(identity.get("canonical_label", "")),
                "prompt": prompt,
                "template_notes": template_notes,
                "identity_notes": clean_str(identity.get("notes", "")),
            })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["axis", "identity_id", "family", "template_id"],
        kind="stable"
    ).reset_index(drop=True)

    if len(out) != expected_row_count:
        raise AssertionError(
            f"Row count mismatch: produced {len(out)} prompts but expected "
            f"{expected_row_count} from sum_over_templates(n_identities_with_works_flag). "
            f"A silent drop has been introduced somewhere."
        )

    bad_regex = "|".join(rf"\b{re.escape(pattern.strip())}\b" for pattern in BAD_PATTERNS)
    bad = out[out["prompt"].str.contains(bad_regex, case=False, regex=True)]
    audit_cols = ["prompt_id", "template_id", "identity_id", "axis", "required_form", "form_used", "prompt"]
    bad[audit_cols].to_csv(AUDIT_PATH, index=False)

    if len(bad):
        print(f"\nWARNING: {len(bad)} suspicious prompts found (see {AUDIT_PATH.name}):")
        print(bad[["template_id", "identity_id", "prompt"]].head(50).to_string(index=False))
    else:
        print(f"\nNo suspicious prompts found. Empty audit written to {AUDIT_PATH.name}.")

    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(out):,} prompts (expected {expected_row_count:,})")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Audit:    {AUDIT_PATH}")

    print("\nCounts by family:")
    print(out["family"].value_counts().sort_index())

    print("\nCounts by axis:")
    print(out["axis"].value_counts().sort_index())


if __name__ == "__main__":
    main()
