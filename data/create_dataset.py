from pathlib import Path
import re

import pandas as pd

DATA_DIR = Path("/Users/brandonlepine/Repositories/Research_Repositories/status_mi/data")

TEMPLATES_PATH = DATA_DIR / "templates" / "mi_identity_templates.csv"
IDENTITIES_PATH = DATA_DIR / "bbq_identity_normalized_forms.csv"
OUTPUT_PATH = DATA_DIR / "mi_identity_prompts.csv"


def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def sentence_case(s: str) -> str:
    s = str(s).strip()
    return s[:1].upper() + s[1:] if s else s


def main():
    templates = pd.read_csv(TEMPLATES_PATH, keep_default_na=False).fillna("")
    identities = pd.read_csv(IDENTITIES_PATH, keep_default_na=False).fillna("")

    rows = []

    for _, template in templates.iterrows():
        template_id = clean_str(template["template_id"])
        family = clean_str(template["family"])
        template_text = clean_str(template["template_text"])
        required_form = clean_str(template["required_form"])
        number = clean_str(template.get("number", ""))
        template_notes = clean_str(template.get("notes", ""))

        if required_form not in identities.columns:
            raise ValueError(
                f"Template {template_id} requires form column '{required_form}', "
                f"but that column is not in identities CSV."
            )

        for _, identity in identities.iterrows():
            form = clean_str(identity[required_form])

            if not form:
                continue

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

    bad_patterns = [
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

    bad_regex = "|".join(rf"\b{re.escape(pattern.strip())}\b" for pattern in bad_patterns)
    bad = out[out["prompt"].str.contains(bad_regex, case=False, regex=True)]

    if len(bad):
        print("\nWARNING: suspicious prompts found:")
        print(bad[["template_id", "identity_id", "prompt"]].head(50).to_string(index=False))

    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(out):,} prompts")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nCounts by family:")
    print(out["family"].value_counts().sort_index())

    print("\nCounts by axis:")
    print(out["axis"].value_counts().sort_index())


if __name__ == "__main__":
    main()