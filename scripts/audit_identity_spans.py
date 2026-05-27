#!/usr/bin/env python3
"""Validate scripts/extract_identity_activations.py:find_identity_span on a
prompt CSV. Runs four integrity checks and exits non-zero if any fails so
the audit can be wired into CI or pre-commit.

Checks (per (prompt, form_used) pair):
  1. A span was found (status != 'failed').
  2. The matched text equals form_used case-insensitively.
  3. The form does not appear more than once in the prompt (no ambiguity
     about which occurrence find_identity_span took).
  4. The match is at a clean word boundary — the chars on either side of
     the matched span are not alphabetic. Catches substring-overlap bugs
     like form 'man' matching inside 'woman'.

Run:
  python scripts/audit_identity_spans.py
  python scripts/audit_identity_spans.py --input_csv data/mi_identity_prompts.csv \
                                          --output /tmp/find_identity_span_audit.csv

The script exits 0 on full pass, 1 on any failure.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_identity_activations import find_identity_span  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "mi_identity_prompts.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "find_identity_span_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT,
                        help="Prompt CSV with at least 'prompt_id', 'prompt', 'form_used' columns.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write the per-prompt audit CSV.")
    parser.add_argument("--show_examples", type=int, default=10,
                        help="How many example failures to print per category.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv, keep_default_na=False)
    for col in ("prompt_id", "prompt", "form_used"):
        if col not in df.columns:
            print(f"ERROR: input_csv is missing required column '{col}'", file=sys.stderr)
            return 2

    print(f"Auditing find_identity_span on {len(df):,} rows from {args.input_csv}...\n")

    rows = []
    for _, r in df.iterrows():
        prompt = str(r["prompt"])
        form = str(r["form_used"])
        span_start, span_end, status = find_identity_span(prompt, form)

        matched_text = prompt[span_start:span_end] if span_start is not None else ""
        matched_equals_form = matched_text.lower() == form.lower()

        n_occurrences = len(re.findall(re.escape(form), prompt, flags=re.I))

        if span_start is not None:
            left_ch = prompt[span_start - 1] if span_start > 0 else " "
            right_ch = prompt[span_end] if span_end < len(prompt) else " "
            clean_boundary = (not left_ch.isalpha()) and (not right_ch.isalpha())
        else:
            clean_boundary = None

        rows.append({
            "prompt_id": r["prompt_id"],
            "template_id": r.get("template_id", ""),
            "required_form": r.get("required_form", ""),
            "form_used": form,
            "prompt": prompt,
            "span_status": status,
            "span_start": span_start,
            "span_end": span_end,
            "matched_text": matched_text,
            "matched_equals_form": matched_equals_form,
            "n_occurrences": n_occurrences,
            "clean_boundary": clean_boundary,
        })

    res = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output, index=False)

    n_total = len(res)
    n_failed = (res["span_status"] == "failed").sum()
    n_mismatch = ((~res["matched_equals_form"]) & (res["span_status"] != "failed")).sum()
    n_ambiguous = (res["n_occurrences"] > 1).sum()
    n_bad_bound = ((res["clean_boundary"] == False) & (res["span_status"] != "failed")).sum()

    print(f"Total prompts:           {n_total:,}")
    print(f"Status counts:           {Counter(res['span_status'].tolist())}")
    print(f"")
    print(f"Check 1 (span found):    {n_total - n_failed:,} / {n_total:,}  {'PASS' if n_failed == 0 else f'FAIL ({n_failed} failed)'}")
    print(f"Check 2 (text==form):    {(n_total - n_failed - n_mismatch):,} / {(n_total - n_failed):,}  {'PASS' if n_mismatch == 0 else f'FAIL ({n_mismatch} mismatched)'}")
    print(f"Check 3 (no ambiguity):  {(n_total - n_ambiguous):,} / {n_total:,}  {'PASS' if n_ambiguous == 0 else f'FAIL ({n_ambiguous} ambiguous)'}")
    print(f"Check 4 (clean boundary):{(n_total - n_failed - n_bad_bound):,} / {(n_total - n_failed):,}  {'PASS' if n_bad_bound == 0 else f'FAIL ({n_bad_bound} substring overlaps)'}")

    failures = []
    if n_failed:
        failures.append(("FAILED span", res[res["span_status"] == "failed"]))
    if n_mismatch:
        failures.append(("MISMATCH text vs form", res[(~res["matched_equals_form"]) & (res["span_status"] != "failed")]))
    if n_ambiguous:
        failures.append(("AMBIGUOUS (multiple occurrences)", res[res["n_occurrences"] > 1]))
    if n_bad_bound:
        failures.append(("SUBSTRING overlap (bad word boundary)", res[(res["clean_boundary"] == False) & (res["span_status"] != "failed")]))

    for label, subset in failures:
        print(f"\n--- {label} ({len(subset):,} rows; showing up to {args.show_examples}) ---")
        cols = ["prompt_id", "form_used", "matched_text", "prompt"]
        print(subset.head(args.show_examples)[cols].to_string(index=False))

    print(f"\nAudit CSV: {args.output}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
