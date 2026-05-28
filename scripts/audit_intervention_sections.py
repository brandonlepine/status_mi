#!/usr/bin/env python3
"""Audit the audit-3.3 section-aware intervention-position machinery against
the real Llama tokenizer + real prepared BBQ rows.

The 3.3 fix (commit afb3ee3) added section-aware token flags in Step 19 and
section-explicit position names + an intervention_section output column in
Step 20. Both were validated synthetically on one prompt with a whitespace-
splitting fake tokenizer — enough to verify the geometric logic but NOT
enough to trust against the real Llama tokenizer or the real BBQ prompt
format. The step 19 / step 20 docs explicitly list these gaps.

This script runs the four recommended diagnostic checks before any RunPod
steering job that depends on the new section-explicit positions:

  1. find_section_spans returns non-empty {context, question, ans0, ans1,
     ans2} for every sampled row. Counts failures per section.

  2. For every token where the legacy is_target_identity_token would fire,
     at least one of the three is_target_identity_token_in_* flags is True
     (no "orphaned" matches whose section lookup silently failed).

  3. For each of the six section-explicit position names, the
     intervention_section distribution across the sampled rows. The
     corresponding section should dominate; a non-trivial 'final' fraction
     means the section lookup is failing and the fix is upstream.

  4. Per-(position, section) cross-tab so the operator can eyeball drift.

Exits non-zero if any check trips a threshold (default 5% for checks 1, 2;
default 20% non-target-section fraction for check 3). Operator-tunable.

Usage:
    python scripts/audit_intervention_sections.py \\
        --prepared_data /workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet \\
        --tokenizer_path /workspace/status_mi/models/llama-3.1-8b \\
        --n_examples 50
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_bbq_sae_steering import (  # noqa: E402
    find_section_spans,
    find_spans,
    intersect_spans_with_section,
    position_section_for,
    positions_for,
    stereotype_terms,
    _SECTION_AWARE_POSITION_SECTIONS,
)
from extract_bbq_token_level_sae_activations import (  # noqa: E402
    overlap,
    overlap_in_section,
)


DEFAULT_PREPARED = Path(
    "/workspace/status_mi/results/bbq_steering/llama-3.1-8b/prepared/bbq_prepared_examples.parquet"
)
DEFAULT_TOKENIZER = Path("/workspace/status_mi/models/llama-3.1-8b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared_data", type=Path, default=DEFAULT_PREPARED)
    p.add_argument("--tokenizer_path", type=str, default=str(DEFAULT_TOKENIZER),
                   help="Local path or HF hub id of the Llama tokenizer. Must support return_offsets_mapping.")
    p.add_argument("--n_examples", type=int, default=50,
                   help="Number of prepared BBQ rows to sample for the audit.")
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--section_failure_threshold", type=float, default=0.05,
                   help="Exit non-zero if >this fraction of rows have any missing section span.")
    p.add_argument("--flag_consistency_threshold", type=float, default=0.05,
                   help="Exit non-zero if >this fraction of legacy-identity-token tokens have zero "
                        "section-aware flags set (silent section-lookup failure).")
    p.add_argument("--fallback_threshold", type=float, default=0.20,
                   help="Exit non-zero if any section-explicit position routes >this fraction of "
                        "examples to intervention_section='final' (silent fallback).")
    return p.parse_args()


def load_tokenizer(path: str):
    """Load a fast tokenizer with offset support. Fail loudly if unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed in this environment; "
            "cannot run the real-tokenizer audit. Install transformers or run "
            "this script on the RunPod where Llama is available."
        ) from exc
    try:
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception as exc:
        raise SystemExit(
            f"Could not load tokenizer from {path!r}. Pass --tokenizer_path "
            f"pointing at a local Llama-3.1-8B checkout (the standard path is "
            f"{DEFAULT_TOKENIZER!s}). Underlying error: {exc}"
        ) from exc
    # Sanity-check offset support
    probe = tok("hello world", return_offsets_mapping=True)
    if "offset_mapping" not in probe:
        raise SystemExit(
            f"Tokenizer at {path!r} does not return offset_mapping. The audit "
            "requires a fast (Rust-backed) tokenizer."
        )
    return tok


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def check_section_spans(rows: pd.DataFrame) -> dict[str, int]:
    """Check 1: find_section_spans returns non-empty {context, question, ans0,
    ans1, ans2} for every row. Returns per-section failure counts."""
    failures: Counter = Counter()
    needed = ["context", "question", "ans0", "ans1", "ans2"]
    for _, row in rows.iterrows():
        prompt = str(row["prompt"])
        spans = find_section_spans(prompt, row)
        for name in needed:
            if name not in spans:
                failures[name] += 1
    return dict(failures)


def check_flag_consistency(tokenizer, rows: pd.DataFrame, max_length: int) -> dict[str, int]:
    """Check 2: every token with legacy is_target_identity_token=True should
    have >=1 of the three is_target_identity_token_in_* flags True. Returns
    counters: legacy_true_total, no_section_match (orphans), boundary_straddle
    (>=2 section flags True), zero_section_with_section_lookup_ok (the
    diagnostic case where sections were found but the match somehow doesn't
    overlap any of them).
    """
    n_legacy_true = 0
    n_no_section_match = 0
    n_boundary_straddle = 0
    n_orphan_with_sections_present = 0
    for _, row in rows.iterrows():
        prompt = str(row["prompt"])
        target_label = str(row.get("target_identity_label", "") or "")
        target_idx = row.get("target_answer_idx", np.nan)
        terms = [target_label]
        if pd.notna(target_idx):
            terms.append(str(row.get(f"ans{int(target_idx)}", "") or ""))
        target_spans = find_spans(prompt, terms)
        if not target_spans:
            continue
        section_map = find_section_spans(prompt, row)
        ctx = section_map.get("context")
        qst = section_map.get("question")
        ans_spans = [section_map[f"ans{i}"] for i in range(3) if f"ans{i}" in section_map]
        sections_present = ctx is not None and qst is not None and len(ans_spans) >= 1

        encoded = tokenizer(prompt, truncation=True, max_length=max_length, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        for s, e in offsets:
            if e <= s:
                continue
            legacy = overlap(s, e, target_spans)
            if not legacy:
                continue
            n_legacy_true += 1
            in_ctx = overlap_in_section(s, e, target_spans, ctx)
            in_qst = overlap_in_section(s, e, target_spans, qst)
            in_ans = any(overlap_in_section(s, e, target_spans, sp) for sp in ans_spans)
            n_true = int(in_ctx) + int(in_qst) + int(in_ans)
            if n_true == 0:
                n_no_section_match += 1
                if sections_present:
                    n_orphan_with_sections_present += 1
            elif n_true >= 2:
                n_boundary_straddle += 1
    return {
        "legacy_true_total": n_legacy_true,
        "no_section_match": n_no_section_match,
        "boundary_straddle": n_boundary_straddle,
        "orphan_with_sections_present": n_orphan_with_sections_present,
    }


def check_position_section_distribution(
    tokenizer, rows: pd.DataFrame, max_length: int,
) -> dict[str, Counter]:
    """Check 3 & 4: for each section-explicit position name, run positions_for
    on every row and tally intervention_section. The section-of-interest
    (per _SECTION_AWARE_POSITION_SECTIONS) should dominate."""
    out: dict[str, Counter] = {}
    for pos_name in sorted(_SECTION_AWARE_POSITION_SECTIONS):
        tally: Counter = Counter()
        for _, row in rows.iterrows():
            try:
                positions = positions_for(tokenizer, str(row["prompt"]), row, max_length, pos_name)
                section = position_section_for(tokenizer, str(row["prompt"]), row, max_length, positions)
            except Exception as exc:
                tally[f"error_{type(exc).__name__}"] += 1
                continue
            tally[section] += 1
        out[pos_name] = tally
    return out


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def print_section_check(failures: dict[str, int], n_rows: int, threshold: float) -> bool:
    """Returns True if check passed (failures below threshold)."""
    print("=" * 72)
    print(f"CHECK 1: find_section_spans non-empty per row (n={n_rows})")
    print("=" * 72)
    if not failures:
        print("  ✓ All sections located in every row.")
        return True
    max_fail = 0
    for section, count in sorted(failures.items()):
        frac = count / max(n_rows, 1)
        marker = "✗" if frac > threshold else "·"
        print(f"  {marker} {section:>10}: {count}/{n_rows} rows missing ({frac:6.1%})")
        max_fail = max(max_fail, frac)
    if max_fail > threshold:
        print(f"\n  FAIL: section failure rate exceeds threshold ({threshold:.0%}).")
        print(f"  Likely cause: prompt format changed (e.g. --few_shot_pool prefix) and")
        print(f"  find_section_spans's substring lookup no longer matches.")
        return False
    return True


def print_flag_consistency(stats: dict[str, int], threshold: float) -> bool:
    print("\n" + "=" * 72)
    print("CHECK 2: section-aware flag consistency")
    print("=" * 72)
    n_total = stats["legacy_true_total"]
    if n_total == 0:
        print("  · No legacy is_target_identity_token=True tokens in sample.")
        print("    Either the sample is small or the identity labels are not")
        print("    matching with the real Llama offsets — investigate.")
        return True
    print(f"  Tokens with legacy is_target_identity_token = True: {n_total}")
    no_match_frac = stats["no_section_match"] / n_total
    orphan_frac = stats["orphan_with_sections_present"] / n_total
    straddle_frac = stats["boundary_straddle"] / n_total
    marker_nm = "✗" if no_match_frac > threshold else "·"
    print(f"  {marker_nm} 0 section flags True: {stats['no_section_match']}/{n_total} ({no_match_frac:6.1%})")
    print(f"        of which sections WERE located: {stats['orphan_with_sections_present']} ({orphan_frac:6.1%})")
    print(f"  · ≥2 section flags True (boundary straddle): {stats['boundary_straddle']} ({straddle_frac:6.1%})")
    if no_match_frac > threshold:
        print(f"\n  FAIL: >{threshold:.0%} of legacy-identity tokens have no section flag.")
        if orphan_frac > 0:
            print(f"  {orphan_frac:.0%} of those had sections located — the legacy match falls")
            print(f"  outside every section span. Likely a Llama BPE offset issue where the")
            print(f"  token's char range extends past a section boundary.")
        return False
    return True


def print_position_distribution(
    distributions: dict[str, Counter], n_rows: int, threshold: float,
) -> bool:
    print("\n" + "=" * 72)
    print(f"CHECK 3 & 4: intervention_section distribution per section-explicit position (n={n_rows})")
    print("=" * 72)
    all_sections = sorted({s for d in distributions.values() for s in d})
    header = f"  {'position':<46}  " + "  ".join(f"{s:>10}" for s in all_sections)
    print(header)
    print(f"  {'-' * 46}  " + "  ".join("-" * 10 for _ in all_sections))
    overall_ok = True
    for pos_name, expected_section in sorted(_SECTION_AWARE_POSITION_SECTIONS.items()):
        tally = distributions.get(pos_name, Counter())
        cells = []
        for s in all_sections:
            n = tally.get(s, 0)
            frac = n / max(n_rows, 1)
            cells.append(f"{n:4d} ({frac:.0%})")
        print(f"  {pos_name:<46}  " + "  ".join(f"{c:>10}" for c in cells))
        final_frac = tally.get("final", 0) / max(n_rows, 1)
        expected_frac = tally.get(expected_section, 0) / max(n_rows, 1)
        if final_frac > threshold:
            overall_ok = False
            print(f"      ✗ FAIL: {final_frac:.0%} silently fell back to 'final' "
                  f"(threshold {threshold:.0%}). Expected '{expected_section}' to dominate.")
        elif expected_frac < 0.5:
            print(f"      · WARNING: only {expected_frac:.0%} landed in expected section "
                  f"'{expected_section}'. Investigate.")
    return overall_ok


def main() -> int:
    args = parse_args()
    if not args.prepared_data.exists():
        raise SystemExit(
            f"--prepared_data not found at {args.prepared_data!s}. Pass the path "
            f"to the bbq_prepared_examples.parquet from prepare_bbq_for_steering.py."
        )
    print(f"Loading prepared BBQ rows from {args.prepared_data}")
    df = pd.read_parquet(args.prepared_data) if args.prepared_data.suffix == ".parquet" else pd.read_csv(args.prepared_data)
    print(f"  Total rows: {len(df):,}")
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = load_tokenizer(args.tokenizer_path)
    print(f"  Tokenizer: {type(tokenizer).__name__}")

    if len(df) > args.n_examples:
        sample = df.sample(args.n_examples, random_state=args.random_seed).reset_index(drop=True)
    else:
        sample = df.reset_index(drop=True)
    print(f"  Audit sample: {len(sample)} rows (seed {args.random_seed})\n")

    section_failures = check_section_spans(sample)
    ok1 = print_section_check(section_failures, len(sample), args.section_failure_threshold)

    consistency = check_flag_consistency(tokenizer, sample, args.max_length)
    ok2 = print_flag_consistency(consistency, args.flag_consistency_threshold)

    distributions = check_position_section_distribution(tokenizer, sample, args.max_length)
    ok3 = print_position_distribution(distributions, len(sample), args.fallback_threshold)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Check 1 (section lookup):     {'PASS' if ok1 else 'FAIL'}")
    print(f"  Check 2 (flag consistency):   {'PASS' if ok2 else 'FAIL'}")
    print(f"  Check 3 (position routing):   {'PASS' if ok3 else 'FAIL'}")
    if ok1 and ok2 and ok3:
        print("\n  ✓ Audit 3.3 fix behaves correctly against real Llama tokenizer + real BBQ format.")
        return 0
    print("\n  ✗ One or more checks failed. Do NOT use section-explicit positions for headline")
    print("    causal claims until the failures are diagnosed and fixed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
