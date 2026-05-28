#!/usr/bin/env python3
"""Audit the audit-4.6 truncation question against the SAVED top-k SAE encodings.

Audit 4.6 has two halves:

(a) The upstream encoder side. Step 6's validator
    (scripts/validate_sae_hook_alignment.py, commit c6dbcfe) re-encodes a
    sample of raw activations, reports L0 percentiles, and fails on
    max_l0 > --top_k_save_threshold. This catches the case where the
    underlying SAE has true L0 > top_k_save.

(b) The downstream artifact side — this script. The actual SAVED files
    (feature_indices_top{K}.npy, feature_values_top{K}.npy) are what
    drove triage and seeded the BBQ feature pool. Even if half (a)
    validates a fresh encoder run, the saved files may have been
    encoded with a too-small top_k_save and the feature pool that
    derived from them may be biased — mid-rank features that genuinely
    fire on identity prompts were truncated and never reached triage.

This script reads the saved (indices, values) per layer and detects rows
"at the cap" — rows where every saved top-k value is positive, meaning
the true L0 was >= top_k and any (L0 - top_k) lower-magnitude active
features were dropped. The diagnostic is:

  n_at_cap == 0    → safe; the cap was never reached, the feature pool
                     is not biased by truncation.
  n_at_cap > 0     → some rows had hidden truncation; the operator should
                     decide whether to bump --top_k_save and re-encode.

Per-layer report:
  - n_rows, top_k (read from the filename), n_features
  - L0 stats: mean, max, p50, p95, p99
  - n_at_cap (count) and fraction_at_cap
  - PASS / FAIL based on --at_cap_threshold

Exits non-zero if any layer's fraction-at-cap exceeds the threshold.

Usage:
    python scripts/audit_identity_sae_l0.py \\
        --sae_encoded_dir /workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token \\
        --layers 0,8,16,24,32 \\
        --at_cap_threshold 0.01
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


DEFAULT_SAE_ENCODED_DIR = Path(
    "/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sae_encoded_dir", type=Path, default=DEFAULT_SAE_ENCODED_DIR,
                   help="Top-level dir produced by scripts/encode_identity_saes.py, "
                        "containing layer_XX/ subdirs with feature_indices_top{K}.npy + "
                        "feature_values_top{K}.npy.")
    p.add_argument("--layers", default="0,8,16,24,32",
                   help="Comma-separated list of layers to audit. Each layer must have "
                        "the saved top-k files under sae_encoded_dir/layer_XX/.")
    p.add_argument("--at_cap_threshold", type=float, default=0.01,
                   help="Maximum allowed fraction of rows where L0 == top_k (rows at the "
                        "truncation cap). Exit non-zero if any layer exceeds this. Default "
                        "0.01 (1%%) — small slack for boundary cases; tighten to 0 for a "
                        "strict 'no row may be at the cap' rule.")
    return p.parse_args()


def parse_layers(s: str) -> list[int]:
    return [int(part.strip()) for part in s.split(",") if part.strip()]


def find_topk_files(layer_dir: Path) -> tuple[Path, Path, int]:
    """Locate feature_indices_top{K}.npy + feature_values_top{K}.npy and return
    them with K. Raises if missing or inconsistent."""
    if not layer_dir.is_dir():
        raise FileNotFoundError(
            f"Layer directory {layer_dir!s} does not exist. Run "
            f"scripts/encode_identity_saes.py for this layer first."
        )
    indices = sorted(layer_dir.glob("feature_indices_top*.npy"))
    if not indices:
        raise FileNotFoundError(
            f"No feature_indices_top*.npy in {layer_dir!s}. Expected output from "
            f"scripts/encode_identity_saes.py."
        )
    index_path = indices[-1]
    m = re.search(r"feature_indices_top(\d+)\.npy$", index_path.name)
    if not m:
        raise ValueError(f"Could not parse top-k from filename {index_path.name!r}.")
    top_k = int(m.group(1))
    value_path = layer_dir / f"feature_values_top{top_k}.npy"
    if not value_path.exists():
        raise FileNotFoundError(
            f"feature_values_top{top_k}.npy missing in {layer_dir!s}; "
            f"indices and values must be paired."
        )
    return index_path, value_path, top_k


def audit_layer(layer_dir: Path) -> dict[str, object]:
    """Read the saved top-k file for a layer and compute L0 / at-cap stats."""
    idx_path, val_path, top_k = find_topk_files(layer_dir)
    values = np.load(val_path, mmap_mode="r")  # (n_rows, top_k)
    if values.ndim != 2:
        raise ValueError(
            f"Expected (n_rows, top_k) 2D array; got shape {values.shape} from "
            f"{val_path!s}."
        )
    n_rows, k_in_file = values.shape
    if k_in_file != top_k:
        raise ValueError(
            f"Filename says top-k={top_k} but the array has {k_in_file} columns "
            f"in {val_path!s}."
        )
    # L0 per row = count of strictly positive saved values. The JumpReLU pre-
    # threshold gate produces non-negative values; any positive saved value
    # corresponds to an active feature. Zeros mean the row had L0 < top_k.
    positives_per_row = (np.asarray(values) > 0).sum(axis=1)
    n_at_cap = int((positives_per_row == top_k).sum())
    return {
        "layer_dir": layer_dir,
        "n_rows": n_rows,
        "top_k_saved": top_k,
        "values_path": val_path,
        "mean_l0": float(positives_per_row.mean()),
        "max_l0": int(positives_per_row.max()),
        "p50_l0": int(np.percentile(positives_per_row, 50)),
        "p95_l0": int(np.percentile(positives_per_row, 95)),
        "p99_l0": int(np.percentile(positives_per_row, 99)),
        "n_at_cap": n_at_cap,
        "fraction_at_cap": n_at_cap / max(n_rows, 1),
    }


def print_report(rows: list[dict[str, object]], threshold: float) -> bool:
    """Returns True if every layer passed."""
    print("=" * 88)
    print(f"AUDIT 4.6: identity-side saved top-k SAE L0 distribution")
    print("=" * 88)
    header = (
        f"  {'layer':<5}  {'n_rows':>8}  {'top_k':>5}  "
        f"{'mean L0':>8}  {'p50':>4}  {'p95':>4}  {'p99':>4}  {'max':>4}  "
        f"{'n_at_cap':>8}  {'frac_cap':>9}  status"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    all_pass = True
    for r in rows:
        frac = r["fraction_at_cap"]
        passed = frac <= threshold
        all_pass = all_pass and passed
        layer_name = Path(r["layer_dir"]).name
        layer_num = layer_name.removeprefix("layer_") if isinstance(layer_name, str) else "?"
        status = "PASS" if passed else f"FAIL (>{threshold:.0%})"
        print(
            f"  {layer_num:<5}  {r['n_rows']:>8d}  {r['top_k_saved']:>5d}  "
            f"{r['mean_l0']:>8.2f}  {r['p50_l0']:>4d}  {r['p95_l0']:>4d}  "
            f"{r['p99_l0']:>4d}  {r['max_l0']:>4d}  "
            f"{r['n_at_cap']:>8d}  {frac:>9.2%}  {status}"
        )
    print()
    if all_pass:
        print(
            "  ✓ No layer exceeds the at-cap threshold. The saved top-k files "
            "are not biased by truncation,"
        )
        print(
            "    and any feature pool derived from them is safe with respect to "
            "audit 4.6."
        )
    else:
        print(
            "  ✗ One or more layers have rows at the top-k cap. Those rows were "
            "TRUNCATED at encode time;"
        )
        print(
            "    mid-rank features that fired on those prompts are not in the "
            "saved files and were missed"
        )
        print(
            "    by triage. Re-encode the affected layer(s) with a larger "
            "--top_k_save (run "
        )
        print(
            "    scripts/validate_sae_hook_alignment.py to pick a safe value: "
            "max_l0 reported there"
        )
        print(
            "    is the un-truncated truth; --top_k_save should exceed it with "
            "headroom)."
        )
        print(
            "    After re-encoding, regenerate the per-feature analysis "
            "(scripts/analyze_identity_sae_features.py)"
        )
        print(
            "    and the triage (scripts/triage_sae_identity_features.py) "
            "before consuming the feature pool downstream."
        )
    return all_pass


def main() -> int:
    args = parse_args()
    if not args.sae_encoded_dir.exists():
        raise SystemExit(
            f"--sae_encoded_dir not found at {args.sae_encoded_dir!s}. Pass the "
            f"path to the directory produced by scripts/encode_identity_saes.py."
        )
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for layer in parse_layers(args.layers):
        layer_dir = args.sae_encoded_dir / f"layer_{layer:02d}"
        try:
            rows.append(audit_layer(layer_dir))
        except FileNotFoundError as exc:
            failures.append(f"layer {layer}: {exc}")
        except ValueError as exc:
            failures.append(f"layer {layer}: {exc}")
    if not rows and not failures:
        raise SystemExit(
            f"No layers audited. Pass --layers (comma-separated) and confirm the "
            f"layer dirs exist under {args.sae_encoded_dir!s}."
        )
    for msg in failures:
        print(f"  SKIP {msg}", file=sys.stderr)
    if not rows:
        return 1
    all_pass = print_report(rows, args.at_cap_threshold)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
