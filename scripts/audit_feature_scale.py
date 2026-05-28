#!/usr/bin/env python3
"""Audit the audit-3.2 per-feature amplitude scaling against the SAVED
feature_stats.csv + the triage feature pool, BEFORE a steering job depends on it.

Audit 3.2 makes clamp/steer amplitude comparable across features by scaling the
perturbation by each feature's own activation statistic (p95/p99/max/mean_nonzero)
from encode_identity_saes.py's per-layer feature_stats.csv:

    clamp target  = clamp_multiplier * scale[f]
    steer increment = alpha * sign * scale[f]

The silent-failure modes against real saved stats — which synthetic tests on
fabricated tables cannot catch — are:

  (a) A feature selected for intervention (kept_for_intervention in the triage
      CSV) is ABSENT from feature_stats.csv. run_bbq_sae_steering.py assigns it
      scale=0, so its clamp target / steer increment collapses to ~0: clamp
      degenerates to ablate, steer to a no-op. The headline effect for that
      feature would be measured at zero amplitude while looking like a normal
      run.
  (b) A selected feature has a NON-POSITIVE stat (p95<=0 — a dead or extremely
      rare feature, or a NaN that compute_feature_stats filled with 0.0). Same
      collapse as (a).
  (c) A layer in the feature pool has no feature_stats.csv at all (wrong
      --feature_stats_dir, or that layer was never encoded), which would make
      run_bbq_sae_steering.py raise at load time — caught here up front.

This script does NOT need a GPU or the model. It reads the triage CSV (the
feature pool) and the per-layer feature_stats.csv and reports, per layer:

  - n_kept_features          features kept_for_intervention on this layer
  - n_missing                kept features absent from feature_stats.csv  (mode a)
  - n_nonpositive            kept features with stat <= 0                  (mode b)
  - n_degenerate             n_missing + n_nonpositive (would run at ~0 amplitude)
  - frac_degenerate          n_degenerate / n_kept_features
  - scale percentiles        p5/p50/p95 of the kept features' scale values
                             (a sanity check that the clamp targets are sane)
  - PASS / FAIL              based on --degenerate_threshold

Exits non-zero if any layer's degenerate fraction exceeds the threshold, or if a
required feature_stats.csv is missing. Run this before any clamp/steer headline.

Usage:
    python scripts/audit_feature_scale.py \\
        --triage_csv /workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/triage/intervention_candidate_features_triaged.csv \\
        --feature_stats_dir /workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token \\
        --feature_scale_stat p95 \\
        --layers 24 \\
        --degenerate_threshold 0.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Mirror run_bbq_sae_steering.py's stat -> column map so this audit checks
# exactly the column the steering run would use.
FEATURE_SCALE_STAT_COLUMN = {
    "p95": "p95_activation",
    "p99": "p99_activation",
    "max": "max_activation",
    "mean_nonzero": "mean_activation_nonzero",
}

DEFAULT_TRIAGE = Path(
    "/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token/"
    "triage/intervention_candidate_features_triaged.csv"
)
DEFAULT_FEATURE_STATS_DIR = Path(
    "/workspace/status_mi/results/sae_identity/llama-3.1-8b/final_token"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--triage_csv", type=Path, default=DEFAULT_TRIAGE,
                   help="The intervention_candidate_features_triaged.csv that seeds the "
                        "BBQ feature pool. Needs columns 'layer' and 'feature_id'; "
                        "'keep_for_intervention' is honored if present.")
    p.add_argument("--feature_stats_dir", type=Path, default=DEFAULT_FEATURE_STATS_DIR,
                   help="encode_identity_saes.py output dir holding "
                        "layer_<NN>/feature_stats.csv (same value passed to "
                        "run_bbq_sae_steering.py --feature_stats_dir).")
    p.add_argument("--feature_scale_stat", default="p95",
                   choices=sorted(FEATURE_SCALE_STAT_COLUMN.keys()),
                   help="Which stat column the steering run will scale by. Default p95.")
    p.add_argument("--layers", default="24",
                   help="Comma-separated layers to audit (the layers your steering run "
                        "will touch).")
    p.add_argument("--degenerate_threshold", type=float, default=0.0,
                   help="Maximum allowed fraction of kept features whose scale is missing "
                        "or non-positive (would run at ~0 amplitude). Exit non-zero if any "
                        "layer exceeds it. Default 0.0 — no degenerate feature tolerated; "
                        "raise it if you knowingly accept a few dead features in a bundle.")
    return p.parse_args()


def parse_layers(s: str) -> list[int]:
    return [int(part.strip()) for part in s.split(",") if part.strip()]


def bool_series(s: pd.Series) -> pd.Series:
    """Coerce a kept-flag column to bool (mirrors run_bbq_sae_steering bool_series
    semantics for the common string/int encodings)."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def kept_features_for_layer(triage: pd.DataFrame, layer: int) -> list[int]:
    df = triage
    if "keep_for_intervention" in df.columns:
        df = df[bool_series(df["keep_for_intervention"])]
    df = df[df["layer"].astype(int) == layer]
    return sorted(df["feature_id"].astype(int).unique().tolist())


def audit_layer(
    triage: pd.DataFrame, feature_stats_dir: Path, layer: int, stat: str
) -> dict[str, object]:
    column = FEATURE_SCALE_STAT_COLUMN[stat]
    kept = kept_features_for_layer(triage, layer)
    stats_path = feature_stats_dir / f"layer_{layer:02d}" / "feature_stats.csv"
    if not stats_path.exists():
        return {
            "layer": layer, "n_kept_features": len(kept),
            "missing_stats_file": True, "stats_path": stats_path,
            "n_missing": len(kept), "n_nonpositive": 0,
            "n_degenerate": len(kept),
            "frac_degenerate": 1.0 if kept else 0.0,
            "scale_p5": float("nan"), "scale_p50": float("nan"), "scale_p95": float("nan"),
        }
    stats = pd.read_csv(stats_path)
    if "feature_id" not in stats.columns or column not in stats.columns:
        raise ValueError(
            f"{stats_path} is missing required columns; expected 'feature_id' and "
            f"{column!r}, found {list(stats.columns)}."
        )
    lookup = dict(zip(stats["feature_id"].astype(int), stats[column].astype(float)))
    scales, missing, nonpositive = [], [], []
    for fid in kept:
        v = lookup.get(int(fid))
        if v is None:
            missing.append(fid)
            continue
        if not (v > 0) or np.isnan(v):
            nonpositive.append(fid)
            continue
        scales.append(float(v))
    n_degenerate = len(missing) + len(nonpositive)
    scale_arr = np.asarray(scales, dtype=float)
    return {
        "layer": layer, "n_kept_features": len(kept),
        "missing_stats_file": False, "stats_path": stats_path,
        "n_missing": len(missing), "n_nonpositive": len(nonpositive),
        "n_degenerate": n_degenerate,
        "frac_degenerate": n_degenerate / max(len(kept), 1),
        "scale_p5": float(np.percentile(scale_arr, 5)) if scale_arr.size else float("nan"),
        "scale_p50": float(np.percentile(scale_arr, 50)) if scale_arr.size else float("nan"),
        "scale_p95": float(np.percentile(scale_arr, 95)) if scale_arr.size else float("nan"),
        "missing_ids": missing[:15],
        "nonpositive_ids": nonpositive[:15],
    }


def print_report(rows: list[dict[str, object]], stat: str, threshold: float) -> bool:
    print("=" * 96)
    print(f"AUDIT 3.2: per-feature amplitude scale soundness (stat={stat})")
    print("=" * 96)
    header = (
        f"  {'layer':<5}  {'n_kept':>6}  {'n_miss':>6}  {'n_nonpos':>8}  "
        f"{'n_degen':>7}  {'frac_deg':>9}  {'scale p5':>9}  {'p50':>9}  {'p95':>9}  status"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    all_pass = True
    for r in rows:
        passed = (not r["missing_stats_file"]) and r["frac_degenerate"] <= threshold
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        if r["missing_stats_file"]:
            status = "FAIL (no feature_stats.csv)"
        print(
            f"  {r['layer']:<5}  {r['n_kept_features']:>6}  {r['n_missing']:>6}  "
            f"{r['n_nonpositive']:>8}  {r['n_degenerate']:>7}  {r['frac_degenerate']:>9.4f}  "
            f"{r['scale_p5']:>9.3f}  {r['scale_p50']:>9.3f}  {r['scale_p95']:>9.3f}  {status}"
        )
        if r.get("missing_ids"):
            print(f"        missing feature_ids (first 15): {r['missing_ids']}")
        if r.get("nonpositive_ids"):
            print(f"        non-positive {stat} feature_ids (first 15): {r['nonpositive_ids']}")
    print("  " + "-" * (len(header) - 2))
    print(f"  threshold: frac_degenerate <= {threshold}")
    print(f"  RESULT: {'ALL PASS' if all_pass else 'FAILED'}")
    print("=" * 96)
    if not all_pass:
        print(
            "\nA degenerate feature scales to ~0 amplitude: clamp degenerates to ablate, "
            "steer to a no-op. Either re-encode the affected layer with a wider activation "
            "sample so feature_stats.csv covers these features, drop the dead features from "
            "the triage pool, or accept them by raising --degenerate_threshold. Until this "
            "passes, no clamp/steer headline should be cited as a per-feature-scaled result."
        )
    return all_pass


def main() -> int:
    args = parse_args()
    if not args.triage_csv.exists():
        print(f"ERROR: triage CSV not found: {args.triage_csv}", file=sys.stderr)
        return 2
    triage = pd.read_csv(args.triage_csv, low_memory=False)
    for col in ("layer", "feature_id"):
        if col not in triage.columns:
            print(f"ERROR: triage CSV missing required column {col!r}.", file=sys.stderr)
            return 2
    rows = [
        audit_layer(triage, args.feature_stats_dir, layer, args.feature_scale_stat)
        for layer in parse_layers(args.layers)
    ]
    return 0 if print_report(rows, args.feature_scale_stat, args.degenerate_threshold) else 1


if __name__ == "__main__":
    raise SystemExit(main())
