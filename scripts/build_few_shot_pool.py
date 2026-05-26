#!/usr/bin/env python3
"""Build a stratified, seeded few-shot exemplar pool from BBQ.

The pool is K=4 exemplars covering every (context_condition, question_polarity) cell:
  (ambig, neg) / (ambig, nonneg) / (disambig, neg) / (disambig, nonneg)

Each exemplar is drawn from a distinct BBQ category to avoid axis over-representation.
Output JSON contains both the raw fields needed to reconstruct prompts and a pre-rendered
`exemplar_block` string in the exact format used by `prepare_bbq_for_steering.py`.

Run:
  python scripts/build_few_shot_pool.py \
      --bbq_data_dir data/bbq/data \
      --output data/bbq/few_shot_pool.json \
      --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BBQ_DIR = REPO_ROOT / "data" / "bbq" / "data"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "bbq" / "few_shot_pool.json"

STRATA = [
    ("ambig", "neg"),
    ("ambig", "nonneg"),
    ("disambig", "neg"),
    ("disambig", "nonneg"),
]

INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C"}


def render_choices(row: dict) -> str:
    return (
        f"A. {row['ans0']}\n"
        f"B. {row['ans1']}\n"
        f"C. {row['ans2']}"
    )


def render_exemplar_block(row: dict) -> str:
    letter = INDEX_TO_LETTER[int(row["label"])]
    answer_text = row[f"ans{int(row['label'])}"]
    return (
        f"Context:\n{row['context']}\n\n"
        f"Question:\n{row['question']}\n\n"
        "Answer choices:\n"
        f"{render_choices(row)}\n\n"
        f"Answer: {letter}. {answer_text}"
    )


def read_bbq_jsonl(bbq_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(bbq_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj["_source_file"] = path.name
                rows.append(obj)
    return rows


def select_pool(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[dict]] = {s: [] for s in STRATA}
    for r in rows:
        key = (r.get("context_condition"), r.get("question_polarity"))
        if key in by_stratum:
            by_stratum[key].append(r)

    for stratum, bucket in by_stratum.items():
        if not bucket:
            raise RuntimeError(f"No BBQ rows found for stratum {stratum}.")
        rng.shuffle(bucket)

    used_categories: set[str] = set()
    chosen: list[dict] = []

    for stratum in STRATA:
        pick = next(
            (r for r in by_stratum[stratum] if r["category"] not in used_categories),
            by_stratum[stratum][0],  # fall back to first if all categories are used
        )
        used_categories.add(pick["category"])
        chosen.append(pick)

    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbq_data_dir", type=Path, default=DEFAULT_BBQ_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_bbq_jsonl(args.bbq_data_dir)
    if not rows:
        raise RuntimeError(f"No BBQ rows loaded from {args.bbq_data_dir}.")

    chosen = select_pool(rows, args.seed)

    pool = {
        "seed": args.seed,
        "stratification": "one_per_(context_condition, question_polarity); distinct category per row when possible",
        "answer_format": "Answer: <LETTER>. <answer_text>",
        "k": len(chosen),
        "exemplars": [],
    }

    for row in chosen:
        pool["exemplars"].append({
            "example_id": int(row["example_id"]),
            "question_index": row.get("question_index"),
            "category": row["category"],
            "context_condition": row["context_condition"],
            "question_polarity": row["question_polarity"],
            "source_file": row["_source_file"],
            "context": row["context"],
            "question": row["question"],
            "ans0": row["ans0"],
            "ans1": row["ans1"],
            "ans2": row["ans2"],
            "label": int(row["label"]),
            "exemplar_block": render_exemplar_block(row),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(pool, f, indent=2)
        f.write("\n")

    print(f"Wrote {args.output} with K={pool['k']} exemplars.")
    for ex in pool["exemplars"]:
        print(
            f"  category={ex['category']:<22} "
            f"({ex['context_condition']}, {ex['question_polarity']}) "
            f"example_id={ex['example_id']}  "
            f"label={INDEX_TO_LETTER[ex['label']]}"
        )


if __name__ == "__main__":
    main()
