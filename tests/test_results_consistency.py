"""The published tables must compare like with like: every backend of a gate is scored on the same examples."""
import csv
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_every_backend_of_a_gate_uses_the_same_examples():
    sizes = defaultdict(set)
    for r in csv.DictReader(open(RESULTS / "gates.csv", encoding="utf-8")):
        sizes[(r["gate"], r["split"])].add(r["n"])
    assert sizes and all(len(v) == 1 for v in sizes.values()), {k: v for k, v in sizes.items() if len(v) > 1}


def test_summary_has_a_header_and_no_duplicate_rows():
    rows = list(csv.reader(open(RESULTS / "summary.csv", encoding="utf-8")))
    assert rows[0] == ["variant", "metric", "mean", "ci_lo", "ci_hi"]
    keys = [(r[0], r[1]) for r in rows[1:]]
    assert len(keys) == len(set(keys))
