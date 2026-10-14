#!/usr/bin/env python3
"""Summarize Dyck 30-seed causal results, separating label flips from single-class collapse.

Reads either per-seed files written by train_dyck_30seed.py (results_dyck_30seed/seed_*.json,
which carry prediction_stats) or the combined results/dyck_30seed/dyck_30seed_causal.json
(old format, accuracy only, so ~0.5 results are "chance_ambiguous").
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from dyck_outcomes import classify


def load(path: Path):
    if path.is_dir():
        return [json.loads(p.read_text()) for p in sorted(path.glob("seed_*.json"))]
    d = json.loads(path.read_text())
    seeds = d["seeds"] if isinstance(d, dict) else d
    return list(seeds.values()) if isinstance(seeds, dict) else seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results_dyck_30seed")
    args = ap.parse_args()
    rows = load(Path(args.path))
    if not rows:
        sys.exit(f"no Dyck seed results found at {args.path}")
    counts = {k: Counter() for k in ("contrastive", "algorithmic", "random")}
    print(f"{'seed':>5} {'base':>6} | {'contrastive':>24} | {'algorithmic':>24} | {'random':>24}")
    for r in rows:
        stats = r.get("prediction_stats", {})
        cells = []
        for k in counts:
            acc = r[f"acc_cancel_{k}"]
            ppr = stats.get(k, {}).get("pred_pos_rate")
            o = classify(acc, ppr, r["acc_base"])
            counts[k][o] += 1
            cells.append(f"{acc:.3f} {o:>18}")
        print(f"{r['seed']:>5} {r['acc_base']:>6.3f} | " + " | ".join(cells))
    for k, c in counts.items():
        print(f"{k:<12} " + ", ".join(f"{o}={n}" for o, n in sorted(c.items())))


if __name__ == "__main__":
    main()
