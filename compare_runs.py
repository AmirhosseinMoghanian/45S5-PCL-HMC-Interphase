"""Combine the structural results of several runs.

    python compare_runs.py runs/composite_r1 runs/composite_r2 runs/composite_r3
    python compare_runs.py --replicas runs/composite_r1 runs/composite_r2 runs/composite_r3 \
                           --compare runs/neat_r1 runs/sensitivity_eps0.8 runs/sensitivity_eps1.2

Replicas are averaged: the reported uncertainty is the standard error over the replicas
(their spread), which - unlike the block errors of a single run - includes the
structural variability of independently built glasses.  Other runs are listed next to
that average for comparison.  Output: runs/comparison.md and runs/comparison.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and "mean" in v:
            out[key] = float(v["mean"])
        elif isinstance(v, dict):
            out.update(flatten(v, key + " / "))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v):
                out[f"{key} [{i}]"] = float(x)
    return out


def load(run):
    path = Path(run) / "05_structure" / "structure_summary.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run 'python run_pipeline.py structure' for {run}")
    return flatten(json.loads(path.read_text()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="replica runs (same as --replicas)")
    ap.add_argument("--replicas", nargs="*", default=[])
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--out", default="runs/comparison")
    args = ap.parse_args()
    replicas = args.replicas + args.runs
    if not replicas and not args.compare:
        ap.error("give at least one run")
    rep = [load(r) for r in replicas]
    cmp_ = [load(r) for r in args.compare]
    keys = [k for k in (rep or cmp_)[0] if not k.startswith("frames")]
    rows, table = [], {}
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in rep], float)
        entry = {}
        if len(rep):
            entry["replica_mean"] = float(np.nanmean(vals))
            entry["replica_sem"] = float(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(np.isfinite(vals))))\
                if np.sum(np.isfinite(vals)) > 1 else None
            entry["replicas"] = vals.tolist()
        for name, r in zip(args.compare, cmp_):
            entry[Path(name).name] = r.get(k)
        table[k] = entry
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"replicas": replicas, "compare": args.compare,
                                                     "table": table}, indent=2))
    head = ["quantity"]
    if rep:
        head += [f"mean ± SEM ({len(rep)} replicas)"]
    head += [Path(c).name for c in args.compare]
    lines = ["# Comparison of runs", "", f"Replicas: {', '.join(replicas) or '-'}  ", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    fmt = lambda x: "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4g}"
    for k, e in table.items():
        cells = [k]
        if rep:
            cells.append(f"{fmt(e['replica_mean'])} ± {fmt(e['replica_sem'])}" if e["replica_sem"] is not None
                         else fmt(e["replica_mean"]))
        cells += [fmt(e.get(Path(c).name)) for c in args.compare]
        lines.append("| " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.json')} ({len(table)} quantities)")


if __name__ == "__main__":
    main()
