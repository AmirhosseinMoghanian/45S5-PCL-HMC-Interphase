"""Pool the profiles of independent replicas.

    python pool_replicas.py runs/composite_r1 runs/composite_r2 runs/composite_r3 --neat runs/neat_r1

Averages the radial density and group-enrichment profiles of the replicas (with the standard
error over replicas), takes the first peaks of the glass-polymer RDFs from the pooled partial
RDFs, and reads the interphase thickness from the pooled density, which is far less noisy than
any single run.  Output: runs/pooled/ (pooled_radial_density.csv, pooled_summary.json/.md).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import warnings

import numpy as np


def read_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    head, data = rows[0], np.array([[float(v) if v not in ("", "nan") else np.nan for v in r]
                                    for r in rows[1:]])
    return head, data


def pooled(runs, name, key_col=0):
    """Align the replicas' tables on their first column and return x, mean, sem (per column)."""
    tables = [read_csv(Path(r) / "05_structure" / "data" / name) for r in runs]
    head = tables[0][0]
    keys = sorted(set.intersection(*[set(np.round(t[1][:, key_col], 4)) for t in tables]))
    stack = np.array([[t[1][np.flatnonzero(np.round(t[1][:, key_col], 4) == k)[0]] for k in keys]
                      for t in tables])                       # replica, row, column
    mean = np.nanmean(stack, axis=0)
    n = np.sum(np.isfinite(stack), axis=0)
    sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)) if len(runs) > 1 else np.zeros_like(mean)
    return head, np.array(keys), mean, sem


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty / single-valued bins
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--neat", default=None, help="neat-PCL control run, for its bulk density")
    ap.add_argument("--out", default="runs/pooled")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((Path(r) / "05_structure" / "structure_summary.json").read_text()) for r in args.runs]
    r_eq = float(np.mean([s["nanoparticle"]["R_equivalent_A"] for s in summaries]))
    neat = json.loads((Path(args.neat) / "05_structure" / "structure_summary.json").read_text()) if args.neat else None
    rho_neat = neat["neat_PCL"]["density_gcc"]["mean"] if neat else None
    res = {"replicas": args.runs, "neat_reference": args.neat, "R_eq_A": round(r_eq, 3)}

    # ---- radial density and interphase from the pooled profile
    head, r, m, e = pooled(args.runs, "radial_density.csv")
    rho, rho_e, rho_g = m[:, 2], e[:, 2], m[:, 1]
    dr = np.median(np.diff(r))
    bulk = r > r_eq + 10.0
    rho_bulk = float(np.nanmean(rho[bulk]))
    n_s = max(1, int(round(2.0 / dr)))
    smooth = np.convolve(rho, np.ones(n_s) / n_s, mode="same")
    peak = np.flatnonzero((r > r_eq) & (r < r_eq + 7))
    i_peak = peak[np.nanargmax(rho[peak])]
    ok = np.abs(smooth / rho_bulk - 1.0) < 0.05
    run = int(round(3.0 / dr))
    r_rec = next((float(r[i]) for i in range(i_peak, len(r) - run) if ok[i:i + run].all()), float("nan"))
    minima = [i for i in range(i_peak + 1, len(r) - 1) if smooth[i] < smooth[i - 1] and smooth[i] <= smooth[i + 1]]
    res["interphase"] = {
        "PCL_bulk_density_gcc (pooled, r > R_eq + 10 A)": round(rho_bulk, 4),
        "neat_PCL_density_gcc": rho_neat,
        "first_peak_r_A": round(float(r[i_peak]), 3),
        "first_peak_distance_from_R_eq_A": round(float(r[i_peak] - r_eq), 3),
        "first_peak_rho_gcc": round(float(rho[i_peak]), 4),
        "first_peak_relative_to_bulk": round(float(rho[i_peak] / rho_bulk), 4),
        "first_minimum_r_A": round(float(r[minima[0]]), 3) if minima else None,
        "first_minimum_relative_to_bulk": round(float(smooth[minima[0]] / rho_bulk), 4) if minima else None,
        "recovery_within_5pct_for_3A_from_r_A": round(r_rec, 3),
        "interphase_thickness_A (pooled)": round(r_rec - r_eq, 3),
    }
    groups = head[4:]
    with open(out / "pooled_radial_density.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["r_A", "rho_glass", "rho_PCL_mean", "rho_PCL_sem_over_replicas"] + [f"{g}_mean" for g in groups])
        for k in range(len(r)):
            w.writerow([f"{r[k]:.4f}", f"{rho_g[k]:.5f}", f"{rho[k]:.5f}", f"{rho_e[k]:.5f}"]
                       + [f"{m[k, 4 + j]:.5f}" for j in range(len(groups))])

    # ---- glass-polymer RDFs
    head, x, m, e = pooled(args.runs, "rdf_glass_polymer.csv")
    names = head[1:]
    rdf_res = {}
    for j, n in enumerate(names):
        g = m[:, 1 + j]
        near = (x > 1.0) & (x < 4.0)
        if np.nanmax(g[near]) < 0.5:
            continue
        ip = np.flatnonzero(near)[np.nanargmax(g[near])]
        rdf_res[n] = {"first_peak_A": round(float(x[ip]), 3), "first_peak_g": round(float(g[ip]), 3),
                      "first_peak_g_sem": round(float(e[ip, 1 + j]), 3)}
    res["rdf_first_peaks"] = rdf_res

    (out / "pooled_summary.json").write_text(json.dumps(res, indent=2))
    i = res["interphase"]
    lines = ["# Pooled replica analysis", "", f"Replicas: {', '.join(args.runs)}; neat reference: {args.neat or '-'}.", "",
             "| quantity | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in i.items()] + \
            ["", "| RDF pair | first peak (Å) | g(peak) |", "|---|---|---|"] + \
            [f"| {k} | {v['first_peak_A']} | {v['first_peak_g']} ± {v['first_peak_g_sem']} |" for k, v in rdf_res.items()]
    (out / "pooled_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"interphase from the pooled profile: first peak {i['first_peak_relative_to_bulk']}x bulk at "
          f"{i['first_peak_distance_from_R_eq_A']} A from R_eq, recovery at {i['recovery_within_5pct_for_3A_from_r_A']} A "
          f"(thickness {i['interphase_thickness_A (pooled)']} A); PCL bulk {i['PCL_bulk_density_gcc (pooled, r > R_eq + 10 A)']} "
          f"vs neat {rho_neat}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
