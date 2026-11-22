"""Additional analyses of the PCL / 45S5 runs: interface profiles pooled over the replicas and a chain-relaxation test.

    python extra_analysis.py runs/composite_r1 runs/composite_r2 runs/composite_r3 --neat runs/neat_r1 --pooled runs/pooled

1. Periodic-image bridging: chains that touch two different periodic images of the nanoparticle in the final structure
   (finite-size check; contact = any atom within 3.0 A of a glass atom).
2. Chain relaxation: autocorrelation of the unit end-to-end vector, C(t) = < u(0) . u(t) >, over the trajectory frames of the
   equilibration and production stages.  C close to 1 after thousands of moves means that the chains do not reorient
   under hybrid Monte Carlo, so chain statistics rest on the initial growth and are not equilibrium averages.
3. Interphase threshold sensitivity: the interphase thickness of the pooled radial PCL density for other values of the recovery
   tolerance, recovery window, smoothing window and bulk offset (needs pool_replicas.py output).

Output: runs/extra_analysis/ (bridging.csv, chain_autocorrelation.csv, threshold_sensitivity.csv,
extra_summary.json).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from composite.io import System
from composite.structure import DEFAULTS, Topology, read_frames

CONTACT = float(DEFAULTS["contact"])


# ------------------------------------------------------------------ 1. bridging
def unwrap_chains(pos, mol, bonds, box):
    adj = {}
    for i, j in bonds:
        adj.setdefault(int(i), []).append(int(j))
        adj.setdefault(int(j), []).append(int(i))
    out = pos.copy()
    seen = set()
    for start in np.where(mol > 0)[0]:
        start = int(start)
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for b in adj.get(a, []):
                if b in seen or mol[b] != mol[a]:
                    continue
                d = pos[b] - out[a]
                d -= box * np.round(d / box)
                out[b] = out[a] + d
                seen.add(b)
                queue.append(b)
    return out


def bridging(run):
    s = System.load(Path(run) / "03_equilibration" / "system_final.npz")
    L, pos, mol = s.box, s.pos, s.mol
    glass = pos[mol == 0]
    chains = np.unique(mol[mol > 0])
    up = unwrap_chains(pos, mol, s.bonds, L)
    shifts = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    touched = {}
    idx = np.where(mol > 0)[0]
    for sh in shifts:
        tree = cKDTree(glass + np.array(sh) * L)
        for a, h in zip(idx, tree.query_ball_point(up[idx], CONTACT)):
            if len(h):
                touched.setdefault(int(mol[a]), set()).add(sh)
    r_max = np.linalg.norm(glass - glass.mean(axis=0), axis=1).max()
    n_bridge = sum(1 for v in touched.values() if len(v) >= 2)
    return dict(run=Path(run).name, box_A=L, particle_max_radius_A=r_max, surface_to_surface_gap_A=L - 2 * r_max,
                n_chains=len(chains), n_chains_touching_particle=len(touched), n_chains_touching_two_images=n_bridge,
                fraction_of_all_chains=n_bridge / len(chains), fraction_of_touching_chains=n_bridge / max(1, len(touched)))


# ------------------------------------------------------------------ 2. chain relaxation
def autocorrelation(run, stages=("equilibrate", "production"), max_lag_moves=8000):
    s = System.load(Path(run) / "03_equilibration" / "system_final.npz")
    top = Topology(s, dict(DEFAULTS))
    ends = np.array([ch["ends"] for ch in top.chains])
    frames = list(read_frames(Path(run) / "03_equilibration" / "trajectory.xyz", s.n, set(stages)))
    moves = np.array([f[3] for f in frames])
    u = []
    for box, wpos, _, _ in frames:
        unw = top.unwrap(wpos, box)
        v = unw[ends[:, 1]] - unw[ends[:, 0]]
        u.append(v / np.linalg.norm(v, axis=1)[:, None])
    u = np.array(u)                                              # frame, chain, 3
    step = int(np.median(np.diff(moves)))
    lags = np.arange(1, max_lag_moves // step + 1)
    rows = []
    for k in lags:
        c = np.einsum("fci,fci->fc", u[:-k], u[k:])              # over origins and chains
        per_chain = c.mean(axis=0)
        rows.append((int(k * step), float(per_chain.mean()), float(per_chain.std(ddof=1) / np.sqrt(len(per_chain)))))
    return dict(run=Path(run).name, stages=list(stages), n_frames=len(frames), frame_spacing_moves=step, rows=rows,
                n_chains=int(u.shape[1]))


# ------------------------------------------------------------------ 3. interphase threshold sensitivity
def thickness(r, rho, r_eq, tol, window, smooth_w, offset):
    dr = float(np.median(np.diff(r)))
    bulk = float(np.nanmean(rho[r > r_eq + offset]))
    n_s = max(1, int(round(smooth_w / dr)))
    smooth = np.convolve(rho, np.ones(n_s) / n_s, mode="same")
    peak = np.flatnonzero((r > r_eq) & (r < r_eq + 7))
    i_peak = peak[np.nanargmax(rho[peak])]
    ok = np.abs(smooth / bulk - 1.0) < tol
    run = int(round(window / dr))
    r_rec = next((float(r[i]) for i in range(i_peak, len(r) - run) if ok[i:i + run].all()), float("nan"))
    return r_rec - r_eq, float(rho[i_peak] / bulk)


def threshold_sensitivity(pooled_dir, r_eq):
    with open(Path(pooled_dir) / "pooled_radial_density.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    r = data[:, head.index("r_A")]
    rho = data[:, head.index("rho_PCL_mean")]
    out = []
    for tol, window, smooth_w, offset in itertools.product((0.03, 0.05, 0.10), (2.0, 3.0, 4.0), (1.0, 2.0, 3.0), (6.0, 8.0, 10.0)):
        t, peak = thickness(r, rho, r_eq, tol, window, smooth_w, offset)
        out.append([tol * 100, window, smooth_w, offset, t, peak])
    return out


def write_csv(path, head, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--neat", default=None)
    ap.add_argument("--pooled", default="runs/pooled")
    ap.add_argument("--out", default="runs/extra_analysis")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}

    br = [bridging(r) for r in args.runs]
    keys = list(br[0])
    write_csv(out / "bridging.csv", keys, [[b[k] for k in keys] for b in br])
    res["bridging"] = br
    for b in br:
        print(f"{b['run']}: box {b['box_A']:.2f} A, gap {b['surface_to_surface_gap_A']:.1f} A, "
              f"{b['n_chains_touching_particle']} of {b['n_chains']} chains touch the particle, "
              f"{b['n_chains_touching_two_images']} touch two images")

    ac = [autocorrelation(r) for r in args.runs + ([args.neat] if args.neat else [])]
    with open(out / "chain_autocorrelation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "lag_moves", "C_ee", "sem_over_chains"])
        for a in ac:
            for lag, c, e in a["rows"]:
                w.writerow([a["run"], lag, f"{c:.5f}", f"{e:.5f}"])
    res["chain_autocorrelation"] = [{"run": a["run"], "frames": a["n_frames"], "spacing_moves": a["frame_spacing_moves"],
                                     "C_at_max_lag": a["rows"][-1][1], "max_lag_moves": a["rows"][-1][0],
                                     "C_at_2000": next((c for l, c, _ in a["rows"] if l == 2000), None)} for a in ac]
    for a in res["chain_autocorrelation"]:
        print(f"{a['run']}: end-to-end autocorrelation C({a['max_lag_moves']} moves) = {a['C_at_max_lag']:.3f}")

    if (Path(args.pooled) / "pooled_radial_density.csv").exists():
        r_eq = float(np.mean([json.loads((Path(r) / "01_nanoparticle" / "nanoparticle.json").read_text())["radius_equivalent"]
                              for r in args.runs]))
        sens = threshold_sensitivity(args.pooled, r_eq)
        write_csv(out / "threshold_sensitivity.csv",
                  ["tolerance_percent", "recovery_window_A", "smoothing_window_A", "bulk_offset_from_R_eq_A",
                   "interphase_thickness_A", "first_peak_relative_to_bulk"], sens)
        th = np.array([o[4] for o in sens], float)
        res["interphase_threshold_sensitivity"] = {
            "n_settings": len(sens), "n_nan": int(np.isnan(th).sum()), "min_A": float(np.nanmin(th)),
            "max_A": float(np.nanmax(th)), "median_A": float(np.nanmedian(th)),
            "reference_5pct_3A_2A_10A": [o[4] for o in sens if o[:4] == [5.0, 3.0, 2.0, 10.0]],
            "first_peak_relative_to_bulk_range": [min(o[5] for o in sens), max(o[5] for o in sens)]}
        for tol in (3.0, 5.0, 10.0):
            sub = np.array([o[4] for o in sens if o[0] == tol], float)
            res["interphase_threshold_sensitivity"][f"tol_{tol:g}pct_range_A"] = [float(np.nanmin(sub)), float(np.nanmax(sub))]
        print("interphase thickness over", len(sens), "settings:", f"{np.nanmin(th):.2f} to {np.nanmax(th):.2f} A "
              f"(median {np.nanmedian(th):.2f}, nan {int(np.isnan(th).sum())})")
    else:
        print("no pooled profile found (run pool_replicas.py first); threshold sensitivity skipped")

    (out / "extra_summary.json").write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
