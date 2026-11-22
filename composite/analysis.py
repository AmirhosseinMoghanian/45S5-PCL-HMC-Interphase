"""Stage 4 - convergence checks and structural analysis of the equilibrated composite."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import forcefield as ff
from .io import System


def read_log(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    num = lambda key: np.array([float(r[key]) if r[key] else np.nan for r in rows])
    return {"stage": np.array([r["stage"] for r in rows]),
            **{k: num(k) for k in rows[0] if k != "stage"}}


def drift(values):
    """Change along the series (linear fit, first to last point) in units of its
    standard deviation about the fit; |drift| below ~1 means no visible trend."""
    if len(values) < 4:
        return float("nan")
    x = np.linspace(-0.5, 0.5, len(values))
    slope, intercept = np.polyfit(x, values, 1)
    resid = values - (slope * x + intercept)
    return float(slope / max(resid.std(), 1e-12))


def radial_density(system, dr=0.5):
    """Mass density of glass and polymer in spherical shells around the particle centre."""
    box = system.box
    pos = system.pos - box * np.floor(system.pos / box + 0.5)
    r = np.linalg.norm(pos, axis=1)
    edges = np.arange(0.0, box / 2 + dr, dr)
    shell = 4.0 / 3.0 * np.pi * np.diff(edges ** 3)
    glass = system.mol == 0
    rho = lambda sel: (np.histogram(r[sel], edges, weights=system.mass[sel])[0]
                       / shell * ff.AMU_A3_TO_GCC)
    return 0.5 * (edges[1:] + edges[:-1]), rho(glass), rho(~glass)


def chain_dimensions(system):
    rg, ree = [], []
    for m in range(1, int(system.mol.max()) + 1):
        sel = system.mol == m
        xyz, mass = system.pos[sel], system.mass[sel]
        com = np.average(xyz, axis=0, weights=mass)
        rg.append(np.sqrt(np.average(((xyz - com) ** 2).sum(1), weights=mass)))
        ree.append(np.linalg.norm(xyz[-3] - xyz[0]))      # acid O ... hydroxyl O
    return np.array(rg), np.array(ree)


def interface_contacts(system, cutoff=3.0):
    """Polymer atoms in contact with the glass surface, by polymer atom type."""
    box = system.box
    wrapped = np.mod(system.pos + box / 2, box)
    wrapped[wrapped >= box] = 0.0
    glass = np.flatnonzero(system.mol == 0)
    poly = np.flatnonzero(system.mol > 0)
    tree_g = cKDTree(wrapped[glass], boxsize=box)
    dist, nearest = tree_g.query(wrapped[poly], distance_upper_bound=cutoff)
    hit = np.isfinite(dist)
    pairs = {}
    for p, g in zip(poly[hit], glass[nearest[hit]]):
        key = f"{ff.TYPE_NAMES[system.type[p]]}...{ff.TYPE_NAMES[system.type[g]]}"
        pairs[key] = pairs.get(key, 0) + 1
    d_min = tree_g.query(wrapped[poly])[0].min()
    return dict(sorted(pairs.items(), key=lambda kv: -kv[1])), float(d_min)


def analyse(cfg, mc_dir, outdir):
    mc_dir, outdir = Path(mc_dir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    system = System.load(mc_dir / "system_final.npz")
    log = read_log(mc_dir / "log.csv")
    last = log["stage"] == log["stage"][-1]
    half = np.flatnonzero(last)[np.count_nonzero(last) // 2:]

    r, rho_glass, rho_poly = radial_density(system)
    rg, ree = chain_dimensions(system)
    if system.has_particle:
        contacts, d_min = interface_contacts(system)
        far = r > r[rho_glass > 0.05 * rho_glass.max()].max() + 6.0
    else:                                   # neat-PCL control
        contacts, d_min = {}, float("nan")
        far = np.ones_like(r, bool)
    report = {
        "n_atoms": system.n,
        "box_A": system.box,
        "density_gcc": system.density,
        "glass_weight_percent": 100.0 * system.glass_weight_fraction,
        "polymer_weight_percent": 100.0 * (1.0 - system.glass_weight_fraction),
        "last_stage": str(log["stage"][-1]),
        "last_stage_second_half": {
            "density_mean": float(np.mean(log["density_gcc"][half])),
            "density_std": float(np.std(log["density_gcc"][half])),
            "density_drift_sigma": drift(log["density_gcc"][half]),
            "energy_mean": float(np.mean(log["E_total"][half])),
            "energy_std": float(np.std(log["E_total"][half])),
            "energy_drift_sigma": drift(log["E_total"][half]),
            "hmc_acceptance": float(np.nanmean(log["acc_hmc"][half])),
        },
        "polymer_density_far_from_particle_gcc": float(rho_poly[far].mean()) if far.any() else None,
        "radius_of_gyration_A": {"mean": float(rg.mean()), "std": float(rg.std())},
        "end_to_end_A": {"mean": float(ree.mean()), "std": float(ree.std())},
        "min_polymer_glass_distance_A": d_min,
        "interface_contacts_within_3A": contacts,
    }
    (outdir / "analysis.json").write_text(json.dumps(report, indent=2))
    np.savetxt(outdir / "radial_density.dat", np.c_[r, rho_glass, rho_poly],
               header="r[A]  rho_glass[g/cm3]  rho_polymer[g/cm3]", fmt="%.5f")

    s = report["last_stage_second_half"]
    print(f"  density {system.density:.4f} g/cm3 | polymer : glass = "
          f"{report['polymer_weight_percent']:.2f} : {report['glass_weight_percent']:.2f} wt%")
    print(f"  stage '{report['last_stage']}' (second half): rho = {s['density_mean']:.4f} "
          f"+/- {s['density_std']:.4f}, drift {s['density_drift_sigma']:+.2f} sigma; "
          f"E drift {s['energy_drift_sigma']:+.2f} sigma")
    print(f"  Rg = {rg.mean():.2f} +/- {rg.std():.2f} A"
          + (f", closest polymer-glass contact {d_min:.2f} A" if system.has_particle else ""))
    return report
