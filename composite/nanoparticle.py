"""Stage 1 - cut a charge-neutral spherical nanoparticle out of the periodic bulk glass.

A plain geometric cut slices through SiO4 / PO4 tetrahedra and leaves a net
charge.  Here the particle is built so that neither happens:

  1. network formers (Si, P) inside the sphere are selected;
  2. every oxygen bonded to a selected former is kept (whole tetrahedra), as
     are oxygens inside the sphere that belong to no former at all;
  3. modifiers (Na, Ca) inside the sphere are selected;
  4. the remaining charge is removed exactly by adding the nearest modifier
     cations just outside the sphere (or dropping the outermost ones).

The sphere centre is chosen among random trial centres as the one whose neutral
particle is closest to the bulk oxide composition.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import forcefield as ff
from .io import box_from_comment, read_xyz, write_extxyz

FORMERS = ("Si", "P")
MODIFIERS = ("Na", "Ca")
OXIDES = {"Si": ("SiO2", 1), "P": ("P2O5", 2), "Na": ("Na2O", 2), "Ca": ("CaO", 1)}


def oxide_mol_percent(symbols):
    count = Counter(symbols)
    units = {name: count.get(el, 0) / per for el, (name, per) in OXIDES.items()}
    total = sum(units.values())
    return {name: 100.0 * v / total for name, v in units.items()}


def formal_charge(symbols):
    return int(sum(ff.FORMAL_CHARGE[s] for s in symbols))


class Bulk:
    def __init__(self, symbols, pos, box, bond_cutoff):
        self.symbols, self.box = symbols, box
        self.pos = pos - box * np.floor(pos / box)
        self.is_former = np.isin(symbols, FORMERS)
        self.is_modifier = np.isin(symbols, MODIFIERS)
        self.is_oxygen = symbols == "O"
        self.q = np.array([ff.FORMAL_CHARGE[s] for s in symbols])
        self.mass = np.array([ff.MASS[ff.TYPE_INDEX[ff.GLASS_TYPE[s]]] for s in symbols])
        # former-oxygen bond graph
        pairs = cKDTree(self.pos, boxsize=box).query_pairs(bond_cutoff, output_type="ndarray")
        fo = self.is_former[pairs[:, 0]] & self.is_oxygen[pairs[:, 1]]
        of = self.is_oxygen[pairs[:, 0]] & self.is_former[pairs[:, 1]]
        self.bond_former = np.concatenate([pairs[fo, 0], pairs[of, 1]])
        self.bond_oxygen = np.concatenate([pairs[fo, 1], pairs[of, 0]])
        self.o_coordination = np.bincount(self.bond_oxygen, minlength=len(symbols))
        self.density = self.mass.sum() / box ** 3 * ff.AMU_A3_TO_GCC

    def displacement(self, centre):
        d = self.pos - centre
        return d - self.box * np.round(d / self.box)

    def select(self, centre, radius):
        """Boolean mask of the whole-tetrahedra cut, and distances from the centre."""
        r = np.linalg.norm(self.displacement(centre), axis=1)
        inside = r <= radius
        mask = inside & (self.is_former | self.is_modifier)
        mask[self.bond_oxygen[inside[self.bond_former]]] = True
        mask |= inside & self.is_oxygen & (self.o_coordination == 0)
        return mask, r

    def neutralise(self, mask, r):
        """Make the selection exactly neutral by adding / removing modifier cations."""
        mask = mask.copy()
        excess = int(self.q[mask].sum())
        changed = []
        if excess < 0:      # add the closest unselected cations
            candidates = np.flatnonzero(self.is_modifier & ~mask)
            candidates = candidates[np.argsort(r[candidates])]
        else:               # drop the outermost selected cations
            candidates = np.flatnonzero(self.is_modifier & mask)
            candidates = candidates[np.argsort(-r[candidates])]
        for i in candidates:
            if excess == 0:
                break
            if self.q[i] <= abs(excess):
                mask[i] = excess < 0
                excess += self.q[i] if excess < 0 else -self.q[i]
                changed.append(int(i))
        if excess != 0:
            raise RuntimeError("could not neutralise the nanoparticle")
        return mask, changed


def hydroxylate(sym, xyz, fraction, bond_cutoff, shell=3.0, rng=None):
    """Exchange a fraction of the surface Na+ ions for H+ on non-bridging oxygens.

    Na+/H+ exchange is the first step of the surface reaction of 45S5 glass; it turns
    Si-O(-) Na(+) into Si-OH and keeps the particle exactly neutral (both ions carry
    +0.6 e in the Teter scheme).  Each removed Na protonates the most exposed free NBO
    within 3.2 A of it; the H is placed 0.96 A from the O at a Si-O-H angle of 118 deg,
    in the direction that keeps it farthest from the other glass atoms."""
    if fraction <= 0.0:
        return sym, xyz, 0
    rng = rng or np.random.default_rng(0)
    r = np.linalg.norm(xyz, axis=1)
    rg = np.sqrt(np.mean(r ** 2))
    r_eq = np.sqrt(5.0 / 3.0) * rg
    tree = cKDTree(xyz)
    formers = np.flatnonzero(np.isin(sym, FORMERS))
    is_o = sym == "O"
    former_nb = {i: [j for j in tree.query_ball_point(xyz[i], bond_cutoff) if j in set(formers)]
                 for i in np.flatnonzero(is_o)}
    nbo = {i for i, nb in former_nb.items() if len(nb) == 1 and r[i] > r_eq - shell - 1.0}
    surface_na = [i for i in np.argsort(-r) if sym[i] == "Na" and r[i] > r_eq - shell]
    n_exchange = int(round(fraction * len(surface_na)))
    remove, new_h = [], []
    others = cKDTree(xyz)
    for na in surface_na:
        if len(remove) == n_exchange:
            break
        placed = None
        for o in sorted((o for o in tree.query_ball_point(xyz[na], 3.2) if o in nbo),
                        key=lambda k: -r[k]):
            u = xyz[o] - xyz[former_nb[o][0]]
            u /= np.linalg.norm(u)
            perp = np.cross(u, rng.normal(size=3))
            perp /= np.linalg.norm(perp)
            best, best_d = None, -1.0
            for phi in np.linspace(0, 2 * np.pi, 72, endpoint=False):
                w = np.cos(phi) * perp + np.sin(phi) * np.cross(u, perp)
                h = xyz[o] + 0.96 * (np.cos(np.radians(62.0)) * u + np.sin(np.radians(62.0)) * w)
                d, k = others.query(h, k=4)
                d = min(dd for dd, kk in zip(d, k) if kk not in (o, na, former_nb[o][0]))
                if d > best_d:
                    best, best_d = h, d
            if best_d >= 2.0:              # no clash with the rest of the glass
                placed = (o, best)
                break
        if placed is None:
            continue
        nbo.discard(placed[0])
        remove.append(na)
        new_h.append(placed[1])
    keep = np.setdiff1d(np.arange(len(sym)), remove)
    sym = np.concatenate([sym[keep], np.array(["H"] * len(new_h))])
    xyz = np.concatenate([xyz[keep], np.array(new_h).reshape(-1, 3)])
    return sym, xyz, len(new_h)


def cut_nanoparticle(cfg, outdir):
    np_cfg = cfg["nanoparticle"]
    radius = float(np_cfg["radius"])
    bond_cutoff = float(np_cfg["bond_cutoff"])
    symbols, pos, comment = read_xyz(cfg["system"]["bulk_glass"])
    box = float(np_cfg.get("bulk_box") or box_from_comment(comment) or 0.0)
    if box <= 0.0:
        raise ValueError("bulk box size not found: set nanoparticle.bulk_box in the config")
    if radius + bond_cutoff + 3.0 > box / 2:
        raise ValueError(f"radius {radius} A is too large for a {box:.2f} A bulk cell "
                         f"(maximum about {box / 2 - bond_cutoff - 3.0:.1f} A)")

    bulk = Bulk(symbols, pos, box, bond_cutoff)
    rng = np.random.default_rng(cfg["system"]["seed"])
    if np_cfg["center"] == "auto":
        centres = rng.random((int(np_cfg["center_trials"]), 3)) * box
    else:
        centres = np.array([np_cfg["center"]], dtype=float)

    # keep the neutral particle whose oxide composition is closest to the bulk glass
    target = oxide_mol_percent(bulk.symbols)
    best = None
    for centre in centres:
        raw_mask, r = bulk.select(centre, radius)
        mask, changed = bulk.neutralise(raw_mask, r)
        comp = oxide_mol_percent(bulk.symbols[mask])
        score = sum(abs(comp[k] - target[k]) for k in target)
        if best is None or score < best[0]:
            best = (score, centre, raw_mask, mask, changed)
    _, centre, raw_mask, mask, changed = best
    raw_excess = abs(int(bulk.q[raw_mask].sum()))

    sym = bulk.symbols[mask]
    xyz = bulk.displacement(centre)[mask]
    mass = bulk.mass[mask]
    xyz -= np.average(xyz, axis=0, weights=mass)       # centre of mass at the origin
    order = np.lexsort((np.linalg.norm(xyz, axis=1), sym))
    sym, xyz, mass = sym[order], xyz[order], mass[order]
    sym, xyz, n_silanol = hydroxylate(sym, xyz, float(np_cfg.get("hydroxyl_fraction", 0.0)),
                                      bond_cutoff, rng=rng)
    mass = np.array([ff.MASS[ff.TYPE_INDEX[ff.GLASS_TYPE[s]]] for s in sym])

    # ---- diagnostics
    dist = np.linalg.norm(xyz, axis=1)
    rg = np.sqrt(np.average(dist ** 2, weights=mass))
    tree = cKDTree(xyz)
    o_idx = np.flatnonzero(sym == "O")
    former_idx = np.flatnonzero(np.isin(sym, FORMERS))
    former_cn = np.array([np.sum(sym[tree.query_ball_point(xyz[i], bond_cutoff)] == "O")
                          for i in former_idx])
    o_cn = np.array([np.sum(np.isin(sym[tree.query_ball_point(xyz[i], bond_cutoff)], FORMERS))
                     for i in o_idx])
    bulk_o_cn = bulk.o_coordination[bulk.is_oxygen]
    report = {
        "radius_requested": radius,
        "radius_max": float(dist.max()),
        "radius_equivalent": float(np.sqrt(5.0 / 3.0) * rg),
        "centre_in_bulk": centre.tolist(),
        "n_atoms": int(len(sym)),
        "composition": dict(sorted(Counter(sym.tolist()).items())),
        "mass_amu": float(mass.sum()),
        "bulk_density_gcc": float(bulk.density),
        "volume_A3": float(mass.sum() / bulk.density * ff.AMU_A3_TO_GCC),
        "net_charge_e": 0.6 * formal_charge(sym),
        "charge_before_neutralisation_e": 0.6 * raw_excess * np.sign(bulk.q[raw_mask].sum()),
        "modifiers_adjusted": len(changed),
        "silanol_groups": n_silanol,
        "silanol_per_nm2": n_silanol / (4 * np.pi * (np.sqrt(5.0 / 3.0) * rg) ** 2 / 100.0),
        "oxide_mol_percent": oxide_mol_percent(sym),
        "oxide_mol_percent_bulk": oxide_mol_percent(bulk.symbols),
        "former_4fold_fraction": float(np.mean(former_cn == 4)),
        "nonbridging_oxygen_fraction": float(np.mean(o_cn == 1)),
        "nonbridging_oxygen_fraction_bulk": float(np.mean(bulk_o_cn == 1)),
    }

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_extxyz(outdir / "nanoparticle.xyz", sym, xyz,
                 info=f'radius={radius} charge={report["net_charge_e"]:.3f}')
    (outdir / "nanoparticle.json").write_text(json.dumps(report, indent=2))

    print(f"  bulk: {len(symbols)} atoms, box {box:.3f} A, density {bulk.density:.3f} g/cm3")
    print(f"  cut : R = {radius:.1f} A -> {len(sym)} atoms "
          f"(R_max {report['radius_max']:.2f} A, R_equiv {report['radius_equivalent']:.2f} A)")
    print(f"  composition: {report['composition']}")
    print("  oxide mol% (particle / bulk): " + ", ".join(
        f"{k} {v:.1f}/{report['oxide_mol_percent_bulk'][k]:.1f}"
        for k, v in report["oxide_mol_percent"].items()))
    print(f"  charge {report['charge_before_neutralisation_e']:+.1f} e -> "
          f"{report['net_charge_e']:+.1f} e after adjusting {len(changed)} modifier cations")
    if n_silanol:
        print(f"  surface hydroxylation: {n_silanol} Na+ exchanged for H+ "
              f"({report['silanol_per_nm2']:.2f} Si-OH per nm^2)")
    print(f"  4-fold Si/P: {100 * report['former_4fold_fraction']:.1f} %, "
          f"non-bridging O: {100 * report['nonbridging_oxygen_fraction']:.1f} % "
          f"(bulk {100 * report['nonbridging_oxygen_fraction_bulk']:.1f} %)")
    return report
