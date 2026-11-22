"""Stage 5 - structural analysis of the equilibrated composite.

All polymer quantities are averaged over the trajectory frames of the selected
stages (default: production); error bars are standard errors of block averages.
The nanoparticle is rigid, so its own structure is analysed once.

Results (numbers only, no plots)
  structure_summary.json   every scalar result (mean, standard error)
  structure_report.md      the same quantities as markdown tables
  data/*.csv               every profile and distribution of the run
"""

from __future__ import annotations

import csv
import warnings
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from . import forcefield as ff
from .io import System

# --------------------------------------------------------------------------- settings
DEFAULTS = {
    "stages": ["production"],      # trajectory frames used for the polymer averages
    "n_blocks": 4,                 # blocks for the standard errors
    "dr": 0.25,                    # radial bin width (A)
    "contact": 3.0,                # polymer-glass contact distance (A), any atom pair
    "hbond_distance": 2.5,         # H...acceptor distance (A)
    "hbond_angle": 120.0,          # donor-H...acceptor angle (deg)
    "rdf_rmax": 8.0,
    "former_cutoff": 2.1,          # Si/P-O bond inside the glass
    "modifier_cutoff": 3.0,        # Na/Ca-O first shell inside the glass
    "surface_shell": 3.0,          # glass atoms with r > R_eq - shell are "surface"
    "free_volume_spacing": 0.5,    # grid spacing (A)
    "free_volume_frames": 8,
    "probes": [0.0, 1.0, 1.4],     # probe radii (A) for the free-volume fractions
}
HB_KINDS = ("PCL OH -> NBO", "PCL OH -> BO", "PCL OH -> Si-OH", "PCL OH -> PCL O", "Si-OH -> PCL O")
VDW = {"H": 1.20, "C": 1.70, "O": 1.52, "Si": 2.10, "P": 1.80, "Na": 2.27, "Ca": 2.31}  # Bondi


# --------------------------------------------------------------------------- helpers
def mic(d, box):
    return d - box * np.round(d / box)


def wrap(x, box):
    w = np.mod(x + box / 2, box)
    w[w >= box] = 0.0
    return w


def dihedral(a, b, c, d):
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    x = np.einsum("ij,ij->i", n1, n2)
    y = np.einsum("ij,ij->i", np.cross(n1, n2), b2 / np.linalg.norm(b2, axis=1)[:, None])
    return np.degrees(np.arctan2(y, x))


def block_stats(values, n_blocks):
    """Mean and standard error from block averages of a per-frame series."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    blocks = np.array([b.mean(axis=0) for b in np.array_split(v, min(n_blocks, len(v)))])
    err = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks)) if len(blocks) > 1 else np.zeros(v.shape[1])
    return v.mean(axis=0).squeeze(), err.squeeze()


def ms(values, n_blocks, digits=4):
    m, e = block_stats(values, n_blocks)
    return {"mean": round(float(m), digits), "sem": round(float(e), digits)}


def write_csv(path, header, columns):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in zip(*columns):
            w.writerow([f"{v:.6g}" if isinstance(v, (float, np.floating)) else v for v in row])


def read_frames(path, n_atoms, stages):
    """Yield (box, wrapped positions, stage, move) for frames of the selected stages."""
    with open(path) as fh:
        while True:
            head = fh.readline()
            if not head:
                return
            comment = fh.readline()
            lines = [fh.readline() for _ in range(int(head))]
            tags = dict(t.split("=", 1) for t in comment.split() if "=" in t and '"' not in t)
            stage = tags.get("stage", "")
            if stage not in stages:
                continue
            box = float(comment.split('Lattice="')[1].split()[0])
            pos = np.array([l.split()[1:4] for l in lines], dtype=float)
            yield box, pos, stage, int(tags.get("move", 0))


# --------------------------------------------------------------------------- topology
def backbone_torsions(unw, ch):
    """Three backbone torsions per repeat unit (deg), columns:
    ester  : C5(previous unit) - O - C(=O) - C1   (trans = 180; NaN for the first unit)
    tau_oc : O - C(=O) - C1 - C2
    tau_cc : C(=O) - C1 - C2 - C3"""
    O, C, CH = ch["O"], ch["C"], ch["CH"]
    ester = np.r_[np.nan, dihedral(unw[CH[:-1, 4]], unw[O[1:]], unw[C[1:]], unw[CH[1:, 0]])]
    return np.c_[ester, dihedral(unw[O], unw[C], unw[CH[:, 0]], unw[CH[:, 1]]),
                 dihedral(unw[C], unw[CH[:, 0]], unw[CH[:, 1]], unw[CH[:, 2]])]


class Topology:
    def __init__(self, system: System, cfg):
        self.s = s = system
        self.names = np.array(ff.TYPE_NAMES)[s.type]
        self.elem = s.symbols
        self.glass = np.flatnonzero(s.mol == 0)
        self.poly = np.flatnonzero(s.mol > 0)
        n = s.n
        neigh = [[] for _ in range(n)]
        for i, j in s.bonds:
            neigh[i].append(j)
            neigh[j].append(i)
        self.neigh = neigh

        # chains: contiguous blocks, 18 n + 3 atoms (see pcl.py): acid-end unit 19 atoms, middle units 18,
        # hydroxyl-end unit 20.  Backbone of a unit: O - C(=O) - C1 - C2 - C3 - C4 - C5 - O(next unit / end OH).
        self.chains = []
        for m in range(1, int(s.mol.max()) + 1):
            idx = np.flatnonzero(s.mol == m)
            nu = (len(idx) - 3) // 18
            b = idx[0]
            start = np.r_[b, b + 19 + 18 * np.arange(nu - 1)]           # first atom of every unit
            unit_of = np.empty(len(idx), int)
            unit_of[:19] = 0
            unit_of[19:19 + 18 * (nu - 1)] = 1 + np.arange(18 * (nu - 1)) // 18
            unit_of[19 + 18 * (nu - 1):] = nu - 1
            lead = np.r_[1, np.zeros(nu - 1, int)]                      # unit 0 starts with HO, OH
            O = start + lead
            C = O + 1
            OC = O + 2
            CH = C[:, None] + 2 + 3 * np.arange(5)[None, :]             # the five CH2 carbons C1..C5
            O_end = b + len(idx) - 2                                     # hydroxyl oxygen of the last unit
            O_next = np.r_[O[1:], O_end]
            bb = np.concatenate([O[:, None], C[:, None], CH, O_next[:, None]], axis=1)     # (n, 8)
            self.chains.append(dict(idx=idx, n=nu, unit_of=unit_of, O=O, CA=CH[:, 0], C=C, OC=OC, CH=CH,
                                    O_next=O_next, bb=bb, ends=np.array([b + 1, O_end])))
        self.unit_id = np.full(n, -1)                       # global monomer id per atom
        self.monomer_chain, self.monomer_pos = [], []
        k = 0
        for ci, ch in enumerate(self.chains):
            self.unit_id[ch["idx"]] = k + ch["unit_of"]
            self.monomer_chain += [ci] * ch["n"]
            self.monomer_pos += list(range(ch["n"]))
            ch["first_monomer"] = k
            k += ch["n"]
        self.n_monomers = k
        self.monomer_chain = np.array(self.monomer_chain)

        # unwrapping order: breadth-first from the first atom of every chain
        parent = np.full(n, -1)
        depth = np.zeros(n, int)
        for ch in self.chains:
            root = ch["idx"][0]
            seen = {root}
            q = deque([root])
            while q:
                a = q.popleft()
                for c in neigh[a]:
                    if c not in seen:
                        seen.add(c)
                        parent[c], depth[c] = a, depth[a] + 1
                        q.append(c)
        self.levels = [np.flatnonzero(depth == d) for d in range(1, depth.max() + 1)]
        self.parent = parent

        # hydrogen-bond donors (polymer hydroxyl H, glass silanol H) and acceptors
        donors = np.flatnonzero(np.isin(self.names, ["HO_acid", "HO_end"]))
        self.donor_h = donors
        self.donor_o = np.array([neigh[h][0] for h in donors], dtype=int)
        self.poly_acceptors = np.flatnonzero(np.isin(
            self.names, ["O_carbonyl", "O_ester", "O_acid", "OH_acid", "OH_end"]))

        # glass structure (rigid)
        g = self.glass
        gp = s.pos[g]
        tree = cKDTree(gp)
        is_former = np.isin(self.elem[g], ["Si", "P"])
        is_o = self.elem[g] == "O"
        o_formers = np.zeros(len(g), int)
        for i in np.flatnonzero(is_o):
            nb = tree.query_ball_point(gp[i], cfg["former_cutoff"])
            o_formers[i] = int(np.sum(is_former[nb]))
        self.glass_o_formers = o_formers
        is_h = self.elem[g] == "H"
        protonated = np.zeros(len(g), bool)
        self.silanol_h, self.silanol_o = g[is_h], np.zeros(int(is_h.sum()), int)
        for k, i in enumerate(np.flatnonzero(is_h)):
            o = [j for j in tree.query_ball_point(gp[i], 1.2) if is_o[j]]
            protonated[o[0]] = True
            self.silanol_o[k] = g[o[0]]
        self.nbo = g[is_o & (o_formers == 1) & ~protonated]
        self.bo = g[is_o & (o_formers >= 2)]
        self.free_o = g[is_o & (o_formers == 0)]

    def unwrap(self, pos, box):
        out = pos.copy()
        for lev in self.levels:
            p = self.parent[lev]
            out[lev] = out[p] + mic(out[lev] - out[p], box)
        return out


# --------------------------------------------------------------------------- nanoparticle
def nanoparticle_structure(top, cfg, r_eq, outdir):
    s, g = top.s, top.glass
    gp, el = s.pos[g], top.elem[g]
    r = np.linalg.norm(gp, axis=1)
    surface = r > r_eq - cfg["surface_shell"]
    tree = cKDTree(gp)
    res = {"R_equivalent_A": r_eq, "R_max_A": float(r.max()),
           "surface_shell_A": cfg["surface_shell"]}

    # first-shell distances and coordination numbers inside the glass
    shells = {}
    o_idx = np.flatnonzero(el == "O")
    o_tree = cKDTree(gp[o_idx])
    for e, cut in (("Si", cfg["former_cutoff"]), ("P", cfg["former_cutoff"]),
                   ("Na", cfg["modifier_cutoff"]), ("Ca", cfg["modifier_cutoff"])):
        sel = np.flatnonzero(el == e)
        core = sel[~surface[sel]]
        d = [np.linalg.norm(gp[o_idx[nb]] - gp[i], axis=1) for i in core
             for nb in [o_tree.query_ball_point(gp[i], cut)]]
        cn = np.array([len(x) for x in d])
        dist = np.concatenate(d) if d else np.array([])
        shells[f"{e}-O"] = {"mean_distance_A": round(float(dist.mean()), 4),
                            "coordination_number_core": round(float(cn.mean()), 3),
                            "cutoff_A": cut}
    res["first_shell"] = shells

    # Q^n speciation of Si (bridging O = O bonded to two network formers)
    bo_mask = np.zeros(s.n, bool)
    bo_mask[top.bo] = True
    qn = {"core": np.zeros(5), "surface": np.zeros(5)}
    si = np.flatnonzero(el == "Si")
    angles_osio, angles_siosi = [], []
    for i in si:
        nb = o_idx[o_tree.query_ball_point(gp[i], cfg["former_cutoff"])]
        n_bo = int(np.sum(bo_mask[g[nb]]))
        qn["surface" if surface[i] else "core"][min(n_bo, 4)] += 1
        v = gp[nb] - gp[i]
        v /= np.linalg.norm(v, axis=1)[:, None]
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                angles_osio.append(np.degrees(np.arccos(np.clip(v[a] @ v[b], -1, 1))))
    formers = np.flatnonzero(np.isin(el, ["Si", "P"]))
    for o in np.flatnonzero(np.isin(g, top.bo)):
        nb = formers[np.linalg.norm(gp[formers] - gp[o], axis=1) < cfg["former_cutoff"]]
        nb = nb[el[nb] == "Si"]
        if len(nb) == 2:
            v1, v2 = gp[nb[0]] - gp[o], gp[nb[1]] - gp[o]
            angles_siosi.append(np.degrees(np.arccos(np.clip(
                v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
    res["Qn_percent"] = {k: {f"Q{n}": round(100 * float(v[n] / max(1, v.sum())), 2) for n in range(5)}
                         for k, v in qn.items()}
    res["Qn_percent"]["all"] = {f"Q{n}": round(100 * float((qn["core"][n] + qn["surface"][n])
                                                          / max(1, qn["core"].sum() + qn["surface"].sum())), 2)
                                for n in range(5)}
    res["network_connectivity"] = {k: round(float(np.dot(np.arange(5), v) / max(1, v.sum())), 3)
                                   for k, v in qn.items()}
    res["O_Si_O_angle_deg"] = {"mean": round(float(np.mean(angles_osio)), 2),
                               "std": round(float(np.std(angles_osio)), 2)}
    res["Si_O_Si_angle_deg"] = {"mean": round(float(np.mean(angles_siosi)), 2),
                                "std": round(float(np.std(angles_siosi)), 2)}

    # surface vs core composition and non-bridging oxygen
    comp = {}
    for region, mask in (("core", ~surface), ("surface", surface)):
        c = {e: int(np.sum(el[mask] == e)) for e in ("Si", "P", "Na", "Ca", "O")}
        tot = sum(c.values())
        comp[region] = {e: round(100 * v / tot, 2) for e, v in c.items()}
        o_m = mask & (el == "O")
        comp[region]["NBO_fraction_of_O"] = round(float(np.isin(g[o_m], top.nbo).mean()), 4)
    res["composition_atom_percent"] = comp
    area = 4 * np.pi * r_eq ** 2 / 100.0                     # nm^2
    surf_nbo = int(np.sum(np.isin(g[surface], top.nbo)))
    res["surface_density_per_nm2"] = {
        "NBO": round(surf_nbo / area, 3),
        "Na": round(int(np.sum(el[surface] == "Na")) / area, 3),
        "Ca": round(int(np.sum(el[surface] == "Ca")) / area, 3)}

    h1, e1 = np.histogram(angles_siosi, bins=np.arange(100, 181, 2), density=True)
    h2, e2 = np.histogram(angles_osio, bins=np.arange(80, 141, 1), density=True)
    write_csv(outdir / "data" / "np_Si_O_Si.csv", ["Si_O_Si_deg", "P"], [0.5 * (e1[1:] + e1[:-1]), h1])
    write_csv(outdir / "data" / "np_O_Si_O.csv", ["O_Si_O_deg", "P"], [0.5 * (e2[1:] + e2[:-1]), h2])
    return res, (qn, angles_siosi, angles_osio, comp)


# --------------------------------------------------------------------------- interface energy
KCAL_MOL_A2_TO_MJ_M2 = 4184.0 / 6.02214076e23 / 1e-20 * 1e3


def interface_energy(system, top, frames, cfg_all, r_eq, n_blocks):
    """Polymer-glass interaction energy E_int = E(composite) - E(polymer alone) - E(glass alone)
    per frame (the rigid glass has no internal energy), split into Lennard-Jones and Coulomb."""
    from .gpu_engine import Engine, cp
    f = cfg_all["forcefield"]
    kw = dict(cutoff=f["cutoff"], skin=f["skin"], dsf_alpha=f["dsf_alpha"], lj_tail=False,
              precision="double", cross_epsilon_scale=f.get("cross_epsilon_scale", 1.0))
    off = len(top.glass)
    s = system
    poly = System(s.box, s.pos[off:], s.type[off:], s.mol[off:], s.mobile[off:], s.bonds - off,
                  s.angles - off, s.dihedrals - off, s.impropers - off, s.cmaps - off)
    full, alone = Engine(s, **kw), Engine(poly, **kw)
    rows = []
    for box, wpos, _, _ in frames:
        unw = top.unwrap(wpos, box)
        res = []
        for eng, pos in ((full, unw), (alone, unw[off:])):
            eng.pos, eng.box = cp.asarray(pos), box
            eng.detach_nlist()
            eng.compute()
            res.append((eng.terms["lj"], eng.terms["coul"]))
        rows.append([res[0][0] - res[1][0], res[0][1] - res[1][1]])
    rows = np.array(rows)
    total = rows.sum(1)
    area = 4 * np.pi * r_eq ** 2
    return {"note": "E(composite) - E(PCL alone), same coordinates; negative = attractive",
            "E_int_kcal_mol": ms(total, n_blocks, 2),
            "E_int_LJ_kcal_mol": ms(rows[:, 0], n_blocks, 2),
            "E_int_Coulomb_kcal_mol": ms(rows[:, 1], n_blocks, 2),
            "E_int_per_area_mJ_m2": ms(total / area * KCAL_MOL_A2_TO_MJ_M2, n_blocks, 2),
            "area_A2 (4 pi R_eq^2)": round(float(area), 1)}


# --------------------------------------------------------------------------- main analysis
def shell_volumes(box, edges, spacing=0.3):
    """Volume of each spherical shell that lies inside the cubic cell (the cell is the
    Wigner-Seitz cell of the particle lattice, so |r| is the distance to the nearest particle)."""
    x = (np.arange(int(box / spacing)) + 0.5) * spacing - box / 2
    counts = np.zeros(len(edges) - 1)
    yy, zz = np.meshgrid(x, x, indexing="ij")
    for xi in x:
        r = np.sqrt(xi * xi + yy * yy + zz * zz).ravel()
        counts += np.histogram(r, edges)[0]
    return counts * (box / len(x)) ** 3


def analyse_neat(cfg, system, top, mc_dir, outdir):
    """Reference analysis of the neat-PCL control: the bulk quantities of analyse_structure."""
    nb = int(cfg["n_blocks"])
    frames = list(read_frames(mc_dir / "trajectory.xyz", system.n, set(cfg["stages"])))
    if not frames:
        raise RuntimeError(f"no trajectory frames from stages {cfg['stages']}")
    print(f"  neat PCL control: {len(frames)} frames from stages {cfg['stages']}")
    acc_set = set(top.poly_acceptors.tolist())
    cos_cut = np.cos(np.radians(cfg["hbond_angle"]))
    rg_rows, hb, dens, dih = [], [], [], []
    for box, wpos, _, _ in frames:
        dens.append(system.mass.sum() / box ** 3 * ff.AMU_A3_TO_GCC)
        unw = top.unwrap(wpos, box)
        rg_f = []
        for ch in top.chains:
            x, m = unw[ch["idx"]], system.mass[ch["idx"]]
            d = x - np.average(x, axis=0, weights=m)
            rg2 = np.average((d ** 2).sum(1), weights=m)
            ree = np.linalg.norm(unw[ch["ends"][1]] - unw[ch["ends"][0]])
            bb = np.diff(unw[ch["bb"]], axis=1).reshape(-1, 3)
            rg_f.append([np.sqrt(rg2), ree, ree ** 2 / (7 * ch["n"] * (bb ** 2).sum(1).mean())])
            dih.append(backbone_torsions(unw, ch))
        rg_f = np.array(rg_f)
        rg_rows.append([rg_f[:, 0].mean(), np.sqrt((rg_f[:, 0] ** 2).mean()), rg_f[:, 1].mean(),
                        np.sqrt((rg_f[:, 1] ** 2).mean()),
                        (rg_f[:, 1] ** 2).mean() / (rg_f[:, 0] ** 2).mean(), rg_f[:, 2].mean()])
        wb = wrap(wpos, box)
        tree = cKDTree(wb, boxsize=box)
        n_hb = 0
        for h, o in zip(top.donor_h, top.donor_o):
            for a_ in tree.query_ball_point(wb[h], cfg["hbond_distance"]):
                if a_ in acc_set and a_ != o:
                    v1, v2 = mic(wpos[o] - wpos[h], box), mic(wpos[a_] - wpos[h], box)
                    n_hb += v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2) <= cos_cut
        hb.append(n_hb)
    rg, dih = np.array(rg_rows), np.concatenate(dih)
    # free volume on a subset of frames
    probes = [float(p) for p in cfg["probes"]]
    fv = []
    for box, wpos, _, _ in frames[::max(1, len(frames) // int(cfg["free_volume_frames"]))][:int(cfg["free_volume_frames"])]:
        n_g = int(box / cfg["free_volume_spacing"])
        x = (np.arange(n_g) + 0.5) * box / n_g
        grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
        wb = wrap(wpos, box)
        clearance = np.full(len(grid), np.inf)
        for e, rad in VDW.items():
            sel = top.elem == e
            if sel.any():
                np.minimum(clearance, cKDTree(wb[sel], boxsize=box).query(grid, workers=-1)[0] - rad,
                           out=clearance)
        fv.append([np.mean(clearance > p) for p in probes])
    summary = {
        "frames": {"stages": cfg["stages"], "n_frames": len(frames), "n_blocks": nb},
        "neat_PCL": {"density_gcc": ms(dens, nb), "box_A": ms([f[0] for f in frames], nb, 3)},
        "hydrogen_bonds_per_frame": {"PCL OH -> PCL O": ms(hb, nb, 3),
                                     "PCL_hydroxyl_donors": int(len(top.donor_h))},
        "chain_conformation": {
            "Rg_mean_A": ms(rg[:, 0], nb, 3), "Rg_rms_A": ms(rg[:, 1], nb, 3),
            "Ree_mean_A": ms(rg[:, 2], nb, 3), "Ree_rms_A": ms(rg[:, 3], nb, 3),
            "Ree2_over_Rg2": ms(rg[:, 4], nb, 3),
            "characteristic_ratio_C_N (backbone bonds)": ms(rg[:, 5], nb, 3)},
        "backbone_torsions": {k: {"trans_fraction_|x|>150": round(float(np.mean(
                                  np.abs(dih[np.isfinite(dih[:, c]), c]) > 150)), 4)}
                              for k, c in (("ester", 0), ("tau_oc", 1), ("tau_cc", 2))},
        "free_volume": {"probes_A": probes, "fraction": [round(float(v), 4) for v in np.mean(fv, 0)],
                        "n_frames": len(fv)},
    }
    # the same quantities under the keys of the composite summary, for compare_runs.py
    summary["interphase"] = {"PCL_bulk_density_gcc (r > R_eq + 10 A)": summary["neat_PCL"]["density_gcc"]["mean"]}
    summary["free_volume"]["fraction_bulk_region (r > R_eq + 10 A)"] = summary["free_volume"]["fraction"]
    summary["hydrogen_bonds_per_frame"]["PCL OH -> PCL O"] = summary["hydrogen_bonds_per_frame"]["PCL OH -> PCL O"]
    summary["backbone_torsions"] = {k: {"bulk": v} for k, v in summary["backbone_torsions"].items()}
    (outdir / "structure_summary.json").write_text(json.dumps(summary, indent=2))
    tb = np.arange(-180, 181, 5)
    write_csv(outdir / "data" / "torsions.csv", ["angle_deg", "ester_bulk", "tau_oc_bulk", "tau_cc_bulk"],
              [0.5 * (tb[1:] + tb[:-1])] + [np.histogram(dih[np.isfinite(dih[:, c]), c], tb, density=True)[0]
                                            for c in range(3)])
    L = ["# Neat PCL control", "", f"{len(frames)} frames of stage(s) {', '.join(cfg['stages'])}.", "",
         "| quantity | value |", "|---|---|",
         f"| density | {summary['neat_PCL']['density_gcc']['mean']} ± {summary['neat_PCL']['density_gcc']['sem']} g/cm³ |"]
    for k, x in summary["chain_conformation"].items():
        L.append(f"| {k} | {x['mean']} ± {x['sem']} |")
    L.append(f"| H-bonds PCL OH -> PCL O | {summary['hydrogen_bonds_per_frame']['PCL OH -> PCL O']['mean']} |")
    for k, x in summary["backbone_torsions"].items():
        L.append(f"| {k} trans fraction | {x['bulk']['trans_fraction_|x|>150']} |")
    for p_, x in zip(probes, summary["free_volume"]["fraction"]):
        L.append(f"| free-volume fraction, probe {p_} Å | {x} |")
    (outdir / "structure_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  neat PCL: density {summary['neat_PCL']['density_gcc']['mean']} g/cm3, "
          f"Rg {summary['chain_conformation']['Rg_mean_A']['mean']} A")
    return summary


def analyse_structure(cfg_all, mc_dir, np_dir, outdir):
    # shells near the cell corners are empty in some frames; their NaN averages are expected
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    cfg = dict(DEFAULTS)
    cfg.update(cfg_all.get("structure", {}))
    mc_dir, outdir = Path(mc_dir), Path(outdir)
    (outdir / "data").mkdir(parents=True, exist_ok=True)
    nb = int(cfg["n_blocks"])
    system = System.load(mc_dir / "system_final.npz")
    top = Topology(system, cfg)
    if not system.has_particle:
        return analyse_neat(cfg, system, top, mc_dir, outdir)
    np_report = json.loads((Path(np_dir) / "nanoparticle.json").read_text())
    r_eq = float(np_report["radius_equivalent"])
    s = system
    mass = s.mass
    names, elem = top.names, top.elem
    poly, glass = top.poly, top.glass

    print("  nanoparticle structure (rigid)...")
    np_res, np_extra = nanoparticle_structure(top, cfg, r_eq, outdir)

    # ---------------------------------------------------------------- frame loop
    groups = {"PCL (all atoms)": poly,
              "carbonyl O": np.flatnonzero(np.isin(names, ["O_carbonyl", "O_acid"])),
              "ester O": np.flatnonzero(names == "O_ester"),
              "alkyl CH2": np.flatnonzero(names == "CH2"),
              "chain-end O": np.flatnonzero(np.isin(names, ["OH_acid", "OH_end"]))}
    rdf_pairs = {"Na-O(carbonyl)": (glass[elem[glass] == "Na"], groups["carbonyl O"]),
                 "Ca-O(carbonyl)": (glass[elem[glass] == "Ca"], groups["carbonyl O"]),
                 "Na-O(ester)": (glass[elem[glass] == "Na"], groups["ester O"]),
                 "Ca-O(ester)": (glass[elem[glass] == "Ca"], groups["ester O"]),
                 "NBO-H(hydroxyl)": (top.nbo, top.donor_h),
                 "NBO-H(C-H)": (top.nbo, np.flatnonzero(np.isin(names, ["HC", "HC_a", "HC_oh"])))}
    if len(top.silanol_h):
        rdf_pairs["H(silanol)-O(carbonyl)"] = (top.silanol_h, groups["carbonyl O"])
        rdf_pairs["H(silanol)-O(ester)"] = (top.silanol_h, groups["ester O"])
    rdf_edges = np.arange(0.0, cfg["rdf_rmax"] + 1e-9, 0.05)
    frames = list(read_frames(mc_dir / "trajectory.xyz", s.n, set(cfg["stages"])))
    if not frames:
        raise RuntimeError(f"no trajectory frames from stages {cfg['stages']}")
    print(f"  {len(frames)} frames from stages {cfg['stages']}")
    box_mean = float(np.mean([f[0] for f in frames]))
    dr = cfg["dr"]
    edges = np.arange(0.0, np.sqrt(3) * box_mean / 2 + dr, dr)
    centres = 0.5 * (edges[1:] + edges[:-1])

    per = {k: [] for k in ("rho_glass", "rho_pcl", "groups", "chain_ends", "mid_monomers",
                           "rdf", "hb", "cation", "ads", "rg", "rg_r", "orient", "dih", "box")}
    trains_all, loops_all, tails_all = [], [], []
    chain_rows = []
    for fi, (box, wpos, stage, move) in enumerate(frames):
        per["box"].append(box)
        vol = shell_volumes(box, edges) if fi == 0 or abs(box - frames[fi - 1][0]) > 1e-3 else vol
        w = wpos - box * np.floor(wpos / box + 0.5)              # cell centred on the particle
        r = np.linalg.norm(w, axis=1)
        unw = top.unwrap(wpos, box)

        # radial densities (g/cm3) and group number densities
        with np.errstate(invalid="ignore", divide="ignore"):
            per["rho_glass"].append(np.histogram(r[glass], edges, weights=mass[glass])[0]
                                    / vol * ff.AMU_A3_TO_GCC)
            per["rho_pcl"].append(np.histogram(r[poly], edges, weights=mass[poly])[0]
                                  / vol * ff.AMU_A3_TO_GCC)
            per["groups"].append([np.histogram(r[idx], edges)[0] / vol for idx in groups.values()])

        # radial distribution functions (glass species -> polymer species)
        V = box ** 3
        wb = wrap(wpos, box)
        rdfs = []
        for a, b in rdf_pairs.values():
            ta, tb = cKDTree(wb[a], boxsize=box), cKDTree(wb[b], boxsize=box)
            d = ta.sparse_distance_matrix(tb, cfg["rdf_rmax"], output_type="ndarray")["v"]
            h = np.histogram(d, rdf_edges)[0]
            shell = 4 / 3 * np.pi * np.diff(rdf_edges ** 3)
            rdfs.append(h / (len(a) * len(b) / V * shell))
        per["rdf"].append(rdfs)

        # hydrogen bonds: hydroxyl H donors -> glass O (NBO / BO) or polymer O
        t_all = cKDTree(wb, boxsize=box)
        counts = dict.fromkeys(HB_KINDS, 0)
        nbo_set, bo_set = set(top.nbo.tolist()), set(top.bo.tolist())
        sil_set = set(top.silanol_o.tolist())
        acc_set = set(top.poly_acceptors.tolist())
        cos_cut = np.cos(np.radians(cfg["hbond_angle"]))
        pairs = [(h, o, "PCL OH") for h, o in zip(top.donor_h, top.donor_o)] + \
                [(h, o, "Si-OH") for h, o in zip(top.silanol_h, top.silanol_o)]
        for h, o, donor in pairs:
            for a in t_all.query_ball_point(wb[h], cfg["hbond_distance"]):
                if a == o or a == h:
                    continue
                if donor == "PCL OH":
                    kind = ("PCL OH -> NBO" if a in nbo_set else "PCL OH -> BO" if a in bo_set
                            else "PCL OH -> Si-OH" if a in sil_set
                            else "PCL OH -> PCL O" if a in acc_set else None)
                else:
                    kind = "Si-OH -> PCL O" if a in acc_set else None
                if kind is None:
                    continue
                v1, v2 = mic(wpos[o] - wpos[h], box), mic(wpos[a] - wpos[h], box)
                if v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2) <= cos_cut:
                    counts[kind] += 1
        per["hb"].append([counts[k] for k in HB_KINDS])

        # surface cations coordinated by polymer oxygen (within the contact distance)
        t_po = cKDTree(wb[top.poly_acceptors], boxsize=box)
        cat = []
        for e in ("Na", "Ca"):
            idx = glass[elem[glass] == e]
            cnt = np.array([len(x) for x in t_po.query_ball_point(wb[idx], cfg["contact"])])
            cat += [int(np.sum(cnt > 0)), float(cnt[cnt > 0].mean()) if np.any(cnt > 0) else 0.0]
        per["cation"].append(cat)

        # adsorption: monomer in contact if any of its atoms lies within `contact` of the glass
        t_g = cKDTree(wb[glass], boxsize=box)
        dmin = t_g.query(wb[poly], distance_upper_bound=cfg["contact"])[0]
        touch = np.zeros(top.n_monomers, bool)
        touch[top.unit_id[poly[np.isfinite(dmin)]]] = True
        n_ads_chains, ads_mass = 0, 0.0
        tr_f, lo_f, ta_f = 0, 0, 0
        for ch in top.chains:
            t = touch[ch["first_monomer"]:ch["first_monomer"] + ch["n"]]
            if not t.any():
                continue
            n_ads_chains += 1
            ads_mass += t.sum()
            on = np.flatnonzero(t)
            first, last = on[0], on[-1]
            if first > 0:
                tails_all.append(first)
            if last < ch["n"] - 1:
                tails_all.append(ch["n"] - 1 - last)
            ta_f += first + (ch["n"] - 1 - last)
            run, state = 0, t[first]
            for v in t[first:last + 1]:
                if v == state:
                    run += 1
                    continue
                (trains_all if state else loops_all).append(run)
                state, run = v, 1
            (trains_all if state else loops_all).append(run)
            tr_f += int(t.sum())
            lo_f += int((last - first + 1) - t.sum())
        n_tot = top.n_monomers
        area_nm2 = 4 * np.pi * r_eq ** 2 / 100.0
        gamma = ads_mass * 72.0627 / 6.02214e23 * 1e3 / (area_nm2 * 1e-18)   # mg/m^2
        per["ads"].append([n_ads_chains / len(top.chains), touch.sum() / n_tot,
                           tr_f / n_tot, lo_f / n_tot, ta_f / n_tot, gamma])

        # chain conformation and its dependence on the distance from the particle
        rg_f, rgr = [], []
        for ci, ch in enumerate(top.chains):
            x = unw[ch["idx"]]
            m = mass[ch["idx"]]
            com = np.average(x, axis=0, weights=m)
            d = x - com
            rg2 = np.average((d ** 2).sum(1), weights=m)
            cw = mic(com, box)
            dist = np.linalg.norm(cw)
            n_hat = cw / dist
            rad2 = np.average((d @ n_hat) ** 2, weights=m)
            ree = np.linalg.norm(unw[ch["ends"][1]] - unw[ch["ends"][0]])
            bb = np.diff(unw[ch["bb"]], axis=1).reshape(-1, 3)
            l2 = (bb ** 2).sum(1).mean()
            rg_f.append([np.sqrt(rg2), ree, ree ** 2 / (7 * ch["n"] * l2)])
            rgr.append([dist - r_eq, np.sqrt(rg2), rad2 / ((rg2 - rad2) / 2)])
            chain_rows.append([fi, ci, dist - r_eq, np.sqrt(rg2), ree, rad2 / ((rg2 - rad2) / 2)])
        rg_f = np.array(rg_f)
        per["rg"].append([rg_f[:, 0].mean(), np.sqrt((rg_f[:, 0] ** 2).mean()), rg_f[:, 1].mean(),
                          np.sqrt((rg_f[:, 1] ** 2).mean()),
                          (rg_f[:, 1] ** 2).mean() / (rg_f[:, 0] ** 2).mean(), rg_f[:, 2].mean()])
        per["rg_r"].append(np.array(rgr))

        # orientation relative to the radial direction, binned by distance from the surface
        orient = []
        for ch in top.chains:
            o, on_, c, oc = ch["O"], ch["O_next"], ch["C"], ch["OC"]
            mid = 0.5 * (unw[o] + unw[on_])
            mw = mic(mid, box)
            rr = np.linalg.norm(mw, axis=1)
            nh = mw / rr[:, None]
            chord = unw[on_] - unw[o]
            chord /= np.linalg.norm(chord, axis=1)[:, None]
            co = unw[oc] - unw[c]
            co /= np.linalg.norm(co, axis=1)[:, None]
            orient.append(np.c_[rr - r_eq, np.einsum("ij,ij->i", chord, nh),
                                np.einsum("ij,ij->i", co, nh)])
        per["orient"].append(np.concatenate(orient))

        # backbone torsions per monomer, tagged with the distance of the monomer from the surface
        dih = []
        for ch in top.chains:
            tors = backbone_torsions(unw, ch)
            phi, psi, omg = tors[:, 0], tors[:, 1], tors[:, 2]
            rr = np.linalg.norm(mic(unw[ch["CA"]], box), axis=1) - r_eq
            dih.append(np.c_[rr, phi, psi, omg])
        per["dih"].append(np.concatenate(dih))
        print(f"    frame {fi + 1}/{len(frames)} ({stage} {move})", end="\r")
    print()

    print("  polymer-glass interaction energy...")
    interaction = interface_energy(s, top, frames, cfg_all, r_eq, nb)

    # ---------------------------------------------------------------- averages
    rho_g = np.nanmean(per["rho_glass"], axis=0)
    rho_p_frames = np.array(per["rho_pcl"])
    rho_p, rho_p_err = block_stats(rho_p_frames, nb)
    # shells beyond half the box lie partly outside the cell; keep those still >= 20 % inside
    in_cell = shell_volumes(box_mean, edges) / (4 / 3 * np.pi * np.diff(edges ** 3))
    valid = np.isfinite(rho_p) & (in_cell >= 0.2)
    centres_v = centres[valid]
    bulk_mask = valid & (centres > r_eq + 10.0)
    rho_bulk = float(np.nanmean(rho_p[bulk_mask])) if bulk_mask.any() else float("nan")
    grp = np.array(per["groups"])                    # frames, groups, bins
    grp_mean = np.nanmean(grp, axis=0)
    grp_bulk = np.nanmean(grp_mean[:, bulk_mask], axis=1)

    # interphase: hyperbolic-tangent fit of the polymer density onset and the 5 % recovery point
    def tanh_profile(x, r0, w):
        return 0.5 * rho_bulk * (1 + np.tanh((x - r0) / w))
    fit_mask = valid & (centres > r_eq - 5) & (centres < r_eq + 12)
    (r0, wdt), _ = curve_fit(tanh_profile, centres[fit_mask], rho_p[fit_mask], p0=(r_eq + 1.5, 1.0))
    n_s = max(1, int(round(2.0 / dr)))                    # 2 A running mean
    smooth = np.convolve(np.where(valid, rho_p, np.nan), np.ones(n_s) / n_s, mode="same")
    peak_region = valid & (centres > r0) & (centres < r0 + 6)
    i_peak = np.flatnonzero(peak_region)[np.nanargmax(rho_p[peak_region])]
    # interphase ends where the smoothed density first stays within 5 % of bulk for 3 A
    ok = np.abs(smooth / rho_bulk - 1.0) < 0.05
    run = int(round(3.0 / dr))
    r_recover = float("nan")
    for i in range(i_peak, len(centres) - run):
        if valid[i:i + run].all() and ok[i:i + run].all():
            r_recover = float(centres[i])
            break

    rdf = np.nanmean(per["rdf"], axis=0)
    rdf_c = 0.5 * (rdf_edges[1:] + rdf_edges[:-1])
    rdf_res = {}
    for k, (key, (a, b)) in enumerate(rdf_pairs.items()):
        g = rdf[k]
        near = rdf_c < 4.0                                 # first-shell (contact) peak only
        ip = int(np.argmax(np.where(near, g, -1)))
        if g[ip] < 0.5:
            rdf_res[key] = {"first_peak_A": None, "first_minimum_A": None,
                            "coordination_per_glass_atom": None,
                            "note": f"no contact peak below 4 A (max g = {g[ip]:.2f})"}
            continue
        after = np.flatnonzero((rdf_c > rdf_c[ip]) & (rdf_c < rdf_c[ip] + 2.0))
        imin = after[np.argmin(g[after])] if len(after) else ip
        rho_b = len(b) / box_mean ** 3
        n_r = np.cumsum(g * rho_b * 4 * np.pi * rdf_c ** 2 * 0.05)
        rdf_res[key] = {"first_peak_A": round(float(rdf_c[ip]), 3),
                        "first_peak_g": round(float(g[ip]), 3),
                        "first_minimum_A": round(float(rdf_c[imin]), 3),
                        "coordination_per_glass_atom": round(float(n_r[imin]), 4)}

    hb = np.array(per["hb"], float)
    cat = np.array(per["cation"], float)
    ads = np.array(per["ads"], float)
    rg = np.array(per["rg"], float)
    rgr = np.concatenate(per["rg_r"])
    orient = np.concatenate(per["orient"])
    dih = np.concatenate(per["dih"])

    # binned profiles vs distance from the particle surface
    def binned(x, y, bins):
        idx = np.digitize(x, bins) - 1
        out = np.full((len(bins) - 1, 3), np.nan)
        for k in range(len(bins) - 1):
            v = y[idx == k]
            v = v[np.isfinite(v)]
            if len(v) >= 20:
                out[k] = [v.mean(), v.std() / np.sqrt(len(v)), len(v)]
        return out
    sbins = np.arange(-2.0, np.sqrt(3) * box_mean / 2 - r_eq, 1.0)
    p2_chord = binned(orient[:, 0], 1.5 * orient[:, 1] ** 2 - 0.5, sbins)
    p2_co = binned(orient[:, 0], 1.5 * orient[:, 2] ** 2 - 0.5, sbins)
    cos_co = binned(orient[:, 0], orient[:, 2], sbins)
    cbins = np.arange(0.0, np.sqrt(3) * box_mean / 2 - r_eq, 3.0)
    rg_vs = binned(rgr[:, 0], rgr[:, 1], cbins)
    an_vs = binned(rgr[:, 0], rgr[:, 2], cbins)

    interphase_cut = r_recover - r_eq
    k1 = int(np.clip(np.digitize(centres[i_peak] - r_eq, sbins) - 1, 0, len(sbins) - 2))   # first-layer bin
    inter = dih[:, 0] < interphase_cut
    tors = {}
    for name, col in (("ester", 1), ("tau_oc", 2), ("tau_cc", 3)):
        for region, mask in (("interphase", inter), ("bulk", ~inter)):
            v = dih[mask, col]
            v = v[np.isfinite(v)]
            tors.setdefault(name, {})[region] = {
                "trans_fraction_|x|>150": round(float(np.mean(np.abs(v) > 150)), 4),
                "n": int(len(v))}
    tors["ester"]["mean_abs_deviation_from_180"] = round(float(np.nanmean(180 - np.abs(dih[:, 1]))), 2)

    # free volume (grid), on a subset of frames
    print("  free volume...")
    fv_frames = frames[::max(1, len(frames) // int(cfg["free_volume_frames"]))][:int(cfg["free_volume_frames"])]
    probes = [float(p) for p in cfg["probes"]]
    fv_total, fv_bulk, fv_profile = [], [], []
    fv_bins = np.arange(r_eq - 2, np.sqrt(3) * box_mean / 2, 1.0)
    for box, wpos, _, _ in fv_frames:
        h = cfg["free_volume_spacing"]
        n_g = int(box / h)
        x = (np.arange(n_g) + 0.5) * box / n_g
        grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
        wb = wrap(wpos, box)
        clearance = np.full(len(grid), np.inf)
        for e, rad in VDW.items():
            sel = elem == e
            if not sel.any():
                continue
            d = cKDTree(wb[sel], boxsize=box).query(grid, workers=-1)[0] - rad
            np.minimum(clearance, d, out=clearance)
        gr = np.linalg.norm(grid - box / 2, axis=1)
        in_poly = gr > r_eq + 2.0          # outside the particle and its first contact layer
        fv_total.append([np.mean(clearance[in_poly] > p) for p in probes])
        bulk_pts = gr > r_eq + 10.0
        fv_bulk.append([np.mean(clearance[bulk_pts] > p) for p in probes])
        idx = np.digitize(gr, fv_bins) - 1
        prof = []
        for p in probes:
            free = clearance > p
            prof.append([free[idx == k].mean() if np.any(idx == k) else np.nan
                         for k in range(len(fv_bins) - 1)])
        fv_profile.append(prof)
    fv_profile = np.nanmean(fv_profile, axis=0)

    # ---------------------------------------------------------------- summary
    dens_frames = np.array([s.mass.sum() / b ** 3 * ff.AMU_A3_TO_GCC for b in per["box"]])
    summary = {
        "frames": {"stages": cfg["stages"], "n_frames": len(frames), "n_blocks": nb},
        "composite": {"density_gcc": ms(dens_frames, nb), "box_A": ms(per["box"], nb, 3),
                      "glass_weight_percent": round(100 * s.glass_weight_fraction, 3),
                      "glass_volume_percent": round(100 * np_report["volume_A3"] / box_mean ** 3, 3)},
        "nanoparticle": np_res,
        "interphase": {
            "PCL_bulk_density_gcc (r > R_eq + 10 A)": round(rho_bulk, 4),
            "tanh_fit_r0_A": round(float(r0), 3), "tanh_fit_width_A": round(float(abs(wdt)), 3),
            "polymer_onset_from_R_eq_A": round(float(r0 - r_eq), 3),
            "first_density_peak_r_A": round(float(centres[i_peak]), 3),
            "first_density_peak_rho_gcc": round(float(rho_p[i_peak]), 4),
            "first_peak_relative_to_bulk": round(float(rho_p[i_peak] / rho_bulk), 4),
            "density_within_5pct_of_bulk_from_r_A (for >= 3 A)": round(r_recover, 3),
            "interphase_thickness_A (R_eq to 5 % recovery)": round(r_recover - r_eq, 3),
            "group_enrichment_first_layer": {
                k: round(float(np.nanmax(grp_mean[i, peak_region]) / grp_bulk[i]), 3)
                for i, k in enumerate(groups) if grp_bulk[i] > 0},
        },
        "rdf_glass_polymer": rdf_res,
        "hydrogen_bonds_per_frame": {
            "criterion": f"H...A <= {cfg['hbond_distance']} A, D-H...A >= {cfg['hbond_angle']} deg",
            **{k: ms(hb[:, i], nb, 3) for i, k in enumerate(HB_KINDS)},
            "PCL_hydroxyl_donors": int(len(top.donor_h)), "silanol_donors": int(len(top.silanol_h))},
        "interaction_energy": interaction,
        "cation_coordination_by_PCL_O": {
            "cutoff_A": cfg["contact"],
            "Na_with_PCL_O": ms(cat[:, 0], nb, 2), "Na_mean_PCL_O_per_contact_cation": ms(cat[:, 1], nb, 3),
            "Ca_with_PCL_O": ms(cat[:, 2], nb, 2), "Ca_mean_PCL_O_per_contact_cation": ms(cat[:, 3], nb, 3)},
        "adsorption": {
            "contact_criterion_A": cfg["contact"],
            "adsorbed_chain_fraction": ms(ads[:, 0], nb), "monomers_in_contact_fraction": ms(ads[:, 1], nb),
            "train_fraction": ms(ads[:, 2], nb), "loop_fraction": ms(ads[:, 3], nb),
            "tail_fraction": ms(ads[:, 4], nb),
            "adsorbed_amount_mg_per_m2": ms(ads[:, 5], nb, 3),
            "mean_train_length_monomers": round(float(np.mean(trains_all)), 3) if trains_all else None,
            "mean_loop_length_monomers": round(float(np.mean(loops_all)), 3) if loops_all else None,
            "mean_tail_length_monomers": round(float(np.mean(tails_all)), 3) if tails_all else None},
        "chain_conformation": {
            "units_per_chain": f"{min(c['n'] for c in top.chains)}-{max(c['n'] for c in top.chains)}",
            "Rg_mean_A": ms(rg[:, 0], nb, 3), "Rg_rms_A": ms(rg[:, 1], nb, 3),
            "Ree_mean_A": ms(rg[:, 2], nb, 3), "Ree_rms_A": ms(rg[:, 3], nb, 3),
            "Ree2_over_Rg2": ms(rg[:, 4], nb, 3),
            "characteristic_ratio_C_N (backbone bonds)": ms(rg[:, 5], nb, 3)},
        "orientation": {"note": "P2 = 0 isotropic, -0.5 tangential to the surface, 1 radial",
                        "P2_backbone_first_layer": round(float(p2_chord[k1, 0]), 4),
                        "P2_carbonyl_first_layer": round(float(p2_co[k1, 0]), 4),
                        "cos_carbonyl_first_layer": round(float(cos_co[k1, 0]), 4),
                        "P2_backbone_bulk (> 10 A)": round(float(np.nanmean(p2_chord[sbins[:-1] >= 10, 0])), 4),
                        "first_layer_bin_A": [float(sbins[k1]), float(sbins[k1 + 1])]},
        "backbone_torsions": tors,
        "free_volume": {
            "probes_A": probes, "vdW_radii": "Bondi",
            "fraction_polymer_region (r > R_eq + 2 A)": [round(float(v), 4) for v in np.mean(fv_total, 0)],
            "fraction_bulk_region (r > R_eq + 10 A)": [round(float(v), 4) for v in np.mean(fv_bulk, 0)],
            "n_frames": len(fv_frames)},
    }
    (outdir / "structure_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------- data files
    write_csv(outdir / "data" / "radial_density.csv",
              ["r_A", "rho_glass_gcc", "rho_PCL_gcc", "rho_PCL_sem"] + [f"n_{k}_rel_bulk" for k in groups],
              [centres_v, rho_g[valid], rho_p[valid], rho_p_err[valid]]
              + [grp_mean[i, valid] / grp_bulk[i] for i in range(len(groups))])
    write_csv(outdir / "data" / "rdf_glass_polymer.csv", ["r_A"] + list(rdf_pairs), [rdf_c, *rdf])
    sb = 0.5 * (sbins[1:] + sbins[:-1])
    write_csv(outdir / "data" / "orientation.csv",
              ["distance_from_R_eq_A", "P2_backbone", "sem", "P2_carbonyl", "sem", "cos_carbonyl", "sem"],
              [sb, p2_chord[:, 0], p2_chord[:, 1], p2_co[:, 0], p2_co[:, 1], cos_co[:, 0], cos_co[:, 1]])
    cb = 0.5 * (cbins[1:] + cbins[:-1])
    write_csv(outdir / "data" / "chain_size_vs_distance.csv",
              ["chain_COM_distance_from_R_eq_A", "Rg_A", "sem", "Rg2_radial_over_tangential", "sem", "n"],
              [cb, rg_vs[:, 0], rg_vs[:, 1], an_vs[:, 0], an_vs[:, 1], rg_vs[:, 2]])
    write_csv(outdir / "data" / "chains_per_frame.csv",
              ["frame", "chain", "COM_distance_from_R_eq_A", "Rg_A", "Ree_A", "Rg2_rad_over_tan"],
              list(zip(*chain_rows)))
    fb = 0.5 * (fv_bins[1:] + fv_bins[:-1])
    write_csv(outdir / "data" / "free_volume_profile.csv", ["r_A"] + [f"free_fraction_probe_{p}A" for p in probes],
              [fb, *fv_profile])
    tb = np.arange(-180, 181, 5)
    hist = {}
    for name, col in (("ester", 1), ("tau_oc", 2), ("tau_cc", 3)):
        for region, mask in (("interphase", inter), ("bulk", ~inter)):
            v = dih[mask, col]
            hist[f"{name}_{region}"] = np.histogram(v[np.isfinite(v)], tb, density=True)[0]
    write_csv(outdir / "data" / "torsions.csv", ["angle_deg"] + list(hist), [0.5 * (tb[1:] + tb[:-1]), *hist.values()])
    for name, data in (("train", trains_all), ("loop", loops_all), ("tail", tails_all)):
        if data:
            k = np.arange(1, max(data) + 1)
            write_csv(outdir / "data" / f"{name}_lengths.csv", ["length_monomers", "probability"],
                      [k, np.bincount(data, minlength=max(data) + 1)[1:] / len(data)])
    report(outdir, summary)
    return summary


# --------------------------------------------------------------------------- report
def report(outdir, s):
    def v(x, unit=""):
        if isinstance(x, dict) and "mean" in x:
            return f"{x['mean']} ± {x['sem']}{unit}"
        return f"{x}{unit}"
    L = ["# Structural analysis of the 45S5 / PCL composite", "",
         f"Averages over {s['frames']['n_frames']} frames of stage(s) {', '.join(s['frames']['stages'])}; "
         f"uncertainties are standard errors of {s['frames']['n_blocks']} block averages.", ""]
    c = s["composite"]
    L += ["## System", "", "| quantity | value |", "|---|---|",
          f"| composite density | {v(c['density_gcc'], ' g/cm³')} |", f"| box edge | {v(c['box_A'], ' Å')} |",
          f"| glass content | {c['glass_weight_percent']} wt% / {c['glass_volume_percent']} vol% |", ""]
    n = s["nanoparticle"]
    L += ["## Nanoparticle (rigid)", "", "| quantity | value |", "|---|---|",
          f"| R_eq / R_max | {n['R_equivalent_A']:.2f} / {n['R_max_A']:.2f} Å |"]
    for k, x in n["first_shell"].items():
        L.append(f"| {k} distance / CN (core) | {x['mean_distance_A']} Å / {x['coordination_number_core']} |")
    L += [f"| O–Si–O angle | {n['O_Si_O_angle_deg']['mean']} ± {n['O_Si_O_angle_deg']['std']}° |",
          f"| Si–O–Si angle | {n['Si_O_Si_angle_deg']['mean']} ± {n['Si_O_Si_angle_deg']['std']}° |",
          f"| network connectivity (core / surface) | {n['network_connectivity']['core']} / {n['network_connectivity']['surface']} |",
          f"| NBO fraction of O (core / surface) | {n['composition_atom_percent']['core']['NBO_fraction_of_O']} / {n['composition_atom_percent']['surface']['NBO_fraction_of_O']} |",
          f"| surface NBO / Na / Ca per nm² | {n['surface_density_per_nm2']['NBO']} / {n['surface_density_per_nm2']['Na']} / {n['surface_density_per_nm2']['Ca']} |",
          "", "| Qⁿ (%) | Q0 | Q1 | Q2 | Q3 | Q4 |", "|---|---|---|---|---|---|"]
    for region in ("core", "surface", "all"):
        q = n["Qn_percent"][region]
        L.append(f"| {region} | " + " | ".join(str(q[f'Q{i}']) for i in range(5)) + " |")
    i = s["interphase"]
    L += ["", "## Interphase", "", "| quantity | value |", "|---|---|"]
    for k, x in i.items():
        if k != "group_enrichment_first_layer":
            L.append(f"| {k} | {x} |")
    for k, x in i["group_enrichment_first_layer"].items():
        L.append(f"| first-layer enrichment, {k} | {x} |")
    L += ["", "## Glass–polymer pair correlations", "", "| pair | first peak (Å) | first minimum (Å) | coordination per glass atom |",
          "|---|---|---|---|"]
    for k, x in s["rdf_glass_polymer"].items():
        if x["first_peak_A"] is None:
            L.append(f"| {k} | {x['note']} | – | – |")
        else:
            L.append(f"| {k} | {x['first_peak_A']} (g = {x['first_peak_g']}) | {x['first_minimum_A']} | "
                     f"{x['coordination_per_glass_atom']} |")
    h, cc = s["hydrogen_bonds_per_frame"], s["cation_coordination_by_PCL_O"]
    L += ["", "## Hydrogen bonds and cation coordination", "", f"Criterion: {h['criterion']}.", "",
          "| quantity | value |", "|---|---|",
          *[f"| {k} | {v(h[k])} |" for k in HB_KINDS],
          f"| PCL hydroxyl donors / silanol donors | {h['PCL_hydroxyl_donors']} / {h['silanol_donors']} |",
          f"| Na with PCL O within {cc['cutoff_A']} Å | {v(cc['Na_with_PCL_O'])} |",
          f"| PCL O per contacting Na | {v(cc['Na_mean_PCL_O_per_contact_cation'])} |",
          f"| Ca with PCL O within {cc['cutoff_A']} Å | {v(cc['Ca_with_PCL_O'])} |",
          f"| PCL O per contacting Ca | {v(cc['Ca_mean_PCL_O_per_contact_cation'])} |"]
    ie = s["interaction_energy"]
    L += ["", "## Polymer–glass interaction energy", "", f"{ie['note']}.", "", "| quantity | value |", "|---|---|",
          f"| E_int | {v(ie['E_int_kcal_mol'], ' kcal/mol')} |", f"| LJ part | {v(ie['E_int_LJ_kcal_mol'], ' kcal/mol')} |",
          f"| Coulomb part | {v(ie['E_int_Coulomb_kcal_mol'], ' kcal/mol')} |",
          f"| per particle area | {v(ie['E_int_per_area_mJ_m2'], ' mJ/m²')} |"]
    a = s["adsorption"]
    L += ["", "## Adsorption", "", "| quantity | value |", "|---|---|"]
    for k, x in a.items():
        L.append(f"| {k} | {v(x)} |")
    g = s["chain_conformation"]
    L += ["", "## Chain conformation", "", "| quantity | value |", "|---|---|"]
    for k, x in g.items():
        L.append(f"| {k} | {v(x)} |")
    o = s["orientation"]
    L += ["", "## Orientation (first layer, " + f"{o['first_layer_bin_A'][0]} to {o['first_layer_bin_A'][1]} Å from R_eq)", "",
          f"{o['note']}.", "", "| quantity | value |", "|---|---|",
          f"| P2 backbone | {o['P2_backbone_first_layer']} |", f"| P2 carbonyl | {o['P2_carbonyl_first_layer']} |",
          f"| ⟨cos θ⟩ carbonyl | {o['cos_carbonyl_first_layer']} |",
          f"| P2 backbone, bulk | {o['P2_backbone_bulk (> 10 A)']} |"]
    t = s["backbone_torsions"]
    L += ["", "## Backbone torsions (fraction with |angle| > 150°)", "", "| torsion | interphase | bulk |", "|---|---|---|"]
    for k in ("ester", "tau_oc", "tau_cc"):
        L.append(f"| {k} | {t[k]['interphase']['trans_fraction_|x|>150']} | {t[k]['bulk']['trans_fraction_|x|>150']} |")
    f = s["free_volume"]
    L += ["", "## Free volume (Bondi radii)", "", "| probe (Å) | polymer region | bulk region |", "|---|---|---|"]
    for p, x, y in zip(f["probes_A"], f["fraction_polymer_region (r > R_eq + 2 A)"], f["fraction_bulk_region (r > R_eq + 10 A)"]):
        L.append(f"| {p} | {x} | {y} |")
    (outdir / "structure_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
