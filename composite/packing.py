"""Stage 2 - centre the nanoparticle in a periodic box and fill the rest with PCL.

The amount of polymer follows from the requested polymer : glass ratio.  Chains
are grown as clash-free random coils at a reduced density; the Monte Carlo
stage then compresses the cell to its equilibrium density.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import forcefield as ff
from . import polymer
from .io import System, read_xyz, write_system_xyz


def chain_lengths(polymer_mass, chain_length):
    """Split the target polymer mass into chains of (almost) equal length."""
    n_chains = max(1, round(polymer_mass / polymer.chain_mass(chain_length)))
    n_units = max(n_chains, round((polymer_mass - n_chains * polymer.END_MASS) / polymer.UNIT_MASS))
    lengths = np.full(n_chains, n_units // n_chains)
    lengths[:n_units % n_chains] += 1
    return lengths


def build_composite(cfg, np_dir, outdir):
    c = cfg["composite"]
    rng = np.random.default_rng(cfg["system"]["seed"] + 1
                                + 104729 * int(cfg["system"].get("replica", 1)))
    neat = bool(c.get("neat", False))
    report_np = json.loads((Path(np_dir) / "nanoparticle.json").read_text())
    g_sym, g_pos, _ = read_xyz(Path(np_dir) / "nanoparticle.xyz")
    g_type = np.array([ff.TYPE_INDEX[ff.GLASS_TYPE[s]] for s in g_sym], np.int32)
    glass_mass = ff.MASS[g_type].sum()
    glass_volume = report_np["volume_A3"]

    # ---- how much polymer: 70 : 30 (polymer : glass) by weight or by volume
    x_poly = float(c["polymer_fraction"])
    if c["basis"] == "weight":
        polymer_mass = glass_mass * x_poly / (1.0 - x_poly)
    elif c["basis"] == "volume":
        polymer_mass = (glass_volume * x_poly / (1.0 - x_poly)
                        * c["polymer_density"] / ff.AMU_A3_TO_GCC)
    else:
        raise ValueError("composite.basis must be 'weight' or 'volume'")
    lengths = chain_lengths(polymer_mass, int(c["chain_length"]))
    polymer_mass = sum(polymer.chain_mass(n) for n in lengths)
    if neat:      # control system: the same chains, without the particle
        g_sym, g_pos, g_type = g_sym[:0], g_pos[:0], g_type[:0]
        glass_mass, glass_volume = 0.0, 0.0

    def box_for(density):
        return (glass_volume + polymer_mass * ff.AMU_A3_TO_GCC / density) ** (1.0 / 3.0)
    box = box_for(c["build_density"])
    box_final = box_for(c["polymer_density"])
    r_np = np.linalg.norm(g_pos, axis=1).max() if len(g_pos) else 0.0
    cutoff = cfg["forcefield"]["cutoff"] + cfg["forcefield"]["skin"]
    if box_final < 2.0 * cutoff:
        raise ValueError(f"expected equilibrium box ({box_final:.1f} A) is smaller than twice the "
                         f"interaction range ({cutoff:.1f} A): use a larger nanoparticle radius")
    print(f"  glass  : {len(g_pos)} atoms, {glass_mass:.0f} amu")
    print(f"  polymer: {len(lengths)} chains x {lengths.min()}-{lengths.max()} units, "
          f"{polymer_mass:.0f} amu")
    print(f"  box    : {box:.2f} A at build density, about {box_final:.2f} A after compression "
          f"(surface-to-image gap {box_final - 2 * r_np:.1f} A)")

    # ---- grow the chains
    d_poly, d_glass = float(c["min_distance"]), float(c["min_distance_glass"])
    def wrap(x):                                 # into [0, box) for the periodic k-d tree
        w = np.mod(x + box / 2, box)
        return np.where(w >= box, 0.0, w)
    glass_tree = cKDTree(wrap(g_pos), boxsize=box) if len(g_pos) else None
    placed = np.empty((0, 3))
    chains = []

    for ci, n_units in enumerate(lengths):
        tree = cKDTree(wrap(placed), boxsize=box) if len(placed) else None

        def accept(new, chain):
            w = wrap(new)
            if glass_tree is not None and                     glass_tree.query(w, distance_upper_bound=d_glass)[0].min() < d_glass:
                return False
            if tree is not None and tree.query(w, distance_upper_bound=d_poly)[0].min() < d_poly:
                return False
            own = np.delete(chain, polymer.ATTACH, axis=0) if len(chain) >= -polymer.ATTACH else chain
            if len(own):                        # the atom the new unit is bonded to is left out
                d = new[:, None, :] - own[None, :, :]
                d -= box * np.round(d / box)
                d2 = (d * d).sum(-1)
                limit = np.full(len(own), d_poly ** 2)
                limit[-polymer.JUNCTION:] = 1.5 ** 2     # previous unit: 1-3 / 1-4 partners
                if (d2 < limit).any():
                    return False
            return True

        for attempt in range(int(c["max_chain_restarts"])):
            start = (rng.random(3) - 0.5) * box
            if not accept(start[None, :], np.empty((1, 3))):
                continue
            xyz = polymer.grow_chain(n_units, start, rng, accept, int(c["torsion_trials"]))
            if xyz is not None:
                break
        else:
            raise RuntimeError(f"chain {ci + 1} could not be placed: lower composite.build_density")
        chains.append(xyz)
        placed = np.concatenate([placed, xyz])
        if (ci + 1) % max(1, len(lengths) // 10) == 0 or ci + 1 == len(lengths):
            print(f"    chain {ci + 1}/{len(lengths)} placed")

    # ---- assemble the system
    pos, types, mol = [g_pos], [g_type], [np.zeros(len(g_pos), np.int32)]
    bonds, angles, dihedrals, impropers, cmaps = [], [], [], [], []
    offset = len(g_pos)
    for ci, (xyz, n_units) in enumerate(zip(chains, lengths)):
        t, b, a, d, i, cm = polymer.chain_topology(int(n_units))
        pos.append(xyz), types.append(t), mol.append(np.full(len(t), ci + 1, np.int32))
        bonds.append(b + offset), angles.append(a + offset)
        dihedrals.append(d + offset), impropers.append(i + offset), cmaps.append(cm + offset)
        offset += len(t)
    system = System(box=float(box), pos=np.concatenate(pos), type=np.concatenate(types),
                    mol=np.concatenate(mol), mobile=np.concatenate(mol) > 0,
                    bonds=np.concatenate(bonds), angles=np.concatenate(angles),
                    dihedrals=np.concatenate(dihedrals), impropers=np.concatenate(impropers),
                    cmaps=np.concatenate(cmaps))

    net_charge = system.charge.sum()
    report = {
        "neat_polymer_control": neat, "n_atoms": system.n, "n_glass_atoms": int(len(g_pos)),
        "n_chains": int(len(lengths)), "units_per_chain": [int(n) for n in lengths],
        "glass_mass_amu": float(glass_mass), "polymer_mass_amu": float(polymer_mass),
        "glass_weight_percent": 100.0 * system.glass_weight_fraction,
        "polymer_weight_percent": 100.0 * (1.0 - system.glass_weight_fraction),
        "glass_volume_percent_expected": 100.0 * glass_volume / box_final ** 3,
        "box_A": float(box), "density_gcc": system.density,
        "expected_equilibrium_box_A": float(box_final), "net_charge_e": float(net_charge),
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    system.save(outdir / "system.npz")
    write_system_xyz(outdir / "composite_initial.xyz", system,
                     wrap=cfg["system"].get("xyz_wrap", "atom"))
    (outdir / "composite.json").write_text(json.dumps(report, indent=2))
    print(f"  built  : {system.n} atoms, polymer : glass = "
          f"{report['polymer_weight_percent']:.2f} : {report['glass_weight_percent']:.2f} wt%, "
          f"density {system.density:.3f} g/cm3, net charge {net_charge:+.4f} e")
    return report
