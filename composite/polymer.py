"""All-atom poly(epsilon-caprolactone):  HOOC-(CH2)5-[O-CO-(CH2)5]n-2 ... -O-CH2-(CH2)4-CH2-OH

The topology (atom order, types, charges, bonded terms) comes from composite/pcl.py and equals the authors'
GROMACS topology.  This module adds the three-dimensional chain growth.

Atom order of a chain of n units:
    unit 0 (19 atoms)      : HO OH C O   C H H  C H H  C H H  C H H  C(490) H H
    units 1 .. n-2 (18)    : O C O       C H H  C H H  C H H  C H H  C(490) H H
    unit n-1 (20 atoms)    : O C O       C H H  C H H  C H H  C H H  C(157) H H  OH HO
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from . import forcefield as ff
from . import pcl

UNIT_MASS = pcl.chain_mass(4) - pcl.chain_mass(3)           # C6H10O2
END_MASS = pcl.chain_mass(3) - 3 * UNIT_MASS                # H + OH
ATTACH = -3          # index (within the atoms already placed) of the atom the next unit's oxygen bonds to
JUNCTION = 8         # trailing atoms of the placed chain that are 1-3 / 1-4 partners of the next unit
T_BUILD = 900.0      # K, torsions are drawn from Boltzmann weights at this temperature
DEG = np.pi / 180.0


def chain_mass(n_units):
    return pcl.chain_mass(n_units)


def chain_topology(n_units):
    """Types and bonded terms of one chain (local atom indices); the last array (CMAP) is empty."""
    types, bonds, angles, dihedrals, impropers = pcl.chain_topology(n_units)
    return types, bonds, angles, dihedrals, impropers, np.zeros((0, 5), np.int32)


def exclusions(n_atoms, bonds, angles, dihedrals):
    """Padded per-atom lists (-1 filled) of excluded (1-2, 1-3) and scaled (1-4) partners."""
    excl = [set() for _ in range(n_atoms)]
    for i, j in bonds:
        excl[i].add(j), excl[j].add(i)
    for i, _, k in angles:
        excl[i].add(k), excl[k].add(i)
    p14 = [set() for _ in range(n_atoms)]
    for i, _, _, l in dihedrals:
        if l not in excl[i] and i != l:
            p14[i].add(l), p14[l].add(i)

    def pad(sets):
        width = max(1, max((len(s) for s in sets), default=1))
        out = np.full((n_atoms, width), -1, np.int32)
        for i, s in enumerate(sets):
            out[i, :len(s)] = sorted(s)
        return out
    return pad(excl), pad(p14)


# ----------------------------------------------------------------------------- geometry
def place(a, b, c, bond, angle, torsion):
    """Position of atom d from the internal coordinates |cd|, angle(b,c,d), torsion(a,b,c,d)."""
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    return c + bond * (-np.cos(angle) * bc
                       + np.sin(angle) * (np.cos(torsion) * m + np.sin(torsion) * n))


def random_unit_vector(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _bond(c1, c2):
    return pcl.lookup(pcl.BONDS, (c1, c2))[1]


def _angle(c1, c2, c3):
    return pcl.lookup(pcl.ANGLES, (c1, c2, c3))[1] * DEG


_GRID = np.linspace(-np.pi, np.pi, 72, endpoint=False)


def _torsion(rng, key):
    """Dihedral drawn from the Boltzmann weight of the force-field torsion of this class (flat if there is none)."""
    table = pcl.DIHEDRALS_BY_VARIANT[pcl.DEFAULT_VARIANT]
    try:
        v1, v2, v3 = pcl.lookup(table, key)
    except KeyError:
        return rng.uniform(-np.pi, np.pi)
    e = 0.5 * (v1 * (1 + np.cos(_GRID)) + v2 * (1 - np.cos(2 * _GRID)) + v3 * (1 + np.cos(3 * _GRID)))
    w = np.exp(-(e - e.min()) / (ff.KB * T_BUILD))
    k = rng.choice(len(_GRID), p=w / w.sum())
    return _GRID[k] + rng.uniform(-1, 1) * (np.pi / len(_GRID))


def _step(rng, a, b, c, ca, cb, cc, cd, torsion=None):
    """Next backbone atom of class cd after (a, b, c) of classes (ca, cb, cc)."""
    t = _torsion(rng, (ca, cb, cc, cd)) if torsion is None else torsion
    return place(a, b, c, _bond(cc, cd), _angle(cb, cc, cd), t)


def _carbonyl_oxygen(prev, c, nxt):
    """Planar carbonyl oxygen of the carbon c between its two chain neighbours."""
    u1 = (prev - c) / np.linalg.norm(prev - c)
    u2 = (nxt - c) / np.linalg.norm(nxt - c)
    d = -(u1 + u2)
    return c + _bond("C_2", "O_2") * d / np.linalg.norm(d)


def _ch2_hydrogens(p, x, n):
    """The two hydrogens of a CH2 group at x between its chain neighbours p and n."""
    u1 = (p - x) / np.linalg.norm(p - x)
    u2 = (n - x) / np.linalg.norm(n - x)
    bis = u1 + u2
    bis /= np.linalg.norm(bis)
    perp = np.cross(u1, u2)
    perp /= np.linalg.norm(perp)
    half = 0.5 * _angle("HC", "CT", "HC")
    r = _bond("CT", "HC")
    return [x + r * (-np.cos(half) * bis + s * np.sin(half) * perp) for s in (1.0, -1.0)]


def _build_chain_unit(rng, frame, pending, last):
    """Middle or last unit.  frame = (C3, C4, C5) of the previous unit, pending = position of its oxygen."""
    c3p, c4p, c5p = frame
    o = pending
    c = _step(rng, c4p, c5p, o, "CT", "CT", "OS", "C_2")
    c1 = _step(rng, c5p, o, c, "CT", "OS", "C_2", "CT", torsion=np.pi + rng.normal(0.0, 10.0 * DEG))
    oc = _carbonyl_oxygen(o, c, c1)
    c2 = _step(rng, o, c, c1, "OS", "C_2", "CT", "CT")
    c3 = _step(rng, c, c1, c2, "C_2", "CT", "CT", "CT")
    c4 = _step(rng, c1, c2, c3, "CT", "CT", "CT", "CT")
    c5 = _step(rng, c2, c3, c4, "CT", "CT", "CT", "CT")
    nxt = _step(rng, c3, c4, c5, "CT", "CT", "CT", "OH" if last else "OS")
    atoms = [o, c, oc]
    carbons = [c1, c2, c3, c4, c5]
    nbr = [(c, c2), (c1, c3), (c2, c4), (c3, c5), (c4, nxt)]
    for x, (p, n) in zip(carbons, nbr):
        h1, h2 = _ch2_hydrogens(p, x, n)
        atoms += [x, h1, h2]
    if last:
        ho = place(c4, c5, nxt, _bond("OH", "HO"), _angle("CT", "OH", "HO"), rng.uniform(-np.pi, np.pi))
        atoms += [nxt, ho]
    return np.array(atoms), (c3, c4, c5), nxt


def _build_acid_unit(rng, start):
    """First unit, anchored at the acid hydroxyl oxygen `start`."""
    v1 = random_unit_vector(rng)
    c = start + _bond("OH", "C_2") * v1
    perp = np.cross(v1, random_unit_vector(rng))
    perp /= np.linalg.norm(perp)
    ang = _angle("HO", "OH", "C_2")
    ho = start + _bond("OH", "HO") * (np.cos(ang) * v1 + np.sin(ang) * perp)
    c1 = _step(rng, ho, start, c, "HO", "OH", "C_2", "CT")
    oc = _carbonyl_oxygen(start, c, c1)
    c2 = _step(rng, start, c, c1, "OH", "C_2", "CT", "CT")
    c3 = _step(rng, c, c1, c2, "C_2", "CT", "CT", "CT")
    c4 = _step(rng, c1, c2, c3, "CT", "CT", "CT", "CT")
    c5 = _step(rng, c2, c3, c4, "CT", "CT", "CT", "CT")
    nxt = _step(rng, c3, c4, c5, "CT", "CT", "CT", "OS")
    atoms = [ho, start, c, oc]
    for x, (p, n) in zip([c1, c2, c3, c4, c5], [(c, c2), (c1, c3), (c2, c4), (c3, c5), (c4, nxt)]):
        h1, h2 = _ch2_hydrogens(p, x, n)
        atoms += [x, h1, h2]
    return np.array(atoms), (c3, c4, c5), nxt


def grow_chain(n_units, start, rng, accept, k_trials=12, max_failures=200):
    """Grow one chain by configurational sampling of the backbone torsions.

    start is the position of the acid hydroxyl oxygen.  accept(new_xyz, chain_xyz) -> bool decides whether the atoms of
    a trial unit are clash free; chain_xyz holds the atoms already placed.  Returns the coordinates in topology order,
    or None when the chain is trapped."""
    xyz = []
    frames, pendings, ends = [], [], [0]
    failures = 0
    u = 0
    kinds, bonds, angles, dihedrals, _, _ = chain_topology(n_units)
    n_atoms = len(kinds)
    near = np.eye(n_atoms, dtype=bool)                  # pairs within three bonds are not checked
    near[bonds[:, 0], bonds[:, 1]] = near[bonds[:, 1], bonds[:, 0]] = True
    near[angles[:, 0], angles[:, 2]] = near[angles[:, 2], angles[:, 0]] = True
    near[dihedrals[:, 0], dihedrals[:, 3]] = near[dihedrals[:, 3], dihedrals[:, 0]] = True
    is_h = np.array([ff.ELEMENT[k] == "H" for k in kinds])

    def self_clash(trial):
        """True if the trial atoms overlap atoms of the same chain that are more than three bonds away."""
        first = len(xyz)
        new_idx = np.arange(first, first + len(trial))
        have = np.vstack([np.array(xyz).reshape(-1, 3), trial])
        d = np.linalg.norm(trial[:, None, :] - have[None, :, :], axis=-1)
        idx = np.arange(len(have))
        limit = np.where(is_h[new_idx][:, None] | is_h[idx][None, :], 1.9, 2.5)
        mask = ~near[np.ix_(new_idx, idx)]
        return bool(((d < limit) & mask).any())

    while u < n_units:
        placed = False
        for _ in range(k_trials):
            if u == 0:
                trial, frame, nxt = _build_acid_unit(rng, start)
            else:
                trial, frame, nxt = _build_chain_unit(rng, frames[u - 1], pendings[u - 1], u == n_units - 1)
            test = trial if u == n_units - 1 else np.vstack([trial, nxt])
            if not self_clash(test):
                if accept(test, np.array(xyz).reshape(-1, 3)):
                    xyz.extend(trial)
                    frames.append(frame)
                    pendings.append(nxt)
                    ends.append(len(xyz))
                    placed = True
                    break
        if placed:
            u += 1
            continue
        failures += 1
        if failures > max_failures:
            return None
        back = min(u, 2)                        # dead end: step back two units
        u -= back
        del xyz[ends[u]:]
        del frames[u:]
        del pendings[u:]
        del ends[u + 1:]
    return np.array(xyz)
