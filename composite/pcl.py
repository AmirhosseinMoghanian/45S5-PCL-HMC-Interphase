"""All-atom poly(epsilon-caprolactone):  HOOC-(CH2)5-[O-CO-(CH2)5]n-2 ... -O-CH2-(CH2)4-CH2-OH

Force field: Bacova, Gonzalez Huarte, Harmandaris, Molina, Open Research Europe 5:296 (2025),
DOI 10.12688/openreseurope.21354.2.  L-OPLS (Siu et al., J. Chem. Theory Comput. 8, 1459 (2012)) is the main
source, OPLS-AA (Jorgensen et al., J. Am. Chem. Soc. 118, 11225 (1996)) the second one, and the ester dihedral
table is the one of their polylactide force field.  The GROMACS files of the authors are in data/pcl_source and are
read here (nothing is typed by hand).

Units of the package: kcal/mol, Angstrom, amu.  GROMACS values are converted as
    bond     E = K (r - r0)^2           K = K_gmx / 2 / 100 / 4.184,   r0 = 10 r0_nm
    angle    E = K (theta - theta0)^2   K = K_gmx / 2 / 4.184
    torsion  Ryckaert-Bellemans C0..C3 -> OPLS Fourier V1, V2, V3 (C4 = C5 = 0 is checked)
    LJ       sigma = 10 sigma_nm,  eps = eps_kJ / 4.184   (combination rule 3, 1-4 scaling 0.5)

Atom order of a chain with n repeat units (18 n + 3 atoms, as in the authors' topology):
    unit 0 (acid end, 19 atoms)   : HO OH C O  4 x (CH2: C H H)  C(490) H H
    units 1 .. n-2 (18 atoms)     : O C O  4 x (CH2: C H H)  C(490) H H
    unit n-1 (hydroxyl end, 20)   : O C O  4 x (CH2: C H H)  C(157) H H  OH HO
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

KJ = 1.0 / 4.184
SOURCE = Path(__file__).resolve().parents[1] / "data" / "pcl_source"
ITP_FILES = {"pla_ester": SOURCE / "pcl_ff_final.itp",             # solvated runs: ester dihedrals taken from PLA
             "lopls_ester": SOURCE / "pcl_ff_final_vacuum.itp"}    # vacuum runs: ester dihedrals from L-OPLS (Siu et al., table 2)
DEFAULT_VARIANT = "lopls_ester"
ITP = ITP_FILES[DEFAULT_VARIANT]
LJ14_SCALE = 0.5
COUL14_SCALE = 0.5

# kind: (opls type, charge) in the order of first appearance in the topology
KINDS = {
    "HO_acid": ("opls_270", 0.450), "OH_acid": ("opls_268", -0.530), "C_acid": ("opls_267", 0.520),
    "O_acid": ("opls_269", -0.440),
    "CH2": ("opls_136", -0.148), "HC": ("opls_140", 0.074),
    "CH2_O": ("opls_490", 0.190), "HC_O": ("opls_469", 0.030),
    "O_ester": ("opls_4671", -0.450), "C_ester": ("opls_4651", 0.750), "O_carbonyl": ("opls_4662", -0.550),
    "CH2_OH": ("opls_157", 0.145), "HC_OH": ("opls_176", 0.060),
    "OH_end": ("opls_154", -0.683), "HO_end": ("opls_155", 0.418),
}
KIND_NAMES = list(KINDS)
KIND_INDEX = {k: i for i, k in enumerate(KIND_NAMES)}


def parse_itp(path=ITP):
    """Atom types, bonded types and Ryckaert-Bellemans dihedrals of the GROMACS force-field file."""
    atomtypes, bonds, angles, rb, impropers = {}, {}, {}, {}, {}
    section, improper_line = None, None
    for raw in Path(path).read_text().splitlines():
        line = raw.split(";")[0].strip()
        if raw.strip().startswith("#define improper_O_C_X_Y"):
            improper_line = raw.split()[2:5]
            continue
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        p = line.split()
        if section == "atomtypes":
            if len(p) == 7 and p[4] == "A":
                atomtypes[p[0]] = dict(bclass=p[1], mass=float(p[2]), charge=float(p[3]),
                                       sigma_nm=float(p[5]), eps_kj=float(p[6]))
        elif section == "bondtypes":
            bonds[(p[0], p[1])] = (float(p[3]), float(p[4]))
        elif section == "angletypes":
            angles[(p[0], p[1], p[2])] = (float(p[4]), float(p[5]))
        elif section == "dihedraltypes" and len(p) == 11 and p[4] == "3":
            c = [float(v) for v in p[5:11]]
            if abs(c[4]) > 1e-12 or abs(c[5]) > 1e-12:
                raise ValueError(f"C4/C5 non-zero for {p[:4]}")
            rb[tuple(p[:4])] = c[:4]
    k = float(improper_line[1])
    return atomtypes, bonds, angles, rb, k


ATOMTYPES, _BONDS_GMX, _ANGLES_GMX, _RB, _IMPROPER_K = parse_itp(ITP)
_ATOMTYPES_ALL = {}
_RB_ALL = {}
for _name, _path in ITP_FILES.items():
    _at, _bo, _an, _rb, _k = parse_itp(_path)
    # everything except the Ryckaert-Bellemans table must be identical in the two files
    assert {t: v for t, v in _at.items() if t in ATOMTYPES} == ATOMTYPES and _bo == _BONDS_GMX and _an == _ANGLES_GMX and _k == _IMPROPER_K
    _RB_ALL[_name] = _rb

# ----------------------------------------------------------------------------- per-kind data
ELEMENT = np.array([ATOMTYPES[KINDS[k][0]]["bclass"][0] if ATOMTYPES[KINDS[k][0]]["bclass"][0] in "CHO" else "?" for k in KINDS])
BCLASS = [ATOMTYPES[KINDS[k][0]]["bclass"] for k in KINDS]
MASS = np.array([ATOMTYPES[KINDS[k][0]]["mass"] for k in KINDS])
CHARGE = np.array([KINDS[k][1] for k in KINDS])
SIGMA = np.array([10.0 * ATOMTYPES[KINDS[k][0]]["sigma_nm"] for k in KINDS])
EPSILON = np.array([ATOMTYPES[KINDS[k][0]]["eps_kj"] * KJ for k in KINDS])

BONDS = {k: (v[1] / 2.0 / 100.0 * KJ, 10.0 * v[0]) for k, v in _BONDS_GMX.items()}
ANGLES = {k: (v[1] / 2.0 * KJ, v[0]) for k, v in _ANGLES_GMX.items()}


def _fourier(rb):
    return {k: ((-2.0 * c[1] - 1.5 * c[3]) * KJ, -c[2] * KJ, -0.5 * c[3] * KJ) for k, c in rb.items()}


DIHEDRALS_BY_VARIANT = {name: _fourier(rb) for name, rb in _RB_ALL.items()}
DIHEDRALS = DIHEDRALS_BY_VARIANT[DEFAULT_VARIANT]
IMPROPER_V2 = 2.0 * _IMPROPER_K * KJ


def lookup(table, key):
    if key in table:
        return table[key]
    if key[::-1] in table:
        return table[key[::-1]]
    raise KeyError("no parameters for " + "-".join(key))


# ----------------------------------------------------------------------------- topology
def chain_atoms(n_units):
    """Kinds and bonds of one chain, local atom indices."""
    if n_units < 3:
        raise ValueError("chains need at least 3 repeat units")
    kinds, bonds = [], []

    def add(kind):
        kinds.append(KIND_INDEX[kind])
        return len(kinds) - 1

    def chain_tail(prev_c, end_kind, h_kind):
        """Four CH2 groups from the carbonyl carbon, then the O-side CH2; returns its index."""
        prev = prev_c
        for _ in range(4):
            c = add("CH2")
            bonds.append((prev, c))
            for _h in range(2):
                bonds.append((c, add("HC")))
            prev = c
        c = add(end_kind)
        bonds.append((prev, c))
        for _h in range(2):
            bonds.append((c, add(h_kind)))
        return c

    impropers = []
    # acid end
    ho, oh, c, o = add("HO_acid"), add("OH_acid"), add("C_acid"), add("O_acid")
    bonds += [(ho, oh), (oh, c), (c, o)]
    first_ch2 = len(kinds)
    impropers.append((first_ch2, oh, c, o))
    prev_o_side = chain_tail(c, "CH2_O", "HC_O")
    for u in range(1, n_units):
        last = u == n_units - 1
        os_, c, o = add("O_ester"), add("C_ester"), add("O_carbonyl")
        bonds += [(prev_o_side, os_), (os_, c), (c, o)]
        first_ch2 = len(kinds)
        impropers.append((first_ch2, os_, c, o))
        prev_o_side = chain_tail(c, "CH2_OH" if last else "CH2_O", "HC_OH" if last else "HC_O")
    oh, ho = add("OH_end"), add("HO_end")
    bonds += [(prev_o_side, oh), (oh, ho)]
    return np.array(kinds, np.int32), bonds, impropers


def chain_topology(n_units):
    """(kinds, bonds, angles, dihedrals, impropers) of one chain."""
    kinds, bonds, impropers = chain_atoms(n_units)
    neigh = [[] for _ in kinds]
    for i, j in bonds:
        neigh[i].append(j)
        neigh[j].append(i)
    angles = [(i, j, k) for j, nb in enumerate(neigh) for i, k in combinations(nb, 2)]
    dihedrals = [(i, j, k, l) for j, k in bonds
                 for i in neigh[j] if i != k for l in neigh[k] if l != j]
    return (kinds, np.array(bonds, np.int32), np.array(angles, np.int32),
            np.array(dihedrals, np.int32), np.array(impropers, np.int32))


def chain_mass(n_units):
    kinds, _, _ = chain_atoms(n_units)
    return float(MASS[kinds].sum())


def bonded_parameters(kinds, bonds, angles, dihedrals, impropers, variant=DEFAULT_VARIANT):
    """Per-interaction parameter arrays for a topology given as index arrays (kcal/mol, A, rad)."""
    cls = [BCLASS[t] for t in kinds]
    bond_par = np.array([lookup(BONDS, (cls[i], cls[j])) for i, j in bonds], float).reshape(-1, 2)
    angle_par = np.array([lookup(ANGLES, (cls[i], cls[j], cls[k])) for i, j, k in angles], float).reshape(-1, 2)
    angle_par[:, 1] = np.radians(angle_par[:, 1])
    table = DIHEDRALS_BY_VARIANT[variant]
    dih_par = np.array([lookup(table, (cls[i], cls[j], cls[k], cls[l])) for i, j, k, l in dihedrals],
                       float).reshape(-1, 3)
    imp_par = np.tile([0.0, IMPROPER_V2, 0.0], (len(impropers), 1))
    torsions = np.concatenate([np.asarray(dihedrals).reshape(-1, 4), np.asarray(impropers).reshape(-1, 4)]).astype(np.int32)
    torsion_par = np.concatenate([dih_par, imp_par])
    keep = np.any(np.abs(torsion_par) > 1e-12, axis=1)
    return bond_par, angle_par, torsions[keep], torsion_par[keep]
