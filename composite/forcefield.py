"""Force-field definition for the 45S5 bioactive-glass / PCL composite.

Units used everywhere in the package: kcal/mol, Angstrom, amu, Kelvin, atm.

Polymer   : all-atom poly(epsilon-caprolactone), L-OPLS / OPLS-AA of Bacova, Gonzalez Huarte, Harmandaris and
            Molina, Open Research Europe 5:296 (2025).  The parameters are read from the authors' GROMACS file by
            composite/pcl.py.
              E_bond  = K (r - r0)^2,  E_angle = K (theta - theta0)^2,
              E_dih   = 1/2 [V1 (1 + cos phi) + V2 (1 - cos 2 phi) + V3 (1 + cos 3 phi)],
            geometric combining rules, 1-4 Lennard-Jones and Coulomb scaled by 0.5.
Glass     : rigid body cut from the melt-quenched bulk.  Partial charges are the Teter charges (0.6 x formal)
            the bulk glass was generated with (Du, Springer Handbook of Glass, 2019).
Interface : Coulomb + Lennard-Jones; glass LJ parameters from UFF (Rappe et al., JACS 114, 10024 (1992);
            sigma = x_i / 2^(1/6), eps = D_i), mixed geometrically.  The strength can be scaled
            (cross_epsilon_scale) for a sensitivity analysis, and individual pairs can be overridden (NBFIX).
"""

from __future__ import annotations

import numpy as np

from . import pcl

KB = 0.0019872041            # kcal/(mol K)
COULOMB = 332.06371          # kcal A/(mol e^2)
ATM = 1.458397e-5            # kcal/(mol A^3) per atm
AMU_A3_TO_GCC = 1.66053907   # amu/A^3 -> g/cm^3
TIME_UNIT_FS = 48.88821      # sqrt(amu A^2 / (kcal/mol)) in fs
KJ = pcl.KJ                  # kJ -> kcal

LJ14_SCALE = pcl.LJ14_SCALE
COUL14_SCALE = pcl.COUL14_SCALE

# name: (element, bonded class, mass, charge, sigma [A], epsilon [kcal/mol])
ATOM_TYPES = {}
_ELEMENT_OF_CLASS = {"HO": "H", "HC": "H", "OH": "O", "O_2": "O", "OS": "O", "C_2": "C", "CT": "C"}
for _i, _kind in enumerate(pcl.KIND_NAMES):
    ATOM_TYPES[_kind] = (_ELEMENT_OF_CLASS[pcl.BCLASS[_i]], pcl.BCLASS[_i], float(pcl.MASS[_i]), float(pcl.CHARGE[_i]),
                         float(pcl.SIGMA[_i]), float(pcl.EPSILON[_i]))
# ---- 45S5 glass (Teter charges, UFF Lennard-Jones)
ATOM_TYPES.update({
    "Si":   ("Si", "-", 28.085,  2.400, 3.826, 0.402),
    "P":    ("P",  "-", 30.974,  3.000, 3.695, 0.305),
    "O_gl": ("O",  "-", 15.999, -1.200, 3.118, 0.060),
    "Na":   ("Na", "-", 22.990,  0.600, 2.658, 0.030),
    "Ca":   ("Ca", "-", 40.078,  1.200, 3.028, 0.238),
    "H_si": ("H",  "-",  1.008,  0.600, 2.571, 0.044),   # silanol H (Na+/H+ exchange), UFF H
})
TYPE_NAMES = list(ATOM_TYPES)
TYPE_INDEX = {name: i for i, name in enumerate(TYPE_NAMES)}
GLASS_TYPE = {"Si": "Si", "P": "P", "O": "O_gl", "Na": "Na", "Ca": "Ca", "H": "H_si"}
GLASS_TYPES = {TYPE_INDEX[t] for t in GLASS_TYPE.values()}
N_POLYMER_TYPES = len(pcl.KIND_NAMES)
assert all(TYPE_INDEX[k] == i for i, k in enumerate(pcl.KIND_NAMES))     # polymer kinds come first

ELEMENT = np.array([ATOM_TYPES[t][0] for t in TYPE_NAMES])
BCLASS = [ATOM_TYPES[t][1] for t in TYPE_NAMES]
MASS = np.array([ATOM_TYPES[t][2] for t in TYPE_NAMES])
CHARGE = np.array([ATOM_TYPES[t][3] for t in TYPE_NAMES])
SIGMA = np.array([ATOM_TYPES[t][4] for t in TYPE_NAMES])
EPSILON = np.array([ATOM_TYPES[t][5] for t in TYPE_NAMES])

# Teter Buckingham parameters of the glass,  U = q_i q_j / r + A exp(-r / rho) - C / r^6
# (pair: A [eV], rho [A], C [eV A^6]).  Reference data only: the nanoparticle is a rigid body, so glass-glass
# terms are a constant and are never evaluated.
TETER_BUCKINGHAM = {
    ("Si", "O"): (13702.905, 0.193817, 54.681),
    ("P", "O"): (26655.472, 0.181968, 86.856),
    ("Na", "O"): (4383.7555, 0.243838, 30.70),
    ("Ca", "O"): (7747.1834, 0.252623, 93.109),
    ("O", "O"): (2029.2204, 0.343645, 192.58),
}

# Formal charges in units of the Teter scaling factor (0.6 e); exact integer charge bookkeeping of the particle.
FORMAL_CHARGE = {"Si": 4, "P": 5, "O": -2, "Na": 1, "Ca": 2, "H": 1}

# Pair-specific Lennard-Jones overrides (NBFIX): (type, type): (sigma [A], epsilon [kcal/mol]).
# Hydroxyl hydrogens carry no LJ site in OPLS-AA; next to the -1.2 e glass oxygens the bare Coulomb attraction
# pulls them to an unphysical 1.2-1.3 A.  A small repulsive site restores a hydrogen-bond distance typical of
# hydroxyl groups on silicate surfaces.
# Both values were scanned with tests/calibrate_nbfix.py on a caprolactone trimer; the
# target is H...O(NBO) of about 1.65 A.  HO_end: sigma 1.95 gives 1.68 A.  HO_acid (opls_270, +0.450): sigma 1.98 A
# interpolates 1.63 A (1.95) and 1.67 A (2.00) to 1.65 A.
NBFIX = {
    ("HO_end", "O_gl"): (1.95, 0.10),
    ("HO_acid", "O_gl"): (1.98, 0.10),
}

# --------------------------------------------------------------------------- bonded terms
IMPROPER_V2 = pcl.IMPROPER_V2


def bonded_parameters(types, bonds, angles, dihedrals, impropers, variant=pcl.DEFAULT_VARIANT):
    """Per-interaction parameter arrays for a topology given as index arrays (types index the full type table)."""
    cls = [BCLASS[t] for t in types]
    table = pcl.DIHEDRALS_BY_VARIANT[variant]
    bond_par = np.array([pcl.lookup(pcl.BONDS, (cls[i], cls[j])) for i, j in bonds], dtype=np.float64).reshape(-1, 2)
    angle_par = np.array([pcl.lookup(pcl.ANGLES, (cls[i], cls[j], cls[k])) for i, j, k in angles],
                         dtype=np.float64).reshape(-1, 2)
    angle_par[:, 1] = np.radians(angle_par[:, 1])
    dih_par = np.array([pcl.lookup(table, (cls[i], cls[j], cls[k], cls[l])) for i, j, k, l in dihedrals],
                       dtype=np.float64).reshape(-1, 3)
    imp_par = np.tile([0.0, IMPROPER_V2, 0.0], (len(impropers), 1))
    torsions = np.concatenate([np.asarray(dihedrals).reshape(-1, 4), np.asarray(impropers).reshape(-1, 4)]).astype(np.int32)
    torsion_par = np.concatenate([dih_par, imp_par])
    keep = np.any(np.abs(torsion_par) > 1e-12, axis=1)     # drop all-zero terms
    return bond_par, angle_par, torsions[keep], torsion_par[keep]


def cmap_coefficients():
    raise NotImplementedError("the PCL force field has no CMAP term")


# --------------------------------------------------------------------------- non-bonded
def lj_tables(cutoff, cross_epsilon_scale=1.0):
    """Type-pair tables A = 4 eps sigma^12, B = 4 eps sigma^6 and the energy offset at the cutoff.

    The offset is zero (plain truncation): an energy shift would count every pair inside
    the cutoff in volume moves and cancel the long-range dispersion correction.
    cross_epsilon_scale multiplies eps of every glass-polymer pair (sensitivity analysis)."""
    sig = np.sqrt(np.outer(SIGMA, SIGMA))
    eps = np.sqrt(np.outer(EPSILON, EPSILON))
    glass = np.array([i in GLASS_TYPES for i in range(len(TYPE_NAMES))])
    cross = np.logical_xor.outer(glass, glass)
    eps = np.where(cross, eps * cross_epsilon_scale, eps)
    for (ta, tb), (s, e) in NBFIX.items():
        i, j = TYPE_INDEX[ta], TYPE_INDEX[tb]
        sig[i, j] = sig[j, i] = s
        eps[i, j] = eps[j, i] = e * (cross_epsilon_scale if cross[i, j] else 1.0)
    a = 4.0 * eps * sig ** 12
    b = 4.0 * eps * sig ** 6
    shift = np.zeros_like(a)
    return a, b, shift
