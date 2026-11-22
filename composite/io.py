"""Structure I/O: plain / extended XYZ and the binary system container."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from . import forcefield as ff


@dataclass
class System:
    """Complete simulation state.  The box is cubic and centred on the origin,
    the nanoparticle (molecule 0) sits at the origin, coordinates are unwrapped."""
    box: float
    pos: np.ndarray          # (N, 3) float64
    type: np.ndarray         # (N,) int32, index into forcefield.TYPE_NAMES
    mol: np.ndarray          # (N,) int32, 0 = nanoparticle, 1.. = polymer chains
    mobile: np.ndarray       # (N,) bool
    bonds: np.ndarray        # (nb, 2) int32
    angles: np.ndarray       # (na, 3)
    dihedrals: np.ndarray    # (nd, 4)
    impropers: np.ndarray    # (ni, 4)
    cmaps: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), np.int32))  # (nc, 5)

    @property
    def n(self):
        return len(self.pos)

    @property
    def symbols(self):
        return ff.ELEMENT[self.type]

    @property
    def mass(self):
        return ff.MASS[self.type]

    @property
    def charge(self):
        return ff.CHARGE[self.type]

    @property
    def density(self):
        return self.mass.sum() / self.box ** 3 * ff.AMU_A3_TO_GCC

    @property
    def glass_weight_fraction(self):
        m = self.mass
        return m[self.mol == 0].sum() / m.sum()

    @property
    def has_particle(self):
        return bool(np.any(self.mol == 0))

    def save(self, path):
        np.savez_compressed(path, type_names=np.array(ff.TYPE_NAMES),
                            **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            names = list(data["type_names"]) if "type_names" in data else None
            if names != ff.TYPE_NAMES:
                raise ValueError(f"{path} was written with a different force-field type table "
                                 "(an older version of the pipeline); rebuild it")
            return cls(**{f.name: (float(data[f.name]) if f.name == "box" else data[f.name])
                          for f in fields(cls) if f.name in data})


def read_xyz(path):
    """Return (symbols, positions, comment line) of a single-frame XYZ file."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    rows = [line.split() for line in lines[2:2 + n]]
    symbols = np.array([r[0] for r in rows])
    pos = np.array([[float(v) for v in r[1:4]] for r in rows])
    return symbols, pos, lines[1]


def box_from_comment(comment):
    """Cubic box edge from 'box=53.48' or an extended-XYZ Lattice entry, else None."""
    m = re.search(r"box\s*=\s*([0-9.eE+-]+)", comment)
    if m:
        return float(m.group(1).rstrip(","))
    m = re.search(r'Lattice="([^"]+)"', comment)
    if m:
        return float(m.group(1).split()[0])
    return None


def write_extxyz(path, symbols, pos, box=None, columns=None, info="", append=False, wrap=False):
    """Write an extended-XYZ frame readable by OVITO / ASE / VMD.

    columns: {name: (N,) array} of extra per-atom integer or real properties."""
    columns = columns or {}
    pos = np.asarray(pos)
    if wrap and box:
        pos = pos - box * np.floor(pos / box + 0.5)
    props = "species:S:1:pos:R:3"
    for name, values in columns.items():
        props += f":{name}:{'I' if np.issubdtype(np.asarray(values).dtype, np.integer) else 'R'}:1"
    header = f"Properties={props}"
    if box:
        header = (f'Lattice="{box:.6f} 0 0 0 {box:.6f} 0 0 0 {box:.6f}" '
                  f'Origin="{-box / 2:.6f} {-box / 2:.6f} {-box / 2:.6f}" {header} pbc="T T T"')
    if info:
        header += " " + info
    cols = [np.asarray(v) for v in columns.values()]
    with open(path, "a" if append else "w") as fh:
        fh.write(f"{len(pos)}\n{header}\n")
        for i, (s, (x, y, z)) in enumerate(zip(symbols, pos)):
            extra = "".join(f" {c[i]}" if np.issubdtype(c.dtype, np.integer) else f" {c[i]:.4f}"
                            for c in cols)
            fh.write(f"{s:<2s} {x:12.6f} {y:12.6f} {z:12.6f}{extra}\n")


def write_system_xyz(path, system, pos=None, box=None, info="", append=False, wrap="atom"):
    """Frame of the composite, folded back into the periodic box.

    wrap = "atom"     : every atom inside the box (chains crossing a face are cut in the picture)
    wrap = "molecule" : chains kept whole, placed by their centre (atoms may stick out)"""
    pos = system.pos if pos is None else pos
    box = system.box if box is None else box
    if wrap == "atom":
        out = pos - box * np.floor(pos / box + 0.5)
        out[out >= box / 2] -= box                    # guard against round-off at the upper face
    elif wrap == "molecule":
        n_mol = int(system.mol.max()) + 1
        count = np.bincount(system.mol, minlength=n_mol)[:, None]
        centre = np.stack([np.bincount(system.mol, pos[:, d], n_mol) for d in range(3)], 1) / count
        out = pos - box * np.floor(centre / box + 0.5)[system.mol]
    else:
        raise ValueError("xyz_wrap must be 'atom' or 'molecule'")
    write_extxyz(path, system.symbols, out, box,
                 {"mol": system.mol, "charge": system.charge}, info, append)
