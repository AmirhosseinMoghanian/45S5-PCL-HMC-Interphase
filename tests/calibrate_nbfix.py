"""Calibration of the NBFIX repulsion between hydroxyl hydrogens and glass oxygens.

    python tests/calibrate_nbfix.py

A caprolactone trimer is hydrogen-bonded through its hydroxyl (or carboxylic-acid) end to
the most exposed non-bridging oxygen of a rigid 45S5 particle and relaxed by steepest
descent for several NBFIX sigma values.  The chosen value is the one that gives an
H...O(NBO) distance of about 1.65 A, typical of a strong hydrogen bond to a siloxide
oxygen (O-H...O- 1.5-1.8 A).
"""

import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from composite import forcefield as ff, polymer                     # noqa: E402
from composite.gpu_engine import Engine, cp                         # noqa: E402
from composite.io import System                                     # noqa: E402
from composite.nanoparticle import Bulk, FORMERS                    # noqa: E402
from composite.io import read_xyz, box_from_comment                 # noqa: E402


def particle(radius=10.0):
    cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    sym, pos, comment = read_xyz(ROOT / cfg["system"]["bulk_glass"])
    bulk = Bulk(sym, pos, box_from_comment(comment), 2.1)
    centre = np.full(3, bulk.box / 2)
    mask, r = bulk.select(centre, radius)
    mask, _ = bulk.neutralise(mask, r)
    xyz = bulk.displacement(centre)[mask]
    return bulk.symbols[mask], xyz - xyz.mean(axis=0)


def relax(engine, steps=3000):
    e = engine.compute()
    h = 0.01
    for _ in range(steps):
        f = engine.f
        fmax = float(cp.sqrt(cp.max(cp.sum(f * f, axis=1))))
        if fmax < 0.05:
            break
        snap = engine.snapshot()
        engine.pos += (h / fmax) * f
        new = engine.compute()
        if new < e:
            e, h = new, min(1.2 * h, 0.1)
        else:
            engine.restore(snap)
            h *= 0.5
    return e


def scan(end="hydroxyl", sigmas=(0.0, 1.6, 1.8, 1.95, 2.0), eps=0.10):
    gsym, gxyz = particle()
    gtype = np.array([ff.TYPE_INDEX[ff.GLASS_TYPE[s]] for s in gsym], np.int32)
    # most exposed non-bridging oxygen
    formers = np.isin(gsym, FORMERS)
    r = np.linalg.norm(gxyz, axis=1)
    nbo = [i for i in np.argsort(-r) if gsym[i] == "O"
           and np.sum(formers & (np.linalg.norm(gxyz - gxyz[i], axis=1) < 2.1)) == 1][0]
    n_hat = gxyz[nbo] / np.linalg.norm(gxyz[nbo])

    n = 3
    t, b, a, d, imp, cm = polymer.chain_topology(n)
    xyz = polymer.grow_chain(n, np.zeros(3), np.random.default_rng(1), lambda new, chain: True)
    n_at = len(t)
    h_idx, o_idx = (n_at - 1, n_at - 2) if end == "hydroxyl" else (0, 1)     # HO_end / OH_end, HO_acid / OH_acid
    v = xyz[h_idx] - xyz[o_idx]
    v /= np.linalg.norm(v)
    target = -n_hat                                   # O-H must point at the NBO
    axis = np.cross(v, target)
    s, c = np.linalg.norm(axis), v @ target
    k = axis / s if s > 1e-9 else np.zeros(3)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * K @ K
    xyz = (xyz - xyz[h_idx]) @ R.T + gxyz[nbo] + 2.0 * n_hat

    pos = np.concatenate([gxyz, xyz])
    types = np.concatenate([gtype, t])
    mol = np.concatenate([np.zeros(len(gxyz), np.int32), np.ones(len(t), np.int32)])
    off = len(gxyz)
    system = System(80.0, pos, types, mol, mol > 0, b + off, a + off, d + off, imp + off, cm + off)
    print(f"{end} end -> NBO (glass O at r = {r[nbo]:.2f} A)")
    keys = [k for k in ff.NBFIX if k[1] == "O_gl"]
    saved = {k: ff.NBFIX[k] for k in keys}
    for sigma in sigmas:
        for key in keys:
            if sigma > 0:
                ff.NBFIX[key] = (sigma, eps)
            else:
                ff.NBFIX.pop(key, None)
        eng = Engine(system, precision="double", lj_tail=False)
        relax(eng)
        p = cp.asnumpy(eng.pos)
        dist = np.linalg.norm(p[off + h_idx] - p[nbo])
        dO = np.linalg.norm(p[off + o_idx] - p[nbo])
        label = "none" if sigma == 0 else f"sigma {sigma:.2f} A, eps {eps:.2f}"
        print(f"   NBFIX {label:24s}  H...O(NBO) {dist:.3f} A   O...O(NBO) {dO:.3f} A")
    ff.NBFIX.update(saved)


if __name__ == "__main__":
    scan("hydroxyl")
    scan("acid")
