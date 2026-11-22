"""Validation of the GPU engine and the polymer builder.

    python tests/test_engine.py

1. analytic forces vs. central finite differences of the energy, and mixed vs double precision
2. energy invariance when a whole chain is shifted by a lattice vector
3. hybrid-MC energy error scales as dt^2 (time-reversible, symplectic integrator)
(the PCL force field has no CMAP term and no stereocentres, so those two tests are not needed)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from composite import forcefield as ff, polymer                     # noqa: E402
from composite.gpu_engine import Engine, cp                         # noqa: E402
from composite.io import System                                     # noqa: E402


def small_system(seed=3, box=30.0, n_chains=3, n_units=4):
    rng = np.random.default_rng(seed)
    glass_sym = ["Si", "O", "O", "O", "O", "Na", "H", "Ca", "P"]
    g_pos = rng.normal(scale=2.5, size=(len(glass_sym), 3))
    pos, types, mol = [g_pos], [np.array([ff.TYPE_INDEX[ff.GLASS_TYPE[s]] for s in glass_sym])], \
        [np.zeros(len(glass_sym), int)]
    bonds, angles, dihedrals, impropers, cmaps = [], [], [], [], []
    offset = len(glass_sym)
    for c in range(n_chains):
        others = np.concatenate(pos)

        def accept(new, chain, others=others):      # no overlap with the glass fragment and earlier chains
            return float(np.min(np.linalg.norm(new[:, None, :] - others[None, :, :], axis=-1))) > 2.2

        for attempt in range(500):
            start = 8.0 * polymer.random_unit_vector(rng)
            if accept(start[None, :], None):
                xyz = polymer.grow_chain(n_units, start, rng, accept)
                if xyz is not None:
                    break
        else:
            raise RuntimeError("test chain could not be placed")
        t, b, a, d, i, cm = polymer.chain_topology(n_units)
        pos.append(xyz + rng.normal(scale=0.03, size=xyz.shape))   # off-equilibrium geometry
        types.append(t), mol.append(np.full(len(t), c + 1))
        bonds.append(b + offset), angles.append(a + offset)
        dihedrals.append(d + offset), impropers.append(i + offset), cmaps.append(cm + offset)
        offset += len(t)
    mol = np.concatenate(mol).astype(np.int32)
    return System(box, np.concatenate(pos), np.concatenate(types).astype(np.int32), mol, mol > 0,
                  np.concatenate(bonds), np.concatenate(angles), np.concatenate(dihedrals),
                  np.concatenate(impropers), np.concatenate(cmaps))


def test_forces():
    system = small_system()
    eng = Engine(system, cutoff=10.0, skin=2.0, precision="double")
    eng.compute()
    analytic = cp.asnumpy(eng.f)
    rng = np.random.default_rng(0)
    atoms = rng.choice(np.flatnonzero(system.mobile), 25, replace=False)
    h, worst = 1e-4, 0.0
    for i in atoms:
        for d in range(3):
            e = []
            for sign in (+1, -1):
                eng.pos[i, d] += sign * h
                e.append(eng.compute())
                eng.pos[i, d] -= sign * h
            numeric = -(e[0] - e[1]) / (2 * h)
            worst = max(worst, abs(numeric - analytic[i, d]) / max(1.0, abs(analytic[i, d])))
    assert np.allclose(analytic[~system.mobile], 0.0), "frozen atoms must feel no force"
    assert worst < 1e-5, f"force / finite-difference mismatch: {worst:.2e}"
    mixed = Engine(system, cutoff=10.0, skin=2.0, precision="mixed")
    mixed.compute()
    diff = float(cp.max(cp.abs(mixed.f - cp.asarray(analytic)))) / np.abs(analytic).max()
    assert diff < 1e-5, f"mixed-precision forces deviate: {diff:.2e}"
    print(f"forces vs finite differences   max rel. error {worst:.1e}   OK")
    print(f"mixed vs double precision      max rel. error {diff:.1e}   OK")


def test_periodicity():
    system = small_system()
    eng = Engine(system, precision="double")
    e0 = eng.compute()
    shift = cp.asarray((system.mol == 2)[:, None] * np.array([system.box, -2 * system.box, 0.0]))
    eng.pos += shift
    e1 = eng.compute()
    assert abs(e1 - e0) < 1e-7 * max(1.0, abs(e0)), (e0, e1)
    print(f"lattice-translation invariance  dE = {e1 - e0:.1e}   OK")


def test_integrator():
    system = small_system()
    errors = []
    for dt_fs in (0.5, 0.25):
        eng = Engine(system)
        e0 = eng.compute()
        cp.random.seed(1)
        m = eng.mass[:, None]
        v = cp.sqrt(ff.KB * 300.0 / m) * cp.random.standard_normal((eng.n, 3)) * eng.mobile_f
        h0 = e0 + 0.5 * float(cp.sum(m * v * v))
        dt = dt_fs / ff.TIME_UNIT_FS
        n = int(round(20.0 / dt_fs))
        v += 0.5 * dt * eng.f / m
        for step in range(n):
            eng.pos += dt * v
            e = eng.compute()
            v += (1.0 if step < n - 1 else 0.5) * dt * eng.f / m
        errors.append(abs(e + 0.5 * float(cp.sum(m * v * v)) - h0))
    ratio = errors[0] / errors[1]
    assert 2.0 < ratio < 8.0, f"energy error does not scale as dt^2: {errors}"
    print(f"leapfrog energy error  {errors[0]:.2e} -> {errors[1]:.2e} (x{ratio:.1f} for dt/2)   OK")


if __name__ == "__main__":
    test_forces()
    test_periodicity()
    test_integrator()
    print("all tests passed")
