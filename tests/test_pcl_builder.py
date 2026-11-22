"""PCL chain growth: geometry against the force field and a clash check on a single chain."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from composite import forcefield as ff, pcl, polymer  # noqa: E402


def main():
    rng = np.random.default_rng(1)
    n = 30
    kinds, bonds, angles, dihedrals, impropers, _ = polymer.chain_topology(n)
    def accept(new, chain):                     # self-avoidance as in composite/packing.py (no glass, no periodicity)
        own = np.delete(chain, polymer.ATTACH, axis=0) if len(chain) >= -polymer.ATTACH else chain
        if len(own) == 0:
            return True
        d2 = ((new[:, None, :] - own[None, :, :]) ** 2).sum(-1)
        limit = np.full(len(own), 2.0 ** 2)
        limit[-polymer.JUNCTION:] = 1.5 ** 2
        return not (d2 < limit).any()
    xyz = polymer.grow_chain(n, np.zeros(3), rng, accept)
    assert xyz is not None and len(xyz) == len(kinds) == 543, (None if xyz is None else len(xyz))
    cls = [ff.BCLASS[t] for t in kinds]
    # bond lengths
    r0 = np.array([pcl.lookup(pcl.BONDS, (cls[i], cls[j]))[1] for i, j in bonds])
    r = np.linalg.norm(xyz[bonds[:, 0]] - xyz[bonds[:, 1]], axis=1)
    print(f"bonds     : max |r - r0| = {np.abs(r - r0).max():.3f} A over {len(r)}")
    ok = np.abs(r - r0).max() < 0.05
    # angles
    th0 = np.array([pcl.lookup(pcl.ANGLES, (cls[i], cls[j], cls[k]))[1] for i, j, k in angles])
    v1 = xyz[angles[:, 0]] - xyz[angles[:, 1]]
    v2 = xyz[angles[:, 2]] - xyz[angles[:, 1]]
    th = np.degrees(np.arccos(np.sum(v1 * v2, 1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)))
    err = np.abs(th - th0)
    print(f"angles    : mean |dtheta| = {err.mean():.2f} deg, max = {err.max():.2f} deg over {len(th)}")
    ok &= err.max() < 20.0 and err.mean() < 3.0
    # no non-bonded overlaps closer than 1-4 distance
    from scipy.spatial import cKDTree
    pairs = cKDTree(xyz).query_pairs(1.6, output_type="ndarray")
    bonded = {tuple(sorted(b)) for b in map(tuple, bonds)}
    near13 = {tuple(sorted((a[0], a[2]))) for a in map(tuple, angles)}
    bad = [p for p in map(tuple, pairs) if tuple(sorted(p)) not in bonded | near13]
    print(f"clashes   : {len(bad)} non-bonded pairs closer than 1.6 A")
    ok &= len(bad) == 0
    # end-to-end distance of an ideal chain of this length is about 30 A (30 units)
    ree = np.linalg.norm(xyz[1] - xyz[-2])
    print(f"Ree (acid O ... hydroxyl O) = {ree:.1f} A")
    # many chains: failure rate and statistics
    ree_all = []
    for s in range(40):
        c = polymer.grow_chain(n, np.zeros(3), np.random.default_rng(100 + s), accept)
        assert c is not None
        ree_all.append(np.linalg.norm(c[1] - c[-2]))
    print(f"40 chains : rms Ree = {np.sqrt(np.mean(np.square(ree_all))):.1f} A")
    print("passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
