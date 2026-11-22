"""The PCL topology generator reproduces the authors' 30-unit topology exactly (Zenodo 10.5281/zenodo.16944033)."""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from composite import pcl  # noqa: E402

TOP = ROOT / "data" / "pcl_source" / "30PCL_pairs.itp"


def read_top(path):
    sec, rows = None, {}
    for raw in Path(path).read_text().splitlines():
        m = re.match(r"\s*\[\s*(\w+)\s*\]", raw)
        if m:
            sec = m.group(1)
            rows.setdefault(sec, [])
            continue
        line = raw.split(";")[0].split()
        if sec and line:
            rows[sec].append(line)
    return rows


def canon(t):
    return min(tuple(t), tuple(t)[::-1])


def main():
    z = read_top(TOP)
    n = 30
    kinds, bonds, angles, dihedrals, impropers = pcl.chain_topology(n)
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        print(("ok    " if ok else "FAIL  ") + name + (" " + detail if detail else ""))
        failures += not ok

    # atoms: type name and charge in the same order
    z_atoms = [(r[1], float(r[6])) for r in z["atoms"]]
    mine = [(pcl.KINDS[pcl.KIND_NAMES[k]][0], pcl.KINDS[pcl.KIND_NAMES[k]][1]) for k in kinds]
    check("atom count", len(mine) == len(z_atoms) == 543, f"{len(mine)} / {len(z_atoms)}")
    check("atom types and charges in order", mine == z_atoms)
    check("net charge zero", abs(sum(c for _, c in mine)) < 1e-9, f"{sum(c for _, c in mine):.6f}")
    z_mass = [float(r[7]) for r in z["atoms"]]
    check("masses", np.allclose(pcl.MASS[kinds], z_mass, atol=2e-3))
    # connectivity
    zb = {canon((int(r[0]) - 1, int(r[1]) - 1)) for r in z["bonds"]}
    check("bonds", {canon(b) for b in map(tuple, bonds)} == zb, f"{len(zb)}")
    za = {canon((int(r[0]) - 1, int(r[1]) - 1, int(r[2]) - 1)) for r in z["angles"]}
    check("angles", {canon(a) for a in map(tuple, angles)} == za, f"{len(za)}")
    proper = [r for r in z["dihedrals"] if r[4] == "3"]
    zd = {canon(tuple(int(x) - 1 for x in r[:4])) for r in proper}
    check("proper dihedrals", {canon(d) for d in map(tuple, dihedrals)} == zd, f"{len(zd)}")
    zi = [tuple(int(x) - 1 for x in r[:4]) for r in z["dihedrals"] if r[4] == "1"]
    check("impropers (same atom order)", sorted(map(tuple, impropers)) == sorted(zi), f"{len(zi)}")
    zp = {canon((int(r[0]) - 1, int(r[1]) - 1)) for r in z["pairs"]}
    mp = {canon((int(d[0]), int(d[3]))) for d in dihedrals}
    check("1-4 pairs", mp == zp, f"{len(zp)}")
    # parameters exist for every term
    bp, ap, tors, tpar = pcl.bonded_parameters(kinds, bonds, angles, dihedrals, impropers)
    check("bonded parameters found", len(bp) == len(bonds) and len(ap) == len(angles), f"{len(tors)} torsions kept")
    print(f"info  chain mass of 30 units from the topology masses: {pcl.chain_mass(30):.2f} g/mol (the paper's table gives 3426)")
    # the two copies of the force-field file differ only in the Ryckaert-Bellemans table
    diff = [k for k in pcl._RB_ALL["pla_ester"] if pcl._RB_ALL["pla_ester"][k] != pcl._RB_ALL["lopls_ester"].get(k)]
    print("info  dihedral types that differ between the two published files:", [" ".join(k) for k in diff])
    check("variants differ only in ester dihedrals", all("OS" in k for k in diff) and len(diff) == 6, f"{len(diff)} types")
    print("FAILED" if failures else "all checks passed")
    return failures


if __name__ == "__main__":
    sys.exit(main())
