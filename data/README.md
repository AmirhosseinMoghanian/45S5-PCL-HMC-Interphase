# Data directory

Everything needed as *input* to the simulations is distributed with this repository.
The *output* trajectories are not (see [`runs/README.md`](../runs/README.md) and the
Data availability section of the main README).

## `final_45S5_bulk.xyz` — the starting structure of the nanoparticle

The melt-quenched bulk 45S5 glass from the companion study: 11,340 atoms
(Si 1844, P 208, Na 1952, Ca 1076, O 6260) in a cubic periodic box.
The second (comment) line carries the metadata the pipeline parses:

```
11340
45S5 bulk glass, N=11340, rho=2.673801, box=53.486637, seed=42
```

Required field: `box=<cubic box edge in Å>` (`rho=` and `seed=` are informational).
This is the file referenced by `system.bulk_glass` in `config.toml`.
It can be regenerated with `simulation_bulk_glass_mc.py`, although the released file is the
exact one used for every run of the manuscript; replacing it changes the nanoparticle.

## `pcl_source/` — the published PCL force-field files

Copied from the Zenodo supplement of Bačová, González Huarte, Harmandaris & Molina,
*Open Research Europe* **5**:296 (2025), DOI [10.12688/openreseurope.21354.2](https://doi.org/10.12688/openreseurope.21354.2)
(Zenodo [10.5281/zenodo.16944033](https://doi.org/10.5281/zenodo.16944033); the copies come from
`30mer/prepare`, `100mer/solvated` and `100mer/vacuum`, byte for byte).
They are read by `composite/pcl.py`, so no force-field constant is typed by hand:

| file | content |
|---|---|
| `pcl_ff_final_vacuum.itp` | the force-field file of the manuscript runs: atom types, charges, Lennard-Jones and bonded terms, with the ester dihedrals of L-OPLS (`ester_variant = "lopls_ester"`) |
| `pcl_ff_final.itp` | the same file except for six ester dihedral types, which were taken from the PLA force field (`ester_variant = "pla_ester"`); both were checked against the published melt density at 500 K |
| `30PCL_pairs.itp` | topology of one 30-unit chain (543 atoms), used by `tests/test_pcl_topology.py` to verify the generator |
| `3mer.gro` | the authors' reference coordinates of a three-unit chain (kept for context, not read by the code) |
| `*.c` | the authors' programs that regenerate the files above for another number of units |

These files remain the intellectual property of their authors and are redistributed here only
to make the model reproducible; the MIT license of this repository covers the code written for
this study, not the third-party force-field files.
