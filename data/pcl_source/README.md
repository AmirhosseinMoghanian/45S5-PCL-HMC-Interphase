# Source of the PCL force field

Files copied from the Zenodo supplement of Bačová, González Huarte, Harmandaris, Molina,
Open Research Europe 5:296 (2025), DOI 10.12688/openreseurope.21354.2
(Zenodo 10.5281/zenodo.16944033, folder `30mer/prepare` and `100mer/solvated`):

| file | content |
|---|---|
| `pcl_ff_final.itp` | GROMACS force field: atom types (L-OPLS and OPLS-AA), bonded types, Ryckaert-Bellemans dihedrals |
| `30PCL_pairs.itp` | topology of one 30-unit chain (543 atoms) used to verify the topology generator |
| `3mer.gro` | coordinates of a three-unit chain |
| `*.c` | the authors' programs that generate chains with a chosen number of units |

The force-field file is identical in all chain-length folders that were compared (10-mer and 100-mer).
