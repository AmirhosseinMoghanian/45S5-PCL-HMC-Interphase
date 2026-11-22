# Interphase in a PCL / 45S5 bioactive-glass nanocomposite: GPU-accelerated hybrid Monte Carlo

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%20%2B%20CuPy-76b919.svg)](https://cupy.dev/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Source code accompanying the manuscript **"Atomistic Modelling of the Interphase in a
Poly(ε-caprolactone)/45S5 Bioactive Glass Nanocomposite by GPU-Accelerated Hybrid Monte Carlo"**.

An all-atom model of the interface: a rigid 45S5 nanoparticle at the centre of a periodic cell,
filled with poly(ε-caprolactone) chains grown around it, melted, cooled and sampled with hybrid
Monte Carlo in the NVT and NPT ensembles. The repository contains the simulation and the analysis
code of the study; everything the code writes is a number (JSON, CSV and markdown summaries) —
there is no plotting code in this release, and the tables and figures of the manuscript are not
part of it:

1. **Force field and topology** ([`composite/pcl.py`](composite/pcl.py),
   [`composite/forcefield.py`](composite/forcefield.py)) — the published all-atom PCL parameters are
   parsed from the authors' GROMACS files in [`data/pcl_source/`](data/pcl_source), nothing is typed
   by hand; glass and interface terms come from the Teter charges, UFF Lennard-Jones and damped
   shifted force Coulomb.
2. **System construction** ([`composite/nanoparticle.py`](composite/nanoparticle.py),
   [`composite/polymer.py`](composite/polymer.py), [`composite/packing.py`](composite/packing.py)) —
   the particle is cut from a melt-quenched bulk glass, the chains are grown as clash-free random
   coils in the periodic cell.
3. **Hybrid Monte Carlo** ([`composite/gpu_engine.py`](composite/gpu_engine.py),
   [`composite/montecarlo.py`](composite/montecarlo.py)) — leapfrog trajectories inside every MC
   move, NPT volume moves, energy/force kernels as CUDA source compiled at start-up.
4. **Structure and interphase analysis** ([`composite/structure.py`](composite/structure.py),
   [`pool_replicas.py`](pool_replicas.py), [`extra_analysis.py`](extra_analysis.py),
   [`compare_runs.py`](compare_runs.py)) — radial density, glass–polymer RDFs, orientation, torsions,
   trains/loops/tails, free volume, replica pooling, autocorrelation and blocking errors.

[`simulation_bulk_glass_mc.py`](simulation_bulk_glass_mc.py) is the melt-quench engine of the
companion bioactive-glass study that produced the bulk structure the nanoparticle is cut from.

---

## Repository structure

```
.
├── run_pipeline.py                # cut -> build -> equilibrate -> analyse -> structure
├── config.toml                    # the run of the manuscript (base configuration)
├── config_melt500_lopls.toml      # particle-free PCL melt at 500 K, L-OPLS ester dihedrals
├── config_melt500_pla.toml        # the same melt with the alternative published ester table
├── profiles/                      # --profile overlays (interface-strength sensitivity runs)
├── composite/                     # the package: force field, building, GPU engine, analysis
│   ├── pcl.py                     # reads data/pcl_source, RB -> OPLS Fourier conversion
│   ├── forcefield.py              # atom types, charges, LJ, NBFIX, bonded parameters
│   ├── nanoparticle.py            # stage 1: cut a neutral, tetrahedra-complete sphere
│   ├── polymer.py, packing.py     # stage 2: chain growth and 70:30 filling of the cell
│   ├── gpu_engine.py              # CUDA energy/force kernels, mixed precision, neighbour lists
│   ├── montecarlo.py              # stage 3: hybrid MC in NVT / NPT, schedule, checkpoints
│   ├── structure.py               # stage 5: interphase, RDFs, conformation, free volume
│   └── io.py, analysis.py, ...    # system container, XYZ/NPZ I/O, energy analysis
├── data/                          # inputs: bulk 45S5 glass and the published PCL files
├── runs/                          # created by the pipeline; outputs are not distributed
├── scripts/                       # the reproduction commands, in order (01 … 04)
├── tests/                         # engine, topology and builder checks (test_engine.py needs a GPU)
├── requirements.txt
├── LICENSE                        # MIT
└── README.md
```

## Method summary

### The model

* **Nanoparticle.** A sphere of nominal radius 20 Å cut from the melt-quenched 45S5 bulk
  (`data/final_45S5_bulk.xyz`, 11 340 atoms, 53.487 Å cell). The cut keeps whole SiO₄/PO₄
  tetrahedra, neutralises the particle by adjusting 75 modifier ions, and picks the best of 400
  random centres. Result: 2 638 atoms, R<sub>eq</sub> = 20.44 Å, R<sub>max</sub> = 21.50 Å, net
  charge 0 e. The particle is a rigid body: its atoms act on the polymer but move with the cell.
* **Polymer.** 39 chains of 29–30 caprolactone units (≈ 3.4 kg/mol), grown as random coils with the
  force-field bond lengths and angles and with torsions drawn from the Ryckaert–Bellemans tables of
  the published model. Polymer : glass = 70 : 30 by weight (14.4 vol% glass), 23 761 atoms in all.
* **Non-bonded terms.** Lennard-Jones truncated at 10 Å with long-range dispersion correction in the
  volume moves, damped shifted force Coulomb (α = 0.20 Å⁻¹) with the LAMMPS
  `lj/cut/coul/dsf` exclusion and 1-4 scaling convention, geometric combination rules, and a
  pair-specific override (NBFIX) for the hydroxyl hydrogens next to network oxygens.
  Glass–polymer ε can be scaled (`forcefield.cross_epsilon_scale`) for the sensitivity study.
* **Bonded terms.** Harmonic bonds and angles, OPLS Fourier torsions, built from the authors'
  topology file; the generator reproduces their 30-mer atom by atom
  ([`tests/test_pcl_topology.py`](tests/test_pcl_topology.py)).

### The sampling

Hybrid Monte Carlo: every MC move integrates a leapfrog trajectory (mean 200 steps, randomised ±30 %)
with a fictitious mass of 12 amu for all atoms, then accepts it with the Metropolis criterion. The
time step is tuned towards a target acceptance of 0.70. NPT moves add a volume proposal on the ln V
scale, accepted as

```
P_acc = min(1, exp[ -(ΔU + P ΔV)/kT + (N_c + 1) ln(V'/V) ])
```

with 10 volume attempts per hybrid move. `N_c` is the number of mobile chains (39): the rigid
particle scales with the cell but is not counted in the ideal-gas term. A volume proposal that would
make the cell smaller than twice the interaction range is rejected without evaluating the energy.

The production schedule of `config.toml` is 30 000 hybrid moves:

| stage | ensemble | T (K) | moves | purpose |
|---|---|---|---|---|
| `relax` | NVT | 600 | 50 | remove the overlap of the build |
| `compress` | ramp | 600 | 450 | shrink towards the expected density |
| `melt` | NPT | 600 | 1 500 | equilibrate the melt |
| `anneal` | NPT | 600 → 300 | 8 000 | slow cooling |
| `equilibrate` | NPT | 300 | 12 000 | sampling |
| `production` | NPT | 300 | 8 000 | sampling with frozen step size (`tune = false`) |

Energies and forces are evaluated by CUDA kernels through CuPy, in mixed precision (single-precision
pair terms, double-precision accumulation), from a cell-list neighbour list with a 2.0 Å skin that is
rebuilt whenever an atom has moved more than half the skin.

### The analysis

The `structure` stage computes, per run: the radial PCL density around the particle, group
enrichment, glass–polymer partial RDFs, the orientation of carbonyl and chain-end groups, backbone
torsion distributions, the trains/loops/tails of the adsorbed chains and the free-volume profile.
`pool_replicas.py` averages the profiles of the three replicas (with the standard error over
replicas) and reads the interphase thickness from the pooled density, which is far less noisy than
any single run. Errors in a single run come from the Sokal integrated autocorrelation time and from
Flyvbjerg–Petersen blocking; between replicas they come from the spread of the replicas themselves.

## Installation

Python ≥ 3.11 (`tomllib`), tested with 3.12. An NVIDIA GPU with a CUDA 11 or 12 driver is required
for the sampling stage and for the engine tests; the cut, build and most analysis steps run on CPU.

```bash
git clone https://github.com/AmirhosseinMoghanian/45S5-PCL-HMC-Interphase.git
cd 45S5-PCL-HMC-Interphase
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists the versions this release has been checked with (Python 3.12.13). CuPy must
match the CUDA version of your machine, so it is not pinned there — install one of:

```bash
pip install cupy-cuda12x         # CUDA 12
pip install cupy-cuda11x         # CUDA 11
```

## Data availability

The Monte Carlo outputs (trajectories, checkpoints, `.npz` states — about 150 MB per composite run)
are **not** included, and are available from the corresponding author upon reasonable request. Every
one of them can be regenerated with the commands in [`scripts/`](scripts/);
[`runs/README.md`](runs/README.md) documents the run directories and their layout.

The two inputs the code needs are shipped:

* `data/final_45S5_bulk.xyz` — the melt-quenched 45S5 glass the nanoparticle is cut from;
* `data/pcl_source/` — the GROMACS parameter files of Bačová, Gonzalez Huarte, Harmandaris and
  Molina, Open Research Europe 5:296 (2025), [DOI 10.12688/openreseurope.21354.2](https://doi.org/10.12688/openreseurope.21354.2),
  Zenodo deposit [10.5281/zenodo.16944033](https://doi.org/10.5281/zenodo.16944033). They remain the
  intellectual property of their authors and are redistributed here only so the parser can be
  re-run; see [`data/README.md`](data/README.md).

The randomly packed cell that the bulk-glass engine starts from belongs to the companion study and is
not part of this repository.

## Usage

### Check the installation

```bash
python tests/test_pcl_topology.py     # the generated 30-mer against the authors' topology file
python tests/test_pcl_builder.py      # geometry, clashes and dimensions of the built chains
python tests/test_engine.py           # forces, precision, symplecticity (needs a GPU)
```

### One composite run

```bash
python run_pipeline.py                          # all stages into runs/composite_r1
python run_pipeline.py cut build                # only prepare the system
python run_pipeline.py equilibrate --resume     # continue an interrupted sampling from its checkpoint
python run_pipeline.py equilibrate --from-stage anneal
python run_pipeline.py --set nanoparticle.radius=15 --set mc.stage.4.moves=2000
```

Stage names are `cut build equilibrate analyse structure`; `--config`, `--profile` and `--set`
combine, and the configuration actually used is written to `runs/<run>/config_used.json`.

### The runs of the manuscript

```bash
bash scripts/01_composite_replicas.sh          # 3 replicas + particle-free control
bash scripts/02_sensitivity_runs.sh            # glass-polymer epsilon x 0.8 and x 1.2
bash scripts/03_melt500_forcefield_check.sh    # melt density at 500 K, both ester tables
bash scripts/04_pool_and_compare.sh            # pooling, extra analyses, comparison
```

Each composite or control run is 30 000 hybrid moves and took 23.6 – 33.3 h on an NVIDIA RTX 3050
(0.13 – 0.37 moves/s); a sensitivity run is 5 000 moves (2.4 – 3.9 h). The runs are
independent of each other and can be executed in parallel — with one exception, the sensitivity runs
restart from the equilibrated replica 1.

### Bulk glass (companion study)

```bash
python simulation_bulk_glass_mc.py <randomly packed 45S5 cell>.xyz --seed 42 --density 2.674
# --final-sweeps 20000  --cutoff 12.0  --wolf-alpha 0.20  --continue
```

## Results at a glance

These are the run outputs themselves — `05_structure/structure_summary.json` of each replica,
`runs/pooled/pooled_summary.json` and `runs/extra_analysis/` from `scripts/04`, and the sensitivity
comparison of the two `runs/sensitivity_*` directories:

| quantity | value |
|---|---|
| composite density at 300 K, 1 atm | 1.291 ± 0.001 g/cm³ (three replicas) |
| interphase thickness | 9.7 – 10.4 Å |
| first PCL layer | 1.46 × the bulk density, peak at 2.44 Å from the particle surface |
| polymer far from the particle | 1.071 g/cm³, 0.2 % from particle-free PCL (1.069 g/cm³) |
| adsorption | Ca–O(carbonyl) 2.73 Å (g = 2.44), Na–O(carbonyl) 2.43 Å (g = 1.47), NBO–H(hydroxyl) 1.68 Å (g = 5.6); no ester-oxygen peak above g = 0.2 |
| chain conformation | chain-end groups 3.1 × bulk abundance in the first layer, 0.76 of chains with at least one contact, radius of gyration unchanged |
| ±20 % glass–polymer ε | first-layer peak 1.51 / 1.55 instead of 1.47; every other quantity within 1.2 × the replica scatter |

## Validation

What the four scripts in [`tests/`](tests) print, and the melt-density check of
`config_melt500_*.toml`:

* analytic forces against central differences: 6 · 10⁻⁷ maximum relative error; mixed against double
  precision 5 · 10⁻⁷; a chain moved onto another lattice site changes the energy by 2 · 10⁻¹² kcal/mol;
  the leapfrog energy error falls by a factor 3.9 when the step is halved (4 for O(dt²)).
* the generated 30-unit chain is identical to the authors' topology: 543 atoms, 542 bonds, 1 021
  angles, 1 410 proper dihedrals, 30 impropers, 1 410 1-4 pairs.
* the builder keeps every bond length at its force-field value (0 Å deviation), the mean angle
  deviation is 0.87°, no non-bonded pair is closer than 1.6 Å, and the freshly grown coils have
  ⟨R<sub>e</sub>²⟩<sup>1/2</sup> = 53.2 Å.
* the particle-free melt at 500 K, 1 atm gives 0.905 ± 0.001 g/cm³ with the L-OPLS ester table and
  0.895 ± 0.001 g/cm³ with the alternative published ester table, which is the comparison the
  force-field check in the manuscript is based on.

## Reproducibility notes

* Every random number in the pipeline comes from `system.seed` (42) and `system.replica`: the cut from
  the seed alone (so all replicas share the very same nanoparticle), the chain growth from
  `seed + 1 + 104729 · replica`, the sampling from `seed + 104729 · replica + 7919 · stage + move`.
  Replicas therefore differ in chain conformations and in the sampled path, not in the particle, the
  composition or the target box.
* The cut and the build are deterministic, so a sensitivity run re-creates the identical particle and
  chains; `mc.start_from` in the profile then takes the coordinates and box of the equilibrated
  replica 1.
* `checkpoint.npz` stores the coordinates, the box, the stage and move counters and the tuned `dt` and
  dln*V*, and is renamed atomically, so an interrupted run continues from its last checkpoint. The
  random stream is seeded at the start of each stage from the seed, the replica, the stage index and
  the move counter, so a resumed stage continues statistically but not bit for bit. The production
  stage has `tune = false`, so its step sizes stay fixed and detailed balance is not disturbed by
  adaptation.
* Mixed precision is used for production and double precision only for the validation tests; the
  difference is 5 · 10⁻⁷ in the forces. Last-digit differences between GPU generations and CUDA
  versions are possible and do not affect the reported statistics.
* Statistical quantities are reported with two independent error estimates (autocorrelation and
  blocking) and, for the replicas, with the standard error over the replicas.

## Citation

If you use this code, please cite the accompanying manuscript:

> *Atomistic Modelling of the Interphase in a Poly(ε-caprolactone)/45S5 Bioactive Glass Nanocomposite
> by GPU-Accelerated Hybrid Monte Carlo* — citation details will be updated upon publication.

Please also cite the source of the PCL force field:

> Bačová, P., Gonzalez Huarte, E., Harmandaris, V. & Molina, A., *Open Research Europe* **5**:296
> (2025), DOI 10.12688/openreseurope.21354.2.

## License

This project is released under the [MIT License](LICENSE). The GROMACS parameter files under
`data/pcl_source/` are third-party force-field data and remain subject to their authors' rights; see
[`data/README.md`](data/README.md).
