# `runs/` — Monte Carlo outputs (not distributed)

This directory is created by `run_pipeline.py`; it holds one subdirectory per run.
The trajectories are **not** part of this repository: `03_equilibration/trajectory.xyz` alone
is 150 MB for a composite run, 30 MB for a sensitivity run and 7 MB for a melt check.
They are available from the corresponding author on reasonable request, and they can be
regenerated with the commands below (or with the ready-to-run scripts in [`scripts/`](../scripts/)).

## The runs of the manuscript

| directory | content | command |
|---|---|---|
| `composite_r1` | composite, replica 1 | `python run_pipeline.py cut build equilibrate structure` (base `config.toml`, seed 42) |
| `composite_r2` | composite, replica 2 | `python run_pipeline.py build equilibrate structure --set system.output_dir="runs/composite_r2" --set system.replica=2` |
| `composite_r3` | composite, replica 3 | same with `runs/composite_r3` and `--set system.replica=3` |
| `neat_r1` | particle-free PCL control | `python run_pipeline.py build equilibrate structure --set system.output_dir="runs/neat_r1" --set composite.neat=true` |
| `sensitivity_eps0.8` | glass–polymer ε × 0.8 | `python run_pipeline.py cut build equilibrate structure --profile sensitivity_eps0.8` |
| `sensitivity_eps1.2` | glass–polymer ε × 1.2 | `python run_pipeline.py cut build equilibrate structure --profile sensitivity_eps1.2` |
| `melt500_lopls` | melt-density check at 500 K, L-OPLS ester table | `python run_pipeline.py build equilibrate structure --config config_melt500_lopls.toml` |
| `melt500_pla` | melt-density check at 500 K, PLA ester table | `python run_pipeline.py build equilibrate structure --config config_melt500_pla.toml` |
| `pooled`, `extra_analysis`, `comparison.*` | replica pooling and comparison | see [`scripts/04_pool_and_compare.sh`](../scripts/04_pool_and_compare.sh) |

Replicas 2 and 3 and the control reuse the nanoparticle of replica 1, so their
`01_nanoparticle/` directory is copied over instead of re-cutting the sphere
([`scripts/01_composite_replicas.sh`](../scripts/01_composite_replicas.sh) does this). The
sensitivity runs re-cut and re-grow instead: both stages are deterministic in the seed, so they
rebuild the particle and the chains of replica 1, and `mc.start_from` in the profile then takes the
coordinates and box of the equilibrated replica 1. `--resume` continues an interrupted
equilibration from its checkpoint and `--from-stage NAME` restarts the schedule at a given stage.

One composite run is 30 000 hybrid-MC moves (23.5 – 33.3 h on an NVIDIA GeForce RTX 3050); the
sensitivity runs are 5 000 moves and the melt checks 4 150 moves.

## Layout of one run

```
runs/composite_r1/
├── config_used.json                  # the configuration after profiles and --set overrides
├── 01_nanoparticle/
│   ├── nanoparticle.xyz              # the cut sphere (2638 atoms)
│   └── nanoparticle.json             # radius, composition, charge neutralisation, 4-fold Si/P
├── 02_composite/
│   ├── composite_initial.xyz         # chains grown around the particle
│   ├── composite.json                # atom counts, units per chain, box, weight fractions
│   └── system.npz                    # engine state at the start of the sampling
├── 03_equilibration/
│   ├── log.csv                       # one row per move: stage, T, energy, density, box, acceptance, dt
│   ├── equilibration.json            # per-stage statistics and convergence tests
│   ├── trajectory.xyz                # frames every trajectory_every moves
│   ├── checkpoint.npz                # coordinates, box, stage and move counters, tuned dt and dlnV
│   ├── composite_equilibrated.xyz
│   └── system_final.npz
├── 04_analysis/                      # optional energy analysis (python run_pipeline.py analyse)
└── 05_structure/
    ├── structure_summary.json        # all structural quantities of the run
    ├── structure_report.md
    └── data/*.csv                    # radial density, RDF, orientation, torsions, trains/loops/tails, …
```

The production commands use the stages `cut build equilibrate structure`, so a run of this study
contains `01`, `02`, `03` and `05` plus `config_used.json`.
`pool_replicas.py` and `compare_runs.py` read `05_structure/` (the summary and the profile CSVs);
`extra_analysis.py` additionally reads `01_nanoparticle/nanoparticle.json`,
`03_equilibration/system_final.npz` and `03_equilibration/trajectory.xyz`. All three take the run
directories as command-line arguments, so a renamed run only has to be renamed in the call.
