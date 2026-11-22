#!/usr/bin/env bash
# The three composite replicas and the particle-free control of the manuscript.
# One run is 30 000 hybrid-MC moves: 23.5 - 33.3 h on an NVIDIA GeForce RTX 3050, so these
# commands are normally left running in the background (nohup) or split with --resume.
#
# Set PYTHON to the interpreter of the environment that holds the CuPy wheel matching your CUDA
# toolkit; run_pipeline.py resumes an interrupted stage with --resume and restarts the schedule
# at a given stage with --from-stage.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

# ---- replica 1: cut -> build -> equilibrate -> structure (base config.toml, seed 42)
"$PY" -u run_pipeline.py cut build equilibrate structure > runs/composite_r1.log 2>&1

# The other runs use exactly the same nanoparticle, so the sphere of replica 1 is copied
# instead of being cut again from the bulk glass.
for tag in composite_r2 composite_r3 neat_r1; do
  mkdir -p "runs/$tag/01_nanoparticle"
  cp runs/composite_r1/01_nanoparticle/* "runs/$tag/01_nanoparticle/"
done

# ---- replicas 2 and 3: different seeds for chain growth and for Monte Carlo
for rep in 2 3; do
  "$PY" -u run_pipeline.py build equilibrate structure \
        --set "system.output_dir=\"runs/composite_r$rep\"" --set "system.replica=$rep" \
        > "runs/composite_r$rep.log" 2>&1
done

# ---- particle-free PCL with the same number and length of chains
"$PY" -u run_pipeline.py build equilibrate structure \
      --set 'system.output_dir="runs/neat_r1"' --set "composite.neat=true" \
      > runs/neat_r1.log 2>&1
