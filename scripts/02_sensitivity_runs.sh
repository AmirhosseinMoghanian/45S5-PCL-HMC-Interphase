#!/usr/bin/env bash
# Interface-strength sensitivity: the glass-polymer Lennard-Jones epsilon multiplied by 0.8 and by
# 1.2 (forcefield.cross_epsilon_scale), with the same particle, the same chains and the same
# starting configuration as replica 1.  Each run is 5 000 moves: reheat to 400 K (1 000),
# anneal 400 -> 300 K (2 000), production at 300 K (2 000).
#
# profiles/sensitivity_eps*.toml set
#   system.output_dir            the run directory
#   forcefield.cross_epsilon_scale
#   mc.start_from = "runs/composite_r1/03_equilibration/system_final.npz"
# The cut and build stages are deterministic in the seed, so they rebuild the very particle and the
# very chains of replica 1 in the new directory; the sampling then starts from the coordinates and
# box of replica 1's equilibrated state instead of the compressed build.
# Run scripts/01_composite_replicas.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

for k in 0.8 1.2; do
  "$PY" -X utf8 -u run_pipeline.py cut build equilibrate structure --profile "sensitivity_eps$k" \
        > "runs/sensitivity_eps$k.log" 2>&1
done
