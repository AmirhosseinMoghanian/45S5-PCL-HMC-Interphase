#!/usr/bin/env bash
# Post-processing of the three replicas: pooled profiles, the additional interface analyses and the
# replica-average comparison table.  Run after scripts/01_composite_replicas.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

R=(runs/composite_r1 runs/composite_r2 runs/composite_r3)

# radial density and group enrichment averaged over the replicas, the first peaks of the
# glass-polymer RDFs and the interphase read from the pooled profile (runs/pooled/)
"$PY" -X utf8 pool_replicas.py "${R[@]}" --neat runs/neat_r1 --out runs/pooled

# periodic-image bridging, chain reorientation over the trajectory and the sensitivity of the
# interphase thickness to its thresholds (runs/extra_analysis/)
"$PY" -X utf8 extra_analysis.py "${R[@]}" --neat runs/neat_r1 --pooled runs/pooled --out runs/extra_analysis

# replica mean ± SEM of every structural quantity against the particle-free control
# (runs/comparison.json and runs/comparison.md)
"$PY" -X utf8 compare_runs.py --replicas "${R[@]}" --compare runs/neat_r1
