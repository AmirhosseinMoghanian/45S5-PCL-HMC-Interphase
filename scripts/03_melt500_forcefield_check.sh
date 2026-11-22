#!/usr/bin/env bash
# Force-field check: particle-free PCL melt at 500 K and 1 atm, once with the L-OPLS and once with
# the PLA-style ester dihedral table.  The production densities are the last column of the
# "production" stage in runs/melt500_<variant>/03_equilibration/log.csv and in the final line of
# 03_equilibration/equilibration.json; the manuscript compares them with the published melt density
# of Bačová et al., Open Research Europe 5:296 (2025).
#
# config_melt500_lopls.toml / config_melt500_pla.toml are self-contained: composite.neat = true
# deletes the nanoparticle after the chains have been grown, so no cutting stage is needed and the
# output directories come from [system].output_dir inside each file.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

for v in lopls pla; do
  "$PY" -X utf8 -u run_pipeline.py build equilibrate structure \
        --config "config_melt500_$v.toml" > "runs/melt500_$v.log" 2>&1
done
