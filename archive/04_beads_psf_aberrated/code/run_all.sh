#!/bin/bash
# Run every configuration in config/ sequentially into log/<name>/.
set -u
PY=${PY:-/workspace/.conda_envs/neural-graph-linux/bin/python}
DEVICE=${DEVICE:-cuda:1}
mkdir -p log
for c in config/*.yaml; do
  name=$(basename "$c" .yaml)
  echo "=============== $name ==============="
  $PY simulate.py -c "$c" --device "$DEVICE" 2>&1 | grep -v "it/s\]$"
done
echo "ALL DONE"
