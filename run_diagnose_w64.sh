#!/usr/bin/env bash
# Convenience wrapper so you don't have to paste the long diagnose_w64.py
# command line. Activate the venv first (source myenv/bin/activate), then run:
#   ./run_diagnose_w64.sh
set -e
cd "$(dirname "$0")"
python diagnose_w64.py \
    --inference-file examples/drilling/inference.csv \
    --n-shot-files examples/drilling/nshot_drilling.csv examples/drilling/nshot_not_drilling.csv \
    --class-names drilling not_drilling \
    --data-columns BPOS DBTM FLWI HDTH HKLD ROP RPM SPPA WOB \
    --timestamp-column DATE_TIME \
    --label-column label
