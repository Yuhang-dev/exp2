#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source ./env.sh

STUDY_OUT=results/study
mkdir -p "$STUDY_OUT"
python -u check_kernel.py --out "$STUDY_OUT/kernel_check.json"
# Re-run the original generation baseline after correcting the score kernel.
python -u benchmark.py --out "$STUDY_OUT/baseline_native"
python -u prepare_study.py --samples 4 --seed 42 --out "$STUDY_OUT"

bash run_positions.sh
