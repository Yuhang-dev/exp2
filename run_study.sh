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

# A separate process per length releases model and compiler allocations.
for length in 4096 8192 16384 32768 65536 131072; do
    python -u study.py --inputs "$STUDY_OUT/streams.pt" --length "$length" \
        --rope yarn4 --out "$STUDY_OUT/yarn4_$length"
    python study_report.py "$STUDY_OUT"
done

# Same 32K inputs and execution path: isolate the effect of YaRN itself.
python -u study.py --inputs "$STUDY_OUT/streams.pt" --length 32768 \
    --rope native --out "$STUDY_OUT/native_32768"
python study_report.py "$STUDY_OUT"
