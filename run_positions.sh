#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source ./env.sh

# Run after kernel checks, baseline and prepare_study.py have completed.
STUDY_OUT=results/study
for length in 4096 8192 16384 32768 65536 131072; do
    python -u study.py --inputs "$STUDY_OUT/streams.pt" --length "$length" \
        --rope yarn4 --out "$STUDY_OUT/yarn4_$length"
    python study_report.py "$STUDY_OUT"
done

python -u study.py --inputs "$STUDY_OUT/streams.pt" --length 32768 \
    --rope native --out "$STUDY_OUT/native_32768"
python study_report.py "$STUDY_OUT"
