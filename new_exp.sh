#!/bin/bash

EXP_PATH="$1"
EXP_ID=$(printf "%03d" $(($(ls -1 "$EXP_PATH" | wc -l) + 1)))
EXP_DIR="${EXP_PATH}/exp_${EXP_ID}"

mkdir -p "$EXP_DIR"/{logs,slurm,checkpoints,results}

echo "Created $EXP_DIR"
