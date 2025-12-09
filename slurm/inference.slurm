#!/bin/bash

#SBATCH --job-name=llm_inference
#SBATCH --gpus-per-node=1
#SBATCH --output=jobs/infer/llm_infer_%j.out
#SBATCH --error=jobs/infer/llm_infer_%j.err

WORKDIR=/storage/xpfeiffer/thesis
cd $WORKDIR

ENV_FILE="./.env"
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
else
    echo "Warning: No .env file found in $EXP_DIR"
fi

source ../.envs/thesis/bin/activate

# keep a record of the files
cp inference.py ${EXPERIMENT}/inference.py
cp .env  ${EXPERIMENT}/

python3 inference.py

mv jobs/infer/llm_infer_${SLURM_JOB_ID}.out ${EXPERIMENT}/slurm/
mv jobs/infer/llm_infer_${SLURM_JOB_ID}.err ${EXPERIMENT}/slurm/
