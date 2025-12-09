#!/bin/bash

#SBATCH --job-name=llm_sft_full
#SBATCH --gpus-per-node=1
#SBATCH --output=jobs/sft/llm_sft_full_%j.out
#SBATCH --error=jobs/sft/llm_sft_full_%j.err

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
cp qwen_sft.py ${EXPERIMENT}/qwen_sft.py
cp DBLPDataset.py ${EXPERIMENT}/DBLPDataset.py
cp .env  ${EXPERIMENT}/

python3 qwen_sft.py

mv jobs/sft/llm_sft_full_${SLURM_JOB_ID}.out ${EXPERIMENT}/slurm/
mv jobs/sft/llm_sft_full_${SLURM_JOB_ID}.err ${EXPERIMENT}/slurm/
