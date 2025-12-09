#!/bin/bash

#SBATCH --job-name=llm_grpo
#SBATCH --gpus-per-node=2
#SBATCH --output=jobs/grpo/llm_grpo_%j.out
#SBATCH --error=jobs/grpo/llm_grpo_%j.err

# EXPORT RANK=0
# EXPORT LOCAL_RANK=0

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
cp qwen_grpo.py ${EXPERIMENT}/qwen_grpo.py
cp DBLPDataset.py ${EXPERIMENT}/DBLPDataset.py
cp .env  ${EXPERIMENT}/

python3 qwen_grpo.py

mv jobs/grpo/llm_grpo_${SLURM_JOB_ID}.out ${EXPERIMENT}/slurm/
mv jobs/grpo/llm_grpo_${SLURM_JOB_ID}.err ${EXPERIMENT}/slurm/
