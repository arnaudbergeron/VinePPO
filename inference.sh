#!/bin/bash
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_test
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=128Gb
#SBATCH --time=2:00:00

module --quiet purge
module load anaconda/3
module load cuda/11.7

conda activate vine

CONFIGSTR="configs/mixed_rewards/mixed_rewards1x5lr.jsonnet"
APP_DIRECTORY="$SCRATCH/vine/experiments/mixed_rewards1x5lr" #change in file

export APP_SEED="2746318213"

export WANDB_RUN_ID="mixed_rewards1x5lr" # Optional

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# # Run the evaluation
deepspeed --no_local_rank --num_gpus=1   \
         src/treetune/main.py --configs "$CONFIGSTR" \
         --main_process_port=2920 run_evaluation
