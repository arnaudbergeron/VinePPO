#!/bin/bash
#SBATCH --gpus-per-task=a100l:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_train
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=128Gb
#SBATCH --time=2:59:00

module --quiet purge
module load anaconda/3
module load cuda/11.7

conda activate vine

CONFIGSTR="configs/mixed_rewards/mixed_rewards_sppo_clamp_high_only.jsonnet"
APP_DIRECTORY="$SCRATCH/vine/experiments/mixed_rewards_sppo_clamp_high_only_more1" #change in file

export APP_SEED="2746318213"

export WANDB_RUN_ID="mixed_rewards_sppo_clamp_high_only_more1" # Optional

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# # Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  --master_port=2420  \
         src/treetune/main.py --configs "$CONFIGSTR" \
         run_iteration_loop
