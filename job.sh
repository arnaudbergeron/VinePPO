#!/bin/bash
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gsm_rho_ppo
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=128Gb
#SBATCH --time=1:59:00

module --quiet purge
module load anaconda/3
module load cuda/11.7

conda activate vine

CONFIGSTR="configs/polIter_rho1bSft2_ppo_GSM8K.jsonnet"
APP_DIRECTORY="$SCRATCH/vine/experiments/sppo_baseline_128epi_batch_quarter_lr" #change in file

export APP_SEED="2746318213"

export WANDB_RUN_ID="sppo_baseline_128epi_batch_quarter_lr" # Optional

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# # Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS  --master_port=2420  \
         src/treetune/main.py --configs "$CONFIGSTR" \
         run_iteration_loop

# # Run the evaluation
# deepspeed --no_local_rank --num_gpus=1   \
#          src/treetune/main.py --configs "$CONFIGSTR" \
#          --main_process_port=2920 run_evaluation
