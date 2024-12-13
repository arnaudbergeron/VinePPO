#!/bin/bash
#SBATCH --gpus-per-task=a100l:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_train
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=128Gb
#SBATCH --time=12:59:00

module load singularity

mkdir -p experiments
chmod a+x run.sh
singularity exec --nv \
	-H "/network/scratch/a/arnaud.bergeron1/vine_run_sing" \
	-B "/network/scratch/a/arnaud.bergeron1/vine_run_sing" \
	treetune_v15.sif \
	./run.sh ppo_freeze_scale_loss_abs mixed_rewards_freeze/full_ppo.jsonnet