#!/bin/bash
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_test
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=128Gb
#SBATCH --time=2:00:00

module load singularity

mkdir -p experiments
chmod a+x run_inference.sh
singularity exec --nv \
	-H "/network/scratch/a/arnaud.bergeron1/vine_run_sing" \
	-B "/network/scratch/a/arnaud.bergeron1/vine_run_sing" \
	treetune_v15.sif \
	./run_inference.sh full_ppo_vine_fix8 mixed_rewards_freeze/full_ppo.jsonnet