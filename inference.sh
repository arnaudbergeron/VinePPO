#!/bin/bash
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_test
#SBATCH --output=job_output_test.txt
#SBATCH --error=job_error_test.txt
#SBATCH --ntasks=1
#SBATCH --mem=128Gb
#SBATCH --time=2:00:00

module load singularity

mkdir -p experiments
chmod a+x run_inference.sh
singularity exec --nv \
	-H "/network/scratch/a/arnaud.bergeron1/VinePPO" \
	-B "/network/scratch/a/arnaud.bergeron1/VinePPO" \
	treetune_v15.sif \
	./run_inference.sh mixed_rewards_ap_1_bp_1_am_0_bm_1_5 mixed_rewards_vine/mixed_rewards_ap_1_bp_1_am_0_bm_1.jsonnet
