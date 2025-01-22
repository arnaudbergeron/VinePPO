#!/bin/bash
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_train
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=64Gb
#SBATCH --time=00:20:00

module load singularity

mkdir -p experiments
chmod a+x run.sh
singularity exec --nv \
	-H "/network/scratch/a/arnaud.bergeron1/VinePPO" \
	-B "/network/scratch/a/arnaud.bergeron1/VinePPO" \
	--bind $HOME \
	treetune_v15.sif \
	./run.sh llama_grad_clip_ap_1_16 mixed_rewards_vine/mixed_rewards_ap_1_bp_1_am_0_bm_1_llama-8e7.jsonnet