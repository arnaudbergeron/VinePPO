#!/bin/bash
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_test
#SBATCH --output=job_output_test2.txt
#SBATCH --error=job_error_test2.txt
#SBATCH --ntasks=1
#SBATCH --mem=128Gb
#SBATCH --time=4:00:00

module load singularity

mkdir -p experiments
chmod a+x run_inference.sh
singularity exec --nv \
	-H "/network/scratch/a/arnaud.bergeron1/VinePPO" \
	-B "/network/scratch/a/arnaud.bergeron1/VinePPO" \
	--bind $HOME \
	treetune_v15.sif \
	./run_inference.sh llama_8b_grad_clip_mixed_constant_lr_max_grad_norm_0_5 mixed_rewards_vine/mixed_rewards_ap_1_bp_1_am_0_bm_1_llama-5e7.jsonnet
