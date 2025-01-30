#!/bin/bash
#SBATCH --gpus-per-task=a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gsm_train
#SBATCH --output=job_output2.txt
#SBATCH --error=job_error2.txt
#SBATCH --ntasks=1
#SBATCH --mem=256Gb
#SBATCH --time=14:30:00

module load singularity

mkdir -p experiments
chmod a+x run.sh

singularity exec --nv \
	-H "/network/scratch/a/arnaud.bergeron1/VinePPO" \
	-B "/network/scratch/a/arnaud.bergeron1/VinePPO" \
	--bind $HOME \
	treetune_v15.sif \
	./run.sh llama_8b_grad_clip_mixed_constant_lr_1 mixed_rewards_vine/mixed_rewards_ap_1_bp_1_am_0_bm_1_llama-5e7.jsonnet

# /home/mila/a/arnaud.bergeron1/.conda/envs/vine_new/bin/python src/test.py