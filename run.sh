APP_DIRECTORY="experiments/$1"
CONFIGSTR="configs/$2"

export APP_SEED="2746318213"
export WANDB_RUN_ID="$1"
export WANDB_API_KEY="35f62c8cac8a9711b806c204b72ed940e23fca82"
export HF_HOME="$SCRATCH/vine_run_sing"
# export OUTLINES_CACHE_DIR="experiments/$1"
# export XDG_CACHE_HOME="experiments/$1"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Run the training
deepspeed --no_local_rank --num_gpus=$NUM_GPUS --master_port=1252 \
         src/treetune/main.py --configs "$CONFIGSTR" \
            run_iteration_loop