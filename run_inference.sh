APP_DIRECTORY="experiments/$1"
CONFIGSTR="configs/$2"

export APP_SEED="2746318213"
export WANDB_RUN_ID="$1"
export OUTLINES_CACHE_DIR="experiments/$1"
export XDG_CACHE_HOME="experiments/$1"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Run the training
deepspeed --no_local_rank --num_gpus=1 --master_port=1920  \
         src/treetune/main.py --configs "$CONFIGSTR" \
          run_evaluation
