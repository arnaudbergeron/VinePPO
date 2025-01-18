# from datasets import Dataset

# data_dir = '/network/scratch/a/arnaud.bergeron1/VinePPO/experiments/distillation_ap_11/mixed_rewards_ap_1_bp_1_am_0_bm_1/checkpoints/episodes__iter0000/w_actLogp'

# data = Dataset.load_from_disk(data_dir)

# critic_vals = data[2]['scores']
# advantages = data[-1]['query_token_ids']
# resp_ = data[0]['response_token_ids']
# print(critic_vals)
# print('adv')
# # print(advantages)
# print(f'len critic vals {len(critic_vals)}')
# print(f'len advantages {len(advantages)}')
# print(f'len resp_ {len(resp_)}')
# print(len(critic_vals[len(advantages) - 1 :]))


import torch
from transformers import AutoModelForCausalLM
from transformers import AutoConfig

# Clear any existing memory
torch.cuda.empty_cache()

# Load model with basic settings
print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Load model with explicit settings
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_position_embeddings=8192  # Use original context length
)

# Print final memory state
print(f"Final CUDA memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")