# import os
# import json
# tree_path = '/network/scratch/a/arnaud.bergeron1/VinePPO/experiments/llama_8b_grad_clip_ap_1_28/mixed_rewards_ap_1_bp_1_am_0_bm_1_llama-5e7/evaluation/ckpt--iter_0050--epoch_1.00--step_3200/gsm8k_test/inference_results/test__0__1/trees'

# for tree in os.listdir(tree_path):
#     with open(f'{tree_path}/tree', 'r') as file:
#         data = json.load(file)

    
from datasets import DatasetDict, Dataset
import ast
import json
ds = Dataset.load_from_disk('/network/scratch/a/arnaud.bergeron1/VinePPO/experiments/llama_8b_grad_clip_ap_1_28/mixed_rewards_ap_1_bp_1_am_0_bm_1_llama-5e7/evaluation/ckpt--iter_0050--epoch_1.00--step_3200/gsm8k_test/inference_results/test__0__1')
output_json = []
num_good = 0
num_tot = 0
for data in ds:
    tree = data['_treetune__reasoning_tree']
    tree = json.loads(str(tree))
    good_ans = data['answer']
    for child in tree['children']:
        num_tot += 1
        output = {'question': data['question']}
        output['answer'] = data['answer']
        output['generated_answer'] = child['text']
        output['extracted_answer'] = child['answer']
        output_json.append(output)
        try:
            if float(good_ans) == float(child['answer'].replace(',', '').replace('$','')):
                num_good+=1
            else:

                print(good_ans, child['answer'])
        except:
            print(good_ans, child['answer'])

print(num_good/num_tot)


# #save output as json
# with open('/network/scratch/a/arnaud.bergeron1/VinePPO/mixed_rewards_outputs.jsonl', 'w') as file:
#     json.dump(output_json, file)
# ds.to_json('/network/scratch/a/arnaud.bergeron1/VinePPO/mixed_rewards_outputs.jsonl')