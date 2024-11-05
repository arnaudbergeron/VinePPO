# import uncertainties as unp
#load hf dataset
from datasets import Dataset
import os
import json
import numpy as np

results_dir_pre = '/network/scratch/a/arnaud.bergeron1/vine/experiments/mixed_rewards_natural_adj/polIter_rho1bSft2_ppo_GSM8K/evaluation'
results_dir_mid = 'gsm8k_test/analysis/TaskPerformanceAnalyzer/gsm8k_test'
results_dir_post = 'test__0__1/log.json'

results = {'vanilla_ppo': {}}

for ckpt in os.listdir(results_dir_pre):
    if 'ckpt--iter' in ckpt:
        json_path = f'{results_dir_pre}/{ckpt}/{results_dir_mid}/{ckpt}/{results_dir_post}'
        try:
            with open(json_path, 'r') as f:
                results['vanilla_ppo'][ckpt] = json.load(f)['correct_frac']
                
        except Exception as e:
            print(f'Error with {e}')

print((results['vanilla_ppo']))