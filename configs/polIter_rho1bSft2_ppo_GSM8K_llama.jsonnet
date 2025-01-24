local hf_model_name = 'meta-llama/Llama-3.1-8B-Instruct';
local task = (import 'tasks/gsm8k.jsonnet');
local total_num_iterations = 3200;


(import 'polIter_rho1bSft2_ppo_MATH_llama.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: task,
        reward_function+: { math_task: $.episode_generator.task },

        initial_model_name_or_path: hf_model_name,

        inference_strategy+: {
            guidance_llm: (import 'guidance_llms/rho1b-sft-GSM8K_llama.jsonnet') + { api_base: 'none' },
        },
    },
    num_iterations: total_num_iterations,
}
+ (import 'sft_rho1b_for_gsm8k_eval_llama.jsonnet')
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0.jsonnet')
+ (import 'trainers/klLoss.jsonnet')
