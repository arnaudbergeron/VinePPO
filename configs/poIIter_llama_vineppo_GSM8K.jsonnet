local hf_model_name = 'meta-llama/Llama-3.2-1B';
local task = (import 'tasks/gsm8k_orig_format.jsonnet');
local total_num_iterations = 650;


(import 'poIIter_llama_vineppo_MATH.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: task,
        reward_function+: { math_task: $.episode_generator.task },

        initial_model_name_or_path: hf_model_name,

        max_step_for_value_estimation: 25,

        inference_strategy+: {
            guidance_llm: (import 'guidance_llms/llama1b.jsonnet') + { api_base: 'none' },
        },
    },
    num_iterations: total_num_iterations,
}
+ (import 'sft_rho1b_for_gsm8k_eval_llama.jsonnet')
+ (import 'episode_generators/9rolls.jsonnet')
+ (import 'trainers/refKl0.0001.jsonnet')
+ (import 'trainers/klLoss.jsonnet')