local hf_model_name = 'meta-llama/Llama-3.1-8B-Instruct';

local actor_tokenizer = {
    type: 'pretrained',
    hf_model_name: 'meta-llama/Llama-3.1-8B-Instruct',
};

local math_task = (import 'tasks/math_inplace_no_answer_prefix.jsonnet') + {
    prepend_in_context_few_shot: false,
    ensure_fit_in_context_size: false,
};

local num_episodes_per_iteration = 64;
local num_rollouts_per_sample = 1;
local num_dataset_samples_per_iteration = num_episodes_per_iteration / num_rollouts_per_sample;
local total_num_iterations = 50;

local sampling_temperature = 1.0;

(import 'gvar.jsonnet')
+ (import 'prompt_library/llama2_sft_gsm8k.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'episode_generators/math_episode_generator.jsonnet')
+ (import 'trainers/ppo_MATH.jsonnet')
+ {
    episode_generator+: {
        // Override the task
        task: math_task,
        reward_function+: { math_task: $.episode_generator.task },
        reasoning_step_delimiter: '',
        answer_prefix: null,

        initial_model_name_or_path: hf_model_name,

        dataset_sample_with_replacement: false,
        dataset_initial_size: 200,
        dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
        total_num_iterations: $.num_iterations,

        vllm_server+: { swap_space: 64, max_num_seqs: 512, max_model_len: 2048 },
        vllm_min_available_gpu_memory_mb: 10 * 1024,

        inference_strategy: {
            type: 'cot',

            max_concurrent_programs: 16,
            max_concurrent_generations: 16,

            samples: num_rollouts_per_sample,
            max_depth: 100,  // Deprecated parameter. Doesn't do anything.

            node_expander: {
                type: 'efficient_iid',
                program: $.prompt_library.tree.expansion.iid,
                program_kwargs+: {
                    temperature: sampling_temperature,
                    top_p: 0.9,
                    max_tokens: 1024,
                    stop: '"<|eot_id|>"',
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: 2048,
                tokenizer: $.tokenizer,
            },

            answer_extractor: {
                type: 'next_chat_turn',
                program: $.prompt_library.tree.answer_extract.next_chat_turn,
                program_kwargs: {
                temperature: 1.0,
                max_tokens: 20,
            },
            },

            guidance_llm: (import 'guidance_llms/rho1b-sft-GSM8K_llama.jsonnet') + { api_base: 'none' },

            question_field: 'query',
            question_template: $.prompt_library.tree.question_template,

            no_cache: true,
        },
    },

    tokenizer: {
        type: 'pretrained',
        hf_model_name: 'meta-llama/Llama-3.1-8B-Instruct',
    },
    use_deepspeed: true,

    num_iterations: total_num_iterations,
    num_episodes_per_iteration: num_episodes_per_iteration,
    episodes_cloud_log_steps: 50,

    trainer+: {
        params+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
        // temp_checkpoint_dir: '/network/scratch/a/arnaud.bergeron1/rlhf/temp_checkpoints',
        actor_model+: { 
            hf_model_name: $.episode_generator.initial_model_name_or_path,
        //     freeze_config+:{
        //     freeze_first_k_layers: 22,
        //     freeze_embeddings: true,
        // }
        },
        critic_model+: { pretrained_backbone_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path } },
        reference_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },

        actor_deepspeed_config: (import 'deepspeed/zero_2.jsonnet'),
        critic_deepspeed_config: (import 'deepspeed/zero_0.jsonnet'),

        // To prevent OOM errors
        report_entropy: false,

        general_training_args+: {
            target_train_batch_size: 1,
            per_device_train_batch_size: null,  // Will be auto computed
            gradient_accumulation_steps: 1,

            save_steps: 25,
            checkpoint_keep_steps: 10  ,
            warmup_steps: 1.0,
            lr_scheduler_type: 'constant',
        },
    },


    analyzers: [
        (import 'analyzers/valnet_prediction_llama.jsonnet') + {
            task: $.episode_generator.task,
            tokenizer: $.tokenizer,
            vllm_server+: { swap_space: 24 },

            reward_function: $.episode_generator.reward_function,

            // Small model. Can afford more requests.
            max_num_requests: 512,

            inference_strategy+: {
                guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

                // Small model. Can afford more concurrent programs.
                max_concurrent_programs: 16,
                max_concurrent_generations: 16,

                node_expander+: {
                    program_kwargs+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
                    model_context_size: $.episode_generator.inference_strategy.node_expander.model_context_size,
                    tokenizer: $.tokenizer,
                },
            },
        },

        (import 'analyzers/ppo_grad_variance.jsonnet') + {
            per_device_batch_size: 1,
        },

        (import 'analyzers/valnet_action_ranking_llama.jsonnet') + {
            task: $.episode_generator.task,
            tokenizer: $.tokenizer,
            vllm_server+: { swap_space: 24 },

            reward_function: $.episode_generator.reward_function,

            max_num_requests: 512,
            max_num_states: 16,

            append_bos_to_query: $.episode_generator.append_bos_to_query,

            inference_strategy+: {
                guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

                // Small model. Can afford more concurrent programs.
                max_concurrent_programs: 16,
                max_concurrent_generations: 16,

                node_expander+: {
                    program_kwargs+: { temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature },
                    model_context_size: $.episode_generator.inference_strategy.node_expander.model_context_size,
                    tokenizer: $.tokenizer,
                },
            },
        },
    ],
}
+ (import 'sft_rho1b_for_MATH_eval_llama.jsonnet')
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0.jsonnet')
+ (import 'trainers/klLoss.jsonnet')