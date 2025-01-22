local num_samples = 16;
local temperature = 0.35;

local tokenizer = {
    type: 'pretrained',
    hf_model_name: 'meta-llama/Llama-3.2-1B-Instruct',
};

local max_tokens = 1024;
local model_context_size = 2047;

local math_inference_pipeline =
    (import 'prompt_library/llama2_sft_gsm8k.jsonnet')
    + (import 'inference_strategies/tree/iid_expander.jsonnet')
    + (import 'inference_strategies/cot.jsonnet')
    + {
        inference_strategy+: {
            max_concurrent_programs: 512,
            max_concurrent_generations: 128,

            node_expander+: {
                type: 'efficient_iid',
                program_kwargs: {
                    temperature: temperature,
                    top_p: 0.9,
                    max_tokens: max_tokens,
                    stop: '"\nAnswer:"',
                },
                node_text_template: '{chain_of_thought}',

                // Needed to compute max_tokens on the fly
                model_context_size: model_context_size,
                tokenizer: tokenizer,
            },
            answer_extractor+: {
                type: 'next_chat_turn',
                program: $.prompt_library.tree.answer_extract.next_chat_turn,
                program_kwargs: {
                temperature: 0,
                max_tokens: 20,
            },
            },
            samples: num_samples,
            max_depth: 10,

            guidance_llm: (import 'guidance_llms/rho1b-sft-GSM8K_llama.jsonnet') + { api_base: 'none' },
            no_cache: true,
            question_field: 'query',

            seed: 42,
        },
        task: (import 'tasks/math_inplace_no_answer_prefix.jsonnet'),
        analyzers: [(import 'analyzers/task_performance.jsonnet')],
    };

local math_train_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'train',
        dataset_portion: 0.04347826,  // About 500 samples
        dataset_shuffle_before_portion: true,
        inference_name: 'math_train',
    };

local math_test_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'test',
        dataset_portion: 1,
        inference_name: 'math_test',
    };

local math_validation_inference_pipeline =
    math_inference_pipeline
    + {
        dataset_split: 'validation',
        dataset_portion: 1,
        inference_name: 'math_validation',
    };

local konkur_inference_pipeline =
    math_inference_pipeline
    + {
        task: (import 'tasks/konkur_inplace_no_answer_prefix.jsonnet'),
    };

local collegeMath_inference_pipeline =
    math_inference_pipeline
    + {
        task: (import 'tasks/collegeMath_inplace_no_answer_prefix.jsonnet'),
    };

local collegeMath_test_inference_pipeline =
    collegeMath_inference_pipeline
    + {
        dataset_split: 'test',
        dataset_portion: 0.1774308,
        inference_name: 'collegeMath_test',
    };

local olympiadbench_inference_pipeline =
    math_inference_pipeline
    + {
        task: (import 'tasks/olympiadbench_inplace_no_answer_prefix.jsonnet'),
    };

local olympiadbench_test_inference_pipeline =
    olympiadbench_inference_pipeline
    + {
        dataset_split: 'test',
        dataset_portion: 1,
        inference_name: 'olympiadbench_test',
    };

{
    inference_pipelines: [
        math_test_inference_pipeline,
        math_validation_inference_pipeline,
        math_train_inference_pipeline,
        collegeMath_test_inference_pipeline,
        olympiadbench_test_inference_pipeline,
    ],

    evaluation_vllm_server: {},
}
