local num_expansion_rounds = 16;
local guidance_program = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} top_p={top_p} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} n={num_samples}}}';
(import '../prompt_library/llama2_sft_gsm8k.jsonnet')

{
    type: 'mc_value_prediction',

    max_num_checkpoints: 10,
    max_num_requests: 100,

    inference_strategy: {
        type: 'cot',

        max_concurrent_programs: 16,
        max_concurrent_generations: 16,

        samples: 256 / num_expansion_rounds,
        max_depth: 100,

        node_expander: {
            type: 'efficient_iid',
            program: guidance_program,
            program_kwargs+: {
                temperature: 1,
                top_p: 0.9,
                max_tokens: 1024,
                stop: '"<|eot_id|>"',
            },
            node_text_template: '{chain_of_thought}',
            num_expansion_rounds: num_expansion_rounds,
        },


        answer_extractor+: {
            type: 'next_chat_turn',
            program: $.prompt_library.tree.answer_extract.next_chat_turn,
            program_kwargs: {
            temperature: 1.0,
            max_tokens: 20,
        }},

        question_field: 'query',
        question_template: '{query}',

        no_cache: true,
    },

    vllm_server+: {
        swap_space: 32,
        enable_prefix_caching: true,
    },
}
