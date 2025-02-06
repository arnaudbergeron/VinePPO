local orig_library = (import 'llama2.jsonnet');
local tree_expansion_iid = '{{prefix}}{{gen "chain_of_thought" temperature={temperature} max_tokens={max_tokens} save_stop_text="stop_text" stop={stop} logprobs={logprobs} n={num_samples}}}';

local tree_answer_extract_next_chat_turn = '{{prefix}}<|eot_id|><|start_header_id|>user<|end_header_id|>So, what is the final answer? Only output digits.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{gen "final_answer" temperature={temperature} max_tokens={max_tokens}}}';

local tree_question_template = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant solving math questions. Always answer in most accurate way.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.

Question: {query}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>';

orig_library + {
    prompt_library+: {
        tree+: {
            expansion+: {
                iid: tree_expansion_iid,
            },
            answer_extract+: {
                next_chat_turn: tree_answer_extract_next_chat_turn,
            },
            question_template: tree_question_template,
        },
    },
}
