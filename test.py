# from datasets import Dataset

# data_dir = '/network/scratch/a/arnaud.bergeron1/VinePPO/experiments/llama_grad_clip_ap_1_9/mixed_rewards_ap_1_bp_1_am_0_bm_1_llama-8e7/episodes/episodes_0000'

# data = Dataset.load_from_disk(data_dir)

# critic_vals = data[0]['response_token_ids']
# advantages = data[0]['query_token_ids']
# resp_ = data[0]['response_token_ids']
# print(critic_vals)
# print(advantages)
# print('adv')
# # print(advantages)
# print(f'len critic vals {len(critic_vals)}')
# print(f'len advantages {len(advantages)}')
# print(f'len resp_ {len(resp_)}')
# print(len(critic_vals[len(advantages) - 1 :]))


# import torch
# from transformers import AutoModelForCausalLM
# from transformers import AutoConfig

# # Clear any existing memory
# torch.cuda.empty_cache()

# # Load model with basic settings
# print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# # Load model with explicit settings
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.1-8B-Instruct",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     max_position_embeddings=8192  # Use original context length
# )

# # Print final memory state
# print(f"Final CUDA memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", padding_side='left')

# id1 = tokenizer.convert_tokens_to_ids('\n')
# print(id1) # None

# id2 = tokenizer('\n')['input_ids'][1]
# print(id2) # 198

# newline_tok = tokenizer.convert_ids_to_tokens([1805, 55838, 12031, 99923, 22106, 100831, 35121, 42504, 76655, 19708, 45237, 120854, 5809, 27, 51, 3692, 94090, 104414, 115655, 73914, 9937, 102817, 9092, 36563, 99655, 6290, 120795, 20242, 60841, 96116, 82129, 21372, 75423, 45611, 68598, 117507, 91706, 126616, 100674, 13172, 86153, 80184, 45068, 123816, 63833, 38642, 73862, 7178, 94788, 118208, 27291, 88487, 126543, 12084, 28134, 58903, 42854, 5360, 103931, 39008, 47195, 88340, 99996, 82586, 46948, 73122, 70220, 103958, 117591, 4864, 80814, 81728, 39354, 63339, 52429, 8311, 93336, 63373, 2938, 9568, 73854, 32208, 71114, 76744, 50465, 69548, 77063, 44668, 122475, 60309, 93598, 55418, 33044, 117396, 55994, 14822, 13, 9350, 6707, 79733, 111640, 54462, 126483, 100752, 2257, 78861, 26477, 52937, 97056, 26818, 21100, 118494, 47193, 27171, 12340, 1487, 18462, 118084, 62, 2414, 30113, 3023, 119814, 47974, 83677, 107638, 83657, 96269, 16676, 93777, 96208, 106850, 66848, 117167, 118666, 26418, 42450, 97260, 34411, 108171, 72800, 97853, 80659, 47086, 1034, 3343, 36887, 4805, 95197, 42731, 13794, 59462, 61772, 54199, 7661, 116377, 18262, 80220, 92010, 22048, 48839, 85446, 61000, 25714, 82077, 650, 1995, 566, 402, 38330, 65963, 19196, 10192, 31120, 126827, 70073, 27200, 51775, 81483, 41966, 91672, 68600, 122991, 60650, 60501, 93515, 54112, 98283, 54914, 19406, 23107, 41049, 46870, 79803, 124323, 26463, 87992, 3471, 101237, 86168, 28424, 66048, 120230, 30433, 19071, 76886, 55231, 122921, 112068, 82822, 116720, 98628, 57734, 69112, 71824, 57711, 75, 525, 65845, 30535, 14968, 88875, 15909, 43714, 50093, 5359, 124529, 105491, 53935, 87483, 30439, 111053, 117124, 19036, 79493, 34860, 41699, 97379, 98688, 78501, 35777, 71562, 65274, 71784, 55420, 67717, 125331, 104087, 6440, 84497, 33806, 1989, 42335, 125444, 67891, 104434, 15956, 34576, 74542, 81947, 55569, 73283, 87659, 2925, 77928, 7431, 81621, 116129, 2865, 39411, 125199, 123435, 19064, 85621, 105445, 51174, 110045, 42777, 103933, 108643, 64028, 19995, 62668, 92193, 64144, 7887, 18207, 60356, 54344, 95340, 6089, 27075, 53740, 43792, 7619, 31509, 46553, 11750, 114743, 87060, 11883, 7915, 122353, 1237, 35884, 52071, 76899, 74490, 100377, 62583, 11280, 85795, 109482, 45240, 21290, 101127, 5809, 35409, 102720, 30080, 19293, 92973, 75886, 39747, 81392, 31671, 113953, 72881, 101332, 118248, 0, 12, 437, 84985, 122587, 12734, 106222, 56112, 42000, 65364, 122148, 10765, 36160, 74960, 22904, 41990, 44612, 21733, 52588, 103395, 109035, 50428, 13465, 98537, 61769, 81155, 52293, 43271, 21423, 78806, 75121, 93533, 5522, 39376, 18801, 33208, 22280, 93595, 1085, 23328, 61442, 72687, 94439, 107394, 94542, 114545, 97120, 78871, 70352, 4937, 5297, 19774, 68820, 96920, 95181, 90381, 23572, 41981, 7967, 117437, 9528, 34443, 19676, 97970, 116244, 121152, 61623, 80213, 112737, 88741, 120577, 79639, 107838, 37313, 44267, 8241, 115890, 830, 710, 689, 71049, 121202, 3782, 98404, 8627, 76170, 22102, 53661, 78630, 91687, 35349, 66979, 34849, 34846, 5922, 110149, 16351, 125400, 43933, 107177, 94253, 14837, 45767, 27953, 105852, 46589, 36222, 54660, 39724, 125722, 53538, 78519, 114193, 41424, 95419, 81303, 5363, 23454, 8057, 74306, 115940, 75511, 44979, 18922, 72586, 41296, 103604, 83905, 39019, 37827, 39094, 105896, 39848, 66795, 127528, 122024, 19169, 95851, 107384, 15832, 81779, 89957, 28756, 100329, 121803, 48653, 111629, 98421, 112659, 110948, 108104, 88523, 911, 79159, 65404, 64938, 79061, 18690, 90033, 71695, 102350, 5809, 57195, 53448, 33902, 101068, 48045, 36567, 92418, 12294, 112501, 9686, 26298, 983, 49361, 16076, 11734, 113839, 15554, 62, 47189, 37170, 89681, 107663, 22678, 4969, 46027, 12, 24137, 392, 5600, 125137, 95419, 8587, 25771, 86377, 3508, 68072, 2690, 52481, 8773, 87867, 37981, 56760, 106932, 23249, 101838, 101779, 67573, 86771, 18892, 114255, 14870, 58313, 82196, 66451, 57057, 122845, 78056, 38267, 56238, 125712, 66637, 69131, 50342, 50225, 75354, 104842, 116543, 91268, 38660, 8577, 60561, 108941, 50432, 112820, 101799, 49923, 30374, 76154, 118487, 62820, 8084, 107133, 5741, 3106, 112497, 81856, 101474, 2648, 124282, 46424, 9998, 9259, 11992, 60145, 123419, 77852, 41600, 126072, 69742, 67282, 97605, 66543, 106684, 75530, 25321, 14967, 46538, 653, 8510, 59091, 6440, 126614, 34566, 1902, 46453, 5698, 88972, 22979, 10349, 8764, 93291, 18010, 73167, 111672, 83234, 30983, 125629, 80791, 75805, 74237, 6844, 84952, 80252, 18889, 82053, 67155, 12600, 71130, 6741, 106200, 22841, 70248, 49945, 75534, 53905, 33163, 113771, 37840, 112248, 72694, 103059, 11120, 121261, 19464, 16250, 72905, 8168, 27, 45, 63576, 84391, 64515, 84927, 96298, 85577, 25735, 123437, 69681, 24464, 105353, 62923, 85717, 17881, 122530, 81469, 75767, 4639, 74404, 9121, 5295, 19263, 105577, 105990, 79291, 40006, 124135, 23772, 86605, 123440, 124125, 110035, 104130, 86051, 70255, 81202, 94845, 24231, 102178, 6932, 2034, 1610, 349, 123842, 15993, 102765, 43287, 114720, 83408, 116883, 91850, 56850, 100227, 37495, 45331, 81391, 63738, 124855, 33929, 101555, 26051, 22928, 76258, 72454, 123676, 3614, 126351, 120768, 121383, 86534, 86651, 54036, 64369, 127562, 110927, 62525, 3988, 90949, 63017, 19963, 116309, 57916, 64416, 24144, 662, 9426, 5069, 90776, 94022, 13787, 89723, 21745, 57757, 56576, 71472, 3587, 8750, 1565, 2914, 1310, 46773, 80977, 25427, 91807, 95946, 1312, 27702, 100419, 17387, 30433, 27800, 54236, 38980, 101678, 4534, 91239, 41962, 73940, 77001, 124021, 38162, 71682, 41116, 22608, 86958, 59640, 98750, 61050, 77435, 42486, 68231, 4608, 52086, 110905, 48117, 30626, 124188, 59726, 84571, 75624, 77449, 56032, 79991, 121186, 43235, 121316, 124920, 52183, 35431, 108908, 32265, 66706, 65974, 49147, 24872, 119328, 21187, 30201, 74136, 104088, 58138, 13689, 37099, 30341, 7181, 113148, 113775, 2062, 48984, 51528, 89458, 107807, 22721, 21739, 114849, 65728, 5809, 122316, 50347, 74694, 12233, 49125, 112183, 578, 114482, 44276, 46427, 14229, 52477, 46675, 31787, 93640, 52077, 5809, 116995, 76998, 106821, 18196, 76047, 47811, 89199, 52248, 116527, 74239, 15612, 24463, 53920, 462, 38927, 43783, 80417, 4954, 100116, 20356, 27780, 104980, 112687, 64808, 61681, 89879, 50229, 48023, 18156, 49676, 104594, 42284, 2647, 67652, 74086, 53254, 32211, 77621, 122309, 30433, 332, 20744, 10838, 86970, 45047, 52281, 28329, 59343, 43255, 42116, 34832, 93822, 68136, 111913, 7687, 72339, 77200, 98017, 54442, 24673, 46991, 66292, 617, 48086, 108420, 99280, 104608, 84499, 13, 17794, 82050, 47906, 27005, 88682, 118339, 33653, 66981, 68072, 85496, 65975, 30433, 99631, 59190, 5935, 81235, 124839, 86505, 125755, 57177, 25194, 63494, 23462, 83332, 20296, 13679, 22617, 123074, 50991, 127255, 93043, 58616, 34471, 2655, 68419, 48591, 128136, 30433, 101348, 26551, 73263, 106514, 16334, 42879, 69860, 42388, 110992, 98275, 75988, 60967, 93793, 12897, 119541, 551, 4527, 8318, 93022, 3206, 52110, 76815, 109296, 93503, 76634, 111623, 54645, 124507, 105472, 22230, 94212, 69887, 28666, 79550, 40554, 93426, 62870, 118058, 76061, 59020, 79512, 21410, 89615, 57160, 826, 316, 15007, 68762, 23135, 45754, 73427, 79099, 59170, 43921, 44711, 78344, 39929, 62024, 74390, 78880, 108507, 115813, 2234, 18309, 27070, 91174, 43048, 17878, 95882, 7559, 63902, 94894, 106268, 61571, 107053, 72141, 103967, 26951, 5629, 72097, 3416, 19099, 61133, 85181, 92921, 102552, 3862, 406, 3862, 106755, 101581, 76526, 60445, 32113, 70758, 46035, 35341, 108340, 56491, 7838, 26722, 38822, 72786, 56611, 46074, 101635, 13603, 23448])
# char_tok = tokenizer.convert_tokens_to_string(newline_tok)
# print(char_tok) # Ċ

# id3 = tokenizer('Ċ')['input_ids'][1]
# print(id3) # 128

# id4 = tokenizer.convert_tokens_to_ids('Ċ')
# print(id4) # 198

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from typing import Dict, Any
# import numpy as np

# def debug_model_outputs(
#     model_path: str,
#     prompt: str,
#     num_samples: int = 5,
#     temperature_range: list = [0.1, 0.3, 0.7, 1.0]
# ) -> Dict[str, Any]:
#     """
#     Debug model outputs with different sampling parameters and return diagnostics.
    
#     Args:
#         model_path: Path to the model
#         prompt: Input prompt to test
#         num_samples: Number of samples to generate per temperature
#         temperature_range: List of temperatures to test
#     """
#     # Load model and tokenizer
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         max_position_embeddings=8192  # Use original context length
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
    
#     diagnostics = {
#         "token_probabilities": [],
#         "entropy_per_token": [],
#         "samples": {}
#     }
    
#     for temp in temperature_range:
#         samples = []
#         for _ in range(num_samples):
#             # Generate with different parameters
#             outputs = model.generate(
#                 tokenizer.encode(prompt, return_tensors="pt").to(model.device),
#                 max_length=100,
#                 num_return_sequences=1,
#                 temperature=temp,
#                 top_p=0.9,
#                 top_k=50,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id,
#                 return_dict_in_generate=True,
#                 output_scores=True
#             )
            
#             # Get token probabilities
#             probs = torch.stack(outputs.scores).softmax(-1)
#             entropy = -(probs * probs.log()).sum(-1).mean().item()
            
#             decoded = tokenizer.decode(outputs.sequences[0])
#             samples.append({
#                 "text": decoded,
#                 "entropy": entropy
#             })
            
#             diagnostics["token_probabilities"].append(probs.max(-1).values.mean().item())
#             diagnostics["entropy_per_token"].append(entropy)
        
#         diagnostics["samples"][f"temp_{temp}"] = samples
    
#     return diagnostics

# def suggest_fixes(diagnostics: Dict[str, Any]) -> list:
#     """Analyze diagnostics and suggest potential fixes."""
#     suggestions = []
    
#     # Check for very high entropy
#     avg_entropy = np.mean(diagnostics["entropy_per_token"])
#     if avg_entropy > 2.0:
#         suggestions.append("High token entropy detected. Consider:\n"
#                          "- Lowering the temperature (try 0.2-0.4)\n"
#                          "- Reducing top_k (try 20-30)\n"
#                          "- Reducing top_p (try 0.7-0.8)")
    
#     # Check for very low token probabilities
#     avg_prob = np.mean(diagnostics["token_probabilities"])
#     if avg_prob < 0.1:
#         suggestions.append("Low token probabilities detected. Consider:\n"
#                          "- Training for more epochs\n"
#                          "- Checking training data quality\n"
#                          "- Validating model architecture")
    
#     return suggestions

# diagnostics = debug_model_outputs(
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "You are a helpful assistant solving math questions. Always answer in most accurate way. \
#     <</SYS>> \
#     Answer the following middle school math word problems, which require multi-step arithmetic reasoning.\
#     \
#     Q: Oranges have 80 calories and cost $1.20 each. If Timmy has $10 and needs to make sure he gets 400 calories, how much money will he have left after he buys the oranges he needs?[/INST]\
#     A:"
# )
# print(diagnostics)
# suggestions = suggest_fixes(diagnostics)
# print(suggestions)


# import subprocess

# ps_call = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# lines_proc = ps_call.stdout.decode().split("\n")
# print(f"Process ID before: {lines_proc}")


def extract_predicted_answer_from_text(text: str, problem: Optional[str] = None
    ) -> Optional[str]:
    # Extract the final answer based on ####
    if "####" not in text:
        return None
    parts = text.split("####")
    assert len(parts) >= 2
    return parts[-1].strip()

    text = text.replace(",", "")
    pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
    if len(pred_answer) == 0:
        return None
    else:
        # Pick the last number
        pred_answer = pred_answer[-1].strip()
        return pred_answer

def extract_gold_answer_from_text(text: str) -> str:
        return text.split("####")[1].strip()


def grade_answer(
        *,
        given_answer: Optional[str] = None,
        ground_truth: str = None,
        item: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        # if given_answer is None:
        #     return False

        # assert ground_truth is not None
        # ground_truth = ground_truth.replace(",", "")

        # return (
        #     float(given_answer.strip().replace(",", "").lower())
        #     == float(ground_truth.strip().lower())
        # )