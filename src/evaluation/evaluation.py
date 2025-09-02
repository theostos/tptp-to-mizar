import argparse
import os
import json
from collections import defaultdict

from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default='export/dataset/eval.json', help='Dataset')
    parser.add_argument('--model-path', default='mirza_reasoner/', help='model')
    parser.add_argument('--tokenizer-path', default='mirza_reasoner/', help='tokenizer')
    parser.add_argument('--prompt-path', default='export/dataset/prompt.json', help='prompt')
    parser.add_argument('--output', default='export/eval', help='Output directory')
    parser.add_argument('--k', type=int, default=32, help='Number of generation per entry')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Max output len')
    parser.add_argument('--gpus', type=int, default=4, help='Number of gpus')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    with open(args.prompt_path, 'r') as file:
        prompt_template = json.load(file)
    with open(args.eval, 'r') as file:
        dataset = json.load(file)
    

    llm = LLM(model=args.model_path, tokenizer=args.tokenizer_path, max_model_len=10_000, max_num_seqs=32, tensor_parallel_size=args.gpus, dtype="bfloat16", gpu_memory_utilization=0.98, trust_remote_code=True)

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p, stop_token_ids=[151645])
    sampling_params.n = args.k  # This tells vLLM to generate k completions per prompt.
    for entry in tqdm(dataset):
        prompt_text = prompt_template['instruction'].format(tptp_proof=entry['tptp_proof'], formal_statement=entry['formal_statement'])
        outputs = llm.generate([prompt_text], sampling_params)
        # Each output in "outputs" is a RequestOutput object containing a list of completions.
        result = []
        for output in outputs:
            for completion in output.outputs:
                result.append(completion.text)
        
        new_entry = {"source": entry['source'], "outputs": [{"content": completion} for completion in result]}
        filepath = os.path.join(args.output, entry['source'])
        with open(filepath, 'w') as file:
            json.dump(new_entry, file, indent=4)
    