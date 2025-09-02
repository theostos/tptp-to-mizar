import os
import argparse
import json
import sys
sys.setrecursionlimit(10_000) 

from tqdm import tqdm
from transformers import AutoTokenizer

"""
Preprocess dataset
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', default='export/dataset/in/', help='Dataset in')
    parser.add_argument('--data-out', default='export/dataset/out/', help='Dataset out')
    parser.add_argument('--output', default='export/dataset/metadata.json', help='File to export')
    parser.add_argument('--tokenizer', default='deepseek-ai/DeepSeek-Coder-V2-Instruct', help='HF path tokenizer')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    to_do = []


    for filename_in in os.listdir(args.input):
        filepath_in = os.path.join(args.input, filename_in)
        filepath_out = os.path.join(args.output, filename_in)
        to_do.append((filepath_in, filepath_out))

    result = {}
    for filepath_in, filepath_out in tqdm(to_do):
        with open(filepath_in, 'r') as file:
            in_data = file.read()
        with open(filepath_out, 'r') as file:
            out_data = file.read()        
        tokens_term = tokenizer(in_data, return_tensors="pt", truncation=False)["input_ids"]
        result[filepath_in] = {"num_tokens": tokens_term.size(1), "filepath_out": filepath_out}
    
    with open('metadata.json', 'w') as file:
        json.dump(result, file, indent=4)
        