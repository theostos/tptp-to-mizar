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
    parser.add_argument('--input', default='export/dataset/metadata.json')
    parser.add_argument('--output', default='export/dataset_noreasoning/')
    parser.add_argument('--threshold', type=int, default=15_000)
    parser.add_argument('--max-len', type=int, default=10000)
    args = parser.parse_args()
    with open(args.input, 'r') as file:
        metadata = json.load(file)
    
    metadata_list = [(metadata[filepath_in]['num_tokens'], filepath_in, metadata[filepath_in]['filepath_out']) for filepath_in in metadata]
    metadata_list = [(num, f_in, f_out) for num, f_in, f_out in metadata_list if num < args.threshold]
    result_train = []
    result_eval = []
    for _, filepath_in, filepath_out in metadata_list:
        with open(filepath_in, 'r') as file:
            data_in = file.read()
        
        with open(filepath_out, 'r') as file:
            data_out = file.read()
        
        source = filepath_in.split('/')[-1]
        if not "\nproof\n" in data_out:
            entry = {'source': source, 'tptp_proof': data_in, 'formal_statement': data_out}
            result_eval.append(entry)
        elif data_out.count('\nproof\n') == 1:
            formal_statement, proof = data_out.split('\nproof\n')
            proof = proof.strip()
            assert proof.endswith('end;')
            proof = proof[:-4]
            entry = {'source': source, 'tptp_proof': data_in, 'formal_statement': formal_statement, 'target_proof': proof}
            result_train.append(entry)
    
    with open(os.path.join(args.output, 'train.json'), 'w') as file:
        json.dump(result_train[:args.max_len], file, indent=4)
    with open(os.path.join(args.output, 'eval.json'), 'w') as file:
        json.dump(result_eval[:args.max_len], file, indent=4)


