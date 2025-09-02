import os
import random
import argparse
import time
import random
import json
import concurrent.futures

import yaml
from openai import OpenAI
from tqdm import tqdm

"""
Generate reasoning traces
"""

def generate_output(prompt, client, config):
    """
    Sends prompt to client using config.
    """
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        **config
    )
    return {"reasoning": completion.choices[0].message.reasoning, "content": completion.choices[0].message.content}

def process_prompt(prompt, export_path, data, client, config, delay=0):
    """
    Executes multiple generation of the same prompt, export them sequentially.
    """
    time.sleep(delay)
    data['output'] = generate_output(prompt, client, config)
    with open(export_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='export/dataset/train.json', help='Input dataset path')
    parser.add_argument('--suboutput', default='export/dataset/traces/', help='Output traces path')
    parser.add_argument('--output', default='export/dataset/train.json', help='Final dataset')
    parser.add_argument('--max-workers', default=100, type=int, help='Max number of concurrent workers')
    parser.add_argument('--mean-delay', default=10, type=int, help='Mean delay before a request is send: use this parameter to load balance')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
    with open(prompt_path, 'r') as file:
        prompt_template = file.read()

    client = OpenAI(
        base_url=config['base_url'],
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    to_do = []

    with open(args.input, 'r') as file:
        dataset = json.load(file)
    
    for k, entry in enumerate(dataset['train']):
        prompt = prompt_template.format(tptp_proof=entry['tptp_proof'], formal_statement=entry['formal_statement'], target_proof=entry['target_proof'])
        filepath = os.path.join(args.suboutput, entry['source'])
        if not os.path.exists(filepath):
            to_do.append((prompt, filepath, entry))

    delay_max = args.mean_delay*2
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        futures += [executor.submit(process_prompt, prompt, export, entry, client, config['request_config'], delay=random.randint(0, delay_max)) for prompt, export, entry in to_do]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
    
    result = []
    for filename in os.listdir(args.suboutput):
        filepath = os.path.join(args.suboutput, filename)
        with open(filepath, 'r') as file:
            entry = json.load(file)
        entry['reasoning'] = entry['output']['content']
        del entry['output']
        result.append(entry)
    
    with open(args.output, 'w') as file:
        json.dump({"train": result}, file, indent=4)
        
        

