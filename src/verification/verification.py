import argparse
import json
import os

import subprocess

from tqdm import tqdm

def create_subdocument(source_path, formal_statement):
    with open(source_path, 'r') as file:
        content = file.read()
    count = content.count(formal_statement)
    assert count == 1, "formal statement wasn't identified properly"

    idx_formal_statement = content.find(formal_statement) + len(formal_statement)
    return content[:idx_formal_statement]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mml-path', default='/usr/local/share/mizar/mml/', help='mml path')
    parser.add_argument('--eval-path', default='export/eval/eval.json', help='model')
    parser.add_argument('--output', default='export/verification', help='Output directory')
    args = parser.parse_args()

    env = os.environ.copy()
    env["MIZFILES"] = "/usr/local/share/mizar"

    with open(args.eval_path, 'r') as file:
        eval = json.load(file)
    
    for entry in eval:
        print("new source!!")
        formal_statement = "\n".join(entry['formal_statement'].split('\n')[-2:])
        source_file = '_'.join(entry['source'].split('_')[1:])
        source_path = os.path.join(args.mml_path, source_file + '.miz')
        subdocument = create_subdocument(source_path, formal_statement)
        for k, output in tqdm(enumerate(entry['outputs'])):
            
            final_proof = output['final_proof']
            subdocument_bis = subdocument + '\nproof\n' + final_proof + '\nend;'

            temp_filepath = os.path.join(args.output, 'tmp', 'temp.miz')
            err_filepath = os.path.join(args.output, 'tmp', 'temp.err')
            with open(temp_filepath, 'w') as file:
                file.write(subdocument_bis)
            

            result = subprocess.run(
                ["mizf", temp_filepath],
                capture_output=True,
                text=True,
                env=env
            )
            assert "Time of mizaring" in result.stdout, f"Issue with mizf command: {result.stdout}\n{result.stderr}"

            with open(err_filepath, 'r') as file:
                err_log = file.read()
            
            if len(err_log) < 2:
                success_path = os.path.join(args.output, f"success_{entry['source']}_{k}")
                with open(success_path, 'w') as file:
                    subentry = {"source": entry['source'], "formal_statement": entry['formal_statement']} | output
                    json.dump(subentry, file, indent=4)

        


    