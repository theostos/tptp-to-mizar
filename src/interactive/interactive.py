import argparse
import json
from vllm import LLM, SamplingParams

def multiline_input(prompt=""):
    print(prompt + " (end with a blank line):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:  # in case of ctrl+d
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='mirza_reasoner/', help='Path to model')
    parser.add_argument('--tokenizer-path', default='mirza_reasoner/', help='Path to tokenizer')
    parser.add_argument('--prompt-path', default='config/prompts/prompt.json', help='Prompt template')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Max output length')
    parser.add_argument('--gpus', type=int, default=2, help='Number of GPUs')
    args = parser.parse_args()

    # Load prompt template
    with open(args.prompt_path, "r") as f:
        prompt_template = json.load(f)["instruction"]

    # Load model
    llm = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path,
        max_model_len=10_000,
        tensor_parallel_size=args.gpus,
        dtype="bfloat16",
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    # Interactive loop
    while True:
        print("\n--- New Query ---")
        tptp_proof = multiline_input("Enter TPTP proof")
        if not tptp_proof:
            print("Empty input, exiting.")
            break

        formal_statement = multiline_input("Enter formal statement")
        if not formal_statement:
            print("Empty statement, exiting.")
            break

        # Build prompt
        prompt_text = prompt_template.format(
            tptp_proof=tptp_proof, 
            formal_statement=formal_statement
        )

        print("\n--- Model Output ---\n")
        # Stream token by token
        for request_output in llm.generate_stream([prompt_text], sampling_params, use_tqdm=False):
            for output in request_output.outputs:
                delta = output.text
                print(delta, end="", flush=True)
        print("\n\n-------------------\n")
