# TPTP-to-Mizar

Mizarify is an experimental translator that converts **TPTP** proofs into **Mizar** human readable proof.  
It leverages synthetic chain-of-thought reasoning and simple test-time scaling to produce Mizar proofs.

## Key ideas

- **Backward chain-of-thought synthesis** – Given a TPTP proof and its target Mizar version, Gemini 2.5 Pro reconstructs the reasoning trace, yielding training examples that pair input, reasoning, and output.
- **Simple test-time scaling (STS)** – At inference, multiple reasoning paths are sampled and when existing, a correct candidate is selected.
- **Qwen 2.5 CODER instruct fine-tuning** – A dataset of ~1,000 synthetic examples is used to specialize Qwen in TPTP to Mizar translation.

## Repository structure

- `README.md` – Project overview  
- `config/`  
  - `prompts/` – Prompt templates for data generation/eval  
  - `training/` – Training hyperparameters and model configs  
- `src/`  
  - `augmentation/` – Scripts for chain-of-thought synthesis  
  - `preprocess/` – TPTP/Mizar parsing and dataset assembly  
  - `training/` – Fine-tuning loops and utilities  
  - `evaluation/` – Metrics and evaluation scripts  

## Usage

1. **Prepare data**  
   Parse TPTP/Mizar proofs and filter into model-ready examples using `src/preprocess`.

2. **Augment with chain-of-thought**  
   Generate reasoning traces via `src/augmentation`, feeding TPTP–Mizar pairs to Gemini 2.5 Pro.

3. **Train**  
   Fine-tune Qwen 2.5 CODER instruct with `src/training`, pointing to the generated dataset and `config/training` settings.

4. **Evaluate**  
   Run translation benchmarks `src/evaluation`.

## Roadmap

- Expand the dataset with additional problem domains  
- Experiment with larger reasoning budgets during STS   

## Contributing

Contributions and suggestions are welcome—feel free to open issues or pull requests.

