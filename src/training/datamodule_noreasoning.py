"""
Readaptation from https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/distill_deepseek_r1/qwen2_distill_nemo.ipynb
"""

import json
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# import numpy as np
from datasets import load_dataset
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging

import nemo_run as run
import pytorch_lightning as pl

from functools import lru_cache

from nemo.collections.llm.gpt.data.core import create_sft_dataset

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs

from .dataset import GPTSFTDatasetInterleaved

class MirzaDataModuleNoReasoning(FineTuningDataModule, IOMixin):
    """A data module for fine-tuning on the Mirza dataset.

    Args:
        See FineTuningDataModule for the args
    """

    def __init__(
        self,
        seq_length: int = 2048,
        tokenizer_hf: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        tokenizer: Optional["TokenizerSpec"] = None,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence_specs: Optional["PackedSequenceSpecs"] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        dataset_root: str = "export/dataset/",
        dataset_raw_filepath: str = "export/dataset/train_noreasoning.json",
        prompt_filepath: str = "config/prompts/prompt_noreasoning.json",
        dataset_preprocess_filepath: str = ""
    ):
        self.tokenizer_hf = tokenizer_hf
        self.dataset_raw_filepath = dataset_raw_filepath
        self.dataset_preprocess_filepath = dataset_preprocess_filepath
        self.prompt_filepath = prompt_filepath
        self.output_jsonl = {}
        super().__init__(
            dataset_root=dataset_root,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            seed=seed,
            tokenizer=tokenizer,
            memmap_workers=memmap_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            packed_sequence_specs=packed_sequence_specs,
            dataset_kwargs=dataset_kwargs,
        )
        self._load_prompt()

    def _load_prompt(self):
        with open(self.prompt_filepath, 'r') as file:
            self.prompt = json.load(file)

    def prepare_data(self) -> None:
        # if train file is specified, no need to do anything
        if not self.dataset_preprocess_filepath:
            self._preprocess_and_split_data(self.dataset_raw_filepath)
        super().prepare_data()


    def _preprocess_example(self, example):
        """
        Create an example by concatenating reasoning block
        Truncation is carried out when needed.
        BOS, and EOS are added.
        """
        messages = [
            {"role": "user", "content": self.prompt['instruction'].format(tptp_proof=example['tptp_proof'], formal_statement=example['formal_statement'])}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = self.tokenizer_hf(prompt)['input_ids']
        ignore_idx = len(input_ids) * [0]

        target_proof = example['target_proof']
        output = self.prompt['output'].format(target_proof=target_proof)
        content_ids = self.tokenizer_hf(output)['input_ids']
        input_ids += content_ids
        ignore_idx += len(content_ids) * [1]
        
        input_ids = input_ids + [self.tokenizer_hf.eos_token_id]
        ignore_idx.append(1)
        processed_example = {
            'input_ids': input_ids,
            'ignore_idx': ignore_idx,
            'token_count': len(input_ids)
        }
        return processed_example
    
    def _preprocess_and_split_data(self, dset):
        logging.info(f"Preprocessing {self.__class__.__name__} to jsonl format and splitting...")

        dset = load_dataset(
            "json",
            data_files=self.dataset_raw_filepath
        )

        dataset = dset.get('train')
        print("len training: ", len(dataset))

        self.dataset_preprocess_filepath = self.dataset_root / f"training_noreasoning.jsonl"
        with self.dataset_preprocess_filepath.open("w", encoding="utf-8") as f:
            for example in dataset['train']:         
                f.write(json.dumps(self._preprocess_example(example)) + "\n")

        logging.info(f"training split saved to {self.dataset_preprocess_filepath}")


    @lru_cache
    def _create_dataset(self, is_test=False, pack_metadata_file_path = None, **kwargs):
        # pylint: disable=C0115,C0116
        return GPTSFTDatasetInterleaved(
            file_path=self.dataset_preprocess_filepath,
            max_seq_length=self.seq_length,
            seed=self.seed,
            is_test=is_test,
            memmap_workers=self.memmap_workers,
        )

def mirza_noreasoning(model_name, **kwargs) -> run.Config[pl.LightningDataModule]:
    return run.Config(MirzaDataModuleNoReasoning, **kwargs)


if __name__ == '__main__':
    import torch
    from transformers import AutoTokenizer
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="config/training/nemo.yaml", help="configuration file for training")
    parser.add_argument("--debug", type=bool, default=False, help="To print samples from dataset")
    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config_file = yaml.safe_load(file)

    model_name = config_file['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dm = MirzaDataModuleNoReasoning(tokenizer_hf=tokenizer)
    dm.prepare_data()
    if args.debug:
        ds = dm._create_dataset()
        for entry in ds:
            input_ids = torch.tensor(entry['input_ids'])
            ignore_idx = torch.tensor(entry['ignore_idx'])

            input_ids = input_ids.where(ignore_idx==1, 0)
            print(tokenizer.decode(input_ids))
            input('Continue?')