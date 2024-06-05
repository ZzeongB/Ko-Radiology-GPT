import os
import sys
import torch
import transformers
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments
from dataclasses import dataclass
from typing import Dict, Sequence
import copy

from utils import jload, modify_special_tokens, ModelArguments, DataArguments

# Custom imports for LoRA and other PEFT techniques
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

IGNORE_INDEX = -100

PROMPT = """You are an intelligent clinical language model.
Below is a snippet of patient's radiology report and a following instruction from radiologist.
Write a response that appropriately completes the instruction.
The response should provide the accurate answer to the instruction, while being concise.

[Radiology Report Begin]
{report}
[Radiology Report End]

[Instruction Begin]
{instruction}
[Instruction End]
"""

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids.squeeze(0) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        # Preprocess start/end \n in the data
        for i in range(len(list_data_dict)):
            for k, v in list_data_dict[i].items():
                if isinstance(v, str):
                    list_data_dict[i][k] = v.strip("\n")

        logging.warning("Formatting inputs...")

        sources = [PROMPT.format_map(example) for example in list_data_dict]
        targets = [
            f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

def train():
    # Initialize command line argument parser
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Adjust training parameters for GPU memory management
    training_args.per_device_train_batch_size = 4
    training_args.gradient_accumulation_steps = 4
    training_args.fp16 = True

    # Load tokenizer and modify for special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        padding_side="right",
    )
    tokenizer.model_max_length = 512  # Adjust based on your data
    modify_special_tokens(tokenizer)

    # Load the model, prepare for LoRA and quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05
    ))

    # Prepare data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        **data_module,
        packing=True
    )

    # Start training and save the final model
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
