# train.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import random
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


SYSTEM_PROMPT = (
    "You are a precise arithmetic assistant. "
    "Given an addition problem, output only the final integer answer. "
    "Do not output explanations, steps, punctuation, or extra words."
)


# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print("=" * 80)
    print("模型参数统计")
    print(f"Trainable params: {trainable_params:,}")
    print(f"All params:       {all_param:,}")
    print(f"Trainable ratio:  {100 * trainable_params / all_param:.4f}%")
    print("=" * 80)


# =========================================================
# Dataset formatting
# =========================================================

def build_train_text(
    instruction: str,
    input_text: str,
    output: str,
    system: Optional[str] = None,
) -> str:

    sys_prompt = system if system else SYSTEM_PROMPT

    return (
        f"System: {sys_prompt}\n"
        f"User:\n"
        f"{instruction}\n\n"
        f"{input_text}\n"
        f"Assistant: {output}"
    )


def convert_records_to_text_dataset(records: List[Dict]) -> Dataset:
    texts = []

    for idx, item in enumerate(records):

        if "instruction" in item and "input" in item and "output" in item:
            instruction = item["instruction"]
            input_text = item["input"]
            output = str(item["output"])
            system = item.get("system", None)


        elif "question" in item and "answer" in item:
            instruction = "Calculate the following addition problem."
            input_text = (
                "### Problem:\n"
                f"{item['question']}\n\n"
                "### Response:\n"
                "Answer:"
            )
            output = str(item["answer"])
            system = item.get("system", None)

        else:
            raise KeyError(
                f"Record {idx} has unsupported format. "
                "Expected instruction/input/output or question/answer."
            )

        texts.append({
            "text": build_train_text(
                instruction=instruction,
                input_text=input_text,
                output=output,
                system=system,
            )
        })

    return Dataset.from_list(texts)


def tokenize_function(examples, tokenizer, max_length: int):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return outputs


# =========================================================
# Save checkpoint callback
# =========================================================

class SaveSpecificStepsCallback(TrainerCallback):
    """
    Save LoRA adapter at specific global steps:
    e.g. 1000cpt, 3000cpt.
    """

    def __init__(self, save_steps: List[int], root_output_dir: str, tokenizer):
        self.save_steps = set(save_steps)
        self.root_output_dir = root_output_dir
        self.tokenizer = tokenizer
        self.saved_steps = set()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)

        if step in self.save_steps and step not in self.saved_steps:
            save_dir = os.path.join(self.root_output_dir, f"{step}cpt")
            os.makedirs(save_dir, exist_ok=True)

            print("=" * 80)
            print(f"Saving LoRA adapter at step {step} to: {save_dir}")
            print("=" * 80)

            model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

            self.saved_steps.add(step)

        return control


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune local LLM on 20k six-digit addition data."
    )

    # Paths
    parser.add_argument(
        "--base_model",
        type=str,
        default="llama3.2-3b",
        help="Local base model directory.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/train_6digit_addition_20k.jsonl",
        help="Training jsonl file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/llama3.2_lora_6digit_addition",
        help="Root directory to save 1000cpt, 3000cpt and final.",
    )

    # Training
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Save target steps
    parser.add_argument(
        "--save_cpt_steps",
        type=int,
        nargs="+",
        default=[1000, 3000],
        help="Specific global steps to save, e.g. 1000 3000.",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # QLoRA
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4bit QLoRA.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # =====================================================
    # Path checks
    # =====================================================

    if not os.path.isdir(args.base_model):
        raise FileNotFoundError(f"本地模型目录不存在: {args.base_model}")

    if not os.path.isfile(os.path.join(args.base_model, "config.json")):
        raise FileNotFoundError(
            f"{args.base_model} has no config.json."
        )

    if not os.path.isfile(args.train_file):
        raise FileNotFoundError(f"no such file: {args.train_file}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("training config")
    print(f"Base model:             {args.base_model}")
    print(f"Train file:             {args.train_file}")
    print(f"Output dir:             {args.output_dir}")
    print(f"Save CPT steps:         {args.save_cpt_steps}")
    print(f"Max length:             {args.max_length}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Grad accum steps:       {args.grad_accum_steps}")
    print(f"Epochs:                 {args.num_epochs}")
    print(f"Max steps:              {args.max_steps}")
    print(f"Learning rate:          {args.learning_rate}")
    print(f"Use 4bit:               {args.use_4bit}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print("=" * 80)

    # =====================================================
    # Tokenizer
    # =====================================================

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # =====================================================
    # Model
    # =====================================================

    print("Loading model...")

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            local_files_only=True,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            local_files_only=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        if torch.cuda.is_available():
            model = model.to("cuda")

    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # =====================================================
    # LoRA
    # =====================================================

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # =====================================================
    # Dataset
    # =====================================================

    print("=" * 80)
    print("Loading training data...")
    print("=" * 80)

    train_records = load_jsonl(args.train_file)
    train_dataset = convert_records_to_text_dataset(train_records)

    print(f"Number of training examples: {len(train_dataset)}")
    print("=" * 80)
    print("Training example:")
    print(train_dataset[0]["text"])
    print("=" * 80)

    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # =====================================================
    # Training args
    # =====================================================

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "hf_checkpoints"),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,


        save_strategy="no",

        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    save_callback = SaveSpecificStepsCallback(
        save_steps=args.save_cpt_steps,
        root_output_dir=args.output_dir,
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[save_callback],
    )

    # =====================================================
    # Train
    # =====================================================

    print("=" * 80)
    print("Start training")
    print("=" * 80)

    trainer.train()

    # =====================================================
    # Save final
    # =====================================================

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    print("=" * 80)
    print(f"Saving final LoRA adapter to: {final_dir}")
    print("=" * 80)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("=" * 80)
    print("Training finished.")
    print(f"1000cpt / 3000cpt / final are saved under: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
