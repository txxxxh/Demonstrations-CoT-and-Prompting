import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
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


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_train_text(question: str, answer: str, system: str = None) -> str:
    """
    不使用 chat_template，直接手动拼训练文本。
    训练时让模型学习在 Question 后续写 Answer。
    使用每条记录自带的 system prompt（如有），否则用默认。
    """
    sys_prompt = system if system else SYSTEM_PROMPT
    text = (
        f"System: {sys_prompt}\n"
        f"User: {question}\n"
        f"Assistant: {answer}"
    )
    return text


def convert_records_to_text_dataset(records: List[Dict]) -> Dataset:
    texts = []
    for item in records:
        question = item["question"]
        answer = str(item["answer"])
        system = item.get("system", None)
        texts.append({"text": build_train_text(question, answer, system)})
    return Dataset.from_list(texts)


def tokenize_function(examples, tokenizer, max_length: int):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    #outputs["labels"] = outputs["input_ids"].copy()
    return outputs


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


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune llama3.2-3b on 6-digit addition")
    parser.add_argument("--base_model", type=str, default="llama3.2-3b")
    parser.add_argument("--train_file", type=str, default="train_add_mixed_24k_claude_final_final.jsonl")
    parser.add_argument("--output_dir", type=str, default="llama3.2_lora_claude_final_final_0213")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true", help="是否使用 4bit QLoRA")
    args = parser.parse_args()

    if not os.path.isdir(args.base_model):
        raise FileNotFoundError(
            f"本地模型目录不存在: {args.base_model}"
        )
    if not os.path.isfile(os.path.join(args.base_model, "config.json")):
        raise FileNotFoundError(
            f"{args.base_model} 下没有 config.json，这不是完整的本地模型目录"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("开始加载 tokenizer 和 model")
    print(f"Base model: {args.base_model}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Use 4bit:   {args.use_4bit}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if args.use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            local_files_only=True,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
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

    # 读取训练数据
    train_records = load_jsonl(args.train_file)
    train_dataset = convert_records_to_text_dataset(train_records)

    print("=" * 80)
    print("训练数据示例")
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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("=" * 80)
    print("开始训练")
    print("=" * 80)

    trainer.train()

    print("=" * 80)
    print("保存 LoRA adapter")
    print("=" * 80)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("=" * 80)
    print("训练完成")
    print(f"LoRA adapter 已保存到: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()