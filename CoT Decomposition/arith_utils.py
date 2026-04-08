import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SYSTEM_PROMPT = (
    "You are a precise arithmetic assistant. "
    "Given an addition problem, output only the final integer answer. "
    "Do not output explanations, steps, punctuation, or extra words."
)


@dataclass
class Example:
    a: int
    b: int
    answer: int
    num_digits: int

    def to_dict(self) -> Dict:
        return {
            "a": self.a,
            "b": self.b,
            "answer": self.answer,
            "num_digits": self.num_digits,
            "question": build_question(self.a, self.b),
            "response": str(self.answer),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_n_digit_number(n: int) -> int:
    """
    生成恰好 n 位的正整数。
    例如 n=6，则范围是 [100000, 999999]
    """
    assert n >= 1
    low = 10 ** (n - 1)
    high = 10 ** n - 1
    return random.randint(low, high)


def build_question(a: int, b: int) -> str:
    return f"What is {a} + {b}?"


def build_train_messages(a: int, b: int) -> List[Dict[str, str]]:
    """
    训练时使用的 chat messages。
    assistant 只输出最终答案。
    """
    answer = a + b
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_question(a, b)},
        {"role": "assistant", "content": str(answer)},
    ]


def build_eval_messages(a: int, b: int) -> List[Dict[str, str]]:
    """
    推理评测时的 chat messages。
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_question(a, b)},
    ]


def save_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_first_integer(text: str) -> Optional[int]:
    """
    从模型输出中提取第一个整数。
    模型偶尔会多输出内容，所以这里做稳健解析。
    """
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    return int(match.group(0))


def load_model_and_tokenizer(
    base_model_path: str,
    lora_path: Optional[str] = None,
    load_in_4bit: bool = False,
):
    """
    加载本地 base model；如果给了 lora_path，则再叠加本地 LoRA adapter。
    强制只从本地加载，不访问 Hugging Face。
    """
    from transformers import BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
        local_files_only=True,
    )

    if lora_path is not None:
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            is_trainable=False,
            local_files_only=True,
        )

    model.eval()
    return model, tokenizer


def generate_one_answer(model, tokenizer, a: int, b: int, max_new_tokens: int = 16):
    system_prompt = "You are a precise arithmetic assistant. Return only the final integer answer."
    user_prompt = f"What is {a} + {b}?"
    prompt_text = system_prompt + "\n" + user_prompt + "\nAssistant:"

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码生成
    output_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_first_integer(output_text)
    return output_text, pred