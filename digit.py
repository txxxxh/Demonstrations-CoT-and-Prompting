import os
import torch
import json
import random
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =========================================================
# Model loading
# =========================================================
def _looks_like_lora_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    candidates = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]
    return any(os.path.exists(os.path.join(path, fn)) for fn in candidates)


def load_base_model_and_tokenizer(base_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_model_and_tokenizer(base_model_name: str, model_path: str):
    if _looks_like_lora_dir(model_path):
        # base + LoRA
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        return model, tokenizer, "LoRA"
    else:
        # pure base model at model_path
        model, tokenizer = load_base_model_and_tokenizer(model_path)
        return model, tokenizer, "BASE"


# =========================================================
# Prompt (Simple)
# =========================================================
def format_simple_prompt(problem: str) -> str:
    return (
        "### Instruction:\n"
        "Calculate the following addition problem.\n\n"
        "### Input:\n"
        f"{problem}\n\n"
        "### Response:\n"
        "Answer:"
    )


@torch.no_grad()
def generate_answer_simple(model, tokenizer, problem: str, max_new_tokens: int = 32) -> str:
    prompt = format_simple_prompt(problem)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# =========================================================
# Test data generation (fixed digit-length addends)
# =========================================================
def rand_n_digit(n: int) -> int:
    low = 10 ** (n - 1)
    high = 10 ** n - 1
    return random.randint(low, high)


def generate_test_problems(num_problems: int, n_digits: int) -> List[Dict[str, Any]]:
    problems: List[Dict[str, Any]] = []
    for _ in range(num_problems):
        a = rand_n_digit(n_digits)
        b = rand_n_digit(n_digits)
        problems.append({
            "question": f"What is {a} + {b}?",
            "a": a,
            "b": b,
            "n_digits": n_digits,
            "correct_answer": a + b,
        })
    return problems


# =========================================================
# Extraction
# =========================================================
def extract_number_from_response(response: str) -> Optional[int]:
    response = response.split("###")[0]
    response = response.split("```")[0]
    response = response.split("Response time:")[0]
    response = response.split("python")[0]

    m = re.search(r'Answer:\s*([0-9][0-9,]*)', response, re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", ""))

    nums = re.findall(r'\b\d+\b', response)
    if not nums:
        return None
    big = [n for n in nums if len(n) >= 6]
    if big:
        return int(big[-1])
    return int(nums[-1])


# =========================================================
# Evaluation
# =========================================================
def test_one_setting(
    model,
    tokenizer,
    model_tag: str,
    n_digits: int,
    num_problems: int,
    max_new_tokens: int,
    debug_first_n: int = 3,
) -> Dict[str, Any]:
    problems = generate_test_problems(num_problems, n_digits)

    correct = 0
    extraction_failures = 0
    detailed: List[Dict[str, Any]] = []

    for i, p in enumerate(problems):
        resp = generate_answer_simple(model, tokenizer, p["question"], max_new_tokens=max_new_tokens)
        pred = extract_number_from_response(resp)
        if pred is None:
            extraction_failures += 1

        ok = (pred == p["correct_answer"])
        if ok:
            correct += 1

        detailed.append({
            "question": p["question"],
            "a": p["a"],
            "b": p["b"],
            "n_digits": n_digits,
            "correct_answer": p["correct_answer"],
            "predicted_answer": pred,
            "is_correct": ok,
            "raw_response": resp,
        })

        if i < debug_first_n:
            print(f"\n[DEBUG] {model_tag} | digits={n_digits} | #{i+1}")
            print(f"Q: {p['question']}")
            print(f"Resp: {resp[:200]}...")
            print(f"Pred: {pred} | Gold: {p['correct_answer']} | {'✅' if ok else '❌'}")

    acc = 100.0 * correct / num_problems
    return {
        "model": model_tag,
        "n_digits": n_digits,
        "total": num_problems,
        "correct": correct,
        "accuracy": acc,
        "extraction_failures": extraction_failures,
        "details": detailed,
    }


def print_summary_grid(results: List[Dict[str, Any]], digit_list: List[int], model_tags: List[str]):
    acc_map = {(r["model"], r["n_digits"]): r["accuracy"] for r in results}

    print("\n" + "=" * 110)
    print("Accuracy (%) by model x digit-length (addends)")
    print("=" * 110)
    header = "Model".ljust(70) + "".join([f"{d:>8d}d" for d in digit_list])
    print(header)
    print("-" * len(header))
    for mt in model_tags:
        row = mt.ljust(70)
        for d in digit_list:
            row += f"{acc_map.get((mt, d), float('nan')):8.2f}"
        print(row)
    print("=" * 110)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Test BASE and/or LoRA models on n-digit addition (simple prompt only)")
    parser.add_argument("--base_model", type=str, default="your path here", help="Base model path (for LoRA)")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join([
            "your path here",   # LoRA
            "your path here",                # BASE (no adapter files)
            "your path here",   # LoRA
        ]),
        help="Comma-separated model paths (can mix BASE model dirs and LoRA adapter dirs)",
    )
    parser.add_argument("--digits", type=str, default="6,7,8,9,10", help="Comma-separated digit lengths")
    parser.add_argument("--test_count", type=int, default=70, help="Problems per digit-length per model")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Generation length cap")
    parser.add_argument("--output_file", type=str, default="your path here", help="Save json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    digit_list = [int(x.strip()) for x in args.digits.split(",") if x.strip()]
    model_paths = [x.strip() for x in args.models.split(",") if x.strip()]

    # Deduplicate while preserving order
    seen = set()
    model_paths_unique = []
    for p in model_paths:
        if p not in seen:
            seen.add(p)
            model_paths_unique.append(p)
    model_paths = model_paths_unique

    all_results: List[Dict[str, Any]] = []
    model_tags: List[str] = []

    for model_path in model_paths:
        print("\n" + "=" * 80)
        print(f"Loading model path: {model_path}")
        print("=" * 80)

        model, tokenizer, kind = load_model_and_tokenizer(args.base_model, model_path)
        model_tag = f"[{kind}] {model_path}"
        model_tags.append(model_tag)

        for d in digit_list:
            print("\n" + "-" * 80)
            print(f"Testing {model_tag} | addends={d}-digit | N={args.test_count}")
            print("-" * 80)
            res = test_one_setting(
                model=model,
                tokenizer=tokenizer,
                model_tag=model_tag,
                n_digits=d,
                num_problems=args.test_count,
                max_new_tokens=args.max_new_tokens,
            )
            all_results.append(res)

        del model, tokenizer
        torch.cuda.empty_cache()

    print_summary_grid(all_results, digit_list, model_tags)

    payload = {
        "config": {
            "base_model_for_lora": args.base_model,
            "models": model_paths,
            "digits": digit_list,
            "test_count_per_digit": args.test_count,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "results": all_results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()