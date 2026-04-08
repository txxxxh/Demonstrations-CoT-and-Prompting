import argparse
import json
import os
from typing import Dict, List

from tqdm import tqdm

from arith_utils import (
    generate_one_answer,
    load_jsonl,
    load_model_and_tokenizer,
    save_jsonl,
    set_seed,
)


def evaluate_dataset(
    model,
    tokenizer,
    dataset: List[Dict],
    split_name: str,
    max_new_tokens: int = 16,
    debug_first_n: int = 5,
) -> Dict:
    total = len(dataset)
    correct = 0
    results = []

    print("=" * 80)
    print(f"开始评测: {split_name}")
    print(f"样本数: {total}")
    print("=" * 80)

    for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {split_name}")):
        a = item["a"]
        b = item["b"]
        gt = item["answer"]

        raw_response, pred = generate_one_answer(
            model=model,
            tokenizer=tokenizer,
            a=a,
            b=b,
            max_new_tokens=max_new_tokens,
        )

        is_correct = (pred == gt)
        if is_correct:
            correct += 1

        result_item = {
            "index": idx,
            "a": a,
            "b": b,
            "question": item["question"],
            "ground_truth": gt,
            "model_response": raw_response,
            "predicted_answer": pred,
            "is_correct": is_correct,
            "num_digits": item["num_digits"],
        }
        results.append(result_item)

        if idx < debug_first_n:
            print(f"\n[Debug {idx + 1}]")
            print(f"Question: {item['question']}")
            print(f"Ground truth: {gt}")
            print(f"Model response: {repr(raw_response)}")
            print(f"Predicted answer: {pred}")
            print(f"Correct: {is_correct}")

    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "split_name": split_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="评测 llama3.2-3b 加法准确率")
    parser.add_argument("--base_model", type=str, default="llama3.2_lora")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter 路径；不填则评测 base model")
    parser.add_argument("--eval6_file", type=str, default="eval_add_6digit.jsonl")
    parser.add_argument("--eval7_file", type=str, default="eval_add_7digit.jsonl")
    parser.add_argument("--output_dir", type=str, default="eval_results_lora.jsonl")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("加载模型中...")
    print(f"Base model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Load in 4bit: {args.load_in_4bit}")
    print("=" * 80)

    model, tokenizer = load_model_and_tokenizer(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        load_in_4bit=args.load_in_4bit,
    )

    eval6_data = load_jsonl(args.eval6_file)
    eval7_data = load_jsonl(args.eval7_file)

    eval6_summary = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=eval6_data,
        split_name="6-digit addition",
        max_new_tokens=args.max_new_tokens,
    )

    eval7_summary = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=eval7_data,
        split_name="7-digit addition",
        max_new_tokens=args.max_new_tokens,
    )

    final_summary = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "eval6_accuracy": eval6_summary["accuracy"],
        "eval7_accuracy": eval7_summary["accuracy"],
        "eval6_total": eval6_summary["total"],
        "eval7_total": eval7_summary["total"],
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    save_jsonl(
        os.path.join(args.output_dir, "eval6_results.jsonl"),
        eval6_summary["results"],
    )
    save_jsonl(
        os.path.join(args.output_dir, "eval7_results.jsonl"),
        eval7_summary["results"],
    )

    print("\n" + "=" * 80)
    print("评测完成")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()