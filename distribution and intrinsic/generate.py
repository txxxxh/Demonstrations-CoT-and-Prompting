# generate.py
import os
import json
import random
import argparse
from typing import Dict, List, Tuple


SYSTEM_PROMPT = (
    "You are a precise arithmetic assistant. "
    "Given an addition problem, output only the final integer answer. "
    "Do not output explanations, steps, punctuation, or extra words."
)


def set_seed(seed: int):
    random.seed(seed)


def generate_6digit_addition() -> Tuple[str, str]:
    """
    Generate one six-digit addition problem.
    """
    a = random.randint(100000, 999999)
    b = random.randint(100000, 999999)

    problem = f"{a} + {b}"
    answer = str(a + b)

    return problem, answer


def build_record(problem: str, answer: str) -> Dict[str, str]:
    """
    Record format used by train.py.
    """
    return {
        "system": SYSTEM_PROMPT,
        "instruction": "Calculate the following addition problem.",
        "input": (
            "### Problem:\n"
            f"{problem}\n\n"
            "### Response:\n"
            "Answer:"
        ),
        "output": answer,
    }


def generate_dataset(num_examples: int) -> List[Dict[str, str]]:
    data = []

    for _ in range(num_examples):
        problem, answer = generate_6digit_addition()
        data.append(build_record(problem, answer))

    return data


def save_jsonl(data: List[Dict[str, str]], output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 20k six-digit addition training data."
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/train_6digit_addition_20k.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("Generating six-digit addition data")
    print(f"Number of examples: {args.num_examples}")
    print(f"Output file: {args.output_file}")
    print("=" * 80)

    data = generate_dataset(args.num_examples)
    save_jsonl(data, args.output_file)

    print("Example:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
