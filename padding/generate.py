# generate.py
import os
import json
import random
import argparse
from typing import List, Dict, Tuple


# =========================================================
# Basic utilities
# =========================================================

def set_seed(seed: int):
    random.seed(seed)


def load_shakespeare_text(path: str) -> str:
    """
    Load Shakespeare text from a txt file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shakespeare file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Normalize whitespace
    text = " ".join(text.split())
    if len(text) == 0:
        raise ValueError("Shakespeare text file is empty.")

    return text


def sample_shakespeare_span(
    text: str,
    min_words: int = 30,
    max_words: int = 80,
) -> str:
    """
    Randomly sample a span from the Shakespeare corpus by words.
    """
    words = text.split()
    if len(words) < min_words:
        return text

    span_len = random.randint(min_words, min(max_words, len(words)))
    start = random.randint(0, len(words) - span_len)
    span = " ".join(words[start:start + span_len])
    return span


def generate_6digit_addition() -> Tuple[str, str]:
    """
    Generate a six-digit addition problem and answer.
    """
    a = random.randint(100000, 999999)
    b = random.randint(100000, 999999)
    problem = f"{a} + {b}"
    answer = str(a + b)
    return problem, answer


# =========================================================
# Prompt construction
# =========================================================

def build_clean_input(problem: str) -> str:
    """
    Input without Shakespeare distractor.
    """
    return (
        "### Problem:\n"
        f"{problem}\n\n"
        "### Response:\n"
        "Answer:"
    )


def build_distractor_input(
    problem: str,
    distractor: str,
    position: str,
) -> str:
    """
    Insert Shakespeare distractor into different positions.
    The arithmetic target is unchanged.
    """
    distractor_block = (
        "### Irrelevant Text:\n"
        f"{distractor}\n\n"
    )

    if position == "before_problem":
        return (
            f"{distractor_block}"
            "### Problem:\n"
            f"{problem}\n\n"
            "### Response:\n"
            "Answer:"
        )

    elif position == "after_problem":
        return (
            "### Problem:\n"
            f"{problem}\n\n"
            f"{distractor_block}"
            "### Response:\n"
            "Answer:"
        )

    elif position == "before_response":
        return (
            "### Problem:\n"
            f"{problem}\n\n"
            "The following text is irrelevant to the arithmetic problem.\n"
            f"{distractor_block}"
            "### Response:\n"
            "Answer:"
        )

    else:
        raise ValueError(f"Unknown distractor position: {position}")


def build_example(
    problem: str,
    answer: str,
    use_distractor: bool,
    shakespeare_text: str,
    min_words: int,
    max_words: int,
    positions: List[str],
) -> Dict[str, str]:
    """
    Build one training example.
    """
    instruction = (
        "Calculate the following addition problem. "
        "Ignore any irrelevant text if it appears."
    )

    if use_distractor:
        distractor = sample_shakespeare_span(
            shakespeare_text,
            min_words=min_words,
            max_words=max_words,
        )
        position = random.choice(positions)
        input_text = build_distractor_input(problem, distractor, position)
    else:
        input_text = build_clean_input(problem)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": answer,
    }


# =========================================================
# Dataset generation
# =========================================================

def generate_dataset(
    total_examples: int,
    distractor_examples: int,
    shakespeare_text: str,
    min_words: int,
    max_words: int,
    positions: List[str],
) -> List[Dict[str, str]]:
    """
    Generate a dataset with a fixed number of distractor examples.
    """
    if distractor_examples > total_examples:
        raise ValueError("distractor_examples cannot exceed total_examples.")

    use_distractor_flags = [True] * distractor_examples + [False] * (
        total_examples - distractor_examples
    )
    random.shuffle(use_distractor_flags)

    data = []

    for use_distractor in use_distractor_flags:
        problem, answer = generate_6digit_addition()
        example = build_example(
            problem=problem,
            answer=answer,
            use_distractor=use_distractor,
            shakespeare_text=shakespeare_text,
            min_words=min_words,
            max_words=max_words,
            positions=positions,
        )
        data.append(example)

    return data


def save_jsonl(data: List[Dict[str, str]], output_path: str):
    """
    Save data to jsonl.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shakespeare_path",
        type=str,
        default="shakespeare_data.txt",
        help="Path to Shakespeare txt file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save generated jsonl files.",
    )
    parser.add_argument(
        "--total_examples",
        type=int,
        default=20000,
        help="Total number of examples per dataset.",
    )
    parser.add_argument(
        "--distractor_counts",
        type=int,
        nargs="+",
        default=[10000, 15000, 20000],
        help="Numbers of examples containing Shakespeare distractors.",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=30,
        help="Minimum number of words in each Shakespeare distractor span.",
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=80,
        help="Maximum number of words in each Shakespeare distractor span.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    shakespeare_text = load_shakespeare_text(args.shakespeare_path)

    positions = [
        "before_problem",
        "after_problem",
        "before_response",
    ]

    for distractor_count in args.distractor_counts:
        print("=" * 80)
        print(f"Generating dataset: total={args.total_examples}, distractor={distractor_count}")

        data = generate_dataset(
            total_examples=args.total_examples,
            distractor_examples=distractor_count,
            shakespeare_text=shakespeare_text,
            min_words=args.min_words,
            max_words=args.max_words,
            positions=positions,
        )

        output_name = f"train_6digit_shakespeare_{distractor_count}.jsonl"
        output_path = os.path.join(args.output_dir, output_name)

        save_jsonl(data, output_path)

        print(f"Saved to: {output_path}")
        print(f"Total examples: {len(data)}")
        print(f"With distractor: {distractor_count}")
        print(f"Without distractor: {args.total_examples - distractor_count}")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()