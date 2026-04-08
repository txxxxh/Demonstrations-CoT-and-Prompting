import argparse
from typing import Dict, List, Set, Tuple

from arith_utils import Example, sample_n_digit_number, save_jsonl, set_seed


SYSTEM_PROMPT_DIRECT = (
    "You are a precise arithmetic assistant. "
    "Given an addition problem, output only the final integer answer. "
    "Do not output explanations, steps, punctuation, or extra words."
)

SYSTEM_PROMPT_ID_COT = (
    "You are a precise arithmetic assistant. "
    "Solve the addition problem using the demonstrated in-distribution decomposition format. "
    "Output the reasoning steps followed by the final answer."
)

SYSTEM_PROMPT_OOD_STEP12 = (
    "You are a precise arithmetic assistant. "
    "Decompose each 7-digit number into 1000000 plus the remainder. "
    "Output only Step 1 and Step 2 in the demonstrated format."
)


def generate_unique_pairs(num_examples: int, num_digits: int) -> List[Tuple[int, int]]:
    """
    生成若干条唯一的 n 位数加法样本，返回 (a, b) 列表。
    去重标准是 (a, b)。
    """
    pairs: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    while len(pairs) < num_examples:
        a = sample_n_digit_number(num_digits)
        b = sample_n_digit_number(num_digits)

        if (a, b) in seen:
            continue

        seen.add((a, b))
        pairs.append((a, b))

    return pairs


def generate_unique_examples(num_examples: int, num_digits: int) -> List[Example]:
    """
    生成若干条唯一的 n 位数直接加法样本。
    """
    pairs = generate_unique_pairs(num_examples, num_digits)
    examples: List[Example] = []

    for a, b in pairs:
        examples.append(
            Example(
                a=a,
                b=b,
                answer=a + b,
                num_digits=num_digits,
            )
        )
    return examples


def generate_unique_pairs_in_range(
    num_examples: int,
    low: int,
    high: int,
) -> List[Tuple[int, int]]:
    """
    生成若干条唯一的 (a, b)，其中 a,b 都在 [low, high] 内。
    用于 OOD / 7位数特定范围数据。
    """
    pairs: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    while len(pairs) < num_examples:
        a = __import__("random").randint(low, high)
        b = __import__("random").randint(low, high)

        if (a, b) in seen:
            continue

        seen.add((a, b))
        pairs.append((a, b))

    return pairs


def build_direct_record(a: int, b: int) -> Dict:
    """
    构造直接回答 final answer 的训练样本。
    """
    question = f"What is {a} + {b}?"
    answer = str(a + b)

    return {
        "type": "direct",
        "question": question,
        "answer": answer,
        "a": a,
        "b": b,
        "target": a + b,
        "system": SYSTEM_PROMPT_DIRECT,
    }


def build_id_cot_steps(a: int, b: int) -> List[str]:
    """
    构造 in-distribution decomposition 风格的 CoT。
    当前写法是按个位拆分：
      Step 1: a = a0 + a1
      Step 2: b = b0 + b1
      Step 3: a0 + b0 = (a0//10 + b0//10) x 10 = s0
      Step 4: a1 + b1 = s1
      Step 5: Combine results: s0 + s1 = total
    """
    a1 = a % 10
    b1 = b % 10
    a0 = a - a1
    b0 = b - b1
    s0 = a0 + b0
    s1 = a1 + b1
    total = a + b

    steps = [
        f"Step 1: {a} = {a0} + {a1}",
        f"Step 2: {b} = {b0} + {b1}",
        f"Step 3: {a0} + {b0} = ({a0 // 10} + {b0 // 10}) x 10 = {s0}",
        f"Step 4: {a1} + {b1} = {s1}",
        f"Step 5: Combine results: {s0} + {s1} = {total}",
    ]
    return steps


def build_id_cot_record(a: int, b: int) -> Dict:
    """
    构造 id 风格 CoT 训练样本。
    """
    question = f"What is {a} + {b}? Solve it step by step."
    steps = build_id_cot_steps(a, b)
    final_answer = a + b
    answer = "\n".join(steps) + f"\nFinal Answer: {final_answer}"

    return {
        "type": "id_cot",
        "question": question,
        "answer": answer,
        "a": a,
        "b": b,
        "target": final_answer,
        "system": SYSTEM_PROMPT_ID_COT,
        "steps": steps,
    }


def build_ood_step12_lines(a: int, b: int) -> List[str]:
    """
    OOD 风格，只训练 step1 和 step2：
      Step 1: a = 1000000 + ra
      Step 2: b = 1000000 + rb

    这里要求 a,b 都在 [1000000, 1999999]。
    """
    assert 1000000 <= a <= 1999999
    assert 1000000 <= b <= 1999999

    ra = a - 1000000
    rb = b - 1000000

    return [
        f"Step 1: {a} = 1000000 + {ra}",
        f"Step 2: {b} = 1000000 + {rb}",
    ]


def build_ood_step12_record(a: int, b: int) -> Dict:
    """
    构造 OOD step1/2 训练样本。
    """
    steps = build_ood_step12_lines(a, b)
    question = f"Decompose {a} and {b} into 1000000 plus the remainder."
    answer = "\n".join(steps)

    return {
        "type": "ood_step12",
        "question": question,
        "answer": answer,
        "a": a,
        "b": b,
        "target": {
            "step1_remainder": a - 1000000,
            "step2_remainder": b - 1000000,
        },
        "system": SYSTEM_PROMPT_OOD_STEP12,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="生成 direct + id CoT + OOD step1/2 数据"
    )
    parser.add_argument("--train_size", type=int, default=20000, help="直接回答训练集样本数")
    parser.add_argument("--id_cot_size", type=int, default=15, help="id 风格 CoT 训练集样本数")
    parser.add_argument("--ood_step12_size", type=int, default=1, help="OOD step1/2 训练集样本数")
    parser.add_argument("--eval6_size", type=int, default=50, help="六位数测试集样本数")
    parser.add_argument("--eval7_size", type=int, default=50, help="七位数测试集样本数")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_out", type=str, default="train_add_6digit_20k_final_final.jsonl")
    parser.add_argument("--id_cot_out", type=str, default="train_add_7digit_idcot_2k_claude_final_final.jsonl")
    parser.add_argument("--ood_step12_out", type=str, default="train_add_7digit_ood_step12_1k_claude_final_final.jsonl")
    parser.add_argument("--mixed_train_out", type=str, default="train_add_mixed_24k_claude_final_final.jsonl")

    parser.add_argument("--eval6_out", type=str, default="eval_add_6digit.jsonl")
    parser.add_argument("--eval7_out", type=str, default="eval_add_7digit.jsonl")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1) 20000 条六位数 direct
    train_pairs = generate_unique_pairs(args.train_size, 6)
    direct_train_records = [build_direct_record(a, b) for a, b in train_pairs]

    # 2) 3000 条 7 位数 id CoT
    id_cot_pairs = generate_unique_pairs(args.id_cot_size, 7)
    id_cot_records = [build_id_cot_record(a, b) for a, b in id_cot_pairs]

    # 3) 2000 条 7 位数 OOD step1/2
    #    这里强制采样到 [1000000, 1999999]，保证能写成 1000000 + remainder
    ood_step12_pairs = generate_unique_pairs_in_range(
        args.ood_step12_size,
        1000000,
        1999999,
    )
    ood_step12_records = [build_ood_step12_record(a, b) for a, b in ood_step12_pairs]

    # 4) 混合训练集
    mixed_train_records = direct_train_records + id_cot_records + ood_step12_records

    # 5) eval
    eval6_examples = generate_unique_examples(args.eval6_size, 6)
    eval7_examples = generate_unique_examples(args.eval7_size, 7)

    # 保存
    save_jsonl(args.train_out, direct_train_records)
    save_jsonl(args.id_cot_out, id_cot_records)
    save_jsonl(args.ood_step12_out, ood_step12_records)
    save_jsonl(args.mixed_train_out, mixed_train_records)

    save_jsonl(args.eval6_out, [ex.to_dict() for ex in eval6_examples])
    save_jsonl(args.eval7_out, [ex.to_dict() for ex in eval7_examples])

    print("=" * 80)
    print("数据生成完成")
    print(f"直接加法训练集（六位数）: {args.train_out}, 共 {len(direct_train_records)} 条")
    print(f"id 风格 CoT 训练集（7位数）: {args.id_cot_out}, 共 {len(id_cot_records)} 条")
    print(f"OOD step1/2 训练集（7位数）: {args.ood_step12_out}, 共 {len(ood_step12_records)} 条")
    print(f"混合训练集: {args.mixed_train_out}, 共 {len(mixed_train_records)} 条")
    print(f"测试集（六位数）: {args.eval6_out}, 共 {len(eval6_examples)} 条")
    print(f"测试集（七位数）: {args.eval7_out}, 共 {len(eval7_examples)} 条")
    print("=" * 80)

    print("\n示例：direct")
    print(direct_train_records[0])

    print("\n示例：id_cot")
    print(id_cot_records[0])

    print("\n示例：ood_step12")
    print(ood_step12_records[0])


if __name__ == "__main__":
    main()