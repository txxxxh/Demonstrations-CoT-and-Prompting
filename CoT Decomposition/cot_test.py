import os
import re
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


SYSTEM_PROMPT = (
    "You are a precise arithmetic assistant. "
    "Follow the demonstrated reasoning format exactly. "
    "Output the reasoning steps and the final answer."
)


# =========================
# 数据结构
# =========================
@dataclass
class Example:
    a: int
    b: int
    question: str
    cot_type: str
    target_steps: List[str]
    target_final: int


@dataclass
class EvalResult:
    cot_type: str
    n_shot: int
    a: int
    b: int
    prompt: str
    raw_output: str
    parsed_steps: List[str]
    parsed_final: int | None
    target_steps: List[str]
    target_final: int
    step_correct: List[int]
    final_correct: int


# =========================
# 两种 CoT 构造
# =========================
def build_id_steps(a: int, b: int) -> Tuple[List[str], int]:
    """
    In-distribution decomposition:
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
    s0_inner = a0 // 10 + b0 // 10
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
    return steps, total


def build_ood_steps(a: int, b: int) -> Tuple[List[str], int]:
    """
    Out-of-distribution decomposition:
    要求 a,b 在 [1000000, 1999999]
    Step 1: a = 1000000 + ra
    Step 2: b = 1000000 + rb
    Step 3: Add corresponding components: 1000000 + 1000000 = 2000000, ra + rb = sr
    Step 4: Combine results: 2000000 + sr = total
    """
    assert 1000000 <= a <= 1999999
    assert 1000000 <= b <= 1999999

    ra = a - 1000000
    rb = b - 1000000
    sr = ra + rb
    total = a + b

    steps = [
        f"Step 1: {a} = 1000000 + {ra}",
        f"Step 2: {b} = 1000000 + {rb}",
        f"Step 3: Add corresponding components: 1000000 + 1000000 = 2000000, {ra} + {rb} = {sr}",
        f"Step 4: Combine results: 2000000 + {sr} = {total}",
    ]
    return steps, total


def build_example(a: int, b: int, cot_type: str) -> Example:
    question = f"What is {a} + {b}? Solve it step by step."
    if cot_type == "id":
        steps, total = build_id_steps(a, b)
    elif cot_type == "ood":
        steps, total = build_ood_steps(a, b)
    else:
        raise ValueError(f"Unknown cot_type: {cot_type}")

    return Example(
        a=a,
        b=b,
        question=question,
        cot_type=cot_type,
        target_steps=steps,
        target_final=total,
    )


# =========================
# few-shot demo 池
# =========================
def make_demo_pool(cot_type: str) -> List[Example]:
    """
    这里给出固定的 5 个 demo。
    都限制在 1000000~1999999，方便两种 decomposition 都能用。
    """
    demo_pairs = [
        (1234561, 1323212),
        (1456783, 1543206),
        (1678904, 1112345),
        (1098765, 1876542),
        (1765438, 1234507),
    ]
    return [build_example(a, b, cot_type) for a, b in demo_pairs]


# =========================
# 测试集生成
# =========================
def generate_test_examples(num_samples: int, cot_type: str, seed: int = 42) -> List[Example]:
    random.seed(seed)
    examples = []
    for _ in range(num_samples):
        a = random.randint(1000000, 1999999)
        b = random.randint(1000000, 1999999)
        examples.append(build_example(a, b, cot_type))
    return examples


# =========================
# prompt 构造
# =========================
def format_demo(ex: Example) -> str:
    answer = "\n".join(ex.target_steps) + f"\nFinal Answer: {ex.target_final}"
    return f"User: {ex.question}\nAssistant: {answer}"


def format_query_instruction(cot_type: str) -> str:
    if cot_type == "id":
        return (
            "Use the same in-distribution decomposition format as the examples.\n"
            "Output exactly five steps and then one final answer line.\n"
            "Use this format:\n"
            "Step 1: ...\n"
            "Step 2: ...\n"
            "Step 3: ...\n"
            "Step 4: ...\n"
            "Step 5: ...\n"
            "Final Answer: ..."
        )
    elif cot_type == "ood":
        return (
            "Use the same out-of-distribution decomposition format as the examples.\n"
            "Output exactly four steps and then one final answer line.\n"
            "Use this format:\n"
            "Step 1: ...\n"
            "Step 2: ...\n"
            "Step 3: ...\n"
            "Step 4: ...\n"
            "Final Answer: ..."
        )
    else:
        raise ValueError(cot_type)


def build_prompt(query_ex: Example, demos: List[Example]) -> str:
    parts = [f"System: {SYSTEM_PROMPT}"]
    if len(demos) > 0:
        for d in demos:
            parts.append(format_demo(d))
    parts.append(
        "User: "
        + query_ex.question
        + "\n"
        + format_query_instruction(query_ex.cot_type)
        + "\nAssistant:"
    )
    return "\n\n".join(parts)


# =========================
# 模型加载
# =========================
def load_model_and_tokenizer(base_model: str, lora_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return model, tokenizer


# =========================
# 生成
# =========================
@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


# =========================
# 解析与判分
# =========================
def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("×", "x")
    s = re.sub(r"\s+", " ", s)
    return s


def extract_step_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    step_lines = []
    for ln in lines:
        if re.match(r"^Step\s*\d+\s*:", ln, flags=re.IGNORECASE):
            step_lines.append(ln)
    return step_lines


def extract_final_answer(text: str) -> int | None:
    m = re.search(r"Final\s*Answer\s*:\s*(-?\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    nums = re.findall(r"-?\d+", text)
    if len(nums) == 0:
        return None
    return int(nums[-1])


def score_id_steps(pred_steps: List[str], a: int, b: int) -> List[int]:
    target_steps, total = build_id_steps(a, b)
    target_norm = [normalize_text(x) for x in target_steps]
    pred_norm = [normalize_text(x) for x in pred_steps]

    res = [0] * 5
    for i in range(min(5, len(pred_norm))):
        if pred_norm[i] == target_norm[i]:
            res[i] = 1
    return res


def score_ood_steps(pred_steps: List[str], a: int, b: int) -> List[int]:
    target_steps, total = build_ood_steps(a, b)
    target_norm = [normalize_text(x) for x in target_steps]
    pred_norm = [normalize_text(x) for x in pred_steps]

    res = [0] * 4
    for i in range(min(4, len(pred_norm))):
        if pred_norm[i] == target_norm[i]:
            res[i] = 1
    return res


def evaluate_one_output(ex: Example, prompt: str, raw_output: str, n_shot: int) -> EvalResult:
    pred_steps = extract_step_lines(raw_output)
    pred_final = extract_final_answer(raw_output)

    if ex.cot_type == "id":
        step_correct = score_id_steps(pred_steps, ex.a, ex.b)
    else:
        step_correct = score_ood_steps(pred_steps, ex.a, ex.b)

    final_correct = int(pred_final == ex.target_final)

    return EvalResult(
        cot_type=ex.cot_type,
        n_shot=n_shot,
        a=ex.a,
        b=ex.b,
        prompt=prompt,
        raw_output=raw_output,
        parsed_steps=pred_steps,
        parsed_final=pred_final,
        target_steps=ex.target_steps,
        target_final=ex.target_final,
        step_correct=step_correct,
        final_correct=final_correct,
    )


# =========================
# 汇总统计
# =========================
def summarize_results(results: List[EvalResult], cot_type: str):
    subset = [r for r in results if r.cot_type == cot_type]
    if len(subset) == 0:
        return {}

    n_shot_set = sorted(set(r.n_shot for r in subset))
    summary = {}

    for k in n_shot_set:
        rs = [r for r in subset if r.n_shot == k]
        final_acc = sum(r.final_correct for r in rs) / len(rs)

        num_steps = len(rs[0].step_correct)
        step_acc = []
        for i in range(num_steps):
            step_acc.append(sum(r.step_correct[i] for r in rs) / len(rs))

        all_steps_correct_acc = sum(int(all(x == 1 for x in r.step_correct)) for r in rs) / len(rs)

        summary[k] = {
            "num_examples": len(rs),
            "final_acc": final_acc,
            "step_acc": step_acc,
            "all_steps_correct_acc": all_steps_correct_acc,
        }

    return summary


# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="llama3.2-3b")
    parser.add_argument("--lora_path", type=str, default="llama3.2_lora_claude_final_final_0213")
    parser.add_argument("--num_test", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="cot_eval_results_llama3.2_lora_claude_final_final_0213")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("Loading model...")
    print("=" * 80)
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.lora_path)

    all_results: List[EvalResult] = []

    cot_type = "id"
    print("=" * 80)
    print(f"Evaluating cot_type = {cot_type}")
    print("=" * 80)
    
    demo_pool = make_demo_pool(cot_type)
    test_examples = generate_test_examples(args.num_test, cot_type, seed=args.seed)
    
    for n_shot in range(1, 6):
        print(f"\n---- {cot_type}, {n_shot}-shot ----")
        demos = demo_pool[:n_shot]
    
        for idx, ex in enumerate(test_examples):
            prompt = build_prompt(ex, demos)
            raw_output = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )
            result = evaluate_one_output(ex, prompt, raw_output, n_shot)
            all_results.append(result)
    
            if (idx + 1) % 10 == 0:
                print(f"[{cot_type}][{n_shot}-shot] done {idx + 1}/{len(test_examples)}")

    # 保存明细
    detail_path = os.path.join(args.save_dir, "detail_results.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # 保存汇总
    summary = {
        "id": summarize_results(all_results, "id"),
        "ood": summarize_results(all_results, "ood"),
    }
    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved detail results to: {detail_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()