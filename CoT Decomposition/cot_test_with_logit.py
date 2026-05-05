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

    # step-wise confidence
    step_logprobs: List[float]
    step_probs: List[float]
    step_raw_logits: List[float]
    step_num_tokens: List[int]


# =========================
# two CoT decompositions
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
    a,b within [1000000, 1999999]
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
# few-shot demo pool
# =========================
def make_demo_pool(cot_type: str) -> List[Example]:

    demo_pairs = [
        (1234561, 1323212),
        (1456783, 1543206),
        (1678904, 1112345),
        (1098765, 1876542),
        (1765438, 1234507),
    ]
    return [build_example(a, b, cot_type) for a, b in demo_pairs]


# =========================
# test examples
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
# prompt
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
# Load model
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
# generate
# =========================
@torch.no_grad()
def generate_one(model, tokenizer, prompt, max_new_tokens=256):
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

    new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return text, new_token_ids



# =========================
# evaluation
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

# =======================
# step-wise helper
# =======================
def extract_step_answer_spans(text: str) -> List[Tuple[str, int, int]]:

    spans = []
    step_pattern = re.compile(r"^Step\s*\d+\s*:.*$", flags=re.IGNORECASE | re.MULTILINE)

    for m in step_pattern.finditer(text):
        line = m.group(0)

        nums = list(re.finditer(r"-?\d+", line))
        if len(nums) == 0:
            continue

        last_num = nums[-1]
        ans_text = last_num.group(0)

        start = m.start() + last_num.start()
        end = m.start() + last_num.end()

        spans.append((ans_text, start, end))

    return spans


@torch.no_grad()
def compute_stepwise_logits(
    model,
    tokenizer,
    prompt: str,
    output_text: str,
    output_token_ids,          
):
    step_spans = extract_step_answer_spans(output_text)
    if not step_spans:
        return [], [], [], []


    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True
    )["input_ids"]
    prompt_token_len = prompt_ids.shape[1]

    if torch.cuda.is_available():
        prompt_ids        = prompt_ids.cuda()
        output_token_ids  = output_token_ids.cuda()

    full_ids = torch.cat([prompt_ids, output_token_ids.unsqueeze(0)], dim=1)

    outputs       = model(input_ids=full_ids)
    logits        = outputs.logits[0]          # [seq_len, vocab]
    shifted_logits = logits[:-1]               # predicts input_ids[1:]
    target_ids    = full_ids[0][1:]

    log_probs        = torch.log_softmax(shifted_logits, dim=-1)
    token_logprobs   = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    token_raw_logits = shifted_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)


    output_ids_list = output_token_ids.cpu().tolist()
    n_output = len(output_ids_list)

    token_char_spans = []
    prev_decoded = ""
    for i in range(n_output):
        cur_decoded = tokenizer.decode(
            output_ids_list[: i + 1], skip_special_tokens=True
        )
        token_char_spans.append((len(prev_decoded), len(cur_decoded)))
        prev_decoded = cur_decoded


    full_decoded   = prev_decoded          # decode output token
    leading_offset = len(output_text) - len(output_text.lstrip(" "))

    step_logprobs_out   = []
    step_probs_out      = []
    step_raw_logits_out = []
    step_num_tokens_out = []

    for ans_text, char_start, char_end in step_spans:

        adj_start = char_start - leading_offset
        adj_end   = char_end   - leading_offset

        selected_lp  = []
        selected_rlg = []

        for out_tok_idx, (tok_start, tok_end) in enumerate(token_char_spans):
            if tok_start == tok_end:
                continue
            if tok_end <= adj_start or tok_start >= adj_end:
                continue

            full_tok_idx = prompt_token_len + out_tok_idx
            score_idx    = full_tok_idx - 1
            if score_idx < 0 or score_idx >= token_logprobs.shape[0]:
                continue

            selected_lp.append(token_logprobs[score_idx])
            selected_rlg.append(token_raw_logits[score_idx])

        if not selected_lp:
            step_logprobs_out.append(float("nan"))
            step_probs_out.append(float("nan"))
            step_raw_logits_out.append(float("nan"))
            step_num_tokens_out.append(0)
        else:
            lp  = torch.stack(selected_lp).mean().item()
            rlg = torch.stack(selected_rlg).mean().item()
            step_logprobs_out.append(lp)
            step_probs_out.append(math.exp(lp))
            step_raw_logits_out.append(rlg)
            step_num_tokens_out.append(len(selected_lp))

    return step_logprobs_out, step_probs_out, step_raw_logits_out, step_num_tokens_out
# =======================
def evaluate_one_output(
    ex: Example,
    prompt: str,
    raw_output: str,
    n_shot: int,
    output_token_ids=None,      
    model=None,
    tokenizer=None,
) -> EvalResult:
    pred_steps = extract_step_lines(raw_output)
    pred_final = extract_final_answer(raw_output)

    if ex.cot_type == "id":
        step_correct = score_id_steps(pred_steps, ex.a, ex.b)
    else:
        step_correct = score_ood_steps(pred_steps, ex.a, ex.b)

    final_correct = int(pred_final == ex.target_final)

    if model is not None and tokenizer is not None and output_token_ids is not None:
        step_logprobs, step_probs, step_raw_logits, step_num_tokens = compute_stepwise_logits(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            output_text=raw_output,
            output_token_ids=output_token_ids,   
        )
    else:
        step_logprobs, step_probs, step_raw_logits, step_num_tokens = [], [], [], []

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
        step_logprobs=step_logprobs,
        step_probs=step_probs,
        step_raw_logits=step_raw_logits,
        step_num_tokens=step_num_tokens,
    )


# =========================
# summary
# =========================
def safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) == 0:
        return float("nan")
    return sum(xs) / len(xs)


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
        avg_step_logprobs = []
        avg_step_probs = []
        avg_step_raw_logits = []

        for i in range(num_steps):
            step_acc.append(
                sum(r.step_correct[i] for r in rs) / len(rs)
            )

            avg_step_logprobs.append(
                safe_mean([
                    r.step_logprobs[i]
                    for r in rs
                    if i < len(r.step_logprobs)
                ])
            )

            avg_step_probs.append(
                safe_mean([
                    r.step_probs[i]
                    for r in rs
                    if i < len(r.step_probs)
                ])
            )

            avg_step_raw_logits.append(
                safe_mean([
                    r.step_raw_logits[i]
                    for r in rs
                    if i < len(r.step_raw_logits)
                ])
            )

        all_steps_correct_acc = sum(
            int(all(x == 1 for x in r.step_correct))
            for r in rs
        ) / len(rs)

        summary[k] = {
            "num_examples": len(rs),
            "final_acc": final_acc,
            "step_acc": step_acc,
            "all_steps_correct_acc": all_steps_correct_acc,
            "avg_step_logprobs": avg_step_logprobs,
            "avg_step_probs": avg_step_probs,
            "avg_step_raw_logits": avg_step_raw_logits,
        }

    return summary


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="llama3.2-3b")
    parser.add_argument("--lora_path", type=str, default="your path")
    parser.add_argument("--num_test", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="your path")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("Loading model...")
    print("=" * 80)
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.lora_path)

    all_results: List[EvalResult] = []

    cot_type = "ood"
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
            raw_output, output_token_ids = generate_one(
    model=model, tokenizer=tokenizer,
    prompt=prompt, max_new_tokens=args.max_new_tokens,
)
            result = evaluate_one_output(
    ex=ex, prompt=prompt, raw_output=raw_output,
    n_shot=n_shot, model=model, tokenizer=tokenizer,
    output_token_ids=output_token_ids,   
)
            all_results.append(result)
    
            if (idx + 1) % 10 == 0:
                print(f"[{cot_type}][{n_shot}-shot] done {idx + 1}/{len(test_examples)}")


    detail_path = os.path.join(args.save_dir, "detail_results.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


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
