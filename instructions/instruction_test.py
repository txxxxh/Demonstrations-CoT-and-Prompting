import os
import json
import time
import argparse
from typing import Dict, List, Any, Optional

from tqdm import tqdm
from openai import OpenAI


# =========================================================
# IO
# =========================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: str, item: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


# =========================================================
# Prompt parsing
# =========================================================

def get_prompt(item: Dict[str, Any]) -> str:
    """
    Flexible parser for prompt.jsonl.

    Supported field names:
    - prompt
    - input
    - text
    """
    for key in ["prompt", "input", "text"]:
        if key in item:
            return str(item[key])
    raise KeyError(
        "Each line in prompt.jsonl must contain one of: prompt, input, text."
    )


def get_target_answer(item: Dict[str, Any], default_answer: Optional[str]) -> str:
    """
    Supported field names:
    - answer
    - target
    - target_answer
    - output

    If none exists, use --default_answer.
    """
    for key in ["answer", "target", "target_answer", "output"]:
        if key in item:
            return str(item[key]).strip()

    if default_answer is not None:
        return str(default_answer).strip()

    raise KeyError(
        "Each line must contain answer/target/target_answer/output, "
        "or you should pass --default_answer."
    )


def get_regime(item: Dict[str, Any]) -> str:
    """
    Optional regime label:
    - addition-only
    - multiplication-only
    - mixed-symbol
    """
    for key in ["regime", "mode", "operator_regime"]:
        if key in item:
            return str(item[key])
    return "unknown"


def build_prefixed_prompt(prompt: str, prefix: str) -> str:
    """
    Teacher-forced prefix construction.

    Your prompt.jsonl should preferably end with something like:

        Assistant:
        Answer:

    or

        ### Response:
        Answer:

    Then we append the correct prefix directly after it.

    Example:
        prompt = "... Answer:"
        prefix = "13"
        final prompt = "... Answer:13"

    This asks the model for the next token after the fixed correct prefix.
    """
    return prompt.rstrip() + prefix


# =========================================================
# Logprob extraction
# =========================================================

def token_matches_digit(token: str, digit: str) -> bool:
    """
    Match digit token robustly.

    Usually the next digit token is exactly "1", "2", etc.
    Some APIs may return tokens with spaces or special markers.
    """
    if token == digit:
        return True
    if token.strip() == digit:
        return True
    return False


def extract_digit_logprob_from_choice(choice, digit: str) -> Dict[str, Any]:
    """
    Extract logprob of the correct digit from OpenAI-compatible response.

    Expected response structure:
        choice.logprobs.content[0].top_logprobs

    If the correct digit is not in top_logprobs, return None.
    """
    result = {
        "correct_digit": digit,
        "generated_token": None,
        "generated_token_logprob": None,
        "correct_digit_logprob": None,
        "correct_digit_rank": None,
        "top_logprobs": [],
        "found": False,
    }

    if choice.logprobs is None:
        return result

    content_logprobs = getattr(choice.logprobs, "content", None)
    if not content_logprobs:
        return result

    first_token_info = content_logprobs[0]

    generated_token = getattr(first_token_info, "token", None)
    generated_logprob = getattr(first_token_info, "logprob", None)

    result["generated_token"] = generated_token
    result["generated_token_logprob"] = generated_logprob

    top_logprobs = getattr(first_token_info, "top_logprobs", None) or []

    parsed_top = []
    for rank, tok_info in enumerate(top_logprobs, start=1):
        tok = getattr(tok_info, "token", None)
        lp = getattr(tok_info, "logprob", None)

        parsed_top.append({
            "rank": rank,
            "token": tok,
            "logprob": lp,
        })

        if tok is not None and token_matches_digit(tok, digit):
            result["correct_digit_logprob"] = lp
            result["correct_digit_rank"] = rank
            result["found"] = True

    result["top_logprobs"] = parsed_top

    # Fallback: if generated token itself is the correct digit.
    if not result["found"]:
        if generated_token is not None and token_matches_digit(generated_token, digit):
            result["correct_digit_logprob"] = generated_logprob
            result["correct_digit_rank"] = 1
            result["found"] = True

    return result


def call_one_next_token(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_logprobs: int,
    max_retries: int,
    sleep_time: float,
) -> Any:
    """
    Call Qwen OpenAI-compatible chat completion API.
    """
    last_err = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            return response

        except Exception as e:
            last_err = e
            wait = sleep_time * (2 ** attempt)
            print(f"[Warning] API error: {e}. Retry in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"API failed after {max_retries} retries: {last_err}")


def score_one_example(
    client: OpenAI,
    model: str,
    item: Dict[str, Any],
    default_answer: Optional[str],
    temperature: float,
    top_logprobs: int,
    max_retries: int,
    sleep_time: float,
) -> Dict[str, Any]:
    """
    For target answer a1...aL, extract logprob for each digit position.

    At position ell:
        condition on correct prefix a1...a_{ell-1}
        ask model for one next token
        read logprob of correct digit a_ell
    """
    prompt = get_prompt(item)
    answer = get_target_answer(item, default_answer)
    regime = get_regime(item)

    digit_results = []

    for pos, digit in enumerate(answer, start=1):
        prefix = answer[:pos - 1]
        prefixed_prompt = build_prefixed_prompt(prompt, prefix)

        response = call_one_next_token(
            client=client,
            model=model,
            prompt=prefixed_prompt,
            temperature=temperature,
            top_logprobs=top_logprobs,
            max_retries=max_retries,
            sleep_time=sleep_time,
        )

        choice = response.choices[0]
        digit_info = extract_digit_logprob_from_choice(choice, digit)

        digit_info.update({
            "position": pos,
            "prefix": prefix,
            "prefixed_prompt": prefixed_prompt if False else None,
        })

        digit_results.append(digit_info)

    avg_logprob = None
    found_logprobs = [
        x["correct_digit_logprob"]
        for x in digit_results
        if x["correct_digit_logprob"] is not None
    ]

    if len(found_logprobs) > 0:
        avg_logprob = sum(found_logprobs) / len(found_logprobs)

    return {
        "id": item.get("id", None),
        "regime": regime,
        "model": model,
        "target_answer": answer,
        "num_digits": len(answer),
        "num_found": sum(1 for x in digit_results if x["found"]),
        "avg_correct_digit_logprob": avg_logprob,
        "digit_logprobs": digit_results,
        "metadata": {
            k: v for k, v in item.items()
            if k not in ["prompt", "input", "text"]
        },
    }


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract token-level digit logprobs from Qwen API."
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt.jsonl",
        help="Input prompt jsonl file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/qwen_digit_logprobs.jsonl",
        help="Output jsonl file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-235b-a22b",
        help="Qwen model name.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="DashScope OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="DASHSCOPE_API_KEY",
        help="Environment variable storing API key.",
    )
    parser.add_argument(
        "--default_answer",
        type=str,
        default=None,
        help="Default target answer if prompt.jsonl does not contain answer field.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top_logprobs",
        type=int,
        default=20,
        help="Number of top token logprobs to request.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--sleep_time",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N examples for debugging.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed ids if output file exists.",
    )

    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if api_key is None:
        raise EnvironmentError(
            f"Please set API key environment variable: {args.api_key_env}"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    records = load_jsonl(args.prompt_file)
    if args.limit is not None:
        records = records[:args.limit]

    processed_ids = set()
    if args.resume and os.path.exists(args.output_file):
        old_records = load_jsonl(args.output_file)
        for x in old_records:
            if x.get("id", None) is not None:
                processed_ids.add(x["id"])

    print("=" * 80)
    print("Qwen digit-level logprob extraction")
    print(f"Prompt file:   {args.prompt_file}")
    print(f"Output file:   {args.output_file}")
    print(f"Model:         {args.model}")
    print(f"Base URL:      {args.base_url}")
    print(f"Top logprobs:  {args.top_logprobs}")
    print(f"Temperature:   {args.temperature}")
    print(f"Num examples:  {len(records)}")
    print("=" * 80)

    for item in tqdm(records):
        item_id = item.get("id", None)

        if args.resume and item_id is not None and item_id in processed_ids:
            continue

        result = score_one_example(
            client=client,
            model=args.model,
            item=item,
            default_answer=args.default_answer,
            temperature=args.temperature,
            top_logprobs=args.top_logprobs,
            max_retries=args.max_retries,
            sleep_time=args.sleep_time,
        )

        append_jsonl(args.output_file, result)

    print("=" * 80)
    print("Done.")
    print(f"Saved to: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()