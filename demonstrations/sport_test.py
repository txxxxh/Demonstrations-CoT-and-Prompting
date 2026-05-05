#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import random
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from openai import OpenAI

# =============================
# Config
# =============================
API_BASE_URL = "your url"
MODEL_NAME = "your model name"
# Strongly recommended: set env var DASHSCOPE_API_KEY instead of hard-coding.
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# Fixed seeds 0-7 as requested
DEFAULT_SEEDS = list(range(0, 8))

# =============================
# Plot style (Times New Roman, big fonts)
# =============================
FONT_FAMILY = "Times New Roman"
LABEL_FONTSIZE = 26
TICK_FONTSIZE = 22
LEGEND_FONTSIZE = 22
TITLE_FONTSIZE = 22
LINEWIDTH = 3
MARKERSIZE = 8

legend_font = FontProperties(family=FONT_FAMILY, size=LEGEND_FONTSIZE)

# =============================
# 1) Client + Chat
# =============================
def get_client() -> OpenAI:
    key = API_KEY
    if not key:
        raise RuntimeError(
            "Missing DASHSCOPE_API_KEY.\n"
            "Set it via:\n"
            "  export DASHSCOPE_API_KEY='YOUR_KEY'\n"
        )
    return OpenAI(api_key=key, base_url=API_BASE_URL)


def chat_once(
    client: OpenAI,
    prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 96,
) -> str:
    # Strict JSON for robust parsing
    sys = (
        "Return a single-line JSON only, no extra text.\n"
        "Schema: {\"answer\": string, "
        "\"relation\": one of [\"national_sport\", \"popular_sport\", \"other\"], "
        "\"confidence\": number in [0,1]}.\n"
        "The answer must be a sport name in English (1-3 words). "
        "If unsure, still pick your best guess and lower confidence."
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def safe_parse_json(s: str) -> Dict[str, Any]:
    s2 = s.strip()
    s2 = re.sub(r"^```(json)?\s*", "", s2)
    s2 = re.sub(r"\s*```$", "", s2)
    try:
        obj = json.loads(s2)
        if not isinstance(obj, dict):
            raise ValueError("not dict")
        return obj
    except Exception:
        return {"answer": s2[:80], "relation": "other", "confidence": 0.0}


# =============================
# 2) Task: National sport vs Popular sport
# =============================
def prompt_from_demos(
    demos: List[Tuple[str, str, str]],
    query_country: str,
) -> str:
    demo_str = ", ".join([f"({c}, {rel}, {ans})" for (c, rel, ans) in demos])
    # Keep the same "Question: ... . X, ?" format, but hint "sport name"
    return f"Question: {demo_str}. {query_country}, ? (Answer with a sport name in English.)"


def build_demo_pools() -> Dict[str, List[List[Tuple[str, str, str]]]]:
    """
    ambiguous: demos that can be explained by BOTH:
        A = national_sport, B = popular_sport
    identifying: demos that are (as much as possible) ONLY explained by A = national_sport

    NOTE: You can expand these lists; this is a minimal clean starting set.
    """
    # Ambiguous (A ∩ B): pick cases where the showcased sport is plausibly both national and popular/representative.
    # Keep small to avoid hidden biases; you can add more later after verifying.
    ambiguous = [
        [("Afghanistan", "?", "Buzkashi")],
    ]

    # Identifying (A \ B): "official national sport" examples that are *not* the most popular sport.
    identifying = [
        [("Bangladesh", "?", "Kabaddi")],
        [("Pakistan", "?", "Field hockey")],
        [("Canada", "?", "Lacrosse")],
    ]

    return {"ambiguous": ambiguous, "identifying": identifying}


# =============================
# 3) Stats helpers
# =============================
def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\.\,\;\:\(\)\[\]\{\}\"\']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in counts.values():
        p = v / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return ent


def top1_rate(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return max(counts.values()) / total


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return 1.96 * std(xs) / math.sqrt(len(xs))


# =============================
# 4) Main experiment (single seed)
# =============================
def run_experiment(
    n_trials: int,
    temperature: float,
    sleep_s: float,
    seed: int,
    out_jsonl: str,
    query_countries: List[str],
):
    random.seed(seed)
    client = get_client()
    pools = build_demo_pools()

    results = []
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for t in range(n_trials):
            for typ in ["ambiguous", "identifying"]:
                demos = random.choice(pools[typ])
                query_country = random.choice(query_countries)
                prompt = prompt_from_demos(demos, query_country=query_country)

                raw = chat_once(client, prompt, temperature=temperature)
                obj = safe_parse_json(raw)

                ans = str(obj.get("answer", "")).strip()
                rel = str(obj.get("relation", "other")).strip().lower()
                conf = obj.get("confidence", 0.0)

                rec = {
                    "trial": t,
                    "type": typ,
                    "seed": seed,
                    "query_country": query_country,
                    "prompt": prompt,
                    "demos": demos,
                    "raw": raw,
                    "answer": ans,
                    "answer_norm": normalize_text(ans),
                    "relation": rel,
                    "confidence": conf,
                    "temperature": temperature,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                results.append(rec)

                if sleep_s > 0:
                    time.sleep(sleep_s)

    return results


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    summary: Dict[str, Dict[str, Any]] = {}
    for typ, rows in by_type.items():
        ans_counts = Counter([r["answer_norm"] for r in rows if r["answer_norm"]])
        rel_counts = Counter([r["relation"] for r in rows if r["relation"]])

        summary[typ] = {
            "n": len(rows),
            "top1_rate": top1_rate(ans_counts),
            "entropy_nats": entropy_from_counts(ans_counts),
            "unique_answers": len(ans_counts),
            "answer_counts": ans_counts,
            "rel_counts": rel_counts,
        }
    return summary


# =============================
# 5) Publication-style plots
# =============================
def plot_two_types_mean_ci_with_scatter(
    values_by_type: Dict[str, List[float]],
    ylabel: str,
    title: str,
    out_path: str,
    ylim=None,
):
    labels = ["ambiguous", "identifying"]
    xs = [0, 1]

    means = [mean(values_by_type.get(k, [])) for k in labels]
    cis = [ci95(values_by_type.get(k, [])) for k in labels]

    plt.figure(figsize=(7.2, 5.2))

    # Mean ± CI
    plt.errorbar(
        xs, means, yerr=cis,
        linestyle="none",
        marker="o",
        markersize=MARKERSIZE + 1,
        capsize=7,
        linewidth=LINEWIDTH,
        label="Mean ± 95% CI",
    )

    # Seed scatter + mean bar
    for i, typ in enumerate(labels):
        ys = values_by_type.get(typ, [])
        jittered_x = [i + (random.random() - 0.5) * 0.10 for _ in ys]
        plt.scatter(jittered_x, ys, s=60, marker=("o" if i == 0 else "s"), linewidths=1.2)
        plt.hlines(mean(ys), i - 0.18, i + 0.18, linewidth=LINEWIDTH)

    plt.xticks(xs, ["Ambiguous", "Identifying"], fontsize=TICK_FONTSIZE, fontfamily=FONT_FAMILY)
    plt.yticks(fontsize=TICK_FONTSIZE, fontfamily=FONT_FAMILY)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontfamily=FONT_FAMILY)
    if title:
        plt.title(title, fontsize=TITLE_FONTSIZE, fontfamily=FONT_FAMILY)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=1,
        frameon=False,
        prop=legend_font,
        handlelength=2.5,
        columnspacing=0.7,
    )

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================
# 6) Multi-seed runner (0-7)
# =============================
def run_multi_seed(
    seeds: List[int],
    n_trials: int,
    temperature: float,
    sleep_s: float,
    figdir: str,
    query_countries: List[str],
):
    os.makedirs(figdir, exist_ok=True)

    seed_summaries: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for sd in seeds:
        out_jsonl = os.path.join(figdir, f"task_sports_seed{sd}.jsonl")
        results = run_experiment(
            n_trials=n_trials,
            temperature=temperature,
            sleep_s=sleep_s,
            seed=sd,
            out_jsonl=out_jsonl,
            query_countries=query_countries,
        )
        summ = aggregate(results)

        seed_summaries[sd] = {
            "ambiguous": {
                "top1_rate": summ["ambiguous"]["top1_rate"],
                "entropy_nats": summ["ambiguous"]["entropy_nats"],
                "unique_answers": summ["ambiguous"]["unique_answers"],
                "rel_counts": dict(summ["ambiguous"]["rel_counts"]),
            },
            "identifying": {
                "top1_rate": summ["identifying"]["top1_rate"],
                "entropy_nats": summ["identifying"]["entropy_nats"],
                "unique_answers": summ["identifying"]["unique_answers"],
                "rel_counts": dict(summ["identifying"]["rel_counts"]),
            },
        }

        with open(os.path.join(figdir, f"summary_seed{sd}.json"), "w", encoding="utf-8") as f:
            json.dump(seed_summaries[sd], f, ensure_ascii=False, indent=2)

    # Overall summary across seeds
    overall = {}
    for typ in ["ambiguous", "identifying"]:
        overall[typ] = {}
        for met in ["top1_rate", "entropy_nats", "unique_answers"]:
            xs = [seed_summaries[s][typ][met] for s in seeds]
            overall[typ][met] = {
                "mean": mean(xs),
                "std": std(xs),
                "ci95": ci95(xs),
                "values": xs,
            }

    overall_path = os.path.join(figdir, "summary_over_seeds.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": seeds,
                "trials_per_seed": n_trials,
                "temperature": temperature,
                "query_countries": query_countries,
                "overall": overall,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Publication-style plots
    plot_two_types_mean_ci_with_scatter(
        {
            "ambiguous": overall["ambiguous"]["entropy_nats"]["values"],
            "identifying": overall["identifying"]["entropy_nats"]["values"],
        },
        ylabel="Entropy (nats)",
        title=f"Entropy across seeds (T={temperature})",
        out_path=os.path.join(figdir, "entropy_ci_pub.png"),
        ylim=None,
    )

    plot_two_types_mean_ci_with_scatter(
        {
            "ambiguous": overall["ambiguous"]["top1_rate"]["values"],
            "identifying": overall["identifying"]["top1_rate"]["values"],
        },
        ylabel="Top-1 rate",
        title=f"Top-1 rate across seeds (T={temperature})",
        out_path=os.path.join(figdir, "top1_ci_pub.png"),
        ylim=(0.0, 1.05),
    )

    plot_two_types_mean_ci_with_scatter(
        {
            "ambiguous": overall["ambiguous"]["unique_answers"]["values"],
            "identifying": overall["identifying"]["unique_answers"]["values"],
        },
        ylabel="Unique outputs",
        title=f"Unique outputs across seeds (T={temperature})",
        out_path=os.path.join(figdir, "unique_ci_pub.png"),
        ylim=None,
    )

    # Print concise console summary
    concise = {
        typ: {
            "top1_rate_mean": overall[typ]["top1_rate"]["mean"],
            "entropy_mean": overall[typ]["entropy_nats"]["mean"],
            "unique_mean": overall[typ]["unique_answers"]["mean"],
        }
        for typ in ["ambiguous", "identifying"]
    }
    print("Figures in:", figdir)
    print("Overall summary:", overall_path)
    print(json.dumps(concise, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--figdir", type=str, default="fig_task_sports")

    # Query countries (test countries). You can add more to improve robustness.
    ap.add_argument("--queries", type=str, default="India,Japan,Italy,Brazil,Canada")

    args = ap.parse_args()

    query_countries = [x.strip() for x in args.queries.split(",") if x.strip()]
    run_multi_seed(
        seeds=DEFAULT_SEEDS,
        n_trials=args.trials,
        temperature=args.temp,
        sleep_s=args.sleep,
        figdir=args.figdir,
        query_countries=query_countries,
    )


if __name__ == "__main__":
    main()
