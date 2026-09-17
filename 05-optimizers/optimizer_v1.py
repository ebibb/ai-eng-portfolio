"""
Minimal bootstrap + random-search optimizer.

The optimizer loop is always three moves:
  PROPOSE  — generate candidate programs (each = one instruction + one demo subset)
  EVALUATE — score each candidate on the *val* set using ../03-eval-harness/eval.py
  SELECT   — keep the highest-scoring candidate

Run:
    python 05-optimizers/optimizer_v1.py

Requires:
    LLM_PROVIDER and LLM_MODEL set in your environment (see ../.env.example).
    03-eval-harness/data/dataset.jsonl populated with real examples.
"""
import random
from typing import Callable
import sys
import os
# from dataclasses import dataclass
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm_provider import get_llm  # noqa: E402
from cost_estimator import DRY_RUN, print_dry_run_summary  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "03-eval-harness"))
from eval import evaluate, metric, load_dataset, split  # noqa: E402

load_dotenv()
llm = get_llm()

_v1_run_log: dict = {}

Dataset = list[dict]
Program = Callable[[str], str]

DEFAULT_INSTRUCTION = (
    "Classify the following customer support ticket into exactly one of three categories: "
    "billing, technical, or general. Respond with only the category label."
)


# ── Step 0: Baseline ───────────────────────────────────────────────────────────

def zero_shot_program(instruction: str) -> Program:
    """Return a program that applies the instruction with no examples.

    TODO: import (or copy) call_llm from 01-llm-fundamentals. Format a prompt using instruction
    and the input text, call the LLM at temperature=0, return the stripped response.
    """
    def program(text: str) -> str:
        prompt = f"{instruction}\n\nTicket: \"{text}\"\nCategory:"
        response = llm.generate(user_prompt=prompt, temp=0)
        return response.strip().strip('"')
    return program


# ── Step 1: Bootstrap ──────────────────────────────────────────────────────────

def bootstrap(program: Program, train: Dataset) -> list[dict]:
    """Run program on each training example; return the examples it got right.

    These become the pool of candidate demonstrations — we only keep correct
    traces because we want to show the model good examples, not bad ones.

    TODO: call program(ex["input"]) for each ex in train.
    Keep examples where metric(prediction, ex["gold"]) == 1.0.
    Each kept example should store both "input" and "gold" (and optionally "prediction").
    """
    # empty list[dict] for results
    correct_examples = []
    for ex in train:
        response = program(ex["input"])
        score = metric(response, ex["gold"])
        if score == 1.0:
            correct_examples.append({"input": ex["input"], "gold": ex["gold"]})
    return correct_examples


# ── Step 2: Propose ────────────────────────────────────────────────────────────

def propose_candidates(
    demo_pool: list[dict],
    instruction: str,
    n_candidates: int,
    n_demos: int,
) -> list[dict]:
    """Sample n_candidates random subsets of n_demos from demo_pool.

    Each candidate is a dict: {"instruction": str, "demos": list[dict]}.

    TODO: use random.sample(demo_pool, min(n_demos, len(demo_pool))) repeated
    n_candidates times. If demo_pool has fewer than n_demos examples, use all of them.
    """
    candidates = []
    for _ in range(n_candidates):
        demos = random.sample(demo_pool, min(n_demos, len(demo_pool)))
        candidates.append({"instruction": instruction, "demos": demos})
    return candidates


# ── Step 2b: Build a runnable program from a candidate ────────────────────────

def build_program(candidate: dict) -> Program:
    """Return a callable Program that applies instruction + demos to an input.

    TODO: format a prompt string from candidate["instruction"] and candidate["demos"]
    (use the few_shot.txt format from the prompt engineering section as a template).
    Call the LLM at temperature=0. Return the stripped response text.
    """

    def program(text: str) -> str:
        # text should be the ticket that gets substituted into the prompt as the input
        demos_str = ""
        # loop through candidate["demos"] to create demos string with this format:
        for demo in candidate["demos"]:
            demos_str += f'Ticket: "{demo["input"]}"\n'
            demos_str += f'Category: "{demo["gold"]}"\n'
        # add demos string into prompt
        prompt = f"{candidate['instruction']}\n---\nExamples:\n{demos_str}---\nTicket: \"{text}\"\nCategory:"
        response = llm.generate(user_prompt=prompt, temp=0)
        return response.strip().strip('"')
    return program


def build_prompt_template(candidate: dict) -> str:
    """Return the few-shot prompt for a candidate with '[input]' as the ticket placeholder."""
    demos_str = ""
    for demo in candidate["demos"]:
        demos_str += f'Ticket: "{demo["input"]}"\n'
        demos_str += f'Category: "{demo["gold"]}"\n'
    return f"{candidate['instruction']}\n---\nExamples:\n{demos_str}---\nTicket: \"[input]\"\nCategory:"


# ── Step 3: Select ─────────────────────────────────────────────────────────────

def select_best(candidates: list[dict], val: Dataset) -> tuple[dict, float, list[dict]]:
    """Score each candidate program on val; return (best_candidate, best_score, all_results).

    all_results is a list of {"candidate": dict, "score": float} for every candidate scored.
    TODO: for each candidate, call build_program() then evaluate() on val.
    Track the highest score and the corresponding candidate.
    Always use val here — never train — because we're selecting the best program.
    """
    best_score = 0.0
    best_candidate = None
    all_results = []
    for candidate in candidates:
        program = build_program(candidate)
        predictions = []
        for ex in val:
            pred = program(ex["input"])
            s = metric(pred, ex["gold"])
            predictions.append({"input": ex["input"], "prediction": pred, "gold": ex["gold"], "correct": s == 1.0})
        score = sum(p["correct"] for p in predictions) / len(predictions) if predictions else 0.0
        all_results.append({"candidate": candidate, "score": score, "predictions": predictions})
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate, best_score, all_results


# ── Main loop ──────────────────────────────────────────────────────────────────

def optimize(
    train: Dataset,
    val: Dataset,
    n_candidates: int = 3,
    n_demos: int = 3,
    instruction: str = DEFAULT_INSTRUCTION,
) -> tuple[dict, float]:
    """Run the full PROPOSE → EVALUATE → SELECT loop.

    Returns (best_candidate, best_val_score).

    TODO:
    1. Score the zero-shot baseline on val and print it.
    2. bootstrap() using the zero-shot baseline program.
    3. propose_candidates() from the demo pool.
    4. select_best() on val.
    5. Print the optimized val score and the improvement.
    """
    _v1_run_log["seed_instruction"] = instruction

    # 1. Score the zero-shot baseline on val and print it.
    zero_shot_prog = zero_shot_program(instruction)
    zero_shot_score = evaluate(zero_shot_prog, val)
    print(f"Zero-shot val score: {zero_shot_score:.3f}")

    # 2. bootstrap() using the zero-shot baseline program.
    demo_pool = bootstrap(zero_shot_prog, train=train)

    # 3. propose_candidates() from the demo pool.
    if not demo_pool and not DRY_RUN:
        print("Bootstrap found no correct examples — cannot propose candidates.")
        return None, zero_shot_score
    candidates = propose_candidates(demo_pool=demo_pool, instruction=instruction, n_candidates=n_candidates, n_demos=n_demos)
    _v1_run_log["candidate_prompts"] = [build_prompt_template(c) for c in candidates]

    # 4. select_best() on val.
    best_candidate, best_score, all_results = select_best(candidates, val)
    _v1_run_log["best_candidate"] = best_candidate
    _v1_run_log["best_score"] = best_score
    _v1_run_log["all_candidate_results"] = all_results

    # 5. Print the optimized val score and the improvement.
    print(f"Optimized val score: {best_score:.3f}")
    print(f"Improvement over zero-shot: {best_score - zero_shot_score:.3f}")

    return best_candidate, best_score


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "03-eval-harness", "data", "dataset.jsonl")

    dataset = load_dataset(DATA_PATH)
    if not dataset:
        print("dataset.jsonl is empty — populate 03-eval-harness/data/dataset.jsonl first.")
        raise SystemExit(1)

    train_data = split(dataset, "train")
    val_data = split(dataset, "val")

    print(f"Dataset: {len(train_data)} train, {len(val_data)} val examples")
    best_candidate, best_score = optimize(train_data, val_data)
    if DRY_RUN:
        print_dry_run_summary()
    else:
        print(f"\nBest val score: {best_score:.3f}")
        print(f"Best instruction: {best_candidate.get('instruction', '')}")
        print(f"Best demos ({len(best_candidate.get('demos', []))}): {best_candidate.get('demos', [])}")
