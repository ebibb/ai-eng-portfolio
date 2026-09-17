"""
Instruction + demo optimizer.

Extends optimizer_v1.py so PROPOSE generates both instruction candidates (via
LLM rewriting) and demo subsets, then searches their combinations.
A simple hill-climb over instructions is the minimum; Bayesian search (see
optimizer_v3.py) builds on this.

Run:
    python 05-optimizers/optimizer_v2.py

Requires:
    LLM_PROVIDER and LLM_MODEL set in your environment (see ../.env.example).
    03-eval-harness/data/dataset.jsonl populated with real examples.
"""

import datetime
import random
import sys
import os
from typing import Callable
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-eval-harness"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "04-llm-as-judge"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ensure this dir's optimizer_v1.py is used
from llm_provider import get_llm  # noqa: E402
from cost_estimator import DRY_RUN, print_dry_run_summary  # noqa: E402
from eval import evaluate, metric, load_dataset, split  # noqa: E402
from optimizer_v1 import zero_shot_program, bootstrap, build_program, select_best  # noqa: E402
from optimizer_v1 import optimize as optimize_v1  # noqa: E402
import optimizer_v1 as _opt_v1_module  # noqa: E402

load_dotenv()
llm = get_llm()

_run_log: dict = {}

SEED_INSTRUCTION = (
    "Classify the following customer support ticket into exactly one of three categories: "
    "billing, technical, or general. Respond with only the category label."
)

INSTRUCTION_REWRITE_PROMPT = """\
You are an expert prompt engineer. Your job is to improve a task instruction.

Current instruction:
{instruction}

Here are some correct examples from the training set:
{examples}

Write {n} improved versions of the instruction. Each version should be:
- More specific about what distinguishes the categories
- Clearer about edge cases
- No longer than 3 sentences

Output a numbered list (1. ... 2. ... 3. ...) with one instruction per line.
Do not include any other text.
"""


# ── Propose: instruction candidates ───────────────────────────────────────────

def propose_instruction_candidates(
    seed_instruction: str,
    demo_pool: list[dict],
    n_instructions: int = 3,
) -> list[str]:
    """Use the LLM to propose n_instructions rewrites of seed_instruction.

    TODO:
    - Sample up to 5 examples from demo_pool to include as context.
    - Format INSTRUCTION_REWRITE_PROMPT with seed_instruction, examples, n_instructions.
    - Call llm.generate() with temperature=0.7 (some diversity in proposals is good).
    - Parse the numbered list from the response.
    - Return a list of instruction strings (include seed_instruction as the first element).
    """

    instructions = [seed_instruction]
    demos = random.sample(demo_pool, min(5, len(demo_pool)))
    prompt = INSTRUCTION_REWRITE_PROMPT.format(
        instruction=seed_instruction,
        examples="\n".join([f"- {d['input']} -> {d['gold']}" for d in demos]),
        n=n_instructions,
    )
    response = llm.generate(user_prompt=prompt, temp=0.7)
    _run_log["instruction_proposal_prompt"] = prompt
    _run_log["instruction_proposal_response"] = response
    for line in response.splitlines():
        if line.strip() and line[0].isdigit() and "." in line:
            instruction = line.split(".", 1)[1].strip()
            instructions.append(instruction)
    # NOTE: RETURNS n_instructions + 1 BECAUSE WE INCLUDE THE SEED INSTRUCTION AS THE FIRST ELEMENT
    return instructions

# ── Propose: instruction × demo combinations ──────────────────────────────────

def propose_candidates_v2(
    demo_pool: list[dict],
    instructions: list[str],
    n_demo_subsets: int,
    n_demos: int,
) -> list[dict]:
    """Cross-product: for each instruction, sample n_demo_subsets random demo subsets.

    Each candidate: {"instruction": str, "demos": list[dict]}.

    TODO: for each instruction in instructions, sample n_demo_subsets subsets of
    size n_demos from demo_pool. Collect all candidates into one flat list.
    """
    candidates = []
    for instruction in instructions:
        for _ in range(n_demo_subsets):
            demos = random.sample(demo_pool, min(n_demos, len(demo_pool)))
            candidates.append({"instruction": instruction, "demos": demos})
    return candidates


# ── Main optimizer loop v2 ─────────────────────────────────────────────────────

def optimize_v2(
    train,
    val,
    seed_instruction: str = SEED_INSTRUCTION,
    n_instructions: int = 3,
    n_demo_subsets: int = 5,
    n_demos: int = 3,
) -> tuple[dict, float]:
    """Full v2 loop: bootstrap → propose instructions → propose demo subsets → select best.

    Returns (best_candidate, best_val_score).

    TODO:
    1. Establish zero-shot baseline score on val and print it.
    2. bootstrap() to get the demo pool (same as v1).
    3. propose_instruction_candidates() to get instruction variants.
    4. propose_candidates_v2() to get all instruction × demo candidates.
    5. select_best() on val.
    6. Print: zero-shot score, v2 score, improvement.
    7. Assert v2 score > v1 score (import and run optimize from optimizer_v1 to get v1 score).
    """
    
    _run_log["seed_instruction"] = seed_instruction

    # 1. Establish zero-shot baseline score on val and print it.
    zero_shot_prog = zero_shot_program(seed_instruction)
    zero_shot_score = evaluate(zero_shot_prog, val)

    # 2. bootstrap() to get the demo pool (same as v1).
    demo_pool = bootstrap(zero_shot_prog, train=train)
    if not demo_pool and not DRY_RUN:
        print("Bootstrap found no correct examples — cannot propose candidates.")
        return None, zero_shot_score
    
    # 3. propose_instruction_candidates() to get instruction variants.
    instructions = propose_instruction_candidates(seed_instruction=seed_instruction, demo_pool=demo_pool, n_instructions=n_instructions)
    if not instructions:
        print("No instruction candidates proposed.")
        return None, zero_shot_score
    
    # 4. propose_candidates_v2() to get all instruction × demo candidates.
    candidates = propose_candidates_v2(demo_pool=demo_pool, instructions=instructions, n_demo_subsets=n_demo_subsets, n_demos=n_demos)
    if not candidates:
        print("No candidates proposed.")
        return None, zero_shot_score
    
    # 5. select_best() on val.
    best_candidate, best_score, all_results = select_best(candidates, val)
    v2_score = best_score
    _run_log["v2_best_candidate"] = best_candidate
    _run_log["v2_best_score"] = best_score
    _run_log["v2_candidate_results"] = all_results

    # 6. Print: zero-shot score, v2 score, improvement.
    print(f"Zero-shot val score: {zero_shot_score:.3f}")
    print(f"Optimized v2 val score: {best_score:.3f}")
    print(f"Improvement over zero-shot: {best_score - zero_shot_score:.3f}")

    # 7. Assert v2 score > v1 score (import and run optimize from optimizer_v1 to get v1 score).
    v1_best_candidate, v1_score = optimize_v1(train, val)
    _run_log["v1_best_candidate"] = v1_best_candidate
    _run_log["v1_best_score"] = v1_score
    if not DRY_RUN:
        assert v2_score > v1_score, f"v2 score ({v2_score:.3f}) is not greater than v1 score ({v1_score:.3f})"
        print(f"v2 score ({v2_score:.3f}) is greater than v1 score ({v1_score:.3f})")
    return best_candidate, best_score


def write_run_log(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("=== OPTIMIZER V2 RUN LOG ===\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        f.write("--- SEED INSTRUCTION ---\n")
        f.write(_run_log.get("seed_instruction", "(not recorded)") + "\n")
        f.write("\n")

        f.write("--- LLM INPUT (Instruction Proposal) ---\n")
        f.write(_run_log.get("instruction_proposal_prompt", "(not recorded)") + "\n")
        f.write("\n")

        f.write("--- LLM OUTPUT (Instruction Proposal) ---\n")
        f.write(_run_log.get("instruction_proposal_response", "(not recorded)") + "\n")
        f.write("\n")

        f.write("--- V2 CANDIDATE SCORES ---\n")
        v2_results = _run_log.get("v2_candidate_results", [])
        if v2_results:
            seed = _run_log.get("seed_instruction", "")
            current_instruction = None
            inst_num = 0
            for n, result in enumerate(v2_results, 1):
                instruction = result["candidate"].get("instruction", "")
                if instruction != current_instruction:
                    current_instruction = instruction
                    label = "Seed" if instruction == seed else f"Instruction {inst_num}"
                    inst_num += 1
                    f.write(f"\n[{label}]\n")
                f.write(f"\nCandidate {n} | Val score: {result['score']:.3f}\n")
                f.write(_opt_v1_module.build_prompt_template(result["candidate"]) + "\n")
                for p in result.get("predictions", []):
                    mark = "✓" if p["correct"] else "✗"
                    f.write(f"  [{mark}] predicted: {p['prediction']:<12} | gold: {p['gold']:<12} | \"{p['input'][:60]}\"\n")
        else:
            f.write("(not recorded)\n")
        f.write("\n")

        f.write("--- V2 OPTIMIZED OUTPUT ---\n")
        best = _run_log.get("v2_best_candidate")
        score = _run_log.get("v2_best_score", 0.0)
        if best:
            f.write(f"Val score: {score:.3f}\n")
            f.write(f"Instruction: {best.get('instruction', '')}\n")
            demos = best.get("demos", [])
            f.write(f"Demos ({len(demos)}):\n")
            for i, d in enumerate(demos, 1):
                f.write(f"  [{i}] \"{d['input']}\" -> {d['gold']}\n")
        else:
            f.write("(not recorded)\n")
        f.write("\n")

        f.write("=== OPTIMIZER V1 COMPARISON ===\n")
        f.write("\n")

        v1_log = _opt_v1_module._v1_run_log

        f.write("--- V1 SEED INSTRUCTION ---\n")
        f.write(v1_log.get("seed_instruction", "(not recorded)") + "\n")
        f.write("\n")

        f.write("--- V1 CANDIDATE PROMPTS ---\n")
        v1_results = v1_log.get("all_candidate_results", [])
        if v1_results:
            for n, result in enumerate(v1_results, 1):
                f.write(f"\nCandidate {n} | Val score: {result['score']:.3f}\n")
                f.write(_opt_v1_module.build_prompt_template(result["candidate"]) + "\n")
                for p in result.get("predictions", []):
                    mark = "✓" if p["correct"] else "✗"
                    f.write(f"  [{mark}] predicted: {p['prediction']:<12} | gold: {p['gold']:<12} | \"{p['input'][:60]}\"\n")
        else:
            f.write("(not recorded)\n")
        f.write("\n")

        f.write("--- V1 OPTIMIZED OUTPUT ---\n")
        v1_best = v1_log.get("best_candidate")
        v1_score = v1_log.get("best_score", 0.0)
        if v1_best:
            f.write(f"Val score: {v1_score:.3f}\n")
            f.write(f"Instruction: {v1_best.get('instruction', '')}\n")
            v1_demos = v1_best.get("demos", [])
            f.write(f"Demos ({len(v1_demos)}):\n")
            for i, d in enumerate(v1_demos, 1):
                f.write(f"  [{i}] \"{d['input']}\" -> {d['gold']}\n")
        else:
            f.write("(not recorded)\n")


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "03-eval-harness", "data", "dataset.jsonl")

    dataset = load_dataset(DATA_PATH)
    if not dataset:
        print("dataset.jsonl is empty — populate 03-eval-harness/data/dataset.jsonl first.")
        raise SystemExit(1)

    train_data = split(dataset, "train")
    val_data = split(dataset, "val")

    print(f"Dataset: {len(train_data)} train, {len(val_data)} val examples")
    log_path = os.path.join(
        os.path.dirname(__file__), "artifacts", "logs_v2",
        f"optimizer_v2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    try:
        best_candidate, best_score = optimize_v2(train_data, val_data)
        if DRY_RUN:
            print_dry_run_summary()
        else:
            print(f"\nBest val score (v2): {best_score:.3f}")
            if best_candidate:
                print(f"Best instruction: {best_candidate.get('instruction', '')[:120]}")
    finally:
        if not DRY_RUN:
            write_run_log(log_path)
            print(f"\nRun log written to: {log_path}")
