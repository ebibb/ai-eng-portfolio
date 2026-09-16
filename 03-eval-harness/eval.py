"""
Reusable eval harness.

Imported by every section from here onward — treat as production code.
Keep metric() and evaluate() signatures stable; changing them breaks all
downstream optimizers.

Run:
    python 03-eval-harness/eval.py
"""


import json
import os
from typing import Callable

Dataset = list[dict]   # each dict: {"input": str, "gold": str, "split": str}
Program = Callable[[str], str] # means input is a str and output is a str


def metric(prediction: str, gold: str) -> float:
    """Score a single prediction against the gold answer. Returns a float in [0.0, 1.0].

    TODO: Implement exact-match as the baseline.
    - Strip leading/trailing whitespace from both strings.
    - Lowercase both strings.
    - Return 1.0 if they match, else 0.0.
    This gets swapped out in the LLM-as-judge section for judge_metric (a drop-in replacement).
    """
    prediction = prediction.strip().lower()
    gold = gold.strip().lower()
    return 1.0 if prediction == gold else 0.0


def evaluate(program: Program, dataset: Dataset, log_path: str = None, log_failures_only: bool = False) -> float:
    """Run program over every example in dataset; return the mean metric score.

    TODO:
    - For each example in dataset, call program(example["input"]).
    - Score the prediction with metric(prediction, example["gold"]).
    - Return the mean score across all examples.
    - Never call this with the test split during optimization — only train or val.
    """
    scores = []
    log_file = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w")

    for example in dataset:
        prediction = program(example["input"])
        score = metric(prediction, example["gold"])
        scores.append(score)
        if log_file and (not log_failures_only or score == 0.0):
            log_file.write(json.dumps({
                "input": example["input"],
                "prediction": prediction,
                "gold": example["gold"],
                "score": score,
            }) + "\n")

    if log_file:
        log_file.close()

    return sum(scores) / len(scores) if scores else 0.0


def load_dataset(path: str) -> Dataset:
    """Load a JSONL dataset file; return a list of example dicts.

    TODO: open the file, json.loads() each non-empty line, collect into a list.
    """
    file = open(path, "r")
    dataset = []
    example = {}
    for line in file:
        if line.strip():
            example = json.loads(line)
            dataset.append(example)
    file.close()
    return dataset


def split(dataset: Dataset, split_name: str) -> Dataset:
    """Return only the examples where example["split"] == split_name."""
    return [example for example in dataset if example["split"] == split_name]

if __name__ == "__main__":
    # Smoke test: run on 3 hand-checked examples before trusting the harness.
    examples: Dataset = [
        {"input": "I was charged twice this month.", "gold": "billing", "split": "val"},
        {"input": "The app crashes on file upload.", "gold": "technical", "split": "val"},
        {"input": "What are your business hours?", "gold": "general", "split": "val"},
    ]

    val = split(examples, "val")

    def always_general(text: str) -> str:
        # Dumb baseline: always predicts "general". Expected score: 1/3 ≈ 0.333
        return "general"

    score = evaluate(always_general, val)
    print(f"Always-general baseline score: {score:.3f}  (expected ~0.333)")
    assert abs(score - 1 / 3) < 0.01, f"Unexpected score {score} — check metric() and evaluate()"
    print("Smoke test passed.")
