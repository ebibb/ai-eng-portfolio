"""
Run zero-shot eval + bootstrap exactly once and print hard-codeable values
for ablation.py.

Run:
    python 04-llm-as-judge/bootstrap_once.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "03-eval-harness"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "05-optimizers"))
from cost_estimator import DRY_RUN, print_dry_run_summary  # noqa: E402
from eval import evaluate, load_dataset, split  # noqa: E402
from optimizer_v1 import bootstrap, zero_shot_program, DEFAULT_INSTRUCTION  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "03-eval-harness", "data", "dataset.jsonl")

dataset = load_dataset(DATA_PATH)
train_data = split(dataset, "train")
val_data = split(dataset, "val")
print(f"Dataset: {len(train_data)} train, {len(val_data)} val\n")

print("Evaluating zero-shot on val...")
zero_shot_prog = zero_shot_program(DEFAULT_INSTRUCTION)
zero_shot_score = evaluate(zero_shot_prog, val_data)

print("Running bootstrap on train...")
demo_pool = bootstrap(zero_shot_prog, train_data)

if DRY_RUN:
    print_dry_run_summary()
else:
    print("\n── Paste these into ablation.py ──────────────────────────")
    print(f"ZERO_SHOT_SCORE = {zero_shot_score}")
    print(f"DEMO_POOL = {json.dumps(demo_pool, indent=2)}")
