# Datasets, Metrics, and a Reusable Eval Harness

## Goal
Build train/val/test splits and a metric function. Understand why you optimize on train/val and only *report* on held-out test — and why a noisy or gameable metric ruins everything downstream.

## Resources
- Start here: https://hamel.dev/blog/posts/evals/
- Metrics reading:
    - https://www.fast.ai/posts/2019-09-24-metrics.html
- Dataset reading:
    - https://www.fast.ai/posts/2017-11-13-validation-sets.html
    - https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets
- Evals bible (literally): https://hamel.dev/blog/posts/evals-faq/
    - Getting Started & Fundamentals are the core sections; worth keeping handy as reference.
- Bonus reading (advanced): https://eugeneyan.com/writing/evals/
- Extra:
    - "Don't stray from binary pass/fail judgements when starting out" https://hamel.dev/blog/posts/llm-judge/#dont-stray-from-binary-passfail-judgments-when-starting-out
    - https://glowing-adventure-r33jeg1.pages.github.io/evaluations/llm-as-judge/#binary-passfail

## What I built
- **`data/dataset.jsonl`** — ~50–100 synthetically created labeled examples. Each line: `{"input": "...", "gold": "billing|technical|general", "split": "train|val|test"}`. Split logic lives in code (70% train / 15% val / 15% test), not applied by hand.
- **`eval.py`** — the reusable harness: `metric(prediction, gold) -> float` and `evaluate(program, dataset) -> score`.
- **`data/data_card.md`** — dataset description, label definitions, and known ambiguous cases.

## Why the test set stays untouched
`evaluate()` runs on all three splits without error. A model is never selected based on the test set — selecting on the same data you report the final number on would let you overfit the *reporting*, not just the model. Test only gets used once, at the very end, to report a number nothing was tuned against.

## Importing eval.py from other sections
Other sections import this module with:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '03-eval-harness'))
from eval import evaluate, metric
```
The `metric()` and `evaluate()` signatures stay stable — changing them would break every downstream section.

## Running it
```bash
python 03-eval-harness/eval.py
```
Runs the smoke test in `__main__` and prints the val score.
