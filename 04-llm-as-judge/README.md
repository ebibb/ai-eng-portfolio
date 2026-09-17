# LLM-as-Judge and a First Optimizer

## Builds on
Two things build directly on the earlier sections:
1. **Judge generated outputs.** The eval harness gave an exact-match `metric()`. Here it's replaced with an *LLM-as-judge* that returns both a scalar score **and** a textual critique — the critique is the signal reflective optimizers (GEPA) rely on.
2. **Build the simplest optimizer.** Prompt engineering named the parts of a prompt (`instruction + demos + format`); the eval harness gave `evaluate()`. Now the `demos` slot is searched automatically: bootstrap correct traces, sample candidates, evaluate on val, keep the best. This is the PROPOSE → EVALUATE → SELECT loop every later optimizer reuses.

This is the answer to the friction log from the prompt engineering section — no more tuning prompts by hand.

## Two LLMs: the solver and the judge
Keep these straight — it's the thing people trip on:

- The **solver** reads an input and produces a **generated output** — the answer.
- The **judge** (`judge.py`) does **not** answer the input. It *grades* the generated output the solver already produced, and explains the grade.

```
input ──▶ [ SOLVER ] ──▶ generated output ──┐
                                            ├──▶ [ JUDGE ] ──▶ score + critique
input ──────────────────────────────────────┘
```

The judge needs the **generated output** because you cannot grade an answer you cannot see. If you handed the judge only the input, it would have to solve the task itself — and then it's just a second solver. Throughout this section, **expected output** refers to the known correct answer (the eval harness's `gold`), and **generated output** to what the solver produced.

**Judging:**
- **Hamel Husain — "Creating a LLM-as-a-Judge That Drives Business Results"** — https://hamel.dev/blog/posts/llm-judge/ — how to actually build, align, and trust a judge (same author as the eval harness's evals reading).
- **Eugene Yan — "LLM-Evaluators (aka LLM-as-Judge)"** — https://eugeneyan.com/writing/llm-evaluators/ — survey of judge patterns (direct scoring, pairwise, reference-based vs. reference-free) with the trade-offs laid out.
- **Judge bias & calibration** — **Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"** — https://arxiv.org/abs/2306.05685 — §3.3 ("Limitations") is the source that first measured *position bias*, *verbosity bias*, and *self-enhancement bias*. For a plain-English take, Eugene Yan's "Biases" section (linked above) summarizes the same failure modes with mitigations.
- **GEPA paper, abstract only** — https://arxiv.org/abs/2507.19457 — the claim that language is a "richer learning medium" than scalar rewards.

## Bootstrapping: where the demos come from
In prompt engineering, the few-shot examples were hand-written. The optimizer can't do that — so it **bootstraps** them: run the solver on the training set, keep the examples it already gets right, and reuse those `(input, expected output)` pairs as the **demo pool**. Then it *searches* — sample random subsets of demos, score each candidate on **val** with `evaluate()`, keep the highest. That's the PROPOSE → EVALUATE → SELECT loop.

Resources:
- **DSPy — "Optimizers" / `BootstrapFewShot`** — https://dspy.ai/learn/optimization/optimizers/ — what bootstrapping is and how a real framework samples and selects demos. `../05-optimizers/optimizer_v1.py` mirrors this.
- **Khattab et al., "DSPy" paper** — https://arxiv.org/abs/2310.03714 — the academic source for *why* self-generated demos work (intro + bootstrap section).
- **Optional preview:** MIPRO abstract + Figure 1 — https://arxiv.org/abs/2406.11695 (the optimizers section builds on it).

## What I built

### Part 1 — Judge (extends the eval harness's `metric()`)
`judge.py` is an LLM-as-judge that returns both a score (`float`) and a one-sentence critique (`str`). It supports two modes: *reference-based* (pass the **expected output**, compare to the correct answer) and *reference-free* (`expected_output=None`, grade against a rubric only — the mode GEPA usually relies on). The scalar adapter `metric(generated_output, expected_output) -> float` keeps the eval harness's exact signature, so it drops straight into `evaluate()`. The critique matters more than the score — it's the textual feedback GEPA passes to another LLM to rewrite the prompt.

`artifacts/comparison_table.md` — "scalar score vs. textual feedback" comparison, with results from running the judge in both modes.

**How it plugs into the harness:** `../03-eval-harness/eval.py`'s `evaluate()` is just a loop — for each example it asks the solver for a `generated_output`, then calls `metric(generated_output, expected_output)` to score it, and averages the scores. Because `judge.py`'s `metric()` has the same signature as the eval harness's exact-match `metric()`, one swaps in for the other with no change to the loop. `evaluate()` only uses the number; the critique is for a human reader, and later, GEPA.

### Part 2 — Optimizer (reuses the eval harness's `evaluate()`)
`../05-optimizers/optimizer_v1.py` is a from-scratch bootstrap + random-search optimizer, importing `evaluate`, `metric`, `load_dataset`, `split` from `../03-eval-harness/eval.py`. A candidate is `instruction + demos`; the optimizer searches the `demos` slot. It beats the zero-shot baseline on val. (The code lives in `05-optimizers/` alongside `v2`/`v3` so the whole optimizer progression stays in one place — this section is where the approach was first built.)

`artifacts/ablation_note.md` — val score vs. number of demos, and vs. number of candidates, with a takeaway for each.

`artifacts/loop_diagram.md` — the PROPOSE → EVALUATE → SELECT loop, drawn from memory.

> **Why does the optimizer use exact-match, not the judge?** For this 3-way classification, exact-match is enough and free — no LLM call per score. The judge exists to demonstrate the *scalar vs. feedback* distinction and to produce the critique GEPA consumes later; swapping `judge.py`'s `metric` into `evaluate()` is a one-line change, but wasn't necessary here.

## What this confirmed
- Running `judge.py` on a clearly wrong generated output yields a critique that names the **specific** failure — not a generic "the answer is incorrect."
- `../05-optimizers/optimizer_v1.py` beats the zero-shot baseline on val.
- SELECT uses the val split, not train — selecting on train would be overfitting, the same train/val/test rule from the eval harness applied here.

## How this connects
- **Back to prompt engineering:** a "candidate" is just `instruction + demos` — the anatomy already named there.
- **Back to the eval harness:** every candidate is scored with `evaluate()` on val; test stays untouched.
- **Forward to the optimizers section:** `optimizer_v2.py` extends this to also search the `instruction` slot.
