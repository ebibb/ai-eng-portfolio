# Instruction Optimization & Bayesian Search

## Builds on
- **Prompt anatomy:** a candidate = `instruction + demos`. The LLM-as-judge section searched the `demos` slot; this section unlocks the `instruction` slot.
- **The optimizer loop:** the same `PROPOSE → EVALUATE → SELECT` loop from `optimizer_v1.py` — only what PROPOSE produces changes.
- **The train/val/test rule:** still select on **val**, never train; test stays untouched. The `evaluate()` / `metric()` harness is unchanged.

## Goal
In `optimizer_v1.py`, PROPOSE only sampled demo subsets and the instruction was frozen. The limit: if the *wording* of the instruction is what's holding back the score, no demo subset can fix it. This section has PROPOSE also **rewrite the instruction with the LLM**, so the search covers `instruction × demos` instead of demos alone. Three approaches, in increasing sophistication:

1. **`optimizer_v1.py` — random search (baseline).** Bootstrap demos, sample random subsets, keep the best on val. Frozen instruction.
2. **`optimizer_v2.py` — hill-climbing on instructions (COPRO-style).** Propose instruction rewrites, keep whichever scores higher on val.
3. **`optimizer_v3.py` — Bayesian optimization over instructions + demos jointly (MIPROv2-style).** Propose data-grounded instructions and search their combinations more sample-efficiently, using a Gaussian Process surrogate.

MIPROv2 is the strong classical baseline GEPA is measured against.

## Resources
- MIPROv2 paper — "Optimizing Instructions and Demonstrations with Multi-Stage Few-Shot Prompting" — https://arxiv.org/abs/2406.11695 (primary baseline and search design)
- GEPA paper — "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" — https://arxiv.org/abs/2507.19457 (modern reflection-based optimizer to compare against MIPRO)
- DSPy docs (GEPA optimization) — https://dspy.ai/getting-started/gepa-optimization/ (practical API walkthrough)
- DSPy docs (choosing an optimizer) — https://dspy.ai/diving-deeper/choosing-an-optimizer/ (when to use hill-climb vs broader search)
- Hill climbing — Decagon glossary, "What is Hill Climbing?": https://decagon.ai/glossary/what-is-hill-climbing — local search, neighbor proposals, restart strategy, early stopping.
- Bayesian optimization — "Exploring Bayesian Optimization" (Distill): https://distill.pub/2020/bayesian-optimization/ — surrogate model, acquisition function, exploration vs exploitation.
- Extra: https://sassafras13.github.io/BayesianOptimization/
- Extra: https://arxiv.org/abs/2510.04384

## How `optimizer_v2.py` works
Extends the earlier `PROPOSE → EVALUATE → SELECT` loop. In v1 a candidate was `instruction (fixed) + demos`; in v2 the instruction is also searched:

1. **Starts from v1.** `optimizer_v1.py`'s score on val is the reference to beat.
2. **PROPOSE also generates instructions.** `propose_instruction_candidates()` asks the LLM to rewrite the seed instruction `n` different ways (clearer wording, edge-case rules, tighter output format). The seed itself stays in as candidate #0.
3. **Searches instruction × demo combinations.** A candidate is now `{instruction, demos}`. Each proposed instruction is paired with demo subsets from the existing bootstrap pool, built into a runnable program with `build_program()`.
4. **EVALUATE on val only.** Reuses `evaluate()` from `../03-eval-harness/eval.py`. Never selects on train.
5. **SELECT with a hill-climb rule.** Keeps the best-scoring `{instruction, demos}` pair seen so far — a greedy accept-if-better search.
6. **Logs every iteration.** Records `instruction_id`, `demo_set_id`, `val_score`, and `llm_calls` — see `artifacts/logs_v2/sample_run.txt` for a full run, and `docs/decision-table.md` for the resulting cost-vs-benefit numbers.

`v2` beats `v1` on val, confirmed by re-running `v1` inside `v2` for direct comparison (see `optimize_v2()`).

## How `optimizer_v3.py` works
`v2`'s hill-climb still evaluates every candidate it proposes. `v3` replaces that exhaustive evaluation with real Bayesian optimization: a Gaussian Process (GP) surrogate model decides *which* candidate is most worth evaluating next, so the search finds a strong result while testing only a fraction of the candidate pool.

Built from scratch with just `numpy`/`math` — no scikit-learn:
- Every candidate (instruction × demo combination) is embedded into a vector via the embeddings API, cached so each unique candidate is only embedded once.
- An RBF kernel over those embeddings feeds a GP that predicts a candidate's expected val score and uncertainty *before* it's actually evaluated.
- An Expected Improvement acquisition function picks the next candidate to test: high predicted score, high uncertainty, or both.
- A handful of candidates are evaluated up front to warm-start the GP, then each further iteration evaluates one more candidate and updates the surrogate.

Every run logs the full math — kernel matrix, Cholesky factor, GP mean/std, and EI/z-scores per iteration — to `artifacts/logs/`. The design rationale (why RBF from scratch, why Cholesky instead of matrix inversion, why the kernel hyperparameters are fixed) is written up in `docs/bayesian-optimizer-spec.md`.

**Result from a sample run** (`artifacts/logs/sample_run.txt`): zero-shot baseline scored 0.933; a warm start of 4 candidates plus 10 BO iterations — 14 evaluations out of a candidate pool of 20 — found a candidate scoring 1.000. The GP reached a candidate matching the best possible score in the pool without evaluating every candidate exhaustively.

## What I built
- **`optimizer_v1.py`, `optimizer_v2.py`, `optimizer_v3.py`** — the three optimizers described above.
- **`test_embed.py`** — live test of the embeddings API: returns vectors of the expected dimension, semantically similar texts land closer together, identical text yields near-identical vectors.
- **`test_optimizer_v3.py`** — test suite split into fast, no-API math tests (kernel, GP fit/predict, EI correctness) and slower tests that make real LLM calls.
- **`optimizer_v3_viz.html`** — a standalone interactive visualization of the PROPOSE/EVALUATE/SELECT phases and the Bayesian optimization iterations, with sliders for the search parameters.
- **`artifacts/generated_instructions_review.md`** — three instructions `optimizer_v2.py` proposed, annotated with why each is better or worse than the seed, and their actual val scores.
- **`docs/optimizer-v1-deep-dive.md`** — function-by-function reference for `optimizer_v1.py`'s bootstrap/propose/select mechanics.
- **`docs/bayesian-optimizer-spec.md`** — full design spec for `optimizer_v3.py`.

## What surprised me
Annotating the LLM-proposed instructions in `artifacts/generated_instructions_review.md` was the most useful exercise here. My intuition about which rewritten instruction would score best was wrong — the seed instruction with a fixed demo set actually outperformed every proposed rewrite on val in that run, and running the seed zero-shot beat the seed-with-demos. That's the concrete argument for why val score has to be the deciding signal, not read-it-and-guess: without it I'd have picked the worst-performing prompt.

## Cost/benefit
Bayesian search is worth it when each val evaluation is expensive (a large val set, an expensive model, or a large candidate pool) — it finds a strong candidate while evaluating a fraction of the pool. A plain hill-climb is good enough when val evaluations are cheap and the candidate pool is small. See `docs/decision-table.md` for the full comparison across every optimizer here.
