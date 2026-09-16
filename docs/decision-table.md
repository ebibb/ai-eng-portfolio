# Decision Table: Which Optimizer When?

| Optimizer | Core idea | When to use | Cost (LLM calls) | Built in | Notes |
|-----------|-----------|-------------|------------------|----------|-------|
| Zero-shot baseline | No optimization — just a fixed instruction | Sanity check / lower bound | None | `02-prompt-engineering/` | Always establish this first |
| Random demo search | Bootstrap demos from training traces; random-sample subsets | Small task, fast feedback loop wanted | Low (n_candidates × dataset size) | `04-llm-as-judge/` | Good starting point before instruction tuning |
| Hill-climb instructions (COPRO-style) | LLM proposes instruction rewrites; keep whichever scores higher on val | Zero-shot baseline is weak; demos alone insufficient | Medium (+ instruction rewrite calls) | `05-optimizers/optimizer_v2.py` | Greedy — can get stuck in local optima |
| Instruction + demo search (MIPROv2-style, Bayesian) | Bootstrap demos, generate data-grounded instructions, Gaussian Process + Expected Improvement search over their combinations | Want best possible quality; val evaluations are expensive | High for a full exhaustive search; lower in practice since the GP picks which candidates to evaluate | `05-optimizers/optimizer_v3.py` | Matched the pool's best score while evaluating 14 of 20 candidates in a sample run |
| GEPA | LLM critique → targeted prompt edit → re-evaluate (gradient-like in language) | Scalar score is insufficient; need to understand *why* failures happen | Medium–High (judge call + edit call per iteration) | Not implemented here | Discussed in `04-llm-as-judge/`; requires the LLM-as-judge critique as its feedback signal |

---

*"Cost" is a rough order of magnitude, not exact.*
