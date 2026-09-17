# AI Engineering Learnings

This repo documents a progression through core LLM engineering skills: prompting, evaluation, LLM-as-judge, and prompt/instruction optimization. Each folder covers one stage and builds on the ones before it.

## Structure

1. **[`01-llm-fundamentals/`](01-llm-fundamentals/)** — temperature, sampling, and calling an LLM API directly.
2. **[`02-prompt-engineering/`](02-prompt-engineering/)** — prompt anatomy (instruction + demonstrations + format) and manual prompt tuning.
3. **[`03-eval-harness/`](03-eval-harness/)** — a train/val/test split and a reusable `metric()` / `evaluate()` pair.
4. **[`04-llm-as-judge/`](04-llm-as-judge/)** — an LLM-as-judge (score + critique) and a first optimizer that searches demonstration subsets.
5. **[`05-optimizers/`](05-optimizers/)** — the most developed section. Three optimizers that search the instruction and demonstrations together: random search, LLM-driven hill-climbing, and a from-scratch Bayesian optimizer using a Gaussian Process over embedding space.

## Motivation

The friction log in [`02-prompt-engineering/friction_log.md`](02-prompt-engineering/friction_log.md) documents manually comparing three prompt variants with no consistent way to measure which was better. The rest of the repo builds the infrastructure (eval harness, judge, optimizers) to make that comparison measurable and repeatable.

## Reference material

- **[`docs/cheat-sheet.md`](docs/cheat-sheet.md)** — glossary of terms used in this repo, with code pointers.
- **[`docs/decision-table.md`](docs/decision-table.md)** — comparison of the optimizers built here: cost, use case, and tradeoffs.
- **[`docs/diagrams/`](docs/diagrams/)** — diagrams of the PROPOSE → EVALUATE → SELECT loop used by every optimizer.

## Running the code

Install dependencies with `pip install -r requirements.txt`, then copy `.env.example` to `.env`
and fill in one provider's values. LLM calls go through `llm_provider.py`, which reads
`LLM_PROVIDER` (`openai` | `anthropic` | `azure` | `ollama`) and dispatches to that backend —
switching providers is an env var change, not a code change. `ollama` runs fully local against
a model you've pulled (e.g. `gemma4:latest`), no API key or cost required.

Each section's README has the exact command to run its scripts. Each optimizer writes a full run log to its own `artifacts/` folder; one representative log per optimizer is committed here.

## License

MIT — see [`LICENSE`](LICENSE).
