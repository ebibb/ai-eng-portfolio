# Cheat-Sheet / Wiki

A running glossary of terms used in this repo: one-line plain-English definition + a code pointer to where it shows up.

| Term | Plain-English definition | Code pointer |
|------|--------------------------|--------------|
| LLM | A neural network trained to predict the next token given a sequence of tokens; "intelligence" emerges from scale and data | `01-llm-fundamentals/llm_practice.py` |
| Temperature | - Temperature is a number used for scaling logits - the last token’s output embedding, enriched with context from all previous tokens and multiplied by learned weights in the final layer. Low temperature makes large logits even larger and small ones even smaller, favoring the highest scoring tokens and leading to more predictable choices. A high temperature flattens the differences, making less likely tokens more competitive and leading to more creative outputs. If temperature = 0, it will always pick the most likely token, neglecting all others. | `01-llm-fundamentals/llm_practice.py` |
| Context window | The maximum number of tokens (input + output) the model can "see" at once; older tokens fall off the left edge, "working memory" | `01-llm-fundamentals/` |
| Word Embedding | A learned numerical vector encoding a word's contextual meaning from training data. | `01-llm-fundamentals/` |
| Positional Encoding | Adds order information so the model knows token positions in a sequence. | `01-llm-fundamentals/` |
| Transformer | Neural architecture using self-attention to process entire sequences in parallel. | `01-llm-fundamentals/` |
| Self-Attention | Mechanism where each token weighs the importance of other tokens to understand context. | `01-llm-fundamentals/` |
| Masked Attention | Prevents tokens from attending to future tokens during generation. | `01-llm-fundamentals/` |
| Top-k Sampling | Limits choices to the k most likely tokens. | `01-llm-fundamentals/` |
| Top-p Sampling | Keeps the smallest set of tokens whose cumulative probability ≥ p. | `01-llm-fundamentals/` |
| Few-shot | Including 1–N example input/output pairs inside the prompt so the model imitates the pattern | `02-prompt-engineering/prompts/few_shot.txt` |
| Chain-of-thought | Asking the model to reason step by step before giving a final answer; often improves accuracy on complex tasks | `02-prompt-engineering/prompts/few_shot_cot.txt` |
| Self-Ask | Breaking problems into sub-questions iteratively. | `02-prompt-engineering/` |
| Tree of Thoughts | Exploring multiple reasoning paths and selecting the best. | `02-prompt-engineering/` |
| Majority Label Bias | Favoring outputs that appear most in examples. | `02-prompt-engineering/challenge/confidence_visualizer_mcq.py` |
| Recency Bias | Overweighting the most recent example in a prompt. | `02-prompt-engineering/challenge/confidence_visualizer_mcq.py` |
| Common Token Bias | Preferring frequently occurring tokens. | `02-prompt-engineering/challenge/confidence_visualizer_mcq.py` |
| Eval harness | The reusable `metric()` + `evaluate(program, dataset) -> score` pair; the objective ground truth for all optimization | `03-eval-harness/eval.py` |
| Train/val/test split | Train = tune on, val = select best model on, test = report final number on (never touch during tuning) | `03-eval-harness/eval.py:split` |
| Scalar score | A single number measuring output quality (e.g. 0.0 or 1.0 for exact-match); tells you *whether* wrong, not *why* | `03-eval-harness/eval.py:metric` |
| Unit Tests (L1) | Fast, scoped assertions that check specific behaviors during development. | `03-eval-harness/` |
| Model & Human Eval (L2) | Combining human judgment and model-based scoring to assess outputs. | `03-eval-harness/` |
| A/B Testing (L3) | Comparing two system versions with real users to measure impact. | `03-eval-harness/` |
| Trace | A full record of interactions (inputs, outputs, steps) for one request. | `03-eval-harness/` |
| Logging | Recording system activity to enable debugging and analysis. | `03-eval-harness/` |
| Proxy Metric | Indirect measure used when the true goal cannot be measured. | `03-eval-harness/` |
| Goodhart’s Law | When a measure becomes a target, it stops being a good measure. | `03-eval-harness/` |
| LLM-as-judge | Using an LLM to score and critique another LLM's output; returns both a scalar and a textual explanation | `04-llm-as-judge/judge.py` |
| Textual feedback | A natural-language critique of an output; tells you *why* it was wrong — the richer signal GEPA exploits | `04-llm-as-judge/judge.py:JudgeResult.critique` |
| Bootstrap | Run a program on training inputs, keep the traces it gets right, reuse those as candidate few-shot demos | `05-optimizers/optimizer_v1.py:bootstrap` |
| PROPOSE → EVALUATE → SELECT | The three-step loop every optimizer runs: generate candidates, score them, keep the best | `05-optimizers/optimizer_v1.py:optimize` |

---
Scalar Score vs. Textual Feedback for Judges
| Dimension | Scalar score | Textual feedback |
|-----------|-------------|------------------|
| **What it tells you** | Whether (and how much) the generated output was right/wrong | *Why* it was wrong — which specific aspect failed |
| **What it hides** | The direction and cause of the failure | May be verbose, inconsistent, or hallucinated across runs |
| **Optimizers that use it** | All classical optimizers (exact-match, F1, BLEU, random search) | Reflective / language-driven optimizers (GEPA, TextGrad) |
| **Example output** | `0.0` | "The generated output was 'billing' but the ticket describes a software crash, which is a technical issue." |
| **Cost per evaluation** | Cheap — string comparison or simple calculation | Expensive — requires an LLM call per example |
| **Signal richness** | Low — only magnitude | High — direction + cause, actionable for prompt editing |

