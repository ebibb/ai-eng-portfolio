# LLM Fundamentals

## Goal
Build a working mental model of next-token prediction, temperature/sampling, and a simple look at transformers.

## Resources
- Andrej Karpathy, *Intro to Large Language Models* (1 hr) — https://www.youtube.com/watch?v=zjkBMFhNj_g
- FT Transformers Explanation: http://ig.ft.com/generative-ai/
- Georgia Tech's Transformers Visualization: https://poloclub.github.io/transformer-explainer/
- OpenAI chat completions API reference — https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create

## What I built
`llm_practice.py` calls the chat completions API and sweeps temperature — `[0, 0.3, 0.7, 1.0]` — on one fixed prompt, printing the output at each setting for direct comparison. It calls through `llm_provider.py`, a provider-agnostic interface with retry/backoff and clean parameter passthrough (temperature, logprobs, etc.) that works against OpenAI, Anthropic, Azure OpenAI, or a local Ollama model — see `../llm_provider.py` for the full implementation.

## What I found
Temperature scales the logits (the raw per-token scores a model produces before they're converted into a probability distribution) before sampling. Lower temperature sharpens the distribution toward the highest-scoring tokens, producing more repeatable output. Higher temperature flattens the distribution, making lower-scoring tokens more likely to be picked. At `temperature=0`, the model deterministically picks the highest-probability token every time. The full write-up on why that matters is in `docs/cheat-sheet.md`.

## Running it
```bash
python 01-llm-fundamentals/llm_practice.py
```
Requires the environment variables listed in the root README.
