"""
Provider-agnostic cost estimation for --dry-run / DRY_RUN=1.

DRY_RUN is true if either the DRY_RUN env var is set to a truthy value or
"--dry-run" is passed on the command line. llm_provider.get_llm() checks this
flag and returns a DryRunProvider instead of a real backend, so no script needs
to check DRY_RUN itself just to avoid making a real call — it only needs to
skip printing/writing real-looking results afterward (see the `if DRY_RUN:`
branches in each script's __main__).

Token counts use tiktoken for openai/azure (OpenAI's cl100k_base encoding is a
close proxy for Azure OpenAI deployments of the same model family) and a
~4-characters-per-token heuristic for anthropic/ollama, since neither
publishes a local tokenizer package. Output token counts are always a fixed
assumption (ASSUMED_OUTPUT_TOKENS) since the real output length isn't known
until a real call is made — the estimate is therefore an approximation, not a
guarantee, and assumes every planned call in a script's normal control flow
actually happens (adaptive branching in a real run could make fewer calls).

Pricing is a small hardcoded table of well-known models, in USD per 1,000,000
tokens. Local Ollama models are always $0. An unrecognized model falls back to
a conservative generic rate so dry-run still produces a number.
"""

import os
import sys

DRY_RUN = os.getenv("DRY_RUN", "").strip().lower() in ("1", "true", "yes") or "--dry-run" in sys.argv

ASSUMED_OUTPUT_TOKENS = 100

# USD per 1,000,000 tokens
CHAT_PRICING = {
    ("openai", "gpt-4o"): {"input": 2.50, "output": 10.00},
    ("openai", "gpt-4o-mini"): {"input": 0.15, "output": 0.60},
    ("anthropic", "claude-3-5-sonnet-latest"): {"input": 3.00, "output": 15.00},
    ("anthropic", "claude-3-5-haiku-latest"): {"input": 0.80, "output": 4.00},
}
EMBEDDING_PRICING = {
    ("openai", "text-embedding-3-small"): 0.02,
    ("openai", "text-embedding-3-large"): 0.13,
}
FALLBACK_CHAT_PRICING = {"input": 3.00, "output": 15.00}
FALLBACK_EMBEDDING_PRICING = 0.10


def count_tokens(text, provider, model=""):
    if provider in ("openai", "azure"):
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass
    # anthropic, ollama, or tiktoken unavailable: ~4 chars/token heuristic
    return max(1, len(text) // 4)


def chat_cost(provider, model, input_tokens, output_tokens):
    if provider == "ollama":
        return 0.0
    pricing = CHAT_PRICING.get((provider, model), FALLBACK_CHAT_PRICING)
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


def embedding_cost(provider, model, input_tokens):
    if provider == "ollama":
        return 0.0
    rate = EMBEDDING_PRICING.get((provider, model), FALLBACK_EMBEDDING_PRICING)
    return (input_tokens / 1_000_000) * rate


class _Tracker:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0


_tracker = _Tracker()


def record_generate_call(provider, model, user_prompt, system_prompt=""):
    """Record one planned chat call; return (input_tokens, estimated_cost) for this call alone."""
    input_tokens = count_tokens((system_prompt or "") + user_prompt, provider, model)
    cost = chat_cost(provider, model, input_tokens, ASSUMED_OUTPUT_TOKENS)
    _tracker.calls += 1
    _tracker.input_tokens += input_tokens
    _tracker.output_tokens += ASSUMED_OUTPUT_TOKENS
    _tracker.cost += cost
    return input_tokens, cost


def record_embed_call(provider, model, text):
    """Record one planned embedding call; return (input_tokens, estimated_cost) for this call alone."""
    input_tokens = count_tokens(text, provider, model)
    cost = embedding_cost(provider, model, input_tokens)
    _tracker.calls += 1
    _tracker.input_tokens += input_tokens
    _tracker.cost += cost
    return input_tokens, cost


def dry_run_summary_text():
    total_tokens = _tracker.input_tokens + _tracker.output_tokens
    lines = [
        "── DRY RUN — no LLM calls were made ─────────────────────────",
        f"  Estimated calls:  {_tracker.calls}",
        f"  Estimated tokens: {total_tokens} ({_tracker.input_tokens} in / {_tracker.output_tokens} out, "
        f"out is a fixed {ASSUMED_OUTPUT_TOKENS}/call assumption)",
        f"  Estimated cost:   ${_tracker.cost:.4f}",
        "  (assumes the script's nominal control-flow path; a real run's adaptive",
        "   branching could make fewer or more calls than this)",
    ]
    return "\n".join(lines)


def print_dry_run_summary():
    print(dry_run_summary_text())
