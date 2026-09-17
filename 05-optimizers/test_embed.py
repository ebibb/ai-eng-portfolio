"""
Test that the configured provider's embed() works against a live embedding endpoint.

Run:
    python 05-optimizers/test_embed.py

Requires:
    LLM_PROVIDER and LLM_EMBEDDING_MODEL set in your environment (see ../.env.example).
    Anthropic has no embeddings API — use openai, azure, or ollama for this file.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_provider import get_llm
from cost_estimator import DRY_RUN, print_dry_run_summary
from dotenv import load_dotenv

load_dotenv()
llm = get_llm()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def run_tests():
    if DRY_RUN:
        # Real vectors are placeholder/random in dry-run mode, so the semantic-similarity
        # and consistency assertions below don't apply — just tally the same 5 calls.
        for text in [
            "Classify the support ticket into billing, technical, or general.",
            "my credit card was charged twice",
            "the app crashes every time I open it",
            "I was billed the wrong amount on my invoice",
            "Respond with only the category label.",
            "Respond with only the category label.",
        ]:
            llm.embed(text)
        print_dry_run_summary()
        return

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
            failed += 1

    print("\n── embed() basic contract ────────────────────────────────────────────")

    vec = llm.embed("Classify the support ticket into billing, technical, or general.")

    check("returns a list", isinstance(vec, list))
    check("elements are floats", all(isinstance(x, float) for x in vec))
    check("non-empty vector", len(vec) > 0)
    print(f"  (embedding dimension: {len(vec)} — depends on the configured model)")

    print("\n── embed() semantic properties ───────────────────────────────────────")

    vec_billing = llm.embed("my credit card was charged twice")
    vec_technical = llm.embed("the app crashes every time I open it")
    vec_billing2 = llm.embed("I was billed the wrong amount on my invoice")

    sim_same_category = cosine_similarity(vec_billing, vec_billing2)
    sim_diff_category = cosine_similarity(vec_billing, vec_technical)

    check(
        "similar texts are closer than dissimilar texts",
        sim_same_category > sim_diff_category,
        f"same={sim_same_category:.4f}, diff={sim_diff_category:.4f}",
    )

    print("\n── embed() consistency ───────────────────────────────────────────────")

    text = "Respond with only the category label."
    vec_a = llm.embed(text)
    vec_b = llm.embed(text)
    sim_identical = cosine_similarity(vec_a, vec_b)

    check(
        "same text produces near-identical vectors (cosine > 0.9999)",
        sim_identical > 0.9999,
        f"cosine similarity={sim_identical:.6f}",
    )

    print(f"\n{'─' * 54}")
    print(f"  {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
