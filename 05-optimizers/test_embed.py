"""
Test that AzureLLMWrapper.embed() works against the live Azure embedding endpoint.

Run:
    python 05-optimizers/test_embed.py

Requires:
    AZURE_APIM_ENDPOINT, AZURE_APIM_SUBSCRIPTION_KEY, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_MODEL set in your environment.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from azure_llm_wrapper import AzureLLMWrapper
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY", "")
deployment = "text-embedding-3-small"
version = os.getenv("AZURE_OPENAI_API_VERSION", "")
endpoint = os.getenv("AZURE_APIM_ENDPOINT", "")
embedding_url = f"{endpoint}/openai/deployments/{deployment}/embeddings"
llm = AzureLLMWrapper(endpoint=embedding_url, api_key=api_key)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def run_tests():
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

    vec = llm.embed("Classify the support ticket into billing, technical, or general.", embedding_url, api_version=version)

    check("returns a list", isinstance(vec, list))
    check("elements are floats", all(isinstance(x, float) for x in vec))
    check("non-empty vector", len(vec) > 0)
    check(
        "expected embedding dimension (1536 or 3072)",
        len(vec) in (1536, 3072),
        f"got {len(vec)}",
    )

    print("\n── embed() semantic properties ───────────────────────────────────────")

    vec_billing = llm.embed("my credit card was charged twice", embedding_url, api_version=version)
    vec_technical = llm.embed("the app crashes every time I open it", embedding_url, api_version=version)
    vec_billing2 = llm.embed("I was billed the wrong amount on my invoice", embedding_url, api_version=version)

    sim_same_category = cosine_similarity(vec_billing, vec_billing2)
    sim_diff_category = cosine_similarity(vec_billing, vec_technical)

    check(
        "similar texts are closer than dissimilar texts",
        sim_same_category > sim_diff_category,
        f"same={sim_same_category:.4f}, diff={sim_diff_category:.4f}",
    )

    print("\n── embed() consistency ───────────────────────────────────────────────")

    text = "Respond with only the category label."
    vec_a = llm.embed(text, embedding_url, api_version=version)
    vec_b = llm.embed(text, embedding_url, api_version=version)
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
