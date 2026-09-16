"""
Test suite for optimizer_v3.py — select a test via the CLI menu.

Run:
    python 05-optimizers/test_optimizer_v3.py

Tests marked [NO API] use only numpy/math — fast, free, no credentials needed.
Tests marked [LLM]    make real LLM calls — slow and cost money.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03-eval-harness"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04-llm-as-judge"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import optimizer_v3 as v3
from eval import load_dataset, split

# ── Shared test fixtures ───────────────────────────────────────────────────────

CANDIDATE_A = {
    "instruction": "Classify the ticket into billing, technical, or general.",
    "demos": [
        {"input": "my card was charged twice", "gold": "billing"},
        {"input": "app keeps crashing", "gold": "technical"},
    ],
}

CANDIDATE_B = {
    "instruction": "Categorize the support ticket as billing, technical, or general.",
    "demos": [
        {"input": "I was charged the wrong amount", "gold": "billing"},
        {"input": "I cannot log in", "gold": "technical"},
    ],
}

CANDIDATE_C = {
    "instruction": "Label each ticket: billing, technical, or general. Output only the label.",
    "demos": [
        {"input": "where is my refund", "gold": "billing"},
        {"input": "the website is down", "gold": "technical"},
    ],
}

POOL = [CANDIDATE_A, CANDIDATE_B, CANDIDATE_C]

# ── Helpers ────────────────────────────────────────────────────────────────────

passed = 0
failed = 0

def check(name: str, condition: bool, expected=None, got=None):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        detail = ""
        if expected is not None:
            detail = f"\n          expected: {expected}\n          got:      {got}"
        print(f"  FAIL  {name}{detail}")
        failed += 1

def approx(a, b, tol=1e-4):
    return abs(a - b) < tol

def reset_counts():
    global passed, failed
    passed = 0
    failed = 0

def print_summary():
    print(f"\n  {'─' * 50}")
    print(f"  {passed} passed, {failed} failed")

# ── 1. candidate_to_text ──────────────────────────────────────────────────────

def test_candidate_to_text():
    print("\n── candidate_to_text [NO API] ────────────────────────────────────────")
    print("  Serializes a candidate dict to one flat string.")
    print("  Punctuation, capitalization, and word order must be preserved.\n")

    result = v3.candidate_to_text(CANDIDATE_A)

    print(f"  Input:    {CANDIDATE_A}")
    print(f"  Got:      {result!r}\n")

    expected = "Classify the ticket into billing, technical, or general. | my card was charged twice -> billing | app keeps crashing -> technical"

    check("returns a string",           isinstance(result, str), expected="str", got=type(result))
    check("instruction is preserved",   result is not None and "Classify the ticket into billing, technical, or general." in result)
    check("demo inputs are preserved",  result is not None and "my card was charged twice" in result)
    check("demo labels are preserved",  result is not None and "billing" in result and "technical" in result)
    check("preserves capitalization",   result is not None and "Classify" in result)
    check("preserves punctuation",      result is not None and "," in result)
    check("matches expected format",    result == expected, expected=expected, got=result)

    print_summary()

# ── 2. embed_candidate ────────────────────────────────────────────────────────

def test_embed_candidate():
    print("\n── embed_candidate [NO API] ──────────────────────────────────────────")
    print("  Converts a candidate dict to a fixed-size float vector via TF-IDF.\n")

    result = v3.embed_candidate(CANDIDATE_A)

    print(f"  Input:  CANDIDATE_A")
    print(f"  Got:    {type(result)} shape={getattr(result, 'shape', 'N/A')}\n")

    check("returns np.ndarray",         isinstance(result, np.ndarray), expected="np.ndarray", got=type(result))
    check("is 1-dimensional",           result is not None and result.ndim == 1)
    check("contains floats",            result is not None and result.dtype in (np.float32, np.float64))
    check("non-zero vector",            result is not None and np.any(result != 0))
    check("finite values only",         result is not None and np.all(np.isfinite(result)))

    if result is not None:
        result_b = v3.embed_candidate(CANDIDATE_B)
        result_c = v3.embed_candidate(CANDIDATE_C)
        check("same shape for all candidates",  result.shape == result_b.shape == result_c.shape)
        check("different candidates differ",    result is not None and not np.allclose(result, result_b))

    print_summary()

# ── 3. embed_pool ─────────────────────────────────────────────────────────────

def test_embed_pool():
    print("\n── embed_pool [NO API] ───────────────────────────────────────────────")
    print("  Embeds all candidates; returns (N × D) matrix index-aligned with pool.\n")

    result = v3.embed_pool(POOL)

    print(f"  Input:  {len(POOL)} candidates")
    print(f"  Got:    {type(result)} shape={getattr(result, 'shape', 'N/A')}\n")

    check("returns np.ndarray",         isinstance(result, np.ndarray), expected="np.ndarray", got=type(result))
    check("2-dimensional",              result is not None and result.ndim == 2)
    check("N rows = pool size",         result is not None and result.shape[0] == len(POOL), expected=len(POOL), got=result.shape[0] if result is not None else None)
    check("rows are distinct",          result is not None and not np.allclose(result[0], result[1]))
    check("index-aligned: row 0 == embed_candidate(POOL[0])",
          result is not None and np.allclose(result[0], v3.embed_candidate(POOL[0])))

    print_summary()

# ── 4. rbf_kernel ─────────────────────────────────────────────────────────────

def test_rbf_kernel():
    print("\n── rbf_kernel [NO API] ───────────────────────────────────────────────")
    print("  k(x1, x2) = signal_var * exp(−‖x1−x2‖² / (2 * length_scale²))\n")

    x_zero  = np.array([0.0, 0.0])
    x_unit  = np.array([1.0, 0.0])
    x_unit2 = np.array([0.0, 1.0])
    x_far   = np.array([100.0, 100.0])

    # identical vectors → signal_var
    k_self = v3.rbf_kernel(x_unit, x_unit, length_scale=1.0, signal_var=1.0)
    print(f"  rbf_kernel([1,0], [1,0], l=1, sv=1)  expected=1.0       got={k_self}")
    check("identical vectors → signal_var=1.0",  approx(k_self, 1.0), expected=1.0, got=k_self)

    # signal_var scaling
    k_sv2 = v3.rbf_kernel(x_unit, x_unit, length_scale=1.0, signal_var=2.0)
    print(f"  rbf_kernel([1,0], [1,0], l=1, sv=2)  expected=2.0       got={k_sv2}")
    check("identical vectors → signal_var=2.0",  approx(k_sv2, 2.0), expected=2.0, got=k_sv2)

    # orthogonal vectors distance²=2, exp(-2/2)=exp(-1)≈0.3679
    k_orth = v3.rbf_kernel(x_unit, x_unit2, length_scale=1.0, signal_var=1.0)
    expected_orth = math.exp(-1.0)
    print(f"  rbf_kernel([1,0], [0,1], l=1, sv=1)  expected={expected_orth:.4f}  got={k_orth}")
    check("orthogonal unit vectors → exp(-1)",   approx(k_orth, expected_orth), expected=round(expected_orth,4), got=round(k_orth,4) if k_orth is not None else None)

    # far apart → near 0
    k_far = v3.rbf_kernel(x_zero, x_far, length_scale=1.0, signal_var=1.0)
    print(f"  rbf_kernel([0,0], [100,100], l=1)    expected≈0.0       got={k_far}")
    check("very far vectors → near 0",           k_far is not None and k_far < 1e-10, expected="< 1e-10", got=k_far)

    # output always in (0, signal_var]
    check("output ≤ signal_var",                 k_orth is not None and k_orth <= 1.0 + 1e-9)
    check("output > 0",                          k_orth is not None and k_orth > 0)

    # symmetry: k(x1,x2) == k(x2,x1)
    k_ab = v3.rbf_kernel(x_unit, x_unit2, length_scale=1.0, signal_var=1.0)
    k_ba = v3.rbf_kernel(x_unit2, x_unit, length_scale=1.0, signal_var=1.0)
    check("symmetric: k(x1,x2) == k(x2,x1)",   approx(k_ab, k_ba))

    print_summary()

# ── 5. covariance_matrix ──────────────────────────────────────────────────────

def test_covariance_matrix():
    print("\n── covariance_matrix [NO API] ────────────────────────────────────────")
    print("  K[i,j] = rbf_kernel(X1[i], X2[j]) for all pairs.\n")

    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    K = v3.covariance_matrix(X, X, length_scale=1.0, signal_var=1.0)

    print(f"  Input:  X shape {X.shape}, l=1.0, signal_var=1.0")
    print(f"  Got K:\n{K}\n")

    check("returns np.ndarray",             isinstance(K, np.ndarray), expected="np.ndarray", got=type(K))
    check("shape is (M, M)",                K is not None and K.shape == (3, 3), expected=(3,3), got=K.shape if K is not None else None)
    check("diagonal = signal_var",          K is not None and np.allclose(np.diag(K), 1.0), expected=1.0, got=np.diag(K) if K is not None else None)
    check("symmetric",                      K is not None and np.allclose(K, K.T))
    check("all values in (0, signal_var]",  K is not None and np.all(K > 0) and np.all(K <= 1.0 + 1e-9))

    # rectangular: K_star shape (K_rows, M_cols)
    X_query = np.array([[0.5, 0.5], [1.0, 1.0]])
    K_rect = v3.covariance_matrix(X_query, X, length_scale=1.0, signal_var=1.0)
    print(f"  Rectangular K_star shape: {K_rect.shape if K_rect is not None else 'None'} (expected (2, 3))")
    check("rectangular shape (K, M)",       K_rect is not None and K_rect.shape == (2, 3), expected=(2,3), got=K_rect.shape if K_rect is not None else None)

    # spot check K[0,1] matches rbf_kernel directly
    expected_01 = v3.rbf_kernel(X[0], X[1], 1.0, 1.0)
    check("K[0,1] matches rbf_kernel(X[0], X[1])", K is not None and approx(K[0, 1], expected_01))

    print_summary()

# ── 6. gp_fit ─────────────────────────────────────────────────────────────────

def test_gp_fit():
    print("\n── gp_fit [NO API] ───────────────────────────────────────────────────")
    print("  Fits GP on observed (embedding, score) pairs via Cholesky decomposition.")
    print("  Returns state dict with X_obs, L, alpha, hyperparams.\n")

    X_obs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    y_obs = np.array([0.8, 0.6, 0.7])

    print(f"  X_obs shape: {X_obs.shape}  y_obs: {y_obs}")

    state = v3.gp_fit(X_obs, y_obs, length_scale=1.0, signal_var=1.0, noise_var=0.01)

    print(f"  Got keys: {list(state.keys()) if isinstance(state, dict) else state}\n")

    check("returns a dict",                     isinstance(state, dict), expected="dict", got=type(state))
    check("contains 'X_obs'",                   state is not None and "X_obs" in state)
    check("contains 'L'",                       state is not None and "L" in state)
    check("contains 'alpha'",                   state is not None and "alpha" in state)
    # check("contains 'hyperparams'",             state is not None and "hyperparams" in state)
    check("L is lower triangular (M × M)",      state is not None and "L" in state and state["L"].shape == (3, 3))
    check("alpha shape is (M,)",                state is not None and "alpha" in state and state["alpha"].shape == (3,))
    check("L is finite",                        state is not None and "L" in state and np.all(np.isfinite(state["L"])))
    check("alpha is finite",                    state is not None and "alpha" in state and np.all(np.isfinite(state["alpha"])))

    print_summary()

# ── 7. gp_predict ─────────────────────────────────────────────────────────────

def test_gp_predict():
    print("\n── gp_predict [NO API] ───────────────────────────────────────────────")
    print("  Returns (mean, std) for query points given a fitted GP.")
    print("  At training points: mean ≈ training score, std ≈ 0.\n")

    X_obs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    y_obs = np.array([0.8, 0.6, 0.7])
    state = v3.gp_fit(X_obs, y_obs, length_scale=1.0, signal_var=1.0, noise_var=1e-6)

    if state is None:
        print("  SKIP — gp_fit not implemented yet")
        return

    # predict at training points
    mean, std = v3.gp_predict(state, X_obs)

    print(f"  Predicting at training points:")
    print(f"  mean: {mean}  (expected ≈ {y_obs})")
    print(f"  std:  {std}   (expected ≈ [0, 0, 0])\n")

    check("mean is np.ndarray shape (M,)",       isinstance(mean, np.ndarray) and mean.shape == (3,))
    check("std is np.ndarray shape (M,)",        isinstance(std, np.ndarray) and std.shape == (3,))
    check("mean ≈ training scores at obs points", np.allclose(mean, y_obs, atol=0.05), expected=y_obs, got=mean)
    check("std ≈ 0 at observed points",           np.all(std < 0.05), expected="< 0.05", got=std)
    check("std ≥ 0 everywhere",                   np.all(std >= 0))

    # predict at an unseen point — std should be higher than at training points
    X_new = np.array([[5.0, 5.0]])
    mean_new, std_new = v3.gp_predict(state, X_new)
    print(f"  Predicting at unseen point [5, 5]:")
    print(f"  mean: {mean_new}  std: {std_new}  (std should be > std at training points)")
    check("std higher at unseen point than at training points", std_new[0] > np.mean(std))

    print_summary()

# ── 8. normal_pdf ─────────────────────────────────────────────────────────────

def test_normal_pdf():
    print("\n── normal_pdf [NO API] ───────────────────────────────────────────────")
    print("  phi(z) = exp(−z²/2) / sqrt(2π)\n")

    cases = [
        (0.0,   1 / math.sqrt(2 * math.pi)),   # ≈ 0.39894
        (1.0,   math.exp(-0.5) / math.sqrt(2 * math.pi)),   # ≈ 0.24197
        (-1.0,  math.exp(-0.5) / math.sqrt(2 * math.pi)),   # symmetric
        (2.0,   math.exp(-2.0) / math.sqrt(2 * math.pi)),   # ≈ 0.05399
    ]

    for z, expected in cases:
        got = v3.normal_pdf(z)
        print(f"  normal_pdf({z:5.1f})  expected={expected:.6f}  got={got}")
        check(f"normal_pdf({z})", got is not None and approx(got, expected), expected=round(expected,6), got=round(got,6) if got is not None else None)

    # symmetry
    check("symmetric: phi(-1) == phi(1)",   approx(v3.normal_pdf(-1.0), v3.normal_pdf(1.0)))
    # always positive
    check("phi(z) > 0 for all z",           v3.normal_pdf(0) is not None and v3.normal_pdf(0) > 0)
    # peak at 0
    check("peak at z=0",                    v3.normal_pdf(0) is not None and v3.normal_pdf(0) > v3.normal_pdf(1))

    print_summary()

# ── 9. normal_cdf ─────────────────────────────────────────────────────────────

def test_normal_cdf():
    print("\n── normal_cdf [NO API] ───────────────────────────────────────────────")
    print("  Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))\n")

    cases = [
        (0.0,   0.5),
        (1.0,   0.8413447),
        (-1.0,  0.1586553),
        (1.96,  0.9750021),
        (-1.96, 0.0249979),
    ]

    for z, expected in cases:
        got = v3.normal_cdf(z)
        print(f"  normal_cdf({z:5.2f})  expected={expected:.6f}  got={got}")
        check(f"normal_cdf({z})", got is not None and approx(got, expected, tol=1e-5), expected=round(expected,6), got=round(got,6) if got is not None else None)

    check("Phi(0) = 0.5",                  approx(v3.normal_cdf(0), 0.5))
    check("Phi(-z) = 1 - Phi(z)",          v3.normal_cdf(-1) is not None and approx(v3.normal_cdf(-1), 1 - v3.normal_cdf(1)))
    check("Phi(large) ≈ 1",                v3.normal_cdf(10) is not None and approx(v3.normal_cdf(10), 1.0))
    check("Phi(-large) ≈ 0",               v3.normal_cdf(-10) is not None and approx(v3.normal_cdf(-10), 0.0))

    print_summary()

# ── 10. expected_improvement ──────────────────────────────────────────────────

def test_expected_improvement():
    print("\n── expected_improvement [NO API] ────────────────────────────────────")
    print("  EI = (mean − best − xi)·Phi(Z) + std·phi(Z)\n")

    # case 1: mean clearly above best, std=0 → pure exploitation
    mean1 = np.array([0.9])
    std1  = np.array([1e-9])
    ei1   = v3.expected_improvement(mean1, std1, best_so_far=0.8, xi=0.01)
    # Z → +inf, Phi→1, phi→0; EI = (0.9-0.8-0.01)*1 = 0.09
    expected1 = 0.09
    print(f"  EI(mean=0.9, std≈0, best=0.8, xi=0.01)  expected≈{expected1}  got={ei1}")
    check("above best with no uncertainty → EI ≈ mean−best−xi", ei1 is not None and approx(ei1[0], expected1, tol=1e-3))

    # case 2: mean below best, std=0 → EI = 0
    mean2 = np.array([0.5])
    std2  = np.array([1e-9])
    ei2   = v3.expected_improvement(mean2, std2, best_so_far=0.8, xi=0.01)
    print(f"  EI(mean=0.5, std≈0, best=0.8)            expected=0.0       got={ei2}")
    check("below best with no uncertainty → EI = 0",  ei2 is not None and approx(ei2[0], 0.0, tol=1e-6))

    # case 3: mean = best, but std is large → exploration gives EI > 0
    mean3 = np.array([0.8])
    std3  = np.array([0.2])
    ei3   = v3.expected_improvement(mean3, std3, best_so_far=0.8, xi=0.01)
    print(f"  EI(mean=0.8, std=0.2, best=0.8)          expected>0         got={ei3}")
    check("high uncertainty gives EI > 0 even at best",  ei3 is not None and ei3[0] > 0)

    # case 4: higher std → higher EI (exploration)
    mean4 = np.array([0.75, 0.75])
    std4  = np.array([0.1,  0.3])
    ei4   = v3.expected_improvement(mean4, std4, best_so_far=0.8, xi=0.01)
    print(f"  EI(mean=[0.75,0.75], std=[0.1,0.3])      expected ei[1]>ei[0]  got={ei4}")
    check("higher std → higher EI",  ei4 is not None and ei4[1] > ei4[0])

    # returns array same length as mean
    mean5 = np.array([0.7, 0.8, 0.9])
    std5  = np.array([0.1, 0.1, 0.1])
    ei5   = v3.expected_improvement(mean5, std5, best_so_far=0.75, xi=0.01)
    check("output shape matches input", ei5 is not None and ei5.shape == (3,))
    check("EI ≥ 0 always",             ei5 is not None and np.all(ei5 >= 0))

    print_summary()

# ── 11. select_next_by_ei ────────────────────────────────────────────────────

def test_select_next_by_ei():
    print("\n── select_next_by_ei [NO API] ───────────────────────────────────────")
    print("  Should return the candidate with the highest Expected Improvement.\n")

    X_obs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_obs = np.array([0.8, 0.6])
    state = v3.gp_fit(X_obs, y_obs, length_scale=1.0, signal_var=1.0, noise_var=0.01)

    if state is None:
        print("  SKIP — gp_fit not implemented yet")
        return

    # place remaining candidates at known distances from training points
    X_remaining = np.array([
        [0.9, 0.1],   # close to X_obs[0] (score 0.8) → should have high mean
        [5.0, 5.0],   # far from everything → high uncertainty
        [0.1, 0.9],   # close to X_obs[1] (score 0.6) → lower mean
    ])
    remaining_candidates = [CANDIDATE_A, CANDIDATE_B, CANDIDATE_C]

    idx, chosen = v3.select_next_by_ei(remaining_candidates, X_remaining, state, best_so_far=0.7)

    print(f"  Got index: {idx}  chosen instruction: {chosen.get('instruction','')[:60] if chosen else None}\n")

    check("returns int index",          isinstance(idx, int), expected="int", got=type(idx))
    check("index in valid range",       idx is not None and 0 <= idx < len(remaining_candidates))
    check("returned candidate matches index",  chosen is remaining_candidates[idx] if idx is not None else False)

    print_summary()

# ── 12. evaluate_candidate [LLM] ─────────────────────────────────────────────

def test_evaluate_candidate():
    print("\n── evaluate_candidate [LLM] ─────────────────────────────────────────")
    print("  Scores one candidate on the val set — makes real LLM calls.\n")

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "03-eval-harness", "data", "dataset.jsonl")
    dataset = load_dataset(DATA_PATH)
    if not dataset:
        print("  SKIP — dataset.jsonl not populated")
        return
    val = split(dataset, "val")

    print(f"  Evaluating CANDIDATE_A on {len(val)} val examples...")
    score = v3.evaluate_candidate(CANDIDATE_A, val)
    print(f"  Score: {score}\n")

    check("returns a float",            isinstance(score, float), expected="float", got=type(score))
    check("score in [0, 1]",            score is not None and 0.0 <= score <= 1.0, expected="[0,1]", got=score)

    print_summary()

# ── 13. warm_start [LLM] ─────────────────────────────────────────────────────

def test_warm_start():
    print("\n── warm_start [LLM] ─────────────────────────────────────────────────")
    print("  Randomly evaluates n_initial candidates; seeds GP observations.\n")

    X_all = v3.embed_pool(POOL)
    if X_all is None:
        print("  SKIP — embed_pool not implemented yet")
        return

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "03-eval-harness", "data", "dataset.jsonl")
    dataset = load_dataset(DATA_PATH)
    if not dataset:
        print("  SKIP — dataset.jsonl not populated")
        return
    val = split(dataset, "val")

    n_initial = 2
    print(f"  Pool size: {len(POOL)}, n_initial: {n_initial}")
    X_obs, y_obs, used_indices = v3.warm_start(POOL, X_all, val, n_initial)
    print(f"  X_obs shape: {X_obs.shape if X_obs is not None else None}")
    print(f"  y_obs: {y_obs}")
    print(f"  used_indices: {used_indices}\n")

    check("X_obs shape (n_initial, D)",     X_obs is not None and X_obs.shape[0] == n_initial)
    check("y_obs shape (n_initial,)",       y_obs is not None and y_obs.shape == (n_initial,))
    check("used_indices length = n_initial", len(used_indices) == n_initial)
    check("indices are valid",              all(0 <= i < len(POOL) for i in used_indices))
    check("indices are unique",             len(set(used_indices)) == n_initial)
    check("scores in [0, 1]",              y_obs is not None and np.all((y_obs >= 0) & (y_obs <= 1)))

    print_summary()

# ── 14. optimize_v3 [LLM] ────────────────────────────────────────────────────

def test_optimize_v3():
    print("\n── optimize_v3 [LLM] ────────────────────────────────────────────────")
    print("  Full BO run — makes many LLM calls. This will take several minutes.\n")

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "03-eval-harness", "data", "dataset.jsonl")
    dataset = load_dataset(DATA_PATH)
    if not dataset:
        print("  SKIP — dataset.jsonl not populated")
        return
    train = split(dataset, "train")
    val   = split(dataset, "val")

    best_candidate, best_score = v3.optimize_v3(
        train, val,
        n_instructions=2, n_demo_subsets=3, n_demos=2,
        n_initial=2, n_iterations=3,
    )

    print(f"\n  best_score: {best_score}")
    print(f"  best_candidate instruction: {best_candidate.get('instruction','')[:100] if best_candidate else None}\n")

    check("returns a dict candidate",       isinstance(best_candidate, dict))
    check("candidate has 'instruction'",    best_candidate is not None and "instruction" in best_candidate)
    check("candidate has 'demos'",          best_candidate is not None and "demos" in best_candidate)
    check("best_score is float in [0,1]",   isinstance(best_score, float) and 0.0 <= best_score <= 1.0)

    print_summary()

# ── Menu ──────────────────────────────────────────────────────────────────────

TESTS = {
    "1":  ("candidate_to_text      [NO API]", test_candidate_to_text),
    "2":  ("embed_candidate        [NO API]", test_embed_candidate),
    "3":  ("embed_pool             [NO API]", test_embed_pool),
    "4":  ("rbf_kernel             [NO API]", test_rbf_kernel),
    "5":  ("covariance_matrix      [NO API]", test_covariance_matrix),
    "6":  ("gp_fit                 [NO API]", test_gp_fit),
    "7":  ("gp_predict             [NO API]", test_gp_predict),
    "8":  ("normal_pdf             [NO API]", test_normal_pdf),
    "9":  ("normal_cdf             [NO API]", test_normal_cdf),
    "10": ("expected_improvement   [NO API]", test_expected_improvement),
    "11": ("select_next_by_ei      [NO API]", test_select_next_by_ei),
    "12": ("evaluate_candidate     [LLM]   ", test_evaluate_candidate),
    "13": ("warm_start             [LLM]   ", test_warm_start),
    "14": ("optimize_v3            [LLM]   ", test_optimize_v3),
}

NO_API_TESTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

def print_menu():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║         optimizer_v3 test suite                     ║")
    print("╠══════════════════════════════════════════════════════╣")
    for key, (label, _) in TESTS.items():
        print(f"║  {key:>2}.  {label:<46}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║   0.  Run all [NO API] tests (1–11)                 ║")
    print("║   q.  Quit                                          ║")
    print("╚══════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    while True:
        print_menu()
        choice = input("\nSelect test: ").strip().lower()

        if choice == "q":
            break
        elif choice == "0":
            reset_counts()
            for key in NO_API_TESTS:
                _, fn = TESTS[key]
                fn()
            print(f"\n{'═'*54}")
            print(f"  TOTAL: {passed} passed, {failed} failed across all [NO API] tests")
        elif choice in TESTS:
            reset_counts()
            _, fn = TESTS[choice]
            fn()
        else:
            print(f"  Unknown option: {choice!r}")
