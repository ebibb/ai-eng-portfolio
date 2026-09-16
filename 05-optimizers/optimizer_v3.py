"""
Bayesian optimization over instruction × demo candidate space.

Replaces the random proposal-and-select from v2 with a Gaussian Process surrogate +
Expected Improvement acquisition loop. Each iteration fits the GP on all observed
(candidate, val_score) pairs, then uses EI to pick the single candidate most worth
evaluating next. All math is from scratch using numpy and the standard library only.

Same PROPOSE → EVALUATE → SELECT skeleton as v1/v2. PROPOSE is identical to v2.
What changes: instead of scoring every candidate, we score strategically.

Run:
    python 05-optimizers/optimizer_v3.py

Requires:
    AZURE_APIM_ENDPOINT, AZURE_APIM_SUBSCRIPTION_KEY, AZURE_OPENAI_MODEL,
    AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_MODEL set in your environment.
    03-eval-harness/data/dataset.jsonl populated with real examples.
"""

import datetime
import math
import os
import random
import sys
from typing import Callable

import numpy as np

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "03-eval-harness"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "04-llm-as-judge"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from azure_llm_wrapper import AzureLLMWrapper  # noqa: E402
from eval import evaluate, metric, load_dataset, split  # noqa: E402
from optimizer_v1 import zero_shot_program, bootstrap, build_program, build_prompt_template  # noqa: E402
from optimizer_v2 import propose_instruction_candidates, propose_candidates_v2, SEED_INSTRUCTION, optimize_v2  # noqa: E402

load_dotenv()
api_key = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY", "")
deployment = "text-embedding-3-small"
version = os.getenv("AZURE_OPENAI_API_VERSION", "")
endpoint = os.getenv("AZURE_APIM_ENDPOINT", "")
embedding_url = f"{endpoint}/openai/deployments/{deployment}/embeddings"
llm = AzureLLMWrapper(endpoint=embedding_url, api_key=api_key)

Dataset = list[dict]

_run_log: dict = {}
_embed_cache: dict[str, np.ndarray] = {}


# ── Section 1: Text Embedding ──────────────────────────────────────────────────


def candidate_to_text(candidate: dict) -> str:
    # Serialize a candidate to one string for the embedding API, preserving
    # original punctuation, capitalization, and word order.
    instruction = candidate.get("instruction", "")
    demos = candidate.get("demos", [])
    demo_texts = [f"{demo['input']} -> {demo['gold']}" for demo in demos]
    parts = [instruction] + demo_texts
    return " | ".join(parts)


def embed_candidate(candidate: dict) -> np.ndarray:
    # Call the Azure embedding API on the candidate's text representation;
    # return a 1536-dim semantic vector as a numpy array.
    text = candidate_to_text(candidate)
    if text not in _embed_cache:
        response = llm.embed(text, embedding_url, api_version=version)
        _embed_cache[text] = np.array(response)
    return _embed_cache[text]


def embed_pool(candidates: list[dict]) -> np.ndarray:
    # Embed every candidate in the pool upfront; return an (N × 1536) matrix
    # index-aligned with the candidates list.
    embeddings = []
    for candidate in candidates:
        embeddings.append(embed_candidate(candidate))
    return np.array(embeddings)


# ── Math Logging Helpers ──────────────────────────────────────────────────────


def _fmt_vec(v: np.ndarray, label: str = "") -> list[str]:
    prefix = f"  {label}" if label else "  vec"
    return [f"{prefix}: {np.round(v, 6).tolist()}"]


def _fmt_mat(M: np.ndarray, label: str = "") -> list[str]:
    rows, cols = M.shape
    prefix = f"  {label}" if label else "  mat"
    if cols > 100:
        return [f"{prefix}: shape={M.shape}"]
    lines = [f"{prefix}: shape={M.shape}"]
    for row in M:
        lines.append("    [" + "  ".join(f"{x:9.5f}" for x in row) + "]")
    return lines


# ── Section 2: Gaussian Process (from scratch) ────────────────────────────────


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float, signal_var: float) -> float:
    # RBF kernel: signal_var * exp(−‖x1−x2‖² / (2 * length_scale²)).
    # Returns how similar two candidate embeddings are.
    # x1 and x2 are vector embeddings of shape (1536,).
    # length_scale controls how quickly similarity decays with distance
    # signal_var controls the overall scale of the kernel. (overall amplitude of the kernel)
    num = np.linalg.norm(x1 - x2) ** 2
    den = 2 * (length_scale ** 2)
    return signal_var * np.exp(-num / den)


def covariance_matrix(X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float) -> np.ndarray:
    # Build the full (M × N) pairwise kernel matrix between two sets of embeddings
    # using vectorized numpy broadcasting.
    result = np.vectorize(rbf_kernel, signature='(n),(n),(),()->()')(X1[:, np.newaxis, :], X2[np.newaxis, :, :], length_scale, signal_var)
    return result


def gp_fit(X_obs: np.ndarray, y_obs: np.ndarray, length_scale: float, signal_var: float, noise_var: float) -> dict:
    # Fit the GP surrogate on observed (embedding, score) pairs via Cholesky
    # decomposition; return a state dict with X_obs, L, alpha, and hyperparams.
    noise_matrix = noise_var * np.eye(len(X_obs)) # identity matrix * noise_var
    K_y = covariance_matrix(X_obs, X_obs, length_scale, signal_var) + noise_matrix
    L = np.linalg.cholesky(K_y)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    return {
        "X_obs": X_obs,
        "K_y": K_y,
        "L": L,
        "alpha": alpha,
        "length_scale": length_scale,
        "signal_var": signal_var,
        "noise_var": noise_var
    }


def gp_predict(gp_state: dict, X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Query the fitted GP for posterior mean and std at un-evaluated candidate
    # locations; return (mean, std, k_star) where mean and std are shape (K,)
    # and k_star is the (K x n_obs) cross-covariance matrix.
    # For the RBF kernel, diag(K(X_query, X_query)) == signal_var everywhere,
    # so posterior var = signal_var - sum(v**2) without computing K_ss.
    k_star = covariance_matrix(X_query, gp_state["X_obs"], gp_state["length_scale"], gp_state["signal_var"])
    mean = k_star @ gp_state["alpha"]
    v = np.linalg.solve(gp_state["L"], k_star.T)
    var = gp_state["signal_var"] - np.sum(v ** 2, axis=0)
    std = np.sqrt(np.maximum(var, 0))
    return mean, std, k_star


# ── Section 3: Acquisition Function (from scratch) ────────────────────────────


def normal_pdf(z: float) -> float:
    # Standard normal PDF: exp(−z²/2) / sqrt(2π) using math.exp and math.pi.
    return math.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)


def normal_cdf(z: float) -> float:
    # Standard normal CDF: 0.5 * (1 + erf(z / sqrt(2))) using math.erf.
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def expected_improvement(mean: np.ndarray, std: np.ndarray, best_so_far: float, xi: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    # Score each remaining candidate by expected improvement over best_so_far;
    # balances exploitation (high mean) and exploration (high std).
    # Returns (ei, z) so callers can log the z-scores used in Φ(z) and φ(z).
    std_safe = np.maximum(std, 1e-9)
    z = (mean - best_so_far - xi) / std_safe
    ei = (mean - best_so_far - xi) * np.vectorize(normal_cdf)(z) + std * np.vectorize(normal_pdf)(z)
    return np.maximum(ei, 0.0), z


def select_next_by_ei(
    remaining_candidates: list[dict], X_remaining: np.ndarray, gp_state: dict, best_so_far: float
) -> tuple[int, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Acquisition step: predict with GP, compute EI, return argmax candidate
    # and its local index along with intermediate math arrays (mean, std,
    # k_star, ei, z) for logging.
    mean, std, k_star = gp_predict(gp_state, X_remaining)
    ei, z = expected_improvement(mean, std, best_so_far)
    best_index = int(np.argmax(ei))
    return best_index, remaining_candidates[best_index], mean, std, k_star, ei, z


# ── Section 4: Evaluation and Warm Start ──────────────────────────────────────


def evaluate_candidate(candidate: dict, val: Dataset) -> float:
    # Score one candidate on the val set; wraps build_program() + metric().
    # This is the expensive black-box objective — one LLM call per val example.
    program = build_program(candidate)
    predictions = []
    for ex in val:
        pred = program(ex["input"])
        s = metric(pred, ex["gold"])
        predictions.append({"input": ex["input"], "prediction": pred, "gold": ex["gold"], "correct": s == 1.0})
    return sum(p["correct"] for p in predictions) / len(predictions) if predictions else 0.0


def warm_start(candidates: list[dict], X_all: np.ndarray, val: Dataset, n_initial: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    # Randomly evaluate n_initial candidates to seed the GP before it takes over;
    # return (X_observed, y_observed, evaluated_indices).
    initial_indices = random.sample(range(len(candidates)), min(n_initial, len(candidates)))
    X_observed = X_all[initial_indices]
    y_observed = np.array([evaluate_candidate(candidates[i], val) for i in initial_indices])
    return X_observed, y_observed, initial_indices


# ── Section 5: Main Loop ───────────────────────────────────────────────────────


def optimize_v3(
    train: Dataset,
    val: Dataset,
    seed_instruction: str = SEED_INSTRUCTION,
    n_instructions: int = 3,
    n_demo_subsets: int = 5,
    n_demos: int = 3,
    n_initial: int = 4,
    n_iterations: int = 10,
    length_scale: float = 1.0,
    signal_var: float = 1.0,
    noise_var: float = 0.1,
) -> tuple[dict, float]:
    # Full Bayesian optimization loop: zero-shot baseline → bootstrap → propose
    # candidate pool (identical to v2) → embed pool → warm start → BO iterations
    # (fit GP, acquire next by EI, evaluate, update) → compare vs v2 → return best.

    _run_log.clear()
    _run_log["math"] = {"embedding": {}, "rbf_kernel_sample": {}, "warm_start": {}, "bo_iterations": []}
    _run_log["hyperparams"] = {
        "seed_instruction": seed_instruction,
        "n_instructions": n_instructions,
        "n_demo_subsets": n_demo_subsets,
        "n_demos": n_demos,
        "n_initial": n_initial,
        "n_iterations": n_iterations,
        "length_scale": length_scale,
        "signal_var": signal_var,
        "noise_var": noise_var,
    }


    # ── PROPOSE ───────────────────────────────────────────────
    # 1. Establish zero-shot baseline score on val and print it.
    zero_shot_prog = zero_shot_program(seed_instruction)
    zero_shot_score = evaluate(zero_shot_prog, val)
    print(f"Zero-shot baseline: {zero_shot_score:.3f}")
    _run_log["zero_shot_score"] = float(zero_shot_score)

    # 2. bootstrap() to get the demo pool (same as v1 & v2).
    demo_pool = bootstrap(zero_shot_prog, train=train)
    if not demo_pool:
        print("Bootstrap found no correct examples — cannot propose candidates.")
        return None, zero_shot_score

    # 3. propose_instruction_candidates() to get instruction variants.
    instructions = propose_instruction_candidates(seed_instruction=seed_instruction, demo_pool=demo_pool, n_instructions=n_instructions)
    if not instructions:
        print("No instruction candidates proposed.")
        return None, zero_shot_score

    # 4. propose_candidates_v2() to get all instruction × demo candidates.
    candidates = propose_candidates_v2(demo_pool=demo_pool, instructions=instructions, n_demo_subsets=n_demo_subsets, n_demos=n_demos)
    if not candidates:
        print("No candidates proposed.")
        return None, zero_shot_score
    _run_log["n_candidates"] = len(candidates)

    # 5. embed candidate pool upfront
    X_all = embed_pool(candidates)
    _run_log["math"]["embedding"] = {"X_all_shape": list(X_all.shape)}

    # ── EVALUATE ───────────────────────────────────────────
    # 6. warm start: randomly evaluate n_initial candidates to seed the GP.
    X_observed, y_observed, evaluated_indices = warm_start(candidates, X_all, val, n_initial)
    _run_log["warm_start"] = {
        "indices": list(evaluated_indices),
        "scores": y_observed.tolist(),
    }
    # -- math log: warm start embeddings
    _run_log["math"]["warm_start"] = {
        "X_observed_shape": list(X_observed.shape),
        "y_observed": y_observed.copy(),
    }

    # 7. BO iterations: fit GP, acquire next candidate by EI, evaluate, update observed set.
    _run_log["iterations"] = []
    for iteration in range(n_iterations):
        if not [i for i in range(len(candidates)) if i not in evaluated_indices]:
            print("All candidates evaluated — stopping early.")
            break
        gp_state = gp_fit(X_observed, y_observed, length_scale, signal_var, noise_var)
        remaining_indices = [i for i in range(len(candidates)) if i not in evaluated_indices]
        X_remaining = X_all[remaining_indices]
        best_so_far = float(max(y_observed)) if len(y_observed) > 0 else 0.0
        next_index_local, next_candidate, mean_all, std_all, k_star, ei_all, z_all = select_next_by_ei(
            [candidates[i] for i in remaining_indices], X_remaining, gp_state, best_so_far
        )
        next_index_global = remaining_indices[next_index_local]
        pred_mean = float(mean_all[next_index_local])
        pred_std = float(std_all[next_index_local])
        next_score = evaluate_candidate(next_candidate, val)
        X_observed = np.vstack([X_observed, X_all[next_index_global]])
        y_observed = np.append(y_observed, next_score)
        evaluated_indices.append(next_index_global)
        print(f"Iteration {iteration + 1}/{n_iterations}: candidate {next_index_global}  "
              f"gp_mean={pred_mean:.3f}  gp_std={pred_std:.3f}  actual={next_score:.3f}")
        _run_log["iterations"].append({
            "iteration": iteration + 1,
            "candidate_index": next_index_global,
            "instruction": next_candidate.get("instruction", ""),
            "demos": next_candidate.get("demos", []),
            "gp_predicted_mean": pred_mean,
            "gp_predicted_std": pred_std,
            "actual_score": float(next_score),
            "best_so_far_before": best_so_far,
        })
        _run_log["math"]["bo_iterations"].append({
            "iteration": iteration + 1,
            "n_observed_before": len(y_observed) - 1,
            "best_so_far": best_so_far,
            "K_y": gp_state["K_y"].copy(),
            "L": gp_state["L"].copy(),
            "alpha": gp_state["alpha"].copy(),
            "k_star": k_star.copy(),
            "mean_remaining": mean_all.copy(),
            "std_remaining": std_all.copy(),
            "z_remaining": z_all.copy(),
            "ei_remaining": ei_all.copy(),
            "selected_local_index": next_index_local,
            "selected_global_index": next_index_global,
            "selected_mean": pred_mean,
            "selected_std": pred_std,
            "selected_ei": float(ei_all[next_index_local]),
            "actual_score": float(next_score),
        })

    # ── SELECT ─────────────────────────────────
    # 8. Select the best candidate seen across warm-start and BO iterations.
    best_v3_index = int(np.argmax(y_observed))
    best_v3_candidate = candidates[evaluated_indices[best_v3_index]]
    best_v3_score = float(y_observed[best_v3_index])
    _run_log["best_v3_score"] = best_v3_score
    _run_log["best_v3_instruction"] = best_v3_candidate.get("instruction", "")
    _run_log["best_v3_demos"] = best_v3_candidate.get("demos", [])

    return best_v3_candidate, best_v3_score


def write_run_log(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["=" * 62, "optimizer_v3 run log", "=" * 62, ""]

    if "hyperparams" in _run_log:
        lines.append("── Hyperparameters ─────────────────────────────────────────")
        for k, v in _run_log["hyperparams"].items():
            lines.append(f"  {k:<20}: {v}")
        lines.append("")

    if "zero_shot_score" in _run_log:
        lines.append(f"  zero-shot baseline : {_run_log['zero_shot_score']:.3f}")
    if "n_candidates" in _run_log:
        lines.append(f"  candidate pool size: {_run_log['n_candidates']}")
    lines.append("")

    if "warm_start" in _run_log:
        ws = _run_log["warm_start"]
        lines.append("── Warm Start ──────────────────────────────────────────────")
        for idx, score in zip(ws["indices"], ws["scores"]):
            lines.append(f"  candidate {idx:3d}: score={score:.3f}")
        lines.append("")

    if "iterations" in _run_log and _run_log["iterations"]:
        lines.append("── BO Iterations ───────────────────────────────────────────")
        for it in _run_log["iterations"]:
            lines.append(f"  Iter {it['iteration']}  candidate={it['candidate_index']}  "
                         f"gp_mean={it['gp_predicted_mean']:.4f}  gp_std={it['gp_predicted_std']:.4f}  "
                         f"actual={it['actual_score']:.3f}  best_before={it['best_so_far_before']:.3f}")
            lines.append(f"    instruction: {it['instruction']}")
            for i, demo in enumerate(it.get("demos", [])):
                lines.append(f"    demo[{i}]     input: {demo.get('input', '')}")
                lines.append(f"             gold : {demo.get('gold', '')}")
        lines.append("")

    # ── Math detail sections ──────────────────────────────────────────────────
    math = _run_log.get("math", {})

    emb = math.get("embedding", {})
    if emb:
        lines.append("── Math: Embedding ─────────────────────────────────────────")
        lines.append(f"  X_all: shape={emb['X_all_shape']}  (candidates x embedding_dim)")
        lines.append("")

    ws_math = math.get("warm_start", {})
    if ws_math:
        lines.append("── Math: Warm Start ────────────────────────────────────────")
        lines.append(f"  X_observed: shape={ws_math['X_observed_shape']}")
        lines += _fmt_vec(ws_math["y_observed"], label="y_observed")
        lines.append("")

    bo_iters = math.get("bo_iterations", [])
    for it in bo_iters:
        n = it["iteration"]
        lines.append(f"── Math: BO Iteration {n} ({'─' * (38 - len(str(n)))})")
        lines.append(f"  best_so_far: {it['best_so_far']:.6f}")
        lines.append("")

        lines.append("  [GP Fit]")
        lines += _fmt_mat(it["K_y"], label="K_y = K(X_obs, X_obs) + noise*I")
        lines += _fmt_mat(it["L"],   label="L   = cholesky(K_y)")
        lines += _fmt_vec(it["alpha"], label="alpha = L.T \\ (L \\ y_obs)")
        lines.append("")

        lines.append("  [GP Predict]")
        lines += _fmt_mat(it["k_star"], label="k_star = K(X_remaining, X_obs)")
        lines += _fmt_vec(it["mean_remaining"], label="mean = k_star @ alpha")
        lines += _fmt_vec(it["std_remaining"],  label="std  = sqrt(signal_var - sum(v**2))")
        lines.append("")

        lines.append("  [EI  =  (mean - best - xi)*Phi(z) + std*phi(z)]")
        lines += _fmt_vec(it["z_remaining"],  label="z  = (mean - best - xi) / std")
        lines += _fmt_vec(it["ei_remaining"], label="EI")
        lines.append(f"  -> selected global_idx={it['selected_global_index']}  "
                     f"mean={it['selected_mean']:.6f}  std={it['selected_std']:.6f}  "
                     f"EI={it['selected_ei']:.6f}  actual={it['actual_score']:.6f}")
        lines.append("")

    lines.append("── Final Comparison ────────────────────────────────────────")
    if "zero_shot_score" in _run_log:
        lines.append(f"  zero-shot  : {_run_log['zero_shot_score']:.3f}")
    if "best_v3_score" in _run_log:
        lines.append(f"  v3 (best)  : {_run_log['best_v3_score']:.3f}")
        if "zero_shot_score" in _run_log:
            delta = _run_log["best_v3_score"] - _run_log["zero_shot_score"]
            lines.append(f"  improvement: {delta:+.3f}")
    lines.append("")

    if "best_v3_instruction" in _run_log:
        lines.append("── Best Candidate ──────────────────────────────────────────")
        lines.append(f"  instruction: {_run_log['best_v3_instruction']}")
        for i, demo in enumerate(_run_log.get("best_v3_demos", [])):
            lines.append(f"  demo[{i}]     input: {demo.get('input', '')}")
            lines.append(f"           gold : {demo.get('gold', '')}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "03-eval-harness", "data", "dataset.jsonl")

    dataset = load_dataset(DATA_PATH)
    if not dataset:
        print("dataset.jsonl is empty — populate 03-eval-harness/data/dataset.jsonl first.")
        raise SystemExit(1)

    train_data = split(dataset, "train")
    val_data = split(dataset, "val")

    print(f"Dataset: {len(train_data)} train, {len(val_data)} val examples")
    log_path = os.path.join(
        os.path.dirname(__file__), "artifacts", "logs",
        f"optimizer_v3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    try:
        best_candidate, best_score = optimize_v3(train_data, val_data)
        print(f"\nBest val score (v3): {best_score:.3f}")
        if best_candidate:
            print(f"Best instruction: {best_candidate.get('instruction', '')[:120]}")
    finally:
        write_run_log(log_path)
        print(f"\nRun log written to: {log_path}")
