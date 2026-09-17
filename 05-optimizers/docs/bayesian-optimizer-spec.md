# optimizer_v3.py — Bayesian Optimization Outline

---

## Level 1 — High Level: The Loop

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│       PROPOSE        │────▶│       EVALUATE       │────▶│        SELECT        │
│                      │     │                      │     │                      │
│  Generate the full   │     │  Score candidates    │     │  Return best-seen    │
│  candidate pool      │     │  strategically via   │     │  across all iters    │
│  (instruction×demo)  │     │  the Bayesian loop   │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

> Same 3-stage skeleton as v1/v2. What changes: EVALUATE no longer scores everyone — it uses a surrogate to pick who to score next. SELECT is implicit — "best seen so far" is tracked across iterations.

---

## Level 2 — Mid Level: Processes Inside Each Stage

```
PROPOSE
│
├── Bootstrap Demo Pool
│     Correct examples from train become the demo reservoir
│
├── Generate Instruction Candidates
│     LLM rewrites the seed instruction N ways; seed kept as candidate 0
│
├── Cross-Product → Candidate Pool
│     Each instruction × each random demo subset = one candidate
│     This finite pool is the entire search space for the BO loop
│
└── Embed Pool
      Serialize each candidate to one string, call the embedding API,
      stack results → one row per candidate (N × 1536 matrix)


EVALUATE  ◀── iterative; replaces "score everyone then pick max"
│
├── Warm Start
│     Randomly evaluate n_initial candidates to seed the GP
│     (GP can't predict anything useful with zero observations)
│
└── BO Iteration Loop  [×n_iterations]
      │
      ├── Fit Surrogate
      │     Build GP over observed (embedding, score) pairs
      │     Produces a belief distribution over all candidates' scores
      │
      ├── Acquire Next
      │     Query GP for predicted mean + uncertainty on remaining candidates
      │     Score each with Expected Improvement; pick the highest
      │
      └── Score & Update
            Evaluate chosen candidate on val (the expensive LLM call)
            Append result to observed set; update best-seen if improved


SELECT
│
└── Track & Return Best
      Best-seen is updated every time a new score beats the current max
      No separate selection pass — BO loop implicitly does this per iteration
```

---

## Level 3 — Low Level: Functions Inside Each Process

```
PROPOSE
│
├── Bootstrap Demo Pool
│     zero_shot_program(instruction)  →  build a zero-shot callable
│     bootstrap(program, train)       →  run it; keep only correct examples
│
├── Generate Instruction Candidates
│     propose_instruction_candidates(seed, demo_pool, n)
│         └── llm.generate(INSTRUCTION_REWRITE_PROMPT, temp=0.7)
│             └── parse numbered list from response
│
├── Cross-Product → Candidate Pool
│     propose_candidates_v2(demo_pool, instructions, n_subsets, n_demos)
│         └── random.sample(demo_pool, n_demos)  [for each instruction × subset]
│
└── Embed Pool
      candidate_to_text(candidate)  →  "instruction | input→gold | input→gold …"
      embed_candidate(candidate)    →  candidate_to_text → embedding API → np.array (1536,)
      embed_pool(candidates)        →  embed_candidate() for each → stack → (N × 1536)


EVALUATE
│
├── Warm Start
│     warm_start(candidates, X_all, val, n_initial)
│         └── evaluate_candidate(candidate, val)  [×n_initial, random picks]
│               └── build_program(candidate)  →  program callable
│                   metric(pred, gold)         →  0.0 or 1.0 per example
│                   mean(scores)               →  val accuracy
│
└── BO Iteration Loop
      │
      ├── Fit Surrogate
      │     gp_fit(X_obs, y_obs, length_scale, signal_var, noise_var)
      │         ├── covariance_matrix(X_obs, X_obs, …)
      │         │     └── rbf_kernel(x1, x2, l, σ²)
      │         │           = σ² · exp(−‖x1−x2‖² / 2l²)
      │         ├── K_noisy = K + noise_var · I
      │         ├── L = cholesky(K_noisy)          [numpy.linalg.cholesky]
      │         └── alpha = solve(Lᵀ, solve(L, y)) [two triangular solves]
      │
      ├── Acquire Next
      │     gp_predict(gp_state, X_remaining)
      │         ├── k_star = covariance_matrix(X_remaining, X_obs, …)
      │         ├── mean   = k_star @ alpha
      │         ├── v      = solve(L, k_star.T)
      │         └── std    = sqrt(σ² − sum(v², axis=0))
      │
      │     expected_improvement(mean, std, best_so_far, xi=0.01)
      │         ├── Z   = (mean − best − xi) / max(std, ε)
      │         ├── Φ(Z) = 0.5 · (1 + erf(Z/√2))    [normal_cdf, uses math.erf]
      │         ├── φ(Z) = exp(−Z²/2) / √(2π)        [normal_pdf, uses math.exp]
      │         └── EI  = (mean − best − xi)·Φ(Z) + std·φ(Z)
      │
      │     select_next_by_ei(remaining, X_rem, gp_state, best)
      │         └── argmax(EI)  →  chosen candidate + index
      │
      └── Score & Update
            evaluate_candidate(chosen, val)   →  score
            stack X_new into X_obs            →  grow observed matrix
            append score into y_obs           →  grow score vector
            if score > best_score: update best_candidate, best_score


SELECT
│
└── Track & Return Best
      compare best_score across all warm-start + BO iterations
      return (best_candidate, best_score)
```

---

## Overview

Bayesian optimization over the instruction × demo candidate space. Replaces the random
proposal-and-select from v1/v2 with a Gaussian Process (GP) surrogate + Expected
Improvement (EI) acquisition loop. Each iteration fits the surrogate on all observed
(candidate, val_score) pairs, then uses EI to select the single candidate most worth
evaluating next — trading off exploitation (high predicted score) against exploration
(high uncertainty). This is the MIPROv2-style stretch goal from the README.

Same PROPOSE → EVALUATE → SELECT skeleton as v1/v2. Only EVALUATE and SELECT change:
instead of scoring every candidate, we score strategically.

---

## Imports

```
# Standard library
sys, os, random, datetime, math               — math.erf, math.exp, math.sqrt, math.pi for EI

# Numerical
numpy                                          — all matrix ops (dot, linalg.cholesky, linalg.solve, exp)

# Project modules (reused from v1/v2)
llm_provider.get_llm()                         — provider-agnostic LLM + embedding calls (instruction generation, same env-var setup)
eval: evaluate, metric, load_dataset, split    — scoring harness (unchanged from the eval harness section)
optimizer_v1: zero_shot_program, bootstrap,
              build_program, build_prompt_template — program construction and evaluation helpers
optimizer_v2: propose_instruction_candidates,
              propose_candidates_v2, SEED_INSTRUCTION — candidate pool generation
```

---

## Functions

---

### `candidate_to_text(candidate: dict) -> str`

**Purpose:** Serialize a candidate to a single string for the embedding API.

**Input:**
- `candidate` — dict with keys `"instruction"` (str) and `"demos"` (list of dicts with `"input"` and `"gold"`)

**Output:**
- A single string: instruction followed by each demo as `"input -> gold"`, separated by ` | `

**Transformation:**
Concatenate the instruction and all demo pairs with ` | ` as a separator, preserving
original punctuation, capitalization, and word order. No tokenization or lowercasing —
the embedding model handles text as-is, and stripping formatting loses signal.

---

### `embed_candidate(candidate: dict) -> np.ndarray`

**Purpose:** Produce a single 1536-dim semantic vector for a candidate via the embedding API (via `llm_provider.py`; 1536 dims when using `text-embedding-3-small` — dimension depends on the configured embedding model).

**Shape:** `(1536,)`

**Input:**
- `candidate` — dict with `"instruction"` and `"demos"`

**Output:**
- 1D numpy float array of length 1536

**Transformation:**
Call `candidate_to_text(candidate)` to get the full string, POST it to the Azure
embedding endpoint (same API key and endpoint base as the chat completions calls, different
path: `/openai/deployments/<embedding-model>/embeddings`), parse the `"embedding"` field
from the JSON response, return as `np.array`. The embedding model captures semantic
meaning — two candidates with similar intent but different wording will be close in this
space, which is what the GP needs to generalize across candidates.

---

### `embed_pool(candidates: list[dict]) -> np.ndarray`

**Purpose:** Embed every candidate in the pool upfront, producing the full feature matrix for the GP.

**Shape:** `N × 1536` where N = pool size

**Input:**
- `candidates` — full list of candidate dicts

**Output:**
- 2D numpy float array, one row per candidate, index-aligned with the candidates list

**Transformation:**
Call `embed_candidate()` for each candidate and stack the results with `np.vstack`.
Done once before the BO loop starts — embeddings are fixed for the life of the run
since the candidate pool doesn't change. Index alignment is critical: row `i` must
always correspond to `candidates[i]` so the BO loop can slice by remaining indices.

---

### `rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float, signal_var: float) -> float`

**Purpose:** Compute the similarity between two candidate embeddings.

**Input:**
- `x1`, `x2` — two 1D embedding vectors of shape `(1536,)`
- `length_scale` — controls how quickly correlation decays with distance
- `signal_var` — scales the overall output variance

**Output:** scalar kernel value k(x1, x2)

**Transformation:**
```
k(x1, x2) = signal_var * exp(−‖x1 − x2‖² / (2 * length_scale²))
```
Use `np.dot` to compute the squared distance. Returns `signal_var` when x1 == x2,
decays toward 0 as distance grows. RBF assumes the objective is smooth — appropriate
here since semantically similar candidates (close in embedding space) tend to score
similarly on val.

---

### `covariance_matrix(X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float) -> np.ndarray`

**Purpose:** Build the full pairwise kernel matrix between two sets of embeddings.

**Shape:** `(M, N)`

**Input:**
- `X1` — shape `(M, 1536)`, `X2` — shape `(N, 1536)`
- `length_scale`, `signal_var` — kernel hyperparameters

**Output:** matrix where entry `[i, j] = rbf_kernel(X1[i], X2[j])`

**Transformation:**
Vectorized with numpy broadcasting:
`-0.5 * ((X1[:,None,:] - X2[None,:,:]) ** 2).sum(axis=2) / length_scale**2`
then multiply by `signal_var` and exponentiate. When X1 == X2 this produces the
symmetric square covariance matrix K used in `gp_fit`.

---

### `gp_fit(X_obs: np.ndarray, y_obs: np.ndarray, length_scale: float, signal_var: float, noise_var: float) -> dict`

**Purpose:** Pre-compute everything needed for fast GP predictions given observed (embedding, score) pairs.

**Input:**
- `X_obs` — shape `(M, 1536)`, embeddings of evaluated candidates
- `y_obs` — shape `(M,)`, their val scores
- `length_scale`, `signal_var`, `noise_var` — kernel hyperparameters

**Output:** GP state dict with keys `"X_obs"`, `"L"`, `"alpha"`, `"hyperparams"`

**Transformation:**
1. K = `covariance_matrix(X_obs, X_obs, length_scale, signal_var)`
2. K_noisy = K + `noise_var * np.eye(M)`
3. L = `np.linalg.cholesky(K_noisy)` — stable factorization of the covariance matrix
4. alpha = `np.linalg.solve(L.T, np.linalg.solve(L, y_obs))` — equivalent to K_noisy⁻¹ @ y but numerically stable
5. Return `{"X_obs": X_obs, "L": L, "alpha": alpha, "hyperparams": {...}}`

Cholesky avoids explicit matrix inversion, which is numerically unstable when two
candidates have very similar embeddings (near-singular K).

---

### `gp_predict(gp_state: dict, X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]`

**Purpose:** Compute posterior mean and uncertainty for un-evaluated candidates.

**Input:**
- `gp_state` — output of `gp_fit`
- `X_query` — shape `(K, 1536)`, embeddings of remaining candidates

**Output:** `(mean, std)` — both shape `(K,)`

**Transformation:**
1. k_star = `covariance_matrix(X_query, X_obs, ...)` → shape `(K, M)`
2. mean = `k_star @ alpha`
3. v = `np.linalg.solve(L, k_star.T)` → shape `(M, K)`
4. var = `signal_var - (v**2).sum(axis=0)` — clip to 0 to prevent negative variance from numerical error
5. std = `np.sqrt(var)`

---

### `normal_pdf(z: float) -> float`

**Purpose:** Standard normal probability density at z.

**Transformation:** `exp(−z²/2) / sqrt(2π)` using `math.exp` and `math.pi`.

---

### `normal_cdf(z: float) -> float`

**Purpose:** Standard normal cumulative probability up to z.

**Transformation:** `0.5 * (1 + erf(z / sqrt(2)))` using `math.erf` and `math.sqrt`.

---

### `expected_improvement(mean: np.ndarray, std: np.ndarray, best_so_far: float, xi: float = 0.01) -> np.ndarray`

**Purpose:** Score each remaining candidate by how much it is expected to improve on the current best.

**Input:**
- `mean`, `std` — GP posterior, both shape `(K,)`
- `best_so_far` — highest val score observed so far
- `xi` — exploration tradeoff; 0.01 biases toward exploitation

**Output:** EI scores shape `(K,)` — higher means more worth evaluating next

**Transformation:**
```
Z   = (mean − best_so_far − xi) / max(std, ε)      # ε ~ 1e-9 avoids div-by-zero
EI  = (mean − best_so_far − xi) * Φ(Z) + std * φ(Z)
```
Apply `normal_cdf` and `normal_pdf` element-wise. EI = 0 for any candidate where
std ≈ 0 (GP is already confident it won't beat the best).

---

### `select_next_by_ei(remaining_candidates: list[dict], X_remaining: np.ndarray, gp_state: dict, best_so_far: float) -> tuple[int, dict]`

**Purpose:** The acquisition step — pick exactly one candidate to evaluate next.

**Input:**
- `remaining_candidates`, `X_remaining` — un-evaluated candidates and their embeddings
- `gp_state` — fitted GP from `gp_fit`
- `best_so_far` — current best observed val score

**Output:** `(local_idx, candidate)` — index into remaining list and the chosen dict

**Transformation:**
Call `gp_predict` → `(mean, std)`, then `expected_improvement` → EI array,
then return `np.argmax(EI)` and the corresponding candidate.

---

### `evaluate_candidate(candidate: dict, val: Dataset) -> float`

**Purpose:** Score one candidate on the validation set — the expensive black-box objective.

**Input:**
- `candidate` — dict `{"instruction": str, "demos": list[dict]}`
- `val` — list of `{"input": str, "gold": str}` validation examples

**Output:**
- Val accuracy (float in [0, 1])

**Transformation:**
Wrap the candidate into a runnable program with `build_program(candidate)` (from
optimizer_v1), call it on each val example, compare the prediction to the gold label
with `metric()`, and return the fraction correct. Each call costs one LLM call per val
example — this is why Bayesian optimization exists: to minimize the number of times we
must call this function.

---

### `warm_start(candidates: list[dict], val: Dataset, n_initial: int) -> tuple[list[dict], list[float]]`

**Purpose:** Randomly evaluate a small set of candidates to give the GP its first observations before any surrogate is fitted.

**Input:**
- `candidates` — full candidate pool
- `val` — validation dataset
- `n_initial` — number of candidates to evaluate randomly (typically 3–5)

**Output:**
- `(evaluated_candidates, scores)` — list of candidate dicts and their corresponding val scores

**Transformation:**
Randomly sample `n_initial` candidates from the pool without replacement, call
`evaluate_candidate` on each, and return the results. The GP requires at least a few
data points before its predictions are useful — this solves the cold-start problem.
Without warm start, the GP posterior is dominated by the prior and EI degenerates to
pure exploration (random sampling). Remove the sampled candidates from the pool so they
are not re-evaluated in the BO loop.

---

### `optimize_v3(train: Dataset, val: Dataset, seed_instruction: str = SEED_INSTRUCTION, n_instructions: int = 3, n_demo_subsets: int = 5, n_demos: int = 3, n_initial: int = 4, n_iterations: int = 10) -> tuple[dict, float]`

**Purpose:** Full Bayesian optimization loop — the main entry point of v3.

**Input:**
- `train`, `val` — datasets from the eval harness split
- `seed_instruction` — starting instruction text
- `n_instructions` — how many LLM-rewritten instruction variants to propose
- `n_demo_subsets` — demo subsets sampled per instruction (pool size = n_instructions × n_demo_subsets)
- `n_demos` — demonstrations per candidate
- `n_initial` — warm-start budget (random evaluations before GP is used)
- `n_iterations` — BO iterations after warm start (GP-guided evaluations)

**Output:**
- `(best_candidate, best_val_score)` — the best-found candidate dict and its val accuracy

**Transformation (step by step):**

1. **Zero-shot baseline.** Call `zero_shot_program(seed_instruction)` then `evaluate` on val. Print and record score.

2. **Bootstrap.** Call `bootstrap(zero_shot_prog, train)` to build the demo pool of correctly-predicted training examples. Guard against empty pool.

3. **Generate candidate pool (PROPOSE).** Call `propose_instruction_candidates` (from v2) to get LLM-rewritten instruction variants, then `propose_candidates_v2` to cross-product them with random demo subsets. This is the finite search space BO will intelligently sample from — identical to v2's PROPOSE.

4. **Embed full pool.** Call `embed_pool(all_candidates)` to get `X_all` (shape: pool_size × 1536). Done once before the loop — embeddings are fixed for the run. Keep pool list and `X_all` rows index-aligned throughout.

5. **Warm start.** Call `warm_start(all_candidates, val, n_initial)` to get the first observed (candidate, score) pairs. Remove warm-start candidates from the remaining pool. Initialize `best_candidate` and `best_score` from warm-start results.

6. **Bayesian optimization loop** (repeat `n_iterations` times or until pool is exhausted):
   - a. Fit surrogate: call `fit_surrogate(X_observed, y_observed)` → `(gp, scaler)`
   - b. Get embeddings of remaining un-evaluated candidates (index into `X_all` using remaining pool indices)
   - c. Acquisition: call `select_next_by_ei(remaining_candidates, X_remaining, gp, scaler, best_score)` → `(idx, next_candidate)`
   - d. Evaluate: call `evaluate_candidate(next_candidate, val)` → `score`
   - e. Update: append `(X_next, score)` to observed arrays; remove candidate from remaining pool; update `best_candidate` and `best_score` if improved
   - f. Log: print iteration number, EI-selected candidate summary, predicted mean/std, actual score, whether it improved the best

7. **Comparison.** Run `optimize_v2` (from v2) to get v2 score. Print: zero-shot → v2 → v3 scores and improvements. Assert v3 score ≥ v2 score.

8. Return `(best_candidate, best_score)`.

---

### `write_run_log(path: str) -> None`

**Purpose:** Persist a human-readable record of the full BO run for auditing and analysis.

**Input:**
- `path` — output file path (e.g. `artifacts/logs/optimizer_v3_<timestamp>.txt`)

**Output:** none (writes file to disk)

**Transformation:**
Same structure as v2's `write_run_log`. Sections to include:
- Seed instruction and hyperparameters (`n_instructions`, `n_demo_subsets`, `n_initial`, `n_iterations`)
- Full candidate pool size
- Warm-start results: each randomly-evaluated candidate, its prompt template, its val score
- Per-BO-iteration record: EI-selected candidate, GP predicted mean and std, actual val score, running best
- GP diagnostic: log-marginal-likelihood after each fit (indicates how well the surrogate is calibrated)
- Final best candidate: instruction, demos, val score
- Comparison table: zero-shot / v1 / v2 / v3 scores side by side

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Embedding API, single string (provider-agnostic via `llm_provider.py`) | Captures semantic similarity — synonymous instructions are close in embedding space; TF-IDF only measures lexical overlap and misses this |
| Single string, not instruction + demos separately | Splitting and concatenating doubles dimensionality without giving the GP more signal; high-D hurts RBF kernels |
| RBF kernel from scratch (numpy only) | No sklearn/scipy dependency; all math visible; appropriate since semantically similar candidates tend to score similarly |
| Cholesky solve instead of matrix inversion | Numerically stable when two candidates have similar embeddings (near-singular K) |
| Fixed kernel hyperparameters | Avoids implementing marginal likelihood optimization; reasonable defaults work for small pools |
| normal_cdf via math.erf | Standard library only; `Φ(z) = 0.5*(1 + erf(z/√2))` is exact |
| Finite candidate pool pre-enumerated | Prompt space is discrete; BO over a finite pool avoids optimizing the acquisition function continuously |
| Warm start before GP | GP with zero observations defaults to prior; warm start gives it signal to condition on |
| EI with xi=0.01 | Small xi biases toward exploitation — sensible when eval budget is small |
| Assert v3 ≥ v2 | Enforces that BO is at least as good as random search over the same pool |
