"""
Ablation 1: Val score vs. number of demos.

Reconstructed from optimizer_v1.py. Zero-shot score and demo pool are
hard-coded from a one-time bootstrap_once.py run so those expensive calls
are never repeated. Change N_DEMOS and N_CANDIDATES at the top to run
different ablation configurations.

Run:
    python 04-llm-as-judge/ablation.py

Output:
    04-llm-as-judge/artifacts/ablation_logs/ablation_<N_CANDIDATES>_<N_DEMOS>_log.txt
"""
import random
import os
import sys
from typing import Callable
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm_provider import get_llm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "03-eval-harness"))
from eval import evaluate, metric, load_dataset, split  # noqa: E402

load_dotenv()
llm = get_llm()

Dataset = list[dict]
Program = Callable[[str], str]

DEFAULT_INSTRUCTION = (
    "Classify the following customer support ticket into exactly one of three categories: "
    "billing, technical, or general. Respond with only the category label."
)

# ── Ablation settings — change these to run different configurations ───────────
N_CANDIDATES = 50
N_DEMOS = 3

# ── Hard-coded from bootstrap_once.py — do not re-run ─────────────────────────
ZERO_SHOT_SCORE = 0.9333333333333333
DEMO_POOL = [
  {
    "input": "I was charged twice for my subscription this month.",
    "gold": "billing"
  },
  {
    "input": "The app keeps crashing when I try to upload a file larger than 10 MB.",
    "gold": "technical"
  },
  {
    "input": "What are your customer support hours on weekends?",
    "gold": "general"
  },
  {
    "input": "Can I get a refund for last month's payment?",
    "gold": "billing"
  },
  {
    "input": "How do I download a copy of my latest invoice?",
    "gold": "billing"
  },
  {
    "input": "I canceled my account but was charged again this month.",
    "gold": "billing"
  },
  {
    "input": "My credit card was declined when my subscription tried to renew.",
    "gold": "billing"
  },
  {
    "input": "I applied a promo code at checkout but the discount wasn't reflected.",
    "gold": "billing"
  },
  {
    "input": "I need to update the billing address on my account.",
    "gold": "billing"
  },
  {
    "input": "My team was billed for 10 seats but we only have 8 users.",
    "gold": "billing"
  },
  {
    "input": "I received a payment failure notification — what should I do next?",
    "gold": "billing"
  },
  {
    "input": "I want to switch from monthly to annual billing.",
    "gold": "billing"
  },
  {
    "input": "I'm being billed for an add-on I never enabled.",
    "gold": "billing"
  },
  {
    "input": "I want to close my account and get a refund for the unused days.",
    "gold": "billing"
  },
  {
    "input": "Why was I charged a different amount than what the pricing page shows?",
    "gold": "billing"
  },
  {
    "input": "The API is returning a 500 error on all POST requests.",
    "gold": "technical"
  },
  {
    "input": "My Slack integration stopped syncing after your latest update.",
    "gold": "technical"
  },
  {
    "input": "The CSV export always produces an empty file.",
    "gold": "technical"
  },
  {
    "input": "I get a 'connection timeout' every time I try to sync my data.",
    "gold": "technical"
  },
  {
    "input": "The mobile app crashes immediately on launch on my iPhone.",
    "gold": "technical"
  },
  {
    "input": "My webhook endpoint is not receiving any events.",
    "gold": "technical"
  },
  {
    "input": "The search bar returns no results, even for items I know exist.",
    "gold": "technical"
  },
  {
    "input": "Two-factor authentication codes are not being delivered to my phone.",
    "gold": "technical"
  },
  {
    "input": "Notification emails from your platform stopped arriving two days ago.",
    "gold": "technical"
  },
  {
    "input": "The dashboard shows a blank white screen after I log in.",
    "gold": "technical"
  },
  {
    "input": "File uploads freeze at 99% and never complete.",
    "gold": "technical"
  },
  {
    "input": "The billing portal page throws a '403 Forbidden' error when I try to open it.",
    "gold": "technical"
  },
  {
    "input": "How do I change the email address associated with my account?",
    "gold": "general"
  },
  {
    "input": "Where can I find your privacy policy?",
    "gold": "general"
  },
  {
    "input": "I'm new and would like a quick walkthrough of the main features.",
    "gold": "general"
  },
  {
    "input": "Do you offer discounts for nonprofit organizations?",
    "gold": "general"
  },
  {
    "input": "How do I invite a teammate to my workspace?",
    "gold": "general"
  },
  {
    "input": "What file formats does the export feature support?",
    "gold": "general"
  },
  {
    "input": "How long do you retain my data after I cancel my account?",
    "gold": "general"
  },
  {
    "input": "I'd like to request a dark mode option for the app.",
    "gold": "general"
  },
  {
    "input": "Where do I find the onboarding guide for new team members?",
    "gold": "general"
  },
  {
    "input": "I need a VAT receipt for my last payment.",
    "gold": "billing"
  },
  {
    "input": "My free trial ended and I was automatically charged without any warning.",
    "gold": "billing"
  },
  {
    "input": "Can I pay for an annual plan upfront to get a lower rate?",
    "gold": "billing"
  },
  {
    "input": "I need to add a purchase order number to my invoices for our accounting team.",
    "gold": "billing"
  },
  {
    "input": "My subscription renewed at full price even though I was on a promotional rate.",
    "gold": "billing"
  },
  {
    "input": "How do I remove a saved credit card from my account?",
    "gold": "billing"
  },
  {
    "input": "I was promised a 20% discount during the sales call but it was never applied to my bill.",
    "gold": "billing"
  },
  {
    "input": "I accidentally purchased the wrong plan — can I switch and get a partial refund?",
    "gold": "billing"
  },
  {
    "input": "Our organization is being charged sales tax but we are tax-exempt — can you fix this?",
    "gold": "billing"
  },
  {
    "input": "Importing a spreadsheet from Google Sheets fails with a permissions error.",
    "gold": "technical"
  },
  {
    "input": "The calendar view doesn't render correctly on Firefox.",
    "gold": "technical"
  },
  {
    "input": "My API key stopped working immediately after I regenerated it in the settings panel.",
    "gold": "technical"
  },
  {
    "input": "The Zapier zap I configured isn't triggering at all.",
    "gold": "technical"
  },
  {
    "input": "I'm getting a 'rate limit exceeded' error even though I'm well under my usage quota.",
    "gold": "technical"
  },
  {
    "input": "The email template editor freezes whenever I try to insert an image.",
    "gold": "technical"
  },
  {
    "input": "My saved filters in the reporting dashboard reset every time I reload the page.",
    "gold": "technical"
  },
  {
    "input": "Dark mode is broken — text is invisible against the dark background.",
    "gold": "technical"
  },
  {
    "input": "Push notifications on Android stopped working after I reinstalled the app.",
    "gold": "technical"
  },
  {
    "input": "How do I transfer workspace ownership to another user?",
    "gold": "general"
  },
  {
    "input": "What is your SLA guarantee for enterprise customers?",
    "gold": "general"
  },
  {
    "input": "Is your platform HIPAA compliant?",
    "gold": "general"
  },
  {
    "input": "Do you have a native desktop app for Windows?",
    "gold": "general"
  },
  {
    "input": "Is there a public status page where I can check for ongoing outages?",
    "gold": "general"
  },
  {
    "input": "Can I have a dedicated account manager for our enterprise team?",
    "gold": "general"
  },
  {
    "input": "Can I have multiple workspaces under a single login?",
    "gold": "general"
  },
  {
    "input": "I'd like to receive my invoices in a different currency.",
    "gold": "billing"
  },
  {
    "input": "The mobile app doesn't sync offline edits when the connection is restored.",
    "gold": "technical"
  },
  {
    "input": "What is your policy on data portability if I decide to leave?",
    "gold": "general"
  },
  {
    "input": "The API returns stale data even after I clear the cache.",
    "gold": "technical"
  }
]

LOG_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "ablation_logs")


# ── Step 2: Propose ────────────────────────────────────────────────────────────

def propose_candidates(
    demo_pool: list[dict],
    instruction: str,
    n_candidates: int,
    n_demos: int,
) -> list[dict]:
    candidates = []
    for _ in range(n_candidates):
        demos = random.sample(demo_pool, min(n_demos, len(demo_pool)))
        candidates.append({"instruction": instruction, "demos": demos})

    return candidates


# ── Step 2b: Build a runnable program from a candidate ────────────────────────

def build_program(candidate: dict) -> Program:
    def program(text: str) -> str:
        demos_str = ""
        for demo in candidate["demos"]:
            demos_str += f'Ticket: "{demo["input"]}"\n'
            demos_str += f'Category: "{demo["gold"]}"\n'
        prompt = f"{candidate['instruction']}\n---\nExamples:\n{demos_str}---\nTicket: \"{text}\"\nCategory:"
        print(f"  LLM call: {text[:50]!r}")
        response = llm.generate(user_prompt=prompt, temp=0)
        print(f"  LLM response: {response.strip()!r}")
        return response.strip()
    return program


# ── Step 3: Select ─────────────────────────────────────────────────────────────

def select_best(candidates: list[dict], val: Dataset) -> tuple[dict, float, list[tuple[dict, float]]]:
    """Score every candidate; return (best_candidate, best_score, all_scored_sorted)."""
    scored = []
    for candidate in candidates:
        program = build_program(candidate)
        score = evaluate(program, val)
        scored.append((candidate, score))
        print(f"  candidate scored {score:.3f}  demos={[d['input'][:30] for d in candidate['demos']]}")

    scored.sort(key=lambda x: x[1], reverse=True)
    best_candidate, best_score = scored[0]
    return best_candidate, best_score, scored


# ── Output ─────────────────────────────────────────────────────────────────────

def write_log(n_demos, n_candidates, best_candidate, best_score, all_scored, seed):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"ablation_{n_candidates}_{n_demos}_log.txt")

    scores = [s for _, s in all_scored]
    lines = [
        f"Ablation 1 — n_demos={n_demos}, n_candidates={n_candidates}",
        f"Random seed: {seed}",
        f"Zero-shot val score:  {ZERO_SHOT_SCORE:.3f}",
        f"Optimized val score:  {best_score:.3f}",
        f"Improvement:          {best_score - ZERO_SHOT_SCORE:.3f}",
        "",
        f"Score distribution across {n_candidates} candidates:",
        f"  min:  {min(scores):.3f}",
        f"  max:  {max(scores):.3f}",
        f"  mean: {sum(scores) / len(scores):.3f}",
        "",
        "── All candidates (ranked) ──────────────────────────────",
    ]

    for rank, (candidate, score) in enumerate(all_scored, 1):
        lines.append(f"\nRank {rank}  |  score={score:.3f}")
        for i, demo in enumerate(candidate.get("demos", []), 1):
            lines.append(f"  Demo {i}:")
            lines.append(f"    input: {demo['input']}")
            lines.append(f"    gold:  {demo['gold']}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def print_result(n_demos, n_candidates, best_candidate, best_score, all_scored, seed):
    scores = [s for _, s in all_scored]
    print(f"\n── n_demos={n_demos}, n_candidates={n_candidates}  seed={seed} ──")
    print(f"Zero-shot val score:  {ZERO_SHOT_SCORE:.3f}")
    print(f"Optimized val score:  {best_score:.3f}")
    print(f"Improvement:          {best_score - ZERO_SHOT_SCORE:.3f}")
    print(f"Score distribution — min: {min(scores):.3f}  max: {max(scores):.3f}  mean: {sum(scores)/len(scores):.3f}")
    print(f"\nBest candidate demos:")
    for i, demo in enumerate(best_candidate.get("demos", []), 1):
        print(f"  Demo {i}: input={demo['input']!r}  gold={demo['gold']!r}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "03-eval-harness", "data", "dataset.jsonl")
    dataset = load_dataset(DATA_PATH)
    if not dataset:
        print("dataset.jsonl is empty — populate 03-eval-harness/data/dataset.jsonl first.")
        raise SystemExit(1)

    val_data = split(dataset, "val")
    print(f"Val examples: {len(val_data)}")
    print(f"Running: n_demos={N_DEMOS}, n_candidates={N_CANDIDATES}")

    # Set and log random seed for reproducibility
    seed = random.randrange(2**32)
    random.seed(seed)
    print(f"Random seed: {seed}")

    # 1. Propose candidates from the hard-coded demo pool
    candidates = propose_candidates(
        demo_pool=DEMO_POOL,
        instruction=DEFAULT_INSTRUCTION,
        n_candidates=N_CANDIDATES,
        n_demos=N_DEMOS,
    )

    # 2. Score all candidates, select best
    best_candidate, best_score, all_scored = select_best(candidates, val_data)

    # 3. Print and log results
    print_result(N_DEMOS, N_CANDIDATES, best_candidate, best_score, all_scored, seed)
    log_path = write_log(N_DEMOS, N_CANDIDATES, best_candidate, best_score, all_scored, seed)
    print(f"\nLog written to: {log_path}")
