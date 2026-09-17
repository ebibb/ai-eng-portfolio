"""
import eval.py to use the functions we just implemented
import azure wrapper
parse jsonl dataset

1) train model on training set
2) evaluate model on validation set to get a more realistic estimate of performance
3) tweak model according to results on validation set
4) repeat 1-3 until satisfied, track results
5) pick model that does best on validation set
6) confirm results on test set
"""

import os
import sys
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_provider import get_llm
from cost_estimator import DRY_RUN, print_dry_run_summary
from eval import evaluate, load_dataset, split

load_dotenv()
llm = get_llm()

def classify(input: str) -> str:
    return llm.generate(
        system_prompt="<label definitions> | Label | Definition | Borderline cases |\n"+
                        "|-------|------------|------------------|\n"+
                        "| billing | Tickets about payments, invoices, charges, subscription tiers, refunds, saved payment information, or pricing questions | Account closure that involves a refund — is it billing or general? |\n"+
                        "| technical | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |\n"+
                        "| general | Everything else: policies, business hours, account information, onboarding, non-urgent questions, reminder settings, or plans affecting integrations | User settings or display preferences - is it a simple setting or a technical bug? |</label definitions>\n" + 
                        "Classify the support ticket into exactly one of: billing, technical, general. Reply with only the label.",
        user_prompt=input
    )

dataset = load_dataset("data/dataset.jsonl")
train_set = split(dataset, "train")
val_set = split(dataset, "val")
test_set = split(dataset, "test")

train_score = evaluate(
    classify, train_set,
    log_path=None if DRY_RUN else "logs/train_failures_1.jsonl",
    log_failures_only=True,
)
if DRY_RUN:
    print_dry_run_summary()
else:
    print(f"Train score: {train_score:.3f}")
