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
from azure_llm_wrapper import AzureLLMWrapper
from eval import evaluate, load_dataset, split

load_dotenv()
api_key = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY", "")
model = os.getenv("AZURE_OPENAI_MODEL", "")
version = os.getenv("AZURE_OPENAI_API_VERSION", "")
endpoint = os.getenv("AZURE_APIM_ENDPOINT", "") + "/openai/deployments/" + model + "/chat/completions?api-version=" + version
llm = AzureLLMWrapper(endpoint=endpoint, api_key=api_key)

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

train_score = evaluate(classify, train_set, log_path="logs/train_failures_1.jsonl", log_failures_only=True)
print(f"Train score: {train_score:.3f}")
