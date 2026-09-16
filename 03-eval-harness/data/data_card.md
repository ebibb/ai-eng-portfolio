# Data Card

Fill this in as you build the dataset. This document travels with the data — anyone reading it should understand what the dataset is and how to use it responsibly.

## Task
Customer support ticket classification into three categories: **billing**, **technical**, and **general**.

This ticket classification system is best-fit for a B2B SaaS company - think project management tools, developer platforms, or CRM software.
The classifier would be most useful for tagging tickets before sending them to different teams to solve.
The test cases address questions ranging from login issues, API access, subscription questions, to legal compliance.

## Label definitions

| Label | Definition | Borderline cases |
|-------|------------|------------------|
| `billing` | Tickets about payments, invoices, charges, subscription tiers, refunds, or pricing questions | Account closure that involves a refund — is it billing or general? |
| `technical` | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |
| `general` | Everything else: policies, business hours, account information, onboarding, non-urgent questions | — |

## Dataset statistics

| Split | Count |
|-------|-------|
| train | 70 |
| val | 15 |
| test | 15 |
| **total** | 100 |

## Known ambiguous cases

List 2–3 examples that could plausibly belong to more than one label, and explain how you resolved them. This is important for reproducibility and for catching metric gaming.

1. {"input": "Can I bulk-import users via a CSV file?", "prediction": "technical", "gold": "general", "score": 0.0} - This details a tool capability not necessarily a technical bug.
2. {"input": "What happens to my integrations if I downgrade my plan?", "prediction": "billing", "gold": "general", "score": 0.0} - This could be either a billing or technical plan - but is put into general because of its ambiguity.
3. {"input": "How do I remove a saved credit card from my account?", "prediction": "general", "gold": "billing", "score": 0.0} - Add to the prompt that billing includes saved financial information.

## Collection method

LLM-generated prompt:
```xml
<purpose>I want to evaluate my model's ability to classify customer support tickets into three categories: billing, technical, and general.</purpose>
 
<answer template>{"input": "...", "gold": "billing|technical|general", "split": "train|val|test"}</answer template>
 
<label definitions> | Label | Definition | Borderline cases |
|-------|------------|------------------|
| `billing` | Tickets about payments, invoices, charges, subscription tiers, refunds, or pricing questions | Account closure that involves a refund — is it billing or general? |
| `technical` | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |
| `general` | Everything else: policies, business hours, account information, onboarding, non-urgent questions | — |</label definitions>
 
<examples>{"input": "I was charged twice for my subscription this month.", "gold": "billing", "split": "train"}
{"input": "The app keeps crashing when I try to upload a file larger than 10 MB.", "gold": "technical", "split": "train"}
{"input": "What are your customer support hours on weekends?", "gold": "general", "split": "train"}
{"input": "My invoice shows an amount I don't recognize.", "gold": "billing", "split": "val"}
{"input": "I cannot log in after resetting my password — the page just hangs.", "gold": "technical", "split": "test"}</examples>
 
<instructions>Generate 95 more labeled examples and append them to 03-eval-harness/data/dataset.jsonl. These examples should represent cases that clearly fit into one "gold" label, as well as some that are more ambiguous and could pass for multiple "gold" labels. "split" should be 70% train / 15% val / 15% test.</instructions>
```
## Known biases or limitations

*(Are certain ticket types over-represented? Is the language limited to English? Any other caveats a future user of this dataset should know?)*
- Tried to provide equal representation of billing/technical/general tickets, ratios are 35/33/32
- Language is limited to English as far as I know
- The training might be different for a more specific use case than this. Right now it seems to best fit a SaaS tool, so it's not trained for a different niche like renting recreational beach equipment.
