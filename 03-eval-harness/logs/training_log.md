- First run of classifier_eval.py on train data set resulted in train score: 0.914
- Classify function:
```python
def classify(input: str) -> str:
    return llm.generate(
        system_prompt="<label definitions> | Label | Definition | Borderline cases |\n"+
                        "|-------|------------|------------------|\n"+
                        "| `billing` | Tickets about payments, invoices, charges, subscription tiers, refunds, or pricing questions | Account closure that involves a refund — is it billing or general? |\n"+
                        "| `technical` | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |\n"+
                        "| `general` | Everything else: policies, business hours, account information, onboarding, non-urgent questions | — |</label definitions>\n" + 
                        "Classify the support ticket into exactly one of: billing, technical, general. Reply with only the label.",
        user_prompt=input
    )
```
- Incorrect:
    - {"input": "I get a 'connection timeout' every time I try to sync my data.", "prediction": "`technical`", "gold": "technical", "score": 0.0}
    - {"input": "Notification emails from your platform stopped arriving two days ago.", "prediction": "`technical`", "gold": "technical", "score": 0.0}
    - {"input": "The billing portal page throws a '403 Forbidden' error when I try to open it.", "prediction": "`technical`", "gold": "technical", "score": 0.0}
    - {"input": "How many users can I add under my current plan?", "prediction": "billing", "gold": "general", "score": 0.0}
    - {"input": "I'd like to request a dark mode option for the app.", "prediction": "technical", "gold": "general", "score": 0.0}
    - {"input": "Is there a way to set up automatic reminders for my team inside the platform?", "prediction": "technical", "gold": "general", "score": 0.0}
- Fix 1: removed backticks
```python
def classify(input: str) -> str:
    return llm.generate(
        system_prompt="<label definitions> | Label | Definition | Borderline cases |\n"+
                        "|-------|------------|------------------|\n"+
                        "| billing | Tickets about payments, invoices, charges, subscription tiers, refunds, or pricing questions | Account closure that involves a refund — is it billing or general? |\n"+
                        "| technical | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |\n"+
                        "| general | Everything else: policies, business hours, account information, onboarding, non-urgent questions | — |</label definitions>\n" + 
                        "Classify the support ticket into exactly one of: billing, technical, general. Reply with only the label.",
        user_prompt=input
    )
```
- Second run of classifier_eval.py on train data set resulted in train score: 0.971
- Incorrect: 
{"input": "Is there a way to set up automatic reminders for my team inside the platform?", "prediction": "technical", "gold": "general", "score": 0.0}
{"input": "How do I remove a saved credit card from my account?", "prediction": "general", "gold": "billing", "score": 0.0}
- Fix 2: 
```python
def classify(input: str) -> str:
    return llm.generate(
        system_prompt="<label definitions> | Label | Definition | Borderline cases |\n"+
                        "|-------|------------|------------------|\n"+
                        "| billing | Tickets about payments, invoices, charges, subscription tiers, refunds, or pricing questions | Account closure that involves a refund — is it billing or general? |\n"+
                        "| technical | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |\n"+
                        "| general | Everything else: policies, business hours, account information, onboarding, non-urgent questions | User settings or display preferences - is it a simple setting or a technical bug? |</label definitions>\n" + 
                        "Classify the support ticket into exactly one of: billing, technical, general. Reply with only the label.",
        user_prompt=input
    )
```
- Third run of classifier_eval.py on train data set resulted in train score: 0.943
- Incorrect: 
{"input": "Do you offer discounts for nonprofit organizations?", "prediction": "billing", "gold": "general", "score": 0.0}
{"input": "Is there a way to set up automatic reminders for my team inside the platform?", "prediction": "technical", "gold": "general", "score": 0.0}
{"input": "How do I remove a saved credit card from my account?", "prediction": "general", "gold": "billing", "score": 0.0}
{"input": "What happens to my integrations if I downgrade my plan?", "prediction": "billing", "gold": "general", "score": 0.0}
- Fix 3: undo Fix 2, try Fix 2 on val set
```python
def classify(input: str) -> str:
    return llm.generate(
        system_prompt="<label definitions> | Label | Definition | Borderline cases |\n"+
                        "|-------|------------|------------------|\n"+
                        "| billing | Tickets about payments, invoices, charges, subscription tiers, refunds, or pricing questions | Account closure that involves a refund — is it billing or general? |\n"+
                        "| technical | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |\n"+
                        "| general | Everything else: policies, business hours, account information, onboarding, non-urgent questions | — |</label definitions>\n" + 
                        "Classify the support ticket into exactly one of: billing, technical, general. Reply with only the label.",
        user_prompt=input
    )
```
- Fourth run of classifier_eval.py on val data set resulted in val score: 0.867
- Incorrect: 
{"input": "What is the difference between the Basic and Pro plans?", "prediction": "billing", "gold": "general", "score": 0.0}
{"input": "How do I set permissions so certain users can only view and not edit content?", "prediction": "technical", "gold": "general", "score": 0.0}
- Fix 4: Restore fix 2, the second failure should be fixed by this
```python
def classify(input: str) -> str:
    return llm.generate(
        system_prompt="<label definitions> | Label | Definition | Borderline cases |\n"+
                        "|-------|------------|------------------|\n"+
                        "| billing | Tickets about payments, invoices, charges, subscription tiers, refunds, or pricing questions | Account closure that involves a refund — is it billing or general? |\n"+
                        "| technical | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |\n"+
                        "| general | Everything else: policies, business hours, account information, onboarding, non-urgent questions | User settings or display preferences - is it a simple setting or a technical bug? |</label definitions>\n" + 
                        "Classify the support ticket into exactly one of: billing, technical, general. Reply with only the label.",
        user_prompt=input
    )
```
- Fifth run of classifier_eval.py on val data set resulted in val score: 1.000
- Sixth run of classifier_eval.py on train data set resulted in train score: 0.943
- Incorrect: 
{"input": "Do you offer discounts for nonprofit organizations?", "prediction": "billing", "gold": "general", "score": 0.0}
{"input": "Is there a way to set up automatic reminders for my team inside the platform?", "prediction": "technical", "gold": "general", "score": 0.0}
{"input": "How do I remove a saved credit card from my account?", "prediction": "general", "gold": "billing", "score": 0.0}
{"input": "What happens to my integrations if I downgrade my plan?", "prediction": "billing", "gold": "general", "score": 0.0}
- Fix 6:
```python
def classify(input: str) -> str:
    return llm.generate(
        system_prompt="<label definitions> | Label | Definition | Borderline cases |\n"+
                        "|-------|------------|------------------|\n"+
                        "| billing | Tickets about payments, invoices, charges, subscription tiers, refunds, saved financial information, or pricing questions | Account closure that involves a refund — is it billing or general? |\n"+
                        "| technical | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |\n"+
                        "| general | Everything else: policies, business hours, account information, onboarding, non-urgent questions, reminder settings, or multi-genre issues | User settings or display preferences - is it a simple setting or a technical bug? |</label definitions>\n" + 
                        "Classify the support ticket into exactly one of: billing, technical, general. Reply with only the label.",
        user_prompt=input
    )
```
- Seventh run of classifier_eval.py on train data set resulted in train score: 0.971
- Incorrect: 
{"input": "How do I remove a saved credit card from my account?", "prediction": "general", "gold": "billing", "score": 0.0}
{"input": "What happens to my integrations if I downgrade my plan?", "prediction": "billing", "gold": "general", "score": 0.0}
- Fix 7:
```python
def classify(input: str) -> str:
    return llm.generate(
        system_prompt="<label definitions> | Label | Definition | Borderline cases |\n"+
                        "|-------|------------|------------------|\n"+
                        "| billing | Tickets about payments, invoices, charges, subscription tiers, refunds, saved payment information, or pricing questions | Account closure that involves a refund — is it billing or general? |\n"+
                        "| technical | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |\n"+
                        "| general | Everything else: policies, business hours, account information, onboarding, non-urgent questions, reminder settings, or multi-genre issues (like technical and billing combined) | User settings or display preferences - is it a simple setting or a technical bug? |</label definitions>\n" + 
                        "Classify the support ticket into exactly one of: billing, technical, general. Reply with only the label.",
        user_prompt=input
    )
```
- Eighth run of classifier_eval.py on train data set resulted in train score: 0.986
- Incorrect: 
{"input": "What happens to my integrations if I downgrade my plan?", "prediction": "billing", "gold": "general", "score": 0.0}
- Fix 8:
```python
def classify(input: str) -> str:
    return llm.generate(
        system_prompt="<label definitions> | Label | Definition | Borderline cases |\n"+
                        "|-------|------------|------------------|\n"+
                        "| billing | Tickets about payments, invoices, charges, subscription tiers, refunds, saved payment information, or pricing questions | Account closure that involves a refund — is it billing or general? |\n"+
                        "| technical | Tickets about software bugs, crashes, errors, features not working, API issues, or integration problems | A feature request — is it technical or general? |\n"+
                        "| general | Everything else: policies, business hours, account information, onboarding, non-urgent questions, or reminder settings | User settings or display preferences - is it a simple setting or a technical bug? / Technical and billing questions - can it be answered by one team or both (general)? |</label definitions>\n" + 
                        "Classify the support ticket into exactly one of: billing, technical, general. Reply with only the label.",
        user_prompt=input
    )
```
- Ninth run of classifier_eval.py on train data set resulted in train score: 0.986
- Incorrect: 
{"input": "What happens to my integrations if I downgrade my plan?", "prediction": "billing", "gold": "general", "score": 0.0}
- Fix 9:
```python
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
```
- Tenth run of classifier_eval.py on train data set resulted in train score: 0.986
- Incorrect: 
{"input": "What happens to my integrations if I downgrade my plan?", "prediction": "billing", "gold": "general", "score": 0.0}
- Fix 10:
```python
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
```
- Eleventh run of classifier_eval.py on val data set resulted in val score: 1.000
- Twelth run of classifier_eval.py on test data set resulted in test score: 0.933
    - I'm happy with these scores - only missing one out of the training set, as well as one out of the test set. 
    - I think if we wanted fixes for these
        - The failing training case would just have to be included in few shot
        - The failing test case could be fixed by specifying that general is more for questions about capabilities and technical is more about problems
    - Otherwise, I think this is a good output
    