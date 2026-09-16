# Friction Log

Write roughly half a page answering: **what was annoying about tuning prompts by hand?**

Prompts to get you started:
- When you had two prompt variants, how did you decide which was better?
- How confident were you in that judgment? What would shake your confidence?
- If you ran the same prompt twice, would you get the same answer?
- What would "good enough" even mean without a number?
- How long did it take to write and compare the three variants? Was that time well spent?

---

*(Write your answer below this line — aim for 150-200 words.)*

After reading the resources provided, I expected the few-shot CoT prompt to have the best results, but that wasn't necessarily the case. The first round of prompting with the train_tickets resulted in zero-shot - 100% accuracy, few-shot - 80% accuracy, and few-shot-cot - 80% accuracy. The first round of prompting with the test_tickets resulted in zero-shot - 90% accuracy, few-shot - 100% accuracy, and few-shot-cot - 100% accuracy. I started out by tuning the zero-shot prompt, which took a few iterations, but still resulted in a simpler prompt than both few-shot and few-shot-cot. Few-shot required the least amount of tweaking, and few-shot-cot was actually the most stubborn of the three. Even with the same prompt changes as few-shot and zero-shot, few-shot-cot was still missing ones that it shouldn't have been. From the resources I read, I concluded that CoT and few-shot would improve responses - even causing the model to change its answer from incorrect to correct, but I think discernment is still required in situations where these techniques could be used. After working with these different styles of prompting for this scenario, I've decided that few-shot was the most effective. One thing I remember from the first document in the resources is that CoT prompts with complex examples can improve the accuracy of complex questions, but perform poorly in simple questions. I think this exercise is a good example of where CoT might've been overkill for the needs of the task. Few-shot seems like the most logical fit for this scenario because it offers extra context without requiring an excessive amount of reasoning from the model. I speculate that requiring chain of thought might have caused the model to "overthink" in scenarios where it wasn't required, skewing the answer from what it should've been.

## Prompt Tuning Log

### Summary Metrics

Train tickets:
- zero-shot: **100%**
- few-shot: **80%**
- few-shot-cot: **80%**

Test tickets (initial):
- zero-shot: **90%**
- few-shot: **100%**
- few-shot-cot: **100%**

Final results:
- all prompts on test tickets: **100%**
- all prompts on train tickets: **100%**

---

### Iteration 1 - few_shot_cot.txt

Before:
Think step by step before giving the final category.

After:
Think step by step before giving the final category. Respond in plain text only - no markdown, no bullet points, no bold or italic formatting.

Reasoning:
The model output used inconsistent markdown formats (for example `**Category:**` and `### ...`), which made parsing and accuracy checks less reliable. The plain-text instruction keeps output structure consistent.

---

### Iteration 2 - zero_shot.txt

Before:
- billing
- technical
- general

After:
- billing    — payment, invoices, charges, subscriptions, refunds
- technical  — bugs, errors, crashes, features not working, integrations
- general    — hours, policies, account info, non-urgent questions

Reasoning:
Accuracy was **90%**.
Mistake:
- Ticket: I tried to update my payment method but the page just spins and never saves.
- Expected: technical
- Output: billing

---

### Iteration 3 - zero_shot.txt

Before:
- billing    — payment, invoices, charges, subscriptions, refunds
- technical  — bugs, errors, crashes, features not working, integrations
- general    — hours, policies, account info, non-urgent questions

After:
- billing    — payment, invoices, charges, subscriptions, refunds
- technical  — bugs, errors, crashes, features not working, integrations, requires technical intervention to fix
- general    — hours, policies, account info, non-urgent questions

Reasoning:
Accuracy was **90%**.
Mistake:
- Ticket: I'm locked out of my account after too many failed login attempts — can someone reset it?
- Expected: technical
- Output: general

---

### Iteration 4 - zero_shot.txt

Before:
- billing    — payment, invoices, charges, subscriptions, refunds
- technical  — bugs, errors, crashes, features not working, integrations, requires technical intervention to fix
- general    — hours, policies, account info, non-urgent questions

After:
- billing    — payment, invoices, charges, subscriptions, refunds
- technical  — bugs, errors, crashes, features not working, integrations, requires IT intervention
- general    — hours, policies, account info, non-urgent questions

Reasoning:
Accuracy was **90%**.
Mistake:
- Ticket: I'm locked out of my account after too many failed login attempts — can someone reset it?
- Expected: technical
- Output: general

---

### Iteration 5 - zero_shot.txt

Before:
- billing    — payment, invoices, charges, subscriptions, refunds
- technical  — bugs, errors, crashes, features not working, integrations, requires IT intervention
- general    — hours, policies, account info, non-urgent questions

After:
- billing    — payment, invoices, charges, subscriptions, refunds
- technical  — bugs, errors, crashes, features not working, integrations, login troubles
- general    — hours, policies, account info, non-urgent questions

Reasoning:
Accuracy was **90%**.
Mistake:
- Ticket: I'm locked out of my account after too many failed login attempts — can someone reset it?
- Expected: technical
- Output: general

---

### Iteration 6 - few_shot_cot.txt

Before:
- billing    — payment, invoices, charges, subscriptions, refunds
- technical  — bugs, errors, crashes, features not working, integrations
- general    — hours, policies, account info, non-urgent questions

After:
- billing    — payment, invoices, charges, subscriptions, refunds
- technical  — bugs, errors, crashes, features not working, integrations, login troubles
- general    — hours, policies, account info, non-urgent questions

Reasoning:
Accuracy was **90%**.
Mistake:
- Ticket: I'm locked out of my account after too many failed login attempts — can someone reset it?
- Expected: technical
- Output reasoning: Being locked out of an account and requesting a reset is related to account information and management rather than payment or technical malfunctions like bugs or crashes. This falls under general assistance.
- Output category: general

---

### Iteration 7 - zero_shot.txt, few_shot.txt, few_shot_cot.txt

Before:
Slightly different for all three prompts.

After:
- billing    — payment, invoices, charges, subscriptions, refunds, promo codes
- technical  — bugs, errors, crashes, features not working, integrations, login troubles
- general    — hours, policies, account info, non-urgent questions

Reasoning:
Even with **100%** accuracy on test_tickets.txt, few_shot and few_shot_cot were still **80%** on train_tickets.