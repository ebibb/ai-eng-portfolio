# Scalar Score vs. Textual Feedback

Fill in this table after running judge.py on several examples.
Copy the completed version into `portfolio/cheat-sheet.md`.

| Dimension | Scalar score | Textual feedback |
|-----------|-------------|------------------|
| **What it tells you** | Whether (and how much) the generated output was right/wrong | *Why* it was wrong — which specific aspect failed |
| **What it hides** | The direction and cause of the failure | May be verbose, inconsistent, or hallucinated across runs |
| **Optimizers that use it** | All classical optimizers (exact-match, F1, BLEU, random search) | Reflective / language-driven optimizers (GEPA, TextGrad) |
| **Example output** | `0.0` | "The generated output was 'billing' but the ticket describes a software crash, which is a technical issue." |
| **Cost per evaluation** | Cheap — string comparison or simple calculation | Expensive — requires an LLM call per example |
| **Signal richness** | Low — only magnitude | High — direction + cause, actionable for prompt editing |

## Why this matters

GEPA's core claim (GEPA paper abstract): language feedback is a "richer learning medium"
than scalar rewards because it tells the optimizer *which part of the prompt to change and how*.
A scalar score of 0.0 says "wrong"; a critique says "the instruction doesn't distinguish billing from
technical — add a clarification about software-related issues."

## Your observations from running judge.py


### Results and Conclusions

- Results:
    - I tested 9 of the same pieces of data for reference-free and reference-based prompting for the judge.
    - 4/9 of the examples had different scoring, and some of the scores that were the same had different critiques.
- Why these discrepancies exist:
    - Reference-free model focuses on absolute correctness and thinks about more possible categories, as it has no specified constraint for the three categories used in this scenario. Reference-based model focuses on agreement, inspecting wether or not predicted and gold match up. This leads me to believe that the reference-based judge would be more effective when trying to see if the solver's output is in alignment with user sentiment.
- Prompt tuning:
    - Potentially add solver prompt (at least some of it).
    - Add categories and descriptions to limit model coming up with new categories that are not in question for the task.
### Explanations and Applications


<div align="center">

| Ex | Predicted | Gold | Ref score | Ref-free score | Delta |
|----|:---------:|:----:|:---------:|:--------------:|:-----:|
| 1 | billing | general | 0.0 | 0.5 | +0.5 |
| 2 | technical | general | 0.5 | 1.0 | +0.5 |
| 3 | general | billing | 0.5 | 0.0 | -0.5 |
| 4 | billing | general | 0.5 | 0.5 | 0 |
| 5 | billing | billing | 1.0 | 1.0 | 0 |
| 6 | technical | technical | 1.0 | 1.0 | 0 |
| 7 | general | general | 1.0 | 0.0 | -1.0 |
| 8 | billing | billing | 1.0 | 1.0 | 0 |
| 9 | billing | billing | 1.0 | 1.0 | 0 |

</div>

- Example 1: (predicted=billing, gold=general) Reference-free judge scores 0.5 and suggests a more precise category like "nonprofit policies" or "discounts". This can be fixed by including the set categories the solver LLM has access to. It would be easier for the judge to grade the solver's decision if it also had the prompt the solver was given. Reference-based judge scores 0.0.

- Example 2: (predicted=technical, gold=general) Reference-free judge sees this as a good fit and scores 1.0. Reference-based judge knows the actual correct answer and scores 0.5 because it sees it as partially correct.

- Example 3: (predicted=general, gold=billing) Reference-free judge scores 0.0 and suggests a more precise classification like "account settings" or "payment methods." Reference-based judge scores 0.5. Same issue as example 1, needs the solver prompt or at least constrained categories for context.

- Example 7: (predicted=general, gold=general) Reference-free judges scores 0.0 and suggests a more precise classification like "customer service" or "business hours". Reference-based judge scores 1.0 as predicted matches gold. Same issue as example 1.

- Application: 
    - How does this apply to a production judge system?
        - Many real-world judges use set criteria to judge generated content, but don't have a "gold" answer for their judgement. 
    - How can we optimize the judges?
        - Compares human decisions on wether the content is acceptable or not, use that to optimize judge prompts.
        - We want **agreement** between human and model, not the model's opinion based on its own training context.