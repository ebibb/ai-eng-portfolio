# Generated Instructions Review

After running `optimizer_v2.py`, paste the 3 instructions it proposed and annotate each.
This forces you to actually read what the optimizer generated — don't skip it.

## Seed instruction

```
Classify the following customer support ticket into exactly one of three categories: billing, technical, or general. Respond with only the category label.
```

## Proposed instructions

### Instruction 1

```
Classify the following customer support ticket into one of these categories: "billing" for issues related to payments, subscriptions, or invoices; "technical" for problems or questions about product functionality, bugs, or errors; "general" for inquiries not related to billing or technical issues. If the ticket could fit into multiple categories, choose the most specific one. Respond with only the category label.
```

**Better or worse than seed? Why?**
I think it is better because it describes each of the categories, which should help the llm discern which category fits each scenario. I don't think that the second to last sentence is necessary because if a ticket's category is unclear it should go to whichever category it is most related to or to general. Instructing it to choose the most specific one seems redundant to me.
*(Your annotation — be specific: "better because it now distinguishes X from Y" or "worse because it's vague about Z")*

---

### Instruction 2

```
Classify the following customer support ticket by selecting one category: "billing" for concerns about charges, pricing, or account payments; "technical" for issues involving system errors, product features, or troubleshooting; "general" for all other inquiries, such as product availability or general questions. When in doubt, prioritize "billing" over "general" and "technical" over "general" if there's overlap. Provide only the category name as your response.
```

**Better or worse than seed? Why?**
Once again I think it is better that it is providing a description of the categories, but I don't think the descriptions are as good as the first instruction. Billing should include subscriptions, and I'm not sure if what the llm categorizes as "product availability" will always fall into the general category. I'm not sure if the second to last sentence will always be accurate. In a scenario where the category could be technical or billing, what would the llm do? I also prefer "Respond with only the category label" over "Provide only the category name as your response".
---

### Instruction 3

```
Assign the following customer support ticket to one category: "billing" if it's about payment, subscription plans, or financial adjustments; "technical" if it involves errors, bugs, or using the product; "general" if it pertains to other non-technical, non-billing questions. Choose the single best category even if the ticket seems relevant to more than one. Respond with just the category label.
```

**Better or worse than seed? Why?**
I think this is better than the seed and perhaps better than the other proposed instructions. It describes the categories, and it is concise. I like "Choose the single best category even if the ticket seems relevant to more than one" more than the other ways this behavior is described in the proposed instructions.
---

## Patterns you noticed

**What kinds of changes did the LLM tend to make?**
The LLM tended to add category definitions, distinguish the difference between general and the other categories, and propose new ways of instructing the LLM to only answer with one category.
The scores were actually shockingly terrible for all of the proposed instructions. With a fixed set of demos the scores were the following:
- Seed = 0.867
- Instruction 1 = 0.333
- Instruction 2 = 0.600
- Instruction 3 = 0.200
I liked the third one the most, but didn't necessarily expect it to perform the best. While concise wording is important in many settings, LLMs appear to benefit from more context. While that context must be accurate, more accurate information about a situation usually leads to better performance. My intuition was wrong. I thought instruction 2 was better than instruction 1 and I was also wrong about that.

The seed instruction with the demos actually performed worse than just running the validation set on the prompt zero-shot. This highlights the importance of testing different sets of demos along with different instructions.

Val score always needs to be used as a benchmark. Without this guidance I probably would've chosen the worst prompt.

## Connection to GEPA
**In 1–2 sentences: how does GEPA automate the annotation step you just did by hand?**
GEPA looks at your evaluators results, data traces, test failures, etc., identifies the problem, critiques the prompt, and uses that feedback to generate a better prompt. Then it continues to loop through this process, generating new prompts, scoring them on different metrics, and using Pareto sampling to choose the dominant performing prompt.
