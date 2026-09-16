# Optimizer

## Concept
- Optimizing is a more methodical way to try out different prompts, without needing constant human oversight. Typically prompt engineering involves hand tuning a prompt, but optimizing automates that process!
- Optimizers can't handwrite few-shot examples to improve your prompt. Instead, it bootstraps them. It runs the solver on the training set, keeps the examples it gets right, and samples from that correct set to ***propose*** few-shot examples that can be appended to your prompt. Then it scores each of these candidates on the validation set to ***evaluate*** the performance. From those scores it ***selects*** the highest performing set to be the new prompt.


<div align="center">
<div style="display:inline-block; border:2px solid #1a4971; border-radius:8px; padding:20px 28px;">
<table style="border-collapse:collapse; font-family:monospace; font-size:12px;">
  <tr>
    <td valign="top">
      <div style="width:155px; border:2px solid #1a4971; border-radius:8px; overflow:hidden;">
        <div style="background:#4a90d9; padding:8px 12px;">
          <span style="color:#fff; font-weight:700; font-size:13px;">PROPOSE:</span>
        </div>
        <div style="background:#b8d9f0; padding:10px 12px; color:#1a4971; line-height:1.6; height:72px;">
          Generate n candidates (instruction + demo subsets)
        </div>
      </div>
    </td>
    <td valign="middle" style="padding:0 4px; font-size:30px; color:#b8d9f0; width:40px; text-align:center;">&#8594;</td>
    <td valign="top">
      <div style="width:155px; border:2px solid #1a4971; border-radius:8px; overflow:hidden;">
        <div style="background:#4a90d9; padding:8px 12px;">
          <span style="color:#fff; font-weight:700; font-size:13px;">EVALUATE:</span>
        </div>
        <div style="background:#b8d9f0; padding:10px 12px; color:#1a4971; line-height:1.6; height:72px;">
          Score each candidate on the VAL set using eval.py
        </div>
      </div>
    </td>
    <td valign="middle" style="padding:0 4px; font-size:30px; color:#b8d9f0; width:40px; text-align:center;">&#8594;</td>
    <td valign="top">
      <div style="width:155px; border:2px solid #1a4971; border-radius:8px; overflow:hidden;">
        <div style="background:#4a90d9; padding:8px 12px;">
          <span style="color:#fff; font-weight:700; font-size:13px;">SELECT:</span>
        </div>
        <div style="background:#b8d9f0; padding:10px 12px; color:#1a4971; line-height:1.6; height:72px;">
          Keep the best-scoring candidate
        </div>
      </div>
    </td>
  </tr>
  <tr>
    <td colspan="5" style="padding:0;">
      <svg viewBox="0 0 561 36" width="561" height="36">
        <polygon points="77.5,0 71,11 84,11" fill="#b8d9f0"/>
        <line x1="77.5" y1="9" x2="77.5" y2="28" stroke="#b8d9f0" stroke-width="2"/>
        <line x1="77.5" y1="28" x2="542" y2="28" stroke="#b8d9f0" stroke-width="2"/>
        <line x1="542" y1="0" x2="542" y2="28" stroke="#b8d9f0" stroke-width="2"/>
        <text x="315" y="24" text-anchor="middle" font-family="monospace" font-size="12" font-style="italic" fill="#b8d9f0">iterate</text>
      </svg>
    </td>
  </tr>
</table>
<p style="font-size:12px; color:#888; margin-top:10px;"><strong>Diagram 1.</strong> Optimizer Loop</p>
</div>
</div>
<br><br>

- This version of optimizer uses scalar feedback for improvement. It simply scores consistency of the model's prediction and the gold answer. Future development would involve switching to textual feedback. 
Textual feedback can be acquired by using a judge llm to score the solver llm's answer. Textual feedback is more expensive to acquire, as it requires an llm call for each case, but provides a richer medium for optimization.
- The goal of optimizing, either with scalar or textual feedback is to improve performance and reduce cost. 
- Cost for optimizing can be broken down into two parts: upfront cost and long term reduction in cost.
  - **Upfront cost**: Especially when employing textual feedback, the sheer number of llm calls guarantess an expensive process, especially as feedback is better when coming from a more advanced model.
  - **Cost reduction**: We can optimize a smaller, less advanced model. By optimizing our prompt on a smaller model, we methodically find prompt 'tricks' that a particular model responds to. This means we can get higher performance from a cheaper model! Improving performance and reducing cost long term is a win-win.
- The downside of prompt optimization is that it is *non-transferrable*. This means that if we optimize a prompt on a smaller model, we cannot use the same prompt on a larger model and expect the same level of performance!

## Code

- **Imports & API Setup**
  - Custom data structures
    - <code style="font-size:13px;">Program = Callable[[str], str]</code>
    - <code style="font-size:13px;">Dataset = list[dict]</code>
  - <code style="font-size:13px;">llm = AzureLLMWrapper(endpoint, api_key)</code> - configured from env vars (<code style="font-size:13px;">AZURE_APIM_ENDPOINT</code>, <code style="font-size:13px;">AZURE_APIM_SUBSCRIPTION_KEY</code>, <code style="font-size:13px;">AZURE_OPENAI_MODEL</code>, <code style="font-size:13px;">AZURE_OPENAI_API_VERSION</code>)
  - <code style="font-size:13px;">DEFAULT_INSTRUCTION</code> - the baseline classification prompt used when no instruction is specified
---
- **Functions**
  - **0) Baseline**: <code style="font-size:13px;">zero_shot_program()</code>
    - Transformation: Take an instruction string and return a callable program that applies that instruction with no examples (zero-shot).
    - Input: <code style="font-size:13px;">instruction (str)</code> <span style="font-size:11px; color:#888; font-style:italic;">— e.g. "Classify this ticket..."</span>
    - Output: <code style="font-size:13px;">program (Program)</code>
      - Transformation: Combine instruction & ticket into prompt, call llm, return stripped response
      - Input: <code style="font-size:13px;">text (str)</code> <span style="font-size:11px; color:#888; font-style:italic;">— ticket</span>
      - Output: <code style="font-size:13px;">llm_response (str)</code> <span style="font-size:11px; color:#888; font-style:italic;">— category</span>
  - **1) Bootstrap**: <code style="font-size:13px;">bootstrap()</code>
    - Transformation: Call program on each item in training data, return correct ones.
    - Input: <code style="font-size:13px;">program (Program)</code>, <code style="font-size:13px;">train (Dataset)</code>
    - Output: <code style="font-size:13px;">correct_examples (list[dict])</code>
  - **2a) Propose**: <code style="font-size:13px;">propose_candidates()</code>
    - Transformation: Sample *n_candidates* random subsets of *n_demos* demos from demo pool
    - Input:
      - <code style="font-size:13px;">demo_pool (list[dict])</code> <span style="font-size:11px; color:#888; font-style:italic;">— bootstrap output, correct examples from train</span>
      - <code style="font-size:13px;">instruction (str)</code> <span style="font-size:11px; color:#888; font-style:italic;">— same as zero_shot input</span>
      - <code style="font-size:13px;">n_candidates (int)</code> <span style="font-size:11px; color:#888; font-style:italic;">— (~1-50), num candidates generated</span>
      - <code style="font-size:13px;">n_demos (int)</code> <span style="font-size:11px; color:#888; font-style:italic;">— (~0-3), num examples as few-shot</span>
    - Output: <code style="font-size:13px;">candidates (list[dict])</code> <span style="font-size:11px; color:#888; font-style:italic;">— each candidate is <code style="font-size:11px;">{"instruction": str, "demos": list[dict]}</code></span>
  <br><br>
<div align="center">
<div style="display:inline-block; border:2px solid #1a4971; border-radius:8px; padding:20px 28px;">
<table style="border-collapse:collapse; font-family:monospace; font-size:12px; margin:10px auto;">
  <tr>
    <td align="center" style="padding-bottom:10px;">
      <span style="font-size:15px; font-weight:700; color:#fff; background:#1a4971; padding:5px 16px; border-radius:6px;">dimensions:</span>
    </td>
    <td></td>
    <td align="center" style="padding-bottom:10px;">
      <span style="font-size:15px; font-weight:700; color:#fff; background:#1a4971; padding:5px 16px; border-radius:6px;">candidates:</span>
    </td>
    <td></td>
    <td align="center" style="padding-bottom:10px;">
      <span style="font-size:15px; font-weight:700; color:#fff; background:#1a4971; padding:5px 16px; border-radius:6px;">demo:</span>
    </td>
  </tr>
  <tr>
    <td valign="middle">
      <div style="display:flex; align-items:center; gap:6px;">
        <div style="writing-mode:vertical-lr; transform:rotate(180deg); font-size:12px; color:#fff; background:#1a4971; padding:4px 6px; border-radius:4px; font-weight:600; white-space:nowrap;">&#8592; n_candidates &#8594;</div>
        <table style="border-collapse:collapse;">
          <tr>
            <td style="border:none; width:62px;"></td>
            <td colspan="3" align="center" style="border:none; padding-bottom:5px;">
              <span style="font-size:12px; color:#fff; background:#1a4971; padding:2px 10px; border-radius:4px; font-weight:600;">&#8592; n_demos &#8594;</span>
            </td>
          </tr>
          <tr>
            <td style="border:2px solid #1a4971; background:#4a90d9; width:62px; height:28px; text-align:center; font-size:10px; color:#fff; font-weight:600;">instruction</td>
            <td style="border:2px solid #1a4971; background:#b8d9f0; width:28px; height:28px;"></td>
            <td style="border:2px solid #1a4971; background:#b8d9f0; width:28px; height:28px;"></td>
            <td style="border:2px solid #1a4971; background:#b8d9f0; width:28px; height:28px;"></td>
          </tr>
          <tr>
            <td style="border:2px solid #1a4971; background:#4a90d9; height:28px; text-align:center; font-size:10px; color:#fff; font-weight:600;">instruction</td>
            <td style="border:2px solid #1a4971; background:#b8d9f0;"></td>
            <td style="border:2px solid #1a4971; background:#b8d9f0;"></td>
            <td style="border:2px solid #1a4971; background:#b8d9f0;"></td>
          </tr>
          <tr>
            <td style="border:2px solid #1a4971; background:#4a90d9; height:28px; text-align:center; font-size:10px; color:#fff; font-weight:600;">instruction</td>
            <td style="border:2px solid #1a4971; background:#b8d9f0;"></td>
            <td style="border:2px solid #1a4971; background:#b8d9f0;"></td>
            <td style="border:2px solid #1a4971; background:#b8d9f0;"></td>
          </tr>
          <tr>
            <td style="border:2px solid #1a4971; background:#4a90d9; height:28px; text-align:center; font-size:10px; color:#fff; font-weight:600;">instruction</td>
            <td style="border:2px solid #1a4971; background:#b8d9f0;"></td>
            <td style="border:2px solid #1a4971; background:#b8d9f0;"></td>
            <td style="border:2px solid #1a4971; background:#b8d9f0;"></td>
          </tr>
        </table>
      </div>
    </td>
    <td valign="middle" align="center" style="padding:0 16px; font-size:30px; color:#b8d9f0;">&#8594;</td>
    <td valign="middle">
      <pre style="border:2px solid #1a4971; padding:10px 14px; background:#e8f4fb; margin:0; line-height:1.7; color:#1a4971;">[{"instruction": str, "demos": list[dict]},
 {"instruction": str, "demos": list[dict]},
 {"instruction": str, "demos": list[dict]},
 {"instruction": str, "demos": list[dict]}]</pre>
    </td>
    <td valign="middle" align="center" style="padding:0 16px; font-size:30px; color:#b8d9f0;">&#8594;</td>
    <td valign="middle">
      <pre style="border:2px solid #1a4971; padding:10px 14px; background:#b8d9f0; margin:0; line-height:1.6; color:#1a4971;">{"input": str,
 "gold": str}</pre>
    </td>
  </tr>
</table>
<p style="font-size:12px; color:#888; margin-top:10px;"><strong>Diagram 2.</strong> Candidates Data Structure</p>
</div>
</div>
<br><br>

  - **2b) Build Program from Candidate**: <code style="font-size:13px;">build_program()</code> <span style="font-size:11px; color:#888; font-style:italic;">— helper for select_best</span>
    - Transformation: takes instruction + demos from candidate, returns a program that applies them (few-shot)
    - Input: <code style="font-size:13px;">candidate (dict)</code> <span style="font-size:11px; color:#888; font-style:italic;">— one candidate, not all</span>
    - Output: <code style="font-size:13px;">program (Program)</code>
      - Transformation: create few-shot prompt, call llm with it, return stripped response
      - Input: <code style="font-size:13px;">text (str)</code> <span style="font-size:11px; color:#888; font-style:italic;">— ticket</span> <span style="font-size:11px; color:#888; font-style:italic;">· prompt = instruction + demos + ticket</span>
      - Output: <code style="font-size:13px;">response (str)</code> <span style="font-size:11px; color:#888; font-style:italic;">— category</span>
  - **3) Select**: <code style="font-size:13px;">select_best()</code>
    - Transformation: call build_program on each candidate, call evaluate on val keep highest score + candidate
    - Input: <code style="font-size:13px;">candidates (list[dict])</code> <span style="font-size:11px; color:#888; font-style:italic;">— from propose_candidates</span>, <code style="font-size:13px;">val (Dataset)</code>
    - Output: <code style="font-size:13px;">(best_candidate, best_score) (tuple[dict, float]))</code>
  - **4) Optimize**: <code style="font-size:13px;">optimize()</code>
    - Transformation: 
      - Create zero-shot program, evaluate on val, print score
      - Create demo pool by calling bootstrap with zero-shot program and train
      - Create candidates by calling propose_candidates with demo_pool, instruction, n_candidates, and n_demos
      - Select best candidate and best score on val
      - Print best score and improvement over zero-shot
    - Input: <code style="font-size:13px;">train (Dataset)</code>, <code style="font-size:13px;">val (Dataset)</code>, <code style="font-size:13px;">n_candidates (int)</code> <span style="font-size:11px; color:#888; font-style:italic;">— (~1-50)</span>, <code style="font-size:13px;">n_demos (int)</code> <span style="font-size:11px; color:#888; font-style:italic;">— (~0-3)</span>, <code style="font-size:13px;">instruction (str)</code>
    - Output: <code style="font-size:13px;">best_candidate, best_val_score (tuple[dict, float])</code>
---
- **Main**
  - Note that build_program() and evaluate() listed under EVALUATE: are apart of the select_best() function and are contained within it in the code. Hence, Evaluate and Select stages happen within one function call.
<div align="center">
<div style="display:inline-block; border:2px solid #1a4971; border-radius:8px; padding:20px 28px;">
<table style="border-collapse:collapse; font-family:monospace; font-size:12px;">
  <tr>
    <td valign="top">
      <div style="width:160px; border:2px solid #1a4971; border-radius:8px; overflow:hidden;">
        <div style="background:#4a90d9; padding:8px 12px;">
          <span style="color:#fff; font-weight:700; font-size:13px;">START:</span>
        </div>
        <div style="background:#b8d9f0; padding:10px 12px; line-height:1.8; height:108px;">
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">AzureLLMWrapper()</code><br>
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">load_dataset()</code><br>
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">split()</code>
        </div>
      </div>
    </td>
    <td valign="middle" style="padding:0 4px; font-size:30px; color:#b8d9f0; width:40px; text-align:center;">&#8594;</td>
    <td valign="top">
      <div style="width:160px; border:2px solid #1a4971; border-radius:8px; overflow:hidden;">
        <div style="background:#4a90d9; padding:8px 12px;">
          <span style="color:#fff; font-weight:700; font-size:13px;">PROPOSE:</span>
        </div>
        <div style="background:#b8d9f0; padding:10px 12px; line-height:1.8; height:108px;">
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">zero_shot_program()</code><br>
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">evaluate()</code><br>
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">bootstrap()</code><br>
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block; margin-left:12px;">metric()</code><br>
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">propose_candidates()</code>
        </div>
      </div>
    </td>
    <td valign="middle" style="padding:0 4px; font-size:30px; color:#b8d9f0; width:40px; text-align:center;">&#8594;</td>
    <td valign="top">
      <div style="width:160px; border:2px solid #1a4971; border-radius:8px; overflow:hidden;">
        <div style="background:#4a90d9; padding:8px 12px;">
          <span style="color:#fff; font-weight:700; font-size:13px;">EVALUATE:</span>
        </div>
        <div style="background:#b8d9f0; padding:10px 12px; line-height:1.8; height:108px;">
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">build_program()</code><br>
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">evaluate()</code>
        </div>
      </div>
    </td>
    <td valign="middle" style="padding:0 4px; font-size:30px; color:#b8d9f0; width:40px; text-align:center;">&#8594;</td>
    <td valign="top">
      <div style="width:160px; border:2px solid #1a4971; border-radius:8px; overflow:hidden;">
        <div style="background:#4a90d9; padding:8px 12px;">
          <span style="color:#fff; font-weight:700; font-size:13px;">SELECT:</span>
        </div>
        <div style="background:#b8d9f0; padding:10px 12px; line-height:1.8; height:108px;">
          <code style="font-size:10px; background:#1a4971; color:#fff; padding:2px 6px; border-radius:3px; display:inline-block;">select_best()</code>
        </div>
      </div>
    </td>
  </tr>
  <tr>
    <td colspan="7" style="padding:0;">
      <svg viewBox="0 0 784 36" width="784" height="36">
        <polygon points="325,0 318,11 332,11" fill="#b8d9f0"/>
        <line x1="325" y1="9" x2="325" y2="28" stroke="#b8d9f0" stroke-width="2"/>
        <line x1="325" y1="28" x2="782" y2="28" stroke="#b8d9f0" stroke-width="2"/>
        <line x1="782" y1="0" x2="782" y2="28" stroke="#b8d9f0" stroke-width="2"/>
        <text x="555" y="24" text-anchor="middle" font-family="monospace" font-size="12" font-style="italic" fill="#b8d9f0">iterate</text>
      </svg>
    </td>
  </tr>
</table>
<p style="font-size:12px; color:#888; margin-top:10px;"><strong>Diagram 3.</strong> Function Flow</p>
</div>
</div>

## Data Formats

### <span style="background:#1a4971; color:#fff; padding:3px 12px; border-radius:4px; font-weight:600;">zero_shot_program()</span>

<span style="font-size:13px; color:#1a4971; font-weight:600;">Input</span> — instruction string

```python
instruction = "Classify the following customer support ticket into exactly one of three categories: billing, technical, or general. Respond with only the category label."
```

<span style="font-size:13px; color:#1a4971; font-weight:600;">Prompt sent to LLM</span>

```python
prompt = """\
Classify the following customer support ticket into exactly one of three categories: billing, technical, or general. Respond with only the category label.

Ticket: "I was charged twice this month."
Category:"""
```

---

### <span style="background:#1a4971; color:#fff; padding:3px 12px; border-radius:4px; font-weight:600;">bootstrap()</span>

<span style="font-size:13px; color:#1a4971; font-weight:600;">Input</span> — train dataset (with `split` field)

```python
train = [
    {"input": "I was charged twice this month.",   "gold": "billing",   "split": "train"},
    {"input": "The app crashes on file upload.",    "gold": "technical", "split": "train"},
    {"input": "What are your business hours?",      "gold": "general",   "split": "train"},
    {"input": "My invoice shows the wrong amount.", "gold": "billing",   "split": "train"},
    {"input": "I can't log in to my account.",      "gold": "technical", "split": "train"},
]
```

<span style="font-size:13px; color:#1a4971; font-weight:600;">Output</span> — correct examples only, `split` field dropped

```python
demo_pool = [
    {"input": "I was charged twice this month.",    "gold": "billing"},
    {"input": "My invoice shows the wrong amount.", "gold": "billing"},
]
```

---

### <span style="background:#1a4971; color:#fff; padding:3px 12px; border-radius:4px; font-weight:600;">propose_candidates()</span>

<span style="font-size:13px; color:#1a4971; font-weight:600;">Input</span> — demo_pool from bootstrap

```python
demo_pool = [
    {"input": "I was charged twice this month.",    "gold": "billing"},
    {"input": "My invoice shows the wrong amount.", "gold": "billing"},
]
```

<span style="font-size:13px; color:#1a4971; font-weight:600;">Output</span> — list of candidates, each with instruction + randomly sampled demo subset

```python
candidates = [
    {
        "instruction": "Classify the following customer support ticket into exactly one of three categories: billing, technical, or general. Respond with only the category label.",
        "demos": [
            {"input": "I was charged twice this month.",    "gold": "billing"},
            {"input": "My invoice shows the wrong amount.", "gold": "billing"},
        ],
    },
    {
        "instruction": "Classify the following customer support ticket into exactly one of three categories: billing, technical, or general. Respond with only the category label.",
        "demos": [
            {"input": "My invoice shows the wrong amount.", "gold": "billing"},
            {"input": "I was charged twice this month.",    "gold": "billing"},
        ],
    },
    # ... n_candidates total
]
```

---

### <span style="background:#1a4971; color:#fff; padding:3px 12px; border-radius:4px; font-weight:600;">build_program()</span>

<span style="font-size:13px; color:#1a4971; font-weight:600;">Input</span> — one candidate dict

```python
candidate = {
    "instruction": "Classify the following customer support ticket into exactly one of three categories: billing, technical, or general. Respond with only the category label.",
    "demos": [
        {"input": "I was charged twice this month.",    "gold": "billing"},
        {"input": "My invoice shows the wrong amount.", "gold": "billing"},
    ],
}
```

<span style="font-size:13px; color:#1a4971; font-weight:600;">Prompt sent to LLM</span>

```python
prompt = """\
Classify the following customer support ticket into exactly one of three categories: billing, technical, or general. Respond with only the category label.
---
Examples:
Ticket: "I was charged twice this month."
Category: "billing"
Ticket: "My invoice shows the wrong amount."
Category: "billing"
---
Ticket: "My card was charged three times."
Category:"""
```

---

### <span style="background:#1a4971; color:#fff; padding:3px 12px; border-radius:4px; font-weight:600;">select_best()</span>

<span style="font-size:13px; color:#1a4971; font-weight:600;">Input</span> — candidates list + val dataset

```python
candidates = [...]  # list of {"instruction": str, "demos": list[dict]}

val = [
    {"input": "I was billed incorrectly.", "gold": "billing"},
    {"input": "The app keeps crashing.",   "gold": "technical"},
]
```

<span style="font-size:13px; color:#1a4971; font-weight:600;">Output</span> — best candidate dict + its val score

```python
best_candidate = {
    "instruction": "Classify the following customer support ticket into exactly one of three categories: billing, technical, or general. Respond with only the category label.",
    "demos": [
        {"input": "My invoice shows the wrong amount.", "gold": "billing"},
        {"input": "I was charged twice this month.",    "gold": "billing"},
    ],
}
best_score = 0.8
```

---

