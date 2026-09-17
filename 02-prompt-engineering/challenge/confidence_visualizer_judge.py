import math
import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_provider import get_llm
from cost_estimator import DRY_RUN, record_generate_call

load_dotenv()
llm = get_llm()

DEFAULT_JUDGE_CRITERIA = (
    "Evaluate the content based on the following criteria: The tone must be 'excited' and "
    "'celebratory,' evoking a sense of anticipation and excitement (e.g., 'Look forward to what's "
    "next'). The message must include personalization using the customer's membership tier and a "
    "clear call-to-action (e.g., 'Explore your benefits'). Avoid any overpromising phrases such as "
    "'unlimited rewards,' and ensure the tone does not become overly casual, as this would dilute "
    "the celebratory atmosphere. Messages that fail to meet these tone expectations, lack "
    "personalization, or omit the required call-to-action should be rated as unacceptable. If "
    "structural elements or tone descriptors are missing, infer reasonable defaults based on the "
    "context of a membership rewards email campaign."
)

# --- Functions ---
def build_prompt(question: str, choices: list[str]) -> tuple[str, dict[str, str]]:
    #
    letters = [chr(65 + i) for i in range(len(choices))]
    letter_to_answer = dict(zip(letters, choices))
    options_str = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    prompt = f"{question}\n{options_str}\n"
    return prompt, letter_to_answer


def query_llm(question: str, choices: list[str], system_prompt: str) -> tuple[list[tuple[str, float]], str]:
    prompt, letter_to_answer = build_prompt(question, choices)
    letters = list(letter_to_answer.keys())
    system_prompt = (
        system_prompt
        + " Respond with only the single letter that best matches your answer. "
        + "Your only valid outputs are the answer letters provided, with no explanation, "
        + "no markdown formatting, no symbols, and no extra text."
    )
    # print(system_prompt)
    output = llm.generate(
        user_prompt=prompt,
        system_prompt=system_prompt,
        temp=None,
        logprobs=True,
        top_logprobs=len(choices),
        return_raw=True,
    )
    content = output["choices"][0]["message"]["content"].strip()
    top_logprobs = output["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    seen = set()
    results = []
    for entry in top_logprobs:
        letter = entry["token"].strip()
        if letter in letter_to_answer and letter not in seen:
            seen.add(letter)
            prob = math.exp(entry["logprob"])
            results.append((letter_to_answer[letter], prob))

    for letter in letters:
        if letter not in seen:
            results.append((letter_to_answer[letter], 0.0))

    chosen = letter_to_answer.get(content, content)
    return results, chosen


def render_chart(results: list[tuple[str, float]]) -> None:
    labels = [r[0] for r in results]
    probs = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(9, len(labels) * 1.4))
    fig.patch.set_facecolor("white")

    bars = ax.barh(
        range(len(labels)), probs,
        color="white", edgecolor="black", linewidth=1.5, height=0.6,
    )

    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center", ha="left", fontsize=13,
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False)

    plt.tight_layout()
    st.pyplot(fig)


def print_cli_table(results: list[tuple[str, float]], chosen: str) -> None:
    print("\nOutput:")
    print(f"Model choice: {chosen}")
    print()
    print(f"{'Choice':<60} {'Probability':>12}")
    print("-" * 74)
    for answer, prob in results:
        print(f"{answer:<60} {prob:>11.2%}")


# --- Streamlit UI ---

st.title("Judge Criteria Confidence Visualizer")

judge_criteria = st.text_area(
    "Judge Criteria",
    value=DEFAULT_JUDGE_CRITERIA,
)

question = st.text_input(
    "Copy",
    value="",
    placeholder="Enter copy that you want to evaluate",
)
choices_raw = st.text_area(
    "Answer choices",
    value="",
    placeholder="Enter answer choices (one per line)",
)

if st.button("Ask"):
    choices = [c.strip() for c in choices_raw.strip().splitlines() if c.strip()]
    if len(choices) < 2:
        st.error("Enter at least 2 choices.")
    elif DRY_RUN:
        prompt, _ = build_prompt(question, choices)
        input_tokens, cost = record_generate_call(
            os.getenv("LLM_PROVIDER", ""), os.getenv("LLM_MODEL", ""), prompt, judge_criteria
        )
        st.info(f"DRY RUN — no call made. Estimated input tokens: {input_tokens}, cost: ${cost:.4f}")
    else:
        with st.spinner("Querying model..."):
            results, chosen = query_llm(question, choices, judge_criteria)
        st.write(f"**Model chose:** {chosen}")
        print_cli_table(results, chosen)
        render_chart(results)

