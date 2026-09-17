import math
import os
import sys
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_provider import get_llm

load_dotenv()
llm = get_llm()

SYSTEM_PROMPT = (
    "You are a multiple-choice answering assistant. "
    "You will be given a question and a list of lettered options. "
    "Respond with only the single letter that best matches your answer. "
    "Do not explain, punctuate, or include anything else."
)

# --- Functions ---
def build_prompt(question: str, choices: list[str]) -> tuple[str, dict[str, str]]:
    #
    letters = [chr(65 + i) for i in range(len(choices))]
    letter_to_answer = dict(zip(letters, choices))
    options_str = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    prompt = f"{question}\n{options_str}\n"
    return prompt, letter_to_answer


def query_llm(question: str, choices: list[str]) -> tuple[list[tuple[str, float]], str]:
    prompt, letter_to_answer = build_prompt(question, choices)
    output = llm.generate(
        user_prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temp=None,
        logprobs=True,
        top_logprobs=len(choices),
        return_raw=True,
    )
    content = output["choices"][0]["message"]["content"].strip()
    top_logprobs = output["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    results = []
    for entry in top_logprobs:
        letter = entry["token"].strip()
        prob = math.exp(entry["logprob"])
        answer = letter_to_answer.get(letter, letter)
        results.append((answer, prob))

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
            bar.get_x() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.0f}%",
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


# --- Streamlit UI ---

st.title("LLM Answer Probabilities")

question = st.text_input(
    "Question",
    value="",
    placeholder="Enter a question with discrete answers",
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
    else:
        with st.spinner("Querying model..."):
            results, chosen = query_llm(question, choices)
        st.write(f"**Model chose:** {chosen}")
        render_chart(results)
