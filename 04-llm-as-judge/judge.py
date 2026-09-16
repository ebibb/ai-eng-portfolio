"""
The judge.

There are two different LLMs here. Don't mix them up:

  1. The SOLVER: It reads an input and
     produces a *generated output* — the answer.
  2. The JUDGE (this file). It does NOT answer the input. It *grades* the
     generated output the solver already produced, and explains its grade.

         input ──▶ [ SOLVER ] ──▶ generated output ──┐
                                                     ├──▶ [ JUDGE ] ──▶ score + critique
         input ─────────────────────────────────────-┘

The judge needs the generated output because you cannot grade an answer you
cannot see. If you only handed the judge the input, it would have to solve the
task itself — and then it is just a second solver, not a judge.

What the judge returns:
  - score (float):   a number, so the eval harness can average it.
  - critique (str):  a one-sentence explanation of *why*. This — not the score —
    is the point. It is the textual feedback a reflective optimizer (GEPA)
    passes to *another* LLM to rewrite the prompt. The score is secondary.

Two modes:
  - Reference-based  (expected output provided): grade the generated output
    against a known correct answer. Useful when you have ground truth
    (this classification task).
  - Reference-free    (expected output = None):  grade the generated output
    against a rubric only, with no correct answer to compare to. This is how
    judges work on open-ended tasks (summaries, code, essays) where there is no
    single right string — and it is the mode GEPA most often relies on.

How this plugs into the eval harness:
  ../03-eval-harness/eval.py:evaluate() is just a loop:
      for example in dataset:
          generated_output = program(example["input"])      # solver answers
          score            = metric(generated_output, ...)  # grade it
      return mean(scores)
  `metric()` below is a drop-in scalar replacement for the eval harness's
  metric() — same signature, so you pass it straight to evaluate(). It calls
  judge() and returns only the score (evaluate() can only average numbers).
  Use judge() directly when you also want the critique.

Run:
    python 04-llm-as-judge/judge.py

Requires:
    AZURE_APIM_ENDPOINT and AZURE_APIM_SUBSCRIPTION_KEY set in your environment.
"""
import sys
import os
from dataclasses import dataclass
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from azure_llm_wrapper import AzureLLMWrapper  # noqa: E402

load_dotenv()
api_key = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY", "")
model = os.getenv("AZURE_OPENAI_MODEL", "")
version = os.getenv("AZURE_OPENAI_API_VERSION", "")
endpoint = os.getenv("AZURE_APIM_ENDPOINT", "") + "/openai/deployments/" + model + "/chat/completions?api-version=" + version
llm = AzureLLMWrapper(endpoint=endpoint, api_key=api_key)


# Reference-based: a known correct answer is available to compare against.
JUDGE_PROMPT_REFERENCE = """\
You are an impartial judge evaluating a classification answer.

Input: {input_text}
Expected output: {expected_output}
Generated output: {generated_output}

Rate the generated output on a scale of 0 to 1:
  1.0 = fully correct
  0.5 = partially correct (e.g., right general area but wrong specific label)
  0.0 = wrong

Then write a one-sentence critique that names *specifically* what was wrong or right.
Do not write a generic statement like "the answer is incorrect" — name the failure.

Respond in EXACTLY this format (two lines, nothing else):
Score: <number>
Critique: <one sentence>
"""

# Reference-free: no correct answer; grade the generated output against a rubric only.
JUDGE_PROMPT_REFERENCE_FREE = """\
You are an impartial judge evaluating a classification answer.
There is NO expected output — grade the generated output on its own merits.

Input: {input_text}
Generated output: {generated_output}

Rate how well the generated category fits the input on a scale of 0 to 1:
  1.0 = the category clearly fits the input
  0.5 = plausible but not the best fit
  0.0 = the category does not fit the input

Then write a one-sentence critique that names *specifically* why the category
does or does not fit. Do not write a generic statement — name the reason.

Respond in EXACTLY this format (two lines, nothing else):
Score: <number>
Critique: <one sentence>
"""


@dataclass
class JudgeResult:
    score: float
    critique: str

def prompt_substitute(input_text: str, generated_output: str, expected_output=None) -> str:
    if expected_output is not None:
        return JUDGE_PROMPT_REFERENCE.format(
            input_text=input_text,
            expected_output=expected_output,
            generated_output=generated_output
        )
    else:
        return JUDGE_PROMPT_REFERENCE_FREE.format(
            input_text=input_text,
            generated_output=generated_output
        )

def parse_judge_response(response_text: str) -> JudgeResult:
    try:
        lines = response_text.strip().split("\n")
        score_line = lines[0]
        critique_line = lines[1]
        score = float(score_line.split(":")[1].strip())
        critique = critique_line.split(":")[1].strip()
        return JudgeResult(score=score, critique=critique)
    except Exception:
        return JudgeResult(score=0.0, critique="parse error")


def judge(
    generated_output: str,
    expected_output= None,
    input_text: str = "",
) -> JudgeResult:
    """Grade a generated output and explain the grade.

    The judge sits *after* the solver: it receives the solver's generated output
    as the thing to evaluate. If expected_output is provided, it grades
    reference-based (compare to the correct answer). If expected_output is None,
    it grades reference-free (rubric only) — the mode used for open-ended outputs
    and the one GEPA typically relies on.

    TODO:
    - Pick JUDGE_PROMPT_REFERENCE when expected_output is provided, else
      JUDGE_PROMPT_REFERENCE_FREE.
    - Format the chosen prompt with input_text, generated_output
      (and expected_output if present).
    - Call llm.generate() with temperature=0 (deterministic judge).
    - Parse "Score: X" (first line) and "Critique: ..." (second line) from response text.
    - Return JudgeResult(score=float, critique=str).
    - Handle parse errors gracefully (e.g., return score=0.0, critique="parse error").
    """
    user_prompt = prompt_substitute(input_text, generated_output, expected_output)
    judge_output = llm.generate(user_prompt=user_prompt, temp=0)
    return parse_judge_response(judge_output)


def metric(generated_output: str, expected_output: str) -> float:
    """Drop-in scalar replacement for the eval harness's metric() — returns only the score.

    Same signature as the eval harness's exact-match metric, so evaluate() can call it
    unchanged. It throws away the critique because evaluate() can only average
    numbers — use judge() when you want the critique too.

    TODO: call judge(generated_output, expected_output), return result.score.
    """
    return judge(generated_output, expected_output).score


if __name__ == "__main__":
    # Acceptance test: the critique should name the specific failure.
    result = judge(
        generated_output="billing",
        expected_output="technical",
        input_text="My app crashes whenever I try to upload a file.",
    )
    print(f"Score:    {result.score}")
    print(f"Critique: {result.critique}")
    print()
    print("Acceptance check: does the critique mention 'crash', 'software', or 'technical'?")
    print("If it just says 'the answer is incorrect', the judge prompt needs tuning.")

    # ── Judge judge_test_data.jsonl (reference-based) ─────────────────────────
    # import json
    #
    # INPUT_PATH = os.path.join(os.path.dirname(__file__), "judge_test_data.jsonl")
    # OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "judge_logs", "judge_ref_results.txt")
    #
    # os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    #
    # with open(INPUT_PATH) as in_f, open(OUTPUT_PATH, "w") as out_f:
    #     for i, line in enumerate(in_f, start=1):
    #         if not line.strip():
    #             continue
    #         ex = json.loads(line)
    #         prediction = ex.get("prediction", "")
    #         result = judge(
    #             generated_output=prediction,
    #             expected_output=ex["gold"],
    #             input_text=ex["input"],
    #         )
    #         out_f.write(f"Example {i}\n")
    #         out_f.write(f"Input:     {ex['input']}\n")
    #         out_f.write(f"Gold:      {ex['gold']}\n")
    #         out_f.write(f"Predicted: {prediction}\n")
    #         out_f.write(f"Score:     {result.score}\n")
    #         out_f.write(f"Critique:  {result.critique}\n")
    #         out_f.write("\n---\n\n")
    #         print(f"Example {i}: {result.score}  {result.critique}")

    # ── Judge judge_test_data.jsonl (reference-free) ───────────────────────────
    # import json

    # INPUT_PATH = os.path.join(os.path.dirname(__file__), "judge_test_data.jsonl")
    # OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "judge_logs", "judge_ref_free_results.txt")

    # os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # with open(INPUT_PATH) as in_f, open(OUTPUT_PATH, "w") as out_f:
    #     for i, line in enumerate(in_f, start=1):
    #         if not line.strip():
    #             continue
    #         ex = json.loads(line)
    #         prediction = ex.get("prediction", "")
    #         result = judge(
    #             generated_output=prediction,
    #             input_text=ex["input"],
    #         )
    #         out_f.write(f"Example {i}\n")
    #         out_f.write(f"Input:     {ex['input']}\n")
    #         out_f.write(f"Predicted: {prediction}\n")
    #         out_f.write(f"Score:     {result.score}\n")
    #         out_f.write(f"Critique:  {result.critique}\n")
    #         out_f.write("\n---\n\n")
    #         print(f"Example {i}: {result.score}  {result.critique}")
