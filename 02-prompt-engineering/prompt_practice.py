"""
Prompt templates:
- prompts/zero_shot.txt
- prompts/few_shot.txt
- prompts/few_shot_cot.txt
Input {{ticket_text}}:
- substitute {{ticket_text}} with the actual text of the ticket you want to classify.
Prompt after Substitution = Prompt template + {{ticket_text}}:
1) substitute {{ticket_text}} in each prompt template with tickets from train_tickets.txt, evaluate performance of each prompt template, and choose most accurate prompt template
2) substitute {{ticket_text}} in best prompt template with tickets from test_tickets.txt,
3) If tuning necessary - tune the prompt to get the best results; Else - reduce instruction to a basic format and try to tune again
"""

import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from azure_llm_wrapper import AzureLLMWrapper


load_dotenv()
api_key = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY", "")
model = os.getenv("AZURE_OPENAI_MODEL", "")
version = os.getenv("AZURE_OPENAI_API_VERSION", "")
endpoint = os.getenv("AZURE_APIM_ENDPOINT", "") + "/openai/deployments/" + model + "/chat/completions?api-version=" + version
llm = AzureLLMWrapper(endpoint=endpoint, api_key=api_key)

# function to read tickets from file and return data
# input: string - ticket file path
# output: list of strings - tickets, list of strings - expected values
def read_tickets(file_path):
    tickets = []
    expected_values = []
    ticket_line_pattern = re.compile(r'^\s*\d+\.\s*"(.*)"\s*$')
    expected_line_pattern = re.compile(r'^\s*Expected:\s*(\w+)\s*$')

    with open(file_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip empty and comment lines.
            if not line or line.startswith("#"):
                continue

            # Extract expected value lines.
            expected_match = expected_line_pattern.match(line)
            if expected_match:
                expected_values.append(expected_match.group(1))
                continue

            # Extract numbered ticket lines (e.g., 1. "...").
            ticket_match = ticket_line_pattern.match(line)
            if ticket_match:
                tickets.append(ticket_match.group(1))

    return tickets, expected_values

# function to substitute {{ticket_text}} in prompt template with actual ticket text
# input: string - prompt template, string - ticket text
# output: string - prompt after substitution
def substituted_prompt(file_path, ticket):
    prompt = open(file_path, 'r').read()
    prompt = prompt.replace("{{ticket_text}}", ticket)
    return prompt

# function to extract category label from a CoT response that contains
# "Reasoning: ..." and "Category: ..." (handles markdown bold and "Final Category" variants)
# input: string - raw LLM output
# output: string - lowercase category label, or full output if no match
def extract_cot_category(output):
    match = re.search(r'(?:Final\s+)?Category:\s*\*{0,2}(\w+)\*{0,2}', output, re.IGNORECASE)
    return match.group(1).strip().lower() if match else output.strip().lower()


def extract_cot_reasoning(output):
    match = re.search(
        r'Reasoning:\s*(.*?)(?:(?:Final\s+)?Category:)',
        output,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else output.strip()


# function to generate output from LLM and evaluate performance
# input: string - prompt after substitution, string - expected value, callable - optional parser to extract comparable value from output
# output: boolean - whether output matches expected value, string - LLM output
def evaluate_performance(prompt, expected_value, parser=None):
    output = llm.generate(user_prompt=prompt)
    comparable = parser(output) if parser else output.strip().lower()
    is_correct = comparable == expected_value.strip().lower()
    return is_correct, output

# function for training with zero, few, and few CoT shot
# 1) read_tickets - train tickets
# 2) loop through tickets & expected values
    # 3) substitute each ticket into prompt
    # 4) evaluate_performance - result should be is_correct and output of llm generation
    # 5) count correctness, append outputs to a list
# 6) return correctness count and output list
# input: string - prompt_type
# output: int - count, list of strings - output
def process_prompts(prompt_type, ticket_type):
    tickets, expected_values = read_tickets(f"02-prompt-engineering/prompts/{ticket_type}_tickets.txt")
    correct = 0
    output = []
    parser = extract_cot_category if prompt_type == "few_shot_cot" else None
    for ticket, expected_value in zip(tickets, expected_values):
        prompt = substituted_prompt(f"02-prompt-engineering/prompts/{prompt_type}.txt", ticket)
        is_correct, single_output = evaluate_performance(prompt, expected_value, parser=parser)
        if is_correct: correct += 1
        output.append(single_output)
    return correct, output, expected_values

# format:
    # Type of prompt
    # prompt
    # loop through and print all - ticket - expected value - actual value
    # print final accuracy percentage
# function for building results report text
def build_results_report(prompt_type, ticket_type, correct_count, output_list):
    tickets, expected_values = read_tickets(f"02-prompt-engineering/prompts/{ticket_type}_tickets.txt")
    report_lines = []
    report_lines.append("===================================================================\n\n")
    report_lines.append(f"**{prompt_type.replace('_', ' ').title()}**\n")
    report_lines.append("--------------------------------\n")
    prompt_template = open(f"02-prompt-engineering/prompts/{prompt_type}.txt", 'r').read().strip()
    report_lines.append(f"Prompt:\n{prompt_template}\n")
    report_lines.append("--------------------------------\n")
    for ticket, expected_value, output in zip(tickets, expected_values, output_list):
        if prompt_type == "few_shot_cot":
            reasoning = extract_cot_reasoning(output)
            category = extract_cot_category(output)
            report_lines.append(f"Ticket: {ticket}\nExpected: {expected_value}\nOutput:\nReasoning: {reasoning}\nCategory: {category}\n\n")
        else:
            report_lines.append(f"Ticket: {ticket}\nExpected: {expected_value}\nOutput: {output}\n\n")
    accuracy = correct_count / len(tickets)
    report_lines.append("--------------------------------\n")
    report_lines.append(f"**{prompt_type.replace('_', ' ').title()} Accuracy: {accuracy:.2%}**\n")
    report_lines.append("\n===================================================================")
    return "".join(report_lines)


# TRAIN TICKETS
prompt_correct_count = {
    "zero_shot": 0,
    "few_shot": 0,
    "few_shot_cot": 0
}

prompt_outputs = {
    "zero_shot": [],
    "few_shot": [],
    "few_shot_cot": []
}

train_expected_values = []

prompt_correct_count["zero_shot"], prompt_outputs["zero_shot"], train_expected_values = process_prompts("zero_shot", "train")
prompt_correct_count["few_shot"], prompt_outputs["few_shot"], _ = process_prompts("few_shot", "train")
prompt_correct_count["few_shot_cot"], prompt_outputs["few_shot_cot"], _ = process_prompts("few_shot_cot", "train")

reports = []
reports.append(build_results_report("zero_shot", "train", prompt_correct_count["zero_shot"], prompt_outputs["zero_shot"]))
reports.append(build_results_report("few_shot", "train", prompt_correct_count["few_shot"], prompt_outputs["few_shot"]))
reports.append(build_results_report("few_shot_cot", "train", prompt_correct_count["few_shot_cot"], prompt_outputs["few_shot_cot"]))

results_path = os.path.join(os.path.dirname(__file__), "artifacts", "prompt_train_tickets_results.txt")
with open(results_path, "w") as results_file:
    results_file.write("\n".join(reports))


# TEST TICKETS
prompt_correct_count["zero_shot"], prompt_outputs["zero_shot"], test_expected_values = process_prompts("zero_shot", "test")
prompt_correct_count["few_shot"], prompt_outputs["few_shot"], _ = process_prompts("few_shot", "test")
prompt_correct_count["few_shot_cot"], prompt_outputs["few_shot_cot"], _ = process_prompts("few_shot_cot", "test")

reports = []
reports.append(build_results_report("zero_shot", "test", prompt_correct_count["zero_shot"], prompt_outputs["zero_shot"]))
reports.append(build_results_report("few_shot", "test", prompt_correct_count["few_shot"], prompt_outputs["few_shot"]))
reports.append(build_results_report("few_shot_cot", "test", prompt_correct_count["few_shot_cot"], prompt_outputs["few_shot_cot"]))

results_path = os.path.join(os.path.dirname(__file__), "artifacts", "prompt_test_tickets_results.txt")
with open(results_path, "w") as results_file:
    results_file.write("\n".join(reports))

print("Complete! Check artifacts folder for results.")