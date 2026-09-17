"""
Sweep temperature [0, 0.3, 0.7, 1.0] on a fixed prompt.

Goal: observe how temperature affects output diversity.

Run:
    python 01-llm-fundamentals/llm_practice.py
"""

"""
Completed steps:
- initialize the LLM provider
- define prompt and temperatures
- uses the provider to call the LLM and print outputs for each temperature
    - the provider formats the prompt and payload, calls the API, and returns the output
"""


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from llm_provider import get_llm


load_dotenv()
llm = get_llm()


prompt = "Give one word"
print(f"Prompt: {prompt}")
temps = [0, 0.3, 0.7, 1.0]
for t in temps:
    output = llm.generate(user_prompt=prompt, temp=t)
    print(f"Temperature: {t}\n\t {output}")