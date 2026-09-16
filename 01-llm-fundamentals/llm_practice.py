"""
Sweep temperature [0, 0.3, 0.7, 1.0] on a fixed prompt.

Goal: observe how temperature affects output diversity.

Run:
    python 01-llm-fundamentals/llm_practice.py
"""

"""
Completed steps:
- initialize wrapper with endpoint and key
- define prompt and temperatures
- uses wrapper to call LLM and print outputs for each temperature
    - the wrapper formats the prompt and payload, calls the API, and returns the output
"""


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from azure_llm_wrapper import AzureLLMWrapper


load_dotenv()
api_key = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY", "")
model = os.getenv("AZURE_OPENAI_MODEL", "")
version = os.getenv("AZURE_OPENAI_API_VERSION", "")
endpoint = os.getenv("AZURE_APIM_ENDPOINT", "") + "/openai/deployments/" + model + "/chat/completions?api-version=" + version
llm = AzureLLMWrapper(endpoint=endpoint, api_key=api_key)


prompt = "Give one word"
print(f"Prompt: {prompt}")
temps = [0, 0.3, 0.7, 1.0]
for t in temps:
    output = llm.generate(user_prompt=prompt, temp=t)
    print(f"Temperature: {t}\n\t {output}")