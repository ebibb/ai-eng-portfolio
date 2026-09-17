"""
Provider-agnostic LLM interface.

Every call site imports `get_llm` and gets back an object with the same
`generate()` / `embed()` contract, regardless of which backend is configured.
Backend selection happens entirely through environment variables — no code
changes are needed to switch providers.

Environment variables:
    LLM_PROVIDER            "openai" | "anthropic" | "azure" | "ollama"   (required)
    LLM_MODEL               chat/completion model name                    (required for generate())
    LLM_EMBEDDING_MODEL     embedding model name                          (required for embed())

    openai:     OPENAI_API_KEY
    anthropic:  ANTHROPIC_API_KEY   (no embeddings API — embed() raises)
    azure:      AZURE_APIM_ENDPOINT, AZURE_APIM_SUBSCRIPTION_KEY, AZURE_OPENAI_API_VERSION
    ollama:     OLLAMA_HOST (default http://localhost:11434); LLM_MODEL is the
                Ollama tag, e.g. OLLAMA_MODEL=gemma4:latest can be used as an
                alias for LLM_MODEL for a local-only setup.

See .env.example for a filled-in sample of each backend.
"""

import os
import time

import requests


class LLMProvider:
    def generate(self, user_prompt, system_prompt=None, temp=None, logprobs=False,
                 top_logprobs=None, return_raw=False):
        raise NotImplementedError

    def embed(self, text):
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, model="", embedding_model=""):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model

    def generate(self, user_prompt, system_prompt=None, temp=None, logprobs=False,
                 top_logprobs=None, return_raw=False):
        if not self.model:
            raise ValueError("Missing LLM_MODEL environment variable")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        kwargs = {"model": self.model, "messages": messages}
        if temp is not None:
            kwargs["temperature"] = temp
        if logprobs:
            kwargs["logprobs"] = True
        if top_logprobs is not None:
            kwargs["top_logprobs"] = top_logprobs
        response = self.client.chat.completions.create(**kwargs)
        if return_raw:
            return response.model_dump()
        return response.choices[0].message.content

    def embed(self, text):
        if not self.embedding_model:
            raise ValueError("Missing LLM_EMBEDDING_MODEL environment variable")
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding


class AnthropicProvider(LLMProvider):
    def __init__(self, model="", embedding_model=""):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY environment variable")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, user_prompt, system_prompt=None, temp=None, logprobs=False,
                 top_logprobs=None, return_raw=False):
        if not self.model:
            raise ValueError("Missing LLM_MODEL environment variable")
        if logprobs:
            raise NotImplementedError(
                "Anthropic's API does not expose logprobs. Use LLM_PROVIDER=openai or "
                "LLM_PROVIDER=azure for logprobs-based call sites."
            )
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temp is not None:
            kwargs["temperature"] = temp
        response = self.client.messages.create(**kwargs)
        if return_raw:
            return response.model_dump()
        return response.content[0].text

    def embed(self, text):
        raise NotImplementedError(
            "Anthropic has no embeddings API. Set LLM_PROVIDER to openai, azure, or ollama "
            "to run code that calls embed()."
        )


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, model="", embedding_model=""):
        self.api_key = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY", "")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
        base = os.getenv("AZURE_APIM_ENDPOINT", "")
        if not base:
            raise ValueError("Missing AZURE_APIM_ENDPOINT environment variable")
        if not self.api_key:
            raise ValueError("Missing AZURE_APIM_SUBSCRIPTION_KEY environment variable")
        self.chat_url = (
            f"{base}/openai/deployments/{model}/chat/completions?api-version={self.api_version}"
            if model else None
        )
        self.embedding_url = (
            f"{base}/openai/deployments/{embedding_model}/embeddings?api-version={self.api_version}"
            if embedding_model else None
        )

    def _post_with_retry(self, url, payload, max_retries=5, timeout=60):
        headers = {"Content-Type": "application/json", "api-key": self.api_key}
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries:
                    raise
                wait = 2 ** attempt
                print(f"  [retry {attempt}/{max_retries}] Connection error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code == 200:
                return response.json()

            if attempt == max_retries:
                raise Exception(f"Request failed: {response.status_code} at {url} - {response.text}")
            wait = 2 ** attempt
            print(f"  [retry {attempt}/{max_retries}] Status {response.status_code}. Retrying in {wait}s...")
            time.sleep(wait)

    def generate(self, user_prompt, system_prompt=None, temp=None, logprobs=False,
                 top_logprobs=None, return_raw=False):
        if not self.chat_url:
            raise ValueError("Missing LLM_MODEL environment variable")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {"messages": messages}
        if temp is not None:
            payload["temperature"] = temp
        if logprobs:
            payload["logprobs"] = logprobs
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs

        response_json = self._post_with_retry(self.chat_url, payload)
        if return_raw:
            return response_json
        try:
            return response_json["choices"][0]["message"]["content"]
        except Exception:
            return response_json

    def embed(self, text):
        if not self.embedding_url:
            raise ValueError("Missing LLM_EMBEDDING_MODEL environment variable")
        response_json = self._post_with_retry(self.embedding_url, {"input": text})
        return response_json["data"][0]["embedding"]


class OllamaProvider(LLMProvider):
    def __init__(self, model="", embedding_model=""):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "")
        self.embedding_model = embedding_model

    def generate(self, user_prompt, system_prompt=None, temp=None, logprobs=False,
                 top_logprobs=None, return_raw=False):
        if not self.model:
            raise ValueError("Missing LLM_MODEL (or OLLAMA_MODEL) environment variable")
        if logprobs:
            raise NotImplementedError(
                "Ollama's chat API does not expose logprobs. Use LLM_PROVIDER=openai or "
                "LLM_PROVIDER=azure for logprobs-based call sites."
            )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {"model": self.model, "messages": messages, "stream": False}
        if temp is not None:
            payload["options"] = {"temperature": temp}

        response = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        if return_raw:
            return data
        return data["message"]["content"]

    def embed(self, text):
        if not self.embedding_model:
            raise ValueError("Missing LLM_EMBEDDING_MODEL environment variable")
        response = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.embedding_model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]


_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "azure": AzureOpenAIProvider,
    "ollama": OllamaProvider,
}


def get_llm():
    provider_name = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider_name:
        raise ValueError(
            "Missing LLM_PROVIDER environment variable. Set it to one of: "
            f"{', '.join(_PROVIDERS)}."
        )
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider_name}'. Use one of: {', '.join(_PROVIDERS)}.")

    model = os.getenv("LLM_MODEL", "")
    embedding_model = os.getenv("LLM_EMBEDDING_MODEL", "")
    return _PROVIDERS[provider_name](model=model, embedding_model=embedding_model)
