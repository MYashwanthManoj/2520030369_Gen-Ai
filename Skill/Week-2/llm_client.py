import os
import requests
from abc import ABC, abstractmethod
from google import genai


# Common interface
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass


# Gemini cloud provider
class GeminiProvider(LLMProvider):
    def __init__(self, api_key, model="gemini-3.8-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt):
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
        except Exception:
            # Fallback if gemini-3.8-flash experiences temporary high demand (503/504)
            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )
        return response.text


# Ollama local provider
class OllamaProvider(LLMProvider):
    def __init__(self, model="llama3.2"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(
            self.url,
            json=payload,
            timeout=120
        )
        if response.status_code == 404:
            # Fallback to available local models if llama3.2 is not pulled
            try:
                tags_resp = requests.get("http://localhost:11434/api/tags", timeout=5)
                if tags_resp.status_code == 200:
                    models = [m["name"] for m in tags_resp.json().get("models", [])]
                    if models:
                        self.model = models[0]
                        payload["model"] = self.model
                        response = requests.post(self.url, json=payload, timeout=120)
            except Exception:
                pass
        response.raise_for_status()
        return response.json()["response"]


# Common client
class LLMClient:
    def __init__(self, provider):
        self.provider = provider

    def generate(self, prompt):
        return self.provider.generate(prompt)

