"""
LLM Client wrapper for OpenAI-compatible APIs.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, provider="openai", model=None):
        self.provider = provider
        self.model = model or "gpt-4o-mini"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])