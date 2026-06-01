"""
LLM abstraction for xsim.

Supports:
- Ollama (local, recommended default)
- Any OpenAI-compatible endpoint (Groq, Together, OpenRouter, vLLM, etc.)
  by providing base_url + api_key

This single interface powers both the agents (posting + deciding engagement)
and future LLM-powered ranking modes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import ollama
from openai import OpenAI


@dataclass
class LLMConfig:
    provider: str = "ollama"           # "ollama" or "openai_compatible"
    model: str = "qwen2.5:7b"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 600


class LLMClient(Protocol):
    def chat(self, system: str, user: str, **kwargs) -> str: ...


class OllamaClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = ollama

    def chat(self, system: str, user: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response = self.client.chat(
            model=self.config.model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        return response["message"]["content"].strip()


class OpenAICompatibleClient:
    """Works with OpenAI, Groq, Together, OpenRouter, Fireworks, local vLLM, etc."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url or "https://api.openai.com/v1",
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
        )

    def chat(self, system: str, user: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content.strip()


def get_llm_client(config: LLMConfig) -> LLMClient:
    """Factory that returns the right client for the chosen provider."""
    if config.provider == "ollama":
        return OllamaClient(config)
    elif config.provider == "openai_compatible":
        return OpenAICompatibleClient(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


# Convenience helper for quick prompts (used heavily by agents)
def quick_chat(
    client: LLMClient,
    system: str,
    user: str,
) -> str:
    return client.chat(system, user)
