
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"
REQUEST_TIMEOUT_S = 30


class LLMError(RuntimeError):
    """Raised on any API/network failure. Callers must treat this as a
    signal to fail visibly (CLI) or route to a human-support message (UI),
    never to fabricate an answer."""


@dataclass
class LLMResponse:
    text: str
    model: str
    raw: dict


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        if not self.api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and "
                "add your key from https://openrouter.ai/settings/keys"
            )

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 500) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
           
            "HTTP-Referer": "https://github.com/learnforge/support-assistant",
            "X-Title": "LearnForge Support Assistant",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            raise LLMError(f"Network error calling OpenRouter: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"OpenRouter returned HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected OpenRouter response shape: {data}") from exc

        return LLMResponse(text=text, model=data.get("model", self.model), raw=data)
