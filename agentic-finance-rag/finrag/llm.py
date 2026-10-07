"""LLM access for planning and answer writing. Groq / OpenAI / Gemini share one OpenAI-compatible client."""
from __future__ import annotations

import json
import re

import httpx

from . import config

BASE_URLS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
}


class ChatLLM:
    def __init__(self, provider: str, api_key: str, model: str = "") -> None:
        base, default_model = BASE_URLS[provider]
        self.model = model or default_model
        self._client = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {"model": self.model, "temperature": 0.1,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self._client.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def json(self, system: str, user: str) -> dict:
        raw = self.chat(system, user, json_mode=True)
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


def get_llm() -> ChatLLM | None:
    """Returns None in mock mode; the graph then uses its rule-based planner and extractive writer."""
    if config.LLM_PROVIDER == "mock" or not config.LLM_API_KEY:
        return None
    return ChatLLM(config.LLM_PROVIDER, config.LLM_API_KEY, config.LLM_MODEL)
