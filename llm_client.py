from __future__ import annotations

import json
import os
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # Optional dependency.
    load_dotenv = None
from openai import OpenAI

if load_dotenv is not None:
    load_dotenv()


FREE_MODELS = {
    # "auto-free": "openrouter/free",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    # "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    # "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-5-mini": "openai/gpt-5-mini-2025-08-07",
    "deepseek-v3.2": "deepseek/deepseek-v3.2-20251201",
    # "deepseek-v3.1": "deepseek/deepseek-chat-v3.1",
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
}

PAID_MODELS = {
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "claude-haiku": "anthropic/claude-3.5-haiku",
    "claude-opus": "anthropic/claude-opus-4",
    "gpt-4o": "openai/gpt-4o",
}

MODELS = {**FREE_MODELS, **PAID_MODELS}
DEFAULT_MODEL = next(iter(FREE_MODELS.values()))

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    return _client


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    system: str | None = None,
    verbose: bool = True,
) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = get_client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )

    if verbose:
        print(f"[Model: {response.model}]")

    return response.choices[0].message.content


def call_llm_structured(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    system: str | None = None,
    verbose: bool = True,
) -> Any:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = get_client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )

    if verbose:
        print(f"[Model: {response.model}]")

    return response


def call_llm_json(prompt: str, verbose: bool = True, **kwargs: Any) -> Any:
    system = kwargs.pop("system", "")
    system = system + " Respond with valid JSON only, no markdown code blocks."

    response = call_llm(prompt, system=system.strip(), verbose=verbose, **kwargs)

    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    return json.loads(text.strip())
