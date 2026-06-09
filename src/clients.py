"""
Thin provider abstraction for LLM API calls.

Supports three providers behind a unified async interface:
  openai     — standard OpenAI API (AsyncOpenAI)
  gemini     — Google Gemini via OpenAI-compatible endpoint (AsyncOpenAI + base_url)
  anthropic  — Anthropic Claude (AsyncAnthropic)

Usage:
  client = make_client("gemini:gemini-3.1-flash-lite", api_key="...")
  client = make_client("openai:gpt-5.4-mini", api_key="...")
  client = make_client("anthropic:claude-sonnet-4-6", api_key="...")

  text = await client.chat(messages, json_mode=True)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ANTHROPIC_MAX_TOKENS = 4096


class LLMClient:
    """Unified async chat interface across OpenAI, Gemini, and Anthropic."""

    def __init__(self, provider: str, model: str, api_key: str) -> None:
        self.provider = provider
        self.model    = model
        self._client  = _build_raw_client(provider, api_key)

    async def chat(
        self,
        messages: list[dict],
        json_mode: bool = False,
    ) -> str:
        """
        Send a chat request and return the response text.

        For OpenAI/Gemini: json_mode adds response_format={"type": "json_object"}.
        For Anthropic:     json_mode is a no-op — JSON is enforced via prompt instructions
                           already present in PHYSICIAN_SYSTEM and EVALUATOR_SYSTEM.
        """
        if self.provider == "anthropic":
            return await self._chat_anthropic(messages)
        else:
            return await self._chat_openai(messages, json_mode)

    async def _chat_openai(self, messages: list[dict], json_mode: bool) -> str:
        kwargs: dict = {"model": self.model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def _chat_anthropic(self, messages: list[dict]) -> str:
        # Anthropic requires the system message as a separate kwarg
        system_content = ""
        user_messages  = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                user_messages.append(m)

        kwargs: dict = {
            "model":      self.model,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "messages":   user_messages,
        }
        if system_content:
            kwargs["system"] = system_content

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text or ""


def _build_raw_client(provider: str, api_key: str):
    """Instantiate the underlying provider SDK client."""
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=api_key)
    elif provider == "gemini":
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=api_key)
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Expected one of: openai, gemini, anthropic."
        )


def make_client(provider_model: str, api_key: str) -> LLMClient:
    """
    Parse a 'provider:model' string and return an LLMClient.

    Examples:
      make_client("gemini:gemini-3.1-flash-lite", api_key=os.environ["GOOGLE_API_KEY"])
      make_client("openai:gpt-5.4-mini",           api_key=os.environ["OPENAI_API_KEY"])
      make_client("anthropic:claude-sonnet-4-6",   api_key=os.environ["ANTHROPIC_API_KEY"])
    """
    if ":" not in provider_model:
        raise ValueError(
            f"Invalid model string '{provider_model}'. "
            "Expected format: 'provider:model-name' (e.g. 'openai:gpt-4o')."
        )
    provider, model = provider_model.split(":", 1)
    return LLMClient(provider=provider, model=model, api_key=api_key)


# Maps each provider to the environment variable that holds its API key.
# Used by run.py to look up the right key automatically.
PROVIDER_ENV_VARS: dict[str, str] = {
    "openai":    "OPENAI_API_KEY",
    "gemini":    "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def provider_of(provider_model: str) -> str:
    """Return the provider portion of a 'provider:model' string."""
    return provider_model.split(":", 1)[0]
