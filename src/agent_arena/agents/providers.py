"""LLM provider adapters behind one protocol.

* :class:`OpenAICompatProvider` - any endpoint speaking the OpenAI ``/chat/completions`` shape
  (OpenAI, vLLM, llama.cpp server, OpenRouter, ...).
* :class:`AnthropicProvider` - the Anthropic Messages API (``POST /v1/messages``).

Endpoints, API versions, token limits and keys come from :class:`ProviderSettings`; model ids come
from the season config. Both adapters use an injected ``httpx.AsyncClient`` so tests mock them with
``respx`` and never reach the network.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from agent_arena.settings import ProviderSettings

MS_PER_SECOND = 1000.0


class Completion(BaseModel):
    text: str
    model: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    refused: bool = False


class ProviderError(RuntimeError):
    """The provider call failed (transport error, non-2xx status, malformed body)."""


class LLMProvider(Protocol):
    async def complete(self, system: str, user: str) -> Completion: ...


Clock = Callable[[], float]


def _require(settings: ProviderSettings, field: str) -> str:
    value = getattr(settings, field)
    if not value:
        raise ProviderError(f"provider {settings.kind} needs '{field}' in settings")
    return str(value)


class _HttpProvider:
    def __init__(
        self,
        settings: ProviderSettings,
        model: str,
        http: httpx.AsyncClient,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.settings = settings
        self.model = model
        self._http = http
        self._clock = clock

    async def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> tuple[Any, float]:
        started = self._clock()
        try:
            resp = await self._http.post(url, json=body, headers=headers, timeout=self.settings.timeout_s)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.settings.kind} transport error: {exc}") from exc
        latency_ms = (self._clock() - started) * MS_PER_SECOND
        if resp.status_code >= 400:
            raise ProviderError(f"{self.settings.kind} HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json(), latency_ms

    def _api_key(self) -> str:
        key = self.settings.api_key
        if key is None:
            raise ProviderError(f"no API key configured for provider {self.settings.kind}")
        return key.get_secret_value()


class OpenAICompatProvider(_HttpProvider):
    """``POST {base_url}/chat/completions``; usage from ``prompt_tokens``/``completion_tokens``."""

    async def complete(self, system: str, user: str) -> Completion:
        base = _require(self.settings, "base_url").rstrip("/")
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.settings.temperature is not None:
            body["temperature"] = self.settings.temperature
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        data, latency_ms = await self._post(f"{base}/chat/completions", headers, body)
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            usage = data.get("usage") or {}
            refused = bool(choice["message"].get("refusal")) or choice.get("finish_reason") == "content_filter"
            return Completion(
                text=text,
                model=str(data.get("model", self.model)),
                tokens_in=int(usage.get("prompt_tokens", 0)),
                tokens_out=int(usage.get("completion_tokens", 0)),
                latency_ms=latency_ms,
                refused=refused,
            )
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(f"malformed chat completion: {exc}") from exc


class AnthropicProvider(_HttpProvider):
    """``POST {base_url}/v1/messages`` with ``x-api-key`` and ``anthropic-version`` headers.

    Only ``text`` content blocks are read (thinking blocks are skipped). ``stop_reason ==
    "refusal"`` marks the completion as refused. ``effort`` goes in ``output_config`` when set;
    ``temperature`` is sent only when configured, because current models reject sampling params.
    """

    async def complete(self, system: str, user: str) -> Completion:
        base = _require(self.settings, "base_url").rstrip("/")
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.settings.effort is not None:
            body["output_config"] = {"effort": self.settings.effort}
        if self.settings.temperature is not None:
            body["temperature"] = self.settings.temperature
        headers = {
            "x-api-key": self._api_key(),
            "anthropic-version": _require(self.settings, "api_version"),
        }
        data, latency_ms = await self._post(f"{base}/v1/messages", headers, body)
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            usage = data["usage"]
            return Completion(
                text=text,
                model=str(data.get("model", self.model)),
                tokens_in=int(usage.get("input_tokens", 0)),
                tokens_out=int(usage.get("output_tokens", 0)),
                latency_ms=latency_ms,
                refused=data.get("stop_reason") == "refusal",
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(f"malformed Messages API response: {exc}") from exc
