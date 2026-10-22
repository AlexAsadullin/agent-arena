from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from agent_arena.agents.providers import AnthropicProvider, OpenAICompatProvider, ProviderError
from agent_arena.settings import ProviderSettings

BASE = "http://llm.test"


def ticking() -> object:
    times = iter([1.0, 1.25])
    return lambda: next(times)


def anthropic_settings(**kw: object) -> ProviderSettings:
    return ProviderSettings(
        kind="anthropic", base_url=BASE, api_version="2023-06-01", api_key=SecretStr("k"), effort="low", **kw
    )


@respx.mock
async def test_anthropic_request_and_parse() -> None:
    route = respx.post(f"{BASE}/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "claude-opus-5-5",
                "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": '{"targets": {}}'}],
                "usage": {"input_tokens": 120, "output_tokens": 30},
                "stop_reason": "end_turn",
            },
        )
    )
    async with httpx.AsyncClient() as http:
        provider = AnthropicProvider(anthropic_settings(), "claude-opus-5-5", http, clock=ticking())  # type: ignore[arg-type]
        out = await provider.complete("sys", "user")
    sent = route.calls.last.request
    body = json.loads(sent.content)
    assert sent.headers["x-api-key"] == "k"
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "sys"
    assert body["output_config"] == {"effort": "low"}
    assert "temperature" not in body
    assert out.text == '{"targets": {}}'
    assert (out.tokens_in, out.tokens_out) == (120, 30)
    assert out.latency_ms == pytest.approx(250.0)
    assert not out.refused


@respx.mock
async def test_anthropic_refusal_and_errors() -> None:
    respx.post(f"{BASE}/v1/messages").mock(
        side_effect=[
            httpx.Response(200, json={"content": [], "usage": {"input_tokens": 1, "output_tokens": 0}, "stop_reason": "refusal"}),
            httpx.Response(529, text="overloaded"),
            httpx.Response(200, json={"content": "oops"}),
            httpx.ConnectError("boom"),
        ]
    )
    async with httpx.AsyncClient() as http:
        provider = AnthropicProvider(anthropic_settings(temperature=0.0), "m", http)
        assert (await provider.complete("s", "u")).refused
        for _ in range(3):
            with pytest.raises(ProviderError):
                await provider.complete("s", "u")


@respx.mock
async def test_openai_compat_request_and_parse() -> None:
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-x",
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )
    )
    settings = ProviderSettings(kind="openai_compat", base_url=f"{BASE}/v1/", api_key=SecretStr("sk"), temperature=0.2)
    async with httpx.AsyncClient() as http:
        out = await OpenAICompatProvider(settings, "gpt-x", http).complete("s", "u")
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer sk"
    body = json.loads(req.content)
    assert body["messages"][0] == {"role": "system", "content": "s"}
    assert body["temperature"] == 0.2
    assert (out.text, out.tokens_in, out.tokens_out, out.refused) == ("hi", 7, 3, False)


@respx.mock
async def test_openai_compat_refusal_and_malformed() -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": None, "refusal": "no"}}]}),
            httpx.Response(200, json={"choices": []}),
        ]
    )
    settings = ProviderSettings(kind="openai_compat", base_url=BASE, api_key=SecretStr("sk"))
    async with httpx.AsyncClient() as http:
        provider = OpenAICompatProvider(settings, "m", http)
        assert (await provider.complete("s", "u")).refused
        with pytest.raises(ProviderError):
            await provider.complete("s", "u")


async def test_missing_key_or_url_fails_before_network() -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(ProviderError, match="API key"):
            await OpenAICompatProvider(ProviderSettings(kind="openai_compat", base_url=BASE), "m", http).complete("s", "u")
        with pytest.raises(ProviderError, match="base_url"):
            await AnthropicProvider(ProviderSettings(kind="anthropic", api_key=SecretStr("k")), "m", http).complete("s", "u")
