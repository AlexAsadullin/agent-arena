"""Build arena agents from season config, settings and the prompt library."""

from __future__ import annotations

import httpx

from agent_arena.agents.base import Agent
from agent_arena.agents.baselines import BaselineAgent
from agent_arena.agents.llm import LLMAgent, PromptLibrary
from agent_arena.agents.providers import AnthropicProvider, LLMProvider, OpenAICompatProvider
from agent_arena.agents.scripted import ScriptedProvider
from agent_arena.season import AgentConfig, BaselineAgentConfig, LLMAgentConfig, SeasonConfig
from agent_arena.settings import ArenaSettings


class AgentFactory:
    """Creates a fresh agent per run, so no state leaks between runs."""

    def __init__(self, settings: ArenaSettings, prompts: PromptLibrary, http: httpx.AsyncClient) -> None:
        self.settings = settings
        self.prompts = prompts
        self.http = http

    def validate(self, season: SeasonConfig) -> None:
        """Fail before any run starts if a season references missing providers, prices or prompts."""
        for name in season.prompt_variants:
            self.prompts.get(name)
        for agent in season.llm_agents:
            if agent.provider not in self.settings.providers:
                raise ValueError(f"agent {agent.id}: provider {agent.provider!r} not in settings")
            if agent.model not in self.settings.pricing:
                raise ValueError(f"agent {agent.id}: model {agent.model!r} has no price in settings.pricing")

    def provider(self, config: LLMAgentConfig, seed: int) -> LLMProvider:
        provider_settings = self.settings.providers[config.provider]
        if provider_settings.kind == "scripted":
            if config.behaviour is None:
                raise ValueError(f"scripted agent {config.id} needs a behaviour")
            return ScriptedProvider(config.model, config.behaviour, config.params, seed)
        if provider_settings.kind == "anthropic":
            return AnthropicProvider(provider_settings, config.model, self.http)
        return OpenAICompatProvider(provider_settings, config.model, self.http)

    def build(self, config: AgentConfig, variant: str | None, seed: int) -> Agent:
        if isinstance(config, BaselineAgentConfig):
            return BaselineAgent(config, seed)
        if variant is None:
            raise ValueError(f"LLM agent {config.id} needs a prompt variant")
        return LLMAgent(
            config.id,
            self.provider(config, seed),
            self.prompts.get(variant),
            self.prompts.refusal_patterns,
        )
