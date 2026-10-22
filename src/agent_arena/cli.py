"""``arena`` command line: run-season | leaderboard | report."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import httpx
import typer

from agent_arena.gateway.client import GatewayFactory, http_gateway_factory
from agent_arena.gateway.fake import fake_gateway_factory
from agent_arena.logging_setup import configure_logging
from agent_arena.pipeline import rebuild_leaderboard, run_season_pipeline
from agent_arena.report.render import render_html, render_markdown, summary_lines
from agent_arena.season import SeasonConfig, load_season
from agent_arena.settings import ArenaSettings, load_settings
from agent_arena.storage import SeasonStore

app = typer.Typer(help="Agent Arena (S8): statistical leaderboards for LLM trading agents.", no_args_is_help=True)


class GatewayChoice(StrEnum):
    fake = "fake"
    live = "live"


SeasonOption = Annotated[Path, typer.Option("--season", help="Season YAML file.")]


def _settings() -> ArenaSettings:
    settings = load_settings()
    configure_logging(settings.log_level)
    return settings


def _gateway(choice: GatewayChoice, season: SeasonConfig, settings: ArenaSettings, url: str | None) -> GatewayFactory:
    if choice is GatewayChoice.fake:
        return fake_gateway_factory(season, settings.pricing, settings.gateway)
    return http_gateway_factory(settings.gateway if url is None else settings.gateway.at(url))


@app.command("run-season")
def run_season(
    season: SeasonOption,
    gateway: Annotated[GatewayChoice, typer.Option(help="fake: in-process I9 fake; live: the I9 service.")] = (
        GatewayChoice.fake
    ),
    gateway_url: Annotated[
        str | None,
        typer.Option(help="Base URL of a running I9 (admin API and MCP); implies --gateway live."),
    ] = None,
) -> None:
    """Run every agent across episodes, seeds and prompt variants, then publish the leaderboard."""
    settings = _settings()
    config = load_season(season)
    choice = GatewayChoice.live if gateway_url is not None else gateway

    async def main() -> list[str]:
        async with httpx.AsyncClient() as http:
            factory = _gateway(choice, config, settings, gateway_url)
            outcome = await run_season_pipeline(config, settings, factory, http)
        return [
            *summary_lines(outcome.leaderboard),
            f"runs: {len(outcome.runs)}; trials logged to: {outcome.trials_destination}",
            f"artefacts: {outcome.store.dir}",
        ]

    for line in asyncio.run(main()):
        typer.echo(line)


@app.command()
def leaderboard(season: SeasonOption) -> None:
    """Recompute the leaderboard from stored runs."""
    settings = _settings()
    config = load_season(season)

    async def main() -> list[str]:
        async with httpx.AsyncClient() as http:
            return summary_lines(await rebuild_leaderboard(config, settings, http))

    for line in asyncio.run(main()):
        typer.echo(line)


@app.command()
def report(season: SeasonOption) -> None:
    """Re-render the Markdown and HTML leaderboard from the stored leaderboard.json."""
    settings = _settings()
    config = load_season(season)
    store = SeasonStore(settings.arena_dir, config.season_id)
    board = store.load_leaderboard()
    store.save_leaderboard(board, render_markdown(board), render_html(board))
    for line in summary_lines(board):
        typer.echo(line)
    typer.echo(f"written: {store.dir / 'leaderboard.md'} and {store.dir / 'leaderboard.html'}")
