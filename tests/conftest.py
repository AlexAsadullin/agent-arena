from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest

from agent_arena.season import SeasonConfig, load_season
from agent_arena.settings import ArenaSettings

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests never touch the real data root and never read the developer's .env."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("ARENA_SETTINGS_FILE", str(ROOT / "config" / "settings.yaml"))
    monkeypatch.setenv("TRADEQ_DATA_DIR", str(tmp_path / "data"))
    for var in ("ARENA_GATEWAY__ADMIN_KEY", "ARENA_PROVIDERS__ANTHROPIC__API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def settings(tmp_path: Path) -> ArenaSettings:
    return ArenaSettings(
        _env_file=None,
        TRADEQ_DATA_DIR=tmp_path / "data",
        stats={
            "bootstrap_samples": 300,
            "ci_level": 0.95,
            "alpha": 0.05,
            "dsr_pass_threshold": 0.95,
            "bradley_terry_prior": 0.5,
            "bradley_terry_max_iter": 500,
            "bradley_terry_tol": 1e-9,
            "seed": 1,
        },
    )


@pytest.fixture
def season() -> SeasonConfig:
    return load_season(FIXTURES / "tiny_season.yaml")


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client
