"""Opt-in: run a small season through the real I9 gateway (``make integration`` or ``pytest -m integration``).

The test starts I9 from its own repository and virtualenv as a subprocess on a free loopback port with
``gateway serve --offline``, which wires I9's in-process fakes of its downstream services. I9 gets a scratch
TRADEQ_DATA_DIR under this repository's gitignored ``tmp/``. The admin key comes from
ARENA_GATEWAY__ADMIN_KEY; the I9 checkout from ARENA_I9_REPO (default: the sibling directory).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from agent_arena.gateway.client import http_gateway_factory
from agent_arena.pipeline import run_season_pipeline
from agent_arena.season import load_season
from agent_arena.settings import ArenaSettings

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SEASON = ROOT / "config" / "seasons" / "i9-offline.yaml"
STARTUP_TIMEOUT_S = 60.0
POLL_S = 0.25


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def i9() -> Iterator[tuple[str, str, Path]]:
    key = os.environ.get("ARENA_GATEWAY__ADMIN_KEY")
    if not key:
        pytest.skip("set ARENA_GATEWAY__ADMIN_KEY (make integration does)")
    repo = Path(os.environ.get("ARENA_I9_REPO", ROOT.parent / "risk-gated-mcp-gateway"))
    gateway_cli = repo / ".venv" / "bin" / "gateway"
    if not gateway_cli.exists():
        pytest.skip(f"I9 CLI not found at {gateway_cli}")
    scratch = ROOT / "tmp" / "i9-integration" / uuid.uuid4().hex[:8]
    scratch.mkdir(parents=True)
    port = _free_port()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "TRADEQ_DATA_DIR": str(scratch / "i9-data"),
        "GATEWAY_ADMIN_API_KEY": key,
        "GATEWAY_SERVER__HOST": "127.0.0.1",
        "GATEWAY_SERVER__PORT": str(port),
    }
    log = (scratch / "i9.log").open("w")
    # cwd is the scratch dir so I9 does not pick up its own .env; it finds its settings next to its code.
    proc = subprocess.Popen(
        [str(gateway_cli), "serve", "--offline"], cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    try:
        while True:
            if proc.poll() is not None:
                pytest.fail(f"I9 exited with {proc.returncode}; see {scratch / 'i9.log'}")
            try:
                if httpx.get(f"{url}/healthz", timeout=1.0).status_code == httpx.codes.OK:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"I9 did not start within {STARTUP_TIMEOUT_S} s; see {scratch / 'i9.log'}")
            time.sleep(POLL_S)
        yield url, key, scratch
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


async def test_small_season_through_real_i9(i9: tuple[str, str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    url, key, scratch = i9
    monkeypatch.setenv("TRADEQ_DATA_DIR", str(scratch / "arena-data"))
    unreachable = "http://127.0.0.1:9"
    settings = ArenaSettings(
        _env_file=None,
        TRADEQ_DATA_DIR=scratch / "arena-data",
        ledger={**ArenaSettings(_env_file=None).ledger.model_dump(), "base_url": unreachable},
        leakage={**ArenaSettings(_env_file=None).leakage.model_dump(), "base_url": unreachable},
    )
    gateway = settings.gateway.at(url).model_copy(update={"admin_key": SecretStr(key)})
    season = load_season(SEASON)
    async with httpx.AsyncClient() as http:
        outcome = await run_season_pipeline(season, settings, http_gateway_factory(gateway), http)

    assert len(outcome.runs) == len(season.agents) * season.episodes * len(season.seeds)
    for run in outcome.runs:
        assert len(run.steps) == season.steps_per_episode
        assert run.frozen_reason is None, run.frozen_reason
    filled = sum(s.n_orders - s.n_rejected for r in outcome.runs for s in r.steps)
    assert filled > 0
    rejects = sorted({x for r in outcome.runs for s in r.steps for x in s.reject_reasons})
    # Only limit hits that the agents caused; no order-rate refusals from a mis-clocked window.
    assert set(rejects) <= {"capital", "order_notional"}, rejects
    # I9 forwards the session clock to its I6 fake, so each fill price equals I9's reference price at that clock.
    audit = [
        json.loads(line) for f in (scratch / "i9-data").rglob("audit*.jsonl") for line in f.read_text().splitlines()
    ]
    fills = [e["outcome"] for e in audit if e.get("action") == "place_order" and e.get("decision") == "allowed"]
    assert len(fills) == filled
    assert all(f["order"]["price"] == pytest.approx(f["reference_price"]) for f in fills)
    llm = [r for r in outcome.runs if r.is_llm]
    assert all(sum(s.token_cost for s in r.steps) > 0 for r in llm)
    assert {row.agent_id for row in outcome.leaderboard.rows} == {a.id for a in season.agents}
    assert (outcome.store.dir / "leaderboard.md").exists()
    summary = scratch / "summary.txt"
    from agent_arena.report.render import summary_lines

    summary.write_text(
        "\n".join([*summary_lines(outcome.leaderboard), f"filled orders: {filled}", f"reject reasons: {rejects}"])
    )
    print(summary.read_text(), file=sys.stderr)
