"""Serve the real I9 gateway over HTTP with its own in-process fakes of the downstream services.

This script runs under the I9 repository's own interpreter (``risk-gated-mcp-gateway/.venv/bin/python``) with
that repository as the working directory; agent-arena never imports I9 code. I9's ``gateway serve`` has no
offline switch, so this wires the same pieces that ``gateway stdio --offline`` uses (the fake platform and a
manual wall clock) into ``build_http_app``. Host, port, data root and admin key come from the GATEWAY_* and
TRADEQ_DATA_DIR environment variables that the arena test sets.
"""

from __future__ import annotations

import uvicorn
from mcp_gateway.app import build_gateway, build_http_app
from mcp_gateway.fakes.services import FakePlatform
from mcp_gateway.fakes.world import world_from_settings
from mcp_gateway.settings import load_settings
from mcp_gateway.timeutil import ManualClock, parse_iso


def main() -> None:
    settings = load_settings()
    clock = ManualClock(parse_iso(settings.demo.wall_clock))
    platform = FakePlatform(world=world_from_settings(settings.fakes), cfg=settings.fakes, clock=clock)
    gateway = build_gateway(settings, clock=clock, transports=platform.transports())
    http = build_http_app(gateway)
    uvicorn.run(http.app, host=settings.server.host, port=settings.server.port, log_level="warning")


if __name__ == "__main__":
    main()
