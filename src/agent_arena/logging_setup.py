"""structlog configuration for the CLI. Library modules only call ``structlog.get_logger``."""

from __future__ import annotations

import logging
import sys

import structlog

# Third-party loggers that log every request at INFO; the arena issues tens of thousands per season.
_NOISY = ("httpx", "httpcore", "httpx2", "httpcore2", "mcp", "uvicorn", "fastapi")


def configure_logging(level: str) -> None:
    numeric = logging.getLevelName(level.upper())
    logging.basicConfig(stream=sys.stderr, level=numeric, format="%(message)s")
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
