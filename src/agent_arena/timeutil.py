"""UTC time helpers. Wire format is ISO-8601; storage is int64 nanoseconds since the epoch."""

from __future__ import annotations

from datetime import UTC, datetime

NS_PER_SECOND = 1_000_000_000
NS_PER_MICROSECOND = 1_000


def to_ns(ts: datetime) -> int:
    """Nanoseconds since the epoch for a timezone-aware datetime (microsecond precision)."""
    if ts.tzinfo is None:
        raise ValueError("naive datetime; all times must be timezone-aware UTC")
    delta = ts.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    micros = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return micros * NS_PER_MICROSECOND


def from_ns(ns: int) -> datetime:
    seconds, rem = divmod(ns, NS_PER_SECOND)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=rem // NS_PER_MICROSECOND)


def iso(ts: datetime) -> str:
    """ISO-8601 UTC string with a ``Z`` suffix."""
    return ts.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {value}")
    return parsed.astimezone(UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)
