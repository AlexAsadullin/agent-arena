"""Trading fee schedules per venue, shared by the arena's accounting and its fake gateway.

``fees.venues.<venue>`` gives proportional rates on trade value per fill; the MOEX venues use the T-Invest
basic tariff (0.25%, maker and taker alike). A schedule may name a zero-fee set (T-Bank's own funds, the
T-Capital ETFs) by ticker or instrument uid in config, extended by I1's reference file
``market-tape-recorder/reference/tbank_zero_fee_funds.json`` when it exists (top-level ``funds``, each with
``ticker`` and ``instrument_uid``). A missing file is tolerated: the config list is used and a warning is
logged. The arena's symbols are ``<SECID>-RUB-<BOARD>``, so it matches funds by ticker (the SECID).

A venue without a schedule has no rate here (``rate`` returns ``None``) and the caller uses its own
fallback: ``accounting.fallback_fee_bps`` for the arena, ``market.fee_bps`` of the season for the fake.
Rates are ``Decimal`` so that fees are exact in the arena's Decimal ledger; a schedule's ``rounding`` rounds
each fill's fee half-up (to the kopeck on MOEX, as I9's mirror sandbox does).
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

import structlog

from agent_arena.settings import ArenaSettings, FeeSettings, ZeroFeeFundsSettings

log = structlog.get_logger(__name__)

Liquidity = Literal["maker", "taker"]


def load_zero_fee_set(cfg: ZeroFeeFundsSettings, data_root: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Tickers and instrument uids of the zero-fee set: config plus the reference file when it exists."""
    tickers = set(cfg.tickers)
    uids = set(cfg.instrument_uids)
    if cfg.reference_file is not None:
        path = data_root / cfg.reference_file
        if path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            funds = doc.get("funds") if isinstance(doc, dict) else None
            if not isinstance(funds, list):
                raise ValueError(f"{path}: expected a top-level 'funds' list")
            for fund in funds:
                if fund.get("ticker"):
                    tickers.add(str(fund["ticker"]))
                if fund.get("instrument_uid"):
                    uids.add(str(fund["instrument_uid"]))
            log.info("zero_fee_reference_loaded", path=str(path), funds=len(funds), source=doc.get("source"))
        else:
            log.warning("zero_fee_reference_missing", path=str(path), fallback_tickers=sorted(cfg.tickers))
    return frozenset(tickers), frozenset(uids)


class FeeBook:
    """Resolves the fee rate of one fill from the venue schedules in settings."""

    def __init__(self, cfg: FeeSettings, data_root: Path) -> None:
        self._cfg = cfg
        self._data_root = data_root
        self._zero_fee: dict[str, tuple[frozenset[str], frozenset[str]]] = {}

    @classmethod
    def from_settings(cls, settings: ArenaSettings) -> FeeBook:
        return cls(settings.fees, settings.data_root)

    def has_schedule(self, venue: str) -> bool:
        return venue in self._cfg.venues

    def _is_zero_fee(self, venue: str, symbol: str, instrument_uid: str | None) -> bool:
        funds = self._cfg.venues[venue].zero_fee_funds
        if funds is None:
            return False
        if venue not in self._zero_fee:
            self._zero_fee[venue] = load_zero_fee_set(funds, self._data_root)
        tickers, uids = self._zero_fee[venue]
        return symbol.split("-", 1)[0] in tickers or (instrument_uid is not None and instrument_uid in uids)

    def rate(self, venue: str, symbol: str, liquidity: Liquidity, instrument_uid: str | None = None) -> Decimal | None:
        """Fee as a share of trade value, or ``None`` when the venue has no schedule."""
        schedule = self._cfg.venues.get(venue)
        if schedule is None:
            return None
        if self._is_zero_fee(venue, symbol, instrument_uid) and schedule.zero_fee_funds is not None:
            funds = schedule.zero_fee_funds
            return Decimal(repr(funds.maker_fee if liquidity == "maker" else funds.taker_fee))
        return Decimal(repr(schedule.maker_fee if liquidity == "maker" else schedule.taker_fee))

    def fee(
        self, venue: str, symbol: str, qty: float, price: float, liquidity: Liquidity, fallback_bps: float
    ) -> Decimal:
        """Fee of one fill on ``|qty * price|``: the venue schedule (rounded half-up to its ``rounding``), else
        ``fallback_bps``. The notional is the exact decimal product, so the rounding matches I9's mirror."""
        notional = abs(Decimal(repr(float(qty))) * Decimal(repr(float(price))))
        rate = self.rate(venue, symbol, liquidity)
        if rate is None:
            return notional * Decimal(repr(float(fallback_bps))) / Decimal(10_000)
        fee = notional * rate
        step = self._cfg.venues[venue].rounding
        if step is None:
            return fee
        return fee.quantize(Decimal(repr(step)), rounding=ROUND_HALF_UP)
