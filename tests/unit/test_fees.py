"""Fee schedules: the MOEX T-Invest tariff, T-Bank's zero-fee funds, and venues without a schedule."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from agent_arena.fees import FeeBook, load_zero_fee_set
from agent_arena.settings import ArenaSettings

REFERENCE = "market-tape-recorder/reference/tbank_zero_fee_funds.json"


def test_config_rates(settings: ArenaSettings) -> None:
    for venue in ("moex-tbank", "moex-algopack", "moex-iss"):
        sched = settings.fees.venues[venue]
        assert (sched.maker_fee, sched.taker_fee) == (0.0025, 0.0025)
        assert sched.source == "T-Invest basic tariff per user, 2026-11-05"
        assert sched.zero_fee_funds is not None
        assert (sched.zero_fee_funds.maker_fee, sched.zero_fee_funds.taker_fee) == (0.0, 0.0)


def test_moex_tariff_zero_fee_funds_and_fallback(settings: ArenaSettings) -> None:
    book = FeeBook.from_settings(settings)
    assert book.rate("moex-tbank", "SBER-RUB-TQBR", "taker") == Decimal("0.0025")
    assert book.rate("moex-tbank", "GAZP-RUB-TQBR", "maker") == Decimal("0.0025")
    assert book.fee("moex-tbank", "SBER-RUB-TQBR", -100, 285.0, "taker", 5.0) == Decimal("71.25")
    # A config-listed T-Capital fund pays nothing even when I1's reference file is absent.
    assert book.fee("moex-tbank", "TMOS-RUB-TQTF", 1000, 10.0, "taker", 5.0) == 0
    assert book.rate("SYNTH", "BTC-USDT-PERP", "taker") is None
    assert book.fee("SYNTH", "BTC-USDT-PERP", 0.5, 20_000.0, "taker", 5.0) == Decimal("5")


def test_reference_file_extends_the_set(tmp_path: Path, settings: ArenaSettings) -> None:
    path = settings.data_root / REFERENCE
    path.parent.mkdir(parents=True)
    fund = {"ticker": "TXYZ", "figi": "F", "instrument_uid": "uid-txyz", "class_code": "TQTF", "lot": 1}
    path.write_text(json.dumps({"source": "t", "retrieved_at": "2026-11-05", "rule": "r", "funds": [fund]}))
    book = FeeBook.from_settings(settings)
    assert book.rate("moex-tbank", "TXYZ-RUB-TQTF", "taker") == 0
    assert book.rate("moex-tbank", "XYZ-RUB-TQTF", "taker", instrument_uid="uid-txyz") == 0
    assert book.rate("moex-tbank", "SBER-RUB-TQBR", "taker") == Decimal("0.0025")


def test_absent_reference_falls_back_and_bad_reference_fails(settings: ArenaSettings, tmp_path: Path) -> None:
    funds = settings.fees.venues["moex-tbank"].zero_fee_funds
    assert funds is not None
    tickers, _ = load_zero_fee_set(funds, tmp_path / "nowhere")
    assert tickers == frozenset(funds.tickers)
    bad = tmp_path / "bad" / REFERENCE
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps({"funds": {}}))
    with pytest.raises(ValueError, match="funds"):
        load_zero_fee_set(funds, tmp_path / "bad")


def test_moex_fee_is_rounded_half_up_to_the_kopeck(settings: ArenaSettings) -> None:
    book = FeeBook.from_settings(settings)
    assert book.fee("moex-tbank", "SBER-RUB-TQBR", 1, 285.0, "taker", 5.0) == Decimal("0.71")  # 0.7125
    assert book.fee("moex-tbank", "SBER-RUB-TQBR", 1, 282.1, "taker", 5.0) == Decimal("0.71")  # 0.70525
    assert book.fee("moex-tbank", "GAZP-RUB-TQBR", 10, 130.0, "maker", 5.0) == Decimal("3.25")
    # The exact decimal product decides the half-up case (as I9 computes it; 977.925 rounds up).
    assert book.fee("moex-tbank", "SBER-RUB-TQBR", 1000, 391.17, "taker", 5.0) == Decimal("977.93")
