from __future__ import annotations

import pytest

from agent_arena.runner import round_to_lots


def test_round_to_lots_rounds_toward_zero() -> None:
    assert round_to_lots(37.9, 10.0) == 30.0
    assert round_to_lots(-37.9, 10.0) == -30.0
    assert round_to_lots(9.99, 10.0) == 0.0
    assert round_to_lots(30.0, 10.0) == 30.0  # exact multiples survive float noise
    assert round_to_lots(2.5, None) == pytest.approx(2.5)  # no lot configured: unchanged
