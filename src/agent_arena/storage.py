"""Season artefacts under ``$TRADEQ_DATA_DIR/agent-arena/seasons/<season_id>/``.

* ``runs.jsonl`` - one :class:`RunResult` per line (full step records);
* ``steps.parquet`` - flat step table (one row per run x step) for downstream readers such as I8;
* ``meta.json`` - where trials were logged and the leakage status of every LLM agent;
* ``leaderboard.json`` / ``leaderboard.md`` / ``leaderboard.html`` - the leaderboard artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from agent_arena.clients.leakage import LeakageResult
from agent_arena.leaderboard import Leaderboard
from agent_arena.results import RunResult
from agent_arena.timeutil import parse_iso, to_ns


class SeasonMeta(BaseModel):
    season_id: str
    trials_destination: str
    leakage: dict[str, LeakageResult] = Field(default_factory=dict)


class SeasonStore:
    def __init__(self, arena_dir: Path, season_id: str) -> None:
        self.dir = arena_dir / "seasons" / season_id
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def runs_path(self) -> Path:
        return self.dir / "runs.jsonl"

    def save_runs(self, runs: list[RunResult]) -> None:
        with self.runs_path.open("w", encoding="utf-8") as fh:
            for run in runs:
                fh.write(run.model_dump_json() + "\n")
        self._save_steps(runs)

    def load_runs(self) -> list[RunResult]:
        if not self.runs_path.exists():
            raise FileNotFoundError(f"no runs stored for this season at {self.runs_path}")
        with self.runs_path.open(encoding="utf-8") as fh:
            return [RunResult.model_validate_json(line) for line in fh if line.strip()]

    def _save_steps(self, runs: list[RunResult]) -> None:
        rows: dict[str, list[object]] = {
            k: []
            for k in (
                "run_id",
                "agent_id",
                "variant",
                "episode",
                "seed",
                "step",
                "ts_decision",
                "ret_net",
                "ret_gross",
                "fees",
                "token_cost",
                "latency_cost",
                "turnover",
                "n_orders",
                "n_rejected",
                "status",
                "tokens_in",
                "tokens_out",
                "latency_ms",
                "frozen",
            )
        }
        for run in runs:
            for s in run.steps:
                rows["run_id"].append(run.run_id)
                rows["agent_id"].append(run.agent_id)
                rows["variant"].append(run.variant)
                rows["episode"].append(run.episode)
                rows["seed"].append(run.seed)
                rows["step"].append(s.step)
                rows["ts_decision"].append(to_ns(parse_iso(s.ts)))
                for key in (
                    "ret_net",
                    "ret_gross",
                    "fees",
                    "token_cost",
                    "latency_cost",
                    "turnover",
                    "n_orders",
                    "n_rejected",
                    "status",
                    "tokens_in",
                    "tokens_out",
                    "latency_ms",
                    "frozen",
                ):
                    rows[key].append(getattr(s, key))
        pq.write_table(pa.table(rows), self.dir / "steps.parquet")

    def save_meta(self, meta: SeasonMeta) -> None:
        (self.dir / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    def load_meta(self) -> SeasonMeta:
        return SeasonMeta.model_validate_json((self.dir / "meta.json").read_text(encoding="utf-8"))

    def save_leaderboard(self, board: Leaderboard, markdown: str, html: str) -> None:
        (self.dir / "leaderboard.json").write_text(board.model_dump_json(indent=2), encoding="utf-8")
        (self.dir / "leaderboard.md").write_text(markdown, encoding="utf-8")
        (self.dir / "leaderboard.html").write_text(html, encoding="utf-8")

    def load_leaderboard(self) -> Leaderboard:
        return Leaderboard.model_validate(json.loads((self.dir / "leaderboard.json").read_text(encoding="utf-8")))
