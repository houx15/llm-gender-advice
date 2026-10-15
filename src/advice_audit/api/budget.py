"""Budget estimation and hard-cap enforcement (spec §5).

The ledger is rebuilt from the append-only call logs every time (never from memory).
Effective limit for a run = min(phase cap - phase spent, total cap - reserve - total spent).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.config import RUNS_DIR, ApiModelCfg, BudgetCfg, load_budget

LEDGER_COLS = ["run_id", "call_id", "phase", "model_key", "cost_usd", "request_ts"]


def price_for(cfg: ApiModelCfg, peak: bool, live: tuple[float, float] | None = None) -> tuple[float, float]:
    """Budget price per Mtok: max(config, live endpoint price), with the peak multiplier."""
    pin, pout = cfg.prices(peak)
    if live:
        m = cfg.peak_multiplier if peak else 1.0
        pin, pout = max(pin, live[0] * m), max(pout, live[1] * m)
    return pin, pout


def estimate_cost(
    n_calls: int, est_in: float, est_out: float, price_in: float, price_out: float, safety: float = 1.0
) -> float:
    return n_calls * (est_in * price_in + est_out * safety * price_out) / 1e6


class Ledger:
    def __init__(self, runs_dir: Path | None = None):
        self.runs_dir = Path(runs_dir or RUNS_DIR)

    def load(self) -> pd.DataFrame:
        parts = sorted(self.runs_dir.glob("**/logs/*.parquet")) if self.runs_dir.exists() else []
        frames = []
        for p in parts:
            df = pd.read_parquet(p)
            cols = [c for c in LEDGER_COLS if c in df.columns]
            frames.append(df[cols])
        if not frames:
            return pd.DataFrame(columns=LEDGER_COLS)
        df = pd.concat(frames, ignore_index=True)
        df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
        return df

    def spent_total(self) -> float:
        return float(self.load()["cost_usd"].sum())

    def spent_by(self, col: str) -> dict[str, float]:
        df = self.load()
        if df.empty:
            return {}
        return df.groupby(col)["cost_usd"].sum().round(4).to_dict()


class BudgetGuard:
    def __init__(self, cfg: BudgetCfg | None = None, ledger: Ledger | None = None):
        self.cfg = cfg or load_budget()
        self.ledger = ledger or Ledger()

    def remaining(self, phase: str | None, allow_reserve: bool = False) -> tuple[float, str]:
        c = self.cfg
        spent = self.ledger.spent_total()
        total_room = c.total_cap_usd - (0.0 if allow_reserve else c.reserve_usd) - spent
        if phase is None:
            return total_room, "total"
        if phase not in c.phase_caps:
            raise SystemExit(f"phase {phase!r} has no cap in config/budget.yaml")
        phase_room = c.phase_caps[phase] - self.ledger.spent_by("phase").get(phase, 0.0)
        return (phase_room, f"phase:{phase}") if phase_room < total_room else (total_room, "total")

    def check(self, estimate: float, phase: str | None, allow_reserve: bool = False) -> tuple[bool, str]:
        room, binding = self.remaining(phase, allow_reserve)
        msg = f"estimate ${estimate:.2f} vs remaining ${room:.2f} (binding: {binding}; spent ${self.ledger.spent_total():.2f})"
        return (estimate <= room), msg

    def kill_switch(self, estimate: float, run_actual: float, room_at_start: float) -> tuple[bool, str]:
        """Stop if actual run spend > estimate * (1 + overrun) or the cap (minus reserve) is reached (spec §5).

        `room_at_start` is `remaining()` evaluated once before the run, so this check is O(1)."""
        overrun = float(self.cfg.kill_switch.get("overrun_fraction", 0.20))
        if estimate > 0 and run_actual > estimate * (1 + overrun):
            return True, f"run actual ${run_actual:.3f} > estimate ${estimate:.3f} x {1 + overrun:.2f}"
        if run_actual >= room_at_start:
            return True, f"cap reached: run spend ${run_actual:.3f} >= room at start ${room_at_start:.3f}"
        return False, ""

    def status(self) -> str:
        c = self.cfg
        spent = self.ledger.spent_total()
        by_phase = self.ledger.spent_by("phase")
        by_model = self.ledger.spent_by("model_key")
        lines = [
            f"total cap ${c.total_cap_usd:.2f} | reserve ${c.reserve_usd:.2f} | spent ${spent:.4f} | "
            f"spendable ${c.total_cap_usd - c.reserve_usd - spent:.4f}",
            "phase caps:",
        ]
        for ph, cap in c.phase_caps.items():
            s = by_phase.get(ph, 0.0)
            lines.append(f"  {ph:<10} ${s:8.4f} / ${cap:6.2f}")
        if by_model:
            lines.append("by model:")
            for mk, s in sorted(by_model.items()):
                lines.append(f"  {mk:<24} ${s:.4f}")
        return "\n".join(lines)
