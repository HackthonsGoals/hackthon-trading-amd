from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class SimulatedTrade:
    trade_id: str
    timestamp: str
    opened_at: str
    closed_at: str
    symbol: str
    side: str
    entry: float
    exit: float
    quantity: int
    slippage_bps: float
    pnl: float
    return_pct: float
    status: str


class ExecutionSimulator:
    """Fake execution engine with no external execution or credential path."""

    def __init__(self, trade_log_path: str | Path | None = None, seed: int = 7) -> None:
        self.trade_log_path = Path(trade_log_path) if trade_log_path else None
        self.random = random.Random(seed)
        self.trades: list[dict] = []

    def execute(self, signals: list[dict], quantity: int = 10, slippage_bps: float = 5.0) -> list[dict]:
        new_trades: list[dict] = []
        for index, signal in enumerate(signals, start=1):
            side = signal["signal"]
            if side == "HOLD":
                continue

            opened_at = datetime.utcnow() + timedelta(seconds=index * 3)
            closed_at = opened_at + timedelta(minutes=self.random.randint(4, 18))
            entry = float(signal["entry"])
            slip = entry * (slippage_bps / 10_000.0)
            direction = 1 if side == "BUY" else -1
            fill_entry = entry + slip * direction
            # ── UPGRADED: Sentiment-aware drift ──────────────────────────────
            sentiment_score = signal.get("sentiment_score", 0.0)  # range: -1 to +1
            sentiment_bias = sentiment_score * 0.004 * entry  # bias exit in sentiment direction
            noise = self.random.gauss(0, 0.003) * entry  # stochastic component
            drift = (sentiment_bias * direction) + noise
            # ─────────────────────────────────────────────────────────────────
            exit_price = fill_entry + drift
            pnl = (exit_price - fill_entry) * quantity * direction
            notional = max(fill_entry * quantity, 1e-9)

            trade = SimulatedTrade(
                trade_id=f"SIM-{opened_at.strftime('%H%M%S')}-{index:03d}",
                timestamp=closed_at.isoformat(timespec="seconds") + "Z",
                opened_at=opened_at.isoformat(timespec="seconds") + "Z",
                closed_at=closed_at.isoformat(timespec="seconds") + "Z",
                symbol=signal["symbol"],
                side=side,
                entry=round(fill_entry, 2),
                exit=round(exit_price, 2),
                quantity=quantity,
                slippage_bps=slippage_bps,
                pnl=round(pnl, 2),
                return_pct=round((pnl / notional) * 100.0, 4),
                status="CLOSED",
            )
            new_trades.append(asdict(trade))

        self.trades.extend(new_trades)
        self._persist()
        return new_trades

    def _persist(self) -> None:
        if not self.trade_log_path:
            return
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.trade_log_path.write_text(json.dumps(self.trades, indent=2), encoding="utf-8")

    def summary(self) -> dict:
        pnl = sum(float(trade["pnl"]) for trade in self.trades)
        wins = sum(1 for trade in self.trades if float(trade["pnl"]) > 0)
        count = len(self.trades)
        return {
            "trades": count,
            "total_pnl": round(pnl, 2),
            "win_rate": round(wins / count, 3) if count else 0.0,
            "avg_return_pct": round(
                sum(float(trade.get("return_pct", 0.0)) for trade in self.trades) / count,
                4,
            )
            if count
            else 0.0,
        }
