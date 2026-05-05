from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class DemoSignal:
    symbol: str
    signal: str
    entry: float
    sl: float
    target: float
    confidence: float
    sentiment: str
    sentiment_score: float
    sentiment_adjustment: float


def generate_dummy_signals(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    sentiment_by_symbol: dict[str, dict] | None = None,
) -> list[dict]:
    """Create schema-compatible fake signals with a tiny transparent sentiment nudge."""

    latest = (
        frame.sort_values("timestamp")
        .groupby("symbol", as_index=False, group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )
    probs = probabilities[-len(latest) :]
    signals: list[dict] = []

    for row, probability in zip(latest.itertuples(index=False), probs):
        sentiment = (sentiment_by_symbol or {}).get(str(row.symbol), {})
        sentiment_label = sentiment.get("sentiment", "NEUTRAL")
        raw_sentiment_score = float(sentiment.get("signed_score", 0.0))
        sentiment_adjustment = max(min(raw_sentiment_score * 0.12, 0.12), -0.12)

        adjusted = np.array(probability, dtype=np.float32).copy()
        adjusted[2] += max(sentiment_adjustment, 0.0)
        adjusted[0] += max(-sentiment_adjustment, 0.0)
        adjusted[1] += max(0.04 - abs(sentiment_adjustment), 0.0)
        adjusted = adjusted / adjusted.sum()

        label_index = int(np.argmax(adjusted))
        action = ["SELL", "HOLD", "BUY"][label_index]
        confidence = float(np.max(adjusted))
        entry = float(row.close)
        offset = max(entry * 0.01, 0.01)

        if action == "BUY":
            sl = entry - offset
            target = entry + offset * 1.5
        elif action == "SELL":
            sl = entry + offset
            target = entry - offset * 1.5
        else:
            sl = entry
            target = entry

        signals.append(
            asdict(
                DemoSignal(
                    symbol=str(row.symbol),
                    signal=action,
                    entry=round(entry, 2),
                    sl=round(sl, 2),
                    target=round(target, 2),
                    confidence=round(confidence, 4),
                    sentiment=sentiment_label,
                    sentiment_score=round(raw_sentiment_score, 4),
                    sentiment_adjustment=round(sentiment_adjustment, 4),
                )
            )
        )

    return signals
