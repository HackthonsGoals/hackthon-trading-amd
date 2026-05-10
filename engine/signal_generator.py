from __future__ import annotations

"""
Simulated / proof-of-concept signal generator.
These are NOT real trading signals. They combine 30-day price momentum,
recent news sentiment, and static volatility regimes to emit demo BUY/SELL/HOLD labels
for UI demonstration purposes only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from sentiment.sentiment_inference import SentimentAnalyzer, sentiment_score

_SENTIMENT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment-distilbert"
_analyzer: SentimentAnalyzer | None = None


def _get_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer(_SENTIMENT_MODEL_PATH, prefer_gpu=True)
    return _analyzer


def _signal_for_ticker(symbol: str, analyzer: SentimentAnalyzer | None = None, volatility: str = "MED") -> dict | None:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="30d")

    if len(df) < 4:
        return None

    current_close = float(df["Close"].iloc[-1])
    close_3d_ago = float(df["Close"].iloc[-4])
    ret_3d = (current_close - close_3d_ago) / close_3d_ago

    # ── Fetch latest 5 news headlines ────────────────────────────────────
    raw_news = ticker.news or []
    headlines = []
    for item in raw_news[:5]:
        content = item.get("content", {})
        title = content.get("title") or item.get("title") or ""
        if title:
            headlines.append(title)

    # ── Run headlines through SentimentAnalyzer ──────────────────────────
    avg_sentiment_score = 0.0
    if headlines:
        actual_analyzer = analyzer if analyzer is not None else _get_analyzer()
        predictions = actual_analyzer.predict_batch(headlines)
        signed_scores = [sentiment_score(p) for p in predictions]
        avg_sentiment_score = sum(signed_scores) / len(signed_scores)

    if avg_sentiment_score > 0.05:
        sentiment_label = "POSITIVE"
    elif avg_sentiment_score < -0.05:
        sentiment_label = "NEGATIVE"
    else:
        sentiment_label = "NEUTRAL"

    # ── Momentum signal ──────────────────────────────────────────────────
    if ret_3d > 0.01:
        signal = "BUY"
    elif ret_3d < -0.01:
        signal = "SELL"
    else:
        signal = "HOLD"

    base_confidence = min(0.9, max(0.3, 0.3 + abs(ret_3d) * 10))
    sentiment_nudge = max(min(avg_sentiment_score * 0.1, 0.1), -0.1)
    
    # Apply volatility regime penalty
    if volatility == "HIGH":
        if signal in ["BUY", "SELL"]:
            # Reduce confidence in high volatility and maybe downgrade to HOLD
            base_confidence -= 0.1
            if base_confidence < 0.4:
                signal = "HOLD"

    confidence = min(0.9, max(0.3, base_confidence + sentiment_nudge))

    # ── Price levels ─────────────────────────────────────────────────────
    entry = current_close
    offset = max(entry * 0.01, 0.01)
    if signal == "BUY":
        sl = entry - offset
        target = entry + offset * 1.5
    elif signal == "SELL":
        sl = entry + offset
        target = entry - offset * 1.5
    else:
        sl = entry
        target = entry

    explanation = f"{sentiment_label} sentiment ({avg_sentiment_score:+.2f}) + {volatility} volatility \u2192 {signal} (demo rule)"

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": round(float(confidence), 4),
        "sentiment": sentiment_label,
        "sentiment_score": round(avg_sentiment_score, 4),
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "target": round(target, 2),
        "news_headlines": headlines,
        "volatility_regime": volatility,
        "explanation": explanation,
    }


def generate_signals_for_demo(
    analyzer: SentimentAnalyzer | None = None,
    volatility_by_symbol: dict[str, str] | None = None,
) -> list[dict]:
    """Generate signals for tech sector tickers using live data."""
    TECH_SECTOR_TICKERS = ["AMD", "NVDA", "MSFT", "GOOGL", "META", "TSLA", "INTC", "AAPL"]
    results = []
    volatility_by_symbol = volatility_by_symbol or {}
    for symbol in TECH_SECTOR_TICKERS:
        volatility = volatility_by_symbol.get(symbol, "MED")
        try:
            sig = _signal_for_ticker(symbol, analyzer=analyzer, volatility=volatility)
            if sig is not None:
                results.append(sig)
        except Exception as e:
            print(f"Warning: Could not generate signal for {symbol}: {e}")
            continue
    return results
