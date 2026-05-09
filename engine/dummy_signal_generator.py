from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def generate_dummy_signals(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    sentiment_by_symbol: dict[str, dict] | None = None,
) -> list[dict]:
    symbol = "AMD"
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="30d")
    
    if len(df) < 4:
        return []
        
    current_close = float(df['Close'].iloc[-1])
    close_3d_ago = float(df['Close'].iloc[-4])
    
    ret_3d = (current_close - close_3d_ago) / close_3d_ago
    
    if ret_3d > 0.01:
        signal = "BUY"
    elif ret_3d < -0.01:
        signal = "SELL"
    else:
        signal = "HOLD"
        
    confidence = min(0.9, max(0.3, 0.3 + abs(ret_3d) * 10))
    
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

    sentiment_data = (sentiment_by_symbol or {}).get(symbol, {})
    sentiment_label = sentiment_data.get("sentiment", "NEUTRAL")
    sentiment_score = float(sentiment_data.get("signed_score", 0.0))

    return [{
        "symbol": symbol,
        "signal": signal,
        "confidence": round(float(confidence), 4),
        "sentiment": sentiment_label,
        "sentiment_score": round(sentiment_score, 4),
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "target": round(target, 2),
    }]
