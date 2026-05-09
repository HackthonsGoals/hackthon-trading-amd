from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard.ui import render_dashboard
from engine.ai_inference import run_batch_inference
from engine.dummy_signal_generator import generate_dummy_signals
from sentiment.benchmark import benchmark_sentiment
from sentiment.sentiment_inference import SentimentAnalyzer, sentiment_score
from simulator.execution_simulator import ExecutionSimulator
from utils.benchmark import benchmark_inference
from utils.data_loader import load_headlines, load_market_data


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_ohlcv.csv"
HEADLINES_PATH = BASE_DIR / "data" / "sample_headlines.csv"
SIGNALS_PATH = BASE_DIR / "data" / "sample_signals.json"
TRADES_PATH = BASE_DIR / "data" / "simulated_trades.json"
SENTIMENT_MODEL_PATH = BASE_DIR / "models" / "sentiment-distilbert"


def _compute_volatility_regimes(market_data: pd.DataFrame) -> dict[str, str]:
    """Compute basic volatility regime (LOW/MED/HIGH) based on rolling standard deviation of returns."""
    regimes = {}
    for symbol in market_data["symbol"].unique():
        symbol_data = market_data[market_data["symbol"] == symbol].copy()
        symbol_data["returns"] = symbol_data["close"].pct_change()
        volatility = symbol_data["returns"].rolling(window=10, min_periods=1).std().iloc[-1]
        
        # Simple static thresholds for demo purposes
        if pd.isna(volatility) or volatility < 0.015:
            regimes[symbol] = "LOW"
        elif volatility < 0.03:
            regimes[symbol] = "MED"
        else:
            regimes[symbol] = "HIGH"
    return regimes


def main() -> None:
    st.set_page_config(page_title="AMD AI Trading Demo", layout="wide")
    st_autorefresh(interval=60000, key="data_refresh")
    
    st.sidebar.header("Lab Controls")
    model_choice = st.sidebar.selectbox("Sentiment model", ["Fine-tuned", "Baseline"])
    active_model_path = SENTIMENT_MODEL_PATH if model_choice == "Fine-tuned" else "distilbert-base-uncased"
    
    max_batch_size = st.sidebar.select_slider(
        "Max Batch Size (Benchmark)",
        options=[50, 100, 250, 500, 1000],
        value=500
    )
    
    market_data = load_market_data(DATA_PATH)
    headlines = load_headlines(HEADLINES_PATH)
    volatility_by_symbol = _compute_volatility_regimes(market_data)

    sentiment_analyzer = SentimentAnalyzer(active_model_path, prefer_gpu=True)
    sentiment_predictions = sentiment_analyzer.predict_batch(headlines["text"].tolist())
    sentiment_records = []
    for headline, prediction in zip(headlines.to_dict("records"), sentiment_predictions):
        record = {**headline, **prediction}
        record["signed_score"] = sentiment_score(prediction)
        sentiment_records.append(record)

    sentiment_by_symbol = {}
    for record in sentiment_records:
        symbol = str(record["symbol"])
        sentiment_by_symbol.setdefault(symbol, []).append(record["signed_score"])
    sentiment_by_symbol = {
        symbol: {
            "sentiment": "POSITIVE" if sum(scores) > 0.15 else "NEGATIVE" if sum(scores) < -0.15 else "NEUTRAL",
            "signed_score": sum(scores) / len(scores),
        }
        for symbol, scores in sentiment_by_symbol.items()
    }

    inference = run_batch_inference(market_data, prefer_gpu=True)
    signals = generate_dummy_signals(
        market_data, 
        inference.probabilities, 
        sentiment_by_symbol, 
        analyzer=sentiment_analyzer,
        volatility_by_symbol=volatility_by_symbol
    )
    from engine.llm_reasoner import explain_signals_batch
    signals = explain_signals_batch(signals)

    SIGNALS_PATH.write_text(json.dumps(signals, indent=2), encoding="utf-8")

    simulator = ExecutionSimulator(TRADES_PATH)
    trades = simulator.execute(signals)
    trade_summary = simulator.summary()
    metrics = {
        "device": inference.device,
        "latency_ms": inference.latency_ms,
        "throughput_rows_per_second": inference.throughput_rows_per_second,
    }
    
    batch_sizes_to_run = [bs for bs in [50, 100, 250, 500, 1000] if bs <= max_batch_size]
    if not batch_sizes_to_run:
        batch_sizes_to_run = [max_batch_size]
        
    benchmark = benchmark_inference(market_data, batch_sizes=tuple(batch_sizes_to_run))
    sentiment_benchmark = benchmark_sentiment(headlines["text"].tolist(), active_model_path)
    
    render_dashboard(
        market_data,
        signals,
        trades,
        metrics,
        benchmark,
        headlines,
        sentiment_records,
        sentiment_benchmark,
        trade_summary,
        volatility_by_symbol=volatility_by_symbol,
    )


if __name__ == "__main__":
    main()
