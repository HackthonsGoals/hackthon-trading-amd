from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import os
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Configure Hugging Face environment for stability and rate-limiting
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

# Ensure Transformers respects the local cache once populated
if os.environ.get("TRANSFORMERS_OFFLINE") is None:
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

from dashboard.ui import render_dashboard
from engine.ai_inference import run_batch_inference
# NOTE: Renamed from dummy_signal_generator to signal_generator for a more professional presentation.
from engine.signal_generator import generate_signals
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

# Toggle this to True before final demo. Kept False for local dev so Streamlit doesn't refresh constantly.
ENABLE_AUTO_REFRESH = True  


def _compute_volatility_regimes(market_data: pd.DataFrame) -> dict[str, str]:
    """Compute basic volatility regime (LOW/MED/HIGH) based on rolling standard deviation of returns."""
    regimes = {}
    for symbol in market_data["symbol"].unique():
        symbol_data = market_data[market_data["symbol"] == symbol].copy()
        symbol_data["returns"] = symbol_data["close"].pct_change()
        volatility = symbol_data["returns"].rolling(window=10, min_periods=1).std().iloc[-1]
        
        # Static thresholds for regime classification
        if pd.isna(volatility) or volatility < 0.015:
            regimes[symbol] = "LOW"
        elif volatility < 0.03:
            regimes[symbol] = "MED"
        else:
            regimes[symbol] = "HIGH"
    return regimes


@st.cache_resource(show_spinner="Loading sentiment model...")
def get_sentiment_analyzer(model_path: str, prefer_gpu: bool = True) -> SentimentAnalyzer:
    """Return a cached SentimentAnalyzer instance to avoid reloading weights every rerun."""
    return SentimentAnalyzer(model_path, prefer_gpu=prefer_gpu)


@st.cache_data(ttl=300, show_spinner="Generating signals and LLM explanations...")
def get_cached_signals(volatility_by_symbol: dict[str, str]) -> list[dict]:
    from engine.llm_reasoner import explain_signals_batch
    
    # We omit passing the active sentiment_analyzer to avoid Streamlit hashing errors on the model object.
    # The generator will lazily load its own instance internally.
    raw_signals = generate_signals(
        analyzer=None,
        volatility_by_symbol=volatility_by_symbol
    )
    return explain_signals_batch(raw_signals)


def main() -> None:
    st.set_page_config(page_title="AMD AI Signal Pipeline", layout="wide")
    
    if ENABLE_AUTO_REFRESH:
        st_autorefresh(interval=60000, key="data_refresh")
    
    st.sidebar.header("Experiment Controls")
    st.sidebar.caption(
        "Optimize pipeline throughput and select specialized sentiment kernels."
    )
    model_choice = st.sidebar.selectbox(
        "Sentiment model",
        ["Fine-tuned", "Baseline"],
        help="Compare the custom fine-tuned DistilBERT checkpoint with the generic baseline model."
    )
    active_model_path = SENTIMENT_MODEL_PATH if model_choice == "Fine-tuned" else "distilbert-base-uncased"
    
    max_batch_size = st.sidebar.select_slider(
        "Max Batch Size (Benchmark)",
        options=[50, 100, 250, 500, 1000],
        value=500,
        help="Upper limit for benchmark batch sizes used in the Performance Metrics section."
    )
    
    market_data = load_market_data(DATA_PATH)
    headlines = load_headlines(HEADLINES_PATH)
    volatility_by_symbol = _compute_volatility_regimes(market_data)

    sentiment_analyzer = get_sentiment_analyzer(str(active_model_path), prefer_gpu=True)
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
    signals = get_cached_signals(volatility_by_symbol)

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
    sentiment_benchmark = benchmark_sentiment(headlines["text"].tolist(), str(active_model_path))
    
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
