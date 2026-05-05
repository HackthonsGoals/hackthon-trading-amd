from __future__ import annotations

import json
from pathlib import Path

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


def main() -> None:
    market_data = load_market_data(DATA_PATH)
    headlines = load_headlines(HEADLINES_PATH)

    sentiment_analyzer = SentimentAnalyzer(SENTIMENT_MODEL_PATH, prefer_gpu=True)
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
    signals = generate_dummy_signals(market_data, inference.probabilities, sentiment_by_symbol)

    SIGNALS_PATH.write_text(json.dumps(signals, indent=2), encoding="utf-8")

    simulator = ExecutionSimulator(TRADES_PATH)
    trades = simulator.execute(signals)
    trade_summary = simulator.summary()
    metrics = {
        "device": inference.device,
        "latency_ms": inference.latency_ms,
        "throughput_rows_per_second": inference.throughput_rows_per_second,
    }
    benchmark = benchmark_inference(market_data)
    sentiment_benchmark = benchmark_sentiment(headlines["text"].tolist(), SENTIMENT_MODEL_PATH)
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
    )


if __name__ == "__main__":
    main()
