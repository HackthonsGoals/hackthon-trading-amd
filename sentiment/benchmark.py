from __future__ import annotations

import time
from pathlib import Path

import torch

from sentiment.sentiment_inference import SentimentAnalyzer


def benchmark_sentiment(
    texts: list[str],
    model_dir: str | Path = "models/sentiment-distilbert",
    repeats: int = 3,
) -> dict:
    """Measure single vs batch inference and CPU vs GPU availability."""

    if not texts:
        return {}

    results: dict[str, dict] = {}
    records: list[dict] = []
    for name, prefer_gpu in [("cpu", False), ("gpu", True)]:
        if name == "gpu" and not torch.cuda.is_available():
            results[name] = {"available": False, "device": "unavailable"}
            continue

        analyzer = SentimentAnalyzer(model_dir=model_dir, prefer_gpu=prefer_gpu)
        single_latencies = []
        batch_latencies = []
        for _ in range(repeats):
            start = time.perf_counter()
            analyzer.predict_one(texts[0])
            single_latencies.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            analyzer.predict_batch(texts, batch_size=min(16, len(texts)))
            batch_latencies.append((time.perf_counter() - start) * 1000.0)

        avg_batch = sum(batch_latencies) / len(batch_latencies)
        row = {
            "device_type": name.upper(),
            "available": True,
            "device": str(analyzer.device),
            "backend": analyzer.backend,
            "single_latency_ms": round(sum(single_latencies) / len(single_latencies), 4),
            "batch_latency_ms": round(avg_batch, 4),
            "batch_throughput_texts_per_second": round(len(texts) / max(avg_batch / 1000.0, 1e-9), 2),
        }
        results[name] = row
        records.append(row)

    cpu_tps = results.get("cpu", {}).get("batch_throughput_texts_per_second")
    gpu_tps = results.get("gpu", {}).get("batch_throughput_texts_per_second")
    results["batch_speedup"] = round(gpu_tps / cpu_tps, 3) if cpu_tps and gpu_tps else None
    results["records"] = records
    return results
