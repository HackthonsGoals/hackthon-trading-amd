from __future__ import annotations

import time

import pandas as pd
import torch

from engine.ai_inference import run_batch_inference


def _make_batch(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    if len(frame) >= size:
        return frame.head(size).copy()
    repeats = (size // max(len(frame), 1)) + 1
    expanded = pd.concat([frame] * repeats, ignore_index=True).head(size)
    expanded["timestamp"] = pd.date_range("2026-05-05 09:15:00", periods=size, freq="s")
    return expanded


def benchmark_inference(frame: pd.DataFrame, repeats: int = 3, batch_sizes: tuple[int, ...] = (100, 1000)) -> dict:
    """Compare CPU and GPU batch inference with chart-ready metrics."""

    results: dict[str, dict] = {}
    records: list[dict] = []
    for name, prefer_gpu in [("cpu", False), ("gpu", True)]:
        if name == "gpu" and not torch.cuda.is_available():
            results[name] = {
                "available": False,
                "device": "unavailable",
                "batches": [],
            }
            continue

        batch_results = []
        device = "cpu"
        for batch_size in batch_sizes:
            batch = _make_batch(frame, batch_size)
            latencies = []
            throughput = []
            start = time.perf_counter()
            for _ in range(repeats):
                result = run_batch_inference(batch, prefer_gpu=prefer_gpu)
                device = result.device
                latencies.append(result.latency_ms)
                throughput.append(result.throughput_rows_per_second)
            elapsed = time.perf_counter() - start
            row = {
                "device_type": name.upper(),
                "device": device,
                "batch_size": batch_size,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 4),
                "throughput_signals_per_second": round(sum(throughput) / len(throughput), 2),
                "wall_time_seconds": round(elapsed, 4),
            }
            batch_results.append(row)
            records.append(row)

        results[name] = {
            "available": True,
            "device": device,
            "batches": batch_results,
        }

    speedups = []
    for batch_size in batch_sizes:
        cpu_row = next((row for row in records if row["device_type"] == "CPU" and row["batch_size"] == batch_size), None)
        gpu_row = next((row for row in records if row["device_type"] == "GPU" and row["batch_size"] == batch_size), None)
        if cpu_row and gpu_row:
            speedups.append(
                {
                    "batch_size": batch_size,
                    "latency_speedup": round(cpu_row["avg_latency_ms"] / max(gpu_row["avg_latency_ms"], 1e-9), 3),
                    "throughput_speedup": round(
                        gpu_row["throughput_signals_per_second"]
                        / max(cpu_row["throughput_signals_per_second"], 1e-9),
                        3,
                    ),
                }
            )

    results["records"] = records
    results["speedups"] = speedups
    results["best_speedup"] = max((row["throughput_speedup"] for row in speedups), default=None)
    return results
