from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


FEATURE_COLUMNS = ["return_1", "range_pct", "volume_z", "close_position"]


class DemoInferenceNet(nn.Module):
    """Tiny untrained neural net for demonstrating batch inference only."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 3) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


@dataclass
class InferenceResult:
    probabilities: np.ndarray
    labels: list[str]
    device: str
    latency_ms: float
    throughput_rows_per_second: float


def select_device(prefer_gpu: bool = True) -> torch.device:
    """Select CUDA/ROCm when available; otherwise use CPU."""

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create generic, non-proprietary features from OHLCV rows."""

    df = frame.sort_values(["symbol", "timestamp"]).copy()
    df["return_1"] = df.groupby("symbol")["close"].pct_change().fillna(0.0)
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"].clip(lower=1e-9)).fillna(0.0)
    volume_mean = df.groupby("symbol")["volume"].transform("mean")
    volume_std = df.groupby("symbol")["volume"].transform("std").replace(0, np.nan)
    df["volume_z"] = ((df["volume"] - volume_mean) / volume_std).fillna(0.0)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_position"] = ((df["close"] - df["low"]) / candle_range).fillna(0.5)
    return df[FEATURE_COLUMNS].astype("float32")


def run_batch_inference(frame: pd.DataFrame, prefer_gpu: bool = True) -> InferenceResult:
    """Run a demo batch through a randomly initialized model."""

    device = select_device(prefer_gpu)
    features = build_features(frame)
    model = DemoInferenceNet(input_dim=len(FEATURE_COLUMNS)).to(device).eval()
    tensor = torch.tensor(features.to_numpy(), dtype=torch.float32, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)

    labels = ["SELL", "HOLD", "BUY"]
    return InferenceResult(
        probabilities=probabilities.cpu().numpy(),
        labels=labels,
        device=str(device),
        latency_ms=elapsed * 1000.0,
        throughput_rows_per_second=len(frame) / elapsed,
    )

