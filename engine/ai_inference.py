from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

FEATURE_COLUMNS = ["return_1", "range_pct", "volume_z", "close_position"]


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


def generate_financial_snippets(frame: pd.DataFrame) -> list[str]:
    """Generate synthetic financial text from ticker data for transformer inference."""
    snippets = []
    for _, row in frame.iterrows():
        symbol = row.get("symbol", "STOCK")
        ret = row.get("return_1", 0.0)
        direction = "surged" if ret > 0.02 else "declined" if ret < -0.02 else "traded flat"
        snippet = f"{symbol} stock {direction} with volume trends indicating {'bullish' if ret > 0 else 'bearish'} sentiment."
        snippets.append(snippet)
    return snippets


_FINBERT_CACHE = {"model": None, "tokenizer": None, "device": None}

def _get_finbert(device: torch.device):
    """Singleton to ensure FinBERT is only loaded once per process."""
    if _FINBERT_CACHE["model"] is None or _FINBERT_CACHE["device"] != device:
        model_name = "ProsusAI/finbert"
        _FINBERT_CACHE["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        _FINBERT_CACHE["model"] = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        _FINBERT_CACHE["device"] = device
    return _FINBERT_CACHE["model"], _FINBERT_CACHE["tokenizer"]

def run_batch_inference(frame: pd.DataFrame, prefer_gpu: bool = True) -> InferenceResult:
    """Run batch inference using FinBERT transformer for financial sentiment."""
    device = select_device(prefer_gpu)
    
    # Use cached model and tokenizer
    model, tokenizer = _get_finbert(device)
    
    # Generate financial text snippets from dataframe
    snippets = generate_financial_snippets(frame)
    
    # Tokenize batch
    inputs = tokenizer(
        snippets,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    # Benchmark inference
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    with torch.no_grad():
        if device.type == "cuda":
            # Use mixed precision for AMD/NVIDIA GPU acceleration
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
        else:
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = max(time.perf_counter() - start, 1e-9)
    
    # FinBERT labels: positive, negative, neutral
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    return InferenceResult(
        probabilities=probabilities.cpu().numpy(),
        labels=labels,
        device=str(device),
        latency_ms=elapsed * 1000.0,
        throughput_rows_per_second=len(frame) / elapsed,
    )
