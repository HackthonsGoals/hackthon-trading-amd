from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


def load_market_data(path: str | Path) -> pd.DataFrame:
    """Load demo OHLCV data and validate a small public schema."""

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Market data file not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def latest_rows(df: pd.DataFrame, rows_per_symbol: int = 1) -> pd.DataFrame:
    """Return the latest N rows per symbol."""

    return (
        df.sort_values("timestamp")
        .groupby("symbol", as_index=False, group_keys=False)
        .tail(rows_per_symbol)
        .reset_index(drop=True)
    )


def load_headlines(path: str | Path) -> pd.DataFrame:
    """Load sample news/headline text for sentiment inference."""

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Headline file not found: {data_path}")

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    required = ["timestamp", "symbol", "text"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required headline columns: {missing}")
    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
