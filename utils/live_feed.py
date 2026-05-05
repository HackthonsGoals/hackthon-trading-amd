from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class MockLiveFeed:
    """Cycle through historical rows to mimic a live market feed."""

    data: pd.DataFrame
    window_size: int = 12

    def stream(self) -> Iterator[pd.DataFrame]:
        ordered = self.data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        total = len(ordered)
        if total == 0:
            return

        for end in range(1, total + 1):
            start = max(0, end - self.window_size)
            yield ordered.iloc[start:end].reset_index(drop=True)

