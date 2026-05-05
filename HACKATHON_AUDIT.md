# Hackathon Audit

| file/path | issue | impact | fix |
|---|---|---|---|
| `README.md` | Generic demo framing | Judges may not see AMD/GPU value fast enough | Rewritten with GPU-first pitch, architecture, setup, demo flow, screenshots, and GitHub topics |
| `utils/benchmark.py` | Only benchmarked current tiny data | Weak performance story | Added 100/1000 synthetic batch sizes, throughput, speedup, and chart-ready records |
| `sentiment/benchmark.py` | JSON-only sentiment metrics | Hard to visualize | Added chart-ready benchmark records and speedup |
| `dashboard/ui.py` | Developer-like tables and JSON | UI felt less submission-ready | Added top metrics, color-coded signals, sentiment charts, trade panel, benchmark charts |
| `simulator/execution_simulator.py` | Trades were just filled rows | Simulation lacked lifecycle | Added trade IDs, opened/closed timestamps, CLOSED status, slippage, return %, win rate, avg return |
| `data/simulated_trades.json` | Static empty trade log | Fine for reset state | Kept as clean runtime output target |
| `models/` | No checkpoint by default | Sentiment model must be trained locally | README and model folder document generated checkpoint path |
| `IP_SAFETY_CLASSIFICATION.md` | Long but useful | Good safety proof, not dashboard content | Kept as due-diligence artifact |

## Prioritized Fixes Completed

1. Improved GPU benchmark visibility.
2. Upgraded dashboard layout for a live demo.
3. Improved fake execution lifecycle metrics.
4. Rewrote README for hackathon judges.
5. Re-ran IP-safety scan for executable code.

## Dashboard Upgrade Plan

- First viewport: device, latency, throughput, speedup, P&L.
- Middle: live signals and headline sentiment.
- Lower: market chart, trade metrics, benchmark charts.
- Final: trade lifecycle and cumulative P&L.

## Benchmark Logic

- Expand mock market rows to fixed batches of 100 and 1000.
- Run repeated CPU and GPU inference.
- Report latency, throughput, and speedup.
- Expose `records` arrays for Plotly charts.

## Screenshot Guidance

Place images in:

```text
assets/screenshots/
```

Recommended files:

- `dashboard-main.png`
- `performance-metrics.png`
- `sentiment-panel.png`

