# 🚀 AMD GPU-Accelerated AI Signal Pipeline

Hackathon-ready demo of a safe, GPU-visible AI pipeline for market-style data.

This is **not a trading bot**. It is a product demo that shows how market data
and news headlines can flow through batch AI inference, open-weight sentiment,
fake signal generation, and a simulated execution dashboard.

## Overview

The app demonstrates:

- PyTorch batch inference on CPU or AMD GPU through ROCm
- open-weight DistilBERT sentiment fine-tuning and inference
- simple, transparent sentiment-aware dummy signals
- fake trade lifecycle simulation with slippage and P&L
- Streamlit dashboard with charts judges can understand in seconds

## Dashboard Preview

![Dashboard overview](assets/screenshots/dashboard-overview.png)

## GPU Benchmark View

![GPU benchmark](assets/screenshots/gpu-benchmark.png)

## Sentiment Panel

![Sentiment panel](assets/screenshots/sentiment-panel.png)

## Features

- **GPU Diagnostics**: Always-visible hardware context showing ROCm and CUDA availability and device name.
- **Batch Scaling Experiment**: Dynamic Streamlit control to benchmark CPU vs GPU throughput across multiple batch sizes.
- **Multi-Model Sentiment Switcher**: Compare baseline models against fine-tuned checkpoints on the fly.
- **Volatility/Regime Visualization**: Rolling standard deviation classifier feeding directly into dummy signals.
- **Pipeline X-ray Panel**: A debug view allowing judges to inspect the raw headlines, sentiment, volatility regime, and the final signal explanation.
- **Live Signal Feed**: color-coded `BUY`, `SELL`, and `HOLD` demo signals.
- **Sentiment Panel**: headline labels, signed scores, class distribution, and score trend.
- **Trade Simulation Panel**: fake OPEN-to-CLOSED lifecycle, slippage, P&L, win rate, and average return.
- **Performance Metrics**: CPU vs GPU latency, sample batch throughput, and speedup ratio.
- **IP-Safe Design**: no broker APIs, no credentials, no real strategy, no RL/OpenEnv, no private prompts.

## Architecture

```text
Mock OHLCV CSV + sample headlines
  -> PyTorch batch market inference
  -> Fine-tuned / Baseline DistilBERT sentiment inference
  -> Volatility / Regime Classification
  -> Sentiment-aware dummy signal generator
  -> Fake execution simulator
  -> Streamlit dashboard
```

## AMD Optimization

The project uses PyTorch tensors and batch inference. On AMD GPU machines with
ROCm-enabled PyTorch, `torch.cuda.is_available()` exposes the accelerator and
the exact same code path runs on GPU.

Benchmarks are generated at runtime for:

- CPU vs GPU latency
- batch sizes: `100` and `1000`
- throughput in signals/sec
- speedup ratio
- sentiment single vs batch inference

## Performance Results

Open the dashboard and look at **Performance Metrics**. It renders chart-ready
benchmark tables and grouped bar charts:

```text
CPU: latency ms, throughput signals/sec
GPU: latency ms, throughput signals/sec
Speedup: GPU throughput / CPU throughput
```

If no compatible AMD GPU is available, the dashboard clearly shows GPU as
pending/unavailable instead of pretending.

## Sentiment-Aware AI Pipeline

The sentiment module uses `distilbert-base-uncased`, a small open-weight
transformer suitable for quick fine-tuning. The included synthetic dataset has
720 balanced examples:

- `POSITIVE`: 240
- `NEGATIVE`: 240
- `NEUTRAL`: 240

Fine-tuning defaults:

- epochs: 3
- batch size: 16
- max sequence length: 96
- optimizer: AdamW
- device: CPU or ROCm/CUDA GPU through PyTorch

Train the sentiment checkpoint:

```bash
python scripts/generate_sentiment_dataset.py
python scripts/train_sentiment_model.py
```

The trained model is saved to:

```text
models/sentiment-distilbert/
```

The app can still open before training by using a clearly marked keyword demo
fallback. For a polished submission video, train the checkpoint first.

## Setup Instructions

```bash
cd hackathon-amd-trading-demo
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Linux or AMD ROCm machine:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For AMD GPU acceleration, install the ROCm-compatible PyTorch build recommended
for your machine, then run the same Streamlit command.

## Demo Flow

1. Start the dashboard with `streamlit run app.py`.
2. Show the top metrics: device, latency, throughput, GPU speedup, demo P&L.
3. Point to the color-coded live signal feed.
4. Show headlines flowing through sentiment labels and score charts.
5. Show fake trades moving through a CLOSED lifecycle with slippage and P&L.
6. Show CPU vs GPU throughput charts for market and sentiment inference.

## Visual Assets

Submission screenshots live here:

```text
assets/screenshots/dashboard-overview.png
assets/screenshots/gpu-benchmark.png
assets/screenshots/sentiment-panel.png
```

Suggested captures:

- full dashboard first viewport
- performance benchmark section
- sentiment panel with distribution chart

## GitHub Optimization

One-line repo description:

```text
AMD GPU-accelerated AI signal pipeline with open-weight sentiment, batch inference benchmarks, and a safe simulated trading dashboard.
```

Suggested topics:

```text
amd, rocm, pytorch, streamlit, sentiment-analysis, distilbert, gpu-acceleration, hackathon, ai-pipeline, simulation
```

Suggested polish before final submission:

- add dashboard screenshots under `assets/screenshots/`
- record a 60-90 second demo GIF or video
- include real benchmark numbers from an AMD GPU run
- pin Python version in the submission environment

## Safety Boundary

This repository intentionally avoids:

- real trading strategy logic
- broker integrations
- API keys or `.env` loading
- private prompts
- production model weights
- RL/OpenEnv code
- capital allocation or risk formulas

The signal generator is deliberately simple and non-realistic. Sentiment only
nudges dummy probabilities for demo visibility.
