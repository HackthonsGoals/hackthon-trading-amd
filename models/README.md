# Models

This folder is intentionally empty by default except for generated checkpoints.

The demo constructs a tiny untrained PyTorch network at runtime to show a
GPU-ready batch inference path. No production model weights, prompts,
adapters, reinforcement learning assets, or proprietary training artifacts are
stored here.

The sentiment training script saves the lightweight fine-tuned DistilBERT
checkpoint to `models/sentiment-distilbert/`. That folder is generated locally
from the synthetic dataset and should not contain private models or data.
