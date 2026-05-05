from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch


LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


@dataclass
class SentimentPrediction:
    text: str
    sentiment: str
    score: float
    device: str
    latency_ms: float


class SentimentAnalyzer:
    """Batch sentiment inference using a fine-tuned open-weight transformer."""

    def __init__(
        self,
        model_dir: str | Path = "models/sentiment-distilbert",
        prefer_gpu: bool = True,
        allow_demo_fallback: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")
        self.allow_demo_fallback = allow_demo_fallback
        self.tokenizer = None
        self.model = None
        self.backend = "keyword-demo"
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_dir.exists():
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device).eval()
            self.backend = "fine-tuned-distilbert"
        except Exception:
            if not self.allow_demo_fallback:
                raise

    def predict_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        if not texts:
            return []

        if self.model is None or self.tokenizer is None:
            return self._keyword_fallback(texts)

        predictions: list[dict] = []
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for offset in range(0, len(texts), batch_size):
            chunk = texts[offset : offset + batch_size]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu()

            for text, row in zip(chunk, probabilities):
                index = int(torch.argmax(row).item())
                predictions.append(
                    {
                        "text": text,
                        "sentiment": LABELS[index],
                        "score": round(float(row[index].item()), 4),
                        "device": str(self.device),
                    }
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        per_item = latency_ms / max(len(texts), 1)
        for prediction in predictions:
            prediction["latency_ms"] = round(per_item, 4)
            prediction["backend"] = self.backend
        return predictions

    def predict_one(self, text: str) -> dict:
        return self.predict_batch([text], batch_size=1)[0]

    def _keyword_fallback(self, texts: list[str]) -> list[dict]:
        """Transparent demo fallback used only before the checkpoint is trained."""

        positive = {"strong", "beats", "growth", "profit", "upgrade", "record", "rally", "wins"}
        negative = {"falls", "weak", "loss", "downgrade", "uncertainty", "misses", "pressure", "declines"}
        start = time.perf_counter()
        predictions = []
        for text in texts:
            tokens = {token.strip(".,:;!?").lower() for token in text.split()}
            pos = len(tokens & positive)
            neg = len(tokens & negative)
            if pos > neg:
                label = "POSITIVE"
                score = min(0.55 + pos * 0.1, 0.95)
            elif neg > pos:
                label = "NEGATIVE"
                score = min(0.55 + neg * 0.1, 0.95)
            else:
                label = "NEUTRAL"
                score = 0.5
            predictions.append(
                {
                    "text": text,
                    "sentiment": label,
                    "score": round(score, 4),
                    "device": str(self.device),
                    "latency_ms": 0.0,
                    "backend": self.backend,
                }
            )

        per_item = ((time.perf_counter() - start) * 1000.0) / max(len(texts), 1)
        for prediction in predictions:
            prediction["latency_ms"] = round(per_item, 4)
        return predictions


def sentiment_score(prediction: dict) -> float:
    if prediction["sentiment"] == "POSITIVE":
        return float(prediction["score"])
    if prediction["sentiment"] == "NEGATIVE":
        return -float(prediction["score"])
    return 0.0

