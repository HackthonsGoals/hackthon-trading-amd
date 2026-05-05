from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


LABEL_TO_ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


class HeadlineDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 96) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def split_rows(frame: pd.DataFrame, validation_fraction: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_at = int(len(shuffled) * (1.0 - validation_fraction))
    return shuffled.iloc[:split_at], shuffled.iloc[split_at:]


def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=-1)
            correct += int((predictions == batch["labels"]).sum().item())
            total += int(batch["labels"].numel())
    model.train()
    return correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data = pd.read_csv(args.dataset)
    data["label_id"] = data["label"].map(LABEL_TO_ID)
    if data["label_id"].isna().any():
        raise ValueError("Dataset labels must be NEGATIVE, NEUTRAL, or POSITIVE")

    train_frame, valid_frame = split_rows(data)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    train_loader = DataLoader(
        HeadlineDataset(train_frame["text"].tolist(), train_frame["label_id"].astype(int).tolist(), tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        HeadlineDataset(valid_frame["text"].tolist(), valid_frame["label_id"].astype(int).tolist(), tokenizer),
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, steps // 10), num_training_steps=steps)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(outputs.loss.item())

        accuracy = evaluate(model, valid_loader, device)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"epoch={epoch + 1} loss={avg_loss:.4f} validation_accuracy={accuracy:.3f} device={device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned sentiment model to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for demo headline sentiment.")
    parser.add_argument("--dataset", default="data/sentiment_dataset.csv")
    parser.add_argument("--output-dir", default="models/sentiment-distilbert")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

