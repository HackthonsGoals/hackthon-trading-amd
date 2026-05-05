from __future__ import annotations

import csv
from pathlib import Path


OUTPUT = Path(__file__).resolve().parents[1] / "data" / "sentiment_dataset.csv"

POSITIVE_EVENTS = [
    "posts strong quarterly results",
    "beats revenue estimates",
    "raises full year guidance",
    "wins a major cloud contract",
    "reports record order book",
    "announces margin expansion",
    "receives analyst upgrade",
    "sees demand growth",
]
NEGATIVE_EVENTS = [
    "falls after weak quarterly results",
    "misses revenue estimates",
    "cuts full year guidance",
    "faces global uncertainty",
    "reports margin pressure",
    "receives analyst downgrade",
    "warns about demand slowdown",
    "declines after regulatory concerns",
]
NEUTRAL_EVENTS = [
    "schedules investor meeting",
    "announces board update",
    "trades in a narrow range",
    "holds annual general meeting",
    "updates product roadmap",
    "reports stable operations",
    "opens new regional office",
    "maintains existing guidance",
]
COMPANIES = [
    "Aster Motors",
    "BluePeak Finance",
    "Cedar Energy",
    "Delta Retail",
    "Evergreen Tech",
    "Falcon Infra",
    "Granite Foods",
    "Helio Pharma",
    "Indigo Systems",
    "Jade Semiconductors",
]


def build_rows() -> list[dict]:
    rows = []
    templates = [
        ("POSITIVE", POSITIVE_EVENTS),
        ("NEGATIVE", NEGATIVE_EVENTS),
        ("NEUTRAL", NEUTRAL_EVENTS),
    ]
    for label, events in templates:
        for index in range(240):
            company = COMPANIES[index % len(COMPANIES)]
            event = events[index % len(events)]
            suffix = "during morning trade" if index % 2 == 0 else "as volumes remain active"
            rows.append({"text": f"{company} {event} {suffix}", "label": label})
    return rows


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(build_rows())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

