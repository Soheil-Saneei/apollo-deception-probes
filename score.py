#!/usr/bin/env python3
"""Summarize response length and uncertainty language by condition."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


UNCERTAINTY = re.compile(r"\b(maybe|might|uncertain|approximately|about|likely|confidence)\b", re.I)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL file produced by generate.py")
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, int]]] = defaultdict(list)
    for line in args.input.read_text().splitlines():
        record = json.loads(line)
        response = record["response"]
        grouped[record["condition"]].append(
            {
                "words": len(response.split()),
                "uncertainty_markers": len(UNCERTAINTY.findall(response)),
            }
        )

    summary = {}
    for condition, rows in grouped.items():
        summary[condition] = {
            "responses": len(rows),
            "mean_words": sum(row["words"] for row in rows) / len(rows),
            "mean_uncertainty_markers": sum(row["uncertainty_markers"] for row in rows) / len(rows),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
