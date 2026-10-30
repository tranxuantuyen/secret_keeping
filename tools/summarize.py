#!/usr/bin/env python3
"""Summarize completed runs without conflating distinct evaluation protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", default="output")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.output_dir).glob("*/*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                metrics.get("experiment", path.parents[1].name),
                path.parent.name,
                metrics.get("method"),
                metrics.get("evaluation_protocol"),
                metrics.get("accuracy"),
                metrics.get("num_scored"),
            )
        )
    if not rows:
        print("No metrics.json files found")
        return 0
    print("experiment\trun_id\tmethod\tprotocol\taccuracy\tn")
    for row in rows:
        print("\t".join("" if value is None else str(value) for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

