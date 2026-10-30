"""Evaluation metrics shared by all auditors."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any


def classification_metrics(predictions: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    scored = [p for p in predictions if p.get("target") is not None]
    correct = [int(p.get("prediction") == p.get("target")) for p in scored]
    by_index: dict[int, list[int]] = defaultdict(list)
    by_secret: dict[str, list[int]] = defaultdict(list)
    for row, hit in zip(scored, correct):
        by_index[int(row.get("sample_index", 0))].append(hit)
        by_secret[str(row["target"])].append(hit)
    index_acc = {str(k): mean(v) for k, v in sorted(by_index.items())}
    return {
        "evaluation_protocol": protocol,
        "accuracy": mean(correct) if correct else None,
        "num_correct": sum(correct),
        "num_scored": len(correct),
        "num_unparsed": sum(p.get("prediction") is None for p in predictions),
        "accuracy_by_response_index": index_acc,
        "mean_accuracy_across_response_indices": mean(index_acc.values()) if index_acc else None,
        "std_accuracy_across_response_indices": pstdev(index_acc.values()) if len(index_acc) > 1 else 0.0,
        "accuracy_by_secret": {k: mean(v) for k, v in sorted(by_secret.items())},
    }

