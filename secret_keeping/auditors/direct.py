"""Closed-set scoring for methods that already produce class predictions/scores."""

from __future__ import annotations

from .base import BaseAuditor
from .build import AUDITOR_REGISTRY


@AUDITOR_REGISTRY.register("direct")
class DirectAuditor(BaseAuditor):
    protocol = "direct_closed_set_classification"

    def audit(self, results: list) -> list[dict]:
        predictions = []
        for result in results:
            prediction = result.direct_prediction
            if prediction is None and result.scores:
                prediction = max(result.scores, key=result.scores.get)
            predictions.append(
                {
                    "sample_id": result.sample_id,
                    "sample_index": result.sample_index,
                    "method": result.method_name,
                    "prediction": prediction,
                    "target": result.target,
                    "scores": result.scores,
                    "evaluation_protocol": self.protocol,
                }
            )
        return predictions

