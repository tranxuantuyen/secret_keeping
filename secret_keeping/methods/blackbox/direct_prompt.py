"""Standard direct prompting with optional closed-set parsing."""

from __future__ import annotations

import re

from secret_keeping.methods.base import BlackBoxMethod, MethodResult, ResponseRecord
from secret_keeping.methods.build import METHOD_REGISTRY


def parse_class(text: str, classes: list[str]) -> str | None:
    matches = []
    for label in classes:
        match = re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE)
        if match:
            matches.append((match.start(), label))
    return min(matches)[1] if matches else None


@METHOD_REGISTRY.register("direct_prompt")
class DirectPrompt(BlackBoxMethod):
    evidence_type = "model_response"

    def apply(self, record: ResponseRecord, model) -> MethodResult:
        prediction = parse_class(record.response, list(self.cfg.DATA.SECRET_CLASSES))
        return MethodResult(
            record.sample_id,
            self.name,
            self.evidence_type,
            {"response": record.response},
            record.sample_index,
            record.target,
            record.prompt,
            record.response,
            direct_prediction=prediction,
            metadata=record.metadata,
        )

