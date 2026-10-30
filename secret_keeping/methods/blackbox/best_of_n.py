"""Best-of-N without access to the target: modal closed-set mention, then earliest response."""

from __future__ import annotations

from collections import Counter

from secret_keeping.methods.base import MethodResult, ResponseRecord
from secret_keeping.methods.build import METHOD_REGISTRY

from .direct_prompt import DirectPrompt, parse_class


@METHOD_REGISTRY.register("best_of_n")
class BestOfN(DirectPrompt):
    inference_variant = "best_of_n"
    evidence_type = "candidate_responses"

    def generation_count(self) -> int:
        return self.cfg.METHOD.NUM_CANDIDATES

    def assemble_responses(self, sample, outputs: list[str]) -> list[ResponseRecord]:
        parsed = [parse_class(text, list(self.cfg.DATA.SECRET_CLASSES)) for text in outputs]
        counts = Counter(label for label in parsed if label is not None)
        winning_label = counts.most_common(1)[0][0] if counts else None
        selected_index = next(
            (index for index, label in enumerate(parsed) if label == winning_label), 0
        )
        return [
            ResponseRecord(
                sample.sample_id,
                sample.prompt,
                outputs[selected_index],
                0,
                sample.target,
                {
                    "candidates": outputs,
                    "candidate_predictions": parsed,
                    "selection": self.cfg.METHOD.SELECTION,
                    "selected_index": selected_index,
                    "target_was_not_used_for_selection": True,
                },
            )
        ]

    def apply(self, record: ResponseRecord, model) -> MethodResult:
        prediction = parse_class(record.response, list(self.cfg.DATA.SECRET_CLASSES))
        return MethodResult(
            record.sample_id,
            self.name,
            self.evidence_type,
            {
                "selected_response": record.response,
                "candidates": record.metadata["candidates"],
                "selection": record.metadata["selection"],
            },
            record.sample_index,
            record.target,
            record.prompt,
            record.response,
            direct_prediction=prediction,
            metadata=record.metadata,
        )

