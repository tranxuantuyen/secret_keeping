"""Assistant-turn continuation using the legacy Gemma end-of-turn trimming convention."""

from __future__ import annotations

from pathlib import Path

from secret_keeping.methods.base import ResponseRecord
from secret_keeping.methods.build import METHOD_REGISTRY

from .direct_prompt import DirectPrompt


@METHOD_REGISTRY.register("assistant_prefill")
class AssistantPrefill(DirectPrompt):
    inference_variant = "assistant_prefill"

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.METHOD.PREFILL_FILE:
            lines = [
                line.strip()
                for line in Path(cfg.METHOD.PREFILL_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(lines) != 1:
                raise ValueError("assistant_prefill currently requires exactly one prefill line")
            self.prefill = lines[0]
        else:
            self.prefill = cfg.METHOD.PREFILL

    def format_input(self, sample, model) -> str:
        return model.format_prompt(sample.prompt, assistant_prefill=self.prefill)

    def assemble_responses(self, sample, outputs: list[str]) -> list[ResponseRecord]:
        return [
            ResponseRecord(
                sample.sample_id,
                sample.prompt,
                self.prefill + output,
                index,
                sample.target,
                {"prefill": self.prefill},
            )
            for index, output in enumerate(outputs)
        ]

