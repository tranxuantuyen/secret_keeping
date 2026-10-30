"""Stable, method-independent prompt loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class PromptSample:
    sample_id: str
    prompt: str
    target: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptDataset:
    def __init__(self, samples: list[PromptSample], name: str, source: Path):
        self.samples = samples
        self.name = name
        self.source = source

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[PromptSample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> PromptSample:
        return self.samples[index]

    @staticmethod
    def _stable_id(name: str, index: int, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
        return f"{name}:{index:04d}:{digest}"

    @classmethod
    def from_config(cls, cfg) -> "PromptDataset":
        path = Path(cfg.DATA.PROMPT_FILE)
        suffix = path.suffix.lower()
        rows: list[dict[str, Any]] = []
        if suffix in {".json", ".jsonl"}:
            if suffix == ".jsonl":
                source_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            else:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                source_rows = loaded.get("samples", loaded.get("results", loaded)) if isinstance(loaded, dict) else loaded
            for row in source_rows:
                if isinstance(row, str):
                    rows.append({"prompt": row})
                    continue
                prompt = row.get("prompt") or row.get("user_prompt") or row.get("input")
                if prompt is None and row.get("messages"):
                    prompt = next((m.get("content") for m in row["messages"] if m.get("role") == "user"), None)
                if prompt:
                    rows.append({**row, "prompt": prompt})
        else:
            rows = [{"prompt": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

        limit = cfg.DATA.LIMIT
        if limit is not None:
            rows = rows[: int(limit)]
        samples = []
        for index, row in enumerate(rows):
            prompt = str(row.pop("prompt"))
            target = row.pop("target", row.pop("secret", cfg.DATA.SECRET))
            sample_id = row.pop("sample_id", cls._stable_id(cfg.DATA.NAME, index, prompt))
            samples.append(PromptSample(sample_id, prompt, target, dict(row)))
        if not samples:
            raise ValueError(f"No prompts loaded from {path}")
        return cls(samples, cfg.DATA.NAME, path)

