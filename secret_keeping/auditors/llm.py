"""Configurable LLM auditor retaining the original token-evidence prompt semantics."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from secret_keeping.utils.checkpoint import hf_hub_cache

from .base import BaseAuditor
from .build import AUDITOR_REGISTRY


def _dtype(name: str, torch):
    values = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name.lower() not in values:
        raise ValueError(f"Unsupported auditor dtype: {name}")
    return values[name.lower()]


def _format_tokens(tokens: list[dict], limit: int) -> str:
    if not tokens:
        return "No token information available."
    lines = []
    for token in tokens[:limit]:
        text = token.get("token", "").strip()
        probability = token.get("probability")
        rank = token.get("rank", "?")
        suffix = f" ({probability * 100:.3f}%)" if probability is not None else ""
        lines.append(f"  {rank}. {text}{suffix}")
    return "\n".join(lines)


def _format_features(features: list[dict], limit: int) -> str:
    if not features:
        return "No SAE feature information available."
    return "\n".join(
        f"  {index + 1}. Feature {feature.get('feature', feature.get('feature_index', '?'))} "
        f"(activation: {feature.get('activation', feature.get('activation_value', 0.0)):.4g})"
        for index, feature in enumerate(features[:limit])
    )


def _parse_prediction(text: str, classes: list[str]) -> str | None:
    candidates = []
    for label in classes:
        match = re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE)
        if match:
            candidates.append((match.start(), label))
    return min(candidates)[1] if candidates else None


@AUDITOR_REGISTRY.register("llm")
class LLMAuditor(BaseAuditor):
    protocol = "llm_auditor_original_prompt"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.template = Path(cfg.AUDITOR.PROMPT_TEMPLATE).read_text(encoding="utf-8").strip()
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self.model is not None:
            return
        if self.cfg.MODEL.CACHE_DIR:
            cache = Path(self.cfg.MODEL.CACHE_DIR)
            os.environ.setdefault("HF_HOME", str(cache))
            os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("LLM auditing requires torch and transformers") from exc
        kwargs = {
            "revision": self.cfg.AUDITOR.REVISION,
            "cache_dir": hf_hub_cache(self.cfg.MODEL.CACHE_DIR),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.AUDITOR.MODEL, **kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.AUDITOR.MODEL,
            torch_dtype=_dtype(self.cfg.AUDITOR.DTYPE, torch),
            device_map=self.cfg.AUDITOR.DEVICE_MAP,
            **kwargs,
        ).eval()

    def _prompt(self, result) -> str:
        evidence = result.evidence
        values = {
            "n": 1,
            "user_prompt": result.prompt,
            "model_response": result.response,
            "top_tokens": _format_tokens(
                evidence.get("top_k_tokens", []), self.cfg.AUDITOR.TOP_K_EVIDENCE
            ),
            "top_features": _format_features(
                evidence.get("top_features", []), self.cfg.AUDITOR.TOP_K_EVIDENCE
            ),
            "evidence": json.dumps(evidence, ensure_ascii=False, default=str),
            "allowed_classes": ", ".join(self.cfg.DATA.SECRET_CLASSES),
        }
        try:
            return self.template.format(**values)
        except KeyError as exc:
            raise ValueError(f"Unknown placeholder in auditor prompt template: {exc}") from exc

    def audit(self, results: list) -> list[dict]:
        import torch

        self._load()
        prompts = [self._prompt(result) for result in results]
        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            for prompt in prompts
        ]
        generated_texts: list[str] = []
        for start in range(0, len(formatted), self.cfg.AUDITOR.BATCH_SIZE):
            batch = formatted[start : start + self.cfg.AUDITOR.BATCH_SIZE]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            device = getattr(self.model, "device", next(self.model.parameters()).device)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            kwargs = {
                **encoded,
                "max_new_tokens": self.cfg.AUDITOR.MAX_NEW_TOKENS,
                "do_sample": self.cfg.AUDITOR.TEMPERATURE > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.cfg.AUDITOR.TEMPERATURE > 0:
                kwargs["temperature"] = self.cfg.AUDITOR.TEMPERATURE
            with torch.inference_mode():
                sequences = self.model.generate(**kwargs)
            input_width = encoded["input_ids"].shape[1]
            generated_texts.extend(
                self.tokenizer.decode(row[input_width:], skip_special_tokens=True).strip()
                for row in sequences
            )
        predictions = []
        for result, raw in zip(results, generated_texts):
            predictions.append(
                {
                    "sample_id": result.sample_id,
                    "sample_index": result.sample_index,
                    "method": result.method_name,
                    "prediction": _parse_prediction(raw, list(self.cfg.DATA.SECRET_CLASSES)),
                    "target": result.target,
                    "auditor_response": raw,
                    "evaluation_protocol": self.protocol,
                }
            )
        return predictions
