"""Evaluation of a pre-fitted linear classifier; no fitting is performed."""

from __future__ import annotations

from secret_keeping.methods.base import MethodResult, ResponseRecord, WhiteBoxMethod
from secret_keeping.methods.build import METHOD_REGISTRY
from secret_keeping.utils.checkpoint import (
    ArtifactCompatibilityError,
    resolve_checkpoint,
    validate_metadata,
)

from .utils import position_metadata, select_positions


def _find_tensor(mapping, names):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


@METHOD_REGISTRY.register("linear_probe")
class LinearProbe(WhiteBoxMethod):
    evidence_type = "class_scores"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._weight = None
        self._bias = None
        self._classes = None
        self._checkpoint = None

    def required_artifacts(self) -> list[str]:
        return [self.cfg.METHOD.CHECKPOINT]

    def _load(self, model):
        if self._weight is not None:
            return
        import torch

        self._checkpoint = resolve_checkpoint(self.cfg.METHOD, self.cfg.MODEL.CACHE_DIR)
        checkpoint = torch.load(self._checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ArtifactCompatibilityError("Linear probe checkpoint must be a dictionary")
        state = checkpoint.get("state_dict", checkpoint)
        self._weight = _find_tensor(
            state, ("weight", "linear.weight", "classifier.weight", "probe.weight")
        )
        self._bias = _find_tensor(
            state, ("bias", "linear.bias", "classifier.bias", "probe.bias")
        )
        if self._weight is None:
            raise ArtifactCompatibilityError("No linear classifier weight found in checkpoint")
        self._weight = self._weight.float()
        if self._weight.ndim == 1:
            self._weight = self._weight[None]
        self._bias = torch.zeros(self._weight.shape[0]) if self._bias is None else self._bias.float()
        metadata = checkpoint.get("metadata", {})
        validate_metadata(
            metadata,
            model_id=self.cfg.MODEL.BASE_MODEL,
            layer=self.cfg.METHOD.LAYER,
            activation_dim=model.activation_dim,
            hook=self.cfg.METHOD.HOOK,
        )
        if self._weight.shape[1] != model.activation_dim:
            raise ArtifactCompatibilityError(
                f"Probe input dim={self._weight.shape[1]}, model hidden size={model.activation_dim}"
            )
        self._classes = checkpoint.get("classes", metadata.get("classes", self.cfg.DATA.SECRET_CLASSES))
        if self._weight.shape[0] not in {1, len(self._classes)}:
            raise ArtifactCompatibilityError(
                f"Probe has {self._weight.shape[0]} outputs for {len(self._classes)} classes"
            )

    def apply(self, record: ResponseRecord, model) -> MethodResult:
        import torch

        self._load(model)
        formatted = model.format_conversation(record.prompt, record.response)
        tokens = model.encode(formatted)
        positions, assistant_start = select_positions(tokens, model.tokenizer, self.cfg.METHOD.POSITION)
        activation = model.hidden_states(tokens)[self.cfg.METHOD.LAYER + 1][0, positions].float().mean(0).cpu()
        logits = self._weight @ activation + self._bias
        if logits.numel() == 1 and len(self._classes) == 2:
            logits = torch.stack([-logits[0], logits[0]])
        scores = {label: float(logits[index]) for index, label in enumerate(self._classes)}
        prediction = max(scores, key=scores.get)
        return MethodResult(
            record.sample_id,
            self.name,
            self.evidence_type,
            {"scores": scores, "positions": position_metadata(tokens, model.tokenizer, positions)},
            record.sample_index,
            record.target,
            record.prompt,
            record.response,
            direct_prediction=prediction,
            scores=scores,
            metadata={
                "layer": self.cfg.METHOD.LAYER,
                "position_mode": self.cfg.METHOD.POSITION,
                "assistant_start": assistant_start,
                "checkpoint": str(self._checkpoint),
            },
        )

