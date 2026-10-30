"""Direct secret scoring using an existing fitted Template Lens artifact."""

from __future__ import annotations

from secret_keeping.methods.base import MethodResult, ResponseRecord, WhiteBoxMethod
from secret_keeping.methods.build import METHOD_REGISTRY
from secret_keeping.utils.checkpoint import ArtifactCompatibilityError, resolve_checkpoint

from .utils import position_metadata, select_positions


@METHOD_REGISTRY.register("template_lens")
class TemplateLens(WhiteBoxMethod):
    evidence_type = "class_scores"
    inference_variant = "prompt_only"
    requires_generation = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self._artifact = None
        self._checkpoint = None

    def required_artifacts(self) -> list[str]:
        return [self.cfg.METHOD.CHECKPOINT]

    def _load(self, model):
        if self._artifact is not None:
            return
        import torch

        self._checkpoint = resolve_checkpoint(self.cfg.METHOD, self.cfg.MODEL.CACHE_DIR)
        self._artifact = torch.load(self._checkpoint, map_location="cpu", weights_only=True)
        required = {"layers", "words", "T", "mu"}
        missing = required - set(self._artifact)
        if missing:
            raise ArtifactCompatibilityError(f"Template artifact missing keys: {sorted(missing)}")
        if self.cfg.METHOD.LAYER not in self._artifact["layers"]:
            raise ArtifactCompatibilityError(
                f"Layer {self.cfg.METHOD.LAYER} not in template layers={self._artifact['layers']}"
            )
        if int(self._artifact["T"].shape[-1]) != model.activation_dim:
            raise ArtifactCompatibilityError(
                f"Template dim={self._artifact['T'].shape[-1]}, model dim={model.activation_dim}"
            )

    def apply(self, record: ResponseRecord, model) -> MethodResult:
        import torch

        self._load(model)
        formatted = model.format_conversation(record.prompt, record.response)
        tokens = model.encode(formatted)
        positions, assistant_start = select_positions(tokens, model.tokenizer, self.cfg.METHOD.POSITION)
        hidden = model.hidden_states(tokens)[self.cfg.METHOD.LAYER + 1][0, positions].float().cpu()
        artifact_index = list(self._artifact["layers"]).index(self.cfg.METHOD.LAYER)
        centered = hidden - self._artifact["mu"][artifact_index].float()
        word_to_index = {word: index for index, word in enumerate(self._artifact["words"])}
        missing = [word for word in self.cfg.DATA.SECRET_CLASSES if word not in word_to_index]
        if missing:
            raise ArtifactCompatibilityError(f"Secret classes absent from template vocabulary: {missing}")
        indices = [word_to_index[word] for word in self.cfg.DATA.SECRET_CLASSES]
        templates = self._artifact["T"][artifact_index, indices].float()
        raw_scores = centered @ templates.T
        mode = self.cfg.METHOD.ARTIFACT_MODE.lower()
        if mode in {"lda", "template_lda"}:
            if "bias" not in self._artifact:
                raise ArtifactCompatibilityError("LDA scoring requested but template artifact has no bias")
            raw_scores = raw_scores - self._artifact["bias"][artifact_index, indices].float()
        elif mode not in {"raw", "projection", "template_proj"}:
            raise ValueError("Template ARTIFACT_MODE must be projection/raw or lda")
        averaged = raw_scores.mean(dim=0)
        scores = {
            label: float(averaged[index]) for index, label in enumerate(self.cfg.DATA.SECRET_CLASSES)
        }
        prediction = max(scores, key=scores.get)
        evidence = {
            "scores": scores,
            "positions": position_metadata(tokens, model.tokenizer, positions),
        }
        return MethodResult(
            record.sample_id,
            self.name,
            self.evidence_type,
            evidence,
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
                "scoring": "template_lda" if mode in {"lda", "template_lda"} else "template_projection",
            },
        )
