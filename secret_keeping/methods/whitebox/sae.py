"""Checkpoint-only SAE feature extraction (raw checkpoints or sae-lens releases)."""

from __future__ import annotations

from secret_keeping.methods.base import MethodResult, ResponseRecord, WhiteBoxMethod
from secret_keeping.methods.build import METHOD_REGISTRY
from secret_keeping.utils.checkpoint import (
    ArtifactCompatibilityError,
    resolve_checkpoint,
    validate_metadata,
)

from .utils import position_metadata, select_positions


@METHOD_REGISTRY.register("sae")
class SAEReadout(WhiteBoxMethod):
    evidence_type = "sae_features"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._sae = None
        self._raw = None
        self._metadata = {}
        self._checkpoint = None

    def required_artifacts(self) -> list[str]:
        if self.cfg.METHOD.CHECKPOINT:
            return [self.cfg.METHOD.CHECKPOINT]
        return [f"sae-lens://{self.cfg.METHOD.SAE_RELEASE}/{self.cfg.METHOD.SAE_ID}"]

    def _load(self, model):
        if self._sae is not None or self._raw is not None:
            return
        import torch

        if self.cfg.METHOD.SAE_RELEASE:
            try:
                from sae_lens import SAE
            except ImportError as exc:
                raise RuntimeError("sae-lens is required for METHOD.SAE_RELEASE") from exc
            loaded = SAE.from_pretrained(
                release=self.cfg.METHOD.SAE_RELEASE,
                sae_id=self.cfg.METHOD.SAE_ID,
                device=str(model.device),
            )
            self._sae = loaded[0] if isinstance(loaded, tuple) else loaded
            self._metadata = {
                "model_id": getattr(getattr(self._sae, "cfg", None), "model_name", None),
                "hook": getattr(getattr(self._sae, "cfg", None), "hook_name", None),
                "activation_dim": getattr(getattr(self._sae, "cfg", None), "d_in", None),
            }
        else:
            self._checkpoint = resolve_checkpoint(self.cfg.METHOD, self.cfg.MODEL.CACHE_DIR)
            checkpoint = torch.load(self._checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ArtifactCompatibilityError("Raw SAE checkpoint must be a dictionary")
            self._raw = checkpoint.get("state_dict", checkpoint)
            self._metadata = checkpoint.get("metadata", {})
            if not any(
                key in self._raw for key in ("W_enc", "encoder.weight", "encoder_linear.weight")
            ):
                raise ArtifactCompatibilityError(
                    "Raw SAE checkpoint needs W_enc, encoder.weight, or encoder_linear.weight"
                )
        validate_metadata(
            {k: v for k, v in self._metadata.items() if v is not None},
            model_id=self.cfg.MODEL.BASE_MODEL,
            layer=self.cfg.METHOD.LAYER,
            activation_dim=model.activation_dim,
            hook=self.cfg.METHOD.HOOK,
        )

    def _encode(self, activation):
        import torch

        if self._sae is not None:
            return self._sae.encode(activation.to(next(self._sae.parameters()).device)).float().cpu()
        weight = self._raw.get(
            "W_enc", self._raw.get("encoder.weight", self._raw.get("encoder_linear.weight"))
        ).float()
        bias = self._raw.get(
            "b_enc", self._raw.get("encoder.bias", self._raw.get("encoder_linear.bias"))
        )
        if weight.shape[0] == activation.shape[-1]:
            encoded = activation.cpu() @ weight
            feature_dim = weight.shape[1]
        elif weight.shape[1] == activation.shape[-1]:
            encoded = activation.cpu() @ weight.T
            feature_dim = weight.shape[0]
        else:
            raise ArtifactCompatibilityError(
                f"SAE encoder shape={tuple(weight.shape)} incompatible with activation dim={activation.shape[-1]}"
            )
        if bias is not None:
            if bias.numel() != feature_dim:
                raise ArtifactCompatibilityError("SAE encoder bias has incompatible dimensionality")
            encoded = encoded + bias.float()
        return torch.relu(encoded)

    def apply(self, record: ResponseRecord, model) -> MethodResult:
        import torch

        self._load(model)
        formatted = model.format_conversation(record.prompt, record.response)
        tokens = model.encode(formatted)
        positions, assistant_start = select_positions(tokens, model.tokenizer, self.cfg.METHOD.POSITION)
        activation = model.hidden_states(tokens)[self.cfg.METHOD.LAYER + 1][0, positions].float()
        features = self._encode(activation).mean(0)
        k = min(self.cfg.METHOD.TOP_FEATURES, features.numel())
        values, indices = torch.topk(features, k)
        top_features = [
            {"feature": int(index), "activation": float(value)}
            for value, index in zip(values, indices)
        ]
        scores = None
        prediction = None
        class_weights = self._raw.get("class_weights") if self._raw else None
        if class_weights is not None:
            logits = class_weights.float() @ features.cpu()
            scores = {
                label: float(logits[index])
                for index, label in enumerate(self.cfg.DATA.SECRET_CLASSES)
            }
            prediction = max(scores, key=scores.get)
        return MethodResult(
            record.sample_id,
            self.name,
            self.evidence_type,
            {
                "top_features": top_features,
                "positions": position_metadata(tokens, model.tokenizer, positions),
            },
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
                "checkpoint": str(self._checkpoint) if self._checkpoint else None,
                "sae_release": self.cfg.METHOD.SAE_RELEASE,
                "sae_id": self.cfg.METHOD.SAE_ID,
            },
        )
