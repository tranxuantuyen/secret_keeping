"""Evaluation-only integration for a pre-fitted Jacobian Lens artifact."""

from __future__ import annotations

import sys
from pathlib import Path

from secret_keeping.methods.base import MethodResult, ResponseRecord, WhiteBoxMethod
from secret_keeping.methods.build import METHOD_REGISTRY
from secret_keeping.utils.checkpoint import ArtifactCompatibilityError, resolve_checkpoint

from .utils import position_metadata, select_positions, top_k_tokens


@METHOD_REGISTRY.register("jacobian_lens")
class JacobianLens(WhiteBoxMethod):
    evidence_type = "token_probabilities"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._lens = None
        self._lens_model = None
        self._checkpoint = None

    def required_artifacts(self) -> list[str]:
        if self.cfg.METHOD.CHECKPOINT:
            return [self.cfg.METHOD.CHECKPOINT]
        return [f"hf://{self.cfg.METHOD.CHECKPOINT_REPO}/{self.cfg.METHOD.CHECKPOINT_FILE}"]

    def _load(self, model):
        if self._lens is not None:
            return
        external = Path(self.cfg.EXTERNAL.JACOBIAN_LENS)
        if external.is_dir() and str(external) not in sys.path:
            sys.path.insert(0, str(external))
        try:
            import jlens
        except ImportError as exc:
            raise RuntimeError(
                "Jacobian Lens requires the jlens package. Install/vendor it under "
                f"{external} or install the optional Jacobian environment."
            ) from exc
        self._checkpoint = resolve_checkpoint(self.cfg.METHOD, self.cfg.MODEL.CACHE_DIR)
        self._lens = jlens.JacobianLens.load(self._checkpoint)
        layer = self.cfg.METHOD.LAYER
        if layer not in self._lens.source_layers:
            raise ArtifactCompatibilityError(
                f"Layer {layer} is not available in lens source_layers={self._lens.source_layers}"
            )
        if getattr(self._lens, "d_model", model.activation_dim) != model.activation_dim:
            raise ArtifactCompatibilityError(
                f"Lens d_model={self._lens.d_model}, model hidden size={model.activation_dim}"
            )
        self._lens_model = jlens.from_hf(model.lens_model(), model.tokenizer, force_bos=False)
        # The setter updates GemmaTokenizer's Rust post-processor; assignment is intentional.
        self._lens_model.tokenizer.add_bos_token = False

    def apply(self, record: ResponseRecord, model) -> MethodResult:
        import torch

        self._load(model)
        formatted = model.format_conversation(record.prompt, record.response)
        tokens = model.encode(formatted)
        positions, assistant_start = select_positions(tokens, model.tokenizer, self.cfg.METHOD.POSITION)
        if len(tokens) > self.cfg.METHOD.MAX_SEQ_LEN:
            raise ValueError(
                f"Sequence has {len(tokens)} tokens, exceeding METHOD.MAX_SEQ_LEN={self.cfg.METHOD.MAX_SEQ_LEN}"
            )
        lens_logits, _model_logits, lens_ids = self._lens.apply(
            self._lens_model,
            formatted,
            layers=[self.cfg.METHOD.LAYER],
            positions=positions,
            max_seq_len=self.cfg.METHOD.MAX_SEQ_LEN,
            use_jacobian=True,
        )
        if lens_ids[0].tolist() != tokens.tolist():
            raise RuntimeError("Jacobian Lens tokenization differs from add_special_tokens=False encoding")
        logits = lens_logits[self.cfg.METHOD.LAYER]
        probabilities = torch.softmax(logits.float(), dim=-1).mean(dim=0)
        evidence = {
            "top_k_tokens": top_k_tokens(probabilities, model.tokenizer, self.cfg.METHOD.TOP_K),
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
            metadata={
                "layer": self.cfg.METHOD.LAYER,
                "position_mode": self.cfg.METHOD.POSITION,
                "assistant_start": assistant_start,
                "checkpoint": str(self._checkpoint),
                "extraction": "prefitted_jacobian_lens_softmax_then_average_positions",
            },
        )

