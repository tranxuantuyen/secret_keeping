"""Gemma logit lens matching the original normalize-unembed-softmax readout."""

from __future__ import annotations

from secret_keeping.methods.base import MethodResult, ResponseRecord, WhiteBoxMethod
from secret_keeping.methods.build import METHOD_REGISTRY

from .utils import position_metadata, select_positions, top_k_tokens


@METHOD_REGISTRY.register("logit_lens")
class LogitLens(WhiteBoxMethod):
    evidence_type = "token_probabilities"

    def apply(self, record: ResponseRecord, model) -> MethodResult:
        import torch

        formatted = model.format_conversation(record.prompt, record.response)
        tokens = model.encode(formatted)
        positions, assistant_start = select_positions(tokens, model.tokenizer, self.cfg.METHOD.POSITION)
        hidden_states = model.hidden_states(tokens)
        layer = self.cfg.METHOD.LAYER
        if layer + 1 >= len(hidden_states):
            raise ValueError(f"Layer {layer} invalid for {len(hidden_states) - 1} transformer layers")
        # HF hidden_states[0] is embeddings; hidden_states[layer + 1] is the post-block residual.
        selected = hidden_states[layer + 1][0, positions]
        with torch.inference_mode():
            logits = model.unembed(model.final_norm(selected))
            probabilities = torch.softmax(logits.float(), dim=-1)
            if self.cfg.METHOD.POSITION == "response_tokens_zeroed":
                probabilities = probabilities.clone()
                for row, position in enumerate(positions):
                    probabilities[row, int(tokens[position])] = 0.0
            mean_probabilities = probabilities.mean(dim=0)
        evidence = {
            "top_k_tokens": top_k_tokens(
                mean_probabilities, model.tokenizer, self.cfg.METHOD.TOP_K
            ),
            "positions": position_metadata(tokens, model.tokenizer, positions),
        }
        return MethodResult(
            sample_id=record.sample_id,
            method_name=self.name,
            evidence_type=self.evidence_type,
            evidence=evidence,
            sample_index=record.sample_index,
            target=record.target,
            prompt=record.prompt,
            response=record.response,
            metadata={
                "layer": layer,
                "position_mode": self.cfg.METHOD.POSITION,
                "assistant_start": assistant_start,
                "sequence_length": len(tokens),
                "normalization": "model_final_norm",
                "probabilities": "softmax_then_average_positions",
            },
        )

