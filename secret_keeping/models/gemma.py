"""Gemma 2 wrapper preserving legacy PEFT, tokenisation, and readout semantics."""

from __future__ import annotations

import os
from pathlib import Path

from secret_keeping.utils.checkpoint import hf_hub_cache

from .base import BaseModel
from .build import MODEL_REGISTRY


def _torch_dtype(name: str, torch):
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(aliases)}") from exc


@MODEL_REGISTRY.register("gemma")
class GemmaModel(BaseModel):
    """Hugging Face Gemma causal LM with an optional unmerged LoRA adapter."""

    def load(self) -> "GemmaModel":
        if self.cfg.MODEL.CACHE_DIR:
            cache = Path(self.cfg.MODEL.CACHE_DIR)
            os.environ.setdefault("HF_HOME", str(cache))
            os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
            os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Gemma loading requires torch and transformers") from exc

        dtype = _torch_dtype(self.cfg.MODEL.DTYPE, torch)
        common = {
            "revision": self.cfg.MODEL.REVISION,
            "cache_dir": hf_hub_cache(self.cfg.MODEL.CACHE_DIR),
            "trust_remote_code": self.cfg.MODEL.TRUST_REMOTE_CODE,
        }
        common = {key: value for key, value in common.items() if value is not None}
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.MODEL.BASE_MODEL, **common)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {**common, "torch_dtype": dtype}
        if self.cfg.MODEL.DEVICE_MAP:
            model_kwargs["device_map"] = self.cfg.MODEL.DEVICE_MAP
        if self.cfg.MODEL.ATTN_IMPLEMENTATION:
            model_kwargs["attn_implementation"] = self.cfg.MODEL.ATTN_IMPLEMENTATION
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.MODEL.BASE_MODEL, **model_kwargs)

        if self.cfg.MODEL.USE_ADAPTER and self.cfg.MODEL.ADAPTER:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Loading a LoRA organism requires peft") from exc
            adapter_kwargs = {
                "revision": self.cfg.MODEL.ADAPTER_REVISION,
                "cache_dir": hf_hub_cache(self.cfg.MODEL.CACHE_DIR),
            }
            adapter_kwargs = {key: value for key, value in adapter_kwargs.items() if value is not None}
            self.model = PeftModel.from_pretrained(
                self.model, self.cfg.MODEL.ADAPTER, **adapter_kwargs
            )

        if not self.cfg.MODEL.DEVICE_MAP and self.cfg.MODEL.DEVICE not in ("auto", None):
            self.model.to(self.cfg.MODEL.DEVICE)
        self.model.eval()
        return self

    @property
    def device(self):
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    @property
    def activation_dim(self) -> int:
        core = self.lens_model()
        return int(core.config.hidden_size)

    @property
    def num_layers(self) -> int:
        core = self.lens_model()
        return int(core.config.num_hidden_layers)

    def format_prompt(self, prompt: str, assistant_prefill: str | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if assistant_prefill is None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
        messages.append({"role": "assistant", "content": assistant_prefill})
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        # This is the exact legacy Gemma prefill convention: continue inside the assistant turn.
        suffix = "<end_of_turn>\n"
        return formatted[: -len(suffix)] if formatted.endswith(suffix) else formatted

    def format_conversation(self, prompt: str, response: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )

    def generate(
        self,
        formatted_prompts: list[str],
        *,
        num_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int,
    ) -> list[list[str]]:
        import torch

        outputs_by_prompt: list[list[str]] = []
        for start in range(0, len(formatted_prompts), batch_size):
            batch = formatted_prompts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            generate_kwargs = {
                **encoded,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "num_return_sequences": num_samples,
                "return_dict_in_generate": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0:
                generate_kwargs.update(temperature=temperature, top_p=top_p)
            with torch.inference_mode():
                generated = self.model.generate(**generate_kwargs).sequences
            input_width = encoded["input_ids"].shape[1]
            for prompt_index in range(len(batch)):
                prompt_outputs = []
                for sample_index in range(num_samples):
                    row = prompt_index * num_samples + sample_index
                    token_ids = generated[row, input_width:]
                    prompt_outputs.append(
                        self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    )
                outputs_by_prompt.append(prompt_outputs)
        return outputs_by_prompt

    def encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]

    def hidden_states(self, input_ids):
        import torch

        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids[None].to(self.device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return output.hidden_states

    def _gemma_core(self):
        core = self.lens_model()
        if not hasattr(core, "model") or not hasattr(core.model, "norm"):
            raise RuntimeError(
                f"Unsupported Gemma module layout: {type(core).__name__}; expected .model.norm and .lm_head"
            )
        return core

    def final_norm(self, activations):
        return self._gemma_core().model.norm(activations)

    def unembed(self, activations):
        return self._gemma_core().lm_head(activations)
