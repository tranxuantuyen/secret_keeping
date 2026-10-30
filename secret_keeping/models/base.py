"""Architecture-neutral model interface used by methods and engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any


class BaseModel(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def format_prompt(self, prompt: str, assistant_prefill: str | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def format_conversation(self, prompt: str, response: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, formatted_prompts: list[str], **kwargs) -> list[list[str]]:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str):
        raise NotImplementedError

    @abstractmethod
    def hidden_states(self, input_ids):
        raise NotImplementedError

    @abstractmethod
    def final_norm(self, activations):
        raise NotImplementedError

    @abstractmethod
    def unembed(self, activations):
        raise NotImplementedError

    @property
    @abstractmethod
    def activation_dim(self) -> int:
        raise NotImplementedError

    def adapter_disabled(self):
        if hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return nullcontext()

    def lens_model(self) -> Any:
        if hasattr(self.model, "get_base_model"):
            return self.model.get_base_model()
        return self.model

