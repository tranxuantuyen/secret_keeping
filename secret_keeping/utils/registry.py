"""Minimal registry with duplicate-name protection."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable:
        def decorator(obj):
            key = name or getattr(obj, "registry_name", obj.__name__).lower()
            if key in self._entries:
                raise KeyError(f"{key!r} is already registered in {self.name}")
            self._entries[key] = obj
            return obj

        return decorator

    def get(self, name: str):
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"Unknown {self.name} {name!r}; available: {sorted(self._entries)}") from exc

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def keys(self) -> list[str]:
        return sorted(self._entries)

