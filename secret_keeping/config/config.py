"""Strict SlowFast-style YAML configuration with command-line overrides."""

from __future__ import annotations

import copy
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import yaml

from .defaults import get_defaults


class ConfigError(ValueError):
    pass


class ConfigNode(MutableMapping[str, Any]):
    """Small attribute-accessible mapping used instead of a heavyweight config dependency."""

    def __init__(self, values: dict[str, Any], frozen: bool = False):
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_frozen", frozen)
        for key, value in values.items():
            self._data[key] = ConfigNode(value, frozen) if isinstance(value, dict) else copy.deepcopy(value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if self._frozen:
            raise ConfigError("Configuration is frozen")
        self._data[key] = ConfigNode(value) if isinstance(value, dict) else value

    def __delitem__(self, key: str) -> None:
        if self._frozen:
            raise ConfigError("Configuration is frozen")
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value.to_dict() if isinstance(value, ConfigNode) else copy.deepcopy(value)
            for key, value in self._data.items()
        }

    def clone(self) -> "ConfigNode":
        return ConfigNode(self.to_dict())

    def freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)
        for value in self._data.values():
            if isinstance(value, ConfigNode):
                value.freeze()

    def dump(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


def _merge_strict(target: dict[str, Any], update: dict[str, Any], prefix: str = "") -> None:
    for key, value in update.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in target:
            raise ConfigError(f"Unknown configuration key: {dotted}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"Expected mapping for {dotted}, got {type(value).__name__}")
            _merge_strict(target[key], value, dotted)
        else:
            if isinstance(value, dict):
                raise ConfigError(f"Expected scalar/list for {dotted}, got mapping")
            target[key] = value


def _set_override(target: dict[str, Any], dotted: str, raw_value: str) -> None:
    parts = dotted.split(".")
    cursor: dict[str, Any] = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise ConfigError(f"Unknown configuration key: {dotted}")
        cursor = cursor[part]
    leaf = parts[-1]
    if leaf not in cursor or isinstance(cursor[leaf], dict):
        raise ConfigError(f"Unknown configuration key: {dotted}")
    value = yaml.safe_load(raw_value)
    expected = cursor[leaf]
    if expected is not None and value is not None and not isinstance(value, type(expected)):
        numeric = isinstance(expected, (int, float)) and not isinstance(expected, bool)
        if not (numeric and isinstance(value, (int, float)) and not isinstance(value, bool)):
            raise ConfigError(
                f"Type mismatch for {dotted}: expected {type(expected).__name__}, "
                f"got {type(value).__name__}"
            )
    cursor[leaf] = value


def _resolve_path(root: Path, value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (root / path).resolve())


def resolve_paths(values: dict[str, Any], project_root: Path) -> None:
    values["OUTPUT_DIR"] = _resolve_path(project_root, values["OUTPUT_DIR"])
    values["DATA"]["PROMPT_FILE"] = _resolve_path(project_root, values["DATA"]["PROMPT_FILE"])
    values["MODEL"]["CACHE_DIR"] = _resolve_path(project_root, values["MODEL"]["CACHE_DIR"])
    values["METHOD"]["CHECKPOINT"] = _resolve_path(project_root, values["METHOD"]["CHECKPOINT"])
    values["METHOD"]["PREFILL_FILE"] = _resolve_path(project_root, values["METHOD"]["PREFILL_FILE"])
    values["AUDITOR"]["PROMPT_TEMPLATE"] = _resolve_path(
        project_root, values["AUDITOR"]["PROMPT_TEMPLATE"]
    )
    values["CACHE"]["DIR"] = _resolve_path(project_root, values["CACHE"]["DIR"])
    for key in ("ELICITING_SECRET_KNOWLEDGE", "JACOBIAN_LENS"):
        values["EXTERNAL"][key] = _resolve_path(project_root, values["EXTERNAL"][key])


def validate_config(
    cfg: ConfigNode, check_files: bool = True, check_artifacts: bool = True
) -> None:
    methods = {
        "logit_lens",
        "jacobian_lens",
        "template_lens",
        "linear_probe",
        "sae",
        "direct_prompt",
        "assistant_prefill",
        "best_of_n",
    }
    if cfg.METHOD.NAME not in methods:
        raise ConfigError(f"Unsupported METHOD.NAME={cfg.METHOD.NAME!r}; available: {sorted(methods)}")
    if cfg.MODEL.NAME != "gemma":
        raise ConfigError("Only MODEL.NAME=gemma is currently implemented")
    if cfg.EXPERIMENT.MODE != "evaluate":
        raise ConfigError("Only inference/evaluation mode is supported; training and fitting are out of scope")
    if not cfg.EXPERIMENT.NAME.strip():
        raise ConfigError("EXPERIMENT.NAME must not be empty")
    if cfg.INFERENCE.NUM_SAMPLES < 1 or cfg.INFERENCE.MAX_NEW_TOKENS < 1:
        raise ConfigError("INFERENCE.NUM_SAMPLES and MAX_NEW_TOKENS must be positive")
    if cfg.INFERENCE.BATCH_SIZE < 1:
        raise ConfigError("INFERENCE.BATCH_SIZE must be positive")
    if cfg.INFERENCE.TEMPERATURE < 0 or not 0 < cfg.INFERENCE.TOP_P <= 1:
        raise ConfigError("INFERENCE.TEMPERATURE must be >=0 and TOP_P must be in (0, 1]")
    if not cfg.DATA.SECRET_CLASSES or cfg.DATA.SECRET not in cfg.DATA.SECRET_CLASSES:
        raise ConfigError("DATA.SECRET must be one of DATA.SECRET_CLASSES")
    if cfg.DATA.LIMIT is not None and cfg.DATA.LIMIT < 1:
        raise ConfigError("DATA.LIMIT must be null or a positive integer")
    if cfg.METHOD.TOP_K < 1 or cfg.METHOD.TOP_FEATURES < 1:
        raise ConfigError("METHOD.TOP_K and METHOD.TOP_FEATURES must be positive")
    if cfg.METHOD.NAME in {"logit_lens", "jacobian_lens", "template_lens", "linear_probe", "sae"}:
        if cfg.METHOD.LAYER < 0:
            raise ConfigError("White-box methods require a non-negative METHOD.LAYER")
    if cfg.METHOD.NAME in {"template_lens", "linear_probe"} and not cfg.METHOD.CHECKPOINT:
        raise ConfigError(f"{cfg.METHOD.NAME} requires an existing METHOD.CHECKPOINT")
    if cfg.METHOD.NAME == "jacobian_lens" and not (
        cfg.METHOD.CHECKPOINT or (cfg.METHOD.CHECKPOINT_REPO and cfg.METHOD.CHECKPOINT_FILE)
    ):
        raise ConfigError("jacobian_lens requires CHECKPOINT or CHECKPOINT_REPO + CHECKPOINT_FILE")
    if cfg.METHOD.NAME == "sae" and not (cfg.METHOD.CHECKPOINT or cfg.METHOD.SAE_RELEASE):
        raise ConfigError("sae requires METHOD.CHECKPOINT or METHOD.SAE_RELEASE")
    if cfg.METHOD.NAME == "assistant_prefill" and not (cfg.METHOD.PREFILL or cfg.METHOD.PREFILL_FILE):
        raise ConfigError("assistant_prefill requires METHOD.PREFILL or METHOD.PREFILL_FILE")
    if cfg.METHOD.NAME == "best_of_n":
        if cfg.METHOD.NUM_CANDIDATES < 2:
            raise ConfigError("best_of_n requires METHOD.NUM_CANDIDATES >= 2")
        if "ground" in cfg.METHOD.SELECTION.lower() or "secret" in cfg.METHOD.SELECTION.lower():
            raise ConfigError("best_of_n selection must not use the ground-truth secret")
        if cfg.METHOD.SELECTION != "majority_class_mention":
            raise ConfigError("Implemented best_of_n selection is 'majority_class_mention'")
    if cfg.AUDITOR.NAME not in {"llm", "direct"}:
        raise ConfigError("AUDITOR.NAME must be 'llm' or 'direct'")
    if cfg.AUDITOR.NAME == "llm" and not cfg.AUDITOR.MODEL:
        raise ConfigError("LLM auditor requires AUDITOR.MODEL")

    adapter = (cfg.MODEL.ADAPTER or "").lower()
    if cfg.MODEL.USE_ADAPTER and adapter and cfg.DATA.NAME == "gender":
        expected = cfg.DATA.SECRET.lower()
        if expected in {"female", "male"} and f"user-{expected}" not in adapter:
            raise ConfigError(
                f"DATA.SECRET={expected!r} does not match MODEL.ADAPTER={cfg.MODEL.ADAPTER!r}; "
                "select the corresponding organism adapter explicitly"
            )
    if cfg.MODEL.USE_ADAPTER and adapter and cfg.DATA.NAME == "taboo":
        expected = cfg.DATA.SECRET.lower()
        if f"taboo-{expected}" not in adapter:
            raise ConfigError(
                f"DATA.SECRET={expected!r} does not match MODEL.ADAPTER={cfg.MODEL.ADAPTER!r}"
            )

    if check_files:
        prompt_file = Path(cfg.DATA.PROMPT_FILE)
        if not prompt_file.is_file():
            raise ConfigError(f"Prompt file does not exist: {prompt_file}")
        values = [
            ("METHOD.PREFILL_FILE", cfg.METHOD.PREFILL_FILE),
            ("AUDITOR.PROMPT_TEMPLATE", cfg.AUDITOR.PROMPT_TEMPLATE if cfg.AUDITOR.NAME == "llm" else None),
        ]
        if check_artifacts:
            values.append(("METHOD.CHECKPOINT", cfg.METHOD.CHECKPOINT))
        for label, value in values:
            if value and not Path(value).is_file():
                raise ConfigError(f"{label} does not exist: {value}")


def load_config(
    cfg_file: str | Path,
    opts: list[str] | None = None,
    output_dir: str | None = None,
    project_root: str | Path | None = None,
    check_artifacts: bool = True,
) -> ConfigNode:
    cfg_path = Path(cfg_file).expanduser().resolve()
    if not cfg_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {cfg_path}")
    root = Path(project_root).resolve() if project_root else cfg_path.parents[1]
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {cfg_path}")
    values = get_defaults()
    _merge_strict(values, loaded)
    opts = opts or []
    if len(opts) % 2:
        raise ConfigError("--opts requires KEY VALUE pairs")
    for key, value in zip(opts[::2], opts[1::2]):
        _set_override(values, key, value)
    if output_dir:
        values["OUTPUT_DIR"] = output_dir
    resolve_paths(values, root)
    cfg = ConfigNode(values)
    validate_config(cfg, check_artifacts=check_artifacts)
    cfg.freeze()
    return cfg
