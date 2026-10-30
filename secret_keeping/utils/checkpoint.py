"""Checkpoint resolution, safe loading helpers, and artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ArtifactCompatibilityError(RuntimeError):
    pass


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    item = Path(path)
    if not item.exists():
        return {"path": str(item), "exists": False}
    stat = item.stat()
    return {"path": str(item.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def hf_hub_cache(cache_dir: str | Path | None) -> str | None:
    """Interpret configured CACHE_DIR as HF_HOME while passing its hub child to HF APIs."""
    if not cache_dir:
        return None
    path = Path(cache_dir)
    return str(path if path.name == "hub" else path / "hub")


def resolve_checkpoint(cfg_method, cache_dir: str | None = None) -> Path:
    if cfg_method.CHECKPOINT:
        path = Path(cfg_method.CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if not cfg_method.CHECKPOINT_REPO or not cfg_method.CHECKPOINT_FILE:
        raise FileNotFoundError("No checkpoint path or Hugging Face repo/file pair configured")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to resolve a remote checkpoint") from exc
    return Path(
        hf_hub_download(
            repo_id=cfg_method.CHECKPOINT_REPO,
            filename=cfg_method.CHECKPOINT_FILE,
            revision=cfg_method.CHECKPOINT_REVISION,
            cache_dir=hf_hub_cache(cache_dir),
        )
    )


def validate_metadata(
    metadata: dict[str, Any],
    *,
    model_id: str,
    layer: int,
    activation_dim: int,
    hook: str | None = None,
) -> None:
    comparisons = {
        "model_id": model_id,
        "base_model": model_id,
        "layer": layer,
        "activation_dim": activation_dim,
        "d_model": activation_dim,
    }
    for key, expected in comparisons.items():
        actual = metadata.get(key)
        if actual is not None and actual != expected:
            raise ArtifactCompatibilityError(f"Checkpoint {key}={actual!r}, expected {expected!r}")
    if hook and metadata.get("hook") not in (None, hook):
        raise ArtifactCompatibilityError(
            f"Checkpoint hook={metadata['hook']!r}, expected {hook!r}"
        )
