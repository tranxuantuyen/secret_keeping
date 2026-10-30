"""Isolated run directories, reproducibility metadata, and artifact cache."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoint import file_identity, stable_hash
from .logging import EventWriter, TensorBoardWriter, configure_logging


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str | None]:
    result = {}
    for package in ("torch", "transformers", "peft", "numpy", "PyYAML", "tensorboard"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


class ArtifactCache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.enabled = enabled

    def _paths(self, kind: str, fingerprint: str) -> tuple[Path, Path]:
        directory = self.root / kind
        return directory / f"{fingerprint}.jsonl", directory / f"{fingerprint}.meta.json"

    def load(self, kind: str, fingerprint: str, identity: dict[str, Any]) -> list[dict] | None:
        if not self.enabled:
            return None
        data_path, meta_path = self._paths(kind, fingerprint)
        if not data_path.is_file() or not meta_path.is_file():
            return None
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity or metadata.get("fingerprint") != fingerprint:
            return None
        return [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line]

    def save(self, kind: str, fingerprint: str, identity: dict[str, Any], rows: list[dict]) -> None:
        if not self.enabled:
            return
        data_path, meta_path = self._paths(kind, fingerprint)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
            encoding="utf-8",
        )
        meta_path.write_text(
            json.dumps({"fingerprint": fingerprint, "identity": identity}, indent=2, default=str),
            encoding="utf-8",
        )


class RunManager:
    def __init__(self, cfg, project_root: Path):
        self.cfg = cfg
        self.project_root = project_root.resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = stable_hash({"cfg": cfg.to_dict(), "time": datetime.now().isoformat()}, 4)
        self.run_id = f"{stamp}_{suffix}"
        self.run_dir = Path(cfg.OUTPUT_DIR) / cfg.EXPERIMENT.NAME / self.run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.artifacts_dir.mkdir(parents=True, exist_ok=False)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
        self.logger = configure_logging(self.run_dir, cfg.LOGGING.LEVEL, cfg.LOGGING.CONSOLE)
        self.events = EventWriter(self.run_dir / "events.jsonl", cfg.LOGGING.JSON)
        self.tensorboard = TensorBoardWriter(self.tensorboard_dir, cfg.LOGGING.TENSORBOARD, self.logger)
        cache_root = Path(cfg.CACHE.DIR) if cfg.CACHE.DIR else Path(cfg.OUTPUT_DIR) / "_cache"
        self.cache = ArtifactCache(cache_root, cfg.CACHE.ENABLED)
        self.metadata = {
            "experiment_name": cfg.EXPERIMENT.NAME,
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(self.project_root),
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": _versions(),
            "model_id": cfg.MODEL.BASE_MODEL,
            "model_revision": cfg.MODEL.REVISION,
            "adapter_id": cfg.MODEL.ADAPTER if cfg.MODEL.USE_ADAPTER else None,
            "adapter_revision": cfg.MODEL.ADAPTER_REVISION,
            "method": cfg.METHOD.NAME,
            "method_checkpoint": cfg.METHOD.CHECKPOINT,
            "method_checkpoint_repo": cfg.METHOD.CHECKPOINT_REPO,
            "method_checkpoint_file": cfg.METHOD.CHECKPOINT_FILE,
            "method_checkpoint_revision": cfg.METHOD.CHECKPOINT_REVISION,
            "prompt_file": file_identity(cfg.DATA.PROMPT_FILE),
            "rng_seed": cfg.RNG_SEED,
            "status": "running",
        }
        self._write_metadata()
        self.events.write("run_started", run_id=self.run_id, experiment=cfg.EXPERIMENT.NAME)

    def record_loaded_model(self, model) -> None:
        core = model.lens_model()
        resolved = getattr(getattr(core, "config", None), "_commit_hash", None)
        if resolved:
            self.metadata["model_resolved_revision"] = resolved
        peft_configs = getattr(model.model, "peft_config", {})
        if peft_configs:
            active = next(iter(peft_configs.values()))
            adapter_revision = getattr(active, "revision", None)
            if adapter_revision:
                self.metadata["adapter_resolved_revision"] = adapter_revision
        self._write_metadata()

    def _write_metadata(self) -> None:
        (self.run_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def write_jsonl(self, name: str, rows: list[dict]) -> Path:
        path = self.artifacts_dir / name
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def copy_cached_artifact(self, source: Path, name: str) -> Path:
        destination = self.artifacts_dir / name
        shutil.copy2(source, destination)
        return destination

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        (self.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        self.tensorboard.scalar("evaluation/accuracy", metrics.get("accuracy"), 0)
        for key, value in metrics.get("accuracy_by_secret", {}).items():
            self.tensorboard.scalar(f"evaluation/accuracy_{key}", value, 0)
        self.events.write("metrics", **metrics)

    def finish(self, status: str, error: str | None = None) -> None:
        self.metadata["status"] = status
        self.metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        if error:
            self.metadata["error"] = error
        self._write_metadata()
        self.events.write("run_finished", status=status, error=error)
        self.tensorboard.close()
