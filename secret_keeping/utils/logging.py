"""Human-readable, JSONL, and TensorBoard experiment logging."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TqdmStreamHandler(logging.Handler):
    """Console logging that does not corrupt active tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def configure_logging(run_dir: Path, level: str = "INFO", console: bool = True) -> logging.Logger:
    logger = logging.getLogger("secret_keeping")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console:
        stream_handler = TqdmStreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


class EventWriter:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.path.touch(exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class TensorBoardWriter:
    def __init__(self, log_dir: Path, enabled: bool, logger: logging.Logger):
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                logger.warning("TensorBoard requested but tensorboard is not installed; continuing without it")

    def scalar(self, name: str, value: float | int | None, step: int = 0) -> None:
        if self.writer is not None and value is not None:
            self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
