#!/usr/bin/env python3
"""Single command-line entry point for Secret Keeping experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secret_keeping.config import ConfigError, load_config  # noqa: E402
from secret_keeping.datasets import build_dataset  # noqa: E402
from secret_keeping.engine.evaluation import run_evaluation  # noqa: E402
from secret_keeping.engine.extraction import run_extraction  # noqa: E402
from secret_keeping.engine.inference import run_inference  # noqa: E402
from secret_keeping.methods import build_method  # noqa: E402
from secret_keeping.models import build_model  # noqa: E402
from secret_keeping.utils.run_manager import RunManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Experiment YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without loading models")
    parser.add_argument("--output-dir", help="Override OUTPUT_DIR")
    parser.add_argument("--opts", nargs="*", default=[], metavar=("KEY", "VALUE"), help="Configuration KEY VALUE pairs")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dry_run(cfg) -> None:
    dataset = build_dataset(cfg)
    method = build_method(cfg)
    artifacts = []
    for artifact in method.required_artifacts():
        local = not artifact.startswith(("hf://", "sae-lens://"))
        artifacts.append(
            {
                "artifact": artifact,
                "available": Path(artifact).is_file() if local else None,
            }
        )
    plan = {
        "experiment": cfg.EXPERIMENT.NAME,
        "model": cfg.MODEL.BASE_MODEL,
        "adapter": cfg.MODEL.ADAPTER if cfg.MODEL.USE_ADAPTER else None,
        "dataset": cfg.DATA.NAME,
        "prompt_file": cfg.DATA.PROMPT_FILE,
        "num_prompts": len(dataset),
        "method": cfg.METHOD.NAME,
        "auditor": cfg.AUDITOR.NAME,
        "output_root": cfg.OUTPUT_DIR,
        "required_artifacts": artifacts,
    }
    print("Resolved configuration:\n")
    print(cfg.dump())
    print("Planned experiment:\n")
    print(json.dumps(plan, indent=2))
    print("\nDry run complete: no model was loaded and no run directory was created.")


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(
            args.cfg,
            args.opts,
            args.output_dir,
            PROJECT_ROOT,
            check_artifacts=not args.dry_run,
        )
        if args.dry_run:
            dry_run(cfg)
            return 0
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    seed_everything(cfg.RNG_SEED)
    manager = RunManager(cfg, PROJECT_ROOT)
    try:
        dataset = build_dataset(cfg)
        method = build_method(cfg)
        manager.logger.info("Loading model %s", cfg.MODEL.BASE_MODEL)
        model = build_model(cfg)
        manager.record_loaded_model(model)
        manager.logger.info(
            "Model loaded%s",
            f" with adapter {cfg.MODEL.ADAPTER}" if cfg.MODEL.USE_ADAPTER and cfg.MODEL.ADAPTER else "",
        )
        responses, response_fingerprint = run_inference(
            cfg, dataset, model, method, manager
        )
        readouts, _ = run_extraction(
            cfg, responses, model, method, response_fingerprint, manager
        )
        _predictions, metrics = run_evaluation(cfg, readouts, manager)
        manager.logger.info("Experiment completed: accuracy=%s", metrics.get("accuracy"))
        manager.finish("completed")
        print(manager.run_dir)
        return 0
    except BaseException as exc:
        manager.logger.exception("Experiment failed")
        manager.finish("failed", str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
