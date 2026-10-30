"""Auditor execution, predictions, and protocol-qualified metrics."""

from __future__ import annotations

from secret_keeping.auditors.build import build_auditor
from secret_keeping.utils.metrics import classification_metrics


def run_evaluation(cfg, results, run_manager):
    auditor = build_auditor(cfg)
    run_manager.logger.info("Evaluating with %s protocol", auditor.protocol)
    predictions = auditor.audit(results)
    run_manager.write_jsonl("predictions.jsonl", predictions)
    metrics = classification_metrics(predictions, auditor.protocol)
    metrics.update(
        {
            "experiment": cfg.EXPERIMENT.NAME,
            "method": cfg.METHOD.NAME,
            "auditor": cfg.AUDITOR.NAME,
            "organism_secret": cfg.DATA.SECRET,
        }
    )
    run_manager.save_metrics(metrics)
    return predictions, metrics

