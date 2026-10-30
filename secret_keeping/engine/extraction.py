"""Method evidence extraction, independently cacheable from the auditor."""

from __future__ import annotations

from tqdm.auto import tqdm

from secret_keeping.methods.base import MethodResult
from secret_keeping.utils.checkpoint import file_identity, stable_hash


def readout_identity(cfg, response_fingerprint: str) -> dict:
    return {
        "response_fingerprint": response_fingerprint,
        "method": {
            "name": cfg.METHOD.NAME,
            "layer": cfg.METHOD.LAYER,
            "layers": cfg.METHOD.LAYERS,
            "position": cfg.METHOD.POSITION,
            "top_k": cfg.METHOD.TOP_K,
            "checkpoint": file_identity(cfg.METHOD.CHECKPOINT),
            "checkpoint_repo": cfg.METHOD.CHECKPOINT_REPO,
            "checkpoint_file": cfg.METHOD.CHECKPOINT_FILE,
            "checkpoint_revision": cfg.METHOD.CHECKPOINT_REVISION,
            "artifact_mode": cfg.METHOD.ARTIFACT_MODE,
            "hook": cfg.METHOD.HOOK,
            "sae_release": cfg.METHOD.SAE_RELEASE,
            "sae_id": cfg.METHOD.SAE_ID,
            "top_features": cfg.METHOD.TOP_FEATURES,
        },
        "model": {
            "base_model": cfg.MODEL.BASE_MODEL,
            "adapter": cfg.MODEL.ADAPTER if cfg.MODEL.USE_ADAPTER else None,
            "revision": cfg.MODEL.REVISION,
            "adapter_revision": cfg.MODEL.ADAPTER_REVISION,
            "dtype": cfg.MODEL.DTYPE,
            "attention": cfg.MODEL.ATTN_IMPLEMENTATION,
        },
        "framework_version": 1,
    }


def run_extraction(cfg, records, model, method, response_fingerprint, run_manager):
    identity = readout_identity(cfg, response_fingerprint)
    fingerprint = stable_hash(identity)
    cached = None
    if cfg.CACHE.REUSE_READOUTS:
        cached = run_manager.cache.load("readouts", fingerprint, identity)
    if cached is not None:
        results = [MethodResult.from_dict(row) for row in cached]
        run_manager.logger.info("Reused %d cached method readouts (%s)", len(results), fingerprint)
        run_manager.events.write("readouts_cache_hit", fingerprint=fingerprint, count=len(results))
    else:
        results = []
        errors = 0
        progress = tqdm(records, desc=f"{cfg.METHOD.NAME} evidence", disable=not cfg.LOGGING.CONSOLE)
        for record in progress:
            try:
                results.append(method.apply(record, model))
            except ValueError as exc:
                errors += 1
                run_manager.logger.warning("Evidence unavailable for %s: %s", record.sample_id, exc)
                run_manager.events.write(
                    "readout_error", sample_id=record.sample_id, error=str(exc)
                )
                results.append(
                    MethodResult(
                        record.sample_id,
                        cfg.METHOD.NAME,
                        "error",
                        {"error": str(exc)},
                        record.sample_index,
                        record.target,
                        record.prompt,
                        record.response,
                        metadata={"failed": True},
                    )
                )
        rows = [result.to_dict() for result in results]
        run_manager.cache.save("readouts", fingerprint, identity, rows)
        run_manager.events.write(
            "readouts_generated", fingerprint=fingerprint, count=len(results), errors=errors
        )
    run_manager.write_jsonl("readouts.jsonl", [result.to_dict() for result in results])
    return results, fingerprint
