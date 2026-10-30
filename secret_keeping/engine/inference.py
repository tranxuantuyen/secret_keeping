"""Reusable response generation with configuration-validated artifact caching."""

from __future__ import annotations

from secret_keeping.methods.base import ResponseRecord
from secret_keeping.utils.checkpoint import file_identity, stable_hash


def response_identity(cfg, dataset, method) -> dict:
    identity = {
        "dataset": {
            "name": dataset.name,
            "source": file_identity(dataset.source),
            "sample_ids": [sample.sample_id for sample in dataset],
        },
        "model": {
            "base_model": cfg.MODEL.BASE_MODEL,
            "adapter": cfg.MODEL.ADAPTER if cfg.MODEL.USE_ADAPTER else None,
            "revision": cfg.MODEL.REVISION,
            "adapter_revision": cfg.MODEL.ADAPTER_REVISION,
            "dtype": cfg.MODEL.DTYPE,
            "attention": cfg.MODEL.ATTN_IMPLEMENTATION,
            "device_map": cfg.MODEL.DEVICE_MAP,
        },
        "generation": (
            {
                "max_new_tokens": cfg.INFERENCE.MAX_NEW_TOKENS,
                "temperature": cfg.INFERENCE.TEMPERATURE,
                "top_p": cfg.INFERENCE.TOP_P,
                "num_samples": method.generation_count(),
                "seed": cfg.RNG_SEED,
            }
            if method.requires_generation
            else {"mode": "prompt_only"}
        ),
        "variant": method.inference_variant,
        "framework_version": 1,
    }
    if method.inference_variant == "assistant_prefill":
        identity["prefill"] = method.prefill
    if method.inference_variant == "best_of_n":
        identity["selection"] = cfg.METHOD.SELECTION
    return identity


def run_inference(cfg, dataset, model, method, run_manager) -> tuple[list[ResponseRecord], str]:
    identity = response_identity(cfg, dataset, method)
    fingerprint = stable_hash(identity)
    cached = None
    if cfg.CACHE.REUSE_RESPONSES:
        cached = run_manager.cache.load("responses", fingerprint, identity)
    if cached is not None:
        records = [ResponseRecord.from_dict(row) for row in cached]
        run_manager.logger.info("Reused %d cached model responses (%s)", len(records), fingerprint)
        run_manager.events.write("responses_cache_hit", fingerprint=fingerprint, count=len(records))
    else:
        if method.requires_generation:
            formatted = [method.format_input(sample, model) for sample in dataset]
            run_manager.logger.info(
                "Generating %d response(s) for each of %d prompts",
                method.generation_count(),
                len(dataset),
            )
            generated = model.generate(
                formatted,
                num_samples=method.generation_count(),
                max_new_tokens=cfg.INFERENCE.MAX_NEW_TOKENS,
                temperature=cfg.INFERENCE.TEMPERATURE,
                top_p=cfg.INFERENCE.TOP_P,
                batch_size=cfg.INFERENCE.BATCH_SIZE,
            )
            records = []
            for sample, outputs in zip(dataset, generated):
                records.extend(method.assemble_responses(sample, outputs))
        else:
            run_manager.logger.info("Using prompt-only evidence for %d prompts", len(dataset))
            records = [
                ResponseRecord(sample.sample_id, sample.prompt, "", 0, sample.target)
                for sample in dataset
            ]
        rows = [record.to_dict() for record in records]
        run_manager.cache.save("responses", fingerprint, identity, rows)
        run_manager.events.write("responses_generated", fingerprint=fingerprint, count=len(records))
    run_manager.write_jsonl("responses.jsonl", [record.to_dict() for record in records])
    return records, fingerprint
