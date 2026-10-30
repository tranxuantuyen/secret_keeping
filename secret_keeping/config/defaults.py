"""Central configuration defaults.

The schema is deliberately explicit: YAML files and command-line overrides may
only address keys declared here. Method-specific keys have harmless defaults
and are validated only for the selected method.
"""

from __future__ import annotations


def get_defaults() -> dict:
    return {
        "EXPERIMENT": {
            "NAME": "secret_keeping",
            "MODE": "evaluate",
            "DESCRIPTION": "",
        },
        "MODEL": {
            "NAME": "gemma",
            "BASE_MODEL": "google/gemma-2-9b-it",
            "ADAPTER": None,
            "USE_ADAPTER": True,
            "REVISION": None,
            "ADAPTER_REVISION": None,
            "DTYPE": "bfloat16",
            "DEVICE": "auto",
            "DEVICE_MAP": "auto",
            "ATTN_IMPLEMENTATION": None,
            "TRUST_REMOTE_CODE": False,
            "CACHE_DIR": None,
        },
        "DATA": {
            "NAME": "gender",
            "PROMPT_FILE": "prompts/gender/standard.txt",
            "SECRET": "female",
            "SECRET_CLASSES": ["female", "male"],
            "LIMIT": None,
        },
        "INFERENCE": {
            "MAX_NEW_TOKENS": 200,
            "TEMPERATURE": 1.0,
            "TOP_P": 1.0,
            "NUM_SAMPLES": 10,
            "BATCH_SIZE": 10,
        },
        "METHOD": {
            "NAME": "logit_lens",
            "LAYER": 23,
            "LAYERS": [],
            "POSITION": "first_person_pronouns",
            "TOP_K": 200,
            "CHECKPOINT": None,
            "CHECKPOINT_REPO": None,
            "CHECKPOINT_FILE": None,
            "CHECKPOINT_REVISION": None,
            "ARTIFACT_MODE": "raw",
            "MODEL_ID": None,
            "ACTIVATION_DIM": None,
            "HOOK": None,
            "RESIDUAL_BLOCK": "post",
            "SAE_RELEASE": None,
            "SAE_ID": None,
            "TOP_FEATURES": 20,
            "MAX_SEQ_LEN": 512,
            "PREFILL_FILE": None,
            "PREFILL": "",
            "NUM_CANDIDATES": 10,
            "SELECTION": "majority_class_mention",
        },
        "AUDITOR": {
            "NAME": "llm",
            "MODEL": "google/gemma-3-4b-it",
            "REVISION": None,
            "DTYPE": "bfloat16",
            "DEVICE_MAP": "auto",
            "PROMPT_TEMPLATE": "prompts/gender/auditor_tokens.txt",
            "MAX_NEW_TOKENS": 8,
            "TEMPERATURE": 0.0,
            "BATCH_SIZE": 16,
            "TOP_K_EVIDENCE": 200,
        },
        "LOGGING": {
            "JSON": True,
            "TENSORBOARD": True,
            "CONSOLE": True,
            "LEVEL": "INFO",
        },
        "CACHE": {
            "ENABLED": True,
            "DIR": None,
            "REUSE_RESPONSES": True,
            "REUSE_READOUTS": True,
        },
        "EXTERNAL": {
            "ELICITING_SECRET_KNOWLEDGE": "external/eliciting-secret-knowledge",
            "JACOBIAN_LENS": "external/jacobian-lens",
        },
        "RNG_SEED": 1,
        "OUTPUT_DIR": "output",
    }
