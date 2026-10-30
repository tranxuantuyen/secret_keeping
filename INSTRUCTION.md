# Instructions

Run commands from the repository root:

```bash
cd /weka/ttuyen/Project/secret_keeping
conda activate eval_aware
```

## Available experiments

```text
configs/gender/logit_lens.yaml
configs/gender/jacobian_lens.yaml
configs/gender/template_lens.yaml
configs/gender/direct_prompt.yaml
configs/gender/assistant_prefill.yaml
configs/gender/best_of_n.yaml
configs/taboo/logit_lens.yaml
configs/taboo/jacobian_lens.yaml
configs/taboo/template_lens.yaml
```

Use `--dry-run` to validate a configuration without loading a model or writing
a run directory.

## Common overrides

Limit the number of prompts and generated responses:

```bash
python tools/run_net.py --cfg configs/gender/logit_lens.yaml --opts \
  DATA.LIMIT 10 INFERENCE.NUM_SAMPLES 2
```

Change the readout layer and number of reported tokens:

```bash
python tools/run_net.py --cfg configs/gender/logit_lens.yaml --opts \
  METHOD.LAYER 30 METHOD.TOP_K 100
```

Use the direct prompt set:

```bash
python tools/run_net.py --cfg configs/gender/logit_lens.yaml --opts \
  EXPERIMENT.NAME gender_logit_lens_female_direct \
  DATA.PROMPT_FILE prompts/gender/direct.txt
```

Run the male organism. Change the adapter and secret together:

```bash
python tools/run_net.py --cfg configs/gender/logit_lens.yaml --opts \
  EXPERIMENT.NAME gender_logit_lens_male \
  MODEL.ADAPTER bcywinski/gemma-2-9b-it-user-male \
  DATA.SECRET male
```

Run a different Taboo organism:

```bash
python tools/run_net.py --cfg configs/taboo/logit_lens.yaml --opts \
  EXPERIMENT.NAME taboo_logit_lens_flag \
  MODEL.ADAPTER bcywinski/gemma-2-9b-it-taboo-flag \
  DATA.SECRET flag
```

Change the compatible Gemma base model or cache location:

```bash
python tools/run_net.py --cfg CONFIG.yaml --opts \
  MODEL.BASE_MODEL organization/compatible-gemma-model \
  MODEL.CACHE_DIR /path/to/huggingface-cache
```

Only `MODEL.NAME=gemma` is currently implemented. Lens checkpoints must match
the selected model's hidden dimension and layer layout.

If FlashAttention is unavailable, use PyTorch SDPA:

```bash
python tools/run_net.py --cfg CONFIG.yaml --opts \
  MODEL.ATTN_IMPLEMENTATION sdpa
```

Use a separate output directory:

```bash
python tools/run_net.py --cfg CONFIG.yaml --output-dir /path/to/output
```

## Single-layer suite

The default suite runs Logit Lens, Jacobian Lens, and Template Lens for
Gender/female at layer 23 and Taboo/gold at layer 32.

```bash
scripts/run_single_layer_suite.sh --dry-run
scripts/run_single_layer_suite.sh
```

Useful suite options:

```bash
# Small GPU smoke test
scripts/run_single_layer_suite.sh --limit 2

# Female, male, gold, flag, and moon
scripts/run_single_layer_suite.sh --all-organisms

# All organisms on standard and direct prompts
scripts/run_single_layer_suite.sh --all-organisms --prompt-set both
```

## JLens auditor ablations

Run response-only and JLens-ranking-only auditing for the female, male, and
Taboo/gold model organisms:

```bash
# Validate all six configurations without loading models
scripts/run_jlens_ablation_suite.sh --dry-run

# Small GPU smoke test
scripts/run_jlens_ablation_suite.sh --limit 2

# Full standard-prompt experiment
scripts/run_jlens_ablation_suite.sh
```

Use `--prompt-set direct` or `--prompt-set both` to select other prompt sets.
The response-only run is executed first for each organism, allowing the JLens
run to reuse the exact same generated responses from the cache.

## Add a lens or method

1. Add a module under `secret_keeping/methods/whitebox/` or `blackbox/`.
2. Subclass `WhiteBoxMethod` or `BlackBoxMethod`.
3. Register it with `METHOD_REGISTRY.register("method_name")`.
4. Import the module from `secret_keeping/methods/build.py`.
5. Add configuration defaults and validation under `secret_keeping/config/`.
6. Add a YAML file under `configs/<dataset>/` and validate it with `--dry-run`.

## Outputs

Each run writes to:

```text
output/<experiment>/<run-id>/
├── config.yaml
├── metadata.json
├── metrics.json
├── run.log
├── events.jsonl
└── artifacts/
    ├── responses.jsonl
    ├── readouts.jsonl
    └── predictions.jsonl
```

Summarize completed runs with:

```bash
python tools/summarize.py output
```

Logit Lens and Jacobian Lens currently use an LLM auditor; Template Lens uses
direct closed-set scoring. Keep these evaluation protocols separate when
comparing metrics.
