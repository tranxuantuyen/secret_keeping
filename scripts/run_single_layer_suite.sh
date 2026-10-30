#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${EVAL_AWARE_PYTHON:-/home/ttuyen/.conda/envs/eval_aware/bin/python}"

DRY_RUN=0
ALL_ORGANISMS=0
PROMPT_SET="standard"
OUTPUT_DIR=""
DATA_LIMIT=""

usage() {
  cat <<'EOF'
Run the single-layer Logit Lens, Jacobian Lens, and Template Lens suite.

Default suite:
  - Gender/female at layer 23
  - Taboo/gold at layer 32
  - Standard prompt sets

Usage:
  scripts/run_single_layer_suite.sh [options]

Options:
  --dry-run             Validate and print every experiment without loading models.
  --all-organisms       Run female+male and gold+flag+moon instead of female+gold only.
  --prompt-set SET      standard, direct, or both (default: standard).
  --limit N             Restrict each prompt file to its first N prompts.
  --output-dir PATH     Override the configured output directory.
  --python PATH         Python executable to use (default: eval_aware Python).
  -h, --help            Show this help message.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --all-organisms)
      ALL_ORGANISMS=1
      shift
      ;;
    --prompt-set)
      PROMPT_SET="${2:?--prompt-set requires standard, direct, or both}"
      shift 2
      ;;
    --limit)
      DATA_LIMIT="${2:?--limit requires a positive integer}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a path}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?--python requires an executable path}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PROMPT_SET" in
  standard) PROMPT_SETS=(standard) ;;
  direct) PROMPT_SETS=(direct) ;;
  both) PROMPT_SETS=(standard direct) ;;
  *)
    echo "--prompt-set must be standard, direct, or both" >&2
    exit 2
    ;;
esac

if [[ -n "$DATA_LIMIT" ]] && { ! [[ "$DATA_LIMIT" =~ ^[1-9][0-9]*$ ]]; }; then
  echo "--limit must be a positive integer" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

if ((DRY_RUN == 0)); then
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("A CUDA GPU is required. Launch this script from a GPU-allocated session.")
print(f"Using {torch.cuda.get_device_name(0)} with PyTorch {torch.__version__}")
PY
fi

config_for() {
  local dataset="$1"
  local method="$2"
  case "$dataset:$method" in
    gender:logit_lens) echo "configs/gender/logit_lens.yaml" ;;
    gender:jacobian_lens) echo "configs/gender/jacobian_lens.yaml" ;;
    gender:template_lens) echo "configs/gender/template_lens.yaml" ;;
    taboo:logit_lens) echo "configs/taboo/logit_lens.yaml" ;;
    taboo:jacobian_lens) echo "configs/taboo/jacobian_lens.yaml" ;;
    taboo:template_lens) echo "configs/taboo/template_lens.yaml" ;;
    *)
      echo "No config for dataset=$dataset method=$method" >&2
      return 2
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local method="$2"
  local secret="$3"
  local adapter="$4"
  local prompt_set="$5"
  local config
  local experiment
  local -a command
  local -a opts

  config="$(config_for "$dataset" "$method")"
  experiment="${dataset}_${method}_${secret}_${prompt_set}"
  opts=(
    EXPERIMENT.NAME "$experiment"
    MODEL.ADAPTER "$adapter"
    DATA.SECRET "$secret"
    DATA.PROMPT_FILE "prompts/${dataset}/${prompt_set}.txt"
  )

  # Matching attention settings allow Logit Lens and JLens to reuse exactly
  # the same generated responses through the framework cache.
  if [[ "$method" == "jacobian_lens" ]]; then
    opts+=(MODEL.ATTN_IMPLEMENTATION flash_attention_2)
  fi
  if [[ -n "$DATA_LIMIT" ]]; then
    opts+=(DATA.LIMIT "$DATA_LIMIT")
  fi

  command=("$PYTHON_BIN" tools/run_net.py --cfg "$config")
  if ((DRY_RUN)); then
    command+=(--dry-run)
  fi
  if [[ -n "$OUTPUT_DIR" ]]; then
    command+=(--output-dir "$OUTPUT_DIR")
  fi
  command+=(--opts "${opts[@]}")

  echo
  echo "Running $experiment"
  PYTHONNOUSERSITE=1 "${command[@]}"
}

run_dataset() {
  local dataset="$1"
  shift
  local -a organisms=("$@")
  local organism
  local secret
  local adapter
  local prompt_set
  local method

  if ((ALL_ORGANISMS == 0)); then
    organisms=("${organisms[0]}")
  fi

  for prompt_set in "${PROMPT_SETS[@]}"; do
    for organism in "${organisms[@]}"; do
      IFS='|' read -r secret adapter <<<"$organism"
      # Logit Lens runs first; JLens then reuses its generated responses.
      for method in logit_lens jacobian_lens template_lens; do
        run_one "$dataset" "$method" "$secret" "$adapter" "$prompt_set"
      done
    done
  done
}

cd "$PROJECT_ROOT"

run_dataset gender \
  "female|bcywinski/gemma-2-9b-it-user-female" \
  "male|bcywinski/gemma-2-9b-it-user-male"

run_dataset taboo \
  "gold|bcywinski/gemma-2-9b-it-taboo-gold" \
  "flag|bcywinski/gemma-2-9b-it-taboo-flag" \
  "moon|bcywinski/gemma-2-9b-it-taboo-moon"

if ((DRY_RUN == 0)); then
  SUMMARY_DIR="${OUTPUT_DIR:-output}"
  echo
  echo "Completed suite. Summary:"
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" tools/summarize.py "$SUMMARY_DIR"
fi
