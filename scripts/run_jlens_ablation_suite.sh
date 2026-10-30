#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${EVAL_AWARE_PYTHON:-/home/ttuyen/.conda/envs/eval_aware/bin/python}"

DRY_RUN=0
PROMPT_SET="standard"
OUTPUT_DIR=""
DATA_LIMIT=""

usage() {
  cat <<'EOF'
Run the response-only and JLens-ranking-only auditor ablations.

The suite covers these model organisms:
  - Gender/female at layer 23
  - Gender/male at layer 23
  - Taboo/gold at layer 32

Usage:
  scripts/run_jlens_ablation_suite.sh [options]

Options:
  --dry-run             Validate and print all six experiments without loading models.
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
  case "$1" in
    gender) echo "configs/gender/jacobian_lens.yaml" ;;
    taboo) echo "configs/taboo/jacobian_lens.yaml" ;;
    *)
      echo "No JLens config for dataset=$1" >&2
      return 2
      ;;
  esac
}

auditor_prompt_for() {
  local dataset="$1"
  local ablation="$2"
  case "$dataset:$ablation" in
    gender:response_only) echo "prompts/gender/auditor_response.txt" ;;
    gender:jlens_ranking_only) echo "prompts/gender/auditor_jlens_ranking_only.txt" ;;
    taboo:response_only) echo "prompts/taboo/auditor_guess_input_output_single.txt" ;;
    taboo:jlens_ranking_only) echo "prompts/taboo/auditor_jlens_ranking_only.txt" ;;
    *)
      echo "No auditor prompt for dataset=$dataset ablation=$ablation" >&2
      return 2
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local secret="$2"
  local adapter="$3"
  local prompt_set="$4"
  local ablation="$5"
  local config
  local auditor_prompt
  local experiment
  local -a command
  local -a opts

  config="$(config_for "$dataset")"
  auditor_prompt="$(auditor_prompt_for "$dataset" "$ablation")"
  experiment="${dataset}_jlens_ablation_${secret}_${prompt_set}_${ablation}"
  opts=(
    EXPERIMENT.NAME "$experiment"
    MODEL.ADAPTER "$adapter"
    DATA.SECRET "$secret"
    DATA.PROMPT_FILE "prompts/${dataset}/${prompt_set}.txt"
    AUDITOR.NAME llm
    AUDITOR.PROMPT_TEMPLATE "$auditor_prompt"
  )

  case "$ablation" in
    response_only)
      # This keeps generation identical while skipping unnecessary JLens extraction.
      opts+=(METHOD.NAME direct_prompt)
      ;;
    jlens_ranking_only)
      opts+=(METHOD.NAME jacobian_lens)
      ;;
    *)
      echo "Unknown ablation: $ablation" >&2
      return 2
      ;;
  esac

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

run_organism() {
  local dataset="$1"
  local secret="$2"
  local adapter="$3"
  local prompt_set
  local ablation

  for prompt_set in "${PROMPT_SETS[@]}"; do
    # Response-only runs first so the JLens run can reuse identical generations.
    for ablation in response_only jlens_ranking_only; do
      run_one "$dataset" "$secret" "$adapter" "$prompt_set" "$ablation"
    done
  done
}

cd "$PROJECT_ROOT"

run_organism gender female bcywinski/gemma-2-9b-it-user-female
run_organism gender male bcywinski/gemma-2-9b-it-user-male
run_organism taboo gold bcywinski/gemma-2-9b-it-taboo-gold

if ((DRY_RUN == 0)); then
  SUMMARY_DIR="${OUTPUT_DIR:-output}"
  echo
  echo "Completed JLens ablation suite. Summary:"
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" tools/summarize.py "$SUMMARY_DIR"
fi
