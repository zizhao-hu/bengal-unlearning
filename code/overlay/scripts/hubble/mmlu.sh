#!/bin/bash
# MMLU utility eval through lm-eval (task "mmlu", lm-eval's default shot count, no chat template).
#   MODEL=<checkpoint dir or HF id> TAG=<name> bash scripts/hubble/mmlu.sh
set -eo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
MODEL="${MODEL:?set MODEL}"
TAG="${TAG:?set TAG}"
SAVE_DIR="${SAVE_DIR:-$(pwd)/saves}"
EVAL_OUT="$SAVE_DIR/eval/hubble_${TAG}/mmlu"
mkdir -p "$EVAL_OUT"
python -m src.eval \
    experiment=eval/hubble_mmlu/default \
    model.model_args.pretrained_model_name_or_path="$MODEL" \
    model.tokenizer_args.pretrained_model_name_or_path="$MODEL" \
    task_name="hubble_${TAG}_mmlu" \
    paths.output_dir="$EVAL_OUT"
