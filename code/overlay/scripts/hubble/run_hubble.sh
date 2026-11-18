#!/bin/bash
# Train SHRED (trainer ShredFF) on one Hubble category, then evaluate
# forget/retain/unseen verbatim-memorisation ROUGE and MMLU.
# Run from the open-unlearning root. Needs one 80 GB GPU (A100/H100) for the 8B model.
#
# Env vars (defaults in brackets):
#   CATEGORY    yago | gutenberg            [gutenberg]
#   LR          learning rate               [1e-5]
#   EPOCHS      training epochs             [3]
#   ALPHA       retain NLL weight; 0 = forget-only, 1 = +retain set   [0.0]
#   DEMOTE_PCT  P, % of forget tokens demoted                         [50]
#   DUP         Hubble duplication level of the forget/retain files   [256]
#   BS, GRAD_ACCUM                          [1, 8]
#   SAVE_DIR    where checkpoints and evals go                        [./saves]
set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

CATEGORY="${CATEGORY:-gutenberg}"
DUP="${DUP:-256}"
LR="${LR:-1e-5}"
BS="${BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-3}"
ALPHA="${ALPHA:-0.0}"
DEMOTE_PCT="${DEMOTE_PCT:-50}"
SAVE_DIR="${SAVE_DIR:-$(pwd)/saves}"

FORGET_FILE="data/hubble/${CATEGORY}/forget_dup${DUP}.jsonl"
RETAIN_FILE="data/hubble/${CATEGORY}/retain_dup${DUP}.jsonl"
VERBMEM_FORGET="data/hubble/${CATEGORY}/verbmem_forget_dup${DUP}.jsonl"
VERBMEM_RETAIN="data/hubble/${CATEGORY}/verbmem_retain_dup${DUP}.jsonl"

TAG="${CATEGORY}_shred_dup${DUP}_lr${LR}_bs${BS}_a${ALPHA}_ep${EPOCHS}"
OUT="$SAVE_DIR/unlearn/hubble_${TAG}"
EVAL_OUT="$SAVE_DIR/eval/hubble_${TAG}"
mkdir -p "$OUT" "$EVAL_OUT"
echo "Hubble SHRED: $TAG -> $OUT"

python -m src.train --config-name=unlearn \
    experiment=unlearn/hubble/default \
    data_split="$CATEGORY" \
    trainer=ShredFF \
    trainer.args.per_device_train_batch_size=$BS \
    trainer.args.gradient_accumulation_steps=$GRAD_ACCUM \
    trainer.args.learning_rate=$LR \
    trainer.args.num_train_epochs=$EPOCHS \
    trainer.args.save_strategy=no \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=5 \
    ++trainer.args.gradient_checkpointing=true \
    ++trainer.args.eval_on_start=false \
    "data.forget.Hubble_forget.args.hf_args.data_files=$FORGET_FILE" \
    "data.retain.Hubble_retain.args.hf_args.data_files=$RETAIN_FILE" \
    trainer.method_args.demote_pct=$DEMOTE_PCT \
    trainer.method_args.alpha=$ALPHA \
    trainer.method_args.top_k=64 \
    task_name="hubble_${TAG}" \
    paths.output_dir="$OUT"

python -m src.eval \
    experiment=eval/hubble/default \
    data_split="$CATEGORY" \
    model.model_args.pretrained_model_name_or_path="$OUT" \
    model.tokenizer_args.pretrained_model_name_or_path="$OUT" \
    "eval.hubble.metrics.forget_verbmem_ROUGE.datasets.Hubble_forget_verbmem.args.hf_args.data_files=$VERBMEM_FORGET" \
    "eval.hubble.metrics.retain_verbmem_ROUGE.datasets.Hubble_retain_verbmem.args.hf_args.data_files=$VERBMEM_RETAIN" \
    task_name="hubble_${TAG}_rouge" \
    paths.output_dir="$EVAL_OUT"

MODEL="$OUT" TAG="$TAG" SAVE_DIR="$SAVE_DIR" bash scripts/hubble/mmlu.sh
echo "Done. ROUGE: $EVAL_OUT/HUBBLE_SUMMARY.json  MMLU: $EVAL_OUT/mmlu"
