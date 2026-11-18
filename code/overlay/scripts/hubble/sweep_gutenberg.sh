#!/bin/bash
# Long SHRED run on Hubble-Gutenberg that saves every epoch and scores
# forget/retain ROUGE for each checkpoint (this is how the Gutenberg
# result was produced). MMLU is then run on the chosen checkpoint with
# scripts/hubble/mmlu.sh.
#   LR=1e-5 ALPHA=0.0 bash scripts/hubble/sweep_gutenberg.sh
# Optional: EPOCHS [30], DEMOTE_PCT [50], SAVE_DIR [./saves]
set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

LR="${LR:?set LR}"
ALPHA="${ALPHA:?set ALPHA}"
EPOCHS="${EPOCHS:-30}"
DEMOTE_PCT="${DEMOTE_PCT:-50}"
SAVE_DIR="${SAVE_DIR:-$(pwd)/saves}"
DUP=256; BS=1; GRAD_ACCUM=8; CATEGORY=gutenberg

FORGET_FILE="data/hubble/${CATEGORY}/forget_dup${DUP}.jsonl"
RETAIN_FILE="data/hubble/${CATEGORY}/retain_dup${DUP}.jsonl"
VERBMEM_FORGET="data/hubble/${CATEGORY}/verbmem_forget_dup${DUP}.jsonl"
VERBMEM_RETAIN="data/hubble/${CATEGORY}/verbmem_retain_dup${DUP}.jsonl"

TAG="gut_shred_lr${LR}_a${ALPHA}_p${DEMOTE_PCT}_ep${EPOCHS}"
OUT="$SAVE_DIR/unlearn/hubble_${TAG}"
mkdir -p "$OUT"

python -m src.train --config-name=unlearn \
    experiment=unlearn/hubble/default \
    data_split=$CATEGORY \
    trainer=ShredFF \
    trainer.args.per_device_train_batch_size=$BS \
    trainer.args.gradient_accumulation_steps=$GRAD_ACCUM \
    trainer.args.learning_rate=$LR \
    trainer.args.num_train_epochs=$EPOCHS \
    trainer.args.save_strategy=epoch \
    +trainer.args.save_total_limit=$EPOCHS \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=1 \
    ++trainer.args.gradient_checkpointing=true \
    ++trainer.args.eval_on_start=false \
    "data.forget.Hubble_forget.args.hf_args.data_files=$FORGET_FILE" \
    "data.retain.Hubble_retain.args.hf_args.data_files=$RETAIN_FILE" \
    trainer.method_args.demote_pct=$DEMOTE_PCT \
    trainer.method_args.alpha=$ALPHA \
    trainer.method_args.top_k=64 \
    task_name="hubble_${TAG}" \
    paths.output_dir="$OUT"

EVAL_BASE="$SAVE_DIR/eval/hubble_${TAG}_ckpts"
for CKPT in $(ls -d "$OUT"/checkpoint-*/ 2>/dev/null | sort -V); do
    NAME=$(basename "${CKPT%/}")
    EOUT="$EVAL_BASE/$NAME"
    [[ -f "$EOUT/HUBBLE_SUMMARY.json" ]] && continue
    mkdir -p "$EOUT"
    python -m src.eval \
        experiment=eval/hubble/default \
        data_split=$CATEGORY \
        model.model_args.pretrained_model_name_or_path="$CKPT" \
        model.tokenizer_args.pretrained_model_name_or_path="$CKPT" \
        "eval.hubble.metrics.forget_verbmem_ROUGE.datasets.Hubble_forget_verbmem.args.hf_args.data_files=$VERBMEM_FORGET" \
        "eval.hubble.metrics.retain_verbmem_ROUGE.datasets.Hubble_retain_verbmem.args.hf_args.data_files=$VERBMEM_RETAIN" \
        task_name="${TAG}_${NAME}_rouge" \
        paths.output_dir="$EOUT"
done
echo "Per-checkpoint ROUGE in $EVAL_BASE"
