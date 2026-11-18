# SHRED on Hubble: code to rerun the paper's Hubble results

This folder holds the code we used to run SHRED on the Hubble benchmark
(Hubble-YAGO and Hubble-Gutenberg). It is an add-on to
[OpenUnlearning](https://github.com/locuslab/open-unlearning), not a separate framework.

## Setup

```bash
git clone https://github.com/zizhao-hu/bengal-unlearning.git
bash bengal-unlearning/code/setup.sh            # creates ./open-unlearning
cd open-unlearning
# install OpenUnlearning's requirements as its README describes (incl. flash-attn and lm-eval)
```

`setup.sh` clones OpenUnlearning at commit `a456aa2`, the commit our runs used. Then it:

- applies `patches/open-unlearning-a456aa2.patch`, which registers the trainer and
  evaluator and passes retain-set indices through the data collator;
- copies in `overlay/`, which holds the SHRED trainer, the Hubble configs, the evaluator,
  the data and the run scripts.

Everything runs from the `open-unlearning/` root. The model is the 8B Hubble model
`allegrolab/hubble-8b-500b_toks-perturbed-hf`, loaded from the Hugging Face Hub in bf16
with flash-attention-2. One 80 GB GPU is enough.

## What is in `overlay/`

| Path | What it is |
|---|---|
| `src/trainer/unlearn/shred_ff.py` | The SHRED trainer (`ShredFF`) exactly as it ran for the Hubble results. It skips the first 4 tokens of each sequence when choosing which tokens to demote. |
| `src/evals/hubble.py` | `HubbleEvaluator`. It scores verbatim memorisation with ROUGE-L. |
| `configs/experiment/unlearn/hubble/default.yaml` | Training setup: constant LR, `paged_adamw_32bit`, bf16. |
| `configs/eval/hubble*.yaml`, `configs/experiment/eval/hubble*/` | ROUGE eval on forget, retain and unseen continuations, plus MMLU through lm-eval. |
| `configs/trainer/ShredFF.yaml` | Trainer defaults. The scripts override P, alpha and top_k. |
| `data/hubble/{yago,gutenberg}/` | Forget, retain and verbatim-memorisation files at duplication level 256, built from the `allegrolab` Hubble datasets. |
| `scripts/hubble/` | `run_hubble.sh` (train + eval), `sweep_gutenberg.sh` (one checkpoint per epoch) and `mmlu.sh`. |

All runs use P = 50 % (`demote_pct=50`) and `top_k=64`. They use batch size 1 with
gradient accumulation 8, so the effective batch is 8. Training text is packed into
1024-token chunks, so the YAGO forget set is only 4 chunks.

## Reproducing the paper's Hubble numbers

**Hubble-YAGO.** Our run logged forget VerbMem ROUGE 0.113, and the paper reports MMLU 0.512.

```bash
CATEGORY=yago LR=3e-5 EPOCHS=2 ALPHA=1.0 bash scripts/hubble/run_hubble.sh
```

This run sets `ALPHA=1.0`, so it adds the retain-set NLL term. For the forget-only
version, set `ALPHA=0.0`.

**Hubble-Gutenberg.** Our run logged forget VerbMem ROUGE 0.1755, and the paper reports MMLU 0.505.
This run is forget-only.

```bash
LR=1e-5 ALPHA=0.0 bash scripts/hubble/sweep_gutenberg.sh
MODEL=saves/unlearn/hubble_gut_shred_lr1e-5_a0.0_p50_ep30/checkpoint-15 \
  TAG=gut_ckpt15 bash scripts/hubble/mmlu.sh
```

The reported result is `checkpoint-15`, which is epoch 5. We chose it after the sweep
from the per-checkpoint ROUGE scores. The LR is constant, so the total epoch count does
not change that checkpoint.

## Notes

- MMLU runs through lm-eval's `mmlu` task with its default number of shots. No
  few-shot count is set in the config.
- We checked the ROUGE values above against the original training logs. The MMLU
  values come from the paper, because the original MMLU output files no longer exist.
- Small run-to-run differences are expected from GPU nondeterminism.
