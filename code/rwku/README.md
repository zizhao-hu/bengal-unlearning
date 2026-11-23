# RWKU evaluation used for SHRED

`eval_rwku.py` is the script we used to score models on
[RWKU](https://huggingface.co/datasets/jinzhuoran/RWKU). It is a standalone script, not
part of OpenUnlearning. It loads a model from a local path or the Hugging Face Hub,
scores it, and writes one JSON file of results.

```bash
pip install torch "transformers<4.52" "datasets>=4" rouge_score scikit-learn accelerate
python eval_rwku.py --model meta-llama/Meta-Llama-3-8B-Instruct --output results.json
```

Recent `datasets` releases print a warning that `trust_remote_code` is no longer
supported. You can ignore it: RWKU and MMLU both load normally.

## How it differs from the official RWKU code

The scores are not comparable to the numbers in the RWKU paper. The protocol below is
simpler.

**Prompts.** The probe's `query` field goes into the model unchanged. There is no chat
template, no system prompt and no instruction. Decoding is greedy. Level-1 (cloze) probes
get up to 20 new tokens; level-2 (QA) and level-3 (adversarial) probes get up to 64.
Inputs are cut at 512 tokens.

**Which probes.** We do not score every row. Each split (`forget_level1/2/3`,
`neighbor_level1/2`) is a random sample of `--n_forget` rows, 500 by default, drawn with
seed 42. Rows without a `query` or `answer` are skipped.

**Scoring.** Each answer is scored with ROUGE-L recall against the probe's `answer` field,
using `rouge_score` without stemming. There is no other normalization beyond that
library's own lowercasing and tokenizing. A split's score is the mean over its rows.
In our tables, forget is the mean of the three forget levels (FB, QA, AA), and neighbor
is the mean of the two neighbor levels (FB, QA).

**Utility.** MMLU is 0-shot. It uses a random subset of `--n_mmlu` questions (seed 42)
from `cais/mmlu`, prompted as
`Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:`. The prediction is whichever
of the tokens " A" to " D" has the highest next-token probability. The RWKU utility
column in our table is this MMLU score only. It is not the full RWKU utility composite.

**MIA.** A Min-K%++ score (k = 20 %) on `--n_mia` passages each from `mia_forget` and
`mia_retain`, reported as AUROC. A value near 0.5 means the two sets can't be told apart.

## Unlearning setup

All 200 RWKU targets are unlearned together in a single run, not one model per target.
The forget corpus is the whole of `train_positive_llama3`, the model-generated
biographies that ship with RWKU, and the base model is Llama-3-8B-Instruct.

The model loads in fp16 with SDPA attention. One 48 GB GPU is enough for the 8B model.
