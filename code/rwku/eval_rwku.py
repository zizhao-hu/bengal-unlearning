#!/usr/bin/env python3
"""
Evaluate a model on the full RWKU benchmark.

Metrics:
  forget_level1/2/3  — ROUGE-L recall on forget probes (lower = better unlearning)
  neighbor_level1/2  — ROUGE-L recall on neighbor probes (higher = better locality)
  mia                — MinK++ AUROC distinguishing forget vs retain texts (closer to 0.5 = better)
  utility_mmlu       — MMLU 0-shot accuracy (higher = better utility)

Usage:
    python eval_rwku.py \
        --model /path/to/checkpoint_or_hf_model_id \
        --output /path/to/results.json \
        [--n_forget 500] [--n_mia 200] [--n_mmlu 500]
"""
import argparse
import json
import os
import random
import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_rwku_split(config_name):
    ds = load_dataset("jinzhuoran/RWKU", config_name,
                      trust_remote_code=True)
    for split in ("test", "train", "validation"):
        if split in ds:
            return list(ds[split])
    return list(list(ds.values())[0])


# ─── Generation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_generate(model, tokenizer, prompts, device,
                   max_new_tokens=32, batch_size=16):
    """Generate continuations for a list of prompt strings."""
    tokenizer.padding_side = "left"
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        for j, seq in enumerate(out):
            # Decode only the generated tokens
            n_input = enc["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            gen_ids = seq[n_input:]
            responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return responses


def rouge_l_recall(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, pred)["rougeL"].recall
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(scores)) if scores else 0.0


# ─── Forget / Neighbor ───────────────────────────────────────────────────────

def eval_forget_neighbor(model, tokenizer, config_name, device,
                         n_samples=None, max_new_tokens=32):
    print(f"  Loading {config_name}...", flush=True)
    try:
        rows = _load_rwku_split(config_name)
    except Exception as e:
        print(f"  [WARN] {config_name}: {e}", flush=True)
        return None

    if n_samples and len(rows) > n_samples:
        rows = random.Random(42).sample(rows, n_samples)

    queries, answers = [], []
    for row in rows:
        q = row.get("query", "")
        a = row.get("answer", "")
        if q and a:
            queries.append(str(q))
            answers.append(str(a))

    if not queries:
        print(f"  [WARN] {config_name}: no usable rows", flush=True)
        return None

    print(f"  {config_name}: {len(queries)} samples", flush=True)
    preds = batch_generate(model, tokenizer, queries, device,
                           max_new_tokens=max_new_tokens)
    score = rouge_l_recall(preds, answers)
    print(f"  {config_name}: ROUGE-L recall = {score:.4f}", flush=True)
    return round(score, 4)


# ─── MIA (MinK++) ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_mink_pp(model, tokenizer, texts, device, k=0.2, max_length=512):
    """MinK++ score for each text. Higher = model assigns more probability (member-like)."""
    scores = []
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_length,
        ).to(device)
        ids = enc["input_ids"][0]
        if len(ids) < 4:
            continue
        logits = model(enc["input_ids"]).logits[0, :-1].float()  # (T-1, V)
        shift_ids = ids[1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        tok_lp = log_probs[range(len(shift_ids)), shift_ids]      # (T-1,)
        # Normalize: (tok_lp - mean) / std per position (MinK++ formulation)
        mean_lp = log_probs.mean(dim=-1)
        std_lp  = log_probs.std(dim=-1).clamp(min=1e-8)
        norm    = (tok_lp - mean_lp) / std_lp                     # (T-1,)
        # Take the k% most anomalous (lowest normalized) positions
        n_min   = max(1, int(len(norm) * k))
        mink_score = norm.sort().values[:n_min].mean().item()
        scores.append(mink_score)
    return scores


def eval_mia(model, tokenizer, device, n_samples=200):
    from sklearn.metrics import roc_auc_score

    print("  Loading MIA splits...", flush=True)
    split_texts = {}
    for split in ("mia_forget", "mia_retain"):
        try:
            rows = _load_rwku_split(split)
            if len(rows) > n_samples:
                rows = random.Random(42).sample(rows, n_samples)
            split_texts[split] = [str(row.get("text", "")) for row in rows
                                   if row.get("text")]
        except Exception as e:
            print(f"  [WARN] MIA {split}: {e}", flush=True)
            return None

    forget_texts = split_texts.get("mia_forget", [])
    retain_texts = split_texts.get("mia_retain", [])
    if not forget_texts or not retain_texts:
        return None

    print(f"  MIA: forget={len(forget_texts)} retain={len(retain_texts)}", flush=True)
    forget_scores = compute_mink_pp(model, tokenizer, forget_texts, device)
    retain_scores = compute_mink_pp(model, tokenizer, retain_texts, device)

    # Labels: 1=forget (should look like non-member after unlearning), 0=retain
    labels = [1] * len(forget_scores) + [0] * len(retain_scores)
    all_scores = forget_scores + retain_scores
    auc = roc_auc_score(labels, all_scores)

    result = {
        "forget_mink_pp_mean": round(float(np.mean(forget_scores)), 4),
        "retain_mink_pp_mean": round(float(np.mean(retain_scores)), 4),
        # AUC < 0.5 means forget texts look less like members than retain — good
        "auc_roc": round(float(auc), 4),
    }
    print(f"  MIA: {result}", flush=True)
    return result


# ─── Utility (MMLU) ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_mmlu(model, tokenizer, device, n_samples=500):
    print(f"  Loading MMLU ({n_samples} samples)...", flush=True)
    try:
        ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  [WARN] MMLU: {e}", flush=True)
        return None

    rows = list(ds)
    if len(rows) > n_samples:
        rows = random.Random(42).sample(rows, n_samples)

    # Token IDs for answer letters (with leading space)
    choice_ids = []
    for c in ("A", "B", "C", "D"):
        ids = tokenizer.encode(f" {c}", add_special_tokens=False)
        choice_ids.append(ids[-1])  # last token of " A" etc.

    correct = 0
    for row in rows:
        q       = row["question"]
        choices = row["choices"]
        answer  = int(row["answer"])
        prompt  = (
            f"Question: {q}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            f"Answer:"
        )
        enc = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512,
        ).to(device)
        logits = model(**enc).logits[0, -1].float()
        probs  = torch.softmax(logits, dim=-1)
        pred   = int(np.argmax([probs[t].item() for t in choice_ids]))
        if pred == answer:
            correct += 1

    acc = correct / max(len(rows), 1)
    print(f"  MMLU: {correct}/{len(rows)} = {acc:.4f}", flush=True)
    return round(acc, 4)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True,
                        help="Checkpoint path or HuggingFace model ID")
    parser.add_argument("--output",   required=True,
                        help="Output JSON path")
    parser.add_argument("--n_forget", type=int, default=500,
                        help="Max samples per forget/neighbor split (default 500)")
    parser.add_argument("--n_mia",    type=int, default=200)
    parser.add_argument("--n_mmlu",   type=int, default=500)
    parser.add_argument("--skip_mmlu", action="store_true")
    parser.add_argument("--skip_mia",  action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Model:  {args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device).eval()

    results = {}

    # ── Forget set ──────────────────────────────────────────────────────────
    print("\n=== Forget set ===", flush=True)
    for level in (1, 2, 3):
        key = f"forget_level{level}"
        # L1 is fill-in-the-blank (short answer), L2/L3 need more tokens
        max_new = 20 if level == 1 else 64
        results[key] = eval_forget_neighbor(
            model, tokenizer, key, device,
            n_samples=args.n_forget, max_new_tokens=max_new,
        )

    # ── Neighbor set ────────────────────────────────────────────────────────
    print("\n=== Neighbor set ===", flush=True)
    for level in (1, 2):
        key = f"neighbor_level{level}"
        max_new = 20 if level == 1 else 64
        results[key] = eval_forget_neighbor(
            model, tokenizer, key, device,
            n_samples=args.n_forget, max_new_tokens=max_new,
        )

    # ── MIA ─────────────────────────────────────────────────────────────────
    if not args.skip_mia:
        print("\n=== MIA ===", flush=True)
        results["mia"] = eval_mia(model, tokenizer, device, args.n_mia)

    # ── Utility (MMLU) ──────────────────────────────────────────────────────
    if not args.skip_mmlu:
        print("\n=== Utility (MMLU) ===", flush=True)
        results["utility_mmlu"] = eval_mmlu(model, tokenizer, device, args.n_mmlu)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
