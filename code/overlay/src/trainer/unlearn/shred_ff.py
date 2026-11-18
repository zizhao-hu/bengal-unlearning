import os
"""
ShredFF: Shredding via Full Fine-tune

Self-distillation unlearning via bottom-P% logit demotion.
Precomputes top-k target distributions from the initial model.
For forget data: bottom P% tokens get demoted (logit set to min).
For retain data: no demotion (P=0%), just anchor to original distribution.
Both use the same top-k → top-k KL divergence loss.
"""

import logging
import time
import torch
from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to inject _idx into every sub-dict for collator passthrough."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        for key in item:
            if isinstance(item[key], dict):
                item[key]["_idx"] = idx
        if not any(isinstance(v, dict) for v in item.values()):
            item["_idx"] = idx
        return item


class ShredFF(UnlearnTrainer):
    """
    Self-distillation unlearning via logit demotion.
    Tracks KL at demoted vs unmodified positions separately.
    """

    def __init__(self, top_k=64, demote_pct=10, alpha=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.demote_pct = demote_pct
        self.alpha = alpha
        self._precomputed = False
        self._targets = {}          # idx -> (target_log_probs, target_ids)
        self._retain_targets = {}   # idx -> (target_log_probs, target_ids)
        self._demote_mask = {}      # idx -> bool tensor [seq_len]
        self._unlearn_tokens = {}   # idx -> tensor [seq_len]
        self._step_count = 0
        self._kl_demoted_sum = 0.0
        self._kl_unmod_sum = 0.0
        self._n_demoted = 0
        self._n_unmod = 0

    @torch.no_grad()
    def _precompute_one(self, model, input_ids, labels, demote_pct, device):
        """
        Precompute top-k target distribution for one example.
        
        Args:
            model: the initial (memorized) model
            input_ids: [1, seq_len] tensor
            labels: [1, seq_len] tensor (-100 for non-answer positions)
            demote_pct: percentage of answer tokens to demote (0 for retain)
            device: compute device
            
        Returns:
            target_log_probs: [seq_len, top_k] log-softmax over top-k
            target_ids: [seq_len, top_k] vocab indices
            demote_mask: [seq_len] bool, True at demoted positions
            unlearn_token_ids: [seq_len] the answer token id or -100
        """
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]
        seq_len = logits.shape[0]
        probs = torch.softmax(logits, dim=-1)

        # Collect answer token probs at valid label positions
        unlearn_probs = []
        valid_positions = []
        unlearn_token_ids = torch.full((seq_len,), -100, dtype=torch.long)
        SKIP_FIRST_N = 4
        for t in range(SKIP_FIRST_N, seq_len - 1):
            ut = labels[0, t + 1].item()
            if ut >= 0:
                unlearn_probs.append(probs[t, ut].item())
                valid_positions.append(t)
                unlearn_token_ids[t] = ut

        # Determine demote set (bottom P% by probability)
        demote_mask = torch.zeros(seq_len, dtype=torch.bool)

        # Step 1: Get top-k from ORIGINAL logits (before any demotion)
        topk_logits, topk_ids = torch.topk(logits, self.top_k, dim=-1)

        # Step 2: Demote within the top-k set
        if valid_positions and demote_pct > 0:
            n_to_demote = max(1, int(len(valid_positions) * demote_pct / 100))
            sorted_idx = sorted(range(len(unlearn_probs)), key=lambda i: unlearn_probs[i])
            demote_positions = set(valid_positions[i] for i in sorted_idx[:n_to_demote])
            for t in demote_positions:
                ut = labels[0, t + 1].item()
                # Find unlearn token in top-k at position t
                topk_at_t = topk_ids[t]
                ut_pos = (topk_at_t == ut).nonzero(as_tuple=True)[0]
                if len(ut_pos) > 0:
                    # Demote: set its logit to -inf (zero probability)
                    topk_logits[t, ut_pos[0]] = float('-inf')
                demote_mask[t] = True

        target_log_probs = torch.log_softmax(topk_logits, dim=-1)

        return target_log_probs.cpu(), topk_ids.cpu(), demote_mask.cpu(), unlearn_token_ids.cpu()

    @torch.no_grad()
    def _precompute_targets(self):
        model = self.accelerator.unwrap_model(self.model)
        gc_enabled = getattr(model, "is_gradient_checkpointing", False)
        if gc_enabled:
            model.gradient_checkpointing_disable()
        model.eval()
        device = next(model.parameters()).device

        n_examples = len(self.train_dataset)
        t0 = time.time()

        logger.warning(
            "=" * 60 + "\n"
            f"[ShredFF] PRECOMPUTE TARGETS\n"
            f"  Examples:    {n_examples}\n"
            f"  top_k:       {self.top_k}\n"
            f"  demote_pct:  {self.demote_pct}%\n"
            f"  alpha:       {self.alpha}\n"
            f"  Device:      {device}\n"
            + "=" * 60
        )

        has_retain = "retain" in self.train_dataset[0]
        n_demoted_total = 0
        n_positions_total = 0

        for idx in range(n_examples):
            item = self.train_dataset[idx]
            
            # --- Forget targets (with demotion) ---
            forget_item = item["forget"] if "forget" in item else item
            input_ids = forget_item["input_ids"].unsqueeze(0).to(device)
            labels = forget_item["labels"].unsqueeze(0).to(device)

            target_lp, target_ids, dmask, utokens = self._precompute_one(
                model, input_ids, labels, self.demote_pct, device
            )
            self._targets[idx] = (target_lp, target_ids)
            self._demote_mask[idx] = dmask
            self._unlearn_tokens[idx] = utokens
            n_demoted_total += dmask.sum().item()
            n_positions_total += (utokens >= 0).sum().item()

            if (idx + 1) % 100 == 0:
                logger.warning(f"  Forget precompute progress: [{idx + 1}/{n_examples}]")

        # --- Precompute retain targets for ALL retain examples (P=0%) ---
        n_retain = 0
        if has_retain and self.alpha > 0:
            retain_ds = self.train_dataset.retain  # the full retain dataset
            n_retain = len(retain_ds)
            logger.warning(f"[ShredFF] Precomputing {n_retain} retain targets (P=0%)...")
            for ridx in range(n_retain):
                r_item = retain_ds[ridx]
                r_input_ids = r_item["input_ids"].unsqueeze(0).to(device)
                r_labels = r_item["labels"].unsqueeze(0).to(device)
                r_target_lp, r_target_ids, _, _ = self._precompute_one(
                    model, r_input_ids, r_labels, 0, device  # demote_pct=0
                )
                self._retain_targets[ridx] = (r_target_lp, r_target_ids)
                if (ridx + 1) % 500 == 0:
                    logger.warning(f"  Retain precompute progress: [{ridx + 1}/{n_retain}]")

        elapsed = time.time() - t0
        logger.warning(
            "\n" + "=" * 60 + "\n"
            f"[ShredFF] PRECOMPUTE COMPLETE ({elapsed:.1f}s)\n"
            f"  Forget targets:       {len(self._targets)}\n"
            f"  Retain targets:       {n_retain}\n"
            f"  Demoted positions:    {n_demoted_total} / {n_positions_total} "
            f"({100 * n_demoted_total / max(1, n_positions_total):.1f}%)\n"
            + "=" * 60
        )

        self.train_dataset = IndexedDataset(self.train_dataset)
        self.data_collator.index = "_idx"

        model.train()
        if gc_enabled or self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        self._precomputed = True

    def train(self, *args, **kwargs):
        if not self._precomputed:
            self._precompute_targets()
        return super().train(*args, **kwargs)

    def _compute_topk_kl(self, logits, batch_idx, target_lp, target_ids, attention_mask):
        """
        Compute top-k to top-k KL divergence for one sample in a batch.
        Both target and model distributions are normalized over the same k tokens.
        """
        device = logits.device
        t_len = min(target_lp.shape[0], logits.shape[0])

        # Gather raw logits at the same top-k positions, then normalize over k
        model_topk_logits = torch.gather(logits[:t_len], 1, target_ids[:t_len])
        model_at_targets = torch.log_softmax(model_topk_logits, dim=-1)

        target_p = torch.exp(target_lp[:t_len])
        kl_per_token = target_p * (target_lp[:t_len] - model_at_targets)
        kl = torch.nan_to_num(kl_per_token, nan=0.0).sum(dim=-1)

        mask = attention_mask[1:t_len].float()
        kl_valid = kl[:-1][:mask.shape[0]]
        sample_loss = (kl_valid * mask[:kl_valid.shape[0]]).sum() / (
            mask[:kl_valid.shape[0]].sum() + 1e-10
        )
        return sample_loss, kl_valid, mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        input_ids = forget_inputs["input_ids"]
        attention_mask = forget_inputs["attention_mask"]
        indices = forget_inputs["_idx"]
        device = input_ids.device

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        batch_size = input_ids.shape[0]
        losses = []
        batch_kl_demoted = []
        batch_kl_unmod = []
        batch_prob_demoted = []

        for b in range(batch_size):
            idx = indices[b].item()
            target_log_probs, target_ids = self._targets[idx]
            target_log_probs = target_log_probs.to(device)
            target_ids = target_ids.to(device)
            dmask = self._demote_mask[idx].to(device)
            utokens = self._unlearn_tokens[idx].to(device)

            sample_loss, kl_valid, mask = self._compute_topk_kl(
                logits[b], idx, target_log_probs, target_ids, attention_mask[b]
            )
            losses.append(sample_loss)

            # --- Track KL at demoted vs unmodified positions ---
            t_len = min(target_log_probs.shape[0], logits.shape[1])
            with torch.no_grad():
                dm = dmask[:t_len - 1][:mask.shape[0]]
                vm = (mask[:dm.shape[0]] > 0)
                demoted_here = dm & vm
                unmod_here = (~dm) & vm & (utokens[:t_len - 1][:mask.shape[0]] >= 0)

                if demoted_here.any():
                    batch_kl_demoted.append(kl_valid[demoted_here].mean().item())
                if unmod_here.any():
                    batch_kl_unmod.append(kl_valid[unmod_here].mean().item())

                model_probs = torch.softmax(logits[b, :t_len], dim=-1)
                for t_pos in demoted_here.nonzero(as_tuple=True)[0]:
                    ut = utokens[t_pos].item()
                    if ut >= 0:
                        batch_prob_demoted.append(model_probs[t_pos, ut].item())

        loss = torch.stack(losses).mean()

        # --- Retain loss: same top-k KL, but with P=0% targets (no demotion) ---
        retain_loss = None
        if self.alpha > 0 and "retain" in inputs and self._retain_targets:
            retain_inputs = inputs["retain"]
            r_input_ids = retain_inputs["input_ids"]
            r_attention_mask = retain_inputs["attention_mask"]
            r_retain_indices = retain_inputs["_retain_idx"]  # actual retain dataset index
            r_outputs = model(input_ids=r_input_ids, attention_mask=r_attention_mask)
            r_logits = r_outputs.logits

            retain_losses = []
            for rb in range(r_input_ids.shape[0]):
                ridx = r_retain_indices[rb].item()
                if ridx not in self._retain_targets:
                    continue
                r_target_lp, r_target_ids = self._retain_targets[ridx]
                r_target_lp = r_target_lp.to(device)
                r_target_ids = r_target_ids.to(device)

                r_sample_loss, _, _ = self._compute_topk_kl(
                    r_logits[rb], ridx, r_target_lp, r_target_ids, r_attention_mask[rb]
                )
                retain_losses.append(r_sample_loss)

            if retain_losses:
                retain_loss = torch.stack(retain_losses).mean()
                loss = loss + self.alpha * retain_loss

        # --- Periodic logging ---
        self._step_count += 1
        if batch_kl_demoted:
            self._kl_demoted_sum += sum(batch_kl_demoted) / len(batch_kl_demoted)
            self._n_demoted += 1
        if batch_kl_unmod:
            self._kl_unmod_sum += sum(batch_kl_unmod) / len(batch_kl_unmod)
            self._n_unmod += 1

        if self._step_count % 10 == 0:
            avg_kl_d = self._kl_demoted_sum / max(1, self._n_demoted)
            avg_kl_u = self._kl_unmod_sum / max(1, self._n_unmod)
            avg_prob_d = sum(batch_prob_demoted) / max(1, len(batch_prob_demoted))
            retain_str = f" retain_kl={retain_loss.item():.4f}" if retain_loss is not None else ""
            logger.warning(
                f"[Step {self._step_count}] "
                f"loss={loss.item():.4f}{retain_str} | "
                f"KL_demoted={avg_kl_d:.4f} KL_unmod={avg_kl_u:.4f} "
                f"ratio={avg_kl_d / max(1e-10, avg_kl_u):.1f}x | "
                f"prob_at_demoted={avg_prob_d:.4f}"
            )
            self._kl_demoted_sum = 0.0
            self._kl_unmod_sum = 0.0
            self._n_demoted = 0
            self._n_unmod = 0

        return (loss, outputs) if return_outputs else loss
