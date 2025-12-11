import os
import math
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.utils.config import load_yaml_config
from src.data.collate import collate_mt_batch
from src.data.dataset import MTTextDataset
from src.data.utils import compute_length_stats
from src.data.batch_sampler import TokenBatchSampler
from src.models.transformer import Seq2SeqTransformer, generate_square_subsequent_mask
from src.utils.config import sanitize_optimizer_config


# ----------------- Utilities -----------------
class NoamScheduler:
    """Implements the Noam learning rate schedule from the Transformer paper."""

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000, factor: float = 1.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self._step = 0
        self.d_model = d_model
        self.factor = factor

    def step(self):
        self._step += 1
        lr = self.factor * (self.d_model ** -0.5) * min(self._step ** -0.5, self._step * (self.warmup_steps ** -1.5))
        for p in self.optimizer.param_groups:
            p["lr"] = lr
        return lr

    def state_dict(self):
        return {"_step": self._step}

    def load_state_dict(self, state: Dict[str, Any]):
        self._step = state.get("_step", 0)


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, label_smoothing: float, vocab_size: int, ignore_index: int = -100):
        super().__init__()
        assert 0.0 <= label_smoothing < 1.0
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred: [B, T, V] (logits), target: [B, T]
        confidence = 1.0 - self.label_smoothing
        smoothing_value = self.label_smoothing / (self.vocab_size - 1)

        pred = pred.view(-1, pred.size(-1))  # [B*T, V]
        target = target.view(-1)  # [B*T]

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.full_like(pred, smoothing_value)
            mask = target != self.ignore_index
            non_ignored = mask.nonzero(as_tuple=False).squeeze(1)
            if non_ignored.numel() > 0:
                true_dist[non_ignored, target[non_ignored]] = confidence

        # Use KLDiv loss: log_softmax(pred) vs true_dist
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        loss = -(true_dist * log_prob).sum(dim=-1)

        # Mask out ignored positions
        if non_ignored.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        loss = loss[non_ignored].mean()
        return loss


# ----------------- Checkpoint helpers -----------------

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: NoamScheduler, scaler: Optional[GradScaler], step: int, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {},
        "step": step,
        "epoch": epoch,
    }
    if scaler is not None:
        obj["scaler_state"] = scaler.state_dict()

    torch.save(obj, str(path))


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[NoamScheduler] = None, scaler: Optional[GradScaler] = None):
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("step", 0), ckpt.get("epoch", 0)


# ----------------- Trainer -----------------
class Trainer:
    def __init__(self, cfg_path: str, device: Optional[torch.device] = None):
        cfg = load_yaml_config(cfg_path)
        self.cfg = cfg

        # Data config
        data_cfg = cfg["data"]
        dl_cfg = cfg["dataloader"]
        tok_cfg = cfg["tokenizer"]

        # Training config
        t_cfg = cfg["train"]

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Build dataset + dataloader
        train_en = data_cfg["train_en"]
        train_de = data_cfg["train_de"]

        self.dataset = MTTextDataset(
            src_path=train_en,
            tgt_path=train_de,
            spm_model_path=tok_cfg["spm_model"],
            max_source_len=dl_cfg["max_source_len"],
            max_target_len=dl_cfg["max_target_len"],
        )

        # -------------------------------
        # DEBUG SUBSET (from YAML)
        # -------------------------------
        debug_subset = t_cfg.get("debug_subset", None)
        if debug_subset is not None and debug_subset > 0:
            orig_len = len(self.dataset)

            debug_subset = min(debug_subset, orig_len)
            # Slice dataset lists directly
            self.dataset.src_lines = self.dataset.src_lines[:debug_subset]
            self.dataset.tgt_lines = self.dataset.tgt_lines[:debug_subset]

            print(f"[DEBUG] Using only {debug_subset} samples out of {orig_len} for training.")
        # -------------------------------

        # Length stats must be computed *after* subset
        # Compute lengths from the sliced dataset
        src_lens = [len(self.dataset.sp.encode(x)) for x in self.dataset.src_lines]
        tgt_lens = [len(self.dataset.sp.encode(x)) for x in self.dataset.tgt_lines]


        self.batch_sampler = TokenBatchSampler(
            src_lens, tgt_lens,
            max_tokens_per_batch=dl_cfg["max_tokens_per_batch"],
            shuffle=True
        )

        self.train_loader = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=collate_mt_batch,
            num_workers=dl_cfg.get("num_workers", 2),
            pin_memory=True,
        )


        # Model config
        m_cfg = cfg["model"]
        vocab_size = tok_cfg["spm_vocab_size"]

        self.model = Seq2SeqTransformer(
            num_encoder_layers=m_cfg["num_encoder_layers"],
            num_decoder_layers=m_cfg["num_decoder_layers"],
            d_model=m_cfg["d_model"],
            nhead=m_cfg["nhead"],
            d_ff=m_cfg["d_ff"],
            vocab_size=vocab_size,
            max_len=dl_cfg["max_source_len"],
            dropout=m_cfg.get("dropout", 0.1),
            tie_embeddings=m_cfg.get("tie_embeddings", True),
        ).to(self.device)

        # Optimizer + scheduler
        optim_cfg = sanitize_optimizer_config(t_cfg["optimizer"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=optim_cfg["lr"],
            betas=optim_cfg["betas"],
            eps=optim_cfg["eps"],
        )

        self.scheduler = NoamScheduler(self.optimizer, d_model=m_cfg["d_model"], warmup_steps=t_cfg.get("warmup_steps", 4000), factor=1.0)

        # Loss
        self.criterion = LabelSmoothingLoss(label_smoothing=t_cfg.get("label_smoothing", 0.1), vocab_size=vocab_size, ignore_index=-100)

        # AMP
        self.scaler = GradScaler(enabled=t_cfg.get("use_amp", True))

        # Training params
        self.accum_steps = t_cfg.get("grad_accum_steps", 1)
        self.max_grad_norm = t_cfg.get("max_grad_norm", 1.0)
        self.num_epochs = t_cfg.get("num_epochs", 10)
        self.checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "models/checkpoints"))
        self.checkpoint_every = t_cfg.get("checkpoint_every_steps", 1000)

        # bookkeeping
        self.global_step = 0
        self.start_epoch = 0

    def train_one_epoch(self, epoch: int):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=True)

        total_steps = len(self.train_loader)
        start_time = time.time()

        last_logged_loss = None
        last_lr = None

        for step, batch in enumerate(pbar):
            src_ids = batch["src_ids"].to(self.device)
            tgt_input_ids = batch["tgt_input_ids"].to(self.device)
            tgt_labels = batch["tgt_labels"].to(self.device)
            src_pad_mask = batch["src_padding_mask"].to(self.device)
            tgt_pad_mask = batch["tgt_padding_mask"].to(self.device)

            tgt_mask = generate_square_subsequent_mask(tgt_input_ids.size(1), device=self.device)

            with autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(
                    src_ids,
                    tgt_input_ids,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    tgt_mask=tgt_mask,
                )
                loss = self.criterion(logits, tgt_labels) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                lr = self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                last_logged_loss = loss.item() * self.accum_steps
                last_lr = lr

                # update tqdm
                pbar.set_postfix({
                    "loss": f"{last_logged_loss:.4f}",
                    "lr": f"{lr:.3e}"
                })

        # compute epoch-level speed
        elapsed = time.time() - start_time
        steps_per_sec = total_steps / elapsed

        print(f"Epoch {epoch} completed | Loss: {last_logged_loss:.4f} | "
            f"LR: {last_lr:.3e} | Speed: {steps_per_sec:.2f} steps/s")



    def fit(self, resume_from: Optional[str] = None):
        if resume_from is not None:
            step, epoch = load_checkpoint(Path(resume_from), self.model, self.optimizer, self.scheduler, self.scaler)
            self.global_step = step
            self.start_epoch = epoch + 1

        for epoch in range(self.start_epoch, self.num_epochs):
            self.train_one_epoch(epoch)

            # save epoch checkpoint
            ckpt_path = self.checkpoint_dir / f"ckpt_epoch_{epoch}.pt"
            save_checkpoint(ckpt_path, self.model, self.optimizer, self.scheduler, self.scaler, self.global_step, epoch)


# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    trainer = Trainer(args.config)
    trainer.fit(resume_from=args.resume)
