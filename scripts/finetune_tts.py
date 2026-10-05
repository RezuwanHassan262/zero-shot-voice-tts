#!/usr/bin/env python
"""
Script 2 - finetune_tts.py

Fine-tune the pretrained F5-TTS base model (DiT backbone, vocos mel front-end)
on the speaker dataset produced by build_clean_dataset.py, with a custom
training loop tuned for a 12 GB RTX 3060:

  * sample-level micro-batches of <= 4 clips / <= 150 s, one optimizer update each (46/epoch), fp16 autocast
  * activation checkpointing so 40-second clips fit in memory
  * per-epoch validation loss averaged over 4 fixed random draws (comparable between epochs)
  * warm-up + linear LR decay planned over --epochs (60), EarlyStopping(patience=20) on validation loss
  * dynamic extension: if the loss is still declining when the planned epochs are
    reached, training is extended in +10 epoch increments (up to --max_epochs)
  * checkpoints/best_model.pt (lowest val loss) and checkpoints/last_model.pt
  * logs/training_metrics.csv + .xlsx (epoch, training_loss, validation_loss, learning_rate, ...),
    logs/training_convergence.png (high-res curves) and logs/finetune_tts.log (console + warnings + traces)

Usage:
    python scripts/finetune_tts.py                       # defaults: 60 epochs, bs 4, accum 1, patience 20
    python scripts/finetune_tts.py --resume              # continue from last_model.pt
    python scripts/finetune_tts.py --dry_run             # one mini-batch, verifies VRAM fit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import CHECKPOINT_DIR, LOG_DIR, TARGET_SR  # noqa: E402

log = common.get_logger("finetune_tts")  # -> logs/finetune_tts.log (console output, warnings, error traces)

BEST_CKPT = CHECKPOINT_DIR / "best_model.pt"
LAST_CKPT = CHECKPOINT_DIR / "last_model.pt"
# centralised training logs (everything metric-related goes to logs/)
HISTORY_CSV = LOG_DIR / "training_metrics.csv"
HISTORY_XLSX = LOG_DIR / "training_metrics.xlsx"
CURVE_PNG = LOG_DIR / "training_convergence.png"
CONFIG_JSON = LOG_DIR / "train_config.json"

MEL_KWARGS = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
                  target_sample_rate=TARGET_SR, mel_spec_type="vocos")
HOP = MEL_KWARGS["hop_length"]

MODEL_CONFIGS = {
    # matches f5_tts/configs/F5TTS_Base.yaml
    "F5TTS_Base": dict(
        arch=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=False,
                  conv_layers=4, pe_attn_head=1),
        ckpt="hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt",
    ),
    "F5TTS_v1_Base": dict(
        arch=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt="hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    ),
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class SpeakerDataset(Dataset):
    """Loads 24 kHz mono wavs, returns (mel [d, t], token list)."""

    def __init__(self, rows: list[dict], mel_spec, max_duration: float, name: str):
        from f5_tts.model.utils import convert_char_to_pinyin

        kept, dropped = [], []
        for r in rows:
            (kept if r["duration"] <= max_duration else dropped).append(r)
        if dropped:
            log.warning(f"{name}: {len(dropped)} clip(s) longer than {max_duration}s excluded from this split: "
                        + ", ".join(f"{Path(r['audio_path']).stem}({r['duration']:.0f}s)" for r in dropped))
        assert kept, f"{name}: no clips left after duration filtering"
        self.rows = kept
        self.mel_spec = mel_spec
        # character-level tokens; convert_char_to_pinyin leaves Latin text as-is
        self.tokens = convert_char_to_pinyin([r["text"] for r in kept])
        self.durations = [r["duration"] for r in kept]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio, sr = common.load_audio(common.abs_path(r["audio_path"]), sr=TARGET_SR)
        common.assert_audio_ok(audio, r["audio_path"])
        with torch.no_grad():
            mel = self.mel_spec(torch.from_numpy(audio)[None]).squeeze(0)  # [d, t]
        assert torch.isfinite(mel).all(), f"{r['audio_path']}: non-finite mel"
        return {"mel": mel, "text": self.tokens[i]}


def _worker_init(_):
    torch.set_num_threads(1)  # keep DataLoader workers light on RAM/CPU


def collate(batch):
    mels = [b["mel"] for b in batch]
    lens = torch.LongTensor([m.shape[-1] for m in mels])
    T = int(lens.max())
    mel = torch.stack([F.pad(m, (0, T - m.shape[-1])) for m in mels])
    return {"mel": mel, "mel_lengths": lens, "text": [b["text"] for b in batch]}


class BucketBatchSampler(Sampler):
    """
    Duration-sorted batches of up to `batch_size` clips whose summed duration
    stays under `max_batch_seconds` (VRAM budget): short clips train 4 at a
    time, the 60-80 s clips alone or in pairs.  Reshuffled every epoch.
    """

    def __init__(self, durations, batch_size, max_batch_seconds, seed, shuffle=True):
        self.durations = list(durations)
        self.batch_size = batch_size
        self.max_batch_seconds = max_batch_seconds
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self._batches = self._make_batches(random.Random(seed))

    def _make_batches(self, rng):
        idx = list(np.argsort(self.durations))
        if self.shuffle:  # jitter inside the sorted order so buckets vary between epochs
            for k in range(0, len(idx), self.batch_size * 4):
                chunk = idx[k : k + self.batch_size * 4]
                rng.shuffle(chunk)
                idx[k : k + len(chunk)] = chunk
        batches, cur, cur_sec = [], [], 0.0
        for i in idx:
            d = self.durations[i]
            if cur and (len(cur) >= self.batch_size or cur_sec + d > self.max_batch_seconds):
                batches.append(cur)
                cur, cur_sec = [], 0.0
            cur.append(int(i))
            cur_sec += d
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, e):
        self.epoch = e
        self._batches = self._make_batches(random.Random(self.seed + e))

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(exp_name: str, device: str, checkpoint_activations: bool):
    from importlib.resources import files

    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
    arch = dict(MODEL_CONFIGS[exp_name]["arch"], checkpoint_activations=checkpoint_activations)
    model = CFM(
        transformer=DiT(**arch, text_num_embeds=vocab_size, mel_dim=MEL_KWARGS["n_mel_channels"]),
        mel_spec_kwargs=MEL_KWARGS,
        vocab_char_map=vocab_char_map,
    ).to(device)
    return model, vocab_char_map, vocab_file


def load_pretrained(model, exp_name: str, pretrain: str | None, device: str):
    from cached_path import cached_path
    from f5_tts.infer.utils_infer import load_checkpoint

    ckpt = pretrain or str(cached_path(MODEL_CONFIGS[exp_name]["ckpt"]))
    log.info(f"Loading pretrained weights: {ckpt}")
    # fine-tuning starts from the pretrained EMA weights (same as f5_tts finetune_cli)
    load_checkpoint(model, ckpt, device, dtype=torch.float32, use_ema=True)
    return ckpt


def check_vocab(rows, vocab_char_map):
    oov = {}
    for r in rows:
        for c in r["text"]:
            if c not in vocab_char_map:
                oov[c] = oov.get(c, 0) + 1
    if oov:
        log.warning(f"{len(oov)} character(s) not in the pretrained vocab (mapped to <unk>): {oov}")
    else:
        log.info("All transcript characters are covered by the pretrained vocabulary")


# --------------------------------------------------------------------------- #
# Validation / checkpoints / plotting
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, loader, device, seed: int, repeats: int, autocast) -> float:
    """Flow-matching loss on the validation split with fixed random draws (comparable between epochs)."""
    model.eval()
    losses = []
    for rep in range(repeats):
        torch.manual_seed(seed + rep)
        random.seed(seed + rep)
        for batch in loader:
            mel = batch["mel"].permute(0, 2, 1).to(device)
            with autocast():
                loss, _, _ = model(mel, text=batch["text"], lens=batch["mel_lengths"].to(device))
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def _half(state: dict) -> dict:
    return {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in state.items()}


def save_checkpoint(path: Path, model, ema, optimizer, scheduler, scaler_state, meta: dict,
                    with_optimizer: bool = False) -> bool:
    """
    Atomically write a checkpoint.  Online weights are kept in fp32 (exact
    resume), EMA weights - the ones inference uses - in fp16 (halves the file;
    F5-TTS casts them to fp16 at load time anyway).  Optimizer moments (2.7 GB)
    are only included on request.  Returns False (and keeps the previous file)
    if the disk does not have room.
    """
    import shutil

    payload = {
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": _half(ema.state_dict()),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler_state,
        **meta,
    }
    if with_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    need_gb = sum(v.numel() * v.element_size() for v in payload["model_state_dict"].values()) / 1024**3 * 1.5
    if with_optimizer:
        need_gb *= 2.2
    free_gb = shutil.disk_usage(path.parent).free / 1024**3
    if free_gb < need_gb + 0.3:
        log.error(f"NOT saving {path.name}: {free_gb:.1f} GiB free on disk but ~{need_gb:.1f} GiB needed. "
                  "Free up space (previous checkpoint kept).")
        return False
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
        return True
    except Exception as e:  # noqa: BLE001
        log.error(f"Failed to write {path.name}: {e}")
        tmp.unlink(missing_ok=True)
        return False


HISTORY_COLUMNS = ["epoch", "training_loss", "validation_loss", "learning_rate",  # required columns first
                   "validation_loss_ema", "optimizer_updates", "epochs_since_best", "epoch_time_s", "peak_vram_gb"]


def write_history(history: list[dict]):
    """Consolidated per-epoch metrics -> logs/training_metrics.csv and .xlsx (same columns)."""
    if not history:
        return
    import pandas as pd

    df = pd.DataFrame(history)[HISTORY_COLUMNS]
    df.to_csv(HISTORY_CSV, index=False)
    try:
        with pd.ExcelWriter(HISTORY_XLSX, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="metrics", index=False)
            summary = pd.DataFrame({
                "metric": ["best validation loss", "best epoch", "final training loss", "final validation loss",
                           "epochs run", "peak VRAM (GiB)"],
                "value": [df["validation_loss"].min(), int(df.loc[df["validation_loss"].idxmin(), "epoch"]),
                          df["training_loss"].iloc[-1], df["validation_loss"].iloc[-1], int(df["epoch"].iloc[-1]),
                          df["peak_vram_gb"].max()],
            })
            summary.to_excel(xw, sheet_name="summary", index=False)
    except Exception as e:  # noqa: BLE001  (xlsx is a convenience copy; the csv is authoritative)
        log.warning(f"could not write {HISTORY_XLSX.name}: {e}")


def plot_history(history: list[dict], best_epoch: int, planned_epochs: int, patience: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot(ep, [h["training_loss"] for h in history], label="train loss", color="#2a78d6", lw=1.8)
    ax.plot(ep, [h["validation_loss"] for h in history], label="validation loss", color="#eb6834", lw=1.8)
    ax.plot(ep, [h["validation_loss_ema"] for h in history], label="validation loss (EMA weights)",
            color="#1baf7a", lw=1.2, ls="--")
    if best_epoch:
        ax.axvline(best_epoch, color="#52514e", ls=":", lw=1.2, label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("conditional flow-matching MSE loss")
    ax.set_title("Training / validation loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(ep, [h["learning_rate"] for h in history], color="#4a3aa7", lw=1.8, label="learning rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title("Learning-rate schedule (warm-up + linear decay)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[2]
    v = np.array([h["validation_loss"] for h in history])
    best = np.minimum.accumulate(v)
    ax.plot(ep, v, color="#eb6834", lw=1.2, alpha=0.6, label="validation loss")
    ax.plot(ep, best, color="black", lw=1.8, label="best-so-far")
    ax.plot(ep, [h["epochs_since_best"] for h in history], color="#e87ba4", lw=1.2, ls="--",
            label=f"epochs since best (patience={patience})")
    ax.set_xlabel("epoch")
    ax.set_title(f"Early-stopping monitor (planned epochs: {planned_epochs})")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("F5-TTS fine-tuning convergence", fontsize=14)
    fig.tight_layout()
    fig.savefig(CURVE_PNG, dpi=220)  # high-resolution convergence plot in logs/
    plt.close(fig)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune F5-TTS on the speaker dataset (RTX 3060 profile).")
    p.add_argument("--exp_name", default="F5TTS_Base", choices=list(MODEL_CONFIGS))
    p.add_argument("--pretrain", default=None, help="local pretrained checkpoint (default: download from HF)")
    p.add_argument("--batch_size", type=int, default=4, help="max clips per micro-batch (2-4 for 12 GB)")
    p.add_argument("--max_batch_seconds", type=float, default=150.0,
                   help="max summed clip duration per micro-batch (150 s measured at 7.6 GiB peak on a 3060)")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="gradient accumulation steps. 1 = one update per micro-batch (46/epoch); a micro-batch of "
                        "<= 150 s is already ~4x the official F5-TTS fine-tune batch, so accumulating only slows "
                        "convergence (it does not change peak VRAM)")
    p.add_argument("--save_optimizer", action="store_true",
                   help="also store AdamW moments in last_model.pt (+2.7 GB) for an exact resume")
    p.add_argument("--epochs", type=int, default=60,
                   help="initially planned epochs; the LR decays linearly to --final_lr_ratio over exactly this many "
                        "epochs, so keep it close to when the loss actually flattens (extensions run at the LR floor). "
                        "Planning 200 here left the LR at 89 %% of peak when early stopping fired at epoch 34")
    p.add_argument("--patience", type=int, default=20,
                   help="early-stopping patience (epochs); the CFM loss wobbles +-0.005 epoch to epoch, so this must "
                        "be long enough to see through the noise")
    p.add_argument("--min_delta", type=float, default=1e-4, help="min val-loss improvement to reset patience")
    p.add_argument("--extend_epochs", type=int, default=10, help="extension increment when still improving")
    p.add_argument("--extend_window", type=int, default=5,
                   help="extend only if the best epoch is within this many epochs of the end")
    p.add_argument("--max_epochs", type=int, default=200, help="hard cap including extensions")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05, help="fraction of planned updates used for warm-up")
    p.add_argument("--final_lr_ratio", type=float, default=0.1, help="lr floor as a fraction of --lr")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--max_duration", type=float, default=90.0, help="skip clips longer than this (seconds)")
    p.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"])
    p.add_argument("--no_checkpoint_activations", action="store_true")
    p.add_argument("--bnb_optimizer", action="store_true", help="8-bit AdamW from bitsandbytes (less VRAM)")
    p.add_argument("--num_workers", type=int, default=1, help="DataLoader workers (each costs ~1 GB RAM)")
    p.add_argument("--val_repeats", type=int, default=4,
                   help="validation passes with different fixed seeds, averaged. Each pass draws one random "
                        "(timestep, mask span, cond-drop) per batch, so 1 pass on 21 clips is a very noisy estimate")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", action="store_true", help="resume from checkpoints/last_model.pt")
    p.add_argument("--skip_cleanup", action="store_true",
                   help="skip the pre-run disk sweep + free-space gate (common.cleanup_cache) at startup")
    p.add_argument("--require_gb", type=int, default=15,
                   help="abort before training if less than this many GB are free after cleanup")
    p.add_argument("--dry_run", action="store_true",
                   help="run a single optimizer update and exit; writes no checkpoints, metrics or plots")
    return p.parse_args()


def main():
    args = parse_args()
    assert 1 <= args.batch_size <= 8 and 1 <= args.grad_accum <= 32
    common.ensure_layout()
    common.run_cleanup(log, skip=args.skip_cleanup, require_gb=args.require_gb)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = common.gpu_report(log, require_cuda=False, min_free_gb=8.0)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from accelerate import Accelerator
    from ema_pytorch import EMA

    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.grad_accum)

    # ---- data ------------------------------------------------------------- #
    train_rows = common.read_split_csv("train")
    valid_rows = common.read_split_csv("valid")
    model, vocab_char_map, vocab_file = build_model(args.exp_name, device, not args.no_checkpoint_activations)
    check_vocab(train_rows + valid_rows, vocab_char_map)
    from f5_tts.model.modules import MelSpec

    mel_spec = MelSpec(**MEL_KWARGS)  # separate CPU instance: DataLoader workers compute mels on CPU
    train_ds = SpeakerDataset(train_rows, mel_spec, args.max_duration, "train")
    valid_ds = SpeakerDataset(valid_rows, mel_spec, args.max_duration, "valid")
    if args.dry_run:  # worst case for VRAM: one micro-batch of the longest clips
        longest = sorted(range(len(train_ds)), key=lambda i: -train_ds.durations[i])[: args.batch_size]
        train_ds.rows = [train_ds.rows[i] for i in longest]
        train_ds.tokens = [train_ds.tokens[i] for i in longest]
        train_ds.durations = [train_ds.durations[i] for i in longest]
        log.info(f"dry run: single micro-batch of the longest clips: {train_ds.durations}")
    log.info(f"train: {len(train_ds)} clips ({sum(train_ds.durations) / 60:.1f} min) | "
             f"valid: {len(valid_ds)} clips ({sum(valid_ds.durations) / 60:.1f} min)")

    train_sampler = BucketBatchSampler(train_ds.durations, args.batch_size, args.max_batch_seconds, args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
                              num_workers=args.num_workers, pin_memory=device == "cuda",
                              persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init)
    # validation is small (~20 clips); load it in-process so no extra worker (~1 GB RAM each) is spawned
    valid_loader = DataLoader(valid_ds, collate_fn=collate, num_workers=0,
                              batch_sampler=BucketBatchSampler(valid_ds.durations, args.batch_size,
                                                               args.max_batch_seconds, 0, shuffle=False))
    log.info(f"{len(train_sampler)} micro-batches/epoch (<= {args.batch_size} clips, <= {args.max_batch_seconds:.0f} s each)")

    # ---- model / optimiser ------------------------------------------------ #
    pretrain_path = load_pretrained(model, args.exp_name, args.pretrain, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"{args.exp_name}: {n_params:.1f} M parameters | activation checkpointing="
             f"{not args.no_checkpoint_activations} | mixed precision={args.mixed_precision}")

    ema = EMA(model, include_online_model=False, beta=args.ema_decay, update_every=1, update_after_step=0).to(device)

    if args.bnb_optimizer:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    planned_updates = updates_per_epoch * args.epochs
    warmup_updates = max(1, int(planned_updates * args.warmup_ratio))

    def lr_lambda(step):  # warm-up -> linear decay to floor -> constant floor (keeps extensions stable)
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        frac = min(1.0, (step - warmup_updates) / max(1, planned_updates - warmup_updates))
        return 1.0 - (1.0 - args.final_lr_ratio) * frac

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    log.info(f"updates/epoch={updates_per_epoch} planned updates={planned_updates} warm-up={warmup_updates} "
             f"(effective batch = {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} clips)")

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    raw_model = accelerator.unwrap_model(model)

    # ---- state ------------------------------------------------------------ #
    history: list[dict] = []
    start_epoch, global_update = 1, 0
    best_val, best_epoch, planned_epochs = float("inf"), 0, args.epochs
    if args.resume:
        assert LAST_CKPT.exists(), f"--resume given but {LAST_CKPT} not found"
        # Load on CPU: mapping the 1.9 GB checkpoint straight onto the GPU left its tensors resident in
        # the allocator for the rest of training (+1.2-1.9 GiB peak VRAM on every resumed epoch).
        ck = torch.load(LAST_CKPT, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ck["model_state_dict"])
        ema.load_state_dict({k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                             for k, v in ck["ema_model_state_dict"].items()})
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        else:
            log.warning("checkpoint has no optimizer state (saved without --save_optimizer); AdamW moments restart")
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        if ck.get("scaler_state_dict") and accelerator.scaler is not None:
            accelerator.scaler.load_state_dict(ck["scaler_state_dict"])
        history = ck.get("history", [])
        start_epoch = ck["epoch"] + 1
        global_update = ck["update"]
        best_val, best_epoch, planned_epochs = ck["best_val"], ck["best_epoch"], ck["planned_epochs"]
        log.info(f"Resumed from epoch {ck['epoch']} (best val {best_val:.4f} @ {best_epoch}, planned {planned_epochs})")
        del ck
        torch.cuda.empty_cache()

    config = {**vars(args), "pretrain_resolved": pretrain_path, "vocab_file": vocab_file, "device": device,
              "updates_per_epoch": updates_per_epoch, "n_train": len(train_ds), "n_valid": len(valid_ds)}
    CONFIG_JSON.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    def scaler_state():
        return accelerator.scaler.state_dict() if accelerator.scaler is not None else None

    # ---- training loop ---------------------------------------------------- #
    epoch = start_epoch
    stop_reason = "reached planned epochs"
    while epoch <= planned_epochs:
        t0 = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        losses = []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                mel = batch["mel"].permute(0, 2, 1)
                loss, _, _ = model(mel, text=batch["text"], lens=batch["mel_lengths"])
                assert torch.isfinite(loss), f"non-finite loss at epoch {epoch} step {step}"
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                ema.update()
                global_update += 1
            losses.append(float(loss))
            if args.dry_run and step + 1 >= args.grad_accum:
                break

        train_loss = float(np.mean(losses))
        val_loss = validate(raw_model, valid_loader, device, args.seed, args.val_repeats, accelerator.autocast)
        val_loss_ema = validate(ema.ema_model, valid_loader, device, args.seed, args.val_repeats, accelerator.autocast)
        lr_now = scheduler.get_last_lr()[0]
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0

        improved = val_loss < best_val - args.min_delta
        if improved:
            best_val, best_epoch = val_loss, epoch
        since_best = epoch - best_epoch
        history.append({"epoch": epoch, "training_loss": round(train_loss, 6), "validation_loss": round(val_loss, 6),
                        "learning_rate": lr_now, "validation_loss_ema": round(val_loss_ema, 6),
                        "optimizer_updates": global_update, "epochs_since_best": since_best,
                        "epoch_time_s": round(time.time() - t0, 1), "peak_vram_gb": round(peak_gb, 2)})
        log.info(f"epoch {epoch:3d}/{planned_epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
                 f"(ema {val_loss_ema:.4f}) | lr {lr_now:.2e} | best {best_val:.4f}@{best_epoch} "
                 f"| wait {since_best}/{args.patience} | {time.time() - t0:.0f}s | peak {peak_gb:.1f} GiB")

        meta = {"epoch": epoch, "update": global_update, "val_loss": val_loss, "best_val": best_val,
                "best_epoch": best_epoch, "planned_epochs": planned_epochs, "history": history,
                "config": config, "exp_name": args.exp_name}
        if args.dry_run:
            # A dry run is a VRAM/plumbing check on 4 clips for 1 epoch; its "best" val loss is
            # meaningless and must never replace a real checkpoint.  Writing it once destroyed the
            # shipped epoch-91 weights, recoverable only because the EMA tensors happened to exist
            # elsewhere - so the dry-run path is now write-free by construction.
            log.info(f"  dry run: NOT writing {BEST_CKPT.name} / {LAST_CKPT.name} "
                     "(a 1-epoch dry-run loss is not a real checkpoint)")
        else:
            if improved and save_checkpoint(BEST_CKPT, raw_model, ema, optimizer, scheduler, scaler_state(), meta):
                log.info(f"  new best -> {BEST_CKPT} ({BEST_CKPT.stat().st_size / 1024**3:.2f} GB)")
            save_checkpoint(LAST_CKPT, raw_model, ema, optimizer, scheduler, scaler_state(), meta,
                            with_optimizer=args.save_optimizer)
            write_history(history)
            plot_history(history, best_epoch, planned_epochs, args.patience)

        if args.dry_run:
            stop_reason = "dry run"
            break
        if since_best >= args.patience:
            stop_reason = f"early stopping: no val improvement for {args.patience} epochs"
            break
        if epoch == planned_epochs and since_best <= args.extend_window and planned_epochs < args.max_epochs:
            planned_epochs = min(args.max_epochs, planned_epochs + args.extend_epochs)
            log.info(f"  validation loss still declining (best {args.extend_window} epochs ago or less) -> "
                     f"extending training to {planned_epochs} epochs")
        epoch += 1

    log.info(f"Training finished ({stop_reason}). best val {best_val:.4f} at epoch {best_epoch}. "
             f"best -> {BEST_CKPT} | last -> {LAST_CKPT} | metrics -> {HISTORY_CSV} / {HISTORY_XLSX} | "
             f"curves -> {CURVE_PNG} | log -> {LOG_DIR / 'finetune_tts.log'}")


if __name__ == "__main__":
    main()
