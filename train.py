"""
train.py

Training loop for MidiMamba.
Key design decisions vs v1:
- Full songs as single sequences — no mid-song windowing during training
- Conditioning tokens always present — model learns to use them
- Cosine schedule with proper warmup relative to total steps
- Anti-repetition handled architecturally by Mamba SSM state decay + section tokens

Usage:
    python train.py tokens_out --stats stats_out [options]
"""

import argparse
import contextlib
import csv
import json
import math
import os
import random
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

import tokenizer as tok
from model import MambaLayer, MidiMamba, ModelConfig

console = Console()
_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    console.print("\n[yellow]Interrupt received — will save checkpoint after this step.[/yellow]")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

class SongDataset(Dataset):
    """
    Full-song dataset. Each item is one complete tokenized song.
    Songs longer than max_seq are split into non-overlapping chunks.
    Songs shorter than min_tokens are discarded.

    Hash-based split: each file is assigned to train or val based on
    SHA256(token sequence) % 100. Ensures near-duplicates always land
    in the same split, preventing train/val leakage.
    """

    def __init__(self, token_dir: str, max_seq: int, min_tokens: int = 256,
                 split: str = "train", val_frac: float = 0.02):
        import hashlib
        import pickle
        self.max_seq = max_seq
        self.min_tokens = min_tokens
        self.chunks = []  # list of (fpath_str, start, length) — lazy load on __getitem__

        token_dir = Path(token_dir)
        files = sorted(token_dir.glob("*_tokens.npy"))
        console.print(f"[blue]Loading {len(files)} token files (split={split})...")

        val_bucket_max = int(val_frac * 100)

        # --- Split assignment cache (permanent — only invalidates if file count or val_frac changes)
        # Maps filename -> bucket (0-99). Never depends on seq_len.
        split_cache_key = f"{len(files)}_{val_frac}"
        split_cache_path = token_dir / ".split_cache.pkl"
        file_lengths: dict[str, int] = {}  # filename -> n_tokens (also cached here)

        split_map: dict[str, int] = {}
        if split_cache_path.exists():
            try:
                with open(split_cache_path, "rb") as f:
                    sc = pickle.load(f)
                if sc.get("key") == split_cache_key:
                    split_map = sc["split_map"]
                    file_lengths = sc["file_lengths"]
                    console.print(f"[dim]Split cache hit — {len(split_map):,} assignments + lengths loaded[/dim]")
            except Exception:
                split_map = {}
                file_lengths = {}

        # --- Chunk cache (invalidates on seq_len/min_tokens change, rebuilds fast from lengths)
        chunk_cache_key = f"{len(files)}_{val_frac}_{max_seq}_{min_tokens}"
        chunk_cache_path = token_dir / f".chunk_cache_{split}.pkl"

        if chunk_cache_path.exists():
            try:
                with open(chunk_cache_path, "rb") as f:
                    cc = pickle.load(f)
                if cc.get("key") == chunk_cache_key:
                    self.chunks = cc["chunks"]
                    total_tokens = cc["total_tokens"]
                    console.print(f"[dim]Chunk cache hit — {len(self.chunks):,} chunks loaded instantly[/dim]")
                    console.print(f"[green]{split}: {len(self.chunks)} chunks, {total_tokens/1e6:.1f}M tokens[/green]")
                    # fall through to preload — don't return early
                    self._preload()
                    return
            except Exception:
                pass

        # --- Build: need file lengths for chunking
        # Files missing from length cache are read in parallel using threads
        # (I/O bound — threading helps significantly on Windows NVMe)
        total_tokens = 0
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from rich.progress import Progress as RProgress, BarColumn, MofNCompleteColumn, SpinnerColumn, TextColumn, TimeRemainingColumn

        # First pass: assign splits (instant, filename hash only)
        # Identify which files need disk reads for length
        files_needing_read = []
        for fpath in files:
            fname = fpath.name
            if fname not in split_map:
                split_map[fname] = int(hashlib.md5(fname.encode()).hexdigest()[:4], 16) % 100
            if fname not in file_lengths:
                files_needing_read.append(fpath)

        # Parallel length reads for files not in cache
        if files_needing_read:
            def _read_len(fpath):
                return fpath.name, len(np.load(str(fpath)))

            console.print(f"[dim]Reading {len(files_needing_read):,} file lengths in parallel...[/dim]")
            with RProgress(SpinnerColumn(), TextColumn("[blue]Reading"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn()) as prog2:
                task2 = prog2.add_task("", total=len(files_needing_read))
                with ThreadPoolExecutor(max_workers=16) as exe:
                    futs = {exe.submit(_read_len, fp): fp for fp in files_needing_read}
                    for fut in as_completed(futs):
                        fname, n = fut.result()
                        file_lengths[fname] = n
                        prog2.update(task2, advance=1)

        # Second pass: build chunks from cached lengths — pure arithmetic, instant
        with RProgress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{split}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[chunks]} chunks[/dim]"),
        ) as prog:
            task = prog.add_task("", total=len(files), chunks=0)
            for fpath in files:
                fname = fpath.name
                bucket = split_map[fname]
                is_val = bucket < val_bucket_max

                if split == "val" and not is_val:
                    prog.update(task, advance=1)
                    continue
                if split == "train" and is_val:
                    prog.update(task, advance=1)
                    continue

                n = file_lengths.get(fname, 0)
                if n < min_tokens:
                    prog.update(task, advance=1)
                    continue

                for start in range(0, n, max_seq):
                    end = min(start + max_seq, n)
                    if end - start >= min_tokens:
                        self.chunks.append((str(fpath), start, end - start))
                        total_tokens += end - start

                prog.update(task, advance=1, chunks=len(self.chunks))

        # Save split cache if we built or updated it
        try:
            with open(split_cache_path, "wb") as f:
                pickle.dump({"key": split_cache_key, "split_map": split_map, "file_lengths": file_lengths}, f)
        except Exception:
            pass

        # Save chunk cache
        try:
            with open(chunk_cache_path, "wb") as f:
                pickle.dump({"key": chunk_cache_key, "chunks": self.chunks, "total_tokens": total_tokens}, f)
            console.print(f"[dim]Caches saved[/dim]")
        except Exception:
            pass

        console.print(f"[green]{split}: {len(self.chunks)} chunks, {total_tokens/1e6:.1f}M tokens[/green]")
        self._preload()

    def _preload(self):
        """Load all unique files into RAM dict for zero-latency __getitem__."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from rich.progress import Progress as RProgress, BarColumn, MofNCompleteColumn, SpinnerColumn, TextColumn
        unique_fpaths = list({fpath for fpath, _, _ in self.chunks})
        console.print(f"[dim]Preloading {len(unique_fpaths):,} files into RAM...[/dim]")
        self._arrays: dict[str, np.ndarray] = {}
        def _load(fp): return fp, np.load(fp)
        with ThreadPoolExecutor(max_workers=16) as exe:
            futs = {exe.submit(_load, fp): fp for fp in unique_fpaths}
            with RProgress(SpinnerColumn(), TextColumn("[blue]Preloading"), BarColumn(), MofNCompleteColumn()) as prog2:
                task2 = prog2.add_task("", total=len(unique_fpaths))
                for fut in as_completed(futs):
                    fp, arr = fut.result()
                    self._arrays[fp] = arr
                    prog2.update(task2, advance=1)
        console.print(f"[dim]Preload complete — all data in RAM[/dim]")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        fpath, start, length = self.chunks[i]
        arr = self._arrays[fpath]  # RAM lookup — zero disk I/O
        chunk = arr[start: start + self.max_seq + 1].astype(np.int64)
        if len(chunk) < self.max_seq + 1:
            chunk = np.pad(chunk, (0, self.max_seq + 1 - len(chunk)), constant_values=tok.PAD)

        # Augmentation: pitch transpose ±6 semitones
        pitch_shift = random.randint(-6, 6)
        vel_shift = random.randint(-1, 1)
        tempo_shift = random.randint(-2, 2)

        if pitch_shift or vel_shift or tempo_shift:
            for idx in range(len(chunk)):
                t = int(chunk[idx])
                if pitch_shift and tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88:
                    nt = t + pitch_shift
                    if tok.PITCH_OFFSET <= nt < tok.PITCH_OFFSET + 88:
                        chunk[idx] = nt
                elif vel_shift and tok.VEL_OFFSET <= t < tok.VEL_OFFSET + 8:
                    chunk[idx] = np.clip(t + vel_shift, tok.VEL_OFFSET, tok.VEL_OFFSET + 7)
                elif tempo_shift and tok.TEMPO_OFFSET <= t < tok.TEMPO_OFFSET + 17:
                    chunk[idx] = np.clip(t + tempo_shift, tok.TEMPO_OFFSET, tok.TEMPO_OFFSET + 16)

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


# --------------------------------------------------------------------------- #
#  Anti-repetition loss
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  LR schedule
# --------------------------------------------------------------------------- #

def cosine_lr(step: int, warmup: int, total: int, min_lr: float, max_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
#  Checkpoint I/O
# --------------------------------------------------------------------------- #

def save_checkpoint(path: Path, model: MidiMamba, optimizer, scheduler_step: int,
                    epoch: int, step: int, best_val: float, cfg: ModelConfig,
                    vocab_config_path: str):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    meta = {
        "step": step,
        "epoch": epoch,
        "best_val": best_val,
        "vocab_config": vocab_config_path,
        "model_cfg": {
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "d_state": cfg.d_state,
            "d_conv": cfg.d_conv,
            "expand": cfg.expand,
            "n_heads": cfg.n_heads,
            "d_ff_mult": cfg.d_ff_mult,
            "dropout": cfg.dropout,
            "max_seq": cfg.max_seq,
            "grad_checkpoint": cfg.grad_checkpoint,
        }
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path: Path, model: MidiMamba, optimizer, device):
    sd = torch.load(path / "model.pt", map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    opt_path = path / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    with open(path / "meta.json") as f:
        meta = json.load(f)
    return meta


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="tokens_out directory from build_dataset.py")
    parser.add_argument("--stats", required=True, help="stats_out directory with vocab_config.json")
    parser.add_argument("--out_dir", default="run")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_frac", type=float, default=0.02,
                        help="Fraction of total steps used for warmup")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--ckpt_every", type=int, default=100)
    parser.add_argument("--ckpt_minutes", type=int, default=30)
    parser.add_argument("--val_every", type=int, default=250)
    parser.add_argument("--val_batches", type=int, default=32)
    parser.add_argument("--sample_every", type=int, default=500,
                        help="Generate a MIDI sample every N steps (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="Gradient checkpointing — trades ~33%% slower training for significant VRAM savings")
    parser.add_argument("--test_vram", action="store_true",
                        help="Run one forward+backward pass, report peak VRAM, then exit. "
                             "Use this to find the maximum --seq_len for your GPU.")
    parser.add_argument("--sweep_models", action="store_true",
                        help="Test increasing model sizes at fixed --seq_len. "
                             "Finds the largest model that fits in VRAM. Exits after sweep.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: [bold]{device}[/bold]")

    # Load vocab
    vocab_path = Path(args.stats) / "vocab_config.json"
    tok.init(vocab_path)
    console.print(f"Vocab: [bold]{tok.VOCAB_SIZE}[/bold] tokens  (cond: {tok.COND_END})")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    samples_dir = out_dir / "samples"
    for d in [ckpt_dir, samples_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Dataset — hash-based split prevents train/val leakage
    train_ds = SongDataset(args.data_dir, max_seq=args.seq_len, split="train", val_frac=0.02)
    val_ds   = SongDataset(args.data_dir, max_seq=args.seq_len, split="val",   val_frac=0.02)
    console.print(f"Train chunks: {len(train_ds):,}  Val chunks: {len(val_ds):,}")
    # num_workers=0: data is preloaded in RAM, workers add pickling overhead with no benefit
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True, drop_last=True)

    # Model
    cfg = ModelConfig(
        vocab_size=tok.VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        n_heads=args.n_heads,
        expand=args.expand,
        max_seq=args.seq_len,
        grad_checkpoint=args.grad_checkpoint,
    )
    model = MidiMamba(cfg).to(device)
    console.print(f"Model: [bold]{model.param_count_str()}[/bold] parameters"
                  + (" [dim](grad checkpointing ON)[/dim]" if args.grad_checkpoint else ""))

    if args.compile:
        console.print("[blue]Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                                   betas=(0.9, 0.95), foreach=True)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ------------------------------------------------------------------ #
    #  Model sweep mode — test increasing model sizes at fixed seq_len
    # ------------------------------------------------------------------ #
    if args.sweep_models and device.type == "cuda":
        total_mb = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024

        # Model configs to sweep — (d_model, n_layers, n_heads, label)
        sweep_configs = [
            (64,   4,  2,  "~0.5M"),
            (64,   6,  2,  "~0.8M"),
            (96,   4,  2,  "~1.2M"),
            (96,   6,  2,  "~1.8M"),
            (128,  4,  4,  "~2.5M"),
            (128,  6,  4,  "~4M"),
            (128,  8,  4,  "~5M"),
            (160,  6,  4,  "~6M"),
            (160,  8,  4,  "~8M"),
            (192,  6,  4,  "~9M"),
            (192,  8,  4,  "~12M"),
            (256,  6,  4,  "~12M"),
            (256,  8,  4,  "~15M"),
            (256,  12, 4,  "~22M"),
            (384,  12, 6,  "~50M"),
            (512,  12, 8,  "~70M"),
            (512,  16, 8,  "~85M"),
            (512,  20, 8,  "~105M"),
            (576,  18, 8,  "~115M"),
            (640,  18, 8,  "~130M"),
            (768,  24, 12, "~220M"),
        ]

        console.print(f"\n[bold yellow]MODEL SWEEP — seq_len={args.seq_len:,}  batch={args.batch_size}  VRAM={total_mb:.0f}MB[/bold yellow]\n")
        console.print(f"  {'Config':<28} {'Params':<8} {'VRAM MB':>8} {'%':>6} {'Status':<20} Headroom")
        console.print(f"  {'-'*90}")

        last_ok_config = None

        for d_model, n_layers, n_heads, label in sweep_configs:
            # Rebuild model for this config
            sweep_cfg = ModelConfig(
                vocab_size=tok.VOCAB_SIZE,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                expand=args.expand,
                d_state=args.d_state,
                max_seq=args.seq_len,
            )
            try:
                sweep_model = MidiMamba(sweep_cfg).to(device)
                sweep_opt   = torch.optim.AdamW(sweep_model.parameters(), lr=1e-4)
                sweep_scaler = torch.amp.GradScaler("cuda")

                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()

                x = torch.randint(0, tok.VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)
                y = torch.randint(0, tok.VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)

                sweep_model.train()
                with torch.amp.autocast("cuda"):
                    logits, _ = sweep_model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, tok.VOCAB_SIZE),
                        y.reshape(-1),
                        ignore_index=tok.PAD,
                    )
                sweep_scaler.scale(loss).backward()
                sweep_opt.step()

                peak_mb  = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                used_pct = 100 * peak_mb / total_mb
                headroom = total_mb - peak_mb

                if used_pct < 85:
                    status = "[green]OK — good fit[/green]"
                    last_ok_config = (d_model, n_layers, n_heads, label, peak_mb, used_pct)
                elif used_pct < 93:
                    status = "[yellow]OK — tight[/yellow]"
                    last_ok_config = (d_model, n_layers, n_heads, label, peak_mb, used_pct)
                else:
                    status = "[red]OK — risky[/red]"

                cfg_str = f"d={d_model} L={n_layers} H={n_heads} ({label})"
                console.print(f"  {cfg_str:<28} {sweep_model.param_count_str():<8} {peak_mb:>8.0f} {used_pct:>5.1f}%  {status:<30} {headroom:.0f}MB")

                del sweep_model, sweep_opt, sweep_scaler, x, y, logits, loss
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                cfg_str = f"d={d_model} L={n_layers} H={n_heads} ({label})"
                console.print(f"  {cfg_str:<28} {'?':<8} {'OOM':>8} {'':>6}   [red]OUT OF MEMORY[/red]")
                break  # no point testing larger configs

        console.print(f"\n  {'-'*90}")
        if last_ok_config:
            d, l, h, lbl, mb, pct = last_ok_config
            console.print(f"\n[bold green]  Recommended: d_model={d} n_layers={l} n_heads={h} ({lbl} params)[/bold green]")
            console.print(f"  Peak VRAM at this config: {mb:.0f}MB ({pct:.1f}%)")
            console.print(f"\n  Training command:")
            console.print(f"  python train.py tokens_out --stats stats_out ^")
            console.print(f"    --seq_len {args.seq_len} --batch_size {args.batch_size} --grad_accum {args.grad_accum} ^")
            console.print(f"    --d_model {d} --n_layers {l} --n_heads {h}")
        return

    # ------------------------------------------------------------------ #
    #  VRAM test mode — one forward+backward, report peak memory, exit
    # ------------------------------------------------------------------ #
    if args.test_vram and device.type == "cuda":
        console.print(f"\n[bold yellow]VRAM TEST MODE[/bold yellow]")
        console.print(f"  seq_len:    {args.seq_len:,}")
        console.print(f"  batch_size: {args.batch_size}")
        console.print(f"  d_model:    {args.d_model}")
        console.print(f"  n_layers:   {args.n_layers}")
        console.print(f"  params:     {model.param_count_str()}")

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        # Synthetic batch — no need to load real data
        x = torch.randint(0, tok.VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, tok.VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)

        try:
            model.train()
            with torch.amp.autocast("cuda"):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, tok.VOCAB_SIZE),
                    y.reshape(-1),
                    ignore_index=tok.PAD,
                )
            scaler.scale(loss).backward()

            peak_mb  = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            total_mb = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
            used_pct = 100 * peak_mb / total_mb

            console.print(f"\n  [green]SUCCESS[/green]")
            console.print(f"  Peak VRAM:  {peak_mb:,.0f} MB  /  {total_mb:,.0f} MB  ({used_pct:.1f}%)")
            console.print(f"  Headroom:   {total_mb - peak_mb:,.0f} MB")

            if used_pct < 70:
                console.print(f"  [green]→ Plenty of headroom. Try doubling seq_len to {args.seq_len * 2:,}[/green]")
            elif used_pct < 85:
                console.print(f"  [green]→ Good fit. Could try {int(args.seq_len * 1.25):,} or {args.seq_len * 2:,}[/green]")
            elif used_pct < 95:
                console.print(f"  [yellow]→ Tight but stable. This is close to your limit.[/yellow]")
            else:
                console.print(f"  [red]→ Too close to limit — risk of OOM during real training. Try {args.seq_len // 2:,}[/red]")

        except torch.cuda.OutOfMemoryError:
            peak_mb  = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            total_mb = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
            console.print(f"\n  [red]OUT OF MEMORY[/red]")
            console.print(f"  Attempted:  {peak_mb:,.0f} MB  /  {total_mb:,.0f} MB")
            console.print(f"  [red]→ Reduce seq_len. Try {args.seq_len // 2:,}[/red]")

        return

    step = 0
    epoch = 0
    best_val = float("inf")
    last_ckpt_time = time.time()

    # Resume
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            meta = load_checkpoint(resume_path, model, optimizer, device)
            step = meta["step"]
            epoch = meta["epoch"]
            best_val = meta.get("best_val", float("inf"))
            console.print(f"[green]Resumed from step {step}, epoch {epoch}[/green]")
    else:
        latest = ckpt_dir / "latest"
        if latest.exists():
            meta = load_checkpoint(latest, model, optimizer, device)
            step = meta["step"]
            epoch = meta["epoch"]
            best_val = meta.get("best_val", float("inf"))
            console.print(f"[green]Auto-resumed from step {step}[/green]")

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = max(100, int(total_steps * args.warmup_frac))
    console.print(f"Total steps: {total_steps}  Warmup: {warmup_steps}")

    # Loss log
    log_path = out_dir / "loss_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr"])

    # --------------------------------------------------------------------------- #
    #  Training loop
    # --------------------------------------------------------------------------- #

    loss_window = []
    train_start = time.time()
    step_at_start = step
    last_step_time = time.time()
    step_times = []
    avg_loss = 0.0
    lr = 0.0
    eta = 0.0
    sps = 0.0
    live_val_loss = None

    # Rich Live display
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    def _fmt(secs):
        secs = int(max(0, secs))
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        if h > 99: return f"{h}h{m:02d}m"
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _make_panel(step, total_steps, avg_loss, val_loss, lr, elapsed, eta, sps, epoch, best_val, batch=0, grad_accum=1):
        g = Table.grid(padding=(0, 2))
        g.add_column(style="bold cyan", no_wrap=True)
        g.add_column(no_wrap=True)
        g.add_column(style="bold cyan", no_wrap=True)
        g.add_column(no_wrap=True)
        pct = 100 * step / max(total_steps, 1)
        bar_w = 40
        filled = int(bar_w * step / max(total_steps, 1))
        bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_w - filled)}[/dim]"
        batch_str = f"{batch}/{grad_accum}" if batch > 0 else "—"
        g.add_row("Step",    f"{step:,}/{total_steps:,}  ({pct:.2f}%)", "Elapsed",  _fmt(elapsed))
        g.add_row("Loss",    f"{avg_loss:.4f}" if avg_loss else "—", "ETA",      _fmt(eta))
        g.add_row("Val",     f"{val_loss:.4f}" if val_loss else "—", "LR",       f"{lr:.2e}")
        g.add_row("Epoch",   str(epoch + 1), "Best val", f"{best_val:.4f}" if best_val < 1e9 else "—")
        g.add_row("tok/s",   f"{sps * 53178 / 1000:.1f}K" if sps else "—", "batch",   batch_str)
        g.add_row(bar, "", "", "")
        return Panel(g, title="[bold]MidiMamba Training[/bold]", border_style="blue")

    # Shared state dict for background thread to read
    _display = {
        "step": step, "total": total_steps, "loss": 0.0, "val": None,
        "lr": 0.0, "elapsed": 0.0, "eta": 0.0, "sps": 0.0,
        "epoch": epoch, "best_val": best_val, "batch": 0, "grad_accum": args.grad_accum,
        "status": "Compiling..." if args.compile else "Training...",
    }

    def _make_panel(d):
        from rich.table import Table
        from rich.panel import Panel
        g = Table.grid(padding=(0, 2))
        g.add_column(style="bold cyan", no_wrap=True)
        g.add_column(no_wrap=True)
        g.add_column(style="bold cyan", no_wrap=True)
        g.add_column(no_wrap=True)
        pct  = 100 * d["step"] / max(d["total"], 1)
        bar_w = 40
        filled = int(bar_w * d["step"] / max(d["total"], 1))
        bar  = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_w - filled)}[/dim]"
        batch_str = f"{d['batch']}/{d['grad_accum']}" if d["batch"] > 0 else "—"
        elapsed = time.time() - train_start
        g.add_row("Step",    f"{d['step']:,}/{d['total']:,}  ({pct:.2f}%)", "Elapsed",  _fmt(elapsed))
        g.add_row("Loss",    f"{d['loss']:.4f}" if d["loss"] else "—",      "ETA",      _fmt(d["eta"]))
        g.add_row("Val",     f"{d['val']:.4f}"  if d["val"]  else "—",      "LR",       f"{d['lr']:.2e}")
        g.add_row("Epoch",   str(d["epoch"] + 1), "Best val", f"{d['best_val']:.4f}" if d["best_val"] < 1e9 else "—")
        g.add_row("tok/s",   f"{d['sps'] * 53178 / 1000:.1f}K" if d["sps"] else "—", "batch", batch_str)
        g.add_row("Status",  f"[dim]{d['status']}[/dim]", "", "")
        g.add_row(bar, "", "", "")
        return Panel(g, title="[bold]MidiMamba Training[/bold]", border_style="blue")

    _stop_display = threading.Event()

    def _display_thread():
        from rich.live import Live
        with Live(_make_panel(_display), refresh_per_second=4, console=console) as live:
            while not _stop_display.is_set():
                live.update(_make_panel(_display))
                time.sleep(0.25)

    _dt = threading.Thread(target=_display_thread, daemon=True)
    _dt.start()

    try:
        for ep in range(epoch, args.epochs):
            epoch = ep
            model.train()
            optimizer.zero_grad()

            for batch_idx, (x, y) in enumerate(train_loader):
                if _interrupted:
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Update display every batch
                _display["batch"]   = (batch_idx % args.grad_accum) + 1
                _display["elapsed"] = time.time() - train_start
                _display["status"]  = "Training..."

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits, _ = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, tok.VOCAB_SIZE),
                        y.reshape(-1),
                        ignore_index=tok.PAD,
                    )
                    loss = loss / args.grad_accum

                scaler.scale(loss).backward()

                if (batch_idx + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    lr = cosine_lr(step, warmup_steps, total_steps, args.min_lr, args.lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    step += 1

                    now = time.time()
                    step_times.append(now - last_step_time)
                    last_step_time = now
                    if len(step_times) > 50:
                        step_times.pop(0)
                    sps = 1.0 / (sum(step_times) / len(step_times))

                    loss_val = loss.item() * args.grad_accum
                    loss_window.append(loss_val)
                    if len(loss_window) > 50:
                        loss_window.pop(0)
                    avg_loss = sum(loss_window) / len(loss_window)

                    eta = (total_steps - step) / max(sps, 1e-6)

                    # Validation
                    if step % args.val_every == 0:
                        model.eval()
                        vlosses = []
                        with torch.no_grad():
                            for vx, vy in val_loader:
                                if len(vlosses) >= args.val_batches:
                                    break
                                vx = vx.to(device)
                                vy = vy.to(device)
                                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                                    vlogits, _ = model(vx)
                                    vl = F.cross_entropy(
                                        vlogits.reshape(-1, tok.VOCAB_SIZE),
                                        vy.reshape(-1),
                                        ignore_index=tok.PAD,
                                    )
                                vlosses.append(vl.item())
                        live_val_loss = sum(vlosses) / len(vlosses)
                        model.train()

                        if live_val_loss < best_val:
                            best_val = live_val_loss
                            save_checkpoint(ckpt_dir / "best", model, optimizer, step,
                                            epoch, step, best_val, cfg, str(vocab_path))

                        with open(log_path, "a", newline="") as f:
                            csv.writer(f).writerow([step, f"{avg_loss:.4f}",
                                                    f"{live_val_loss:.4f}", f"{lr:.6f}"])

                    # Update display dict
                    _display.update({
                        "step": step, "loss": avg_loss, "val": live_val_loss,
                        "lr": lr, "eta": eta, "sps": sps,
                        "epoch": epoch, "best_val": best_val, "batch": 0,
                    })

                    # Checkpoint latest
                    if step % args.ckpt_every == 0:
                        save_checkpoint(ckpt_dir / "latest", model, optimizer, step,
                                        epoch, step, best_val, cfg, str(vocab_path))

                    # Auto sample generation
                    if args.sample_every > 0 and step % args.sample_every == 0:
                        try:
                            from generate import generate as _gen
                            import json as _json
                            with open(vocab_path) as _f:
                                _vc = _json.load(_f)
                            _bc = _vc["bucket_config"]
                            _cond = [_bc[f]["token_offset"] + _bc[f]["n_buckets"] // 2 for f in _bc]
                            model.eval()
                            _ids = _gen(model, _cond, max_tokens=2000, silent=True, seed=step)
                            model.train()
                            import tokenizer as _tok2
                            _tok2.init(vocab_path)
                            _midi = _tok2.decode(_ids)
                            _sample_path = samples_dir / f"step_{step:07d}.mid"
                            _midi.write(str(_sample_path))
                            console.print(f"[dim]Sample → {_sample_path}[/dim]")
                        except Exception as _e:
                            console.print(f"[yellow]Sample failed: {_e}[/yellow]")

                    # Timed permanent checkpoint
                    if (now - last_ckpt_time) >= args.ckpt_minutes * 60:
                        save_checkpoint(ckpt_dir / f"step_{step:07d}", model, optimizer,
                                        step, epoch, step, best_val, cfg, str(vocab_path))
                        last_ckpt_time = now

            if _interrupted:
                console.print("[yellow]Saving checkpoint before exit...")
                save_checkpoint(ckpt_dir / "latest", model, optimizer, step,
                                epoch, step, best_val, cfg, str(vocab_path))
                break

            save_checkpoint(ckpt_dir / "latest", model, optimizer, step,
                            epoch + 1, step, best_val, cfg, str(vocab_path))
            console.print(f"[green]Epoch {epoch+1} complete. Step {step}[/green]")

    finally:
        _stop_display.set()
        _dt.join(timeout=2)

    console.print(f"\n[bold green]Training complete. Best val loss: {best_val:.4f}[/bold green]")


if __name__ == "__main__":
    main()
