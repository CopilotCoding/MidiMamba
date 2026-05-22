"""
train.py — MidiGen3

Key changes over midigen2:
  - bfloat16 autocast (native on RTX 5060 Ti / Blackwell sm_100)
  - torch.compile with Windows-safe fallback (no Triton required)
  - Best-val-loss checkpoint (was already partially in midigen2 but now primary)
  - Dataset uses dataset.py (memmap, no preload, per-batch padding)
  - Sweep table updated for MidiMamba (no n_heads column)
  - num_workers default 0 on Windows (avoids spawn IPC deadlocks with mmap)
  - Cosine LR with warmup (unchanged from midigen2)
  - Gradient accumulation (unchanged)

Usage:
    # VRAM sweep — find best model size for your GPU
    python train.py tokens_out --stats stats_out --sweep_models

    # Train
    python train.py tokens_out --stats stats_out ^
        --d_model 768 --n_layers 16 ^
        --seq_len 16384 --batch_size 1 --grad_accum 8

    # Resume
    python train.py tokens_out --stats stats_out ^
        --d_model 768 --n_layers 16 ^
        --resume run/checkpoints/latest
"""

import argparse
import contextlib
import csv
import json
import math
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

import tokenizer as tok
from dataset import make_loaders
from model import MidiMamba, ModelConfig

console = Console()
_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    console.print("\n[yellow]Interrupt received — saving checkpoint after this step.[/yellow]")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_sigint)


# --------------------------------------------------------------------------- #
#  LR schedule
# --------------------------------------------------------------------------- #

def cosine_lr(step: int, warmup: int, total: int, min_lr: float, max_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
#  Checkpoint
# --------------------------------------------------------------------------- #

def save_checkpoint(path: Path, model, optimizer, step: int, epoch: int,
                    best_val: float, cfg: ModelConfig, vocab_path: str):
    path.mkdir(parents=True, exist_ok=True)
    # Save raw state dict (no _orig_mod prefix issues on reload)
    sd = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save(sd, path / "model.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    meta = {
        "step":       step,
        "epoch":      epoch,
        "best_val":   best_val,
        "vocab_config": vocab_path,
        "model_cfg":  {
            "vocab_size": cfg.vocab_size,
            "d_model":    cfg.d_model,
            "n_layers":   cfg.n_layers,
            "d_conv":     cfg.d_conv,
            "expand":     cfg.expand,
            "d_ff_mult":  cfg.d_ff_mult,
            "dropout":    cfg.dropout,
            "max_seq":    cfg.max_seq,
            "grad_ckpt":  cfg.grad_ckpt,
        },
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path: Path, model, optimizer, device) -> dict:
    sd = torch.load(path / "model.pt", map_location=device, weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    opt_path = path / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(
            torch.load(opt_path, map_location=device, weights_only=True)
        )
    with open(path / "meta.json") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  torch.compile — Windows-safe wrapper
# --------------------------------------------------------------------------- #

def maybe_compile(model, use_compile: bool):
    """
    torch.compile on Windows requires Triton for the inductor backend,
    which has no official Windows wheel.  We try 'reduce-overhead' (which
    works without Triton on some setups) and fall back to uncompiled silently.
    The real speedups on this setup come from bfloat16, not compile.
    """
    if not use_compile:
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        console.print("[dim]torch.compile: reduce-overhead[/dim]")
        return compiled
    except Exception as e:
        console.print(f"[yellow]torch.compile unavailable ({e}) — running eager[/yellow]")
        return model


# --------------------------------------------------------------------------- #
#  Display
# --------------------------------------------------------------------------- #

def _fmt(secs: float) -> str:
    secs = int(max(0, secs))
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    if h > 99: return f"{h}h{m:02d}m"
    return f"{h:02d}:{m:02d}:{s:02d}"


def _make_panel(d: dict, train_start: float) -> Panel:
    g = Table.grid(padding=(0, 2))
    g.add_column(style="bold cyan", no_wrap=True)
    g.add_column(no_wrap=True)
    g.add_column(style="bold cyan", no_wrap=True)
    g.add_column(no_wrap=True)
    pct     = 100 * d["step"] / max(d["total"], 1)
    bar_w   = 38
    filled  = int(bar_w * d["step"] / max(d["total"], 1))
    bar     = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_w - filled)}[/dim]"
    elapsed = time.time() - train_start
    g.add_row("Step",   f"{d['step']:,}/{d['total']:,}  ({pct:.2f}%)", "Elapsed",  _fmt(elapsed))
    g.add_row("Loss",   f"{d['loss']:.4f}"  if d["loss"] else "—",    "ETA",      _fmt(d["eta"]))
    g.add_row("Val",    f"{d['val']:.4f}"   if d["val"]  else "—",    "LR",       f"{d['lr']:.2e}")
    g.add_row("Epoch",  str(d["epoch"] + 1),                           "Best val", f"{d['best_val']:.4f}" if d["best_val"] < 1e9 else "—")
    g.add_row("tok/s",  f"{d['tps']/1000:.1f}K" if d["tps"] else "—", "Status",   f"[dim]{d['status']}[/dim]")
    g.add_row(bar, "", "", "")
    return Panel(g, title="[bold]MidiMamba Training[/bold]", border_style="blue")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train MidiMamba")
    parser.add_argument("token_dir",             help="Directory of *_tokens.npy files")
    parser.add_argument("--stats",               required=True,  help="stats_out directory (contains vocab_config.json)")
    parser.add_argument("--out",                 default="run",  help="Output directory for checkpoints and logs")
    parser.add_argument("--resume",              default=None,   help="Checkpoint directory to resume from")

    # Model
    parser.add_argument("--d_model",             type=int,   default=512)
    parser.add_argument("--n_layers",            type=int,   default=12)
    parser.add_argument("--d_conv",              type=int,   default=4)
    parser.add_argument("--expand",              type=int,   default=2)
    parser.add_argument("--d_ff_mult",           type=float, default=2.667)
    parser.add_argument("--dropout",             type=float, default=0.1)
    parser.add_argument("--grad_checkpoint",     action="store_true")

    # Training
    parser.add_argument("--seq_len",             type=int,   default=16384)
    parser.add_argument("--batch_size",          type=int,   default=1)
    parser.add_argument("--grad_accum",          type=int,   default=8)
    parser.add_argument("--epochs",              type=int,   default=10)
    parser.add_argument("--lr",                  type=float, default=3e-4)
    parser.add_argument("--min_lr",              type=float, default=3e-5)
    parser.add_argument("--warmup_frac",         type=float, default=0.02)
    parser.add_argument("--weight_decay",        type=float, default=0.1)
    parser.add_argument("--grad_clip",           type=float, default=1.0)
    parser.add_argument("--compile",             action="store_true", help="Attempt torch.compile")

    # Dataset
    parser.add_argument("--val_frac",            type=float, default=0.02)
    parser.add_argument("--min_tokens",          type=int,   default=256)
    # num_workers=0 is safest on Windows with mmap; increase if you're on Linux
    parser.add_argument("--num_workers",         type=int,   default=0)

    # Checkpointing / eval
    parser.add_argument("--val_every",           type=int,   default=500)
    parser.add_argument("--val_batches",         type=int,   default=50)
    parser.add_argument("--ckpt_every",          type=int,   default=1000)
    parser.add_argument("--ckpt_minutes",        type=int,   default=30)
    parser.add_argument("--sample_every",        type=int,   default=0,
                        help="Generate a sample MIDI every N steps (0=off)")

    # Utility modes
    parser.add_argument("--sweep_models",        action="store_true",
                        help="Probe increasing model sizes to find largest that fits in VRAM, then exit.")
    parser.add_argument("--test_vram",           action="store_true",
                        help="Single forward+backward VRAM test, then exit.")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #
    out_dir     = Path(args.out)
    ckpt_dir    = out_dir / "checkpoints"
    samples_dir = out_dir / "samples"
    for d in [ckpt_dir, samples_dir]:
        d.mkdir(parents=True, exist_ok=True)

    vocab_path = Path(args.stats) / "vocab_config.json"
    if not vocab_path.exists():
        console.print(f"[red]vocab_config.json not found in {args.stats}[/red]")
        sys.exit(1)
    tok.init(vocab_path)
    console.print(f"Vocab: [bold]{tok.VOCAB_SIZE}[/bold] tokens  (COND_END={tok.COND_END})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: [bold]{device}[/bold]")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        console.print(f"GPU: {props.name}  {props.total_memory/1024**3:.1f} GB  "
                      f"sm_{props.major}{props.minor}")

    # ------------------------------------------------------------------ #
    #  Model
    # ------------------------------------------------------------------ #
    cfg = ModelConfig(
        vocab_size  = tok.VOCAB_SIZE,
        d_model     = args.d_model,
        n_layers    = args.n_layers,
        d_conv      = args.d_conv,
        expand      = args.expand,
        d_ff_mult   = args.d_ff_mult,
        dropout     = args.dropout,
        max_seq     = args.seq_len,
        grad_ckpt   = args.grad_checkpoint,
    )
    # Assertion in ModelConfig.__post_init__ ensures vocab_size > 0
    model = MidiMamba(cfg).to(device)
    console.print(f"Model: [bold]{model.param_count_str()}[/bold] parameters")

    # bfloat16 on Blackwell (sm_100) — fully supported, no GradScaler needed
    use_bf16   = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype  = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = not use_bf16   # GradScaler only needed for fp16
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)
    console.print(f"AMP: [bold]{'bfloat16' if use_bf16 else 'float16' if device.type=='cuda' else 'none'}[/bold]")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),   # fused AdamW: ~10% faster on CUDA
    )

    # ------------------------------------------------------------------ #
    #  VRAM sweep mode
    # ------------------------------------------------------------------ #
    if args.sweep_models and device.type == "cuda":
        total_mb = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024

        # (d_model, n_layers, label)
        sweep_configs = [

            # ------------------------------------------------------------------
            # Tiny sanity / baseline region
            # ------------------------------------------------------------------
            (32, 2, "~0.15M"),
            (32, 3, "~0.22M"),
            (32, 4, "~0.30M"),

            (40, 2, "~0.28M"),
            (40, 3, "~0.42M"),
            (40, 4, "~0.55M"),

            (48, 2, "~0.35M"),
            (48, 3, "~0.52M"),
            (48, 4, "~0.70M"),

            (56, 2, "~0.45M"),
            (56, 3, "~0.68M"),
            (56, 4, "~0.90M"),

            (64, 2, "~0.60M"),
            (64, 3, "~0.90M"),
            (64, 4, "~1.20M"),
            (64, 6, "~1.80M"),

            # ------------------------------------------------------------------
            # Small model region
            # ------------------------------------------------------------------
            (72, 2, "~0.75M"),
            (72, 3, "~1.10M"),
            (72, 4, "~1.50M"),
            (72, 6, "~2.20M"),

            (80, 2, "~1.00M"),
            (80, 3, "~1.50M"),
            (80, 4, "~2.00M"),
            (80, 6, "~3.00M"),

            (88, 2, "~1.20M"),
            (88, 3, "~1.80M"),
            (88, 4, "~2.40M"),
            (88, 6, "~3.60M"),

            (96, 2, "~1.50M"),
            (96, 3, "~2.20M"),
            (96, 4, "~3.00M"),
            (96, 6, "~4.50M"),

            # ------------------------------------------------------------------
            # Early useful regime
            # ------------------------------------------------------------------
            (112, 2, "~3.80M"),
            (112, 3, "~5.70M"),
            (112, 4, "~7.60M"),
            (112, 5, "~9.50M"),
            (112, 6, "~11.4M"),

            (128, 2, "~6.00M"),
            (128, 3, "~9.00M"),
            (128, 4, "~12.0M"),
            (128, 5, "~15.0M"),
            (128, 6, "~18.0M"),
            (128, 8, "~24.0M"),

            # ------------------------------------------------------------------
            # Mid regime
            # ------------------------------------------------------------------
            (144, 2, "~8.50M"),
            (144, 3, "~12.8M"),
            (144, 4, "~17.0M"),
            (144, 5, "~21.0M"),
            (144, 6, "~25.5M"),
            (144, 8, "~34.0M"),

            (160, 3, "~12.0M"),
            (160, 4, "~16.0M"),
            (160, 6, "~24.0M"),
            (160, 8, "~32.0M"),

            (176, 2, "~9.5M"),
            (176, 3, "~14.5M"),
            (176, 4, "~19.5M"),
            (176, 6, "~29.0M"),
            (176, 8, "~38.5M"),

            (192, 2, "~10.0M"),
            (192, 3, "~15.0M"),
            (192, 4, "~30.0M"),
            (192, 5, "~37.0M"),
            (192, 6, "~45.0M"),
            (192, 8, "~60.0M"),

            # ------------------------------------------------------------------
            # Strong mid-large regime
            # ------------------------------------------------------------------
            (224, 4, "~40.0M"),
            (224, 6, "~60.0M"),
            (224, 8, "~80.0M"),

            (256, 2, "~20.0M"),
            (256, 3, "~30.0M"),
            (256, 4, "~60.0M"),
            (256, 5, "~75.0M"),
            (256, 6, "~90.0M"),
            (256, 8, "~120.0M"),

            # ------------------------------------------------------------------
            # Large regime
            # ------------------------------------------------------------------
            (320, 4, "~100.0M"),
            (320, 6, "~150.0M"),
            (320, 8, "~200.0M"),

            (384, 4, "~120.0M"),
            (384, 6, "~170.0M"),
            (384, 7, "~200.0M"),
            (384, 8, "~230.0M"),

            # ------------------------------------------------------------------
            # Very large regime
            # ------------------------------------------------------------------
            (448, 6, "~240.0M"),
            (448, 8, "~320.0M"),

            (512, 6, "~260.0M"),
            (512, 8, "~320.0M"),
            (512, 10, "~400.0M"),
            (512, 12, "~480.0M"),

            # ------------------------------------------------------------------
            # Stress tail
            # ------------------------------------------------------------------
            (640, 8, "~400.0M+"),
            (640, 12, "~600.0M+"),
        ]

        console.print(f"\n[bold yellow]MODEL SWEEP — seq_len={args.seq_len:,}  "
                      f"batch={args.batch_size}  VRAM={total_mb:.0f}MB[/bold yellow]\n")
        console.print(f"  {'Config':<24} {'Params':<8} {'VRAM MB':>8} {'%':>6}  {'Status':<22} Headroom")
        console.print(f"  {'-'*80}")

        last_ok = None
        for d_model, n_layers, label in sweep_configs:
            sweep_cfg = ModelConfig(
                vocab_size=tok.VOCAB_SIZE, d_model=d_model, n_layers=n_layers,
                d_conv=args.d_conv, expand=args.expand,
                max_seq=args.seq_len, grad_ckpt=args.grad_checkpoint,
            )
            try:
                sm   = MidiMamba(sweep_cfg).to(device)
                sopt = torch.optim.AdamW(sm.parameters(), lr=1e-4)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
                x = torch.randint(0, tok.VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)
                y = torch.randint(0, tok.VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)
                sm.train()
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits, _ = sm(x)
                    loss = F.cross_entropy(logits.reshape(-1, tok.VOCAB_SIZE), y.reshape(-1))
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                sopt.step()
                
                torch.cuda.synchronize()
                
                peak_mb  = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                used_pct = 100 * peak_mb / total_mb
                headroom = total_mb - peak_mb

                # stop immediately if VRAM budget is exceeded (negative headroom)
                if headroom <= 0:
                    console.print(
                        f"  {cfg_str:<24} {'?':<8} {'OVR':>8} {'':>6}   "
                        f"[red]VRAM LIMIT EXCEEDED — stopping sweep[/red]"
                    )
                    del sm, sopt, x, y, logits, loss
                    torch.cuda.empty_cache()
                    break

                if used_pct < 85:
                    status = "[green]OK — good fit[/green]"
                    last_ok = (d_model, n_layers, label, peak_mb, used_pct)
                elif used_pct < 93:
                    status = "[yellow]OK — tight[/yellow]"
                    last_ok = (d_model, n_layers, label, peak_mb, used_pct)
                else:
                    status = "[red]OK — risky[/red]"

                cfg_str = f"d={d_model} L={n_layers} ({sm.param_count_str()})"
                console.print(f"  {cfg_str:<24} {sm.param_count_str():<8} "
                               f"{peak_mb:>8.0f} {used_pct:>5.1f}%  {status:<30} {headroom:.0f}MB")
                del sm, sopt, x, y, logits, loss
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                cfg_str = f"d={d_model} L={n_layers} ({sm.param_count_str()})"
                console.print(f"  {cfg_str:<24} {'?':<8} {'OOM':>8} {'':>6}   [red]OUT OF MEMORY — stopping sweep[/red]")
                break

        console.print(f"\n  {'-'*80}")
        if last_ok:
            d, l, lbl, mb, pct = last_ok
            console.print(f"\n[bold green]  Recommended: d_model={d}  n_layers={l}  ({lbl} params)[/bold green]")
            console.print(f"  Peak VRAM: {mb:.0f}MB ({pct:.1f}%)")
            console.print(f"\n  Training command:")
            console.print(f"  python train.py {args.token_dir} --stats {args.stats} ^")
            console.print(f"    --seq_len {args.seq_len} --batch_size {args.batch_size} "
                          f"--grad_accum {args.grad_accum} ^")
            console.print(f"    --d_model {d} --n_layers {l}")
        return

    # ------------------------------------------------------------------ #
    #  VRAM test mode
    # ------------------------------------------------------------------ #
    if args.test_vram and device.type == "cuda":
        total_mb = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
        console.print(f"\n[bold yellow]VRAM TEST  seq_len={args.seq_len:,}  "
                      f"batch={args.batch_size}  params={model.param_count_str()}[/bold yellow]")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        x = torch.randint(0, tok.VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, tok.VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)
        try:
            model.train()
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, tok.VOCAB_SIZE), y.reshape(-1))
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            torch.cuda.synchronize()
            
            peak_mb  = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            used_pct = 100 * peak_mb / total_mb
            console.print(f"\n  [green]SUCCESS[/green]  {peak_mb:,.0f} MB / {total_mb:,.0f} MB  ({used_pct:.1f}%)")
            if   used_pct < 70: console.print(f"  [green]→ Plenty of headroom.[/green]")
            elif used_pct < 85: console.print(f"  [green]→ Good fit.[/green]")
            elif used_pct < 95: console.print(f"  [yellow]→ Tight but stable.[/yellow]")
            else:               console.print(f"  [red]→ Risky — try reducing seq_len or batch_size.[/red]")
        except torch.cuda.OutOfMemoryError:
            console.print(f"\n  [red]OUT OF MEMORY[/red]")
        return

    # ------------------------------------------------------------------ #
    #  Compile (optional, Windows-safe)
    # ------------------------------------------------------------------ #
    if args.compile:
        model = maybe_compile(model, True)

    # ------------------------------------------------------------------ #
    #  Data loaders
    # ------------------------------------------------------------------ #
    train_loader, val_loader = make_loaders(
        token_dir   = args.token_dir,
        max_seq     = args.seq_len,
        batch_size  = args.batch_size,
        pad_id      = tok.PAD,
        num_workers = args.num_workers,
        val_frac    = args.val_frac,
        min_tokens  = args.min_tokens,
    )

    # ------------------------------------------------------------------ #
    #  Resume
    # ------------------------------------------------------------------ #
    step     = 0
    epoch    = 0
    best_val = float("inf")

    resume_path = None
    if args.resume and Path(args.resume).exists():
        resume_path = Path(args.resume)
    elif (ckpt_dir / "latest").exists():
        resume_path = ckpt_dir / "latest"

    if resume_path:
        meta     = load_checkpoint(resume_path, model, optimizer, device)
        step     = meta["step"]
        epoch    = meta["epoch"]
        best_val = meta.get("best_val", float("inf"))
        console.print(f"[green]Resumed from step {step}, epoch {epoch}[/green]")

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #
    total_steps   = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps  = max(100, int(total_steps * args.warmup_frac))
    console.print(f"Total steps: {total_steps:,}  Warmup: {warmup_steps:,}")

    log_path = out_dir / "loss_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr"])

    _display = {
        "step": step, "total": total_steps, "loss": 0.0, "val": None,
        "lr": 0.0, "eta": 0.0, "tps": 0.0, "epoch": epoch,
        "best_val": best_val, "status": "Starting...",
    }
    _stop_display = threading.Event()
    train_start   = time.time()

    def _display_thread():
        with Live(_make_panel(_display, train_start), refresh_per_second=4,
                  console=console) as live:
            while not _stop_display.is_set():
                live.update(_make_panel(_display, train_start))
                time.sleep(0.25)

    _dt = threading.Thread(target=_display_thread, daemon=True)
    _dt.start()

    loss_window:  list = []
    step_times:   list = []
    live_val_loss       = None
    last_step_time      = time.time()
    last_ckpt_time      = time.time()
    tokens_this_accum   = 0   # actual token count accumulated since last optimizer step

    try:
        for ep in range(epoch, args.epochs):
            epoch = ep
            model.train()
            optimizer.zero_grad()

            # SSM state persisted across chunks of the same song, reset at boundaries
            ssm_states = model.init_states(args.batch_size, device)
            current_song_id = None

            for batch_idx, (x, y, song_ids, is_lasts) in enumerate(train_loader):
                if _interrupted:
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                tokens_this_accum += x.numel()   # actual tokens in this batch (post-padding)
                _display["status"] = "Training..."

                # Reset state at song boundary (new song_id)
                if song_ids[0] != current_song_id:
                    ssm_states = model.init_states(args.batch_size, device)
                    current_song_id = song_ids[0]

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    logits, new_states = model(x, ssm_states)
                    loss = F.cross_entropy(
                        logits.reshape(-1, tok.VOCAB_SIZE),
                        y.reshape(-1),
                        ignore_index=tok.PAD,
                    ) / args.grad_accum

                # Detach states: backprop within chunk only (TBPTT), not through full song
                ssm_states = [s.detach() for s in new_states]

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % args.grad_accum == 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    lr = cosine_lr(step, warmup_steps, total_steps, args.min_lr, args.lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    step += 1

                    now = time.time()
                    step_times.append(now - last_step_time)
                    last_step_time = now
                    if len(step_times) > 50: step_times.pop(0)
                    sps = 1.0 / (sum(step_times) / len(step_times))
                    tps = tokens_this_accum / (sum(step_times) / len(step_times))
                    tokens_this_accum = 0

                    loss_val = loss.item() * args.grad_accum
                    loss_window.append(loss_val)
                    if len(loss_window) > 50: loss_window.pop(0)
                    avg_loss = sum(loss_window) / len(loss_window)
                    eta      = (total_steps - step) / max(sps, 1e-6)

                    # ---- Validation ---- #
                    if step % args.val_every == 0:
                        model.eval()
                        vlosses = []
                        with torch.no_grad():
                            for vx, vy, _sids, _ilasts in val_loader:
                                if len(vlosses) >= args.val_batches: break
                                with torch.amp.autocast("cuda", dtype=amp_dtype,
                                                        enabled=(device.type == "cuda")):
                                    vl = F.cross_entropy(
                                        model(vx.to(device))[0].reshape(-1, tok.VOCAB_SIZE),
                                        vy.to(device).reshape(-1),
                                        ignore_index=tok.PAD,
                                    )
                                vlosses.append(vl.item())
                        live_val_loss = sum(vlosses) / max(len(vlosses), 1)
                        model.train()

                        # Save best checkpoint
                        if live_val_loss < best_val:
                            best_val = live_val_loss
                            save_checkpoint(ckpt_dir / "best", model, optimizer, step,
                                            epoch, best_val, cfg, str(vocab_path))

                        with open(log_path, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [step, f"{avg_loss:.4f}", f"{live_val_loss:.4f}", f"{lr:.6f}"]
                            )

                    _display.update({
                        "step": step, "loss": avg_loss, "val": live_val_loss,
                        "lr": lr, "eta": eta, "tps": tps, "epoch": epoch, "best_val": best_val,
                    })

                    # ---- Checkpoints ---- #
                    if step % args.ckpt_every == 0:
                        save_checkpoint(ckpt_dir / "latest", model, optimizer, step,
                                        epoch, best_val, cfg, str(vocab_path))

                    if (now - last_ckpt_time) >= args.ckpt_minutes * 60:
                        save_checkpoint(ckpt_dir / f"step_{step:07d}", model, optimizer,
                                        step, epoch, best_val, cfg, str(vocab_path))
                        last_ckpt_time = now

                    # ---- Auto-sample ---- #
                    if args.sample_every > 0 and step % args.sample_every == 0:
                        try:
                            from generate import generate as _gen
                            import json as _json
                            with open(vocab_path) as _f:
                                _vc = _json.load(_f)
                            _bc   = _vc["bucket_config"]
                            _cond = [_bc[f]["token_offset"] + _bc[f]["n_buckets"] // 2
                                     for f in _bc]
                            model.eval()
                            _ids  = _gen(model, _cond, max_tokens=2000, silent=True, seed=step)
                            model.train()
                            _pm   = tok.decode(_ids)
                            _path = samples_dir / f"step_{step:07d}.mid"
                            _pm.write(str(_path))
                            _display["status"] = f"Sample → {_path.name}"
                        except Exception as _e:
                            _display["status"] = f"Sample failed: {_e}"

            if _interrupted:
                console.print("[yellow]Saving and exiting...[/yellow]")
                save_checkpoint(ckpt_dir / "latest", model, optimizer, step,
                                epoch, best_val, cfg, str(vocab_path))
                break

            save_checkpoint(ckpt_dir / "latest", model, optimizer, step,
                            epoch + 1, best_val, cfg, str(vocab_path))

    finally:
        _stop_display.set()
        _dt.join(timeout=2)

    console.print(f"\n[bold green]Done.  Best val loss: {best_val:.4f}[/bold green]")


if __name__ == "__main__":
    main()
