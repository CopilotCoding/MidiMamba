"""
eval_generated.py

Post-training generation quality evaluation.
Generates N samples from a checkpoint and computes objective metrics.

Don't trust your ears alone. Your ears forgive patterns. Metrics expose them.

Metrics computed per sample and averaged:
  - Duplicate note %       : same pitch active twice simultaneously
  - Repeated n-gram rate   : fraction of 4-grams that repeat within the sequence
  - Unique token ratio     : unique tokens / total tokens (low = repetitive)
  - Note density           : notes per bar (variance across samples flagged)
  - Pitch entropy          : Shannon entropy of pitch distribution (low = monotone)
  - Bar similarity score   : avg token overlap between consecutive bars

Usage:
    python eval_generated.py run/checkpoints/best --stats stats_out --n 20
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

import tokenizer as tok
from model import MidiMamba, ModelConfig

console = Console()


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def duplicate_note_pct(token_ids: list[int]) -> float:
    """
    Percentage of POS groups where the same pitch appears more than once
    in the same slot — a physical impossibility that indicates generation artifacts.
    """
    total_groups = 0
    dup_groups = 0
    i = 0
    n = len(token_ids)
    while i < n:
        t = token_ids[i]
        if tok.POS_OFFSET <= t < tok.POS_OFFSET + 16:
            i += 1
            if i < n and tok.TRACK_OFFSET <= token_ids[i] < tok.TRACK_OFFSET + 9:
                i += 1
                pitches_here = []
                while i < n and tok.PITCH_OFFSET <= token_ids[i] < tok.PITCH_OFFSET + 88:
                    pitches_here.append(token_ids[i])
                    i += 1
                    # skip dur + vel
                    if i < n and tok.DUR_OFFSET <= token_ids[i] < tok.DUR_OFFSET + 16:
                        i += 1
                    if i < n and tok.VEL_OFFSET <= token_ids[i] < tok.VEL_OFFSET + 8:
                        i += 1
                total_groups += 1
                if len(pitches_here) != len(set(pitches_here)):
                    dup_groups += 1
        else:
            i += 1
    return 100 * dup_groups / max(total_groups, 1)


def repeated_ngram_rate(token_ids: list[int], n: int = 4) -> float:
    """
    Fraction of n-grams in the sequence that have appeared before.
    High rate = model is looping.
    Only considers music tokens (after BOS), ignores conditioning prefix.
    """
    # Start from BOS
    start = 0
    for i, t in enumerate(token_ids):
        if t == tok.BOS:
            start = i + 1
            break

    seq = token_ids[start:]
    if len(seq) < n * 2:
        return 0.0

    seen = set()
    repeated = 0
    total = 0
    for i in range(len(seq) - n):
        gram = tuple(seq[i:i+n])
        if gram in seen:
            repeated += 1
        seen.add(gram)
        total += 1

    return repeated / max(total, 1)


def unique_token_ratio(token_ids: list[int]) -> float:
    """Unique tokens / total tokens. Low = repetitive output."""
    start = 0
    for i, t in enumerate(token_ids):
        if t == tok.BOS:
            start = i + 1
            break
    seq = token_ids[start:]
    if not seq:
        return 0.0
    return len(set(seq)) / len(seq)


def pitch_entropy(token_ids: list[int]) -> float:
    """Shannon entropy of pitch token distribution. Low = monotone melody."""
    pitches = [t for t in token_ids if tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88]
    if not pitches:
        return 0.0
    counts = Counter(pitches)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def note_density_per_bar(token_ids: list[int]) -> float:
    """Average notes per bar."""
    bars = sum(1 for t in token_ids if t == tok.BAR)
    pitches = sum(1 for t in token_ids if tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88)
    return pitches / max(bars, 1)


def bar_similarity_score(token_ids: list[int], n_compare: int = 8) -> float:
    """
    Average token overlap between consecutive bars.
    High = model is generating repetitive bars.
    """
    bar_starts = [i for i, t in enumerate(token_ids) if t == tok.BAR]
    if len(bar_starts) < 2:
        return 0.0

    similarities = []
    for i in range(1, min(len(bar_starts), n_compare + 1)):
        s1 = bar_starts[i-1]
        e1 = bar_starts[i]
        s2 = bar_starts[i]
        e2 = bar_starts[i+1] if i+1 < len(bar_starts) else len(token_ids)
        bar1 = token_ids[s1:e1]
        bar2 = token_ids[s2:e2]
        clen = min(len(bar1), len(bar2), 32)
        if clen < 4:
            continue
        matches = sum(a == b for a, b in zip(bar1[:clen], bar2[:clen]))
        similarities.append(matches / clen)

    return float(np.mean(similarities)) if similarities else 0.0


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint directory")
    parser.add_argument("--stats", required=True, help="stats_out directory with vocab_config.json")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.92)
    parser.add_argument("--top_p", type=float, default=0.93)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    with open(ckpt / "meta.json") as f:
        meta = json.load(f)

    vocab_path = Path(meta["vocab_config"])
    tok.init(vocab_path)

    with open(vocab_path) as f:
        vocab_config = json.load(f)

    cfg_dict = meta["model_cfg"]
    cfg_dict.pop("n_heads", None)   # midigen2 checkpoints may have n_heads — not in ModelConfig
    cfg = ModelConfig(**cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: [bold]{device}[/bold]")

    model = MidiMamba(cfg).to(device)
    sd = torch.load(ckpt / "model.pt", map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    console.print(f"Model: [bold]{model.param_count_str()}[/bold]\n")

    # Import generate function
    from generate import generate

    # Use neutral (unconditioned) prefix for evaluation
    bucket_config = vocab_config["bucket_config"]
    cond_prefix = [
        cfg["token_offset"] + cfg["n_buckets"] // 2
        for cfg in bucket_config.values()
    ]

    # Collect metrics across N samples
    all_metrics = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Generating samples"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("", total=args.n)

        for i in range(args.n):
            token_ids = generate(
                model,
                cond_prefix=cond_prefix,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=i,  # different seed per sample
                silent=True,
            )

            metrics = {
                "n_tokens":          len(token_ids),
                "n_bars":            sum(1 for t in token_ids if t == tok.BAR),
                "duplicate_note_pct": duplicate_note_pct(token_ids),
                "ngram_repeat_rate": repeated_ngram_rate(token_ids, n=4),
                "unique_token_ratio": unique_token_ratio(token_ids),
                "pitch_entropy":     pitch_entropy(token_ids),
                "note_density":      note_density_per_bar(token_ids),
                "bar_similarity":    bar_similarity_score(token_ids),
            }
            all_metrics.append(metrics)
            progress.update(task, advance=1)

    # Aggregate
    def avg(key): return float(np.mean([m[key] for m in all_metrics]))
    def std(key): return float(np.std([m[key] for m in all_metrics]))

    console.print("\n[bold]Generation Quality Report[/bold]")
    console.print(f"  Samples: {args.n}  |  Max tokens: {args.max_tokens}\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric",         width=26)
    table.add_column("Mean",           width=12)
    table.add_column("Std",            width=12)
    table.add_column("Status",         width=20)

    def status(label, val, warn_thresh, bad_thresh, low_is_bad=True):
        if low_is_bad:
            if val < bad_thresh:  return f"[red]{label}: BAD[/red]"
            if val < warn_thresh: return f"[yellow]{label}: CAUTION[/yellow]"
            return f"[green]{label}: OK[/green]"
        else:
            if val > bad_thresh:  return f"[red]{label}: BAD[/red]"
            if val > warn_thresh: return f"[yellow]{label}: CAUTION[/yellow]"
            return f"[green]{label}: OK[/green]"

    rows = [
        ("Tokens per sample",    "n_tokens",           None),
        ("Bars per sample",      "n_bars",             None),
        ("Duplicate note %",     "duplicate_note_pct", status("dup",  avg("duplicate_note_pct"), 5,  15,  low_is_bad=False)),
        ("4-gram repeat rate",   "ngram_repeat_rate",  status("rep",  avg("ngram_repeat_rate"),  0.3, 0.6, low_is_bad=False)),
        ("Unique token ratio",   "unique_token_ratio", status("uniq", avg("unique_token_ratio"),  0.15, 0.08, low_is_bad=True)),
        ("Pitch entropy (bits)", "pitch_entropy",       status("ent",  avg("pitch_entropy"),       3.0, 1.5, low_is_bad=True)),
        ("Notes per bar",        "note_density",        status("den",  avg("note_density"),        2.0, 0.5, low_is_bad=True)),
        ("Bar similarity",       "bar_similarity",      status("sim",  avg("bar_similarity"),      0.7, 0.85, low_is_bad=False)),
    ]

    for label, key, st in rows:
        mean_str = f"{avg(key):.3f}"
        std_str  = f"{std(key):.3f}"
        table.add_row(label, mean_str, std_str, st or "")

    console.print(table)

    # Overall verdict
    bad_flags = []
    if avg("ngram_repeat_rate") > 0.6:   bad_flags.append("high n-gram repetition")
    if avg("unique_token_ratio") < 0.08: bad_flags.append("low token diversity")
    if avg("pitch_entropy") < 1.5:       bad_flags.append("low pitch entropy (monotone)")
    if avg("bar_similarity") > 0.85:     bad_flags.append("high bar-to-bar similarity")
    if avg("duplicate_note_pct") > 15:   bad_flags.append("many duplicate notes")

    if bad_flags:
        console.print(f"\n[red]Model has quality issues: {', '.join(bad_flags)}[/red]")
        console.print("[red]Consider more training, better data filtering, or higher temperature.[/red]")
    else:
        console.print("\n[bold green]Model passes generation quality checks.[/bold green]")


if __name__ == "__main__":
    main()
