"""
validate_tokens.py

Phase 3 (optional but recommended): Validate tokenized dataset quality.
"""

import argparse
import json
import math
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

import tokenizer as tok

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()

    vocab_path = Path(args.stats) / "vocab_config.json"
    tok.init(vocab_path)

    with open(vocab_path) as f:
        vocab_config = json.load(f)
    bucket_config = vocab_config["bucket_config"]

    data_dir = Path(args.data)
    files = sorted(data_dir.glob("*_tokens.npy"))
    if args.max_files:
        files = files[:args.max_files]

    console.print(f"[blue]Validating {len(files)} token files...[/blue]\n")

    token_counter = Counter()
    lengths = np.empty(len(files), dtype=np.int64)

    seen_hashes = set()
    duplicates = 0

    # ------------------------------------------------------------
    # PASS 1
    # ------------------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Loading"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:

        task = progress.add_task("", total=len(files))

        for i, fpath in enumerate(files):
            arr = np.load(fpath, mmap_mode="r")

            lengths[i] = arr.shape[0]

            # FAST PATH: avoid Python list conversion explosion
            uniq, cnt = np.unique(arr, return_counts=True)
            token_counter.update(dict(zip(uniq.astype(np.int64), cnt.astype(np.int64))))

            h = hashlib.blake2b(arr.tobytes()).hexdigest()
            if h in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(h)

            progress.update(task, advance=1)

    total_tokens = sum(token_counter.values())
    n_files = len(files)

    # ------------------------------------------------------------
    # LENGTH STATS
    # ------------------------------------------------------------
    ls = np.sort(lengths)

    p50 = int(ls[int(n_files * 0.50)])
    p90 = int(ls[int(n_files * 0.90)])
    p99 = int(ls[int(n_files * 0.99)])
    pmax = int(ls[-1])
    pmin = int(ls[0])
    pavg = int(lengths.mean())

    console.print("[bold]Sequence lengths:[/bold]")
    console.print(f"  Files:  {n_files:,}")
    console.print(f"  Min:    {pmin:,}")
    console.print(f"  Avg:    {pavg:,}")
    console.print(f"  P50:    {p50:,}")
    console.print(f"  P90:    {p90:,}")
    console.print(f"  P99:    {p99:,}")
    console.print(f"  Max:    {pmax:,}")

    if pmax > p90 * 10:
        console.print(f"  [red]WARNING: Max outliers[/red]")
    if pavg < p50 * 0.5:
        console.print(f"  [red]WARNING: Skewed lengths[/red]")

    # ------------------------------------------------------------
    # TOKEN STATS
    # ------------------------------------------------------------
    console.print(f"\n[bold]Token distribution:[/bold]")
    console.print(f"  Total tokens:  {total_tokens:,}")
    console.print(f"  Unique tokens: {len(token_counter):,} / {tok.VOCAB_SIZE}")

    probs = np.fromiter(token_counter.values(), dtype=np.float64)
    probs /= probs.sum()

    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    max_entropy = math.log2(tok.VOCAB_SIZE)
    entropy_pct = 100 * entropy / max_entropy

    console.print(f"  Entropy: {entropy:.2f} bits ({entropy_pct:.1f}%)")

    # ------------------------------------------------------------
    # DUPLICATES
    # ------------------------------------------------------------
    console.print(f"\n[bold]Duplicates:[/bold]")
    console.print(f"  Duplicate sequences: {duplicates:,}")

    # ------------------------------------------------------------
    # TOP TOKENS
    # ------------------------------------------------------------
    console.print(f"\n[bold]Top {args.top_n} tokens:[/bold]")

    def token_name(tid: int) -> str:
        if tid == tok.PAD: return "PAD"
        if tid == tok.BOS: return "BOS"
        if tid == tok.EOS: return "EOS"
        if tid == tok.BAR: return "BAR"

        if tok.POS_OFFSET <= tid < tok.POS_OFFSET + 16:
            return f"POS_{tid - tok.POS_OFFSET}"
        if tok.TRACK_OFFSET <= tid < tok.TRACK_OFFSET + 9:
            return f"TRACK_{tid - tok.TRACK_OFFSET}"
        if tok.PITCH_OFFSET <= tid < tok.PITCH_OFFSET + 88:
            return f"PITCH_{tid - tok.PITCH_OFFSET + 21}"
        if tok.DUR_OFFSET <= tid < tok.DUR_OFFSET + 16:
            return f"DUR_{tid - tok.DUR_OFFSET + 1}"
        if tok.VEL_OFFSET <= tid < tok.VEL_OFFSET + 8:
            return f"VEL_{tid - tok.VEL_OFFSET + 1}"
        if tok.TEMPO_OFFSET <= tid < tok.TEMPO_OFFSET + 17:
            return f"TEMPO_{40 + (tid - tok.TEMPO_OFFSET)*10}"

        if tid < tok.COND_END:
            for field, cfg in bucket_config.items():
                o = cfg["token_offset"]
                if o <= tid < o + cfg["n_buckets"]:
                    return f"COND_{field}_b{tid-o}"

        return f"UNK_{tid}"

    table = Table(show_header=True)
    table.add_column("Rank")
    table.add_column("Token ID")
    table.add_column("Name")
    table.add_column("Count")
    table.add_column("%")

    for i, (tid, c) in enumerate(token_counter.most_common(args.top_n), 1):
        table.add_row(str(i), str(tid), token_name(tid), str(c), f"{100*c/total_tokens:.2f}")

    console.print(table)

    # ------------------------------------------------------------
    # CONDITIONING
    # ------------------------------------------------------------
    console.print("\n[bold]Conditioning token usage per dimension:[/bold]")

    cond_table = Table(show_header=True)
    cond_table.add_column("Field")
    cond_table.add_column("Buckets")
    cond_table.add_column("Used")
    cond_table.add_column("Coverage")
    cond_table.add_column("Top bucket %")
    cond_table.add_column("Status")

    for field, cfg in bucket_config.items():
        offset = cfg["token_offset"]
        n_buckets = cfg["n_buckets"]

        counts = np.array([token_counter.get(offset + i, 0) for i in range(n_buckets)], dtype=np.int64)

        total = counts.sum()
        used = int(np.count_nonzero(counts))
        coverage = 100 * used / n_buckets
        top_pct = 100 * counts.max() / max(total, 1)

        if top_pct > 80:
            status = "[red]SKEWED[/red]"
        elif top_pct > 60:
            status = "[yellow]UNEVEN[/yellow]"
        elif coverage < 50:
            status = "[yellow]SPARSE[/yellow]"
        else:
            status = "[green]OK[/green]"

        cond_table.add_row(field, str(n_buckets), str(used), f"{coverage:.0f}%", f"{top_pct:.1f}%", status)

    console.print(cond_table)

    # ------------------------------------------------------------
    # DEAD TOKENS
    # ------------------------------------------------------------
    used = set(token_counter.keys())
    dead_pct = 100 * (tok.VOCAB_SIZE - len(used)) / tok.VOCAB_SIZE

    console.print("\n[bold]Dead token detection:[/bold]")
    console.print(f"  Dead tokens: {dead_pct:.1f}%")

    # ------------------------------------------------------------
    # PASS 2 (FINGERPRINT - OPTIMIZED MEMORY ONLY)
    # ------------------------------------------------------------
    console.print("\n[bold]Near-duplicate detection:[/bold]")

    fingerprints = Counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Fingerprinting"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:

        task = progress.add_task("", total=len(files))

        for fpath in files:
            arr = np.load(fpath, mmap_mode="r")

            track = arr[(arr >= tok.TRACK_OFFSET) & (arr < tok.TRACK_OFFSET + 9)] - tok.TRACK_OFFSET
            slots = frozenset(track.tolist())

            pitch = arr[(arr >= tok.PITCH_OFFSET) & (arr < tok.PITCH_OFFSET + 88)]
            pitch_pc = np.bincount((pitch - tok.PITCH_OFFSET) % 12, minlength=12)
            pitch_sig = tuple((pitch_pc / (pitch_pc.sum() + 1e-9)).round(2))

            bars = int((arr == tok.BAR).sum())
            bar_bucket = (bars // 8) * 8

            dur = arr[(arr >= tok.DUR_OFFSET) & (arr < tok.DUR_OFFSET + 16)]
            dur_pc = np.bincount(dur - tok.DUR_OFFSET, minlength=16)
            dur_sig = tuple((dur_pc / (dur_pc.sum() + 1e-9)).round(2))

            fingerprints[(slots, pitch_sig, bar_bucket, dur_sig)] += 1
            progress.update(task, advance=1)

    near_dup = sum(v - 1 for v in fingerprints.values() if v > 1)
    console.print(f"  Near-duplicate files: {near_dup:,}")

    console.print("\n[bold green]Validation complete.[/bold green]")


if __name__ == "__main__":
    main()