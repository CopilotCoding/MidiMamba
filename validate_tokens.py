"""
validate_tokens.py

Phase 3 (optional but recommended): Validate tokenized dataset quality.

Computes:
  - Token frequency table and top 100 tokens
  - Shannon entropy of token distribution
  - Sequence length histogram with P50/P90/P99/max
  - Conditioning token usage per dimension
  - Duplicate sequence detection
  - Warning flags for pathological distributions

Run after build_dataset.py to confirm training data is healthy before
spending GPU time on a poisoned dataset.

Usage:
    python validate_tokens.py --data tokens_out --stats stats_out
"""

import argparse
import json
import math
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
    parser.add_argument("--data", required=True, help="tokens_out directory from build_dataset.py")
    parser.add_argument("--stats", required=True, help="stats_out directory with vocab_config.json")
    parser.add_argument("--top_n", type=int, default=100, help="Top N tokens to show")
    parser.add_argument("--max_files", type=int, default=0, help="Cap files to sample (0=all)")
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

    token_counter: Counter = Counter()
    lengths = []
    seen_hashes: set = set()
    duplicates = 0
    import hashlib

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Loading"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("", total=len(files))
        for fpath in files:
            arr = np.load(str(fpath))
            lengths.append(len(arr))
            token_counter.update(arr.tolist())
            h = hashlib.sha256(arr.tobytes()).hexdigest()
            if h in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(h)
            progress.update(task, advance=1)

    total_tokens = sum(token_counter.values())
    n_files = len(files)

    # ------------------------------------------------------------------ #
    # Sequence length stats
    # ------------------------------------------------------------------ #
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    p50  = lengths_sorted[int(n * 0.50)]
    p90  = lengths_sorted[int(n * 0.90)]
    p99  = lengths_sorted[int(n * 0.99)]
    pmax = lengths_sorted[-1]
    pmin = lengths_sorted[0]
    pavg = int(np.mean(lengths))

    console.print("[bold]Sequence lengths:[/bold]")
    console.print(f"  Files:  {n_files:,}")
    console.print(f"  Min:    {pmin:,}")
    console.print(f"  Avg:    {pavg:,}")
    console.print(f"  P50:    {p50:,}")
    console.print(f"  P90:    {p90:,}")
    console.print(f"  P99:    {p99:,}")
    console.print(f"  Max:    {pmax:,}")

    # Warn if distribution is heavily skewed
    if pmax > p90 * 10:
        console.print(f"  [red]WARNING: Max ({pmax:,}) is >10x P90 ({p90:,}) — extreme outliers present[/red]")
    if pavg < p50 * 0.5:
        console.print(f"  [red]WARNING: Avg ({pavg:,}) << P50 ({p50:,}) — many tiny files skewing average[/red]")

    # ------------------------------------------------------------------ #
    # Token frequency and entropy
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold]Token distribution:[/bold]")
    console.print(f"  Total tokens:  {total_tokens:,}  ({total_tokens/1e6:.1f}M)")
    console.print(f"  Unique tokens: {len(token_counter):,} / {tok.VOCAB_SIZE}")

    # Shannon entropy
    probs = np.array(list(token_counter.values()), dtype=np.float64)
    probs /= probs.sum()
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    max_entropy = math.log2(tok.VOCAB_SIZE)
    entropy_pct = 100 * entropy / max_entropy
    console.print(f"  Entropy:       {entropy:.2f} bits  ({entropy_pct:.1f}% of max {max_entropy:.1f})")

    if entropy_pct < 40:
        console.print(f"  [red]WARNING: Low entropy ({entropy_pct:.1f}%) — token distribution is highly skewed. Training may be dominated by a few tokens.[/red]")
    elif entropy_pct < 60:
        console.print(f"  [yellow]CAUTION: Moderate entropy ({entropy_pct:.1f}%) — some token imbalance present.[/yellow]")
    else:
        console.print(f"  [green]Entropy looks healthy.[/green]")

    # ------------------------------------------------------------------ #
    # Duplicate sequences
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold]Duplicates:[/bold]")
    console.print(f"  Duplicate sequences: {duplicates:,}  ({100*duplicates/max(n_files,1):.1f}%)")
    if duplicates > 0:
        console.print(f"  [yellow]WARNING: {duplicates} duplicate sequences found. Run build_dataset.py again — it deduplicates automatically.[/yellow]")
    else:
        console.print(f"  [green]No duplicates detected.[/green]")

    # ------------------------------------------------------------------ #
    # Top N tokens
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold]Top {args.top_n} tokens:[/bold]")

    # Build reverse lookup: token_id -> human name
    def token_name(tid: int) -> str:
        if tid == tok.PAD:           return "PAD"
        if tid == tok.BOS:           return "BOS"
        if tid == tok.EOS:           return "EOS"
        if tid == tok.BAR:           return "BAR"
        if tid == tok.SECTION_EARLY: return "SECTION_EARLY"
        if tid == tok.SECTION_MID:   return "SECTION_MID"
        if tid == tok.SECTION_LATE:  return "SECTION_LATE"
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
            # Find which conditioning field
            for field, cfg in bucket_config.items():
                o = cfg["token_offset"]
                n = cfg["n_buckets"]
                if o <= tid < o + n:
                    return f"COND_{field}_b{tid-o}"
        return f"UNK_{tid}"

    table = Table(show_header=True, header_style="bold")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Token ID", width=10)
    table.add_column("Name", width=30)
    table.add_column("Count", width=14)
    table.add_column("% of total", width=12)

    top_tokens = token_counter.most_common(args.top_n)
    for rank, (tid, count) in enumerate(top_tokens, 1):
        pct = 100 * count / total_tokens
        color = "red" if pct > 15 else "yellow" if pct > 8 else ""
        name = token_name(tid)
        row = [str(rank), str(tid), name, f"{count:,}", f"{pct:.2f}%"]
        if color:
            table.add_row(*row, style=color)
        else:
            table.add_row(*row)

    console.print(table)

    # Warn if any single token dominates
    top_pct = 100 * top_tokens[0][1] / total_tokens if top_tokens else 0
    if top_pct > 20:
        console.print(f"[red]WARNING: Top token ({token_name(top_tokens[0][0])}) is {top_pct:.1f}% of all tokens — severely imbalanced.[/red]")

    # ------------------------------------------------------------------ #
    # Conditioning token usage per dimension
    # ------------------------------------------------------------------ #
    console.print(f"\n[bold]Conditioning token usage per dimension:[/bold]")

    cond_table = Table(show_header=True, header_style="bold")
    cond_table.add_column("Field", width=22)
    cond_table.add_column("Buckets", width=8)
    cond_table.add_column("Used", width=8)
    cond_table.add_column("Coverage", width=10)
    cond_table.add_column("Top bucket %", width=14)
    cond_table.add_column("Status", width=20)

    for field, cfg in bucket_config.items():
        offset = cfg["token_offset"]
        n_buckets = cfg["n_buckets"]
        bucket_counts = [token_counter.get(offset + i, 0) for i in range(n_buckets)]
        total_field = sum(bucket_counts)
        used = sum(1 for c in bucket_counts if c > 0)
        top_pct_field = 100 * max(bucket_counts) / max(total_field, 1)
        coverage = 100 * used / n_buckets

        if top_pct_field > 80:
            status = "[red]SKEWED[/red]"
        elif top_pct_field > 60:
            status = "[yellow]UNEVEN[/yellow]"
        elif coverage < 50:
            status = "[yellow]SPARSE[/yellow]"
        else:
            status = "[green]OK[/green]"

        cond_table.add_row(
            field,
            str(n_buckets),
            str(used),
            f"{coverage:.0f}%",
            f"{top_pct_field:.1f}%",
            status,
        )

    console.print(cond_table)
    # ------------------------------------------------------------------ #
    # Dead token detection
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Dead token detection:[/bold]")
    all_token_ids = set(range(tok.VOCAB_SIZE))
    used_token_ids = set(token_counter.keys())
    dead_tokens = all_token_ids - used_token_ids
    dead_pct = 100 * len(dead_tokens) / tok.VOCAB_SIZE

    console.print(f"  Vocab size:    {tok.VOCAB_SIZE:,}")
    console.print(f"  Tokens used:   {len(used_token_ids):,}")
    console.print(f"  Dead tokens:   {len(dead_tokens):,}  ({dead_pct:.1f}%)")

    if dead_pct > 30:
        console.print(f"  [red]WARNING: {dead_pct:.1f}% of vocab unused — tokenizer may be over-designed for this corpus.[/red]")
    elif dead_pct > 15:
        console.print(f"  [yellow]CAUTION: {dead_pct:.1f}% of vocab unused.[/yellow]")
    else:
        console.print("  [green]Vocab utilization looks healthy.[/green]")

    # ------------------------------------------------------------------ #
    # Near-duplicate detection via musical fingerprint
    # ------------------------------------------------------------------ #
    console.print("\n[bold]Near-duplicate detection:[/bold]")
    console.print("  [dim]Fingerprint = instrument slots + pitch class histogram + bar count + duration histogram[/dim]")

    fingerprints: Counter = Counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Fingerprinting"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress2:
        task2 = progress2.add_task("", total=len(files))
        for fpath in files:
            arr = np.load(str(fpath)).tolist()

            # Instrument slots used
            slots = frozenset(
                t - tok.TRACK_OFFSET
                for t in arr
                if tok.TRACK_OFFSET <= t < tok.TRACK_OFFSET + 9
            )

            # Pitch class histogram — ignores octave and transposition
            pitch_classes = [0] * 12
            for t in arr:
                if tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88:
                    pitch_classes[(t - tok.PITCH_OFFSET) % 12] += 1
            total_pitches = sum(pitch_classes) or 1
            pitch_sig = tuple(min(3, int(4 * c / total_pitches)) for c in pitch_classes)

            # Bar count bucketed to nearest 8
            n_bars_fp = sum(1 for t in arr if t == tok.BAR)
            bar_bucket = (n_bars_fp // 8) * 8

            # Duration histogram bucketed
            dur_counts = [0] * 16
            for t in arr:
                if tok.DUR_OFFSET <= t < tok.DUR_OFFSET + 16:
                    dur_counts[t - tok.DUR_OFFSET] += 1
            total_durs = sum(dur_counts) or 1
            dur_sig = tuple(min(3, int(4 * c / total_durs)) for c in dur_counts)

            fp = (slots, pitch_sig, bar_bucket, dur_sig)
            fingerprints[fp] += 1
            progress2.update(task2, advance=1)

    near_dup_groups = {fp: c for fp, c in fingerprints.items() if c > 1}
    near_dup_files = sum(near_dup_groups.values()) - len(near_dup_groups)
    near_dup_pct = 100 * near_dup_files / max(len(files), 1)

    console.print(f"  Near-duplicate groups: {len(near_dup_groups):,}")
    console.print(f"  Near-duplicate files:  {near_dup_files:,}  ({near_dup_pct:.1f}%)")

    if near_dup_pct > 20:
        console.print(f"  [red]WARNING: {near_dup_pct:.1f}% near-duplicates — heavy transpose spam or re-exports present.[/red]")
    elif near_dup_pct > 10:
        console.print(f"  [yellow]CAUTION: {near_dup_pct:.1f}% near-duplicates — some repetition in corpus.[/yellow]")
    else:
        console.print("  [green]Near-duplicate rate looks acceptable.[/green]")

    console.print("\n[bold green]Validation complete.[/bold green]")


if __name__ == "__main__":
    main()
