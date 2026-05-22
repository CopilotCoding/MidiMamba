import argparse
import hashlib
import time
from collections import deque
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

import tokenizer as tok

console = Console()


# ------------------------------------------------------------
# FAST ANALYSIS
# ------------------------------------------------------------
def analyze_file(arr, token_counts):
    token_counts += np.bincount(arr, minlength=tok.VOCAB_SIZE)


# ------------------------------------------------------------
# FINGERPRINT (UNCHANGED)
# ------------------------------------------------------------
def fingerprint(arr):
    track = arr[(arr >= tok.TRACK_OFFSET) & (arr < tok.TRACK_OFFSET + 9)]
    slots = frozenset((track - tok.TRACK_OFFSET).tolist()) if len(track) else frozenset()

    pitch = arr[(arr >= tok.PITCH_OFFSET) & (arr < tok.PITCH_OFFSET + 88)]
    pitch_hist = np.bincount((pitch - tok.PITCH_OFFSET) % 12, minlength=12)
    pitch_sig = tuple((pitch_hist / (pitch_hist.sum() + 1e-9)).round(2))

    bars = int(np.count_nonzero(arr == tok.BAR))
    bar_bucket = (bars // 8) * 8

    dur = arr[(arr >= tok.DUR_OFFSET) & (arr < tok.DUR_OFFSET + 16)]
    dur_hist = np.bincount(dur - tok.DUR_OFFSET, minlength=16)
    dur_sig = tuple((dur_hist / (dur_hist.sum() + 1e-9)).round(2))

    return (slots, pitch_sig, bar_bucket, dur_sig)


# ------------------------------------------------------------
# PANEL
# ------------------------------------------------------------
def make_panel(i, total, lt, at, ht, slow=False):
    t = Table(title="Telemetry", expand=True)
    t.add_column("Metric")
    t.add_column("Value")

    t.add_row("progress", f"{i}/{total}")
    t.add_row("load", f"{lt*1000:.3f} ms")
    t.add_row("analyze", f"{at*1000:.3f} ms")
    t.add_row("hash", f"{ht*1000:.3f} ms")

    if slow:
        t.add_row("status", "SLOWDOWN DETECTED")

    return t


# ------------------------------------------------------------
# CHUNKING HELPERS (FIX 2)
# ------------------------------------------------------------
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--chunk_size", type=int, default=1000)
    args = parser.parse_args()

    tok.init(Path(args.stats) / "vocab_config.json")

    files = sorted(Path(args.data).glob("*_tokens.npy"))
    n = len(files)

    token_counts = np.zeros(tok.VOCAB_SIZE, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)

    seen = set()
    duplicates = 0

    load_times = deque(maxlen=200)
    analyze_times = deque(maxlen=200)
    hash_times = deque(maxlen=200)

    layout = Layout()
    layout.split_column(Layout(name="main"))

    total_processed = 0

    with Live(layout, console=console, refresh_per_second=12):

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[blue]Loading"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        )

        task = progress.add_task("", total=n)

        with progress:

            # ----------------------------------------------------
            # FIX 1: BATCH PROCESSING
            # ----------------------------------------------------
            for batch in chunk_list(files, args.chunk_size):

                for fpath in batch:

                    t0 = time.perf_counter()
                    arr = np.load(fpath, mmap_mode="r", allow_pickle=False)
                    t1 = time.perf_counter()

                    lengths[total_processed] = arr.shape[0]

                    analyze_file(arr, token_counts)
                    t2 = time.perf_counter()

                    h = hashlib.blake2b(arr.data, digest_size=16).digest()
                    t3 = time.perf_counter()

                    if h in seen:
                        duplicates += 1
                    else:
                        seen.add(h)

                    t4 = time.perf_counter()

                    lt = t1 - t0
                    at = t2 - t1
                    ht = t4 - t3

                    load_times.append(lt)
                    analyze_times.append(at)
                    hash_times.append(ht)

                    if total_processed % 200 == 0:
                        layout["main"].update(
                            make_panel(total_processed, n, lt, at, ht, False)
                        )

                    progress.update(task, advance=1)
                    total_processed += 1

    # ------------------------------------------------------------
    # FINAL STATS
    # ------------------------------------------------------------
    total_tokens = int(token_counts.sum())
    p50, p90, p99 = np.percentile(lengths, [50, 90, 99]).astype(int)

    console.print("\nSequence stats")
    console.print(f"Files: {n:,}")
    console.print(f"Min: {lengths.min():,}")
    console.print(f"Avg: {int(lengths.mean()):,}")
    console.print(f"P50: {p50:,}")
    console.print(f"P90: {p90:,}")
    console.print(f"P99: {p99:,}")
    console.print(f"Max: {lengths.max():,}")

    console.print("\nToken stats")
    console.print(f"Total tokens: {total_tokens:,}")
    console.print(f"Unique tokens: {np.count_nonzero(token_counts)}")

    probs = token_counts / token_counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    console.print(f"Entropy: {entropy:.2f} bits")

    console.print(f"\nDuplicates: {duplicates:,}")
    console.print("\nDONE")


if __name__ == "__main__":
    main()