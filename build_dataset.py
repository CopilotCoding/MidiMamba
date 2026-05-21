"""
build_dataset.py

Phase 2: Tokenize all MIDI files to .npy token arrays.
- SHA256 deduplication of token sequences
- Structured error logging to errors.log
- Length percentiles (P50/P90/P99/max) in summary
- Chunked pool dispatch for Windows IPC efficiency

Usage:
    python build_dataset.py --stats stats_out --dirs "C:/bach" "C:/maestro" "C:/giantmidi" "C:/midicaps" --out tokens_out
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import traceback
import warnings
from pathlib import Path

import numpy as np
import pretty_midi
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

import tokenizer as tok

console = Console()

_vocab_config_path: str = ""


def _worker_init(vocab_path: str):
    global _vocab_config_path
    _vocab_config_path = vocab_path
    tok.init(vocab_path)


def _process_chunk(args_tuple) -> list[dict]:
    """
    Process a chunk of (path_str, out_dir, features_lookup) tuples.
    features_lookup: dict mapping path_str -> pre-scanned feature dict from corpus_stats.json
    If a file is in features_lookup, skips mido re-extraction entirely.
    Only pretty_midi is needed for note encoding.
    Returns list of result dicts — one IPC round trip per chunk.
    """
    pairs, out_dir, features_lookup = args_tuple
    out_path_base = Path(out_dir)
    results = []

    for path_str in pairs:
        try:
            path = Path(path_str)

            # Look up pre-scanned features — skip mido re-extraction if available
            features = features_lookup.get(path_str)

            if features is None:
                # Not in scan results — file failed pre-filter or feature extraction
                # during scanning. Skip it — no point attempting tokenization.
                results.append({"status": "skip", "src": path_str, "reason": "not_in_scan"})
                continue

            # Only pretty_midi needed now — just for note encoding
            pm = pretty_midi.PrettyMIDI(str(path))
            ids = tok.encode(pm, features)
            if len(ids) < 128:
                results.append({"status": "skip", "src": path_str, "reason": "too_short"})
                continue

            arr = np.array(ids, dtype=np.int32)
            sha = hashlib.sha256(arr.tobytes()).hexdigest()

            stem = hashlib.md5(path_str.encode()).hexdigest()[:16]
            out_path = out_path_base / f"{stem}_tokens.npy"
            np.save(str(out_path), arr)

            results.append({
                "status": "ok",
                "src": path_str,
                "out": str(out_path),
                "n_tokens": len(ids),
                "sha256": sha,
            })

        except Exception as e:
            results.append({
                "status": "error",
                "src": path_str,
                "exc_type": type(e).__name__,
                "exc_msg": str(e)[:200],
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--out", default="tokens_out")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=500)
    args = parser.parse_args()

    stats_dir = Path(args.stats)
    vocab_path = stats_dir / "vocab_config.json"
    if not vocab_path.exists():
        console.print(f"[red]vocab_config.json not found in {stats_dir}. Run scan_dataset.py first.")
        return

    tok.init(vocab_path)
    console.print(f"[green]Vocab loaded. VOCAB_SIZE = {tok.VOCAB_SIZE}, COND_END = {tok.COND_END}[/green]")

    # Load pre-scanned features from corpus_stats.json — eliminates mido re-extraction per file
    stats_path = stats_dir / "corpus_stats.json"
    features_lookup: dict[str, dict] = {}
    if stats_path.exists():
        import json as _json
        with open(stats_path) as f:
            corpus_stats = _json.load(f)
        features_lookup = {r["path"]: r for r in corpus_stats}
        console.print(f"[green]Loaded {len(features_lookup):,} pre-scanned feature records — skipping mido re-extraction[/green]")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    error_log_path = out_dir / "errors.log"

    # Collect paths
    all_paths = []
    for d in args.dirs:
        p = Path(d)
        if not p.exists():
            console.print(f"[yellow]Warning: {d} not found, skipping")
            continue
        found = list(p.rglob("*.mid")) + list(p.rglob("*.midi")) + list(p.rglob("*.MID"))
        # Deduplicate by lowercase path — Windows rglob returns *.mid and *.MID
        # as separate matches for the same file on case-insensitive NTFS
        seen_lower: set[str] = set()
        deduped = []
        for f in found:
            key = str(f).lower()
            if key not in seen_lower:
                seen_lower.add(key)
                deduped.append(f)
        found = deduped
        if args.limit:
            found = found[:args.limit]
        console.print(f"  {d}: {len(found)} files")
        all_paths.extend(found)

    console.print(f"\nTotal: [bold]{len(all_paths)}[/bold] files to tokenize\n")

    # Build chunks
    path_strs = [str(p) for p in all_paths]
    chunk_size = args.chunk_size
    chunks = [path_strs[i:i+chunk_size] for i in range(0, len(path_strs), chunk_size)]
    chunk_args = [(chunk, str(out_dir), features_lookup) for chunk in chunks]

    ok_results = []
    skipped = 0
    errors = 0
    total_tokens = 0
    seen_hashes: dict[str, str] = {}  # sha256 -> first src path
    duplicates = 0
    error_lines = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Tokenizing"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[tokens]}M tok | {task.fields[dups]} dups[/dim]"),
    ) as progress:
        task = progress.add_task("", total=len(path_strs), tokens=0, dups=0)

        with mp.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(str(vocab_path),),
        ) as pool:
            for batch in pool.imap_unordered(_process_chunk, chunk_args):
                for r in batch:
                    if r["status"] == "ok":
                        sha = r["sha256"]
                        if sha in seen_hashes:
                            # Duplicate — delete the file we just wrote
                            duplicates += 1
                            Path(r["out"]).unlink(missing_ok=True)
                        else:
                            seen_hashes[sha] = r["src"]
                            ok_results.append(r)
                            total_tokens += r["n_tokens"]
                    elif r["status"] == "skip":
                        skipped += 1
                    elif r["status"] == "error":
                        errors += 1
                        error_lines.append(
                            f"{r['exc_type']} | {r['exc_msg']} | {r['src']}\n"
                        )
                progress.update(task, advance=chunk_size,
                                tokens=total_tokens/1e6, dups=duplicates)

    # Write error log
    if error_lines:
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.writelines(error_lines)
        console.print(f"[yellow]Error log → {error_log_path} ({len(error_lines)} entries)[/yellow]")

    console.print(f"\n[green]Tokenized:  {len(ok_results):,} files[/green]")
    console.print(f"[yellow]Duplicates: {duplicates:,} removed[/yellow]  ({100*duplicates/max(len(ok_results)+duplicates,1):.1f}%)")
    console.print(f"[dim]Skipped:    {skipped:,}[/dim]")
    console.print(f"[red]Errors:     {errors:,}[/red]")
    console.print(f"\nTotal tokens: [bold]{total_tokens:,}[/bold]  ({total_tokens/1e6:.1f}M)")

    # Length percentiles + outlier trimming
    lengths = sorted([r["n_tokens"] for r in ok_results])
    trimmed = 0
    if lengths:
        n = len(lengths)
        p50  = lengths[int(n * 0.50)]
        p90  = lengths[int(n * 0.90)]
        p99  = lengths[int(n * 0.99)]
        pmax = lengths[-1]
        pavg = total_tokens // max(n, 1)

        # Trim sequences longer than 5x P99 — DAW timelines, broken exports, etc.
        trim_threshold = p99 * 5
        if pmax > trim_threshold:
            trimmed_results = []
            for r in ok_results:
                if r["n_tokens"] > trim_threshold:
                    Path(r["out"]).unlink(missing_ok=True)
                    trimmed += 1
                else:
                    trimmed_results.append(r)
            ok_results = trimmed_results
            total_tokens = sum(r["n_tokens"] for r in ok_results)
            console.print(f"[yellow]Trimmed {trimmed} outlier sequences longer than {trim_threshold:,} tokens (5x P99)[/yellow]")

        console.print(f"\nSequence lengths:")
        console.print(f"  Avg: {pavg:,}  P50: {p50:,}  P90: {p90:,}  P99: {p99:,}  Max: {pmax:,}")
        if trimmed:
            console.print(f"  Trim threshold: {trim_threshold:,}  Removed: {trimmed}")

    # Save manifest
    manifest = {
        "n_files": len(ok_results),
        "n_duplicates": duplicates,
        "n_skipped": skipped,
        "n_errors": errors,
        "total_tokens": total_tokens,
        "vocab_size": tok.VOCAB_SIZE,
        "cond_end": tok.COND_END,
        "length_percentiles": {
            "p50": p50, "p90": p90, "p99": p99, "max": pmax, "avg": pavg
        } if lengths else {},
        "files": [{"src": r["src"], "out": r["out"], "n_tokens": r["n_tokens"]} for r in ok_results],
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    console.print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
