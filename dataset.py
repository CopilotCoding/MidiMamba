"""
dataset.py — MidiGen3

Stateful training support: chunks are served in song order so the training
loop can thread SSM state across chunks of the same song, resetting only at
song boundaries.

Key changes from the stateless version:
  - __getitem__ returns (x, y, song_id, is_last_chunk) instead of (x, y)
  - Shuffle is at song level, not chunk level — chunks within a song stay ordered
  - song_id is a stable integer index into the sorted file list
  - is_last_chunk flags the final chunk of each song so the training loop resets state

Everything else unchanged: memmap, chunk/split cache.
"""

import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

console = Console()


class SongDataset(Dataset):
    """
    Each item: (x, y, song_id, is_last_chunk)
      x, y          token tensors for next-token prediction
      song_id       stable int identifying the source song
      is_last_chunk True if this is the final chunk of the song
    """

    def __init__(self, token_dir, max_seq, min_tokens=256, split="train", val_frac=0.02, pad_id=0):
        self.max_seq    = max_seq
        self.min_tokens = min_tokens
        self.pad_id     = pad_id
        self.chunks     = []  # (fpath, start, length, song_id, is_last_chunk)

        token_dir = Path(token_dir)
        files     = sorted(token_dir.glob("*_tokens.npy"))
        if not files:
            raise FileNotFoundError(f"No *_tokens.npy files found in {token_dir}")

        console.print(f"[blue]SongDataset: {len(files):,} token files  split={split}[/blue]")

        val_bucket_max   = int(val_frac * 100)
        split_cache_key  = f"{len(files)}_{val_frac}"
        split_cache_path = token_dir / ".split_cache.pkl"
        chunk_cache_key  = f"stateful_{len(files)}_{val_frac}_{max_seq}_{min_tokens}"
        chunk_cache_path = token_dir / f".chunk_cache_{split}.pkl"

        if chunk_cache_path.exists():
            try:
                with open(chunk_cache_path, "rb") as f:
                    cc = pickle.load(f)
                if cc.get("key") == chunk_cache_key:
                    self.chunks = cc["chunks"]
                    total_tok   = cc["total_tokens"]
                    console.print(f"[dim]Chunk cache hit — {len(self.chunks):,} chunks, {total_tok/1e6:.1f}M tokens[/dim]")
                    return
            except Exception:
                pass

        split_map    = {}
        file_lengths = {}

        if split_cache_path.exists():
            try:
                with open(split_cache_path, "rb") as f:
                    sc = pickle.load(f)
                if sc.get("key") == split_cache_key:
                    split_map    = sc["split_map"]
                    file_lengths = sc["file_lengths"]
                    console.print(f"[dim]Split cache hit — {len(split_map):,} entries[/dim]")
            except Exception:
                split_map = {}
                file_lengths = {}

        for fpath in files:
            fname = fpath.name
            if fname not in split_map:
                split_map[fname] = int(hashlib.md5(fname.encode()).hexdigest()[:4], 16) % 100

        files_needing_read = [fp for fp in files if fp.name not in file_lengths]
        if files_needing_read:
            console.print(f"[dim]Reading {len(files_needing_read):,} file lengths in parallel...[/dim]")
            def _read_len(fp):
                return fp.name, int(np.load(str(fp), mmap_mode="r").shape[0])
            with Progress(SpinnerColumn(), TextColumn("[blue]Scanning"), BarColumn(),
                          MofNCompleteColumn(), TimeRemainingColumn()) as prog:
                task = prog.add_task("", total=len(files_needing_read))
                with ThreadPoolExecutor(max_workers=16) as exe:
                    futs = {exe.submit(_read_len, fp): fp for fp in files_needing_read}
                    for fut in as_completed(futs):
                        fname, n = fut.result()
                        file_lengths[fname] = n
                        prog.advance(task)

        try:
            with open(split_cache_path, "wb") as f:
                pickle.dump({"key": split_cache_key, "split_map": split_map,
                             "file_lengths": file_lengths}, f)
        except Exception:
            pass

        total_tok = 0
        song_id   = 0
        for fpath in files:
            fname  = fpath.name
            bucket = split_map.get(fname, 0)
            in_val = bucket < val_bucket_max
            if split == "val"   and not in_val: continue
            if split == "train" and in_val:     continue
            n = file_lengths.get(fname, 0)
            if n < min_tokens:
                continue
            song_chunks = []
            for start in range(0, n, max_seq):
                length = min(max_seq, n - start)
                if length < min_tokens:
                    continue
                song_chunks.append((str(fpath), start, length, song_id, False))
                total_tok += length
            if song_chunks:
                last = song_chunks[-1]
                song_chunks[-1] = (last[0], last[1], last[2], last[3], True)
                self.chunks.extend(song_chunks)
                song_id += 1

        console.print(f"[green]{split}: {len(self.chunks):,} chunks from {song_id:,} songs, {total_tok/1e6:.1f}M tokens[/green]")

        try:
            with open(chunk_cache_path, "wb") as f:
                pickle.dump({"key": chunk_cache_key, "chunks": self.chunks, "total_tokens": total_tok}, f)
        except Exception:
            pass

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        fpath, start, length, song_id, is_last = self.chunks[idx]
        arr    = np.load(fpath, mmap_mode="r")[start : start + length].astype(np.int64)
        tokens = torch.from_numpy(arr)
        return tokens[:-1], tokens[1:], song_id, is_last


class SongOrderSampler(Sampler):
    """
    Shuffles at song level, preserves chunk order within each song.
    Ensures chunks of the same song are always served consecutively.
    """
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        songs = {}
        for idx, chunk in enumerate(dataset.chunks):
            sid = chunk[3]
            songs.setdefault(sid, []).append(idx)
        for sid in songs:
            songs[sid].sort(key=lambda i: dataset.chunks[i][1])
        self.songs = list(songs.values())

    def __iter__(self):
        order = torch.randperm(len(self.songs)).tolist() if self.shuffle else list(range(len(self.songs)))
        for song_idx in order:
            yield from self.songs[song_idx]

    def __len__(self):
        return len(self.dataset)


class StatefulCollator:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        xs, ys, song_ids, is_lasts = zip(*batch)
        max_len = max(x.shape[0] for x in xs)
        x_out = torch.full((len(xs), max_len), self.pad_id, dtype=torch.long)
        y_out = torch.full((len(ys), max_len), self.pad_id, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x_out[i, :x.shape[0]] = x
            y_out[i, :y.shape[0]] = y
        return x_out, y_out, list(song_ids), list(is_lasts)


def make_loaders(token_dir, max_seq, batch_size, pad_id=0, num_workers=0, val_frac=0.02, min_tokens=256):
    collator      = StatefulCollator(pad_id)
    train_ds      = SongDataset(token_dir, max_seq, min_tokens, "train", val_frac, pad_id)
    val_ds        = SongDataset(token_dir, max_seq, min_tokens, "val",   val_frac, pad_id)
    train_sampler = SongOrderSampler(train_ds, shuffle=True)
    val_sampler   = SongOrderSampler(val_ds,   shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
        collate_fn=collator, num_workers=num_workers, pin_memory=True, persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler,
        collate_fn=collator, num_workers=num_workers, pin_memory=True, persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None)
    return train_loader, val_loader
