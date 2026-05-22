"""
generate.py — MidiGen3

Generate MIDI from a trained MidiMamba checkpoint.
Recurrent inference: O(1) memory per step, no context window limit.

Usage:
    # Unconditioned
    python generate.py run/checkpoints/best output.mid

    # Conditioned
    python generate.py run/checkpoints/best output.mid --tempo 120 --key_root 0 --key_minor 0

    # Batch: 4 variations in parallel
    python generate.py run/checkpoints/best output.mid --batch 4

    # Long piece
    python generate.py run/checkpoints/best output.mid --max_tokens 80000 --temperature 0.92
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                            SpinnerColumn, TextColumn, TimeRemainingColumn)

import tokenizer as tok
from model import MidiMamba, ModelConfig
from scan_dataset import value_to_bucket, CONTINUOUS_FIELDS, CATEGORICAL_FIELDS

console = Console()


# --------------------------------------------------------------------------- #
#  Sampling
# --------------------------------------------------------------------------- #

def _top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    l = logits.float().cpu().numpy()
    l = l / max(temperature, 1e-6)
    l -= l.max()
    probs = np.exp(l)
    probs /= probs.sum()
    sorted_idx   = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum       = np.cumsum(sorted_probs)
    sorted_probs[(cumsum - sorted_probs) > top_p] = 0.0
    sorted_probs /= sorted_probs.sum()
    return int(sorted_idx[np.random.choice(len(sorted_probs), p=sorted_probs)])


def _rep_penalty(logits: torch.Tensor, tokens: list, penalty: float, window: int) -> torch.Tensor:
    """Divide logits of recently-used pitch tokens."""
    recent = {t for t in tokens[-window:] if tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88}
    if recent:
        ids = torch.tensor(list(recent), dtype=torch.long, device=logits.device)
        logits[ids] /= penalty
    return logits


def _bar_rep_score(tokens: list, n_bars: int = 4) -> float:
    """Return 0–1 similarity between last n_bars and preceding n_bars."""
    bar_starts = [i for i, t in enumerate(tokens) if t == tok.BAR]
    if len(bar_starts) < n_bars * 2 + 1:
        return 0.0
    recent = tokens[bar_starts[-n_bars]:]
    prev   = tokens[bar_starts[-n_bars * 2] : bar_starts[-n_bars]]
    clen   = min(len(recent), len(prev), 64)
    if clen < 8: return 0.0
    return sum(a == b for a, b in zip(recent[:clen], prev[:clen])) / clen


# --------------------------------------------------------------------------- #
#  Conditioning prefix
# --------------------------------------------------------------------------- #

def build_cond_prefix(args, vocab_config: dict) -> list:
    bucket_config = vocab_config["bucket_config"]
    arg_map = {
        "tempo": "tempo", "duration_sec": "duration_sec", "n_bars": "n_bars",
        "pitch_min": "pitch_min", "pitch_max": "pitch_max", "pitch_range": "pitch_range",
        "note_density": "note_density", "avg_dur_sec": "avg_dur_sec",
        "polyphony": "polyphony", "rest_density": "rest_density",
        "total_notes": "total_notes", "ioi_cv": "ioi_cv",
        "pitch_variety": "pitch_variety", "interval_diversity": "interval_diversity",
        "file_size": "file_size", "ts_num": "ts_num", "ts_den": "ts_den",
        "key_root": "key_root", "key_minor": "key_minor", "key_detected": "key_detected",
        "n_tracks": "n_tracks", "n_channels": "n_channels",
        "midi_format": "midi_format", "has_drums": "has_drums",
    }
    tokens = []
    for field, cfg in bucket_config.items():
        offset   = cfg["token_offset"]
        arg_name = arg_map.get(field)
        val      = getattr(args, arg_name, None) if arg_name else None
        if val is None:
            bucket = cfg["n_buckets"] // 2   # neutral/mid-bucket for unconditioned dims
        elif cfg["type"] == "continuous":
            bucket = value_to_bucket(float(val), cfg["boundaries"])
        else:
            try:    bucket = cfg["values"].index(int(val))
            except ValueError: bucket = 0
        tokens.append(offset + bucket)
    return tokens


# --------------------------------------------------------------------------- #
#  Generate
# --------------------------------------------------------------------------- #

def generate(
    model:               MidiMamba,
    cond_prefix:         list,
    max_tokens:          int   = 50000,
    temperature:         float = 0.92,
    top_p:               float = 0.93,
    min_tokens:          int   = 4000,
    rep_penalty:         float = 1.08,
    rep_window:          int   = 256,
    rep_boost_threshold: float = 0.75,
    rep_boost:           float = 0.15,
    seed = None,
    silent:              bool  = False,
    batch_size:          int   = 1,
) -> list:
    """
    Generate using Mamba's recurrent step — O(1) memory, no context window.

    batch_size > 1: generates N independent sequences in parallel on GPU,
    returns list of token lists (all from same random seed + individual noise).
    Single-sequence mode (batch_size=1) returns a flat list for compatibility.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = next(model.parameters()).device
    model.eval()

    B      = batch_size
    prefix = list(cond_prefix) + [tok.BOS]
    states = model.init_inference_states(B, device)

    all_tokens = [list(prefix) for _ in range(B)]

    with torch.no_grad():
        # Prefill: step through prefix tokens
        for t in prefix[:-1]:
            idx = torch.tensor([[t]] * B, dtype=torch.long, device=device)
            _, states = model.step(idx, states)

        idx    = torch.tensor([[prefix[-1]]] * B, dtype=torch.long, device=device)
        logits, states = model.step(idx, states)   # (B, 1, V)

        n_bars       = [0] * B
        active       = [True] * B   # tracks which sequences haven't hit EOS yet
        gen_tokens   = 0
        gen_start    = time.time()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating"),
            BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
            TextColumn("• [dim]{task.fields[tps]:.0f} tok/s[/dim]"),
            TextColumn("• [yellow]bars:{task.fields[bars]}[/yellow]"),
            disable=silent,
        )

        with progress:
            task = progress.add_task("", total=max_tokens, tps=0.0, bars=0)

            while gen_tokens < max_tokens and any(active):
                next_ids = []
                for b in range(B):
                    if not active[b]:
                        next_ids.append(tok.PAD)
                        continue

                    l = logits[b, -1].clone()
                    l[:tok.COND_END] = float("-inf")   # never generate conditioning tokens

                    if gen_tokens == 0:
                        mask      = torch.full_like(l, float("-inf"))
                        mask[tok.BAR] = 0.0
                        l         = l + mask

                    if len(all_tokens[b]) < min_tokens + len(prefix):
                        l[tok.EOS] = float("-inf")

                    if rep_penalty != 1.0:
                        l = _rep_penalty(l, all_tokens[b], rep_penalty, rep_window)

                    rep_score = _bar_rep_score(all_tokens[b])
                    temp      = temperature
                    if rep_score >= rep_boost_threshold:
                        temp = min(temperature + rep_boost * (rep_score - rep_boost_threshold) /
                                   (1.0 - rep_boost_threshold + 1e-6), 1.4)

                    nid = _top_p_sample(l, temp, top_p)
                    next_ids.append(nid)
                    all_tokens[b].append(nid)

                    if nid == tok.BAR:
                        n_bars[b] += 1
                    elif nid == tok.EOS and gen_tokens >= min_tokens:
                        active[b] = False

                gen_tokens += 1
                idx    = torch.tensor([[nid] for nid in next_ids], dtype=torch.long, device=device)
                logits, states = model.step(idx, states)

                if gen_tokens % 20 == 0:
                    elapsed = time.time() - gen_start
                    tps     = gen_tokens / max(elapsed, 1e-6)
                    progress.update(task, completed=gen_tokens, tps=tps,
                                    bars=max(n_bars))

    for b in range(B):
        pitches = {t for t in all_tokens[b] if tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88}
        console.print(f"  [{b}] {len(all_tokens[b])} tokens | {n_bars[b]} bars | "
                      f"{len(pitches)} distinct pitches")

    if B == 1:
        return all_tokens[0]   # backward-compatible flat list
    return all_tokens


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output",     help="Output .mid path (or stem for --batch > 1)")
    parser.add_argument("--batch",    type=int,   default=1,     help="Generate N variations in parallel")
    parser.add_argument("--max_tokens", type=int, default=50000)
    parser.add_argument("--min_tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.92)
    parser.add_argument("--top_p",    type=float, default=0.93)
    parser.add_argument("--seed",     type=int,   default=None)
    parser.add_argument("--rep_penalty", type=float, default=1.08)
    parser.add_argument("--rep_window",  type=int,   default=256)

    # Conditioning (all optional)
    for name, typ, hlp in [
        ("tempo",             float, "BPM"),
        ("duration_sec",      float, "Target duration seconds"),
        ("n_bars",            int,   "Target bar count"),
        ("pitch_min",         int,   "Lowest pitch 21-108"),
        ("pitch_max",         int,   "Highest pitch 21-108"),
        ("pitch_range",       int,   "Pitch span semitones"),
        ("note_density",      float, "Notes per bar"),
        ("avg_dur_sec",       float, "Avg note duration"),
        ("polyphony",         float, "Avg simultaneous notes"),
        ("rest_density",      float, "Fraction of time silent 0-1"),
        ("total_notes",       int,   ""),
        ("ioi_cv",            float, "Rhythmic irregularity 0-3"),
        ("pitch_variety",     float, "0-1"),
        ("interval_diversity",float, ""),
        ("file_size",         int,   ""),
        ("ts_num",            int,   "Time sig numerator"),
        ("ts_den",            int,   "Time sig denominator"),
        ("key_root",          int,   "0=C .. 11=B"),
        ("key_minor",         int,   "0=major 1=minor"),
        ("key_detected",      int,   "1=key sig present"),
        ("n_tracks",          int,   ""),
        ("n_channels",        int,   ""),
        ("midi_format",       int,   "0 or 1"),
        ("has_drums",         int,   "1=drums 0=no drums"),
    ]:
        parser.add_argument(f"--{name}", type=typ, default=None, help=hlp)

    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    with open(ckpt / "meta.json") as f:
        meta = json.load(f)

    vocab_path = Path(meta["vocab_config"])
    tok.init(vocab_path)
    console.print(f"Vocab: {tok.VOCAB_SIZE} tokens")

    with open(vocab_path) as f:
        vocab_config = json.load(f)

    cfg   = ModelConfig(**meta["model_cfg"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: [bold]{device}[/bold]")

    model = MidiMamba(cfg).to(device)
    sd    = torch.load(ckpt / "model.pt", map_location=device, weights_only=True)
    sd    = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    console.print(f"Model: [bold]{model.param_count_str()}[/bold]")

    cond_prefix = build_cond_prefix(args, vocab_config)
    console.print(f"Conditioning prefix: {len(cond_prefix)} tokens")

    results = generate(
        model,
        cond_prefix  = cond_prefix,
        max_tokens   = args.max_tokens,
        min_tokens   = args.min_tokens,
        temperature  = args.temperature,
        top_p        = args.top_p,
        seed         = args.seed,
        rep_penalty  = args.rep_penalty,
        rep_window   = args.rep_window,
        batch_size   = args.batch,
    )

    out_path = Path(args.output)
    if args.batch == 1:
        tok.decode_to_file(results, out_path)
        console.print(f"[green]Saved → {out_path}[/green]")
    else:
        stem   = out_path.stem
        suffix = out_path.suffix or ".mid"
        for i, token_ids in enumerate(results):
            p = out_path.parent / f"{stem}_{i:02d}{suffix}"
            tok.decode_to_file(token_ids, p)
            console.print(f"[green]Saved → {p}[/green]")


if __name__ == "__main__":
    main()
