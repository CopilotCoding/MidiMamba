"""
generate.py

Generate MIDI from a trained MidiMamba checkpoint.
Supports full conditioning config — specify any combination of:
  tempo, duration, key, time signature, pitch range, density, polyphony, etc.

Usage:
    # Unconditioned (model chooses everything)
    python generate.py run/checkpoints/best output.mid

    # Conditioned
    python generate.py run/checkpoints/best output.mid --tempo 120 --key_root 0 --key_minor 0

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
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

import tokenizer as tok
from model import MambaLayer, MidiMamba, ModelConfig
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
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    sorted_probs[(cumsum - sorted_probs) > top_p] = 0.0
    sorted_probs /= sorted_probs.sum()
    chosen = np.random.choice(len(sorted_probs), p=sorted_probs)
    return int(sorted_idx[chosen])


def _rep_penalty(logits: torch.Tensor, tokens: list[int], penalty: float, window: int) -> torch.Tensor:
    """Penalize recently used pitch tokens to discourage melodic loops."""
    recent = set(t for t in tokens[-window:] if tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88)
    if recent:
        ids = torch.tensor(list(recent), dtype=torch.long, device=logits.device)
        logits[ids] /= penalty
    return logits


def _bar_repetition_score(tokens: list[int], n_bars: int = 4) -> float:
    """
    Score how repetitive the last n_bars are vs the preceding n_bars.
    Returns 0..1. High = repetitive.
    """
    bar_starts = [i for i, t in enumerate(tokens) if t == tok.BAR]
    if len(bar_starts) < n_bars * 2 + 1:
        return 0.0
    recent_start = bar_starts[-n_bars]
    prev_start = bar_starts[-n_bars * 2]
    prev_end = bar_starts[-n_bars]
    recent = tokens[recent_start:]
    prev = tokens[prev_start:prev_end]
    clen = min(len(recent), len(prev), 64)
    if clen < 8:
        return 0.0
    matches = sum(a == b for a, b in zip(recent[:clen], prev[:clen]))
    return matches / clen


# --------------------------------------------------------------------------- #
#  Build conditioning prefix from user args
# --------------------------------------------------------------------------- #

def build_cond_prefix(args, vocab_config: dict) -> list[int]:
    """
    Build conditioning token prefix from CLI args.
    Any unspecified dimension uses a UNKNOWN/middle bucket.
    """
    bucket_config = vocab_config["bucket_config"]
    tokens = []

    # Map CLI arg names to feature field names
    arg_map = {
        "tempo": "tempo",
        "duration_sec": "duration_sec",
        "n_bars": "n_bars",
        "pitch_min": "pitch_min",
        "pitch_max": "pitch_max",
        "pitch_range": "pitch_range",
        "note_density": "note_density",
        "avg_dur_sec": "avg_dur_sec",
        "polyphony": "polyphony",
        "rest_density": "rest_density",
        "total_notes": "total_notes",
        "ioi_cv": "ioi_cv",
        "pitch_variety": "pitch_variety",
        "interval_diversity": "interval_diversity",
        "file_size": "file_size",
        "ts_num": "ts_num",
        "ts_den": "ts_den",
        "key_root": "key_root",
        "key_minor": "key_minor",
        "key_detected": "key_detected",
        "n_tracks": "n_tracks",
        "n_channels": "n_channels",
        "midi_format": "midi_format",
        "has_drums": "has_drums",
    }

    for field, cfg in bucket_config.items():
        offset = cfg["token_offset"]
        arg_name = arg_map.get(field)
        val = getattr(args, arg_name, None) if arg_name else None

        if val is None:
            # Use middle bucket as neutral/unconditioned
            bucket = cfg["n_buckets"] // 2
        elif cfg["type"] == "continuous":
            bucket = value_to_bucket(float(val), cfg["boundaries"])
        else:
            try:
                bucket = cfg["values"].index(int(val))
            except ValueError:
                bucket = 0

        tokens.append(offset + bucket)

    return tokens


# --------------------------------------------------------------------------- #
#  Generate
# --------------------------------------------------------------------------- #

def generate(
    model: MidiMamba,
    cond_prefix: list[int],
    max_tokens: int = 50000,
    temperature: float = 0.92,
    top_p: float = 0.93,
    min_tokens: int = 4000,
    rep_penalty: float = 1.08,
    rep_window: int = 256,
    rep_boost_threshold: float = 0.75,
    rep_boost: float = 0.15,
    seed: int | None = None,
    silent: bool = False,
) -> list[int]:
    """
    Generate using Mamba's recurrent inference mode — single token per step,
    O(1) memory regardless of sequence length. No context window limit.

    The SSM state carries full musical history in a fixed-size vector.
    Anti-repetition:
    1. Pitch repetition penalty on logits
    2. Dynamic temperature boost when bar similarity detected
    3. EOS suppression until min_tokens
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = next(model.parameters()).device
    model.eval()

    tokens = list(cond_prefix) + [tok.BOS]
    states = model.init_states(1, device)

    with torch.no_grad():
        # Prefill: run full prefix through training-mode forward to build SSM state
        # Then extract per-layer states for recurrent stepping
        # Simpler: step through prefix tokens one at a time
        for t in tokens[:-1]:
            idx = torch.tensor([[t]], dtype=torch.long, device=device)
            _, states = model.step(idx, states)

        # Get logits for last prefix token
        idx = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
        logits, states = model.step(idx, states)

        n_bars = 0
        current_tempo = 120.0
        gen_start = time.time()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("• [dim]{task.fields[tps]:.0f} tok/s[/dim]"),
            TextColumn("• [green]~{task.fields[dur]}[/green]"),
            TextColumn("• [yellow]bars:{task.fields[bars]}[/yellow]"),
            disable=silent,
        )

        with progress:
            task = progress.add_task("", total=max_tokens, tps=0.0, dur="0:00", bars=0)

            gen_tokens = 0
            while gen_tokens < max_tokens:
                next_logits = logits[0, -1].clone()

                # Always mask conditioning tokens — model should never generate these
                next_logits[:tok.COND_END] = float("-inf")

                # Force first token to be BAR
                if gen_tokens == 0:
                    music_mask = torch.full_like(next_logits, float("-inf"))
                    music_mask[tok.BAR] = 0.0
                    next_logits = next_logits + music_mask

                # EOS suppression
                if len(tokens) < min_tokens + len(cond_prefix):
                    next_logits[tok.EOS] = float("-inf")

                # Pitch rep penalty
                if rep_penalty != 1.0:
                    next_logits = _rep_penalty(next_logits, tokens, rep_penalty, rep_window)

                # Dynamic temperature boost
                rep_score = _bar_repetition_score(tokens)
                temp = temperature
                if rep_score >= rep_boost_threshold:
                    temp = min(temperature + rep_boost * (rep_score - rep_boost_threshold) /
                               (1.0 - rep_boost_threshold + 1e-6), 1.4)

                next_id = _top_p_sample(next_logits, temp, top_p)
                tokens.append(next_id)
                gen_tokens += 1

                if next_id == tok.BAR:
                    n_bars += 1
                elif tok.TEMPO_OFFSET <= next_id < tok.TEMPO_OFFSET + 17:
                    current_tempo = tok.TEMPO_TOKENS[next_id - tok.TEMPO_OFFSET]
                elif next_id == tok.EOS and gen_tokens >= min_tokens:
                    break
                # EOS before min_tokens is ignored — model learned to end pieces early

                # Single recurrent step — O(1) memory
                idx = torch.tensor([[next_id]], dtype=torch.long, device=device)
                logits, states = model.step(idx, states)

                if gen_tokens % 20 == 0:
                    elapsed = time.time() - gen_start
                    tps = gen_tokens / max(elapsed, 1e-6)
                    bars_per_tok = n_bars / max(gen_tokens, 1)
                    est_secs = bars_per_tok * max_tokens * (60.0 / max(current_tempo, 1)) * 4
                    dur_str = f"{int(est_secs)//60}:{int(est_secs)%60:02d}"
                    progress.update(task, completed=gen_tokens, tps=tps, dur=dur_str, bars=n_bars)

    pitches = {t for t in tokens if tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88}
    console.print(f"\nGenerated {len(tokens)} tokens | {n_bars} bars | {len(pitches)} distinct pitches")
    return tokens


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate MIDI from MidiMamba checkpoint")
    parser.add_argument("checkpoint", help="Checkpoint directory (contains model.pt + meta.json)")
    parser.add_argument("output", help="Output .mid file path")

    # Generation params
    parser.add_argument("--max_tokens", type=int, default=50000)
    parser.add_argument("--min_tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.92)
    parser.add_argument("--top_p", type=float, default=0.93)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rep_penalty", type=float, default=1.08)
    parser.add_argument("--rep_window", type=int, default=256)

    # Conditioning (all optional — omit for unconditioned)
    parser.add_argument("--tempo", type=float, default=None, help="BPM e.g. 120")
    parser.add_argument("--duration_sec", type=float, default=None, help="Target duration in seconds")
    parser.add_argument("--n_bars", type=int, default=None, help="Target bar count")
    parser.add_argument("--pitch_min", type=int, default=None, help="Lowest pitch 21-108")
    parser.add_argument("--pitch_max", type=int, default=None, help="Highest pitch 21-108")
    parser.add_argument("--pitch_range", type=int, default=None, help="Pitch span in semitones")
    parser.add_argument("--note_density", type=float, default=None, help="Notes per bar")
    parser.add_argument("--avg_dur_sec", type=float, default=None, help="Avg note duration seconds")
    parser.add_argument("--polyphony", type=float, default=None, help="Avg simultaneous notes")
    parser.add_argument("--rest_density", type=float, default=None, help="Fraction of time silent 0-1")
    parser.add_argument("--total_notes", type=int, default=None)
    parser.add_argument("--ioi_cv", type=float, default=None, help="Rhythmic irregularity 0-3")
    parser.add_argument("--pitch_variety", type=float, default=None, help="Pitch variety 0-1")
    parser.add_argument("--interval_diversity", type=float, default=None)
    parser.add_argument("--file_size", type=int, default=None)
    parser.add_argument("--ts_num", type=int, default=None, help="Time sig numerator e.g. 4")
    parser.add_argument("--ts_den", type=int, default=None, help="Time sig denominator e.g. 4")
    parser.add_argument("--key_root", type=int, default=None, help="Key root 0=C 1=C# ... 11=B")
    parser.add_argument("--key_minor", type=int, default=None, help="0=major 1=minor")
    parser.add_argument("--key_detected", type=int, default=None, help="1=key signature present 0=unknown")
    parser.add_argument("--n_tracks", type=int, default=None, help="Number of instrument tracks")
    parser.add_argument("--n_channels", type=int, default=None)
    parser.add_argument("--midi_format", type=int, default=None, help="0 or 1")
    parser.add_argument("--has_drums", type=int, default=None, help="1=drums present 0=no drums")

    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    meta_path = ckpt / "meta.json"
    if not meta_path.exists():
        console.print(f"[red]meta.json not found in {ckpt}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    vocab_path = Path(meta["vocab_config"])
    tok.init(vocab_path)
    console.print(f"Vocab: {tok.VOCAB_SIZE} tokens")

    with open(vocab_path) as f:
        vocab_config = json.load(f)

    cfg_dict = meta["model_cfg"]
    cfg = ModelConfig(**cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: [bold]{device}[/bold]")

    model = MidiMamba(cfg).to(device)
    sd = torch.load(ckpt / "model.pt", map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    console.print(f"Model: [bold]{model.param_count_str()}[/bold]")

    # Build conditioning prefix
    cond_prefix = build_cond_prefix(args, vocab_config)
    console.print(f"Conditioning prefix: {len(cond_prefix)} tokens")

    # Log what's conditioned
    cond_args = {k: v for k, v in vars(args).items()
                 if v is not None and k in vocab_config["bucket_config"]}
    if cond_args:
        console.print(f"[blue]Conditioning on: {cond_args}[/blue]")
    else:
        console.print("[dim]No conditioning specified — using neutral mid-bucket defaults[/dim]")

    token_ids = generate(
        model,
        cond_prefix=cond_prefix,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        rep_penalty=args.rep_penalty,
        rep_window=args.rep_window,
    )

    pm = tok.decode(token_ids)
    out_path = args.output
    pm.write(out_path)
    console.print(f"[green]Saved → {out_path}[/green]")


if __name__ == "__main__":
    main()
