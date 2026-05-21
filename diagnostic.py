"""
diagnostic.py — MidiMamba generation quality diagnostic

Tests conditioning response, token distribution, and SSM state coherence.
Run from the midigen2 directory:
    python diagnostic.py [--checkpoint run/checkpoints/best]
"""

import argparse
import json
import time
from pathlib import Path
from collections import Counter

import torch

import tokenizer as tok
from model import MidiMamba, ModelConfig
from generate import generate, build_cond_prefix


def load_model(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    meta = json.load(open(ckpt_dir / "meta.json"))
    cfg = ModelConfig(**meta["model_cfg"])
    model = MidiMamba(cfg)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, meta, cfg


def token_breakdown(ids, label=""):
    total = len(ids)
    cond  = sum(1 for t in ids if t < tok.COND_END)
    bar   = sum(1 for t in ids if t == tok.BAR)
    pos   = sum(1 for t in ids if tok.POS_OFFSET   <= t < tok.POS_OFFSET   + 16)
    track = sum(1 for t in ids if tok.TRACK_OFFSET  <= t < tok.TRACK_OFFSET + 9)
    pitch = sum(1 for t in ids if tok.PITCH_OFFSET  <= t < tok.PITCH_OFFSET + 88)
    dur   = sum(1 for t in ids if tok.DUR_OFFSET    <= t < tok.DUR_OFFSET   + 16)
    vel   = sum(1 for t in ids if tok.VEL_OFFSET    <= t < tok.VEL_OFFSET   + 8)
    eos   = sum(1 for t in ids if t == tok.EOS)
    other = total - cond - bar - pos - track - pitch - dur - vel - eos

    pitches = sorted({t - tok.PITCH_OFFSET + 21 for t in ids
                      if tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88})
    pitch_range = (min(pitches), max(pitches)) if pitches else (0, 0)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total tokens : {total}")
    print(f"  Cond prefix  : {cond}  ({100*cond/max(total,1):.1f}%)")
    print(f"  BAR          : {bar}")
    print(f"  POS          : {pos}  ({100*pos/max(total,1):.1f}%)")
    print(f"  TRACK        : {track}  ({100*track/max(total,1):.1f}%)")
    print(f"  PITCH        : {pitch}  ({100*pitch/max(total,1):.1f}%)")
    print(f"  DUR          : {dur}  ({100*dur/max(total,1):.1f}%)")
    print(f"  VEL          : {vel}  ({100*vel/max(total,1):.1f}%)")
    print(f"  EOS          : {eos}")
    print(f"  Other        : {other}")
    print(f"  Distinct pitches : {len(pitches)}  range {pitch_range[0]}-{pitch_range[1]}")
    if bar > 0:
        print(f"  Notes/bar    : {pitch/bar:.1f}")
    note_pct = 100 * pitch / max(total - cond, 1)
    status = "OK" if pitch > 50 and bar > 3 else "WARN — low note density"
    print(f"  Note token % : {note_pct:.1f}%  [{status}]")
    return pitch, bar, len(pitches)


def test_conditioning_response(model, vocab_config, n_tokens=600):
    """Generate with opposite conditioning pairs and compare pitch distributions."""
    print("\n" + "="*60)
    print("  CONDITIONING RESPONSE TEST")
    print("="*60)

    class FakeArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __getattr__(self, name):
            return None

    tests = [
        ("High tempo (180 BPM)", dict(tempo=180)),
        ("Low tempo  (50 BPM)",  dict(tempo=50)),
        ("Major key",            dict(key_minor=0, key_detected=1)),
        ("Minor key",            dict(key_minor=1, key_detected=1)),
        ("No drums",             dict(has_drums=0)),
        ("With drums",           dict(has_drums=1)),
        ("High density",         dict(note_density=20)),
        ("Low density",          dict(note_density=1)),
    ]

    results = {}
    for label, kwargs in tests:
        args = FakeArgs(**kwargs)
        cond = build_cond_prefix(args, vocab_config)
        t0 = time.time()
        ids = generate(model, cond, max_tokens=n_tokens, min_tokens=n_tokens-50, silent=True, seed=42)
        elapsed = time.time() - t0
        pitches, bars, n_unique = token_breakdown(ids, label)
        results[label] = (pitches, bars, n_unique)
        print(f"  Generated in {elapsed:.1f}s")

    # Compare pairs
    print("\n" + "="*60)
    print("  PAIR COMPARISONS")
    print("="*60)
    pairs = [
        ("High tempo (180 BPM)", "Low tempo  (50 BPM)",  "note density"),
        ("Major key",            "Minor key",             "pitch distribution"),
        ("No drums",             "With drums",            "track usage"),
        ("High density",         "Low density",           "notes/bar"),
    ]
    for a, b, metric in pairs:
        pa, ba, ua = results[a]
        pb, bb, ub = results[b]
        diff_notes = abs(pa - pb)
        diff_bars  = abs(ba - bb)
        responding = diff_notes > 10 or diff_bars > 2
        status = "RESPONDING" if responding else "NOT RESPONDING"
        print(f"  {a} vs {b}")
        print(f"    notes: {pa} vs {pb}  bars: {ba} vs {bb}  [{status}]")


def test_state_coherence(model, vocab_config, n_tokens=1000):
    """Check if later tokens in a sequence are more structured than earlier ones."""
    print("\n" + "="*60)
    print("  SSM STATE COHERENCE TEST")
    print("="*60)

    args_obj = type("A", (), {"__getattr__": lambda s, n: None})()
    cond = build_cond_prefix(args_obj, vocab_config)
    ids = generate(model, cond, max_tokens=n_tokens, min_tokens=n_tokens-50, silent=True, seed=123)

    # Split into thirds and compare note density
    music_ids = ids[len(cond):]  # skip cond prefix
    third = len(music_ids) // 3
    parts = [music_ids[:third], music_ids[third:2*third], music_ids[2*third:]]
    labels = ["First third", "Middle third", "Final third"]

    print(f"  Analyzing {len(music_ids)} music tokens in thirds of {third} tokens each\n")
    for label, part in zip(labels, parts):
        pitch = sum(1 for t in part if tok.PITCH_OFFSET <= t < tok.PITCH_OFFSET + 88)
        bar   = sum(1 for t in part if t == tok.BAR)
        cond_leak = sum(1 for t in part if t < tok.COND_END)
        print(f"  {label}: {pitch} pitches, {bar} bars, {cond_leak} cond-leaks")

    cond_total = sum(1 for t in music_ids if t < tok.COND_END)
    if cond_total > 20:
        print(f"\n  WARNING: {cond_total} conditioning tokens in music section — masking may not be working")
    else:
        print(f"\n  Conditioning leak: {cond_total} tokens — OK")


def main():
    parser = argparse.ArgumentParser(description="MidiMamba generation diagnostic")
    parser.add_argument("--checkpoint", default="run/checkpoints/best",
                        help="Checkpoint directory")
    parser.add_argument("--vocab", default="stats_out/vocab_config.json")
    parser.add_argument("--tokens", type=int, default=600,
                        help="Tokens per test generation")
    args = parser.parse_args()

    tok.init(args.vocab)
    vocab_config = json.load(open(args.vocab))

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, meta, cfg = load_model(args.checkpoint)
    print(f"  Step: {meta['step']}")
    print(f"  Best val: {meta['best_val']:.4f}")
    print(f"  Config: d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} S={cfg.d_state}")
    print(f"  Params: {model.param_count_str()}")

    test_conditioning_response(model, vocab_config, n_tokens=args.tokens)
    test_state_coherence(model, vocab_config, n_tokens=args.tokens * 2)

    print("\n" + "="*60)
    print("  DIAGNOSTIC COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
