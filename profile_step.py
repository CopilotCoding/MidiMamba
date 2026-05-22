"""
profile_step.py — Find out where training time ACTUALLY goes.

Runs a handful of forward+backward steps at your real config under
torch.profiler and prints the operations sorted by total CUDA time.
This tells us whether the SSM scan is the bottleneck or whether it's
something else (SwiGLU, checkpoint recompute, conv, etc).

Run:
    python profile_step.py --seq_len 53178
    python profile_step.py --seq_len 8192
Compare the two: whatever grows superlinearly between them is the culprit.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

import tokenizer as tok
from model import MidiMamba, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=53178)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=16)
    ap.add_argument("--grad_checkpoint", action="store_true", default=True)
    ap.add_argument("--no_checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--stats", default="stats_out")
    args = ap.parse_args()

    # tokenizer just for VOCAB_SIZE; fall back if stats missing
    try:
        import json, os
        vc = os.path.join(args.stats, "vocab_config.json")
        tok.init(vc)
        vocab = tok.VOCAB_SIZE
    except Exception:
        vocab = 442

    device = "cuda"
    cfg = ModelConfig(
        vocab_size=vocab, d_model=args.d_model, n_layers=args.n_layers,
        d_state=64, max_seq=args.seq_len,
        grad_ckpt=args.grad_checkpoint,
    )
    model = MidiMamba(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randint(0, vocab, (1, args.seq_len), device=device)
    y = torch.randint(0, vocab, (1, args.seq_len), device=device)

    def one_step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        return loss.item()

    # warmup (kernels, autotune, checkpoint graph)
    for _ in range(2):
        one_step()
    torch.cuda.synchronize()

    print(f"\nProfiling {args.steps} steps  seq_len={args.seq_len}  "
          f"checkpoint={args.grad_checkpoint}\n" + "="*70)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(args.steps):
            one_step()
        torch.cuda.synchronize()

    # Top ops by CUDA total time
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25))

    # Also crude wall-clock tok/s
    import time
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(args.steps):
        one_step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    toks = args.steps * args.seq_len
    print(f"\nWall: {dt:.2f}s for {args.steps} steps  "
          f"=> {toks/dt:,.0f} tok/s  ({dt/args.steps:.2f}s/step)")


if __name__ == "__main__":
    main()
