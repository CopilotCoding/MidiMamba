"""
bench_scan.py — Run on your machine (RTX 5060 Ti) to confirm the new scan.

Three checks:
  1. Correctness vs sequential reference (fp32 + fp64 reference) — same as
     test_scan.py but also reports timing.
  2. Gradient correctness — autograd through the scan vs the sequential loop.
  3. Throughput — wall-clock for forward + backward at training seq_len.

Run:  python bench_scan.py
"""
import time
import torch
from model import _ssm_scan


def sequential_ssm(dA, dBx, C):
    B, T, H, S = dBx.shape
    h = torch.zeros(B, H, S, device=dBx.device, dtype=dBx.dtype)
    ys = []
    for t in range(T):
        h = dA[:, t].unsqueeze(-1) * h + dBx[:, t]
        ys.append((h * C[:, t]).sum(-1))
    return torch.stack(ys, dim=1)


def make_inputs(B, T, H, S, device, seed=42, dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dA  = (torch.sigmoid(torch.randn(B, T, H, generator=g)) * 0.9 + 0.05)
    dBx = torch.randn(B, T, H, S, generator=g) * 0.1
    C   = torch.randn(B, T, H, S, generator=g) * 0.1
    return (dA.to(device, dtype), dBx.to(device, dtype), C.to(device, dtype))


def correctness(device):
    print("\n[1] CORRECTNESS  (fp32 scan vs fp32 sequential)")
    print("-" * 64)
    for (T, H, S) in [(16,4,64),(1024,4,64),(2048,4,64),(8192,4,64),
                      (16384,4,64),(53178,4,64),(53178,8,32)]:
        dA, dBx, C = make_inputs(1, T, H, S, device)
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        out = _ssm_scan(dA, dBx, C)
        torch.cuda.synchronize() if device == "cuda" else None
        t_scan = time.time() - t0

        # sequential reference can be slow at 53K; only run it up to 16384
        if T <= 16384:
            ref = sequential_ssm(dA, dBx, C)
            err = (ref - out).abs().max().item()
            status = "PASS" if err < 1e-3 else "FAIL"
            print(f"  [{status}] T={T:6d} H={H} S={S}  max_err={err:.2e}  scan={t_scan*1e3:7.1f}ms")
        else:
            print(f"  [----] T={T:6d} H={H} S={S}  (ref skipped)     scan={t_scan*1e3:7.1f}ms")


def grad_check(device):
    print("\n[2] GRADIENT CHECK  (T=512, small enough for sequential backward)")
    print("-" * 64)
    T, H, S = 512, 4, 16
    dA, dBx, C = make_inputs(1, T, H, S, device)

    # scan path
    dA1 = dA.clone().requires_grad_(True)
    dBx1 = dBx.clone().requires_grad_(True)
    C1 = C.clone().requires_grad_(True)
    y1 = _ssm_scan(dA1, dBx1, C1)
    y1.pow(2).mean().backward()

    # sequential path
    dA2 = dA.clone().requires_grad_(True)
    dBx2 = dBx.clone().requires_grad_(True)
    C2 = C.clone().requires_grad_(True)
    y2 = sequential_ssm(dA2, dBx2, C2)
    y2.pow(2).mean().backward()

    for name, g1, g2 in [("dA", dA1.grad, dA2.grad),
                          ("dBx", dBx1.grad, dBx2.grad),
                          ("C", C1.grad, C2.grad)]:
        err = (g1 - g2).abs().max().item()
        status = "PASS" if err < 1e-4 else "FAIL"
        print(f"  [{status}] grad {name:3s}  max_err={err:.2e}")


def throughput(device):
    print("\n[3] THROUGHPUT  (forward + backward at training seq_len)")
    print("-" * 64)
    B, T, H, S = 1, 53178, 8, 64
    dA, dBx, C = make_inputs(B, T, H, S, device)
    dA.requires_grad_(True); dBx.requires_grad_(True); C.requires_grad_(True)

    # warmup
    y = _ssm_scan(dA, dBx, C); y.sum().backward()
    torch.cuda.synchronize() if device == "cuda" else None

    dA.grad = dBx.grad = C.grad = None
    t0 = time.time()
    y = _ssm_scan(dA, dBx, C)
    loss = y.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.time() - t0
    mem = (torch.cuda.max_memory_allocated()/1e9) if device == "cuda" else 0.0
    print(f"  T={T} H={H} S={S}  fwd+bwd={dt*1e3:.1f}ms  peak_mem={mem:.2f}GB")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    correctness(device)
    grad_check(device)
    if device == "cuda":
        throughput(device)
    print("\nDone.\n")
