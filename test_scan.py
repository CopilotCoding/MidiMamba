"""
test_scan.py — verify _ssm_scan against sequential reference implementation

Run: python test_scan.py
"""
import torch
import sys
sys.path.insert(0, '.')
from model import _ssm_scan

def sequential_ssm(dA, dBx, C):
    """Dead-simple sequential loop — provably correct reference."""
    B, T, H, S = dBx.shape
    h = torch.zeros(B, H, S)
    ys = []
    for t in range(T):
        h = dA[:, t].unsqueeze(-1) * h + dBx[:, t]
        y = (h * C[:, t]).sum(-1)
        ys.append(y)
    return torch.stack(ys, dim=1)  # (B, T, H)

def test(T, H, S, B=1, desc=""):
    torch.manual_seed(42)
    dA  = torch.sigmoid(torch.randn(B, T, H)) * 0.9 + 0.05  # (0.05, 0.95)
    dBx = torch.randn(B, T, H, S) * 0.1
    C   = torch.randn(B, T, H, S) * 0.1

    ref = sequential_ssm(dA, dBx, C)
    out = _ssm_scan(dA, dBx, C)

    max_err = (ref - out).abs().max().item()
    mean_err = (ref - out).abs().mean().item()
    rel_err  = ((ref - out).abs() / (ref.abs() + 1e-8)).mean().item()

    status = "PASS" if max_err < 1e-3 else "FAIL"
    print(f"  [{status}] T={T:6d} H={H} S={S}  max_err={max_err:.2e}  mean_err={mean_err:.2e}  rel_err={rel_err:.2e}  {desc}")
    return max_err < 1e-3

print("\nSSM Scan Correctness Tests")
print("="*70)
all_pass = True
all_pass &= test(16,    4, 64, desc="tiny")
all_pass &= test(128,   4, 64, desc="small")
all_pass &= test(1024,  4, 64, desc="one segment")
all_pass &= test(2048,  4, 64, desc="two segments")
all_pass &= test(8192,  4, 64, desc="exactly one SEG")
all_pass &= test(16384, 4, 64, desc="two SEGs")
all_pass &= test(53178, 4, 64, desc="full training seq_len")
all_pass &= test(53178, 8, 32, desc="different H/S")

print("="*70)
print(f"  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
print()
