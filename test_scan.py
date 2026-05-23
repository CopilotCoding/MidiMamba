import torch
import sys
sys.path.insert(0, '.')
from model import _ssm_scan

def reference(dA, dBx, C, h0):
    h = h0.double()
    ys = []
    for t in range(dA.shape[1]):
        h = dA[:, t].double() * h + dBx[:, t].double()
        ys.append((h * C[:, t].double()).float())
    return torch.stack(ys, dim=1)

def test(T, F, desc=""):
    torch.manual_seed(0)
    B = 1
    dA  = torch.rand(B, T, F) * 0.6 + 0.2
    dBx = torch.randn(B, T, F) * 0.01
    C   = torch.randn(B, T, F) * 0.01
    h0  = torch.zeros(B, F)
    ref     = reference(dA, dBx, C, h0)
    out, _  = _ssm_scan(dA, dBx, C, h0)
    err = (ref - out).abs().max().item()
    ok  = "PASS" if err < 2e-3 else "FAIL"
    print(f"  [{ok}] T={T:6d} F={F:4d}  max_err={err:.2e}  {desc}")
    return err < 2e-3

print("\nSSM Scan Correctness Tests")
print("=" * 60)
p = True
p &= test(16,    4,    "tiny")
p &= test(256,   4,    "small")
p &= test(1024,  4,    "medium")
p &= test(4096,  4,    "seq_len")
p &= test(4096,  1408, "training shape")
print("=" * 60)
print(f"  Overall: {'ALL PASS' if p else 'FAILURES DETECTED'}")
