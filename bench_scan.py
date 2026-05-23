import time
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

def make(B, T, F, device):
    torch.manual_seed(0)
    dA  = (torch.rand(B, T, F) * 0.6 + 0.2).to(device)
    dBx = (torch.randn(B, T, F) * 0.01).to(device)
    C   = (torch.randn(B, T, F) * 0.01).to(device)
    h0  = torch.zeros(B, F, device=device)
    return dA, dBx, C, h0

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

print("[1] CORRECTNESS")
print("-" * 56)
for T, F in [(16,4),(256,4),(1024,4),(4096,4),(4096,1408)]:
    dA, dBx, C, h0 = make(1, T, F, device)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    out, _ = _ssm_scan(dA, dBx, C, h0)
    if device == "cuda": torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000
    if F <= 64:
        ref = reference(dA.cpu(), dBx.cpu(), C.cpu(), h0.cpu()).to(device)
        err = (ref - out).abs().max().item()
        ok  = "PASS" if err < 2e-3 else "FAIL"
        print(f"  [{ok}] T={T:5d} F={F:4d}  err={err:.2e}  {ms:.1f}ms")
    else:
        print(f"  [----] T={T:5d} F={F:4d}  (ref skipped)  {ms:.1f}ms")

print("\n[2] GRADIENTS")
print("-" * 56)
dA, dBx, C, h0 = make(1, 512, 16, device)
for name, inp in [("dA", dA), ("dBx", dBx), ("C", C)]:
    x = inp.clone().requires_grad_(True)
    args = [dA.clone(), dBx.clone(), C.clone(), h0]
    idx  = ["dA","dBx","C"].index(name)
    args[idx] = x
    y, _ = _ssm_scan(*args)
    y.pow(2).mean().backward()
    g1 = x.grad.clone()
    x2 = inp.clone().requires_grad_(True)
    args2 = [dA.cpu().clone(), dBx.cpu().clone(), C.cpu().clone(), h0.cpu()]
    args2[idx] = x2.cpu().requires_grad_(True) if device=="cuda" else x2
    h = h0.cpu()
    a2 = dA.cpu().clone(); b2 = dBx.cpu().clone(); c2 = C.cpu().clone()
    vs = [a2,b2,c2]; vs[idx] = vs[idx].requires_grad_(True)
    hh = h0.cpu().clone()
    hc = hh.clone()
    ys2 = []
    for t in range(512):
        hc = vs[0][:,t]*hc + vs[1][:,t]
        ys2.append(hc*vs[2][:,t])
    torch.stack(ys2,dim=1).pow(2).mean().backward()
    g2 = vs[idx].grad
    if device == "cuda": g2 = g2.cuda()
    err = (g1 - g2).abs().max().item()
    print(f"  [{'PASS' if err<1e-4 else 'FAIL'}] grad {name:3s}  err={err:.2e}")

if device == "cuda":
    print("\n[3] THROUGHPUT")
    print("-" * 56)
    dA, dBx, C, h0 = make(1, 4096, 1408, device)
    dA.requires_grad_(True); dBx.requires_grad_(True); C.requires_grad_(True)
    y,_ = _ssm_scan(dA,dBx,C,h0); y.sum().backward()
    torch.cuda.synchronize()
    dA.grad=dBx.grad=C.grad=None
    t0=time.time()
    y,_=_ssm_scan(dA,dBx,C,h0); y.pow(2).mean().backward()
    torch.cuda.synchronize()
    dt=time.time()-t0
    mem=torch.cuda.max_memory_allocated()/1e9
    print(f"  T=4096 F=1408  fwd+bwd={dt*1000:.1f}ms  peak_mem={mem:.2f}GB")

print("\nDone.")
