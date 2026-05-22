"""
model.py — MidiMamba

Correct Mamba1 SSM implementation.  Pure PyTorch, Windows-compatible,
bfloat16-safe.  No mamba-ssm / causal-conv1d dependency.

Key fix over midigen2:
  midigen2 computed x_ssm as a scalar per head, then broadcast back to
  d_inner — every position in a head was identical.  The SSM had n_heads
  information channels in a d_inner-wide model.  Wasted capacity, broken
  gradients at width.

  Here x_ssm IS x_inner (full d_inner vector).  A_log and dt_bias are
  (d_inner,) — one decay per feature dimension.  State is (d_inner, d_state).
  This is the original correct Mamba formulation.

SSM per feature f (d_inner total):
  h[t,f] = dA[t,f] * h[t-1,f] + dB[t,f,:] * x[t,f]
  y[t,f] = (h[t,f,:] * C[t,f,:]).sum()

where h[t,f] is a d_state vector, dA[t,f] scalar decay, dB/C d_state vectors.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    vocab_size:    int   = 0        # MUST be set from tokenizer — no default
    d_model:       int   = 512
    n_layers:      int   = 12
    d_state:       int   = 32
    d_conv:        int   = 4
    expand:        int   = 2
    d_ff_mult:     float = 2.667
    dropout:       float = 0.1
    max_seq:       int   = 32768
    grad_ckpt:     bool  = False

    def __post_init__(self):
        assert self.vocab_size > 0, (
            "ModelConfig.vocab_size must be set from tokenizer before model init. "
            "Call tokenizer.init() first, then pass tok.VOCAB_SIZE."
        )

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand


# --------------------------------------------------------------------------- #
#  Primitives
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 2.667):
        super().__init__()
        d_ff = (int(d_model * mult) + 63) // 64 * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --------------------------------------------------------------------------- #
#  SSM scan — chunked Hillis-Steele associative scan
#  h[t] = dA[t] * h[t-1] + dBx[t]
#  y[t] = (h[t] * C[t]).sum(-1)
#
#  Identical algorithm to midigen2 (which is numerically correct).
#  Shape change: F (d_inner) replaces H (n_heads).  Everything else identical.
# --------------------------------------------------------------------------- #

def _assoc_scan_chunk(a: torch.Tensor, b: torch.Tensor):
    """
    In-chunk Hillis-Steele inclusive scan.

    a, b : (B, L, F, S)  — F = d_inner, S = d_state
    Returns:
      h     : (B, L, F, S)  hidden states (carry-in = 0)
      A_cum : (B, L, F, S)  cumulative decay products within chunk

    Monoid: (A1,h1) o (A2,h2) = (A1*A2, A2*h1 + h2).  Identity = (1, 0).
    No per-pass slice/cat — uses F.pad + narrow so the whole tensor updates in
    two fused elementwise ops per pass.
    """
    A = a
    h = b
    L = a.shape[1]
    shift = 1
    while shift < L:
        A_prev = F.pad(A, (0, 0, 0, 0, shift, 0), value=1.0)[:, :L]
        h_prev = F.pad(h, (0, 0, 0, 0, shift, 0), value=0.0)[:, :L]
        h = A * h_prev + h
        A = A_prev * A
        shift *= 2
    return h, A


def _ssm_scan(
    dA:  torch.Tensor,   # (B, T, F)      scalar decay per feature dim
    dBx: torch.Tensor,   # (B, T, F, S)   dB[t] * x[t], additive input
    C:   torch.Tensor,   # (B, T, F, S)   output projection
    CHUNK: int = 2048,
) -> torch.Tensor:
    """
    Returns y: (B, T, F)

    Chunked associative scan.  Within each chunk: log2(CHUNK) parallel passes.
    Between chunks: sequential carry thread (T/CHUNK iterations — ~16 for T=32K).
    fp32 accumulation for numerical stability; cast back to input dtype on return.
    """
    dtype = dBx.dtype
    dA  = dA.float().clamp(1e-6, 1.0)
    dBx = dBx.float()
    C   = C.float()

    B, T, F, S = dBx.shape
    a_full = dA.unsqueeze(-1).expand(B, T, F, S)  # broadcast over S, no copy

    ys    = []
    carry = torch.zeros(B, F, S, device=dBx.device, dtype=torch.float32)

    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        h_local, A_cum = _assoc_scan_chunk(a_full[:, s:e], dBx[:, s:e])
        h = h_local + A_cum * carry.unsqueeze(1)
        carry = h[:, -1]
        ys.append((h * C[:, s:e]).sum(-1))  # (B, L, F)

    return torch.cat(ys, dim=1).to(dtype)  # (B, T, F)


# --------------------------------------------------------------------------- #
#  MambaBlock — correct Mamba1
# --------------------------------------------------------------------------- #

class MambaBlock(nn.Module):
    """
    Mamba1 SSM block.

    x_ssm = x_inner (full d_inner vector, NOT a scalar-per-head).
    A_log, dt_bias, D are all (d_inner,) — one parameter per feature dim.
    State per layer per batch: (d_inner, d_state).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        F_dim = cfg.d_inner    # renamed to avoid shadowing builtin
        S     = cfg.d_state

        self.d_inner = F_dim
        self.d_state = S
        self.d_conv  = cfg.d_conv

        self.in_proj  = nn.Linear(cfg.d_model, F_dim * 2, bias=False)
        self.out_proj = nn.Linear(F_dim, cfg.d_model, bias=False)
        self.norm     = RMSNorm(F_dim)
        self.drop     = nn.Dropout(cfg.dropout)

        # Depthwise causal conv over time
        self.conv1d = nn.Conv1d(
            F_dim, F_dim, kernel_size=cfg.d_conv,
            padding=cfg.d_conv - 1, groups=F_dim, bias=True,
        )

        # Project x_inner → dB (F*S), C (F*S), dt (F)
        # Note: x_ssm IS x_inner, no separate projection needed
        self.x_proj = nn.Linear(F_dim, F_dim * S * 2 + F_dim, bias=False)

        # Per-feature-dim SSM parameters
        self.A_log   = nn.Parameter(torch.empty(F_dim))
        self.dt_bias = nn.Parameter(torch.empty(F_dim))
        self.D       = nn.Parameter(torch.ones(F_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.in_proj.weight,  std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2))
        nn.init.normal_(self.x_proj.weight,   std=0.01)
        nn.init.constant_(self.dt_bias, math.log(math.expm1(1.0)))
        nn.init.uniform_(self.A_log, -2.0, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        F_dim = self.d_inner
        S     = self.d_state

        xz = self.in_proj(x)                                      # (B, T, F*2)
        x_inner, z = xz.chunk(2, dim=-1)                          # (B, T, F) each

        # Causal conv + activation
        x_conv = self.conv1d(x_inner.transpose(1, 2))[..., :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_norm = self.norm(x_conv)                                 # (B, T, F)

        # Project to SSM params
        proj   = self.x_proj(x_norm)                              # (B, T, F*(2S+1))
        B_proj = proj[..., :F_dim * S].reshape(B, T, F_dim, S)
        C_proj = proj[..., F_dim*S : F_dim*S*2].reshape(B, T, F_dim, S)
        dt_raw = proj[..., F_dim*S*2:]                            # (B, T, F)

        dt  = F.softplus(dt_raw + self.dt_bias).clamp(1e-4, 10.0) # (B, T, F)
        A   = -torch.exp(self.A_log)                              # (F,)
        dA  = torch.exp(dt * A)                                   # (B, T, F)

        # dBx: dB[t,f] * x[t,f] outer-producted with dt
        dBx = B_proj * x_norm.unsqueeze(-1) * dt.unsqueeze(-1)   # (B, T, F, S)

        # SSM scan
        y = _ssm_scan(dA, dBx, C_proj)                           # (B, T, F)
        y = y + self.D * x_norm                                   # skip connection

        return self.drop(self.out_proj(y * F.silu(z)))

    def step(
        self,
        x:          torch.Tensor,   # (B, 1, d_model) or (B, d_model)
        ssm_state:  torch.Tensor,   # (B, F, S)
        conv_state: torch.Tensor,   # (B, F, d_conv-1)
    ):
        """Single-token recurrent step.  O(1) memory, O(1) compute."""
        if x.dim() == 3:
            x = x.squeeze(1)        # (B, d_model)
        B = x.shape[0]
        F_dim = self.d_inner
        S     = self.d_state

        xz = self.in_proj(x)        # (B, F*2)
        x_inner, z = xz.chunk(2, dim=-1)

        # Conv step: slide window
        x_padded  = torch.cat([conv_state, x_inner.unsqueeze(2)], dim=2)  # (B, F, d_conv)
        new_conv  = x_padded[:, :, 1:]                                     # (B, F, d_conv-1)
        w         = self.conv1d.weight.squeeze(1)                          # (F, d_conv)
        x_conv    = (x_padded * w.unsqueeze(0)).sum(dim=2) + self.conv1d.bias
        x_conv    = F.silu(x_conv)                                         # (B, F)
        x_norm    = self.norm(x_conv)

        proj   = self.x_proj(x_norm)
        B_proj = proj[:, :F_dim*S].reshape(B, F_dim, S)
        C_proj = proj[:, F_dim*S : F_dim*S*2].reshape(B, F_dim, S)
        dt_raw = proj[:, F_dim*S*2:]

        dt  = F.softplus(dt_raw + self.dt_bias).clamp(1e-4, 10.0)         # (B, F)
        A   = -torch.exp(self.A_log)
        dA  = torch.exp(dt * A)                                            # (B, F)

        dBx     = B_proj * x_norm.unsqueeze(-1) * dt.unsqueeze(-1)        # (B, F, S)
        new_ssm = dA.unsqueeze(-1) * ssm_state + dBx                      # (B, F, S)
        y       = (new_ssm * C_proj).sum(-1) + self.D * x_norm            # (B, F)

        out = self.drop(self.out_proj(y * F.silu(z)))                      # (B, d_model)
        return out, new_ssm, new_conv


# --------------------------------------------------------------------------- #
#  MambaLayer + MidiMamba
# --------------------------------------------------------------------------- #

class MambaLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.ssm   = MambaBlock(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp   = SwiGLU(cfg.d_model, cfg.d_ff_mult)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

    def step(self, x: torch.Tensor, ssm_state: torch.Tensor, conv_state: torch.Tensor):
        h, new_ssm, new_conv = self.ssm.step(self.norm1(x), ssm_state, conv_state)
        x = x.squeeze(1) + h
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, new_ssm, new_conv


class MidiMamba(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([MambaLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, idx: torch.Tensor, states=None):
        """
        idx: (B, T) token ids
        Returns: logits (B, T, vocab_size), [] (states unused in training mode)
        """
        x = self.drop(self.embed(idx))
        for layer in self.layers:
            if self.cfg.grad_ckpt:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.head(self.norm(x)), []

    def step(self, idx: torch.Tensor, states: list):
        """
        Single-token recurrent inference step.
        idx:    (B, 1) token ids
        states: list of (ssm_state, conv_state) per layer
        Returns: logits (B, 1, vocab_size), new_states
        """
        x = self.embed(idx)           # (B, 1, d_model)
        new_states = []
        for i, layer in enumerate(self.layers):
            ssm_s, conv_s = states[i]
            x_out, new_ssm, new_conv = layer.step(x, ssm_s, conv_s)
            x = x_out.unsqueeze(1)
            new_states.append((new_ssm, new_conv))
        logits = self.head(self.norm(x))   # (B, 1, vocab_size)
        return logits, new_states

    def init_states(self, batch_size: int, device: torch.device) -> list:
        """Return zeroed (ssm_state, conv_state) pairs for all layers."""
        F_dim = self.cfg.d_inner
        S     = self.cfg.d_state
        states = []
        for _ in self.layers:
            ssm  = torch.zeros(batch_size, F_dim, S, device=device)
            conv = torch.zeros(batch_size, F_dim, self.cfg.d_conv - 1, device=device)
            states.append((ssm, conv))
        return states

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self) -> str:
        n = self.param_count()
        if n >= 1e9: return f"{n/1e9:.2f}B"
        if n >= 1e6: return f"{n/1e6:.1f}M"
        return f"{n/1e3:.1f}K"
