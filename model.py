"""
model.py — MidiMamba

Correct Mamba2 SSM implementation, pure PyTorch, Windows-compatible.

SSM per head h (d_state):
  h[t] = dA[t] * h[t-1] + dB[t] * x_ssm[t]
  y[t] = C[t] @ h[t]

where x_ssm is a scalar per head (the "inner" value after projection),
dA is scalar decay per head, dB and C are d_state vectors per head.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int  = 512
    d_model: int     = 512
    n_layers: int    = 16
    d_state: int     = 64
    d_conv: int      = 4
    expand: int      = 2
    n_heads: int     = 8
    d_ff_mult: float = 2.667
    dropout: float   = 0.1
    max_seq: int     = 65536
    grad_checkpoint: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x / (x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt())


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 2.667):
        super().__init__()
        d_ff = (int(d_model * mult) + 63) // 64 * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --------------------------------------------------------------------------- #
#  SSM scan — segmented for numerical stability, ~7 Python iters for T=53K
# --------------------------------------------------------------------------- #

def _ssm_scan(
    dA:  torch.Tensor,  # (B, T, H)    decay per head, values in (0,1)
    dBx: torch.Tensor,  # (B, T, H, S) input: dB[t] * x_ssm[t]
    C:   torch.Tensor,  # (B, T, H, S) output projection
) -> torch.Tensor:
    """
    h[t] = dA[t] * h[t-1] + dBx[t]
    y[t] = (h[t] * C[t]).sum(-1)   (dot product over state dim)
    Returns y: (B, T, H)
    """
    dtype = dBx.dtype
    dA  = dA.float().clamp(1e-6, 1.0)
    dBx = dBx.float().clamp(-1e4, 1e4)  # prevent inf inputs
    C   = C.float()

    B, T, H, S = dBx.shape
    SEG = 8192

    out = torch.zeros(B, T, H, S, device=dBx.device, dtype=torch.float32)
    h   = torch.zeros(B, H, S,    device=dBx.device, dtype=torch.float32)

    for s in range(0, T, SEG):
        e    = min(s + SEG, T)
        dA_s = dA  [:, s:e]   # (B, C, H)
        Bx_s = dBx [:, s:e]   # (B, C, H, S)
        C    = C               # unchanged

        # logcumA[t] = log(dA[s]) + ... + log(dA[s+t])
        logcumA      = torch.cumsum(torch.log(dA_s), dim=1)  # (B, C, H)
        cumA         = torch.exp(logcumA.clamp(-80, 0))       # (B, C, H) in (0,1]

        # logcumA_prev: shift by one, first entry = 0 (no decay yet)
        logcumA_prev = torch.cat([
            torch.zeros(B, 1, H, device=dBx.device),
            logcumA[:, :-1]
        ], dim=1)
        inv_cumA     = torch.exp((-logcumA_prev).clamp(-80, 0))  # (B, C, H)

        # Weighted cumsum of inputs
        Bx_norm  = Bx_s * inv_cumA.unsqueeze(-1)                  # (B, C, H, S)
        bu       = cumA.unsqueeze(-1) * torch.cumsum(Bx_norm, dim=1)

        # Carry: propagate h through segment
        carry    = cumA.unsqueeze(-1) * h.unsqueeze(1)            # (B, C, H, S)

        h_seg    = (bu + carry).clamp(-1e6, 1e6)                  # prevent explosion
        out[:, s:e] = h_seg
        h        = h_seg[:, -1]

    # Output: dot product with C per timestep
    y = (out * C.float()).sum(-1)  # (B, T, H)
    return y.to(dtype)


# --------------------------------------------------------------------------- #
#  MambaBlock — correct Mamba formulation
# --------------------------------------------------------------------------- #

class MambaBlock(nn.Module):
    """
    Mamba SSM block with correct math.

    Per head h of size d_state:
      x_ssm: scalar input (one value per head, from inner projection)
      dB:    (d_state,) input gate
      C:     (d_state,) output gate
      dA:    scalar decay
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.d_state  = cfg.d_state
        self.d_conv   = cfg.d_conv

        d_inner = (cfg.d_model * cfg.expand // cfg.n_heads) * cfg.n_heads
        self.d_inner  = d_inner
        self.head_dim = d_inner // cfg.n_heads  # x_inner per head

        self.in_proj  = nn.Linear(cfg.d_model, d_inner * 2, bias=False)
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)
        self.norm     = RMSNorm(d_inner)
        self.drop     = nn.Dropout(cfg.dropout)

        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=cfg.d_conv,
            padding=cfg.d_conv - 1, groups=d_inner, bias=True,
        )

        # Project inner to: x_ssm (H scalars), dB (H*S), C (H*S), dt (H)
        # x_ssm: one scalar per head = H dims total
        self.x_proj = nn.Linear(d_inner, cfg.n_heads * (1 + cfg.d_state * 2 + 1), bias=False)

        self.A_log   = nn.Parameter(torch.zeros(cfg.n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads))
        self.D       = nn.Parameter(torch.ones(cfg.n_heads))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.in_proj.weight,  std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2))
        nn.init.normal_(self.x_proj.weight,   std=0.01)  # small init — prevents NaN
        nn.init.constant_(self.dt_bias, math.log(math.expm1(1.0)))
        nn.init.uniform_(self.A_log, -2.0, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.n_heads; S = self.d_state; Hd = self.head_dim

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[..., :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_norm = self.norm(x_conv)

        # Project to SSM params — all at once
        proj   = self.x_proj(x_norm)                              # (B, T, H*(1+2S+1))
        offset = 0
        x_ssm  = proj[..., offset:offset+H].reshape(B, T, H, 1)  # scalar per head
        offset += H
        B_proj = proj[..., offset:offset+H*S].reshape(B, T, H, S)
        offset += H*S
        C_proj = proj[..., offset:offset+H*S].reshape(B, T, H, S)
        offset += H*S
        dt_raw = proj[..., offset:offset+H]                       # (B, T, H)

        dt  = F.softplus(dt_raw + self.dt_bias).clamp(1e-4, 10.0)
        A   = -torch.exp(self.A_log)
        dA  = torch.exp(dt * A)                                    # (B, T, H)

        # dBx: dB[t] * x_ssm[t] — outer product, shape (B, T, H, S)
        dBx = B_proj * x_ssm * dt.unsqueeze(-1)                   # (B, T, H, S)

        # SSM scan → y_h: (B, T, H)
        y_h = _ssm_scan(dA, dBx, C_proj)

        # Skip connection
        y_h = y_h + self.D * x_ssm.squeeze(-1)

        # Broadcast y_h back to d_inner: repeat each head scalar across head_dim
        y = y_h.unsqueeze(-1).expand(-1, -1, -1, Hd).reshape(B, T, self.d_inner)
        return self.drop(self.out_proj(y * F.silu(z)))

    def step(self, x, ssm_state, conv_state):
        """Single-token recurrent step for inference."""
        B  = x.shape[0]
        H  = self.n_heads; S = self.d_state; Hd = self.head_dim

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_t = x_inner.squeeze(1)

        x_padded = torch.cat([conv_state, x_t.unsqueeze(2)], dim=2)
        new_conv = x_padded[:, :, 1:]
        w = self.conv1d.weight.squeeze(1)
        x_conv = (x_padded * w.unsqueeze(0)).sum(dim=2) + self.conv1d.bias
        x_conv = F.silu(x_conv)
        x_norm = self.norm(x_conv)

        proj   = self.x_proj(x_norm)
        offset = 0
        x_ssm  = proj[:, offset:offset+H]
        offset += H
        B_proj = proj[:, offset:offset+H*S].reshape(B, H, S)
        offset += H*S
        C_proj = proj[:, offset:offset+H*S].reshape(B, H, S)
        offset += H*S
        dt_raw = proj[:, offset:offset+H]

        dt  = F.softplus(dt_raw + self.dt_bias).clamp(1e-4, 10.0)
        A   = -torch.exp(self.A_log)
        dA  = torch.exp(dt * A)
        dBx = B_proj * x_ssm.unsqueeze(-1) * dt.unsqueeze(-1)    # (B, H, S)

        new_ssm = dA.unsqueeze(-1) * ssm_state + dBx
        y_h     = (new_ssm * C_proj).sum(-1) + self.D * x_ssm    # (B, H)

        y   = y_h.unsqueeze(-1).expand(-1, -1, Hd).reshape(B, self.d_inner)
        out = self.drop(self.out_proj(y * F.silu(z.squeeze(1))))
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

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

    def step(self, x, ssm_state, conv_state):
        h, new_ssm, new_conv = self.ssm.step(self.norm1(x), ssm_state, conv_state)
        x = x.squeeze(1) + h
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, new_ssm, new_conv


class MidiMamba(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg    = cfg or ModelConfig()
        c           = self.cfg
        self.embed  = nn.Embedding(c.vocab_size, c.d_model)
        self.drop   = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList([MambaLayer(c) for _ in range(c.n_layers)])
        self.norm   = RMSNorm(c.d_model)
        self.head   = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, idx, states=None):
        x = self.drop(self.embed(idx))
        if self.cfg.grad_checkpoint:
            x = x.requires_grad_(True)
        for layer in self.layers:
            if self.cfg.grad_checkpoint:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.head(self.norm(x)), []

    def step(self, idx, states):
        x = self.embed(idx)
        new_states = []
        for i, layer in enumerate(self.layers):
            ssm_s, conv_s = states[i]
            x_out, new_ssm, new_conv = layer.step(x, ssm_s, conv_s)
            x = x_out.unsqueeze(1)
            new_states.append((new_ssm, new_conv))
        return self.head(self.norm(x)), new_states

    def init_states(self, batch_size, device):
        H = self.cfg.n_heads
        d_inner = (self.cfg.d_model * self.cfg.expand // H) * H
        S = self.cfg.d_state
        states = []
        for _ in self.layers:
            ssm  = torch.zeros(batch_size, H, S, device=device)
            conv = torch.zeros(batch_size, d_inner, self.cfg.d_conv - 1, device=device)
            states.append((ssm, conv))
        return states

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self):
        n = self.param_count()
        if n >= 1e9: return f"{n/1e9:.2f}B"
        if n >= 1e6: return f"{n/1e6:.1f}M"
        return f"{n/1e3:.1f}K"
