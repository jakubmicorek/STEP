"""
models/step.py
==============
STEP energy network: a sigma-modulated Residual MLP with DSM-based score
matching.  This is the only architecture in the released code.

Sigma is fused with a residual skip in every block.

The class also carries the sigma schedule / noising / score machinery
(geometric grid of L sigma values in [sigma_low, sigma_high]).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _ModulatedBlock(nn.Module):
    """Residual block: h_out = x_in + GELU(Linear(x_in)) + enc(sigma) * alpha(sigma)."""

    def __init__(self, dim: int, layernorm: bool = False):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.act = nn.GELU()

        # Sigma modulation path
        enc_dim = max(dim // 4, 1)
        self.sigma_enc = nn.Sequential(
            nn.Linear(1, enc_dim, bias=False),
            nn.GELU(),
            nn.Linear(enc_dim, dim, bias=False),
        )
        self.sigma_norm = nn.LayerNorm(dim)
        self.sigma_alpha = nn.Linear(1, dim, bias=False)  # learned scale, starts near identity
        nn.init.uniform_(self.sigma_alpha.weight, 1e-3, 1.0)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.linear(x)))

        # Sigma modulation: enc(sigma) scaled by alpha(sigma)
        enc = self.sigma_norm(self.sigma_enc(sigma))   # [N, dim]
        alpha = self.sigma_alpha(sigma)                 # [N, dim]
        h = h + enc * alpha

        # Residual
        return x + h


class STEPNetwork(nn.Module):
    """sigma-modulated Residual MLP. default: feature_dim=48, units=[1024]*4,
    L=10, sigma_low=0.1, sigma_high=1.0"""

    def __init__(
        self,
        feature_dim: int,
        units: list[int] = (1024, 1024, 1024, 1024),
        layernorm: bool = False,
        network_jitter: float = 0.0,
        L: int = 10,
        sigma_low: float = 0.1,
        sigma_high: float = 1.0,
    ):
        super().__init__()
        self.network_jitter = network_jitter
        self.L = L
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high

        # Early fusion; sigma.detach() keeps the sigma gradient flowing only
        # through the block modulation paths, not through this projection.
        self.input_proj = nn.Linear(feature_dim + 1, units[0])

        self.blocks = nn.ModuleList()
        in_dim = units[0]
        for out_dim in units:
            if in_dim != out_dim:
                raise ValueError(
                    f"All units must be equal for residual connections "
                    f"(got {in_dim} -> {out_dim})."
                )
            self.blocks.append(_ModulatedBlock(out_dim, layernorm))
            in_dim = out_dim

        self.head = nn.Linear(in_dim, 1)

    # ------------------------------------------------------------------
    # Sigma schedule
    # ------------------------------------------------------------------

    def get_sigma_list(self) -> list[float]:
        """Geometric sequence of L sigma values in [sigma_low, sigma_high]."""
        sigmas = np.exp(np.linspace(np.log(self.sigma_low), np.log(self.sigma_high), self.L))
        return [float(f"{s:.5f}") for s in sigmas]

    def sample_sigma(self, batch_size: int) -> torch.Tensor:
        """Discrete uniform over the geometric sigma grid."""
        grid = np.array(self.get_sigma_list())
        return torch.tensor(np.random.choice(grid, batch_size), dtype=torch.float32).unsqueeze(1)

    # ------------------------------------------------------------------
    # Noising
    # ------------------------------------------------------------------

    def add_noise(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_noisy, sigma, noise) where x_noisy = x + sigma * eps."""
        sigma = self.sample_sigma(x.size(0)).to(x.device)
        noise = torch.randn_like(x) * sigma
        return x + noise, sigma, noise

    # ------------------------------------------------------------------
    # Forward pass - energy only (for evaluation, no grad w.r.t. x)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Energy f(x, sigma)."""
        h = self.input_proj(torch.cat([x, sigma.detach()], dim=1))

        for block in self.blocks:
            if self.training and self.network_jitter > 0:
                h = h + torch.randn_like(h) * self.network_jitter
            h = block(h, sigma)

        return self.head(h)

    # ------------------------------------------------------------------
    # Forward pass + gradient w.r.t. x (for DSM training)
    # ------------------------------------------------------------------

    def score(self, x: torch.Tensor, sigma: torch.Tensor) -> dict:
        """Returns {"energy": [N,1], "score_x": [N,D]} where score_x = grad_x f(x, sigma)."""
        x = x.requires_grad_(True)
        energy = self.forward(x, sigma)
        score_x = torch.autograd.grad(-energy.sum(), x, create_graph=self.training)[0]
        return {"energy": energy, "score_x": score_x}
