"""training/loss.py - DSM loss (pure function, no side effects)."""

import torch


def dsm_loss(
    model,
    x: torch.Tensor,
    confs: torch.Tensor,
    confidence_weighted: bool = True,
) -> torch.Tensor:
    """DSM loss: E_sigma[ sigma^2 * ||grad_x f(x~, sigma) + noise/sigma^2||^2 ], optionally confidence-weighted."""
    x_noisy, sigma, noise = model.add_noise(x)

    result = model.score(x_noisy, sigma)
    score_x = result["score_x"]           # [N, D]

    target    = -noise / (sigma ** 2)     # [N, D]
    diff      = score_x - target
    residuals = (diff ** 2).sum(dim=-1)   # [N]

    if confidence_weighted:
        residuals = residuals * confs.view(-1)

    return (sigma.view(-1) ** 2 * residuals).mean() / 2.0
