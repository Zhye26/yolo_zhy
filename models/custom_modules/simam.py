"""
SimAM (Simple Parameter-free Attention Module) for YOLOv8.
Reference: SimAM: A Simple, Parameter-Free Attention Module for CNNs (ICML 2021)

SimAM is a lightweight attention mechanism that:
- Has ZERO learnable parameters
- Based on neuroscience energy function theory
- Computes 3D attention weights without channel/spatial reduction
- Ideal for small models like YOLOv8n where parameter overhead matters
"""
import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    SimAM: Simple Parameter-free Attention Module.

    Unlike CBAM/SE which use learnable parameters, SimAM computes attention
    weights based on energy minimization theory from neuroscience.

    Args:
        c1: Input channels (unused, kept for API compatibility with CBAM)
        e_lambda: Regularization parameter (default: 1e-4)
    """

    def __init__(self, c1=None, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        n = w * h - 1

        # Guard against 1x1 feature maps (n=0)
        if n <= 0:
            return x

        # Calculate mean and variance along spatial dimensions
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        return x * self.act(y)


class SimAM_YOLOv8(nn.Module):
    """SimAM wrapper compatible with YOLOv8 module registration."""

    def __init__(self, c1, c2=None, e_lambda: float = 1e-4):
        super().__init__()
        self.simam = SimAM(c1, e_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.simam(x)
