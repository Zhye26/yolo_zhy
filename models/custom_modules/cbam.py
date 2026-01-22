"""
CBAM (Convolutional Block Attention Module) for YOLOv8.
Reference: CBAM: Convolutional Block Attention Module (https://arxiv.org/abs/1807.06521)
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(channels // reduction, 8)  # Ensure at least 8 channels
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module.
    Combines channel and spatial attention for feature refinement.

    Args:
        c1: input/output channels (for attention, no change in channels)
        reduction: channel reduction ratio for ChannelAttention
        kernel_size: kernel size for SpatialAttention
    """

    def __init__(self, c1, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


class C2f_CBAM(nn.Module):
    """C2f module with CBAM attention at the end."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cbam = CBAM(c2, c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cbam(self.cv2(torch.cat(y, 1)))


# Import Conv and Bottleneck from ultralytics
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck
