"""
BiFPN (Bidirectional Feature Pyramid Network) module for YOLOv8.
Reference: EfficientDet (https://arxiv.org/abs/1911.09070)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPNConcat(nn.Module):
    """Weighted feature fusion for BiFPN with learnable weights."""

    def __init__(self, c1, c2, num_inputs=2, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.num_inputs = num_inputs
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass with weighted fusion."""
        weights = self.relu(self.weights)
        weights = weights / (weights.sum() + self.epsilon)

        # Resize features to match the first input's size
        target_size = x[0].shape[2:]
        fused = sum(w * (F.interpolate(xi, size=target_size, mode='nearest')
                        if xi.shape[2:] != target_size else xi)
                   for w, xi in zip(weights, x))
        return fused


class BiFPN(nn.Module):
    """
    BiFPN layer that performs bidirectional feature fusion.
    Replaces PANet in YOLOv8 for better multi-scale feature aggregation.
    """

    def __init__(self, channels, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        # Convolutions for feature processing
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_convs = nn.ModuleDict({
                'p3_td': self._make_conv(channels[0]),
                'p4_td': self._make_conv(channels[1]),
                'p4_out': self._make_conv(channels[1]),
                'p5_out': self._make_conv(channels[2]),
            })
            self.conv_layers.append(layer_convs)

        # Weighted fusion modules
        self.fuse_p4_td = BiFPNConcat(channels[1], channels[1], num_inputs=2)
        self.fuse_p3_out = BiFPNConcat(channels[0], channels[0], num_inputs=2)
        self.fuse_p4_out = BiFPNConcat(channels[1], channels[1], num_inputs=3)
        self.fuse_p5_out = BiFPNConcat(channels[2], channels[2], num_inputs=2)

    def _make_conv(self, channels):
        """Create a depthwise separable convolution block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, features):
        """
        Forward pass.
        Args:
            features: list of [P3, P4, P5] feature maps
        Returns:
            list of fused [P3_out, P4_out, P5_out] feature maps
        """
        p3, p4, p5 = features

        for convs in self.conv_layers:
            # Top-down pathway
            p5_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
            p4_td = convs['p4_td'](self.fuse_p4_td([p4, p5_up]))

            p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
            p3_out = convs['p3_td'](self.fuse_p3_out([p3, p4_up]))

            # Bottom-up pathway
            p3_down = F.interpolate(p3_out, size=p4.shape[2:], mode='nearest')
            p4_out = convs['p4_out'](self.fuse_p4_out([p4, p4_td, p3_down]))

            p4_down = F.interpolate(p4_out, size=p5.shape[2:], mode='nearest')
            p5_out = convs['p5_out'](self.fuse_p5_out([p5, p4_down]))

            # Update for next iteration
            p3, p4, p5 = p3_out, p4_out, p5_out

        return [p3, p4, p5]
