#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""DAMOT Training-time Difference Enhancement Module (TDEM)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalConvRefinement(nn.Module):
    """Two 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BilinearUpBlock(nn.Module):
    """Bilinear x2 -> 3x3 channel reduction -> local refinement."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        reduced_channels = in_channels // 2
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.channel_reduction = nn.Conv2d(
            in_channels, reduced_channels, 3, 1, 1
        )
        self.local_refinement = LocalConvRefinement(
            reduced_channels, out_channels
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.channel_reduction(x)
        return self.local_refinement(x)


class BilinearReconstructionModule(nn.Module):
    """BRM reconstructs RGB from stride-8 P3 using three up blocks."""

    def __init__(self, in_channels, decoder_channels=(128, 64, 32)):
        super().__init__()
        channels = (in_channels,) + tuple(decoder_channels)
        self.up_blocks = nn.Sequential(
            *[
                BilinearUpBlock(channels[index], channels[index + 1])
                for index in range(len(decoder_channels))
            ]
        )
        self.output = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, p3, output_size):
        reconstructed = self.output(self.up_blocks(p3))
        if reconstructed.shape[-2:] != output_size:
            reconstructed = F.interpolate(
                reconstructed,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return reconstructed


class DifferenceDomainBackgroundSmoothing(nn.Module):
    """Preserve target residuals and smooth RGB background residuals."""

    def __init__(self):
        super().__init__()
        self.smoother = nn.Sequential(
            nn.Conv2d(3, 1, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 3, 3, 1, 1),
        )

    @staticmethod
    def _foreground_mask(residual_map, targets):
        batch_size, _, height, width = residual_map.shape
        foreground = residual_map.new_zeros((batch_size, 1, height, width))

        for batch_index in range(batch_size):
            image_targets = targets[batch_index]
            valid = (image_targets[:, 3] > 0) & (image_targets[:, 4] > 0)
            boxes = image_targets[valid, 1:5].detach()
            if boxes.numel() == 0:
                continue

            cx, cy, box_width, box_height = boxes.unbind(dim=1)
            xyxy = torch.stack(
                (
                    cx - box_width * 0.5,
                    cy - box_height * 0.5,
                    cx + box_width * 0.5,
                    cy + box_height * 0.5,
                ),
                dim=1,
            )
            xyxy[:, 0::2].clamp_(0, width)
            xyxy[:, 1::2].clamp_(0, height)
            xyxy[:, :2] = torch.floor(xyxy[:, :2])
            xyxy[:, 2:] = torch.ceil(xyxy[:, 2:])

            for x1, y1, x2, y2 in xyxy.to(dtype=torch.int64).cpu().tolist():
                if x2 > x1 and y2 > y1:
                    foreground[batch_index, :, y1:y2, x1:x2] = 1

        return foreground

    def forward(self, residual_map, targets):
        foreground = self._foreground_mask(residual_map, targets)
        background = 1.0 - foreground
        background_residual = residual_map * background
        smoothed_background = background_residual + self.smoother(
            background_residual
        )
        return residual_map * foreground + smoothed_background * background


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        hidden_channels = max(channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
        )

    def forward(self, feature):
        avg_attention = self.mlp(F.adaptive_avg_pool2d(feature, 1).flatten(1))
        max_attention = self.mlp(F.adaptive_max_pool2d(feature, 1).flatten(1))
        return torch.sigmoid(avg_attention + max_attention).unsqueeze(-1).unsqueeze(-1)


class DifferenceGuidedSpatialChannelEnhancement(nn.Module):
    """Difference-guided spatial modulation with channel attention."""

    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)

    def forward(self, p3, difference_map, learnable_threshold):
        binary_map = (
            torch.sign(difference_map - learnable_threshold) + 1.0
        ) * 0.5
        binary_map = F.interpolate(
            binary_map, size=p3.shape[-2:], mode="nearest"
        ).to(dtype=p3.dtype)
        reweighted = p3 * self.channel_attention(p3)
        return reweighted * binary_map + reweighted


class TrainingTimeDifferenceEnhancementModule(nn.Module):
    """TDEM builds enhanced P3 during training and is bypassed at inference."""

    def __init__(
        self,
        p3_channels,
        decoder_channels=(128, 64, 32),
        learnable_threshold=0.0156862,
        reconstruction_loss_weight=1.0,
        rgb_mean=(0.485, 0.456, 0.406),
        rgb_std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.brm = BilinearReconstructionModule(
            p3_channels, decoder_channels=decoder_channels
        )
        self.background_smoothing = DifferenceDomainBackgroundSmoothing()
        self.spatial_channel_enhancement = (
            DifferenceGuidedSpatialChannelEnhancement(p3_channels)
        )
        self.learnable_threshold = nn.Parameter(
            torch.tensor(float(learnable_threshold))
        )
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.register_buffer(
            "rgb_mean", torch.tensor(rgb_mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "rgb_std", torch.tensor(rgb_std).view(1, 3, 1, 1), persistent=False
        )

    def _denormalize(self, normalized_image):
        rgb_std = self.rgb_std.to(dtype=normalized_image.dtype)
        rgb_mean = self.rgb_mean.to(dtype=normalized_image.dtype)
        return (normalized_image * rgb_std + rgb_mean).clamp_(0.0, 1.0)

    def forward(self, p3, normalized_image, targets):
        original_image = self._denormalize(normalized_image)
        reconstructed_image = self.brm(
            p3, output_size=original_image.shape[-2:]
        )
        rgb_residual = torch.abs(reconstructed_image - original_image)
        refined_residual = self.background_smoothing(rgb_residual, targets)
        difference_map = refined_residual.mean(dim=1, keepdim=True)
        enhanced_p3 = self.spatial_channel_enhancement(
            p3, difference_map, self.learnable_threshold
        )
        reconstruction_loss = F.mse_loss(
            reconstructed_image, original_image, reduction="mean"
        ) * self.reconstruction_loss_weight
        return enhanced_p3, reconstruction_loss
