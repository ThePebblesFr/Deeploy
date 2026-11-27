#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Visual Wake Words (VWW) heavy/small model pair for DynReconf, following the MLCommons
Tiny v1.1 VWW benchmark reference: MobileNetV1 with width multiplier alpha=0.25 on
96x96x3 RGB crops from the Visual Wake Words dataset (COCO subset) -> binary
person-present/absent classification.

Architecture-only (no training script): DynMultiDeeploy's hardware metrics (latency,
energy, operation counts) are derived from graph structure and tensor shapes alone,
not weight values, so these definitions are sufficient to plug into the same
export/fuse/theta pipeline used for ImageClassification/AnomalyDetection with random
weights -- see export_resnet8small.py + fuse_requantshift.py for that flow.

Input: (1, 3, 96, 96) float32. Output: (1, 2) logits.
"""
import torch.nn as nn


class DWSepBlock(nn.Module):
    """Depthwise 3x3 conv (given stride) + pointwise 1x1 conv, each followed by BN+ReLU --
    the standard MobileNetV1 depthwise-separable block."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                    padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Two distinct instances, not one shared self.relu called twice: quantlib's
        # PACT replacement/FX pipeline maps each replaced module to exactly one call
        # site, and breaks ("references nonexistent attribute") when a single nn.ReLU
        # instance is invoked from two different points in forward().
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x


class MobileNetV1(nn.Module):
    """Generic MobileNetV1 body: strided 3x3 stem conv + a configurable stack of
    DWSepBlocks (each (out_channels, stride)) + global average pool + FC."""

    def __init__(self, block_cfg, stem_channels, n_classes=2, in_channels=3):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(stem_channels)
        self.relu = nn.ReLU(inplace=True)

        blocks = []
        ch = stem_channels
        for out_ch, stride in block_cfg:
            blocks.append(DWSepBlock(ch, out_ch, stride))
            ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem(x)))
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# Standard MobileNetV1 channel/stride progression (13 DS blocks), scaled by width
# multiplier alpha=0.25 and rounded -- matches the MLCommons Tiny VWW reference.
_HEAVY_CFG = [
    (16, 1), (32, 2), (32, 1), (64, 2), (64, 1), (128, 2),
    (128, 1), (128, 1), (128, 1), (128, 1), (128, 1), (256, 2), (256, 1),
]

# Moderate shrink: alpha ~0.1 stem, 6 DS blocks (vs 13 for heavy). Reverted here
# from an even-more-aggressive 3-block/242-param cut: once heavy was properly
# trained to convergence (~80%, up from an under-trained ~75%, see VWWHeavy's
# training history), an industrial-accuracy-floor requirement (>=70% for any
# model to be embeddable) ruled out shrinking small past this point -- this
# config already demonstrated >=75% test accuracy on its own, comfortably above
# the floor with a real (~5-10pt) gap under heavy.
_SMALL_CFG = [(8, 1), (16, 2), (16, 1), (32, 2), (32, 1), (32, 1)]


class VWWHeavy(MobileNetV1):
    """Reference MLCommons Tiny VWW model: MobileNetV1, width multiplier 0.25, 96x96x3 input."""

    def __init__(self):
        super().__init__(block_cfg=_HEAVY_CFG, stem_channels=8, n_classes=2, in_channels=3)


class VWWSmall(MobileNetV1):
    """Shrunk MobileNetV1: width multiplier ~0.1, 6 DS blocks (vs 13 for heavy)."""

    def __init__(self):
        super().__init__(block_cfg=_SMALL_CFG, stem_channels=4, n_classes=2, in_channels=3)
