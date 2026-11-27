#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Keyword Spotting (KS) heavy/small model pair for DynReconf, following the MLCommons
Tiny v1.1 KS benchmark reference: a depthwise-separable CNN (DS-CNN, Zhang et al.
"Hello Edge", arXiv:1711.07128) operating on a 49x10 MFCC feature map (49 time frames
x 10 MFCC coefficients, 1 channel) from Google Speech Commands -> 12-class softmax
(10 keywords + "unknown" + "silence").

Architecture-only (no training script): DynMultiDeeploy's hardware metrics (latency,
energy, operation counts) are derived from graph structure and tensor shapes alone,
not weight values, so these definitions are sufficient to plug into the same
export/fuse/theta pipeline used for ImageClassification/AnomalyDetection with random
weights -- see export_resnet8small.py + fuse_requantshift.py for that flow.

Input: (1, 1, 49, 10) float32. Output: (1, 12) logits.
"""
import torch.nn as nn


class DSConvBlock(nn.Module):
    """Depthwise 3x3 conv + pointwise 1x1 conv, each followed by BN+ReLU (stride 1, same spatial size)."""

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                                    padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
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


class DSCNN(nn.Module):
    """Strided Conv2d stem (matches the DS-CNN reference's (10,4)/stride-2 first layer,
    approximated with PyTorch-native padding since Keras 'same' padding has no direct
    stride>1 equivalent here) + `n_blocks` DSConvBlocks + global average pool + FC."""

    def __init__(self, channels=64, n_blocks=4, n_classes=12):
        super().__init__()
        # Reference DS-CNN uses a (10,4)/stride-2/pad-(4,1) stem -- Deeploy's
        # PULPConv2DParser requires SQUARE padding (pads[0] == pads[1], i.e. top
        # padding == left padding), which that asymmetric kernel/padding violates.
        # Square 3x3/stride-2/pad-1 keeps the same "strided stem" role (spatial
        # downsampling before the DS-conv blocks) and is fully Deeploy-compatible;
        # already an approximation per this class's own docstring (Keras "same"
        # padding has no direct stride>1 PyTorch equivalent anyway).
        self.stem = nn.Conv2d(1, channels, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(*[DSConvBlock(channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem(x)))
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class KSHeavy(DSCNN):
    """Reference MLCommons Tiny DS-CNN: 64 channels, 4 DS-conv blocks (~24.7K params)."""

    def __init__(self):
        super().__init__(channels=64, n_blocks=4, n_classes=12)


class KSSmall(DSCNN):
    """Shrunk DS-CNN: 8 channels (1/8 width), 2 DS-conv blocks (half depth) -- same topology."""

    def __init__(self):
        super().__init__(channels=8, n_blocks=2, n_classes=12)
