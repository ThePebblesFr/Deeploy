#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Supervised normal/anomaly binary classifier pair for DCASE 2020 ToyCar (see
train_anomaly_classifier.py), heavy/small sized to mirror the ImageClassification
heavy/small capacity gap. Same (1,640) input interface as the reconstruction
autoencoders, 2-class logit output instead of a 640-dim reconstruction.
"""

import torch.nn as nn


class ClassifierSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(640, 2),
        )

    def forward(self, x):
        return self.net(x)


class ClassifierHeavy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)
