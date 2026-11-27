#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Light/heavy dense classifier pair standing in for Task 2 (image classification) while
validating the DynMultiDeeploy pipeline's variant-switching machinery -- not a real
image classifier (no conv, random untrained weights), just a distinct-shape
Linear(+ReLU) MLP pair with the same (1,256) input / 10-class output interface across
both variants, matching create_deeploy_int8_native.py's current Linear-only scope.

Usage (inside the Deeploy container, from CustomTestsGeneration/):
    python3 create_deeploy_int8_native.py \
        --model-path models/tiny_classifier.py --model-class TinyClassifierLight \
        --test-name TinyClassifier_light --input-shape "1,256" \
        --output-dir ../DeeployTest/Tests --seed 42

    python3 create_deeploy_int8_native.py \
        --model-path models/tiny_classifier.py --model-class TinyClassifierHeavy \
        --test-name TinyClassifier_heavy --input-shape "1,256" \
        --output-dir ../DeeployTest/Tests --seed 42
"""

import torch.nn as nn


class TinyClassifierLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


class TinyClassifierHeavy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)
