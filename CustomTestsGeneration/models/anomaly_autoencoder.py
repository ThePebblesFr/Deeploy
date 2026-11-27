#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Light/heavy dense autoencoder pair for the anomaly-detection task, sized to match
DeeployTest/Tests/MLPerf/AnomalyDetection's (1,640) flat feature-vector interface.

Both variants share the same depth and I/O shape and differ only in hidden-layer
(bottleneck) width, so they are drop-in interchangeable variants of the same task:
    AutoencoderLight: 640 -> 64  -> 8  -> 64  -> 640
    AutoencoderHeavy: 640 -> 256 -> 32 -> 256 -> 640

Usage (inside the Deeploy/QuantLib container, from CustomTestsGeneration/):
    python3 create_deeploy_int8_test_from_pytorch.py \
        --model-path models/anomaly_autoencoder.py --model-class AutoencoderLight \
        --test-name AnomalyDetection_light --input-shapes "1,640" \
        --output-dir DeeployTest/Tests --seed 42

    python3 create_deeploy_int8_test_from_pytorch.py \
        --model-path models/anomaly_autoencoder.py --model-class AutoencoderHeavy \
        --test-name AnomalyDetection_heavy --input-shapes "1,640" \
        --output-dir DeeployTest/Tests --seed 42
"""

import torch.nn as nn


class AutoencoderLight(nn.Module):
    def __init__(self):
        super().__init__()
        # Widening the bottleneck past 8 measurably lowers reconstruction MSE but hurts
        # anomaly-detection accuracy (a higher-capacity bottleneck reconstructs anomalies
        # too well too, shrinking the normal/anomalous error gap) -- confirmed empirically
        # on DCASE ToyCar, so 8 (matching the heavy reference) is kept.
        self.net = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 640),
        )

    def forward(self, x):
        return self.net(x)


class AutoencoderHeavy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 640),
        )

    def forward(self, x):
        return self.net(x)


class AutoencoderMLCommonsHeavy(nn.Module):
    """
    The real MLCommons Tiny AnomalyDetection reference architecture (matches
    DeeployTest/Tests/MLPerf/AnomalyDetection's onnx exactly: 10 Gemm layers,
    640-128-128-128-128-8-128-128-128-128-640), used to retrain a "heavy" variant
    whose real accuracy we can verify -- the pre-existing checked-in heavy model has
    no documented input calibration, so its accuracy can't be reliably measured.
    """

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
            nn.Linear(128, 8),
            nn.ReLU(),
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 640),
        )

    def forward(self, x):
        return self.net(x)
