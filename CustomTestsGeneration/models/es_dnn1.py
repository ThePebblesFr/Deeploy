#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
ES_DNN1 model definition, for use with create_deeploy_int8_native.py:

    python3 create_deeploy_int8_native.py \
        --model-path models/es_dnn1.py \
        --model-class ES_DNN1 \
        --test-name ES_DNN1_int8_native \
        --input-shape 1,16 \
        --seed 42 \
        --verbose
"""

import torch.nn as nn


class ES_DNN1(nn.Module):

    def __init__(self):
        super().__init__()
        layers = [
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
