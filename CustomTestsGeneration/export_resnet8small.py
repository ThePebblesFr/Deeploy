#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Export the trained ResNet8Small (quantlab/systems/CIFAR10/ResNet, exp0000) to a
Deeploy-style int8 ONNX, bypassing quantlab/examples/fx_integerization/
integerize_pactnets.py's integerize_network()/DORYHarmonizePass path -- that
wrapper's IntegerizePACTNetPass call uses kwargs (n_levels_in, fix_channel_numbers,
requant_node) that no longer exist on the current quantlib.editing.fx.passes.pact.
IntegerizePACTNetPass, and DORYHarmonizePass itself isn't exported by this quantlib
checkout's quantlib.backends.dory package at all.

quantlib.backends.dory.export_net (used by create_deeploy_int8_test_from_pytorch.py
for the MLP case too) already calls IntegerizePACTNetPass internally with the
current signature, so this script just loads the trained+quantized network via
integerize_pactnets.py's own get_network() helper (that part still works) and
hands it straight to export_net, skipping the broken wrapper entirely.

Usage (inside the container):
    python3 export_resnet8small.py --out-dir /tmp/resnet8small_export
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/app/Deeploy/quantlab")
sys.path.insert(0, "/app/Deeploy/quantlab/examples/fx_integerization")

from integerize_pactnets import get_network, get_config, get_valid_dataset, _QUANT_UTILS  # noqa: E402
from quantlib.backends.dory import export_net  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", type=int, default=0)
    ap.add_argument("--ckpt-id", type=int, default=-1)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--name", type=str, default="ImageClassification_small")
    ap.add_argument("--in-idx", type=int, default=42, help="Validation-set index to use as the test input.")
    args = ap.parse_args()

    key = "ResNetCIFAR"
    qu = _QUANT_UTILS[key]
    cfg = get_config(key, args.exp_id)

    net = get_network(key, args.exp_id, args.ckpt_id, quantized=True)

    ds = get_valid_dataset(key, cfg, quantize="int")
    test_input = ds[args.in_idx][0].unsqueeze(0)
    test_label = ds[args.in_idx][1]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_net(net, name=args.name, out_dir=str(out_dir), eps_in=qu.eps_in,
               in_data=test_input, integerize=True, D=qu.D)

    with torch.no_grad():
        logits = net(test_input.to(torch.float32))
    pred = int(torch.argmax(logits, dim=1).item())
    print(f"Test sample index {args.in_idx}: true label={int(test_label)}, predicted={pred}")
    print(f"Exported to {out_dir}/{args.name}_ql_integerized.onnx")


if __name__ == "__main__":
    main()
