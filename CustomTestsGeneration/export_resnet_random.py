#!/usr/bin/env python3
"""
Export a randomly-initialized (untrained) CIFAR10 ResNet variant to a Deeploy-style
INT8 ONNX, bypassing the logs/exp<NNNN>/config.json + checkpoint indirection that
export_resnet8small.py relies on (there's no trained checkpoint yet -- weights are
coming later). Builds the network + applies the exact same PACT quantization recipe
a real training run would use (read straight from the matching config.*.json in
this repo), but skips loading any checkpoint state_dict, so weights stay at
whatever nn.init.kaiming_normal_ (ResNet._initialize_weights) produces and PACT
clip ranges stay at whatever their init_clip strategy computes from those random
weights at construction time (e.g. "max" computes off the actual random weight
values -- no training/calibration data required for a structurally valid, if
numerically meaningless, export).

Usage (inside the container):
    python3 export_resnet_random.py --config-key ResNet20 --config-json config.json --name ImageClassification_heavy --out-dir /tmp/resnet_heavy_export
    python3 export_resnet_random.py --config-key ResNet8Tiny --config-json config.tiny2.json --name ImageClassification_small --out-dir /tmp/resnet_small_export
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/app/Deeploy/quantlab")
sys.path.insert(0, "/app/Deeploy/quantlab/examples/fx_integerization")

from integerize_pactnets import _QUANT_UTILS  # noqa: E402
from quantlib.backends.dory import export_net  # noqa: E402

_RESNET_DIR = Path("/app/Deeploy/quantlab/systems/CIFAR10/ResNet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-key", type=str, required=True, help="e.g. ResNet20, ResNet8Tiny (the _CONFIGS key in resnet.py)")
    ap.add_argument("--config-json", type=str, required=True, help="e.g. config.json, config.tiny2.json (in quantlab/systems/CIFAR10/ResNet/)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    key = "ResNetCIFAR"
    qu = _QUANT_UTILS[key]

    with open(_RESNET_DIR / args.config_json) as f:
        cfg = json.load(f)

    net_cfg = dict(cfg["network"]["kwargs"])
    quant_cfg = cfg["network"]["quantize"]["kwargs"]

    # Some config.*.json files (e.g. config.json/ResNet20) set harmonize kwargs
    # ('infer_outact'/'infer_sign') that this installed quantlib checkout's
    # _PACTActivation/PACTIntegerAdd don't accept -- a version mismatch between
    # the config file and the vendored quantlib code, not something specific to
    # random-weight export. Both are optional inference-time conveniences, not
    # core requantization behavior, so drop them rather than block the export.
    harmonize_cfg = quant_cfg.get("config", {}).get("harmonize", {})
    for unsupported_kwarg in ("infer_outact", "infer_sign"):
        if unsupported_kwarg in harmonize_cfg:
            print(f"[export_resnet_random] Dropping unsupported harmonize kwarg: {unsupported_kwarg}")
            del harmonize_cfg[unsupported_kwarg]
    # Some config.*.json files (config.json/ResNet20) omit 'learn_clip' in their
    # harmonize block -- this installed quantlib's _PACTActivation now requires it
    # as a mandatory argument. config.tiny2.json's own choice (true) is a sane
    # default: it only affects whether clip ranges keep adapting during any future
    # real training, not this export.
    if "learn_clip" not in harmonize_cfg:
        print("[export_resnet_random] harmonize config missing 'learn_clip', defaulting to true")
        harmonize_cfg["learn_clip"] = True
    # config.json (ResNet20) sets harmonize n_levels to a mixed-precision list
    # ([256, 16]) -- this installed quantlib's PACTAsymmetricAct.get_eps() only
    # supports a scalar n_levels (does plain int arithmetic on it). Normalize to
    # the first/highest-precision entry; this only affects harmonize's own
    # residual-branch-merging activation precision, not each layer's own
    # individually-specified Conv/Linear n_levels (unaffected by this).
    if isinstance(harmonize_cfg.get("n_levels"), list):
        original = harmonize_cfg["n_levels"]
        harmonize_cfg["n_levels"] = original[0]
        print(f"[export_resnet_random] harmonize n_levels {original} -> scalar {harmonize_cfg['n_levels']}")

    assert net_cfg.get("config") == args.config_key, (
        f"--config-key {args.config_key} doesn't match {args.config_json}'s own "
        f"network.kwargs.config ({net_cfg.get('config')}) -- pick the right pair."
    )

    torch.manual_seed(args.seed)
    net = qu.network(**net_cfg, seed=args.seed)
    quant_net = qu.quantize(net, **quant_cfg)
    quant_net = quant_net.eval()

    test_input = torch.randn(*qu.in_shape)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_net(quant_net, name=args.name, out_dir=str(out_dir), eps_in=qu.eps_in,
               in_data=test_input, integerize=True, D=qu.D)

    with torch.no_grad():
        logits = quant_net(test_input)
    print(f"Random-weight logits (meaningless until real weights land): {logits}")
    print(f"Exported to {out_dir}/{args.name}_ql_integerized.onnx")


if __name__ == "__main__":
    main()
