#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Export an arbitrary Conv2d/BatchNorm2d/ReLU/Linear CNN (random weights, no QuantLab
training experiment needed) to a Deeploy-style INT8 ONNX -- the Conv2d-capable
counterpart to create_deeploy_int8_test_from_pytorch.py, which only PACT-wraps
Linear/ReLU (MLP-only: its own all_pact_create_configs_int8/all_pact_f2f_recipe
type2rule dict has no "Conv2d" entry).

Reuses the exact same building blocks as export_resnet_random.py wherever they're
already generic (calibrate_pact_clip_bounds, quantlib.backends.dory.export_net),
but does NOT depend on a QuantLab-registered architecture's own pact_recipe (which
is keyed to that architecture's specific module names, e.g. ResNet's "pilot.0$") --
instead builds a generic PACT-wrapping config/rule set that matches by MODULE TYPE
(any Conv2d/Linear/ReLU, wherever it appears), the same way
create_deeploy_int8_test_from_pytorch.py's own Linear/ReLU-only version works.

BatchNorm2d must be folded into its preceding Conv2d BEFORE PACT-wrapping (PACT has
no notion of BatchNorm; QuantLab's own ResNet recipe relies on its harmonize config
to do this implicitly for a registered architecture -- there is no equivalent here),
via the standard torch.nn.utils.fusion.fuse_conv_bn_eval, applied generically to
every adjacent (Conv2d, BatchNorm2d) child pair, recursively.

Usage (inside the container):
    python3 export_generic_cnn_random.py --model-path models/keyword_spotting.py \
        --model-class KSHeavy --input-shapes 1,1,49,10 --name KeywordSpotting_heavy \
        --out-dir /tmp/ks_heavy_export --seed 42
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

sys.path.insert(0, str(Path(__file__).parent))
from create_deeploy_int8_test_from_pytorch import (  # noqa: E402
    load_model_from_file, calibrate_pact_clip_bounds, compute_symmetric_input_eps,
    quantize_input_i8, generate_random_float_inputs, parse_shape,
)

import quantlib.editing.lightweight as qlw  # noqa: E402
from quantlib.backends.dory import export_net  # noqa: E402


def fold_all_bn(module: nn.Module) -> None:
    """Fold every adjacent (Conv1d/Conv2d, BatchNorm1d/BatchNorm2d) child pair in
    module._modules (in declaration order -- matches attribute assignment order for
    hand-written blocks, and numeric '0','1',... order for nn.Sequential), then
    recurse into every child regardless of whether it was itself touched, so nested
    blocks (e.g. a DSConvBlock/DWSepBlock inside an nn.Sequential) get folded too."""
    names = list(module._modules.keys())
    i = 0
    while i < len(names) - 1:
        n1, n2 = names[i], names[i + 1]
        m1, m2 = module._modules[n1], module._modules[n2]
        if isinstance(m1, (nn.Conv1d, nn.Conv2d)) and isinstance(m2, (nn.BatchNorm1d, nn.BatchNorm2d)):
            fused = fuse_conv_bn_eval(m1, m2)
            module._modules[n1] = fused
            module._modules[n2] = nn.Identity()
            i += 2
            continue
        i += 1
    for child in module.children():
        fold_all_bn(child)


def all_pact_create_configs_cnn(network: nn.Module) -> dict:
    """Generic-by-type PACT config, extending create_deeploy_int8_test_from_pytorch.py's
    own Linear/ReLU-only version with Conv2d -- everything else (defaults) matches it
    exactly for consistency."""
    lwg = qlw.LightweightGraph(network)
    conv_nodes = {n.name for n in lwg.nodes_list if n.module.__class__.__name__ in ("Conv1d", "Conv2d")}
    linear_nodes = {n.name for n in lwg.nodes_list if n.module.__class__.__name__ == "Linear"}
    relu_nodes = {n.name for n in lwg.nodes_list if n.module.__class__.__name__ == "ReLU"}

    from collections import defaultdict

    conv_default = {
        "quantize": "per_layer", "init_clip": "sawb_asymm", "learn_clip": False,
        "symm_wts": True, "tqt": False, "n_levels": 256,
    }
    linear_default = dict(conv_default)
    relu_default = {
        "init_clip": "std", "learn_clip": True, "nb_std": 3, "rounding": False,
        "tqt": False, "n_levels": 256,
    }

    conv_config = defaultdict(lambda: conv_default.copy())
    for n in conv_nodes:
        conv_config[n]
    linear_config = defaultdict(lambda: linear_default.copy())
    for n in linear_nodes:
        linear_config[n]
    relu_config = defaultdict(lambda: relu_default.copy())
    for n in relu_nodes:
        relu_config[n]

    return {**conv_config, **linear_config, **relu_config}


def all_pact_f2f_recipe_cnn(network: nn.Module, name2config: dict) -> nn.Module:
    """Generic-by-type PACT wrapping, extending create_deeploy_int8_test_from_pytorch.py's
    own version with a Conv2d rule (Conv1d mapped too, for completeness, though unused
    by KS/VWW today)."""
    import re

    lwg = qlw.LightweightGraph(network)
    name2type = {n.name: n.module.__class__.__name__ for n in lwg.nodes_list}

    assert set(name2config.keys()).issubset(set(name2type.keys()))

    type2rule = {
        "Conv1d": qlw.rules.pact.ReplaceConvLinearPACTRule,
        "Conv2d": qlw.rules.pact.ReplaceConvLinearPACTRule,
        "Linear": qlw.rules.pact.ReplaceConvLinearPACTRule,
        "ReLU": qlw.rules.pact.ReplaceActPACTRule,
    }

    rhos = [
        type2rule[name2type[n]](qlw.rules.NameFilter(re.escape(n) + "$"), **name2config[n])
        for n in name2config.keys()
    ]

    lwe = qlw.LightweightEditor(lwg)
    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()

    return lwe.graph.net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--model-class", type=str, required=True)
    ap.add_argument("--input-shapes", type=str, required=True, help="e.g. 1,3,96,96")
    ap.add_argument("--input-range", nargs=2, type=float, default=[-1.0, 1.0])
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=2**18)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shape_in = parse_shape(args.input_shapes)

    float_model = load_model_from_file(args.model_path, args.model_class, None)
    float_model.eval()

    # PyTorch's nn.BatchNorm2d.bias defaults to EXACTLY 0.0 for every layer. An
    # integerized zero bias stays exactly zero regardless of that layer's own
    # scale/eps (0 * anything = 0), so every layer's RequantShift "add" constant
    # ends up bit-identical -- torch.onnx.export/onnx_graphsurgeon then dedupe
    # them into ONE shared tensor reused by every RequantShift node in the graph.
    # That single-tensor-multiple-consumers shape corrupts Deeploy's own
    # topology-optimization/parsing search badly enough that EVERY layer fails to
    # parse (confirmed: each node parses fine in isolation, but the full-graph
    # search still reports "backtracking exhausted at root"). Perturbing each
    # BatchNorm2d's bias slightly at init -- harmless for a random-weight/
    # hardware-metrics-only export -- gives every layer a distinct add constant.
    for m in float_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and m.bias is not None:
            with torch.no_grad():
                m.bias.add_(torch.randn_like(m.bias) * 0.01)
    # NOT folding BatchNorm2d into the preceding Conv2d here: quantlib's own
    # IntegerizePACTNetPass integrates a bias-free PACTConv2d -> BatchNorm2d ->
    # PACTAct sequence internally (confirmed via ResNet, which never gets BN-folded
    # externally either) -- integrate_pact_conv_fun explicitly asserts the conv has
    # NO bias, which torch.nn.utils.fusion.fuse_conv_bn_eval would otherwise
    # introduce (it creates a bias term to absorb BN's offset even when the
    # original conv had none), breaking that assumption.

    calib_list = generate_random_float_inputs([shape_in], tuple(args.input_range), np.float32)
    calib_batch = torch.from_numpy(calib_list[0])
    input_eps = compute_symmetric_input_eps(calib_batch.numpy(), n_levels=256)

    name2config = all_pact_create_configs_cnn(float_model)
    pact_model = all_pact_f2f_recipe_cnn(float_model, name2config)
    pact_model.eval()
    calibrate_pact_clip_bounds(pact_model, calib_batch)

    x_i8 = quantize_input_i8(calib_list[0][:1], input_eps)
    test_input = torch.from_numpy(x_i8.astype(np.float32))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Saved so downstream create_deeploy_int_io_from_onnx.py can be given THIS
    # already-int8-domain sample via --input-npz, instead of drawing a fresh
    # --input-range sample of its own: that generic path draws raw floats in the
    # given range with no quantization step, which for a range like [-1,1] never
    # produces integer-valued int8-domain data at all -- Deeploy's own frontend
    # then infers the input type as float32_t instead of int8_t (confirmed via its
    # -v -v -v --debug diagnostic path, which types inputs by inspecting whether
    # they're actually integer-valued) and every downstream INT8 layer parser
    # rejects the graph ("backtracking exhausted"), even though every layer parses
    # fine individually. AnomalyDetection/ImageClassification never hit this
    # because their own --input-npz files happened to already be int8-quantized
    # (a real image / a properly quantize_input_i8'd sample).
    np.savez(out_dir / f"{args.name}_test_input_i8.npz", input_0=x_i8)

    export_net(pact_model, name=args.name, out_dir=str(out_dir), eps_in=input_eps,
               in_data=test_input, integerize=True, D=args.D)

    with torch.no_grad():
        logits = pact_model(test_input)
    print(f"Random-weight logits (meaningless): {logits}")
    print(f"Exported to {out_dir}/{args.name}_ql_integerized.onnx")


if __name__ == "__main__":
    main()
