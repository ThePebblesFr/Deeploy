#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate a genuinely INT8 Deeploy test (network.onnx + inputs.npz + outputs.npz)
directly from a PyTorch model description, WITHOUT going through QuantLib.

Why this script exists
-----------------------
Deeploy does not infer a tensor's bit width from the ONNX tensor's declared
elem_type (it is always TensorProto.FLOAT, even in "real" INT8 Deeploy test
graphs -- see DeeployTest/Tests/simpleRegression/network.onnx). Instead, an
INT8 kernel gets bound only when a `Gemm`/`Conv`/`MatMul` node is immediately
followed by a custom `RequantShift` node (domain "PACTOps") carrying `div`
(power-of-two divisor), `n_levels_out` and `signed` attributes. Deeploy's own
PULPGEMMRequantMergePass / PULPConvRequantMergePass then fuse that pair into
`RequantizedGemm` / `RequantizedConv`, which is what actually binds to the
integer pulp-nn kernels. A plain `Gemm` (even with integer-valued float32
weights) or a fake-quant Div/Round/Clip/QDQ chain WITHOUT an adjacent
RequantShift node is exactly what falls back to FP32 kernels -- this is the
trap the QuantLib-based flow in create_deeploy_int8_test_from_pytorch.py
tends to fall into (it also hand-builds a "RequantizedGemm" node directly,
which is actually an internal fusion-pass output, not a valid input-level op,
so its attribute/bias layout does not match what Deeploy expects).

This script sidesteps all of that: it quantizes the model itself (simple
per-tensor symmetric INT8 calibration), and emits the exact
`Gemm(+bias) -> RequantShift(PACTOps)` pattern that the existing, known-good
test graphs (DeeployTest/Tests/simpleRegression, .../test2DRequantizedConv)
use, then computes the golden reference output by replicating Deeploy's own
RequantShift arithmetic (see TargetLibraries/Generic/src/RequantShift_s8.c)
in numpy so outputs.npz matches bit-exactly what the C kernel will produce.

Scope: currently supports `nn.Linear` (+ optional `nn.ReLU`) chains only,
i.e. plain MLPs such as ES_DNN1/ES_DNN2/NoFS_DNN1/SimpleMLP. Conv2d support
would need per-channel BN-style folding and NHWC layout handling and is not
implemented here.

Usage:
    python3 create_deeploy_int8_native.py \
        --model-path models/es_dnn1.py \
        --model-class ES_DNN1 \
        --test-name ES_DNN1_int8_native \
        --input-shape 1,16 \
        --seed 42 \
        --verbose
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise RuntimeError("PyTorch is required. Install torch first.") from e


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model_from_file(model_path: str, model_class: str, model_kwargs: Optional[Dict] = None) -> nn.Module:
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {model_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, model_class):
        raise RuntimeError(f"Class '{model_class}' not found in {model_path}")

    ModelClass = getattr(module, model_class)
    model = ModelClass(**(model_kwargs or {}))
    model.eval()
    return model


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in shape_str.split(","))
    except ValueError as e:
        raise ValueError(f"Invalid shape string '{shape_str}': {e}")


# -----------------------------------------------------------------------------
# Discover the Linear(+ReLU) chain in forward-execution order
# -----------------------------------------------------------------------------

class LinearStage:

    def __init__(self, name: str, weight: np.ndarray, bias: np.ndarray, relu_after: bool):
        self.name = name
        self.weight = weight  # float32, shape [out_features, in_features]
        self.bias = bias  # float32, shape [out_features]
        self.relu_after = relu_after


def discover_linear_chain(model: nn.Module, example_input: torch.Tensor) -> List[LinearStage]:
    """
    Runs one forward pass with hooks on every leaf module to recover the
    execution order, then folds each nn.Linear with an immediately-following
    nn.ReLU (if present) into a LinearStage.
    """
    order: List[Tuple[str, nn.Module]] = []

    def make_hook(name):

        def hook(module, inputs, output):
            order.append((name, module))

        return hook

    hooks = []
    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(example_input)

    for h in hooks:
        h.remove()

    stages: List[LinearStage] = []
    i = 0
    while i < len(order):
        name, module = order[i]
        if isinstance(module, nn.Linear):
            relu_after = (i + 1 < len(order)) and isinstance(order[i + 1][1], nn.ReLU)
            weight = module.weight.detach().cpu().numpy().astype(np.float64)
            if module.bias is not None:
                bias = module.bias.detach().cpu().numpy().astype(np.float64)
            else:
                bias = np.zeros(weight.shape[0], dtype=np.float64)
            stages.append(LinearStage(name, weight, bias, relu_after))
            i += 2 if relu_after else 1
        elif isinstance(module, nn.ReLU):
            # Standalone ReLU not directly following a Linear: unsupported.
            raise RuntimeError(
                f"Unsupported module sequence: ReLU '{name}' does not immediately follow a Linear layer. "
                "This script only supports Linear(+ReLU) chains."
            )
        else:
            if list(module.parameters(recurse=False)) or not isinstance(module, (nn.Identity, nn.Flatten)):
                # Allow harmless shape-only ops (Flatten/Identity); reject anything else with weights.
                if not isinstance(module, (nn.Identity, nn.Flatten)):
                    raise RuntimeError(
                        f"Unsupported module '{name}' of type {type(module).__name__}. "
                        "This script only supports nn.Linear and nn.ReLU."
                    )
            i += 1

    if not stages:
        raise RuntimeError("No nn.Linear layers found in the model.")

    return stages


# -----------------------------------------------------------------------------
# Calibration (float) forward pass to gather per-stage activation ranges
# -----------------------------------------------------------------------------

def calibrate(stages: List[LinearStage], calib_inputs: np.ndarray) -> Tuple[float, List[float]]:
    """
    Runs the float-domain forward pass through the discovered stages using
    plain numpy, and returns:
      - input_absmax: max(abs(x)) over calib_inputs
      - stage_absmax: for each stage, max(abs(.)) of its *output* (post-ReLU
        if relu_after, else raw), i.e. the range that stage's quantized
        output must cover.
    """
    input_absmax = float(np.max(np.abs(calib_inputs)))
    if input_absmax == 0.0:
        input_absmax = 1.0

    x = calib_inputs.astype(np.float64)
    stage_absmax = []
    for stage in stages:
        x = x @ stage.weight.T + stage.bias
        if stage.relu_after:
            x = np.maximum(x, 0.0)
        absmax = float(np.max(np.abs(x)))
        if absmax == 0.0:
            absmax = 1.0
        stage_absmax.append(absmax)

    return input_absmax, stage_absmax


# -----------------------------------------------------------------------------
# Quantization: per-tensor symmetric INT8 weights/activations + RequantShift
# multiplier/shift derivation
# -----------------------------------------------------------------------------

class QuantizedStage:

    def __init__(self, weight_i8, bias_i32, mul: int, shift: int, out_signed: bool = True):
        self.weight_i8 = weight_i8
        self.bias_i32 = bias_i32
        self.mul = mul
        self.shift = shift  # log2(div)
        self.div = 2 ** shift
        self.out_signed = out_signed


def compute_mul_shift(real_multiplier: float, target_mantissa_bits: int = 14, max_shift: int = 30) -> Tuple[int, int]:
    """
    Approximates `real_multiplier` as mul / 2**shift with mul a "nice" sized
    integer (~2**target_mantissa_bits), matching the magnitude convention
    QuantLib itself emits (see e.g. simpleRegression/network.onnx: mul=514,
    div=65536). Falls back gracefully for extreme scale ratios.
    """
    if real_multiplier <= 0:
        raise ValueError("real_multiplier must be positive")

    frac, exp = math.frexp(real_multiplier)  # real_multiplier == frac * 2**exp, 0.5 <= frac < 1
    shift = target_mantissa_bits - exp
    shift = max(1, min(max_shift, shift))

    mul = int(round(real_multiplier * (2 ** shift)))
    if mul <= 0:
        mul = 1
    # Keep comfortably within int32.
    while mul >= 2 ** 30 and shift > 1:
        shift -= 1
        mul = int(round(real_multiplier * (2 ** shift)))

    return mul, shift


def quantize_stages(stages: List[LinearStage], input_absmax: float, stage_absmax: List[float],
                    weight_clip: int = 127, verbose: bool = False) -> Tuple[float, List[QuantizedStage], List[float]]:
    input_scale = input_absmax / 127.0

    quantized: List[QuantizedStage] = []
    scale_in = input_scale
    out_scales: List[float] = []

    max_accum_exponent = 0

    for idx, (stage, absmax) in enumerate(zip(stages, stage_absmax)):
        weight_scale = float(np.max(np.abs(stage.weight))) / weight_clip
        if weight_scale == 0.0:
            weight_scale = 1.0

        weight_i8 = np.clip(np.round(stage.weight / weight_scale), -weight_clip, weight_clip).astype(np.int32)
        bias_i32 = np.round(stage.bias / (scale_in * weight_scale)).astype(np.int64)

        out_scale = absmax / 127.0
        real_multiplier = (scale_in * weight_scale) / out_scale
        mul, shift = compute_mul_shift(real_multiplier)

        max_abs_accum = float(np.max(np.abs(weight_i8)).astype(np.float64)) * 127.0 * weight_i8.shape[1] + \
            float(np.max(np.abs(bias_i32)))
        max_accum_exponent = max(max_accum_exponent, int(math.log2(max(max_abs_accum, 1))) + 1)

        if verbose:
            print(f"[INFO] stage {idx} ({stage.name}): in_scale={scale_in:.6g} w_scale={weight_scale:.6g} "
                  f"out_scale={out_scale:.6g} mul={mul} shift={shift} relu_after={stage.relu_after}")

        quantized.append(QuantizedStage(weight_i8, bias_i32, mul, shift, out_signed=True))
        out_scales.append(out_scale)
        scale_in = out_scale

    if max_accum_exponent > 24 and verbose:
        print(f"[WARNING] worst-case accumulator needs ~{max_accum_exponent} bits, "
              "which exceeds float32's 24-bit exact-integer mantissa. Values may not "
              "round-trip exactly through the float32 ONNX container for pathological inputs.")

    return input_scale, quantized, out_scales


# -----------------------------------------------------------------------------
# Golden integer reference, replicating Deeploy's RequantShift arithmetic
# exactly (TargetLibraries/Generic/src/RequantShift_s8.c):
#   intermediate = x * mul + add
#   if log2D > 0: intermediate += 1 << (log2D - 1)      # round-half-up
#   intermediate = intermediate >> log2D
#   out = clip(intermediate, -128, 127)                  # signed, n_levels=256
# -----------------------------------------------------------------------------

def integer_forward(x_i8: np.ndarray, quantized: List[QuantizedStage]) -> List[np.ndarray]:
    activations = [x_i8.astype(np.int64)]

    current = x_i8.astype(np.int64)
    for q in quantized:
        acc = current @ q.weight_i8.astype(np.int64).T + q.bias_i32.astype(np.int64)
        y = acc * np.int64(q.mul)
        if q.shift > 0:
            y = y + np.int64(1 << (q.shift - 1))
        y = y >> np.int64(q.shift)
        y = np.clip(y, -128, 127)
        activations.append(y)
        current = y

    return activations


# -----------------------------------------------------------------------------
# ONNX graph construction: Gemm(+bias) -> RequantShift(PACTOps) chain
# -----------------------------------------------------------------------------

def tensor_attr_f32(name: str, value) -> onnx.TensorProto:
    arr = np.array(value, dtype=np.float32)
    return numpy_helper.from_array(arr, name=name)


def build_onnx_graph(onnx_path: Path, input_shape: Tuple[int, ...], quantized: List[QuantizedStage],
                     opset_version: int = 13) -> None:
    if len(input_shape) != 2:
        raise ValueError(f"Expected a rank-2 input shape [batch, features], got {input_shape}")

    nodes = []
    initializers = []
    value_infos = []

    prev = "input_0"
    graph_inputs = [helper.make_tensor_value_info("input_0", TensorProto.FLOAT, list(input_shape))]

    batch = input_shape[0]

    for i, q in enumerate(quantized):
        w_name = f"layer{i}_weight"
        b_name = f"layer{i}_bias"
        mul_name = f"layer{i}_mul"
        add_name = f"layer{i}_add"

        out_features = int(q.weight_i8.shape[0])

        gemm_out = f"layer{i}_gemm_out"
        rqs_out = "output_0" if i == len(quantized) - 1 else f"layer{i}_out"

        initializers.append(numpy_helper.from_array(q.weight_i8.astype(np.float32), w_name))
        initializers.append(numpy_helper.from_array(q.bias_i32.astype(np.float32), b_name))
        initializers.append(numpy_helper.from_array(np.array(q.mul, dtype=np.float32), mul_name))
        initializers.append(numpy_helper.from_array(np.array(0, dtype=np.float32), add_name))

        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[prev, w_name, b_name],
                outputs=[gemm_out],
                name=f"Gemm_{i}",
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=1,
            ))
        # Deeploy does not perform shape inference itself: every intermediate
        # tensor needs an explicit shape annotation.
        value_infos.append(helper.make_tensor_value_info(gemm_out, TensorProto.FLOAT, [batch, out_features]))

        nodes.append(
            helper.make_node(
                "RequantShift",
                inputs=[gemm_out, mul_name, add_name],
                outputs=[rqs_out],
                name=f"RequantShift_{i}",
                domain="PACTOps",
                div=tensor_attr_f32(f"layer{i}_div_attr", float(q.div)),
                n_levels_out=tensor_attr_f32(f"layer{i}_n_levels_attr", [256.0]),
                signed=tensor_attr_f32(f"layer{i}_signed_attr", [1.0]),
            ))
        if rqs_out != "output_0":
            value_infos.append(helper.make_tensor_value_info(rqs_out, TensorProto.FLOAT, [batch, out_features]))

        prev = rqs_out

    out_features = quantized[-1].weight_i8.shape[0]
    graph_outputs = [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, [batch, out_features])]

    graph = helper.make_graph(
        nodes=nodes,
        name="Deeploy_INT8_MLP",
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializer=initializers,
        value_info=value_infos,
    )

    model = helper.make_model(
        graph,
        producer_name="create_deeploy_int8_native.py",
        opset_imports=[
            helper.make_opsetid("", opset_version),
            helper.make_opsetid("PACTOps", 1),
        ],
    )
    model.ir_version = 8

    # Do NOT run onnx.checker.check_model: RequantShift/PACTOps is a Deeploy custom op.
    onnx.save(model, str(onnx_path))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a native (non-QuantLib) INT8 Deeploy test from a PyTorch Linear(+ReLU) model.")

    parser.add_argument("--model-path", type=str, required=True, help="Python file containing the model class.")
    parser.add_argument("--model-class", type=str, required=True)
    parser.add_argument("--model-kwargs", type=str, default=None, help="JSON kwargs for --model-class.")

    parser.add_argument("--test-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="DeeployTest/Tests")
    parser.add_argument("--input-shape", type=str, required=True, help="e.g. '1,16'")
    parser.add_argument("--input-range", nargs=2, type=float, default=[-1.0, 1.0])
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--opset-version", type=int, default=13)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    input_shape = parse_shape(args.input_shape)
    if len(input_shape) != 2:
        raise RuntimeError("This script supports rank-2 (batch, features) inputs only.")

    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None
    model = load_model_from_file(args.model_path, args.model_class, model_kwargs)

    test_dir = Path(args.output_dir) / args.test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1) Discover the Linear(+ReLU) chain.
    example_input = torch.zeros(input_shape, dtype=torch.float32)
    stages = discover_linear_chain(model, example_input)
    if args.verbose:
        print(f"[INFO] discovered {len(stages)} Linear stage(s):")
        for s in stages:
            print(f"       {s.name}: {s.weight.shape} relu_after={s.relu_after}")

    # 2) Calibrate on random float inputs in the target range to derive scales.
    lo, hi = args.input_range
    calib_inputs = np.random.uniform(lo, hi, size=(args.calibration_samples, input_shape[1]))
    input_absmax, stage_absmax = calibrate(stages, calib_inputs)

    # 3) Quantize weights/bias and compute RequantShift mul/shift per stage.
    input_scale, quantized, out_scales = quantize_stages(stages, input_absmax, stage_absmax, verbose=args.verbose)

    # 4) Build the actual INT8 test input (fresh sample, same range/scale as calibration).
    x_float = np.random.uniform(lo, hi, size=input_shape)
    x_i8 = np.clip(np.round(x_float / input_scale), -128, 127).astype(np.int64)

    # 5) Golden integer reference output, replicating Deeploy's RequantShift kernel exactly.
    activations = integer_forward(x_i8, quantized)
    y_i8 = activations[-1]

    # 6) Build the Deeploy-ready ONNX graph.
    network_path = test_dir / "network.onnx"
    build_onnx_graph(network_path, input_shape, quantized, opset_version=args.opset_version)

    # 7) Save inputs.npz / outputs.npz.
    np.savez(test_dir / "inputs.npz", input_0=x_i8.astype(np.int64))
    np.savez(test_dir / "outputs.npz", output_0=y_i8.astype(np.int64))

    # Sanity check against the float reference (informational only).
    if args.verbose:
        x_float_dequant = x_i8.astype(np.float64) * input_scale
        y_float_ref = x_float_dequant
        for stage in stages:
            y_float_ref = y_float_ref @ stage.weight.T + stage.bias
            if stage.relu_after:
                y_float_ref = np.maximum(y_float_ref, 0.0)
        y_dequant = y_i8.astype(np.float64) * out_scales[-1]
        rel_err = np.max(np.abs(y_dequant - y_float_ref)) / (np.max(np.abs(y_float_ref)) + 1e-9)
        print(f"[INFO] max relative error vs float model: {rel_err:.4%}")
        print(f"[INFO] input dtype/range: int8 in [{x_i8.min()}, {x_i8.max()}]")
        print(f"[INFO] output dtype/range: int8 in [{y_i8.min()}, {y_i8.max()}]")

    readme = f"""# {args.test_name}

Generated by `create_deeploy_int8_native.py` (no QuantLib dependency).

## Files

- `network.onnx`: `Gemm -> RequantShift` (domain `PACTOps`) chain, float32
  container tensors holding exact integer values -- matches the convention
  used by Deeploy's own known-good INT8 tests (see
  `DeeployTest/Tests/simpleRegression/network.onnx`). Deeploy's
  `PULPGEMMRequantMergePass` fuses each `Gemm+RequantShift` pair into a
  `RequantizedGemm` bound to an integer pulp-nn kernel.
- `inputs.npz` / `outputs.npz`: `input_0` / `output_0`, INT8-valued (stored
  as int64), computed by replicating Deeploy's own RequantShift arithmetic
  (`TargetLibraries/Generic/src/RequantShift_s8.c`) in numpy.

## Quantization

- Input scale: `{input_scale}`
- Per-stage (mul, shift=log2(div), out_scale):
{chr(10).join(f"  - stage {i}: mul={q.mul}, shift={q.shift}, out_scale={s}" for i, (q, s) in enumerate(zip(quantized, out_scales)))}

## Run

```bash
cd DeeployTest
python3 testRunner_tiled_siracusa.py -t Tests/{args.test_name} --cores 8 --l1 64000
```
"""
    (test_dir / "README.md").write_text(readme)

    print("\n✓ Native INT8 Deeploy test generation complete")
    print(f"Test directory: {test_dir}")


if __name__ == "__main__":
    main()
