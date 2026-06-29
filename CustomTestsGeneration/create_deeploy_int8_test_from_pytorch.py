#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Create a Deeploy INT8 test from a PyTorch model using the same high-level flow as
create_deeploy_test.py, but inserting QuantLib PACT integerization before exporting.

The important difference from a raw torch.onnx.export(tq_pact_model, ...) is that this
script can build a Deeploy-oriented ONNX using custom RequantizedGemm nodes instead of
exporting the fake-quant arithmetic as Gemm -> Mul -> Add -> Div -> Floor -> Clip.

Typical ES_DNN1 usage from a PyTorch model file:
    python3 create_deeploy_int8_test_from_pytorch.py \
        --model-path models/es_dnn1.py \
        --model-class PlainMLP \
        --test-name ES_DNN1_int8 \
        --input-shapes "1,16" \
        --output-dir DeeployTest/Tests \
        --seed 42 \
        --verbose

Typical ES_DNN1 usage from an existing float ONNX:
    python3 create_deeploy_int8_test_from_pytorch.py \
        --source-onnx models/ES-DNN1.onnx \
        --builtin-arch ES_DNN1 \
        --test-name ES_DNN1_int8 \
        --input-shapes "1,16" \
        --output-dir DeeployTest/Tests \
        --seed 42 \
        --verbose

Dependencies inside the Deeploy/QuantLab environment:
    pip install numpy onnx torch onnx2pytorch
    # plus QuantLib/QuantLab, as in your notebook
"""

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise RuntimeError("PyTorch is required. Install torch first.") from e

try:
    import quantlib.algorithms as qa
    import quantlib.editing.fx as qfx
    import quantlib.editing.lightweight as qlw
except ImportError as e:
    raise RuntimeError(
        "QuantLib is required for this script. Install it as in your notebook, e.g. "
        "clone pulp-platform/quantlab and install quantlab/quantlib."
    ) from e


# -----------------------------------------------------------------------------
# Built-in ES_DNN architectures from your notebook
# -----------------------------------------------------------------------------

class ES_DNN1(nn.Module):
    def __init__(self, with_softmax: bool = False):
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
        if with_softmax:
            layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ES_DNN2(nn.Module):
    def __init__(self, with_softmax: bool = False):
        super().__init__()
        layers = [
            nn.Linear(16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 704),
            nn.ReLU(),
            nn.Linear(704, 288),
            nn.ReLU(),
            nn.Linear(288, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ]
        if with_softmax:
            layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


BUILTINS = {
    "ES_DNN1": ES_DNN1,
    "ES_DNN2": ES_DNN2,
}


# -----------------------------------------------------------------------------
# Generic helpers copied/adapted from the original Deeploy generator
# -----------------------------------------------------------------------------

def parse_shape(shape_str: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in shape_str.split(","))
    except ValueError as e:
        raise ValueError(f"Invalid shape string '{shape_str}': {e}")


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


def load_float_onnx_into_builtin(source_onnx: str, builtin_arch: str, device: torch.device) -> nn.Module:
    """
    Same idea as the notebook:
      ConvertModel(onnx) -> copy Linear weights into a clean PlainMLP.
    """
    try:
        from onnx2pytorch import ConvertModel
    except ImportError as e:
        raise RuntimeError("onnx2pytorch is required when using --source-onnx.") from e

    if builtin_arch not in BUILTINS:
        raise ValueError(f"Unknown builtin arch: {builtin_arch}. Choices: {list(BUILTINS)}")

    onnx_model = onnx.load(source_onnx)
    converted = ConvertModel(onnx_model).to(device)
    converted.eval()

    clean = BUILTINS[builtin_arch]().to(device)
    clean.eval()

    src_linears = [m for m in converted.modules() if isinstance(m, nn.Linear)]
    dst_linears = [m for m in clean.modules() if isinstance(m, nn.Linear)]

    if len(src_linears) != len(dst_linears):
        raise RuntimeError(f"Linear layer mismatch: converted={len(src_linears)}, builtin={len(dst_linears)}")

    for i, (src, dst) in enumerate(zip(src_linears, dst_linears)):
        if tuple(src.weight.shape) != tuple(dst.weight.shape):
            raise RuntimeError(f"Layer {i}: weight shape mismatch {tuple(src.weight.shape)} vs {tuple(dst.weight.shape)}")
        dst.weight.data.copy_(src.weight.data)
        if src.bias is not None and dst.bias is not None:
            dst.bias.data.copy_(src.bias.data)

    return clean


def generate_random_float_inputs(
    input_shapes: List[Tuple[int, ...]],
    input_range: Tuple[float, float],
    dtype: np.dtype = np.float32,
) -> List[np.ndarray]:
    low, high = input_range
    return [np.random.uniform(low, high, size=shape).astype(dtype) for shape in input_shapes]


def save_deeploy_npz(test_dir: Path, inputs: List[np.ndarray], outputs: List[np.ndarray]):
    # Matches the original create_deeploy_test.py convention: input_0, output_0, ...
    np.savez(test_dir / "inputs.npz", **{f"input_{i}": x for i, x in enumerate(inputs)})
    np.savez(test_dir / "outputs.npz", **{f"output_{i}": y for i, y in enumerate(outputs)})


# -----------------------------------------------------------------------------
# QuantLib PACT fake-quant and fake-to-true conversion, adapted from the notebook
# -----------------------------------------------------------------------------

def all_pact_f2f_recipe(network: nn.Module, name2config: Dict[str, Dict]) -> nn.Module:
    lwg = qlw.LightweightGraph(network)
    name2type = {n.name: n.module.__class__.__name__ for n in lwg.nodes_list}

    assert set(name2config.keys()).issubset(set(name2type.keys()))

    type2rule = {
        "Linear": qlw.rules.pact.ReplaceConvLinearPACTRule,
        "ReLU": qlw.rules.pact.ReplaceActPACTRule,
    }

    rhos = [
        type2rule[name2type[n]](qlw.rules.NameFilter(n), **name2config[n])
        for n in name2config.keys()
    ]

    lwe = qlw.LightweightEditor(lwg)
    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()

    return lwe.graph.net


def all_pact_create_configs_int8(network: nn.Module, patches: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
    patches = patches or {}

    lwg = qlw.LightweightGraph(network)

    linear_nodes = {n.name for n in lwg.nodes_list if n.module.__class__.__name__ == "Linear"}
    relu_nodes = {n.name for n in lwg.nodes_list if n.module.__class__.__name__ == "ReLU"}

    assert set(patches.keys()).issubset(linear_nodes | relu_nodes)

    linear_default = {
        "quantize": "per_layer",
        "init_clip": "sawb_asymm",
        "learn_clip": False,
        "symm_wts": True,
        "tqt": False,
        "n_levels": 256,
    }

    relu_default = {
        "init_clip": "std",
        "learn_clip": True,
        "nb_std": 3,
        "rounding": False,
        "tqt": False,
        "n_levels": 256,
    }

    linear_config = defaultdict(lambda: linear_default.copy())
    for n in linear_nodes:
        linear_config[n].update(patches[n] if n in patches else {})

    relu_config = defaultdict(lambda: relu_default.copy())
    for n in relu_nodes:
        relu_config[n].update(patches[n] if n in patches else {})

    return {**linear_config, **relu_config}


def compute_symmetric_input_eps(x_float: np.ndarray, n_levels: int = 256) -> float:
    """
    Notebook-equivalent symmetric input quantization scale.
    For signed int8, this is effectively max(abs(x)) / 127.
    """
    max_abs = float(np.max(np.abs(x_float)))
    if max_abs == 0.0:
        return 1.0
    # For 256 signed levels: -128..127. Use 127 for symmetric quant.
    return max_abs / ((n_levels // 2) - 1)


def quantize_input_i8(x_float: np.ndarray, eps: float) -> np.ndarray:
    x_q = np.rint(x_float / eps)
    x_q = np.clip(x_q, -128, 127).astype(np.int8)
    return x_q


def f2t_convert(network: nn.Module, x_q_float: torch.Tensor, input_eps: float, D: int = 2**18) -> nn.Module:
    network.eval()
    network = network.to(device=torch.device("cpu"))

    fake2true_converter = qfx.passes.pact.IntegerizePACTNetPass(
        shape_in=x_q_float.shape,
        eps_in=input_eps,
        D=D,
    )

    tq_network = fake2true_converter(network)
    tq_network.eval()
    return tq_network


# -----------------------------------------------------------------------------
# Integer reference arithmetic from the notebook
# -----------------------------------------------------------------------------

def clip_u8(x: int) -> int:
    return max(0, min(255, int(x)))


def clip_i8(x: int) -> int:
    return max(-128, min(127, int(x)))


def pulp_nn_quant(phi: int, m: int, d: int, signed: bool) -> int:
    phi_i32 = np.int32(phi)
    m_i16 = np.int16(m)

    prod_i64 = np.int64(m_i16) * np.int64(phi_i32)
    prod_i32 = np.int32(prod_i64)
    x_i32 = np.int32(prod_i32 >> np.int8(d))

    return clip_i8(int(x_i32)) if signed else clip_u8(int(x_i32))


def fc_int(
    x,
    w_i8,
    b_i32,
    out_mult: int,
    out_shift: int,
    input_signed: bool,
    output_signed: bool,
) -> np.ndarray:
    x_dtype = np.int8 if input_signed else np.uint8
    y_dtype = np.int8 if output_signed else np.uint8

    x = np.asarray(x, dtype=x_dtype).reshape(-1)
    w_i8 = np.asarray(w_i8, dtype=np.int8)
    b_i32 = np.asarray(b_i32, dtype=np.int32).reshape(-1)

    y = np.zeros(w_i8.shape[0], dtype=y_dtype)
    for i in range(w_i8.shape[0]):
        phi = int(b_i32[i]) + int(np.dot(w_i8[i].astype(np.int64), x.astype(np.int64)))
        y[i] = pulp_nn_quant(phi, int(out_mult), int(out_shift), signed=output_signed)

    return y


def extract_linear_params(linear_module: nn.Linear) -> Tuple[np.ndarray, np.ndarray]:
    w = linear_module.weight.detach().cpu().numpy()
    b = linear_module.bias.detach().cpu().numpy()
    return np.rint(w).astype(np.int8), np.rint(b).astype(np.int32)


def collect_linear_layers_in_execution_order(model: nn.Module, x: torch.Tensor) -> List[nn.Linear]:
    executed = []

    def make_hook(name):
        def hook(module, inputs, output):
            executed.append((name, module))
        return hook

    root = model.net if hasattr(model, "net") else model
    hooks = []
    for name, module in root.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return [module for _, module in executed if isinstance(module, nn.Linear)]


def collect_requant_modules(model: nn.Module) -> List[nn.Module]:
    modules = []
    for _, m in model.named_modules():
        if hasattr(m, "mul") and hasattr(m, "div") and m.__class__.__name__.lower().find("requant") >= 0:
            modules.append(m)

    if not modules:
        # QuantLib integerization pass often names these modules with INTEGERIZE_UNSIGNED_ACT_PASS.
        for name, m in model.named_modules():
            if "INTEGERIZE_UNSIGNED_ACT_PASS" in name and hasattr(m, "mul") and hasattr(m, "div"):
                modules.append(m)

    return modules


def get_requant_params(tq_model: nn.Module, n_linears: int, final_signed: bool) -> Tuple[List[int], List[int], List[bool]]:
    requant_modules = collect_requant_modules(tq_model)

    out_mult = [int(m.mul.detach().cpu().item()) if torch.is_tensor(m.mul) else int(m.mul) for m in requant_modules]
    out_shift = [
        int(round(math.log2(int(m.div.detach().cpu().item() if torch.is_tensor(m.div) else m.div))))
        for m in requant_modules
    ]

    # The PACT true network normally has one requant after each hidden layer, not after the final FC.
    # To obtain an INT8-only Deeploy graph, we add an identity-ish final requant by default.
    while len(out_mult) < n_linears:
        out_mult.append(1)
        out_shift.append(0)

    # First hidden outputs are unsigned activations. Final output is configurable; default signed int8.
    out_signed = [False] * (n_linears - 1) + [final_signed]

    return out_mult[:n_linears], out_shift[:n_linears], out_signed[:n_linears]


def integer_forward(
    x_i8: np.ndarray,
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    out_mult: List[int],
    out_shift: List[int],
    out_signed: List[bool],
) -> List[np.ndarray]:
    outputs = [x_i8.reshape(1, -1)]

    current = x_i8.reshape(-1)
    current_signed = True

    for i, (w, b) in enumerate(zip(weights, biases)):
        y = fc_int(
            current,
            w,
            b,
            out_mult=out_mult[i],
            out_shift=out_shift[i],
            input_signed=current_signed,
            output_signed=out_signed[i],
        )
        outputs.append(y.reshape(1, -1))
        current = y
        current_signed = out_signed[i]

    return outputs


# -----------------------------------------------------------------------------
# Deeploy-oriented custom ONNX generation
# -----------------------------------------------------------------------------

def tensor_attr_i64(name: str, value: int) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array([value], dtype=np.int64), name=name)


def tensor_attr_i32(name: str, value: int) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array([value], dtype=np.int32), name=name)


def make_requantized_gemm_node(
    name: str,
    A: str,
    B: str,
    C: str,
    M: str,
    Y: str,
    div: int,
    shift: int,
    n_levels: int,
    signed: bool,
    transB: int = 1,
) -> onnx.NodeProto:
    # Deeploy's PULPGEMMParser expects op RequantizedGemm, 4 inputs [A, B, C, mul],
    # and RQS attrs div/n_levels/signed plus a shift attr.
    return helper.make_node(
        "RequantizedGemm",
        inputs=[A, B, C, M],
        outputs=[Y],
        name=name,
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=transB,
        div=tensor_attr_i32(f"{name}_div_attr", div),
        n_levels=tensor_attr_i32(f"{name}_n_levels_attr", n_levels),
        signed=tensor_attr_i32(f"{name}_signed_attr", int(signed)),
        shift=tensor_attr_i32(f"{name}_shift_attr", shift),
    )


def make_deeploy_requantized_gemm_onnx(
    onnx_path: Path,
    input_shape: Tuple[int, ...],
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    out_mult: List[int],
    out_shift: List[int],
    out_signed: List[bool],
    opset_version: int = 9,
):
    if len(input_shape) != 2:
        raise ValueError(f"This simple MLP exporter expects rank-2 inputs, got {input_shape}")

    nodes = []
    initializers = []
    value_infos = []

    input_name = "input_0"
    prev = input_name
    prev_dtype = TensorProto.INT8

    graph_inputs = [
        helper.make_tensor_value_info(input_name, TensorProto.INT8, list(input_shape))
    ]

    batch = input_shape[0]

    for i, (W, B, mult, shift, signed) in enumerate(zip(weights, biases, out_mult, out_shift, out_signed)):
        W_name = f"layer{i}_weight"
        B_name = f"layer{i}_bias"
        M_name = f"layer{i}_mul"
        Y_name = f"output_0" if i == len(weights) - 1 else f"layer{i}_out"

        initializers.append(numpy_helper.from_array(W.astype(np.int8), W_name))
        initializers.append(numpy_helper.from_array(B.astype(np.int32), B_name))
        initializers.append(numpy_helper.from_array(np.array([mult], dtype=np.int32), M_name))

        out_dtype = TensorProto.INT8 if signed else TensorProto.UINT8
        out_shape = [batch, int(W.shape[0])]

        nodes.append(
            make_requantized_gemm_node(
                name=f"RequantizedGemm_{i}",
                A=prev,
                B=W_name,
                C=B_name,
                M=M_name,
                Y=Y_name,
                div=2 ** int(shift),
                shift=0,
                n_levels=256,
                signed=signed,
                transB=1,
            )
        )

        # Shape annotations are important because Deeploy does not do general shape inference.
        if i != len(weights) - 1:
            value_infos.append(helper.make_tensor_value_info(Y_name, out_dtype, out_shape))

        prev = Y_name
        prev_dtype = out_dtype

    graph_outputs = [
        helper.make_tensor_value_info("output_0", prev_dtype, [batch, int(weights[-1].shape[0])])
    ]

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
        producer_name="create_deeploy_int8_test_from_pytorch.py",
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Do NOT call onnx.checker.check_model here: RequantizedGemm is a Deeploy custom op.
    onnx.save(model, str(onnx_path))


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a Deeploy INT8 test from PyTorch + QuantLib PACT.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-path", type=str, help="Python file containing a PyTorch model class.")
    src.add_argument("--source-onnx", type=str, help="Existing float ONNX to load and copy into a built-in architecture.")

    parser.add_argument("--model-class", type=str, default=None, help="Class name when using --model-path.")
    parser.add_argument("--model-kwargs", type=str, default=None, help="JSON kwargs for --model-class.")
    parser.add_argument("--builtin-arch", choices=list(BUILTINS), default=None, help="Builtin arch when using --source-onnx.")

    parser.add_argument("--test-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="DeeployTest/Tests")
    parser.add_argument("--input-shapes", nargs="+", required=True)
    parser.add_argument("--input-range", nargs=2, type=float, default=[-1.0, 1.0])
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--opset-version", type=int, default=9)
    parser.add_argument("--D", type=int, default=2**18, help="QuantLib fake-to-true denominator.")
    parser.add_argument("--final-signed", action="store_true", default=True, help="Use signed INT8 final output.")
    parser.add_argument("--unsigned-final", dest="final_signed", action="store_false", help="Use UINT8 final output.")
    parser.add_argument("--emit-raw-quantlib-onnx", action="store_true", help="Also export raw QuantLib ONNX for debugging.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    input_shapes = [parse_shape(s) for s in args.input_shapes]
    if len(input_shapes) != 1:
        raise RuntimeError("This first script version supports one input tensor only.")

    device = torch.device("cpu")

    if args.model_path:
        if not args.model_class:
            raise RuntimeError("--model-class is required with --model-path")
        model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None
        float_model = load_model_from_file(args.model_path, args.model_class, model_kwargs).to(device)
    else:
        if not args.builtin_arch:
            raise RuntimeError("--builtin-arch is required with --source-onnx")
        float_model = load_float_onnx_into_builtin(args.source_onnx, args.builtin_arch, device)

    float_model.eval()

    test_dir = Path(args.output_dir) / args.test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate random float input as in create_deeploy_test.py.
    x_float_list = generate_random_float_inputs(input_shapes, tuple(args.input_range), np.float32)
    x_float = x_float_list[0]

    # 2) Compute an input epsilon and quantize the actual test input to signed int8.
    input_eps = compute_symmetric_input_eps(x_float, n_levels=256)
    x_i8 = quantize_input_i8(x_float, input_eps)
    x_q_float_for_quantlib = torch.from_numpy(x_i8.astype(np.float32))

    # 3) PACT fake-quant model.
    name2config = all_pact_create_configs_int8(deepcopy(float_model), patches={})
    pact_model = all_pact_f2f_recipe(deepcopy(float_model), name2config).to(device)
    pact_model.eval()

    # 4) QuantLib fake-to-true integerization.
    tq_model = f2t_convert(deepcopy(pact_model), x_q_float_for_quantlib, input_eps=input_eps, D=args.D)

    # 5) Extract integerized linear parameters and requant parameters.
    x_torch = torch.from_numpy(x_i8.astype(np.float32))
    linear_layers = collect_linear_layers_in_execution_order(tq_model, x_torch)

    weights, biases = [], []
    for layer in linear_layers:
        W, B = extract_linear_params(layer)
        weights.append(W)
        biases.append(B)

    out_mult, out_shift, out_signed = get_requant_params(
        tq_model,
        n_linears=len(linear_layers),
        final_signed=args.final_signed,
    )

    # 6) Integer reference outputs, not float PyTorch outputs.
    all_outputs = integer_forward(
        x_i8=x_i8,
        weights=weights,
        biases=biases,
        out_mult=out_mult,
        out_shift=out_shift,
        out_signed=out_signed,
    )

    deeploy_inputs = [x_i8.reshape(input_shapes[0])]
    deeploy_outputs = [all_outputs[-1]]

    # 7) Build Deeploy-oriented custom ONNX. This avoids raw Div/Floor/Clip nodes.
    network_path = test_dir / "network.onnx"
    make_deeploy_requantized_gemm_onnx(
        onnx_path=network_path,
        input_shape=input_shapes[0],
        weights=weights,
        biases=biases,
        out_mult=out_mult,
        out_shift=out_shift,
        out_signed=out_signed,
        opset_version=args.opset_version,
    )

    # Optional: raw QuantLib ONNX for comparing what PyTorch export would produce.
    if args.emit_raw_quantlib_onnx:
        raw_path = test_dir / "network_raw_quantlib.onnx"
        torch.onnx.export(
            tq_model,
            x_q_float_for_quantlib,
            str(raw_path),
            input_names=["input_0"],
            output_names=["output_0"],
            opset_version=args.opset_version,
            do_constant_folding=True,
        )

    # 8) Save Deeploy input/output files using the original naming convention.
    save_deeploy_npz(test_dir, deeploy_inputs, deeploy_outputs)

    readme = f"""# {args.test_name}

Generated by `create_deeploy_int8_test_from_pytorch.py`.

## Files

- `network.onnx`: Deeploy-oriented INT8 graph using custom `RequantizedGemm` nodes.
- `inputs.npz`: signed INT8 input stored as `input_0`.
- `outputs.npz`: integer reference output stored as `output_0`.

## Quantization

- Input epsilon: `{input_eps}`
- QuantLib fake-to-true denominator D: `{args.D}`
- Requant multipliers: `{out_mult}`
- Requant shifts: `{out_shift}`
- Output signedness per layer: `{out_signed}`

## Run

```bash
cd DeeployTest
python3 testRunner_tiled_siracusa.py -t Tests/{args.test_name} --cores 8 --l1 64000
```
"""
    (test_dir / "README.md").write_text(readme)

    if args.verbose:
        print("[INFO] test_dir:", test_dir)
        print("[INFO] input_eps:", input_eps)
        print("[INFO] input dtype/min/max:", deeploy_inputs[0].dtype, deeploy_inputs[0].min(), deeploy_inputs[0].max())
        print("[INFO] output dtype/min/max:", deeploy_outputs[0].dtype, deeploy_outputs[0].min(), deeploy_outputs[0].max())
        print("[INFO] weights:", [w.shape for w in weights])
        print("[INFO] out_mult:", out_mult)
        print("[INFO] out_shift:", out_shift)
        print("[INFO] out_signed:", out_signed)
        print("[INFO] wrote:", network_path)

    print("\n✓ Deeploy INT8 test generation complete")
    print(f"Test directory: {test_dir}")


if __name__ == "__main__":
    main()
