#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Generate Deeploy-compatible inputs.npz / outputs.npz from an existing ONNX file,
and save a Deeploy-friendlier network.onnx with inferred shape annotations.

This is useful when the ONNX already comes from an external quantization / DORY
preparation flow and you only need to generate reference IO files for Deeploy.

Main difference compared with a naive ONNXRuntime IO generator:
  - the ONNX copied into the Deeploy test folder is NOT the raw input file;
    it is the shape-inferred ONNX model.
  - this matters because Deeploy does not support missing tensor shapes.

Example:
    python create_deeploy_int_io_from_onnx.py \
        --onnx-path CustomTestsGeneration/models/ES-DNN1.onnx \
        --test-name ES_DNN1_int \
        --output-dir DeeployTest/Tests \
        --input-shapes "1,16" \
        --input-dtype int8 \
        --input-range -128 127

For your ES-DNN1 model, note that the uploaded ONNX graph input is FLOAT,
although weights and intermediate quantized values are integer-valued floats.
Therefore, by default, generated inputs are cast to the ONNX-declared dtype
before ONNX Runtime inference. The saved inputs can optionally be stored as
integer dtype using --save-input-dtype, but only do that if Deeploy expects it.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference


ONNX_TO_NUMPY_DTYPE = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT8: np.int8,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.UINT8: np.uint8,
    TensorProto.UINT16: np.uint16,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
    TensorProto.BOOL: np.bool_,
}

DTYPE_FROM_STRING = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "bool": np.bool_,
    "auto": None,
}


def parse_shape(shape_str: str) -> Tuple[Optional[str], Tuple[int, ...]]:
    """Parse '1,16' or 'input_name:1,16'."""
    if ":" in shape_str:
        name, raw_shape = shape_str.split(":", 1)
        name = name.strip()
    else:
        name = None
        raw_shape = shape_str

    shape = tuple(int(x.strip()) for x in raw_shape.split(",") if x.strip())
    if not shape:
        raise ValueError(f"Invalid empty shape: {shape_str}")
    return name, shape


def get_shape(value_info) -> Tuple[Optional[int], ...]:
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    return tuple(dims)


def get_dtype(value_info):
    elem_type = value_info.type.tensor_type.elem_type
    if elem_type not in ONNX_TO_NUMPY_DTYPE:
        raise RuntimeError(
            f"Unsupported ONNX dtype for tensor '{value_info.name}': elem_type={elem_type}"
        )
    return ONNX_TO_NUMPY_DTYPE[elem_type]


def get_runtime_inputs(model: onnx.ModelProto):
    initializer_names = {init.name for init in model.graph.initializer}
    return [inp for inp in model.graph.input if inp.name not in initializer_names]


def generate_array(shape: Tuple[int, ...], dtype, value_range: Tuple[float, float]):
    lo, hi = value_range
    dtype = np.dtype(dtype)

    if np.issubdtype(dtype, np.floating):
        return np.random.uniform(lo, hi, size=shape).astype(dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo_i = max(int(np.floor(lo)), info.min)
        hi_i = min(int(np.ceil(hi)), info.max)
        if hi_i < lo_i:
            raise ValueError(f"Invalid range {value_range} for dtype {dtype}")
        return np.random.randint(lo_i, hi_i + 1, size=shape).astype(dtype)

    if dtype == np.bool_:
        return np.random.choice([False, True], size=shape).astype(dtype)

    raise RuntimeError(f"Unsupported generated dtype: {dtype}")


def load_npz_inputs(path: Path, input_names: List[str]) -> Dict[str, np.ndarray]:
    data = np.load(path)
    keys = list(data.keys())

    if all(name in keys for name in input_names):
        return {name: data[name] for name in input_names}

    arr_keys = [f"arr_{i}" for i in range(len(input_names))]
    if all(key in keys for key in arr_keys):
        return {name: data[key] for name, key in zip(input_names, arr_keys)}

    raise RuntimeError(
        f"Cannot map NPZ inputs from {path}. Expected keys {input_names} "
        f"or {arr_keys}, got {keys}."
    )


def save_npz(path: Path, arrays: List[np.ndarray], names: List[str], key_style: str):
    if key_style == "arr":
        np.savez(path, *arrays)
    elif key_style == "named":
        np.savez(path, **{name: arr for name, arr in zip(names, arrays)})
    else:
        raise ValueError(f"Unsupported key style: {key_style}")


def infer_shapes_or_die(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run ONNX shape inference and return the inferred model."""
    checker.check_model(model)
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    checker.check_model(inferred)
    return inferred


def collect_known_shapes(model: onnx.ModelProto) -> Dict[str, Tuple[Optional[int], ...]]:
    shapes = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.type.HasField("tensor_type"):
            shapes[vi.name] = get_shape(vi)
    for init in model.graph.initializer:
        shapes[init.name] = tuple(init.dims)
    return shapes


def print_model_diagnostics(model: onnx.ModelProto, inferred: onnx.ModelProto):
    print("[INFO] ONNX diagnostics")
    print(f"       opsets: {[ (o.domain, o.version) for o in model.opset_import ]}")
    print(f"       nodes: {len(model.graph.node)}")
    print(f"       original value_info entries: {len(model.graph.value_info)}")
    print(f"       inferred value_info entries: {len(inferred.graph.value_info)}")

    print("[INFO] Runtime inputs")
    for inp in get_runtime_inputs(inferred):
        print(
            f"       {inp.name}: dtype={np.dtype(get_dtype(inp)).name}, "
            f"shape={get_shape(inp)}"
        )

    float_integer_initializers = []
    for init in inferred.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            if arr.size and np.all(np.isfinite(arr)) and np.allclose(arr, np.round(arr)):
                float_integer_initializers.append(init.name)

    if float_integer_initializers:
        print(
            "[WARNING] Some initializers are FLOAT but contain integer-valued data. "
            "This often means the graph is fake/integerized-quantized, not a true INT8 ONNX graph."
        )
        print(f"          Examples: {float_integer_initializers[:8]}")


def create_readme(test_dir: Path, args, input_names, output_names, input_arrays, output_arrays):
    lines = []
    lines.append(f"# {args.test_name}\n")
    lines.append("Generated from an existing ONNX model.\n")
    lines.append("## Files\n")
    lines.append("- `network.onnx`: shape-inferred ONNX model copied/generated for Deeploy\n")
    lines.append("- `inputs.npz`: generated or supplied test inputs\n")
    lines.append("- `outputs.npz`: ONNX Runtime reference outputs\n")
    lines.append("\n## Inputs\n")
    for name, arr in zip(input_names, input_arrays):
        lines.append(f"- `{name}`: shape `{arr.shape}`, dtype `{arr.dtype}`\n")
    lines.append("\n## Outputs\n")
    for name, arr in zip(output_names, output_arrays):
        lines.append(f"- `{name}`: shape `{arr.shape}`, dtype `{arr.dtype}`\n")
    lines.append("\n## Notes\n")
    lines.append(f"- NPZ key style: `{args.key_style}`\n")
    lines.append("- The saved `network.onnx` is shape-inferred to avoid Deeploy missing-shape errors.\n")
    (test_dir / "README.md").write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Deeploy IO files from an existing ONNX and save a shape-inferred network.onnx"
    )
    parser.add_argument("--onnx-path", required=True, type=Path)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--output-dir", default="DeeployTest/Tests", type=Path)
    parser.add_argument("--input-shapes", nargs="*", default=None)
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--input-range", nargs=2, type=float, default=[-1.0, 1.0])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--input-dtype",
        choices=list(DTYPE_FROM_STRING.keys()),
        default="auto",
        help="dtype used to generate random inputs before casting for ONNX Runtime. Default: ONNX input dtype.",
    )
    parser.add_argument(
        "--save-input-dtype",
        choices=list(DTYPE_FROM_STRING.keys()),
        default="auto",
        help="dtype used when saving inputs.npz. Default: keep ONNX Runtime input dtype.",
    )
    parser.add_argument(
        "--key-style",
        choices=["named", "arr"],
        default="named",
        help="Use named keys input_0/output_0 style, or arr_0/arr_1 style.",
    )
    parser.add_argument(
        "--no-infer-shapes",
        action="store_true",
        help="Copy the raw ONNX instead of saving a shape-inferred model. Not recommended for Deeploy.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if not args.onnx_path.exists():
        raise FileNotFoundError(args.onnx_path)

    test_dir = args.output_dir / args.test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading ONNX model: {args.onnx_path}")
    raw_model = onnx.load(str(args.onnx_path))

    if args.no_infer_shapes:
        model_for_deeploy = raw_model
        checker.check_model(model_for_deeploy)
    else:
        print("[INFO] Running ONNX shape inference for Deeploy network.onnx")
        model_for_deeploy = infer_shapes_or_die(raw_model)

    print_model_diagnostics(raw_model, model_for_deeploy)

    # Save the model Deeploy will actually parse.
    network_path = test_dir / "network.onnx"
    onnx.save(model_for_deeploy, str(network_path))
    print(f"[INFO] Saved Deeploy ONNX to: {network_path}")

    runtime_inputs = get_runtime_inputs(model_for_deeploy)
    input_names = [inp.name for inp in runtime_inputs]
    if not input_names:
        raise RuntimeError("No runtime inputs found in model.")

    # Resolve user-provided shapes.
    named_shapes = {}
    ordered_shapes = []
    if args.input_shapes:
        for s in args.input_shapes:
            name, shape = parse_shape(s)
            if name is None:
                ordered_shapes.append(shape)
            else:
                named_shapes[name] = shape

    # Generate/load inputs for ONNX Runtime.
    if args.input_npz:
        print(f"[INFO] Loading existing inputs from: {args.input_npz}")
        ort_inputs = load_npz_inputs(args.input_npz, input_names)
    else:
        print("[INFO] Generating random inputs")
        ort_inputs = {}
        for idx, inp in enumerate(runtime_inputs):
            name = inp.name
            onnx_dtype = get_dtype(inp)
            gen_dtype = DTYPE_FROM_STRING[args.input_dtype] or onnx_dtype

            if name in named_shapes:
                shape = named_shapes[name]
            elif idx < len(ordered_shapes):
                shape = ordered_shapes[idx]
            else:
                shape = get_shape(inp)
                if any(dim is None for dim in shape):
                    raise RuntimeError(
                        f"Input '{name}' has dynamic shape {shape}. Provide --input-shapes '{name}:...'."
                    )
                shape = tuple(int(dim) for dim in shape)

            generated = generate_array(shape, gen_dtype, tuple(args.input_range))
            # ONNX Runtime requires the declared ONNX dtype.
            ort_inputs[name] = generated.astype(onnx_dtype)

    print("[INFO] Running ONNX Runtime inference")
    session = ort.InferenceSession(str(network_path), providers=["CPUExecutionProvider"])
    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(output_names, ort_inputs)

    # Save inputs. By default, save what ONNX Runtime consumed.
    input_arrays_to_save = []
    save_dtype = DTYPE_FROM_STRING[args.save_input_dtype]
    for name in input_names:
        arr = ort_inputs[name]
        if save_dtype is not None:
            arr = arr.astype(save_dtype)
        input_arrays_to_save.append(arr)

    # For named key style, Deeploy's older generator used input_0/output_0,
    # not necessarily the raw ONNX tensor names. Preserve that behavior.
    if args.key_style == "named":
        input_save_names = [f"input_{i}" for i in range(len(input_arrays_to_save))]
        output_save_names = [f"output_{i}" for i in range(len(outputs))]
    else:
        input_save_names = input_names
        output_save_names = output_names

    inputs_path = test_dir / "inputs.npz"
    outputs_path = test_dir / "outputs.npz"
    save_npz(inputs_path, input_arrays_to_save, input_save_names, args.key_style)
    save_npz(outputs_path, outputs, output_save_names, args.key_style)

    print(f"[INFO] Saved inputs to:  {inputs_path}")
    print(f"[INFO] Saved outputs to: {outputs_path}")

    if args.verbose:
        for name, arr in zip(input_save_names, input_arrays_to_save):
            print(f"[VERBOSE] saved input {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
        for name, arr in zip(output_save_names, outputs):
            print(f"[VERBOSE] saved output {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")

    create_readme(
        test_dir,
        args,
        input_save_names,
        output_save_names,
        input_arrays_to_save,
        outputs,
    )

    print("\n✓ Deeploy IO generation successful!")
    print(f"Test directory: {test_dir}")


if __name__ == "__main__":
    main()
