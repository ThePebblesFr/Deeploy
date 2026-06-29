#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Generate Deeploy input/output test files from an existing ONNX model.

This script is useful when you already have a Deeploy-compatible ONNX model
and only need to generate the reference input/output .npz files.

It performs the following steps:
1. Load and validate an existing ONNX model.
2. Generate random inputs, or load existing inputs from a .npz file.
3. Run inference with ONNX Runtime.
4. Save inputs.npz and outputs.npz in a Deeploy-style test folder.
5. Optionally copy the ONNX model as network.onnx into the test folder.

Examples
--------

Generate random inputs for a model with one static or explicitly provided input:

    python create_deeploy_io_from_onnx.py \
        --onnx-path my_model.onnx \
        --test-name testMyModel \
        --input-shapes "1,16"

Generate random inputs with a named input shape:

    python create_deeploy_io_from_onnx.py \
        --onnx-path my_model.onnx \
        --test-name testMyModel \
        --input-shapes "input_0:1,16"

Use an existing inputs.npz and only regenerate outputs.npz:

    python create_deeploy_io_from_onnx.py \
        --onnx-path my_model.onnx \
        --test-name testMyModel \
        --input-npz inputs.npz
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, shape_inference


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


def parse_shape(shape_str: str) -> Tuple[Optional[str], Tuple[int, ...]]:
    """
    Parse either:
        "1,3,224,224"
    or:
        "input_0:1,3,224,224"

    Returns:
        (input_name or None, shape_tuple)
    """
    if ":" in shape_str:
        name, raw_shape = shape_str.split(":", 1)
        name = name.strip()
    else:
        name = None
        raw_shape = shape_str

    try:
        shape = tuple(int(x.strip()) for x in raw_shape.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid shape string '{shape_str}': {exc}") from exc

    if len(shape) == 0:
        raise ValueError(f"Invalid empty shape string: '{shape_str}'")

    return name, shape


def get_model_inputs(onnx_model: onnx.ModelProto):
    """
    Return real runtime graph inputs, excluding initializers such as weights/biases.
    """
    initializer_names = {init.name for init in onnx_model.graph.initializer}

    inputs = []
    for value_info in onnx_model.graph.input:
        if value_info.name not in initializer_names:
            inputs.append(value_info)

    return inputs


def get_value_info_shape(value_info) -> Tuple[Optional[int], ...]:
    """
    Extract shape from ONNX ValueInfoProto.

    Dynamic dimensions are returned as None.
    """
    tensor_type = value_info.type.tensor_type
    shape = []

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append(None)

    return tuple(shape)


def get_value_info_dtype(value_info) -> np.dtype:
    """
    Extract NumPy dtype from ONNX ValueInfoProto.
    """
    elem_type = value_info.type.tensor_type.elem_type

    if elem_type not in ONNX_TO_NUMPY_DTYPE:
        raise RuntimeError(
            f"Unsupported ONNX input dtype for '{value_info.name}': elem_type={elem_type}"
        )

    return ONNX_TO_NUMPY_DTYPE[elem_type]


def generate_input_array(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    input_range: Tuple[float, float],
) -> np.ndarray:
    """
    Generate one random input array depending on dtype.
    """
    min_val, max_val = input_range

    if np.issubdtype(dtype, np.floating):
        return np.random.uniform(min_val, max_val, size=shape).astype(dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)

        low = max(int(np.floor(min_val)), info.min)
        high = min(int(np.ceil(max_val)), info.max)

        if low > high:
            raise ValueError(
                f"Invalid input range {input_range} for dtype {dtype}: "
                f"clipped low={low}, high={high}"
            )

        # np.random.randint high is exclusive.
        return np.random.randint(low, high + 1, size=shape, dtype=dtype)

    if dtype == np.bool_:
        return np.random.choice([False, True], size=shape).astype(dtype)

    raise RuntimeError(f"Unsupported dtype for random generation: {dtype}")


def load_inputs_from_npz(input_npz: Path, model_input_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Load inputs from an existing .npz file.

    Supports either:
    - named keys matching ONNX inputs, e.g. input_0
    - anonymous NumPy keys, e.g. arr_0, arr_1
    """
    loaded = np.load(input_npz)
    keys = list(loaded.keys())

    inputs = {}

    # Case 1: keys match ONNX input names.
    if all(name in keys for name in model_input_names):
        for name in model_input_names:
            inputs[name] = loaded[name]
        return inputs

    # Case 2: fallback to arr_0, arr_1, ... ordering.
    arr_keys = [f"arr_{i}" for i in range(len(model_input_names))]
    if all(k in keys for k in arr_keys):
        for name, arr_key in zip(model_input_names, arr_keys):
            inputs[name] = loaded[arr_key]
        return inputs

    raise RuntimeError(
        f"Could not map inputs from {input_npz}.\n"
        f"Expected either keys {model_input_names} or {arr_keys}, got {keys}."
    )


def save_npz(
    path: Path,
    arrays: List[np.ndarray],
    names: List[str],
    key_style: str,
):
    """
    Save arrays using either:
    - key_style='arr': arr_0, arr_1, ... like np.savez(path, *arrays)
    - key_style='named': input_0/output_0 style named keys
    """
    if key_style == "arr":
        np.savez(path, *arrays)
    elif key_style == "named":
        np.savez(path, **{name: arr for name, arr in zip(names, arrays)})
    else:
        raise ValueError(f"Unsupported key style: {key_style}")


def create_readme(
    test_dir: Path,
    test_name: str,
    onnx_path: Path,
    input_names: List[str],
    output_names: List[str],
    input_shapes: Dict[str, Tuple[int, ...]],
    key_style: str,
):
    """
    Create a small README next to the generated Deeploy test files.
    """
    content = f"""# {test_name}

Generated from existing ONNX model:

```text
{onnx_path}
```

## Files

- `network.onnx`: ONNX model
- `inputs.npz`: Input tensors
- `outputs.npz`: Expected output tensors

## Inputs

"""

    for name in input_names:
        content += f"- `{name}`: shape `{input_shapes[name]}`\n"

    content += "\n## Outputs\n\n"

    for name in output_names:
        content += f"- `{name}`\n"

    content += f"""

## NPZ key style

This test was generated with key style:

```text
{key_style}
```

- `arr`: saves tensors as `arr_0`, `arr_1`, ...
- `named`: saves tensors using ONNX names such as `input_0`, `output_0`, ...

## Example Deeploy command

```bash
cd DeeployTest
python testRunner_generic.py -t Tests/{test_name}
```
"""

    (test_dir / "README.md").write_text(content)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Deeploy inputs.npz and outputs.npz from an existing ONNX model"
    )

    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Path to the existing ONNX model",
    )

    parser.add_argument(
        "--test-name",
        type=str,
        required=True,
        help="Name of the Deeploy test folder",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DeeployTest/Tests",
        help="Parent output directory",
    )

    parser.add_argument(
        "--input-shapes",
        nargs="*",
        default=None,
        help=(
            "Input shapes. Either ordered shapes like '1,3,224,224' "
            "or named shapes like 'input_0:1,3,224,224'. "
            "Required if ONNX has dynamic dimensions."
        ),
    )

    parser.add_argument(
        "--input-npz",
        type=str,
        default=None,
        help="Optional existing inputs.npz to use instead of generating random inputs",
    )

    parser.add_argument(
        "--input-range",
        nargs=2,
        type=float,
        default=[-1.0, 1.0],
        metavar=("MIN", "MAX"),
        help="Range for generated random values. Default: -1.0 1.0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--key-style",
        choices=["arr", "named"],
        default="named",
        help=(
            "How to save NPZ keys. "
            "'arr' gives arr_0, arr_1, ...; "
            "'named' gives input/output names. Default: named."
        ),
    )

    parser.add_argument(
        "--no-copy-onnx",
        action="store_true",
        help="Do not copy the ONNX model into the test folder as network.onnx",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print more information",
    )

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    test_dir = Path(args.output_dir) / args.test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading ONNX model: {onnx_path}")

    model = onnx.load(str(onnx_path))
    checker.check_model(model)

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:
        print(f"[WARNING] Shape inference failed: {exc}")

    model_inputs = get_model_inputs(model)
    input_names = [inp.name for inp in model_inputs]

    if len(input_names) == 0:
        raise RuntimeError("The ONNX model appears to have no runtime inputs.")

    if args.verbose:
        print("[INFO] Model inputs:")
        for inp in model_inputs:
            print(
                f"  - {inp.name}: shape={get_value_info_shape(inp)}, "
                f"dtype={get_value_info_dtype(inp)}"
            )

    # Parse user-provided input shapes.
    provided_shapes_by_name = {}
    provided_shapes_ordered = []

    if args.input_shapes:
        for raw_shape in args.input_shapes:
            name, shape = parse_shape(raw_shape)
            if name is None:
                provided_shapes_ordered.append(shape)
            else:
                provided_shapes_by_name[name] = shape

    # Prepare inputs.
    if args.input_npz:
        input_npz = Path(args.input_npz)
        if not input_npz.exists():
            raise FileNotFoundError(f"Input NPZ file not found: {input_npz}")

        print(f"[INFO] Loading existing inputs from: {input_npz}")
        ort_inputs = load_inputs_from_npz(input_npz, input_names)
    else:
        print("[INFO] Generating random inputs")
        ort_inputs = {}

        for idx, value_info in enumerate(model_inputs):
            name = value_info.name
            dtype = get_value_info_dtype(value_info)
            model_shape = get_value_info_shape(value_info)

            if name in provided_shapes_by_name:
                shape = provided_shapes_by_name[name]
            elif idx < len(provided_shapes_ordered):
                shape = provided_shapes_ordered[idx]
            else:
                if any(dim is None for dim in model_shape):
                    raise RuntimeError(
                        f"Input '{name}' has dynamic shape {model_shape}. "
                        f"Please provide it with --input-shapes '{name}:...'."
                    )
                shape = tuple(int(dim) for dim in model_shape)

            ort_inputs[name] = generate_input_array(
                shape=shape,
                dtype=dtype,
                input_range=tuple(args.input_range),
            )

    # Run inference.
    print("[INFO] Running ONNX Runtime inference")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(output_names, ort_inputs)

    # Save inputs and outputs.
    input_arrays = [ort_inputs[name] for name in input_names]

    inputs_path = test_dir / "inputs.npz"
    outputs_path = test_dir / "outputs.npz"

    save_npz(
        path=inputs_path,
        arrays=input_arrays,
        names=input_names,
        key_style=args.key_style,
    )

    save_npz(
        path=outputs_path,
        arrays=outputs,
        names=output_names,
        key_style=args.key_style,
    )

    print(f"[INFO] Saved inputs to:  {inputs_path}")
    print(f"[INFO] Saved outputs to: {outputs_path}")

    # Copy ONNX as network.onnx.
    if not args.no_copy_onnx:
        dst_onnx = test_dir / "network.onnx"
        shutil.copyfile(onnx_path, dst_onnx)
        print(f"[INFO] Copied ONNX to:   {dst_onnx}")

    # README.
    input_shapes = {name: tuple(arr.shape) for name, arr in ort_inputs.items()}
    create_readme(
        test_dir=test_dir,
        test_name=args.test_name,
        onnx_path=onnx_path,
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes,
        key_style=args.key_style,
    )

    print("\n✓ Deeploy IO generation successful!")
    print(f"Test directory: {test_dir}")


if __name__ == "__main__":
    main()
