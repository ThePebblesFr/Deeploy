#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Fuse the raw Mul->Add->Div->Floor->Clip chains produced by quantlib's
quantlib.backends.dory.export_net (quantlib.editing.fx.passes.pact.RequantShift's
forward() decomposes to these 5 primitive ops -- it has no custom ONNX symbolic()
registration, and the modern torch.onnx exporter no longer respects the old-style
custom-op export hooks the legacy TorchScript exporter used to) into single
"RequantShift" custom nodes, matching the format Deeploy's own importer expects
(Deeploy/Targets/Generic/Parsers.py:RQSParserInterface/RequantShiftParser):
    inputs  = [data_in, mul (tensor), add (tensor)]
    attrs   = {div: <D>, n_levels: <int>, signed: <0|1>}
    outputs = [data_out]

Leaves everything else (Conv, residual Add, MaxPool/AveragePool, Gemm/Reshape)
untouched -- a residual Add is distinguished from a quantization Add simply by not
being part of a Mul->Add->Div->Floor->Clip chain (its output normally feeds into the
next chain's Mul, rather than a Div).

Usage:
    python3 fuse_requantshift.py --input in.onnx --output out.onnx
"""
import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def _unwrap_identity(tensor):
    """torch.onnx.export's constant-folding deduplicates repeated constants (e.g.
    the same fixed-point divisor D shared by every RequantShift layer): the first
    reference gets a real gs.Constant, every subsequent reference is routed through
    a single Identity node instead of duplicating the tensor. Walk back through one
    such hop so callers see the real constant either way."""
    if isinstance(tensor, gs.Variable) and len(tensor.inputs) == 1 and tensor.inputs[0].op == "Identity":
        return tensor.inputs[0].inputs[0]
    return tensor


def fuse(graph: gs.Graph) -> gs.Graph:
    floor_nodes = [n for n in graph.nodes if n.op == "Floor"]
    n_fused = 0

    for floor_node in floor_nodes:
        div_node = floor_node.inputs[0].inputs[0]
        assert div_node.op == "Div", f"expected Div feeding Floor, got {div_node.op}"
        add_node = div_node.inputs[0].inputs[0]
        assert add_node.op == "Add", f"expected Add feeding Div, got {add_node.op}"
        mul_node = add_node.inputs[0].inputs[0]
        assert mul_node.op == "Mul", f"expected Mul feeding Add, got {mul_node.op}"

        assert len(floor_node.outputs[0].outputs) == 1, "Floor output must feed exactly one Clip"
        clip_node = floor_node.outputs[0].outputs[0]
        assert clip_node.op == "Clip", f"expected Clip consuming Floor, got {clip_node.op}"

        x = mul_node.inputs[0]
        mul_const = _unwrap_identity(mul_node.inputs[1])
        add_const = _unwrap_identity(add_node.inputs[1])
        div_const = float(_unwrap_identity(div_node.inputs[1]).values)
        # Clip's min/max are extra tensor inputs in opset>=11, but plain attributes
        # in opset<11 (e.g. export_net's opset_version=10) -- support both.
        if len(clip_node.inputs) >= 3:
            clip_min = float(_unwrap_identity(clip_node.inputs[1]).values)
            clip_max = float(_unwrap_identity(clip_node.inputs[2]).values)
        else:
            clip_min = float(clip_node.attrs["min"])
            clip_max = float(clip_node.attrs["max"])

        signed = 1 if clip_min < 0 else 0
        n_levels = int(round(clip_max - clip_min + 1))

        rqs_out = clip_node.outputs[0]

        # Deeploy's parsers/passes expect div/n_levels_out/signed as gs.Constant-wrapped
        # tensors (matching DORY's own export convention), not plain Python scalars.
        rqs_node = gs.Node(
            op="RequantShift",
            name=f"RequantShift_{n_fused}",
            inputs=[x, mul_const, add_const],
            outputs=[rqs_out],
            attrs={
                "div": gs.Constant(name=f"RequantShift_{n_fused}_div", values=np.array(div_const, dtype=np.float32)),
                "n_levels_out": gs.Constant(name=f"RequantShift_{n_fused}_n_levels_out",
                                             values=np.array([n_levels], dtype=np.float32)),
                "signed": gs.Constant(name=f"RequantShift_{n_fused}_signed",
                                       values=np.array([signed], dtype=np.float32)),
            },
        )

        # Detach the 5 old nodes (clear their outputs so cleanup() removes them) --
        # note this must happen AFTER constructing rqs_node above: the gs.Node
        # constructor already links rqs_node into rqs_out.inputs, so clearing
        # rqs_out.inputs here (rather than just clip_node.outputs) would wipe that
        # link back out and orphan rqs_out.
        for old_node in (mul_node, add_node, div_node, floor_node, clip_node):
            old_node.outputs.clear()
        rqs_node.domain = "PACTOps"
        graph.nodes.append(rqs_node)
        n_fused += 1

    graph.cleanup().toposort()
    print(f"[fuse_requantshift] Fused {n_fused} Mul->Add->Div->Floor->Clip chains into RequantShift nodes.")
    return graph


def merge_consecutive(graph: gs.Graph) -> gs.Graph:
    """
    Back-to-back RequantShift nodes (e.g. a residual add's own output-activation
    RequantShift immediately followed by the BasicBlock's post-add ReLU's
    RequantShift) are valid individually but Deeploy's PULPOpen parser only matches
    a single RequantShift layer at a time, not two chained ones -- it relies on
    Deeploy/Targets/Generic/TopologyOptimizationPasses/Passes.py's MergeRequantPass
    running first. Rather than depend on that pass being wired into this platform's
    lowering pipeline, do the identical merge (same math as _merge_requant_fun)
    ourselves so the graph hits the parser already reduced to one node.
    """
    n_merged = 0
    changed = True
    while changed:
        changed = False
        for rqs1 in list(graph.nodes):
            if rqs1.op != "RequantShift" or len(rqs1.outputs) == 0:
                continue  # already detached by an earlier merge this pass
            out = rqs1.outputs[0]
            if len(out.outputs) != 1:
                continue
            rqs2 = out.outputs[0]
            if rqs2.op != "RequantShift":
                continue
            if not (isinstance(rqs1.inputs[1], gs.Constant) and isinstance(rqs1.inputs[2], gs.Constant) and
                    isinstance(rqs2.inputs[1], gs.Constant) and isinstance(rqs2.inputs[2], gs.Constant)):
                continue

            div1 = float(rqs1.attrs['div'].values)
            div2 = float(rqs2.attrs['div'].values)
            new_div = max(div1, div2)
            min_div = min(div1, div2)
            # The merged node's output type must match rqs2 (the *second*/final requant's
            # own clip semantics) -- e.g. a signed residual-accumulator requant (rqs1)
            # feeding a post-ReLU unsigned requant (rqs2) must merge to *unsigned*, not
            # max(signed1, signed2)=signed (matching Deeploy's own _merge_requant_fun,
            # which silently clips real values in [128,255] down to 127 in exactly this
            # case -- confirmed empirically: this was the source of a ~55-point accuracy
            # regression on ImageClassification_small).
            n_levels = float(rqs2.attrs['n_levels_out'].values[0])
            signed = float(rqs2.attrs['signed'].values[0])

            mul1, mul2 = rqs1.inputs[1].values, rqs2.inputs[1].values
            add1, add2 = rqs1.inputs[2].values, rqs2.inputs[2].values

            new_mul = np.array(np.round((mul1 * mul2) / min_div), dtype=np.float32)
            new_add = np.array(np.round(((add1 * mul2) + (div1 * add2)) / min_div), dtype=np.float32)

            merged = gs.Node(
                op="RequantShift",
                name=f"{rqs1.name}_merge_{rqs2.name}",
                inputs=[rqs1.inputs[0],
                        gs.Constant(name=f"{rqs1.name}_merged_mul", values=new_mul),
                        gs.Constant(name=f"{rqs1.name}_merged_add", values=new_add)],
                outputs=[rqs2.outputs[0]],
                attrs={
                    "div": gs.Constant(name=f"{rqs1.name}_merged_div", values=np.array(new_div, dtype=np.float32)),
                    "n_levels_out": gs.Constant(name=f"{rqs1.name}_merged_n_levels_out",
                                                 values=np.array([n_levels], dtype=np.float32)),
                    "signed": gs.Constant(name=f"{rqs1.name}_merged_signed",
                                           values=np.array([signed], dtype=np.float32)),
                },
            )
            rqs1.outputs.clear()
            rqs2.outputs.clear()
            merged.domain = "PACTOps"
            graph.nodes.append(merged)
            n_merged += 1
            changed = True
            break  # graph.nodes mutated; restart the scan

    graph.cleanup().toposort()
    print(f"[fuse_requantshift] Merged {n_merged} consecutive RequantShift pairs.")
    return graph


def add_conv_kernel_shape(graph: gs.Graph) -> gs.Graph:
    """
    The modern torch.onnx exporter omits Conv's `kernel_shape` attribute (it's
    optional per the ONNX spec -- inferable from the weight tensor), but the
    reference heavy MLPerf model's Conv nodes all carry it explicitly, and
    Deeploy's tiling constraint solver appears to depend on it being present
    (confirmed empirically: a single Conv+RequantShift tiles fine without it, but
    chaining two -- the minimal case where the solver must propagate a tile size
    across layers -- goes infeasible without it, and adding it back fixes that).
    """
    n_fixed = 0
    for node in graph.nodes:
        if node.op != "Conv" or "kernel_shape" in node.attrs:
            continue
        weight = node.inputs[1]
        node.attrs["kernel_shape"] = list(weight.shape[2:])
        n_fixed += 1
    print(f"[fuse_requantshift] Added kernel_shape attribute to {n_fixed} Conv node(s).")
    return graph


def fix_reduce_mean_axes(graph: gs.Graph) -> gs.Graph:
    """
    The modern torch.onnx exporter emits ReduceMean's axes as a second (int64
    Constant) input, opset>=18 style. Deeploy's generic bind() path creates a
    PULPConstantBuffer for it but never sets that buffer's `_type`
    (Deeploy/Targets/Generic/Parsers.py:ReduceMeanParser does accept this form at
    *parse* time, so this only breaks later, during binding) -- this op is
    evidently only exercised with the older opset<18 axes-as-attribute form
    elsewhere in this codebase. Converting back to that form (single input, `axes`/
    `keepdims` attributes) uses the same values and sidesteps the gap.
    """
    n_fixed = 0
    for node in graph.nodes:
        if node.op != "ReduceMean" or len(node.inputs) != 2:
            continue
        axes_tensor = node.inputs[1]
        assert isinstance(axes_tensor, gs.Constant), "ReduceMean axes input must be a constant"
        rank = len(node.inputs[0].shape)
        # Normalize negative axes (e.g. -1, -2) to their positive equivalents -- the
        # torch.onnx exporter emits negative indices here, but Deeploy's own
        # attribute-based-axes code path may not handle them the way generic Python
        # indexing would, so store them the way an opset<18 exporter always did.
        node.attrs["axes"] = sorted(int(a) if a >= 0 else int(a) + rank for a in axes_tensor.values)
        node.inputs = [node.inputs[0]]
        n_fixed += 1
    graph.cleanup().toposort()
    print(f"[fuse_requantshift] Converted {n_fixed} ReduceMean node(s) to attribute-based axes.")
    return graph


def convert_global_avg_pool(graph: gs.Graph) -> gs.Graph:
    """
    torch's AdaptiveAvgPool2d((1, 1)) (e.g. ResNet's self.avgpool) traces to
    GlobalAveragePool in this exporter path -- Deeploy's PULPOpen engine has no
    operator mapping for it at all (only ever exercised via the older exporter
    path, which expressed the identical operation as ReduceMean(axes=[2,3],
    keepdims=1) instead). Mathematically identical for a 4D NCHW input -- rewrite
    in place, reusing the same input/output tensors so nothing downstream needs
    to change.
    """
    n_converted = 0
    for node in graph.nodes:
        if node.op != "GlobalAveragePool":
            continue
        rank = len(node.inputs[0].shape)
        node.op = "ReduceMean"
        node.attrs["axes"] = list(range(2, rank))
        node.attrs["keepdims"] = 1
        n_converted += 1
    graph.cleanup().toposort()
    print(f"[fuse_requantshift] Converted {n_converted} GlobalAveragePool node(s) to ReduceMean.")
    return graph


def add_final_requant(graph: gs.Graph, div: float = 128.0) -> gs.Graph:
    """
    quantlib's harmonize/integerization pipeline evidently doesn't always append a
    RequantShift after the final classifier Gemm (it does for the heavy MLPerf
    reference model -- Deeploy's PULPGEMMParser expects one, since PULP's GEMM
    kernel produces a wide int32 accumulator that needs requantizing down to int8
    before it's a valid final output). Add a simple mul=1/add=0 RequantShift so the
    raw logits are just floor-divided by `div` and clipped to signed int8 --
    argmax(logits) is invariant under this positive affine rescaling, so it doesn't
    change the network's actual classification behavior.
    """
    last_node = graph.nodes[-1]
    if last_node.op != "Gemm":
        print(f"[fuse_requantshift] Last node is {last_node.op}, not Gemm -- skipping final-requant addition.")
        return graph

    old_out = last_node.outputs[0]
    new_out = gs.Variable(name=old_out.name + "_raw", dtype=old_out.dtype, shape=old_out.shape)
    last_node.outputs = [new_out]

    rqs = gs.Node(
        op="RequantShift",
        name="RequantShift_final",
        inputs=[new_out,
                gs.Constant(name="RequantShift_final_mul", values=np.array([1.0], dtype=np.float32)),
                gs.Constant(name="RequantShift_final_add", values=np.array([0.0], dtype=np.float32))],
        outputs=[old_out],
        attrs={
            "div": gs.Constant(name="RequantShift_final_div", values=np.array(div, dtype=np.float32)),
            "n_levels_out": gs.Constant(name="RequantShift_final_n_levels_out", values=np.array([256.0], dtype=np.float32)),
            "signed": gs.Constant(name="RequantShift_final_signed", values=np.array([1.0], dtype=np.float32)),
        },
    )
    rqs.domain = "PACTOps"
    graph.nodes.append(rqs)
    graph.cleanup().toposort()
    print(f"[fuse_requantshift] Added a final RequantShift (div={div}) after the classifier Gemm.")
    return graph


def strip_nonstandard_attrs(graph: gs.Graph) -> gs.Graph:
    """IntegerizePACTNetPass annotates Conv modules with 'weight_bits'/'bias_bits'
    precision metadata that torch.onnx.export carries straight through as literal
    Conv node attributes -- not part of the ONNX Conv op spec, so onnx.checker (and
    Deeploy's own parser, which reads precision from RequantShift instead) rejects
    them. Same "non-essential metadata" DORY annotations export_net's own
    annotate_onnx() step already treats as skippable -- just never stripped from
    the raw graph before now. Applied to every node/attr regardless of op type:
    precision annotations can land on Conv, Gemm, or a residual Add alike."""
    nonstandard = {"weight_bits", "bias_bits", "add_bits", "mult_bits", "out_bits"}
    n_stripped = 0
    for node in graph.nodes:
        for attr_name in list(node.attrs.keys()):
            if attr_name in nonstandard:
                del node.attrs[attr_name]
                n_stripped += 1
    print(f"[fuse_requantshift] Stripped {n_stripped} non-standard precision attribute(s).")
    return graph


def cast_int_dtypes(graph: gs.Graph) -> gs.Graph:
    """Deeploy's PULP binders type-check literally (PointerClass(int8_t)/(int32_t)),
    not "is this float32 tensor numerically integer-valued" -- they reject a
    Conv/RequantShift input that is still float32/float64 even when every value in it
    is a whole number. quantlib's IntegerizePACTNetPass only performs this int8/int32
    dtype cast for its own recognized PACTConv2d->BatchNorm2d->PACTAct pattern
    (registered architectures like ResNet, used by ImageClassification/
    AnomalyDetection); a hand-rolled nn.Module wrapped generically
    (export_generic_cnn_random.py's Conv2d/Linear/ReLU-only recipe, used for
    KeywordSpotting/VisualWakeWords) keeps its BatchNorm2d as a literal float
    Mul/Add pair that pass never touches, so neither the preceding Conv weight nor
    the BN-derived scale/offset ever gets cast -- confirmed via KeywordSpotting_small
    failing to parse in Deeploy ("Did not find adequate mapping for graph") with
    weight/mul dtype float32 and add dtype float64, vs int8/int32/int32 for the
    working ImageClassification reference. Assert-checked round-trip (not merely
    rounded) before casting: a real fractional value here would silently corrupt
    results rather than just misdeclare a dtype."""
    n_cast = 0

    def _cast(tensor, dtype, label):
        nonlocal n_cast
        if isinstance(tensor, gs.Constant) and tensor.values.dtype != dtype:
            rounded = np.round(tensor.values)
            assert np.allclose(tensor.values, rounded, atol = 1e-3), \
                f"{label} is not integer-valued ({tensor.values.dtype}): cannot cast to {dtype}"
            tensor.values = rounded.astype(dtype)
            n_cast += 1

    for node in graph.nodes:
        if node.op in ("Conv", "Gemm") and len(node.inputs) >= 2:
            _cast(node.inputs[1], np.int8, f"{node.name} weight")
            if len(node.inputs) >= 3:
                _cast(node.inputs[2], np.int32, f"{node.name} bias")
        elif node.op == "RequantShift" and len(node.inputs) >= 3:
            _cast(node.inputs[1], np.int32, f"{node.name} mul")
            _cast(node.inputs[2], np.int32, f"{node.name} add")

    for inp in graph.inputs:
        if inp.dtype != np.int8:
            inp.dtype = np.int8

    print(f"[fuse_requantshift] Cast {n_cast} non-integer-dtype weight/mul/add tensor(s) to int8/int32.")
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--final-requant-div", type=float, default=None,
                     help="If set, add a final mul=1/add=0 RequantShift with this divisor "
                          "after a trailing Gemm that has no trailing quantizer of its own.")
    args = ap.parse_args()

    model = onnx.load(args.input)
    graph = gs.import_onnx(model)
    # Deeploy's frontend reads shapes straight out of the ONNX file's own
    # value_info entries -- it does not re-run shape inference itself. Standard
    # onnx.shape_inference.infer_shapes() stops propagating the moment it hits a
    # custom op it doesn't recognize (RequantShift, once fused), leaving every
    # downstream tensor's shape unresolved. Strip the non-standard attributes
    # and run shape inference on the still-UNFUSED graph (every op still
    # standard) first, then re-import: graphsurgeon's Variable objects carry
    # their now-known .shape through the fuse() below, which reuses the same
    # tensor objects for each chain's input/output, so gs.export_onnx() at the
    # end still emits complete value_info despite the custom op.
    graph = strip_nonstandard_attrs(graph)
    model_for_shapes = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    graph = gs.import_onnx(model_for_shapes)

    graph = fuse(graph)
    graph = merge_consecutive(graph)
    graph = add_conv_kernel_shape(graph)
    graph = convert_global_avg_pool(graph)
    graph = fix_reduce_mean_axes(graph)
    if args.final_requant_div is not None:
        graph = add_final_requant(graph, args.final_requant_div)
    graph = cast_int_dtypes(graph)
    exported = gs.export_onnx(graph)
    # RequantShift is a custom op (matches the domain Deeploy's own
    # create_deeploy_int8_test_from_pytorch.py path already emits) -- declare
    # it in opset_import or onnx.checker (and Deeploy's own loader) reject it.
    # onnx_graphsurgeon's gs.Node.domain assignment (set on rqs_node/merged/rqs
    # above) isn't actually preserved by this version's gs.export_onnx() -- set it
    # directly on the exported protobuf's nodes instead.
    for node in exported.graph.node:
        if node.op_type == "RequantShift":
            node.domain = "PACTOps"
    exported.opset_import.append(onnx.helper.make_opsetid("PACTOps", 1))
    onnx.save(exported, args.output)
    print(f"[fuse_requantshift] Wrote {args.output}")


if __name__ == "__main__":
    main()
