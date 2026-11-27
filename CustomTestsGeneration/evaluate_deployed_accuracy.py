#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Evaluate the actual DEPLOYED (post-quantization) accuracy of the 4 int8
network.onnx graphs in DeeployTest/Tests/{ImageClassification,AnomalyDetection}_
{heavy,small}, on their real test sets -- not the fake-quant/float PyTorch
accuracy measured during training, the real thing that will run on the cluster.

Uses onnx.reference.ReferenceEvaluator with custom op kernels for RequantShift and
RequantizedGemm (Deeploy-specific ops with no standard ONNX semantics); all other
ops (Conv, Add, Gemm, MaxPool, ReduceMean, Reshape) use onnx.reference's own
built-in kernels.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

sys.path.insert(0, str(Path(__file__).parent))


class RequantShift(OpRun):
    op_domain = "PACTOps"

    def _run(self, x, mul, add, div=None, n_levels=None, n_levels_out=None, signed=None):
        y = x.astype(np.int64) * mul.astype(np.int64) + add.astype(np.int64)
        y = np.floor(y / float(div))
        levels = int((n_levels if n_levels is not None else n_levels_out).reshape(-1)[0])
        is_signed = bool(int(np.array(signed).reshape(-1)[0]))
        if is_signed:
            c = int(np.floor(levels / 2 + 0.001))
            lo, hi = (-c, c) if levels % 2 else (-c, c - 1)
        else:
            lo, hi = 0, levels - 1
        y = np.clip(y, lo, hi)
        return (y.astype(np.float32),)


class RequantizedGemm(OpRun):
    op_domain = "PACTOps"

    def _run(self, A, B, C, M, alpha=1.0, beta=1.0, transA=0, transB=0,
             div=None, n_levels=None, signed=None, shift=None):
        a = A.astype(np.int64)
        b = B.astype(np.int64)
        if transB:
            phi = a @ b.T
        else:
            phi = a @ b
        y = phi * M.astype(np.int64) + C.astype(np.int64)
        y = np.floor(y / float(div))
        levels = int(np.array(n_levels).reshape(-1)[0])
        is_signed = bool(int(np.array(signed).reshape(-1)[0]))
        if is_signed:
            c = int(np.floor(levels / 2 + 0.001))
            lo, hi = (-c, c) if levels % 2 else (-c, c - 1)
        else:
            lo, hi = 0, levels - 1
        y = np.clip(y, lo, hi)
        return (y.astype(np.float32),)


def load_evaluator(onnx_path):
    model = onnx.load(onnx_path)
    return ReferenceEvaluator(model, new_ops=[RequantShift, RequantizedGemm])


# -----------------------------------------------------------------------------
# ImageClassification evaluation (CIFAR-10 test set, top-1 accuracy)
# -----------------------------------------------------------------------------

def eval_image_classification(onnx_path, quant_mode, n_samples=None):
    """quant_mode: 'raw_uint8' (heavy) or 'quantlab_pact' (small, via quantlab's own transform)."""
    import torch
    sys.path.insert(0, "/app/Deeploy/quantlab")

    ev = load_evaluator(onnx_path)
    input_name = ev.input_names[0]

    if quant_mode == "raw_uint8":
        import torchvision
        ds = torchvision.datasets.CIFAR10(
            root="/app/Deeploy/quantlab_storage/data/quantlab/systems/CIFAR10/data",
            train=False, download=False)
        n = n_samples or len(ds)
        correct = 0
        for i in range(n):
            img, label = ds[i]
            arr = np.array(img, dtype=np.int64).transpose(2, 0, 1)[None, ...]  # HWC->CHW
            out = ev.run(None, {input_name: arr.astype(np.float32)})[0]
            pred = int(np.argmax(out[0]))
            correct += int(pred == label)
        return correct / n, n

    elif quant_mode == "quantlab_pact":
        from examples.fx_integerization.integerize_pactnets import get_valid_dataset, _QUANT_UTILS, get_config
        cfg = get_config("ResNetCIFAR", 0)
        ds = get_valid_dataset("ResNetCIFAR", cfg, quantize="int")
        n = n_samples or len(ds)
        correct = 0
        for i in range(n):
            x, label = ds[i]
            arr = x.unsqueeze(0).numpy().astype(np.float32)
            out = ev.run(None, {input_name: arr})[0]
            pred = int(np.argmax(out[0]))
            correct += int(pred == int(label))
        return correct / n, n
    else:
        raise ValueError(quant_mode)


# -----------------------------------------------------------------------------
# AnomalyDetection evaluation (DCASE ToyCar test set, AUC + best-accuracy)
# -----------------------------------------------------------------------------

def eval_anomaly_detection(onnx_path, mean, std, eps=None, offset=None, assumption_note=""):
    """
    mean/std: per-feature z-score stats (small model's own, from training).
    eps/offset: for heavy model, a best-effort affine quantization (x_q = round((x-offset)/eps))
    calibrated from OUR OWN training data's raw log-mel range, since heavy's original
    calibration isn't documented anywhere in this repo. Flagged explicitly in the report.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    from pathlib import Path as P

    cache = np.load("data/dcase2020_toycar/features_cache.npz")
    test_vectors = cache["test_vectors"]
    starts, ends = cache["test_ranges_start"], cache["test_ranges_end"]
    test_ranges = list(zip(starts, ends))

    dev_test_dir = P("data/dcase2020_toycar/dev_data/ToyCar/test")
    test_files = sorted(dev_test_dir.glob("*.wav"))
    labels = np.array([0 if "normal" in fp.name else 1 for fp in test_files])

    if eps is not None:
        x_q = np.clip(np.round((test_vectors - offset) / eps), 0, 255).astype(np.float32)
    else:
        # The model's input is INT8-quantized (create_deeploy_int8_test_from_pytorch.py's
        # quantize_input_i8), not raw z-scored floats -- must requantize the same way,
        # using the SAME input_eps the export baked into the graph (README.md).
        x_norm = (test_vectors - mean) / std
        input_eps = 0.023608708944846325
        x_q = np.clip(np.round(x_norm / input_eps), -128, 127).astype(np.float32)

    ev = load_evaluator(onnx_path)
    input_name = ev.input_names[0]

    batch = 2048
    recon = np.zeros_like(x_q)
    for s in range(0, len(x_q), batch):
        chunk = x_q[s:s + batch]
        out = ev.run(None, {input_name: chunk})[0]
        recon[s:s + batch] = out

    per_vec_mse = ((recon - x_q) ** 2).mean(axis=1)
    file_scores = np.array([np.median(per_vec_mse[s:e]) if e > s else 0.0 for s, e in test_ranges])

    auc = roc_auc_score(labels, file_scores)
    fpr, tpr, thresholds = roc_curve(labels, file_scores)
    accs = [((file_scores >= t).astype(int) == labels).mean() for t in thresholds]
    best_acc = max(accs)
    return auc, best_acc, assumption_note


def main():
    results = {}

    print("=== ImageClassification_heavy (quantlab test split, final epoch149 ckpt) ===")
    acc, n = eval_image_classification(
        "/tmp/ic_heavy_export/ImageClassification_heavy_fused.onnx", "quantlab_pact")
    print(f"top-1 accuracy: {acc:.4f} (n={n})")
    results["ImageClassification_heavy"] = acc

    print("\n=== ImageClassification_small (quantlab test split, final epoch119 ckpt) ===")
    acc, n = eval_image_classification(
        "/tmp/ic_small_export/ImageClassification_small_fused.onnx", "quantlab_pact")
    print(f"top-1 accuracy: {acc:.4f} (n={n})")
    results["ImageClassification_small"] = acc

    # AnomalyDetection now uses the supervised classifiers (train_anomaly_classifier.py),
    # not the unsupervised autoencoders these AD eval helpers below were written for --
    # skipped here; classifier accuracy is reported by train_anomaly_classifier.py itself.

    print("\n=== Summary ===")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()
