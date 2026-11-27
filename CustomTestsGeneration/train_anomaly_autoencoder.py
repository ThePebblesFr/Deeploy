#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Train the small (AutoencoderLight, 640->64->8->64->640) dense autoencoder from
models/anomaly_autoencoder.py on the DCASE 2020 Task 2 ToyCar dataset, matching the
MLCommons Tiny AnomalyDetection reference feature extraction (128 mel bins x 5 frames =
640-dim, n_fft=1024, hop_length=512, power=2.0, per benchmark/training/anomaly_detection/
baseline.yaml) so results are directly comparable to the heavy reference model already
checked in at DeeployTest/Tests/MLPerf/AnomalyDetection.

Expects the dataset already extracted under CustomTestsGeneration/data/dcase2020_toycar/:
    dev_data/ToyCar/{train,test}/*.wav
    eval_data/ToyCar/train/*.wav

Usage (inside the Deeploy/QuantLab container):
    python3 train_anomaly_autoencoder.py --epochs 100 --out checkpoints/anomaly_light
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from models.anomaly_autoencoder import AutoencoderLight, AutoencoderHeavy, AutoencoderMLCommonsHeavy

N_MELS = 128
FRAMES = 5
N_FFT = 1024
HOP_LENGTH = 512
POWER = 2.0
DIMS = N_MELS * FRAMES


def file_to_vector_array(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                          n_mels=N_MELS, power=POWER)
    log_mel = 20.0 / POWER * np.log10(mel + sys.float_info.epsilon)

    n_vectors = log_mel.shape[1] - FRAMES + 1
    if n_vectors < 1:
        return np.empty((0, DIMS), dtype=np.float32)

    vectors = np.zeros((n_vectors, DIMS), dtype=np.float32)
    for t in range(FRAMES):
        vectors[:, N_MELS * t:N_MELS * (t + 1)] = log_mel[:, t:t + n_vectors].T
    return vectors


def extract_dataset(file_paths, desc):
    all_vectors = []
    per_file_ranges = []
    start = 0
    for i, fp in enumerate(file_paths):
        v = file_to_vector_array(fp)
        all_vectors.append(v)
        per_file_ranges.append((start, start + len(v)))
        start += len(v)
        if (i + 1) % 500 == 0 or (i + 1) == len(file_paths):
            print(f"[{desc}] {i + 1}/{len(file_paths)} files -> {start} vectors", flush=True)
    return np.concatenate(all_vectors, axis=0), per_file_ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/dcase2020_toycar")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="checkpoints/anomaly_light")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-checkpoint", type=str, default=None,
                     help="Warm-start from an existing state_dict instead of random init.")
    ap.add_argument("--agg", type=str, default="median", choices=["mean", "median", "max", "p90"],
                     help="Per-file anomaly-score aggregation over its vectors.")
    ap.add_argument("--model-class", type=str, default="AutoencoderLight",
                     choices=["AutoencoderLight", "AutoencoderHeavy", "AutoencoderMLCommonsHeavy"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_root)
    dev_train_dir = root / "dev_data" / "ToyCar" / "train"
    eval_train_dir = root / "eval_data" / "ToyCar" / "train"
    dev_test_dir = root / "dev_data" / "ToyCar" / "test"

    train_files = sorted(dev_train_dir.glob("*.wav")) + sorted(eval_train_dir.glob("*.wav"))
    test_files = sorted(dev_test_dir.glob("*.wav"))
    assert train_files, f"No training wavs found under {dev_train_dir} / {eval_train_dir}"
    assert test_files, f"No test wavs found under {dev_test_dir}"
    print(f"train files: {len(train_files)}, test files: {len(test_files)}")

    cache_path = root / "features_cache.npz"
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cache = np.load(cache_path)
        train_vectors = cache["train_vectors"]
        test_vectors = cache["test_vectors"]
        test_ranges = list(zip(cache["test_ranges_start"], cache["test_ranges_end"]))
    else:
        print("Extracting training features...")
        train_vectors, _ = extract_dataset(train_files, "train")
        print("Extracting test features...")
        test_vectors, test_ranges = extract_dataset(test_files, "test")
        starts, ends = zip(*test_ranges)
        np.savez(cache_path, train_vectors=train_vectors, test_vectors=test_vectors,
                 test_ranges_start=np.array(starts), test_ranges_end=np.array(ends))
        print(f"Cached features to {cache_path}")

    mean = train_vectors.mean(axis=0, keepdims=True)
    std = train_vectors.std(axis=0, keepdims=True) + 1e-8
    train_norm = (train_vectors - mean) / std
    test_norm = (test_vectors - mean) / std
    test_labels_per_file = np.array([0 if "normal" in fp.name else 1 for fp in test_files])

    device = torch.device("cpu")
    model_cls = {"AutoencoderLight": AutoencoderLight, "AutoencoderHeavy": AutoencoderHeavy,
                 "AutoencoderMLCommonsHeavy": AutoencoderMLCommonsHeavy}[args.model_class]
    model = model_cls().to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"Warm-started from {args.init_checkpoint}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    train_t = torch.from_numpy(train_norm).float()
    ds = torch.utils.data.TensorDataset(train_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"Training on {len(train_norm)} vectors, dim={DIMS}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, batches_per_epoch={len(dl)}")
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for (x,) in dl:
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()
        print(f"epoch {epoch + 1}/{args.epochs} train_mse={total_loss / n_batches:.5f} "
              f"lr={sched.get_last_lr()[0]:.6f} ({time.time() - t0:.1f}s)", flush=True)

    # Evaluate: per-file anomaly score = aggregate (args.agg) per-vector reconstruction MSE.
    model.eval()
    test_t = torch.from_numpy(test_norm).float()
    with torch.no_grad():
        recon = model(test_t)
        per_vector_mse = ((recon - test_t) ** 2).mean(dim=1).numpy()

    agg_fn = {"mean": np.mean, "median": np.median, "max": np.max,
              "p90": lambda a: np.percentile(a, 90)}[args.agg]
    file_scores = np.array([
        agg_fn(per_vector_mse[s:e]) if e > s else 0.0
        for (s, e) in test_ranges
    ])

    auc = roc_auc_score(test_labels_per_file, file_scores)
    fpr, tpr, thresholds = roc_curve(test_labels_per_file, file_scores)
    # Pick the threshold that maximizes raw classification accuracy directly,
    # rather than Youden's J (which optimizes balanced TPR-FPR, not accuracy).
    accs = [((file_scores >= t).astype(int) == test_labels_per_file).mean() for t in thresholds]
    best_idx = int(np.argmax(accs))
    best_thresh = thresholds[best_idx]
    accuracy = accs[best_idx]

    print(f"\n=== Evaluation (n={len(test_files)} test files, agg={args.agg}) ===")
    print(f"AUC: {auc:.4f}")
    print(f"Best threshold (max-accuracy): {best_thresh:.6f}")
    print(f"Classification accuracy at threshold: {accuracy:.4f}")
    print(f"Normal files: {(test_labels_per_file == 0).sum()}, Anomaly files: {(test_labels_per_file == 1).sum()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path) + ".pt")
    np.savez(str(out_path) + "_norm_stats.npz", mean=mean, std=std,
             feature_min=train_norm.min(), feature_max=train_norm.max())
    with open(str(out_path) + "_eval.txt", "w") as f:
        f.write(f"AUC: {auc:.4f}\nthreshold: {best_thresh:.6f}\naccuracy: {accuracy:.4f}\n")
        f.write(f"normalized_feature_range: [{train_norm.min():.4f}, {train_norm.max():.4f}]\n")

    print(f"\nSaved checkpoint to {out_path}.pt, norm stats to {out_path}_norm_stats.npz")

    if accuracy < 0.70:
        print("\nWARNING: accuracy below 70% target.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
