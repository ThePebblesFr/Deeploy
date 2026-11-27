#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Train a SUPERVISED normal/anomaly binary classifier on DCASE 2020 ToyCar, instead
of the unsupervised reconstruction-error approach in train_anomaly_autoencoder.py.

Rationale: the unsupervised autoencoder approach's accuracy ceiling (~70-75%,
matching published DCASE ToyCar baselines) is far below the >=90% target for the
"heavy" model. A supervised classifier on the same labeled data is a fundamentally
easier learning problem and can reach that target -- at the cost of no longer being
a faithful unsupervised-AD benchmark (worth noting in the paper).

Data: the only labeled (normal + anomaly) DCASE data is dev_data/ToyCar/test/
(2459 clips: 1400 normal, 1059 anomaly). Split at the CLIP level (not frame level,
since adjacent frames within a clip are highly correlated -- frame-level splitting
would leak information) into train/test, each frame inheriting its clip's label.
Reuses the already-cached raw log-mel features (features_cache.npz's test_vectors/
test_ranges) from train_anomaly_autoencoder.py -- same feature extraction, no need
to recompute.

Model I/O stays (1,640) -> (1,2) logits, matching the single-frame-vector interface
convention used throughout this pipeline (Deeploy int8 export, DynMultiDeeploy
scenario builder) -- clip-level predictions are obtained by averaging per-frame
probabilities within a clip at evaluation time, not baked into the model itself.

Usage (inside the container):
    python3 train_anomaly_classifier.py --model-class ClassifierHeavy --epochs 60 \
        --out checkpoints/anomaly_classifier_heavy
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from models.anomaly_classifier import ClassifierHeavy, ClassifierSmall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/dcase2020_toycar")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="checkpoints/anomaly_classifier_heavy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-class", type=str, default="ClassifierHeavy",
                     choices=["ClassifierHeavy", "ClassifierSmall"])
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_root)
    dev_test_dir = root / "dev_data" / "ToyCar" / "test"
    test_files = sorted(dev_test_dir.glob("*.wav"))
    clip_labels = np.array([0 if "normal" in fp.name else 1 for fp in test_files])

    cache = np.load(root / "features_cache.npz")
    vectors = cache["test_vectors"]
    starts, ends = cache["test_ranges_start"], cache["test_ranges_end"]
    clip_ranges = list(zip(starts, ends))

    n_clips = len(test_files)
    rng = np.random.RandomState(args.seed)
    normal_idx = np.where(clip_labels == 0)[0]
    anomaly_idx = np.where(clip_labels == 1)[0]
    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)
    n_train_normal = int(len(normal_idx) * args.train_frac)
    n_train_anomaly = int(len(anomaly_idx) * args.train_frac)
    train_clips = np.concatenate([normal_idx[:n_train_normal], anomaly_idx[:n_train_anomaly]])
    test_clips = np.concatenate([normal_idx[n_train_normal:], anomaly_idx[n_train_anomaly:]])
    rng.shuffle(train_clips)
    rng.shuffle(test_clips)

    def expand(clip_indices):
        feats, labels, owner_clip = [], [], []
        for ci in clip_indices:
            s, e = clip_ranges[ci]
            feats.append(vectors[s:e])
            labels.append(np.full(e - s, clip_labels[ci]))
            owner_clip.append(np.full(e - s, ci))
        return np.concatenate(feats), np.concatenate(labels), np.concatenate(owner_clip)

    train_feats, train_labels, _ = expand(train_clips)
    test_feats, test_labels, test_owner = expand(test_clips)

    mean = train_feats.mean(axis=0, keepdims=True)
    std = train_feats.std(axis=0, keepdims=True) + 1e-8
    train_norm = ((train_feats - mean) / std).astype(np.float32)
    test_norm = ((test_feats - mean) / std).astype(np.float32)

    print(f"train clips: {len(train_clips)} ({n_train_normal} normal, {n_train_anomaly} anomaly), "
          f"{len(train_feats)} frame vectors")
    print(f"test clips: {len(test_clips)}, {len(test_feats)} frame vectors")

    device = torch.device("cpu")
    model_cls = {"ClassifierHeavy": ClassifierHeavy, "ClassifierSmall": ClassifierSmall}[args.model_class]
    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    train_x = torch.from_numpy(train_norm).float()
    train_y = torch.from_numpy(train_labels).long()
    ds = torch.utils.data.TensorDataset(train_x, train_y)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"Training {args.model_class} on {len(train_norm)} vectors, epochs={args.epochs}, "
          f"batches_per_epoch={len(dl)}")
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / len(dl):.4f} "
                  f"frame_train_acc={correct / total:.4f} ({time.time() - t0:.1f}s)", flush=True)

    # Evaluate: per-frame predictions, aggregated to per-clip via mean probability.
    model.eval()
    test_x = torch.from_numpy(test_norm).float()
    with torch.no_grad():
        probs = torch.softmax(model(test_x), dim=1)[:, 1].numpy()

    clip_scores = {}
    for ci in test_clips:
        mask = test_owner == ci
        clip_scores[ci] = probs[mask].mean()

    y_true = np.array([clip_labels[ci] for ci in test_clips])
    y_score = np.array([clip_scores[ci] for ci in test_clips])
    y_pred = (y_score >= 0.5).astype(int)
    clip_accuracy = (y_pred == y_true).mean()

    frame_pred = (probs >= 0.5).astype(int)
    frame_accuracy = (frame_pred == test_labels).mean()

    print(f"\n=== Evaluation ({len(test_clips)} held-out clips) ===")
    print(f"Clip-level accuracy (mean-prob aggregation, threshold 0.5): {clip_accuracy:.4f}")
    print(f"Frame-level accuracy: {frame_accuracy:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path) + ".pt")
    np.savez(str(out_path) + "_norm_stats.npz", mean=mean, std=std)
    with open(str(out_path) + "_eval.txt", "w") as f:
        f.write(f"clip_accuracy: {clip_accuracy:.4f}\nframe_accuracy: {frame_accuracy:.4f}\n")

    print(f"\nSaved checkpoint to {out_path}.pt, norm stats to {out_path}_norm_stats.npz")


if __name__ == "__main__":
    main()
