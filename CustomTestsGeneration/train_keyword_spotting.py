#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Train the KSHeavy/KSSmall DS-CNN models (models/keyword_spotting.py) on Google Speech
Commands v0.02, matching the MLCommons Tiny KS benchmark task: 12-way classification
of 1-second audio clips into {yes, no, up, down, left, right, on, off, stop, go,
_unknown_, _silence_}, using a 49x10 log-mel-ish MFCC feature map (49 frames x 10
MFCC coefficients) as input, per benchmark/training/keyword_spotting/keras_model.py's
own feature extraction (n_mfcc=10, frame_len=40ms/20ms hop -> 49 frames over 1s).

Expects the dataset already extracted under CustomTestsGeneration/data/speech_commands/
(from speech_commands_v0.02.tar.gz) -- each keyword in its own subfolder of .wav files,
plus a _background_noise_/ folder used to synthesize "_silence_" clips, and
validation_list.txt/testing_list.txt defining the official split.

Usage (inside the container):
    python3 train_keyword_spotting.py --model-class KSHeavy --epochs 30 \
        --out checkpoints/ks_heavy
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from models.keyword_spotting import KSHeavy, KSSmall

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE  # 1 second
N_MFCC = 10
FRAME_LEN = int(0.040 * SAMPLE_RATE)   # 640
FRAME_HOP = int(0.020 * SAMPLE_RATE)   # 320 -> 49 frames over 16000 samples
TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = TARGET_WORDS + ["_unknown_", "_silence_"]
LABEL_TO_IDX = {w: i for i, w in enumerate(LABELS)}


def wav_to_mfcc(y):
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    else:
        y = y[:CLIP_SAMPLES]
    mfcc = librosa.feature.mfcc(y = y, sr = SAMPLE_RATE, n_mfcc = N_MFCC,
                                 n_fft = FRAME_LEN, hop_length = FRAME_HOP, center = False)
    # mfcc: (N_MFCC, n_frames) -> pad/crop time axis to exactly 49 frames
    n_frames = 49
    if mfcc.shape[1] < n_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, n_frames - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :n_frames]
    return mfcc.T.astype(np.float32)  # (49, 10)


def load_split(root: Path, split_files: set, background_noise, rng, n_silence):
    feats, labels = [], []
    for word_dir in sorted(root.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("_"):
            continue
        label = word_dir.name if word_dir.name in TARGET_WORDS else "_unknown_"
        for wav_path in word_dir.glob("*.wav"):
            rel = f"{word_dir.name}/{wav_path.name}"
            if rel not in split_files:
                continue
            y, _ = librosa.load(str(wav_path), sr = SAMPLE_RATE, mono = True)
            feats.append(wav_to_mfcc(y))
            labels.append(LABEL_TO_IDX[label])

    for _ in range(n_silence):
        noise = background_noise[rng.randint(len(background_noise))]
        start = rng.randint(0, max(1, len(noise) - CLIP_SAMPLES))
        clip = noise[start:start + CLIP_SAMPLES] * rng.uniform(0.1, 1.0)
        feats.append(wav_to_mfcc(clip))
        labels.append(LABEL_TO_IDX["_silence_"])

    return np.stack(feats), np.array(labels, dtype = np.int64)


def read_split_list(root: Path, name: str) -> set:
    with open(root / name) as f:
        return {line.strip() for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type = str, default = "data/speech_commands")
    ap.add_argument("--epochs", type = int, default = 30)
    ap.add_argument("--batch-size", type = int, default = 128)
    ap.add_argument("--lr", type = float, default = 1e-3)
    ap.add_argument("--out", type = str, default = "checkpoints/ks_heavy")
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--model-class", type = str, default = "KSHeavy", choices = ["KSHeavy", "KSSmall"])
    ap.add_argument("--cache", type = str, default = None,
                     help = "Optional .npz path to cache extracted features across runs.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    root = Path(args.data_root)

    cache_path = Path(args.cache) if args.cache else (root / "features_cache.npz")
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        cache = np.load(cache_path)
        train_x, train_y = cache["train_x"], cache["train_y"]
        val_x, val_y = cache["val_x"], cache["val_y"]
        test_x, test_y = cache["test_x"], cache["test_y"]
    else:
        val_files = read_split_list(root, "validation_list.txt")
        test_files = read_split_list(root, "testing_list.txt")
        all_word_files = set()
        for word_dir in sorted(root.iterdir()):
            if word_dir.is_dir() and not word_dir.name.startswith("_"):
                for wav_path in word_dir.glob("*.wav"):
                    all_word_files.add(f"{word_dir.name}/{wav_path.name}")
        train_files = all_word_files - val_files - test_files

        noise_dir = root / "_background_noise_"
        background_noise = [librosa.load(str(p), sr = SAMPLE_RATE, mono = True)[0]
                             for p in sorted(noise_dir.glob("*.wav"))]

        print("Extracting train split...")
        train_x, train_y = load_split(root, train_files, background_noise, rng,
                                       n_silence = int(len(train_files) * 0.1))
        print(f"  {len(train_x)} clips")
        print("Extracting validation split...")
        val_x, val_y = load_split(root, val_files, background_noise, rng,
                                   n_silence = int(len(val_files) * 0.1))
        print(f"  {len(val_x)} clips")
        print("Extracting test split...")
        test_x, test_y = load_split(root, test_files, background_noise, rng,
                                     n_silence = int(len(test_files) * 0.1))
        print(f"  {len(test_x)} clips")

        np.savez(cache_path, train_x = train_x, train_y = train_y,
                  val_x = val_x, val_y = val_y, test_x = test_x, test_y = test_y)
        print(f"Cached features to {cache_path}")

    mean = train_x.mean()
    std = train_x.std() + 1e-8
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    device = torch.device("cpu")
    model_cls = {"KSHeavy": KSHeavy, "KSSmall": KSSmall}[args.model_class]
    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    train_t = torch.from_numpy(train_x).unsqueeze(1).float()  # (N, 1, 49, 10)
    train_lbl = torch.from_numpy(train_y).long()
    ds = torch.utils.data.TensorDataset(train_t, train_lbl)
    dl = torch.utils.data.DataLoader(ds, batch_size = args.batch_size, shuffle = True, drop_last = True)

    val_t = torch.from_numpy(val_x).unsqueeze(1).float()
    val_lbl = torch.from_numpy(val_y).long()

    print(f"Training {args.model_class} on {len(train_x)} clips, epochs={args.epochs}, "
          f"batches_per_epoch={len(dl)}")
    best_val_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim = 1) == yb).sum().item()
            total += len(yb)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_t)
            val_acc = (val_logits.argmax(dim = 1) == val_lbl).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # Persist on every improvement -- a run killed before the epoch budget
            # ends would otherwise lose its best checkpoint (it never reaches the
            # post-loop torch.save() below).
            out_path = Path(args.out)
            out_path.parent.mkdir(parents = True, exist_ok = True)
            torch.save(best_state, str(out_path) + ".pt")

        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / len(dl):.4f} "
              f"train_acc={correct / total:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.1f}s)",
              flush = True)

    model.load_state_dict(best_state)
    model.eval()
    test_t = torch.from_numpy(test_x).unsqueeze(1).float()
    test_lbl = torch.from_numpy(test_y).long()
    with torch.no_grad():
        test_logits = model(test_t)
        test_acc = (test_logits.argmax(dim = 1) == test_lbl).float().mean().item()

    print(f"\n=== Evaluation ({len(test_x)} test clips, best-val checkpoint) ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(model.state_dict(), str(out_path) + ".pt")
    np.savez(str(out_path) + "_norm_stats.npz", mean = mean, std = std)
    with open(str(out_path) + "_eval.txt", "w") as f:
        f.write(f"test_accuracy: {test_acc:.4f}\nbest_val_accuracy: {best_val_acc:.4f}\n")

    print(f"\nSaved checkpoint to {out_path}.pt, norm stats to {out_path}_norm_stats.npz")


if __name__ == "__main__":
    main()
