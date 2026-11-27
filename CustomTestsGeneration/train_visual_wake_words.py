#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Train the VWWHeavy/VWWSmall MobileNetV1-0.25 models (models/visual_wake_words.py) on
the MLCommons Tiny Visual Wake Words task: binary person-present/absent classification
of 96x96 RGB crops, using the official preprocessed vw_coco2014_96 dataset (COCO 2014
images relabeled by person-bbox area and pre-cropped/resized to 96x96 by the benchmark
authors -- https://www.silabs.com/.../vw_coco2014_96.tar.gz), so no raw COCO download
or pyvww relabeling is needed.

Expects CustomTestsGeneration/data/vww/vw_coco2014_96/{person,non_person}/*.jpg
already extracted.

Usage (inside the container):
    python3 train_visual_wake_words.py --model-class VWWHeavy --epochs 15 \
        --out checkpoints/vww_heavy
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from models.visual_wake_words import VWWHeavy, VWWSmall


def load_split(root: Path, seed: int, train_frac: float = 0.8, val_frac: float = 0.1):
    person_files = sorted((root / "person").glob("*.jpg"))
    non_person_files = sorted((root / "non_person").glob("*.jpg"))

    rng = np.random.RandomState(seed)
    rng.shuffle(person_files)
    rng.shuffle(non_person_files)

    def split(files):
        n = len(files)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]

    p_train, p_val, p_test = split(person_files)
    n_train, n_val, n_test = split(non_person_files)

    train = [(f, 1) for f in p_train] + [(f, 0) for f in n_train]
    val = [(f, 1) for f in p_val] + [(f, 0) for f in n_val]
    test = [(f, 1) for f in p_test] + [(f, 0) for f in n_test]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def load_images(entries):
    imgs = np.zeros((len(entries), 96, 96, 3), dtype = np.uint8)
    labels = np.zeros(len(entries), dtype = np.int64)
    for i, (path, label) in enumerate(entries):
        im = Image.open(path).convert("RGB")
        if im.size != (96, 96):
            im = im.resize((96, 96))
        imgs[i] = np.array(im)
        labels[i] = label
    return imgs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type = str, default = "data/vww/vw_coco2014_96")
    ap.add_argument("--epochs", type = int, default = 15)
    ap.add_argument("--batch-size", type = int, default = 128)
    ap.add_argument("--lr", type = float, default = 1e-3)
    ap.add_argument("--weight-decay", type = float, default = 0.0)
    ap.add_argument("--hflip", action = "store_true",
                     help = "Random horizontal flip augmentation (safe for person detection: "
                            "orientation doesn't matter).")
    ap.add_argument("--out", type = str, default = "checkpoints/vww_heavy")
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--model-class", type = str, default = "VWWHeavy", choices = ["VWWHeavy", "VWWSmall"])
    ap.add_argument("--cache", type = str, default = None,
                     help = "Optional .npz path to cache decoded images across runs.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_root)
    cache_path = Path(args.cache) if args.cache else (root.parent / "images_cache.npz")

    if cache_path.exists():
        print(f"Loading cached images from {cache_path}")
        cache = np.load(cache_path)
        train_x, train_y = cache["train_x"], cache["train_y"]
        val_x, val_y = cache["val_x"], cache["val_y"]
        test_x, test_y = cache["test_x"], cache["test_y"]
    else:
        train_entries, val_entries, test_entries = load_split(root, args.seed)
        print(f"train: {len(train_entries)}, val: {len(val_entries)}, test: {len(test_entries)}")
        print("Decoding train images...")
        train_x, train_y = load_images(train_entries)
        print("Decoding val images...")
        val_x, val_y = load_images(val_entries)
        print("Decoding test images...")
        test_x, test_y = load_images(test_entries)
        np.savez(cache_path, train_x = train_x, train_y = train_y,
                  val_x = val_x, val_y = val_y, test_x = test_x, test_y = test_y)
        print(f"Cached images to {cache_path}")

    mean = train_x.reshape(-1, 3).mean(axis = 0).astype(np.float32)
    std = train_x.reshape(-1, 3).std(axis = 0).astype(np.float32) + 1e-8

    def normalize(x):
        return ((x.astype(np.float32) - mean) / std).transpose(0, 3, 1, 2)  # NHWC -> NCHW

    device = torch.device("cpu")
    model_cls = {"VWWHeavy": VWWHeavy, "VWWSmall": VWWSmall}[args.model_class]
    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    train_t = torch.from_numpy(normalize(train_x)).float()
    train_lbl = torch.from_numpy(train_y).long()
    ds = torch.utils.data.TensorDataset(train_t, train_lbl)
    dl = torch.utils.data.DataLoader(ds, batch_size = args.batch_size, shuffle = True, drop_last = True)

    val_t = torch.from_numpy(normalize(val_x)).float()
    val_lbl = torch.from_numpy(val_y).long()

    print(f"Training {args.model_class} on {len(train_x)} images, epochs={args.epochs}, "
          f"batches_per_epoch={len(dl)}")
    best_val_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in dl:
            if args.hflip:
                flip_mask = torch.rand(xb.size(0)) < 0.5
                xb[flip_mask] = xb[flip_mask].flip(dims = [3])
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
            # Persist on every improvement, not just after the full epoch budget: a
            # long run killed early (e.g. to cut a thermal-throttled slowdown short)
            # would otherwise lose its best checkpoint entirely, since it never
            # reaches the post-loop torch.save() below.
            out_path = Path(args.out)
            out_path.parent.mkdir(parents = True, exist_ok = True)
            torch.save(best_state, str(out_path) + ".pt")

        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / len(dl):.4f} "
              f"train_acc={correct / total:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.1f}s)",
              flush = True)

    model.load_state_dict(best_state)
    model.eval()
    test_t = torch.from_numpy(normalize(test_x)).float()
    test_lbl = torch.from_numpy(test_y).long()
    with torch.no_grad():
        test_logits = model(test_t)
        test_acc = (test_logits.argmax(dim = 1) == test_lbl).float().mean().item()

    print(f"\n=== Evaluation ({len(test_x)} test images, best-val checkpoint) ===")
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
