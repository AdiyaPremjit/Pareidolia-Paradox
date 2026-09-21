"""
train.py — The Pareidolia Paradox

Trains a binary classifier (Class 0 = Depth, Class 1 = Rise) on 256x256
grayscale lunar surface images. Every image is rotated by -sun_azimuth_angle
before it ever reaches the model, to normalize lighting direction across the
dataset (see README.md for why this matters).

Usage:
    python train.py --data_dir /path/to/data --epochs 15

Expects, under --data_dir:
    train_images/            (or a single nested subfolder containing the images)
    train_metadata.csv       (columns: image_id, sun_azimuth_angle, label)
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

SEED = 42
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_sun_angle(img: Image.Image, sun_azimuth_angle: float) -> Image.Image:
    """
    Rotates the image counter-clockwise by -sun_azimuth_angle, as specified
    by the challenge, so that every image has a consistent lighting direction
    before training. Without this, the same physical shape (crater or mound)
    can produce a near-identical shadow pattern depending on the original sun
    angle, making the two classes indistinguishable to the model.
    """
    return img.rotate(-sun_azimuth_angle, resample=Image.BILINEAR, fillcolor=0, expand=False)


def find_image_path(image_id, images_dir, _cache={}):
    """Resolves image_id to a full path, searching the whole folder tree once
    and caching the result — handles zip extractions that produce an extra
    nested subfolder without needing the caller to know about it."""
    if images_dir not in _cache:
        index = {}
        for root, _, files in os.walk(images_dir):
            for f in files:
                index[f] = os.path.join(root, f)
        _cache[images_dir] = index
    index = _cache[images_dir]
    fname = str(image_id)
    if fname in index:
        return index[fname]
    raise FileNotFoundError(f"No image found for id {image_id} anywhere under {images_dir}")


class PareidoliaDataset(Dataset):
    """
    NOTE: deliberately does NOT include horizontal/vertical flip augmentation.
    Flipping an image also flips its shadow direction, which reintroduces the
    exact lighting ambiguity that normalize_sun_angle() removes. In testing,
    removing flips raised validation accuracy from ~0.64 to ~0.74 on this task.
    """

    def __init__(self, df, images_dir, train=True, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.train = train
        self.has_labels = has_labels

        aug = []
        if train:
            aug += [T.ColorJitter(brightness=0.15, contrast=0.15)]
        aug += [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform = T.Compose(aug)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = find_image_path(row["image_id"], self.images_dir)
        img = Image.open(img_path).convert("L")
        img = normalize_sun_angle(img, row["sun_azimuth_angle"])
        img_t = self.transform(img)

        if self.has_labels:
            return img_t, int(row["label"])
        return img_t, row["image_id"]


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1),
    )
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(is_train)

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.float().to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the Pareidolia Paradox classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing train_images/ and train_metadata.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--val_split", type=float, default=0.18)
    parser.add_argument("--output_path", type=str, default=None,
                         help="Where to save the best model checkpoint (default: <data_dir>/best_model.pt)")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_images_dir = os.path.join(args.data_dir, "train_images")
    train_metadata_csv = os.path.join(args.data_dir, "train_metadata.csv")
    output_path = args.output_path or os.path.join(args.data_dir, "best_model.pt")

    train_meta = pd.read_csv(train_metadata_csv)
    print("Loaded", len(train_meta), "training examples")
    print(train_meta["label"].value_counts(normalize=True))

    train_df, val_df = train_test_split(
        train_meta, test_size=args.val_split, stratify=train_meta["label"], random_state=SEED
    )

    train_ds = PareidoliaDataset(train_df, train_images_dir, train=True, has_labels=True)
    val_ds = PareidoliaDataset(val_df, train_images_dir, train=False, has_labels=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_path)
            print(f"  -> New best val_acc {val_acc:.4f}, checkpoint saved to {output_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print("Training complete. Best validation accuracy:", best_val_acc)


if __name__ == "__main__":
    main()
