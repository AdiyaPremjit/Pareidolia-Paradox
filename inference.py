"""
inference.py — The Pareidolia Paradox

Loads a trained checkpoint and generates predictions on the test set,
writing a submission.csv with exactly one row per test image in the
format: image_id,label

Usage:
    python inference.py --data_dir /path/to/data --model_path best_model.pt

Expects, under --data_dir:
    test_images/             (or a single nested subfolder containing the images)
    test_metadata.csv        (columns: image_id, sun_azimuth_angle)
"""

import argparse
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_sun_angle(img: Image.Image, sun_azimuth_angle: float) -> Image.Image:
    """Must exactly match the normalization used in train.py — rotates
    counter-clockwise by -sun_azimuth_angle so lighting direction is
    consistent between training and inference."""
    return img.rotate(-sun_azimuth_angle, resample=Image.BILINEAR, fillcolor=0, expand=False)


def find_image_path(image_id, images_dir, _cache={}):
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


class PareidoliaTestDataset(Dataset):
    def __init__(self, df, images_dir):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = find_image_path(row["image_id"], self.images_dir)
        img = Image.open(img_path).convert("L")
        img = normalize_sun_angle(img, row["sun_azimuth_angle"])
        img_t = self.transform(img)
        return img_t, row["image_id"]


def build_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Run inference for the Pareidolia Paradox challenge")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing test_images/ and test_metadata.csv")
    parser.add_argument("--model_path", type=str, required=True,
                         help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Sigmoid probability threshold for predicting class 1")
    parser.add_argument("--output_csv", type=str, default=None,
                         help="Where to write predictions (default: <data_dir>/submission.csv)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_images_dir = os.path.join(args.data_dir, "test_images")
    test_metadata_csv = os.path.join(args.data_dir, "test_metadata.csv")
    output_csv = args.output_csv or os.path.join(args.data_dir, "submission.csv")

    test_meta = pd.read_csv(test_metadata_csv)
    print("Loaded", len(test_meta), "test examples")

    test_ds = PareidoliaTestDataset(test_meta, test_images_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_ids, all_preds = [], []
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader):
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > args.threshold).int().cpu().numpy()
            all_ids.extend(list(image_ids))
            all_preds.extend(preds.tolist())

    submission = pd.DataFrame({"image_id": all_ids, "label": all_preds})

    # Hard checks before saving — fail loudly rather than submit something broken
    assert len(submission) == len(test_meta), \
        f"Row count mismatch: {len(submission)} predictions vs {len(test_meta)} test rows"
    assert submission["image_id"].notnull().all(), "Found null image_id"
    assert submission["label"].notnull().all(), "Found null label"
    assert set(submission["image_id"]) == set(test_meta["image_id"]), \
        "image_id set does not match test_metadata.csv"
    assert submission["label"].isin([0, 1]).all(), "Labels must be exactly 0 or 1"
    assert submission["image_id"].duplicated().sum() == 0, "Duplicate image_id found"

    submission.to_csv(output_csv, index=False)
    print(f"Saved {len(submission)} predictions to {output_csv}")
    print(submission["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
