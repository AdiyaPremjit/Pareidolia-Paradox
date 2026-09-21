# The Pareidolia Paradox

Binary classification of 256x256 grayscale lunar surface images into:
- **Class 0 (Depth):** craters, holes, surface depressions
- **Class 1 (Rise):** mounds, hills, rocks, boulders

## Methodology Summary

The core difficulty in this task is that a crater and a mound can produce a
visually identical shadow pattern depending on the direction sunlight is
coming from — the same physical bump looks like a hole or a hill purely
based on lighting angle. Left unaddressed, this makes the two classes
indistinguishable to a model, since shadow shape (not true depth) becomes
the dominant visual cue and that shadow shape varies per-photo.

**Sun-angle normalization.** Every image, in both training and inference,
is rotated counter-clockwise by `-sun_azimuth_angle` (the angle provided in
each image's metadata row) before it reaches the model:

```python
img.rotate(-sun_azimuth_angle, resample=Image.BILINEAR, fillcolor=0, expand=False)
```

This fixes the effective lighting direction to be consistent across the
entire dataset, so "shadow falls this way" reliably maps to one physical
class rather than varying by photo. This preprocessing step is applied
identically in `train.py` and `inference.py`.

**Augmentation caveat.** During development we found that standard
horizontal/vertical flip augmentation actively hurts this task: flipping an
image also flips its (now-normalized) shadow direction, reintroducing the
exact ambiguity the rotation step removes. Validation accuracy improved from
~0.64 to ~0.74 after removing flip augmentation, keeping only brightness/
contrast jitter (which doesn't affect shadow geometry). `train.py` reflects
this — flips are deliberately excluded.

**Model.** ResNet18, pretrained on ImageNet, fine-tuned end-to-end with a
dropout layer (p=0.4) before the final linear layer to control overfitting,
trained with `Adam` (lr=1e-4, weight_decay=1e-4) and early stopping
(patience=3 epochs on validation accuracy).

**Best validation accuracy:** ~0.744 (vs. ~0.637 majority-class baseline).

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── train.py          # Trains the model, saves best_model.pt
└── inference.py       # Loads a checkpoint, generates submission.csv
```

## Setup

```bash
pip install -r requirements.txt
```

## Expected Data Layout

```
<data_dir>/
├── train_images/            # or a single nested subfolder containing the .png files
├── train_metadata.csv       # columns: image_id, sun_azimuth_angle, label
├── test_images/             # or a single nested subfolder containing the .png files
└── test_metadata.csv        # columns: image_id, sun_azimuth_angle
```

Both scripts search recursively under the given image directory, so an
extra nested folder from zip extraction (e.g. `train_images/train_images/`)
is handled automatically.

## Training

```bash
python train.py --data_dir /path/to/data --epochs 15
```

Key options:
| Flag | Default | Description |
|---|---|---|
| `--epochs` | 15 | Max training epochs (early stopping may end sooner) |
| `--batch_size` | 32 | |
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 1e-4 | L2 regularization |
| `--patience` | 3 | Early stopping patience, in epochs |
| `--val_split` | 0.18 | Fraction of training data held out for validation |
| `--output_path` | `<data_dir>/best_model.pt` | Where the best checkpoint is saved |

## Inference

```bash
python inference.py --data_dir /path/to/data --model_path /path/to/best_model.pt
```

Key options:
| Flag | Default | Description |
|---|---|---|
| `--threshold` | 0.5 | Sigmoid probability cutoff for predicting Class 1 |
| `--output_csv` | `<data_dir>/submission.csv` | Where predictions are written |

Output format (`image_id,label`), validated before writing:
- Exactly one row per test image, no missing or duplicate IDs
- No null values
- Labels are exactly 0 or 1

## Model Weights

Trained weights (`best_model.pt`): **[ADD YOUR DOWNLOAD LINK HERE]**

(Upload `best_model.pt` to Google Drive, Hugging Face, or Kaggle and set
sharing to "Anyone with the link can view" before pasting the link above.)
