"""
04_augmentation.py
==================
PyTorch Dataset class + GPU-ready DataLoaders for HAGCA-Net training.

Reads from the processed CSV files produced by 03_preprocessing.py:
  data/processed/train_processed.csv
  data/processed/val_processed.csv
  data/processed/test_processed.csv

Each CSV has columns: src_path, out_path, label, label_idx, split, group_id

Usage (after preprocessing):
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\04_augmentation.py
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── bring config into scope ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import CFG, setup_device

# ── image size (must match model backbone input) ─────────────────────────────
IMG_SIZE = 224   # EfficientNet-B3 / Swin-Base both accept 224

# ── ImageNet mean/std (used by all timm pretrained models) ───────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════════════════════
#  1.  Transform pipelines
# ════════════════════════════════════════════════════════════════════════════

def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline for training images.
    All operations run on CPU (PIL/tensor) — batches are then .to(device) in
    the training loop.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        # ── geometric augmentations ─────────────────────────────────────────
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        # ── colour / texture augmentations ──────────────────────────────────
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05
        ),
        transforms.RandomGrayscale(p=0.02),

        # ── zoom / crop augmentation ─────────────────────────────────────────
        transforms.RandomResizedCrop(
            size=IMG_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),

        # ── convert to tensor + normalise ───────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # ── random erasing (occlusion robustness) ───────────────────────────
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Deterministic pipeline for validation and test images.
    No augmentation — only resize + normalise.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ════════════════════════════════════════════════════════════════════════════
#  2.  Dataset
# ════════════════════════════════════════════════════════════════════════════

class LungDataset(Dataset):
    """
    Reads preprocessed lung histopathology images.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'out_path' (preprocessed image path) and 'label_idx'.
    transform : callable, optional
        torchvision transform applied to each PIL image.
    use_processed : bool
        If True (default), load from 'out_path' (preprocessed).
        If False, fall back to 'src_path' (raw) — useful for debugging.
    """

    def __init__(
        self,
        df,
        transform=None,
        use_processed: bool = True,
    ):
        self.df            = df.reset_index(drop=True)
        self.transform     = transform
        self.use_processed = use_processed
        self.path_col      = "processed_path" if use_processed else "filepath"
        # (Path validation removed — preprocessing already confirmed all files exist)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        path  = row[self.path_col]
        label = int(row["label_idx"])

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[LungDataset] Failed to load {path}: {e}")
            img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)

        return img, label


# ════════════════════════════════════════════════════════════════════════════
#  2b. Cached Dataset — pre-loads ALL images into RAM as uint8 arrays.
#
#  Why: with num_workers=0 (Windows), disk I/O is the bottleneck.
#       The GPU finishes each batch in <1s then idles waiting for CPU to
#       read the next batch from disk.  Caching moves all images to RAM
#       once (~30s one-time cost) so the GPU never waits for disk again.
#
#  Memory: 9936 images × 224×224×3 bytes ≈ 1.5 GB RAM.  Safe on any
#  machine with ≥8 GB RAM.
# ════════════════════════════════════════════════════════════════════════════

class CachedLungDataset(Dataset):
    """
    Preloads every image into RAM as a uint8 numpy array.
    Transforms are still applied per-sample (so augmentation stays random),
    but there is zero disk I/O after the initial load.
    """

    def __init__(self, df, transform=None, use_processed: bool = True):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        path_col       = "processed_path" if use_processed else "filepath"

        n = len(self.df)
        print(f"[Cache] Loading {n} images into RAM "
              f"(~{n * IMG_SIZE * IMG_SIZE * 3 / 1e9:.1f} GB) ...", flush=True)

        self.images = []
        self.labels = []
        for i, row in self.df.iterrows():
            try:
                arr = np.array(Image.open(row[path_col]).convert("RGB"), dtype=np.uint8)
            except Exception:
                arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            self.images.append(arr)
            self.labels.append(int(row["label_idx"]))
            if (len(self.images)) % 2000 == 0:
                print(f"  ... {len(self.images)}/{n}", flush=True)

        print(f"[Cache] Done — {n} images in RAM.", flush=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ════════════════════════════════════════════════════════════════════════════
#  3.  DataLoader factory
# ════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    batch_size=None,
    num_workers=None,
    pin_memory=None,
):
    """
    Build train / val / test DataLoaders from processed CSV files.
    Returns: (loaders_dict, class_weights_tensor)
    """
    bs  = batch_size  if batch_size  is not None else CFG.BATCH_SIZE
    nw  = num_workers if num_workers is not None else CFG.DATALOADER_WORKERS
    pin = pin_memory  if pin_memory  is not None else CFG.PIN_MEMORY
    # pin_memory only meaningful with CUDA; disable when nw=0 isn't needed but harmless
    if nw == 0:
        pin = False   # avoids a spurious warning on some torch versions

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    print(f"[DataLoader] Loaded CSVs — "
          f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_ds = LungDataset(train_df, transform=get_train_transforms())
    val_ds   = LungDataset(val_df,   transform=get_val_transforms())
    test_ds  = LungDataset(test_df,  transform=get_val_transforms())

    # Class weights for weighted loss (handles any class imbalance)
    class_counts  = train_df["label_idx"].value_counts().sort_index().values
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
            drop_last=True,
            persistent_workers=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=False,
        ),
    }

    return loaders, class_weights


# ════════════════════════════════════════════════════════════════════════════
#  4.  Smoke test
# ════════════════════════════════════════════════════════════════════════════

def smoke_test(loaders, device):
    """Pull one batch from each split, push to GPU, verify shapes."""
    print("\n" + "=" * 60)
    print("SMOKE TEST — batch shapes on", device)
    print("=" * 60)

    for split, loader in loaders.items():
        t0 = time.time()
        imgs, labels = next(iter(loader))
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        elapsed = time.time() - t0

        print(f"  [{split:5s}]  imgs={tuple(imgs.shape)}  "
              f"labels={tuple(labels.shape)}  "
              f"dtype={imgs.dtype}  device={imgs.device}  "
              f"load={elapsed:.3f}s")

    print("=" * 60)
    print("Smoke test PASSED ✓")


# ════════════════════════════════════════════════════════════════════════════
#  5.  Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_device()
    device = torch.device(CFG.DEVICE)

    proc_dir  = CFG.PROJECT_ROOT / "data" / "splits"
    train_csv = proc_dir / "train_processed.csv"
    val_csv   = proc_dir / "val_processed.csv"
    test_csv  = proc_dir / "test_processed.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            print("  → Run python src\\03_preprocessing.py first.")
            sys.exit(1)

    loaders, class_weights = build_dataloaders(train_csv, val_csv, test_csv)

    print(f"\n[DataLoader] Class weights (for loss fn): {class_weights.tolist()}")
    print(f"  Classes: {CFG.LUNG_CLASSES}")

    smoke_test(loaders, device)

    print("\n[Summary]")
    for split, loader in loaders.items():
        n_batches = len(loader)
        n_samples = len(loader.dataset)
        print(f"  {split:5s}: {n_samples:,} images → {n_batches} batches "
              f"(batch_size={CFG.BATCH_SIZE})")

    print(f"\n[OK] 04_augmentation.py complete.")
    print("[OK] Step 04 complete. Next: python src\05_gan_augment.py  (or: python main.py)")
    print(f"     Import build_dataloaders() in 07_train.py to get DataLoaders.")


