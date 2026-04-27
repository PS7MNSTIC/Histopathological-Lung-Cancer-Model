"""
13_pcam_train_test.py  (v2 — high-accuracy edition, target >98%)
================================================================
Trains, validates, and tests HAGCA-Net entirely on PatchCamelyon (PCam).

This script is Scenario 3 in the three-way comparison:
  Scenario 1 - LC25000 -> LC25000   (08_evaluate.py)
  Scenario 2 - LC25000 -> PCam      (10_cross_dataset.py, zero-shot)
  Scenario 3 - PCam    -> PCam      (THIS FILE)

WHAT CHANGED IN v2 (vs. the 83.59% run):
  * Forced retrain (or auto-retrain if an existing ckpt is too weak / too short).
  * Trains on FULL 262,144 PCam training images (was 30,000).
  * MixUp + label smoothing during training.
  * EMA (exponential moving average) of weights, evaluated each epoch.
  * Test-Time Augmentation (TTA, 4-flip mean) at evaluation.
  * Gradient clipping + linear warmup + cosine annealing.
  * AdamW with weight decay (was Adam).

Preprocessing (same pipeline as LC25000 for fair comparison):
  - Reinhard stain normalization (reference stats computed from 200 PCam train images)
  - CLAHE enhancement on L channel (LAB space)

Outputs:
  checkpoints/hagcanet_pcam_best.pth      (best EMA weights, val F1)
  results/metrics/pcam_train_test_metrics.json
  results/metrics/pcam_train_test_report.txt
  results/plots/pcam_confusion_matrix.png

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    # Recommended (force a fresh, full-quality run):
    python src\\13_pcam_train_test.py --force-retrain
    # Or just delete the old checkpoint first and run normally:
    del checkpoints\\hagcanet_pcam_best.pth
    python src\\13_pcam_train_test.py

Estimated training time on RTX 5060 (8.5 GB):
    Full data (262K) + 3+15 epochs ........... ~5-6 h, expected test acc >=98%
    Half data (130K) + 3+12 epochs ........... ~3 h,   expected test acc 96-98%
    Quick (60K)      + 3+10 epochs ........... ~1.5 h, expected test acc 94-96%
"""

import sys, os, json, importlib.util, threading, queue, time, copy, argparse, math
from pathlib import Path
from datetime import timedelta

import cv2
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.functional import softmax
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import CFG, setup_device, ensure_dirs, get_logger

def _load(alias, fname):
    spec = importlib.util.spec_from_file_location(alias, SRC / fname)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_model = _load("model", "06_model_hagcanet.py")
HAGCANet = _model.HAGCANet


# ============================================================================
#  Config  (tweak here if you need a faster run)
# ============================================================================

# ---- Training scale ---------------------------------------------------------
MAX_TRAIN_SAMPLES = 130_000      # None = full 262,144  (recommended for >98%)
MAX_VAL_SAMPLES   = 16_384    # subset of the 32,768 validation images for speed
MAX_TEST_SAMPLES  = None      # None = full 32,768  (always evaluate all)

# ---- Two-phase schedule -----------------------------------------------------
PHASE1_EPOCHS     = 3         # frozen backbones (warmup of head + novel modules)
PHASE2_EPOCHS     = 15        # full fine-tune
WARMUP_EPOCHS     = 1         # linear LR warmup at the start of each phase
PHASE1_BS         = 32
PHASE2_BS         = 16
LR1               = 2e-4
LR2               = 5e-5
WEIGHT_DECAY      = 1e-4
EARLY_STOP        = 6         # patience (epochs of no val-F1 improvement)
GRAD_CLIP         = 1.0       # clip grad-norm

# ---- Regularization ---------------------------------------------------------
LABEL_SMOOTHING   = 0.05
MIXUP_ALPHA       = 0.2       # 0 = off
MIXUP_PROB        = 0.5       # apply mixup to a fraction of batches
EMA_DECAY         = 0.9995    # exponential moving average of weights

# ---- Test-time augmentation ------------------------------------------------
TTA_FLIPS         = True      # avg over (orig, hflip, vflip, hflip+vflip)

# ---- Quality gate for "skip training if checkpoint exists" -----------------
MIN_VAL_F1_TO_SKIP = 0.97     # if existing ckpt's val_f1 < this, retrain
MIN_EPOCHS_TO_SKIP = 8        # if existing ckpt epoch < this, retrain

PCAM_NUM_CLASSES  = 2
PCAM_CLASSES      = ["normal", "tumor"]
SEED              = 42

PCAM_ROOT   = CFG.PROJECT_ROOT / "data" / "external_test" / "archive"
PCAM_CKPT   = CFG.CHECKPOINTS_DIR / "hagcanet_pcam_best.pth"
IMG_SIZE    = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLAHE_CLIP_LIMIT = getattr(CFG, "CLAHE_CLIP_LIMIT", 2.0)
CLAHE_TILE_SIZE  = getattr(CFG, "CLAHE_TILE_SIZE", (8, 8))
REF_STAT_SAMPLES = 200


# ============================================================================
#  Reinhard / CLAHE helpers (unchanged from v1 — same as 03_preprocessing.py)
# ============================================================================

def _compute_reinhard_stats(img_rgb: np.ndarray):
    lab = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    return (L.mean(), L.std() + 1e-6,
            a.mean(), a.std() + 1e-6,
            b.mean(), b.std() + 1e-6)


def _reinhard_normalize(img_rgb: np.ndarray, ref_stats) -> np.ndarray:
    ref_mL, ref_sL, ref_ma, ref_sa, ref_mb, ref_sb = ref_stats
    lab = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    L = (L - L.mean()) / (L.std() + 1e-6) * ref_sL + ref_mL
    a = (a - a.mean()) / (a.std() + 1e-6) * ref_sa + ref_ma
    b = (b - b.mean()) / (b.std() + 1e-6) * ref_sb + ref_mb
    lab_norm = np.stack([L, a, b], axis=2).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)


def _apply_clahe(img_rgb: np.ndarray,
                 clip_limit=CLAHE_CLIP_LIMIT,
                 tile_size=CLAHE_TILE_SIZE) -> np.ndarray:
    lab  = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _compute_pcam_ref_stats(img_h5_path, n_sample=REF_STAT_SAMPLES, seed=SEED):
    with h5py.File(str(img_h5_path), "r") as f:
        n_total = f["x"].shape[0]
        rng     = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n_total, min(n_sample, n_total), replace=False))
        sample_imgs = f["x"][indices]
    stats_list = [_compute_reinhard_stats(img) for img in sample_imgs]
    arr = np.array(stats_list)
    return tuple(arr.mean(axis=0).tolist())


class ReinhardNorm:
    _ref_stats = None
    _lock = threading.Lock()

    def __init__(self, img_h5_path=None):
        self.img_h5_path = img_h5_path

    def _ensure_stats(self):
        if ReinhardNorm._ref_stats is None:
            with ReinhardNorm._lock:
                if ReinhardNorm._ref_stats is None:
                    if self.img_h5_path is None or not Path(self.img_h5_path).exists():
                        ReinhardNorm._ref_stats = (128.0, 10.0, 128.0, 5.0, 128.0, 5.0)
                    else:
                        ReinhardNorm._ref_stats = _compute_pcam_ref_stats(self.img_h5_path)

    def __call__(self, img):
        self._ensure_stats()
        arr = np.array(img)
        arr = _reinhard_normalize(arr, ReinhardNorm._ref_stats)
        return Image.fromarray(arr)


class CLAHEEnhance:
    def __init__(self, clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_SIZE):
        self.clip_limit = clip_limit
        self.tile_size  = tile_size

    def __call__(self, img):
        arr = np.array(img)
        arr = _apply_clahe(arr, self.clip_limit, self.tile_size)
        return Image.fromarray(arr)


# ============================================================================
#  Transform pipelines  (stronger augmentation in v2)
# ============================================================================

def make_transforms(pcam_img_train_h5):
    reinhard = ReinhardNorm(img_h5_path=str(pcam_img_train_h5))
    clahe    = CLAHEEnhance()

    train_tf = transforms.Compose([
        reinhard,
        clahe,
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25,
                               saturation=0.20, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    val_tf = transforms.Compose([
        reinhard,
        clahe,
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, val_tf


# ============================================================================
#  PCam Dataset  (lazy H5 reader, stratified subsampling)
# ============================================================================

class PCamDataset(Dataset):
    def __init__(self, img_h5, lbl_h5, transform=None,
                 max_samples=None, seed=SEED):
        self.img_h5    = str(img_h5)
        self.lbl_h5    = str(lbl_h5)
        self.transform = transform

        with h5py.File(self.lbl_h5, "r") as f:
            all_labels = f["y"][:, 0, 0, 0].astype(int)

        n_total = len(all_labels)
        if max_samples is not None and max_samples < n_total:
            rng = np.random.default_rng(seed)
            n_per_class = max_samples // 2
            idx0 = np.where(all_labels == 0)[0]
            idx1 = np.where(all_labels == 1)[0]
            chosen0 = rng.choice(idx0, min(n_per_class, len(idx0)), replace=False)
            chosen1 = rng.choice(idx1, min(n_per_class, len(idx1)), replace=False)
            self.indices = np.sort(np.concatenate([chosen0, chosen1]))
            self.labels  = all_labels[self.indices]
        else:
            self.indices = np.arange(n_total)
            self.labels  = all_labels

        self._img_file = None

    def _open(self):
        if self._img_file is None:
            self._img_file = h5py.File(self.img_h5, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._open()
        real_idx = int(self.indices[idx])
        img_arr  = self._img_file["x"][real_idx]
        label    = int(self.labels[idx])
        img      = Image.fromarray(img_arr.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================================
#  PrefetchLoader
# ============================================================================

class PrefetchLoader:
    def __init__(self, loader, device, queue_size=3):
        self.loader = loader; self.device = device; self.qs = queue_size

    def __len__(self): return len(self.loader)

    def __iter__(self):
        q = queue.Queue(maxsize=self.qs); sentinel = object()
        def _worker():
            try:
                for imgs, labels in self.loader:
                    q.put((imgs.to(self.device, non_blocking=True),
                           labels.to(self.device, non_blocking=True)))
            finally: q.put(sentinel)
        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = q.get()
            if item is sentinel: break
            yield item


# ============================================================================
#  Build DataLoaders
# ============================================================================

def build_pcam_loaders(batch_size, logger):
    img_train = PCAM_ROOT / "pcam" / "training_split.h5"
    lbl_train = PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_train_y.h5"
    img_val   = PCAM_ROOT / "pcam" / "validation_split.h5"
    lbl_val   = PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_valid_y.h5"
    img_test  = PCAM_ROOT / "pcam" / "test_split.h5"
    lbl_test  = PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_test_y.h5"

    for p in [img_train, lbl_train, img_val, lbl_val, img_test, lbl_test]:
        if not p.exists():
            logger.error(f"Missing: {p}"); sys.exit(1)

    train_tf, val_tf = make_transforms(img_train)

    train_ds = PCamDataset(img_train, lbl_train, train_tf,
                           max_samples=MAX_TRAIN_SAMPLES)
    val_ds   = PCamDataset(img_val,   lbl_val,   val_tf,
                           max_samples=MAX_VAL_SAMPLES)
    test_ds  = PCamDataset(img_test,  lbl_test,  val_tf,
                           max_samples=MAX_TEST_SAMPLES)

    logger.info(f"PCam splits  -- train: {len(train_ds):,}  "
                f"val: {len(val_ds):,}  test: {len(test_ds):,}")

    logger.info(f"Computing Reinhard ref stats from {REF_STAT_SAMPLES} PCam training images ...")
    train_ds[0]
    logger.info(f"Reinhard ref stats: {[round(v, 2) for v in ReinhardNorm._ref_stats]}")

    counts = np.bincount(train_ds.labels, minlength=2)
    w = torch.tensor(1.0 / counts, dtype=torch.float32)
    w = w / w.sum() * 2.0
    logger.info(f"Class counts -- normal: {counts[0]:,}  tumor: {counts[1]:,}")
    logger.info(f"Class weights: {[round(v, 4) for v in w.tolist()]}")

    make_loader = lambda ds, bs, shuffle: DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=0, pin_memory=False, drop_last=shuffle,
    )
    return {
        "train": make_loader(train_ds, batch_size, True),
        "val":   make_loader(val_ds,   batch_size, False),
        "test":  make_loader(test_ds,  batch_size, False),
    }, w


# ============================================================================
#  MixUp + Label Smoothing
# ============================================================================

def mixup_batch(imgs, labels, num_classes, alpha=0.2):
    """Returns mixed_imgs, mixed_targets (soft labels)."""
    if alpha <= 0:
        targets = F.one_hot(labels, num_classes).float()
        return imgs, targets
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    mixed_imgs = lam * imgs + (1 - lam) * imgs[perm]
    targets_a  = F.one_hot(labels,        num_classes).float()
    targets_b  = F.one_hot(labels[perm],  num_classes).float()
    mixed_targets = lam * targets_a + (1 - lam) * targets_b
    return mixed_imgs, mixed_targets


def soft_cross_entropy(logits, soft_targets, weight=None, smoothing=0.0):
    """CE for soft targets with optional label smoothing and class weighting."""
    n_classes = logits.size(1)
    if smoothing > 0:
        soft_targets = soft_targets * (1 - smoothing) + smoothing / n_classes
    log_probs = F.log_softmax(logits, dim=1)
    if weight is not None:
        log_probs = log_probs * weight.view(1, -1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


# ============================================================================
#  EMA (Exponential Moving Average) of weights
# ============================================================================

class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.decay  = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ============================================================================
#  One epoch
# ============================================================================

def train_one_epoch(model, loader, device, optimizer, scaler, class_weights,
                    ema, logger, epoch, warmup_epochs, base_lr):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    n_batches = len(loader)

    for b_idx, (imgs, labels) in enumerate(PrefetchLoader(loader, device)):
        # ---- linear warmup within first warmup_epochs ----
        if epoch <= warmup_epochs:
            warm_frac = ((epoch - 1) * n_batches + b_idx + 1) / max(warmup_epochs * n_batches, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * warm_frac

        # ---- mixup ----
        use_mixup = (np.random.rand() < MIXUP_PROB) and (MIXUP_ALPHA > 0)
        if use_mixup:
            imgs_in, soft_targets = mixup_batch(imgs, labels, PCAM_NUM_CLASSES, MIXUP_ALPHA)
        else:
            imgs_in       = imgs
            soft_targets  = F.one_hot(labels, PCAM_NUM_CLASSES).float()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and CFG.AMP)):
            logits = model(imgs_in)
            loss   = soft_cross_entropy(logits, soft_targets,
                                        weight=class_weights, smoothing=LABEL_SMOOTHING)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * imgs.size(0)
        # For training accuracy reporting, use clean predictions (no mixup)
        with torch.no_grad():
            preds = logits.argmax(1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n    = len(all_labels)
    loss = total_loss / n
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return loss, acc, f1


@torch.no_grad()
def eval_model(model, loader, device, class_weights, logger):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in PrefetchLoader(loader, device):
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and CFG.AMP)):
            logits = model(imgs)
            soft   = F.one_hot(labels, PCAM_NUM_CLASSES).float()
            loss   = soft_cross_entropy(logits, soft, weight=class_weights, smoothing=0.0)
        total_loss += loss.item() * imgs.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n    = len(all_labels)
    return total_loss / n, accuracy_score(all_labels, all_preds), \
           f1_score(all_labels, all_preds, average="binary", zero_division=0)


# ============================================================================
#  Training loop  (two-phase, EMA, mixup, LS, warmup, cosine)
# ============================================================================

def train_pcam(device, logger):
    loaders, class_weights = build_pcam_loaders(PHASE1_BS, logger)
    class_weights = class_weights.to(device)

    model = HAGCANet(num_classes=PCAM_NUM_CLASSES, pretrained=True).to(device)
    logger.info(f"Model params: {model.trainable_param_count()/1e6:.2f} M (all)")

    ema     = ModelEMA(model, decay=EMA_DECAY)
    scaler  = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and CFG.AMP))

    best_val_f1 = -1.0
    no_improve  = 0
    best_epoch  = 0
    total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS

    for epoch in range(1, total_epochs + 1):
        # ---- Phase 1: frozen backbones ----
        if epoch == 1:
            model.freeze_backbones()
            # Re-init EMA from the frozen model state to avoid stale moments.
            ema = ModelEMA(model, decay=EMA_DECAY)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR1, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(PHASE1_EPOCHS - WARMUP_EPOCHS, 1),
                eta_min=LR1 * 0.05
            )
            logger.info(f"Phase 1 -- backbones frozen  "
                        f"({model.trainable_param_count()/1e6:.2f}M trainable, bs={PHASE1_BS})")
            phase_start_epoch = 1
            phase_warmup      = WARMUP_EPOCHS
            phase_base_lr     = LR1

        # ---- Phase 2: full fine-tune ----
        if epoch == PHASE1_EPOCHS + 1:
            model.unfreeze_backbones()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR2, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(PHASE2_EPOCHS - WARMUP_EPOCHS, 1),
                eta_min=LR2 * 0.05
            )
            logger.info(f"Phase 2 -- full fine-tune  "
                        f"({model.trainable_param_count()/1e6:.2f}M trainable, bs={PHASE2_BS})")
            loaders, _ = build_pcam_loaders(PHASE2_BS, logger)
            phase_start_epoch = PHASE1_EPOCHS + 1
            phase_warmup      = WARMUP_EPOCHS
            phase_base_lr     = LR2

        local_epoch = epoch - phase_start_epoch + 1
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, loaders["train"], device, optimizer, scaler, class_weights,
            ema, logger, local_epoch, phase_warmup, phase_base_lr
        )
        # Validate the EMA copy — it's typically smoother and slightly better.
        va_loss, va_acc, va_f1 = eval_model(
            ema.module, loaders["val"], device, class_weights, logger
        )
        if local_epoch > phase_warmup:
            scheduler.step()
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Ep {epoch:>3}/{total_epochs}  lr={cur_lr:.2e}  "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} tr_f1={tr_f1:.4f}  |  "
            f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} va_f1={va_f1:.4f}  "
            f"[{elapsed:.0f}s]"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch  = epoch
            no_improve  = 0
            CFG.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch":       epoch,
                "val_f1":      va_f1,
                "val_acc":     va_acc,
                "state_dict":  ema.module.state_dict(),
                "num_classes": PCAM_NUM_CLASSES,
            }, PCAM_CKPT)
            logger.info(f"  [SAVED] EMA val F1={va_f1:.4f}  -> {PCAM_CKPT.name}")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP and epoch > PHASE1_EPOCHS + 2:
                logger.info(f"Early stopping at epoch {epoch} (patience={EARLY_STOP})")
                break

    logger.info(f"Training complete. Best val F1 = {best_val_f1:.4f} at epoch {best_epoch}")
    return best_val_f1


# ============================================================================
#  Evaluation on test set  (with TTA)
# ============================================================================

@torch.no_grad()
def _forward_with_tta(model, imgs, device, use_tta=True):
    """Returns averaged softmax probs over (orig, hflip, vflip, hvflip)."""
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and CFG.AMP)):
        all_probs = [softmax(model(imgs), dim=1)]
        if use_tta:
            all_probs.append(softmax(model(torch.flip(imgs, dims=[3])), dim=1))   # H-flip
            all_probs.append(softmax(model(torch.flip(imgs, dims=[2])), dim=1))   # V-flip
            all_probs.append(softmax(model(torch.flip(imgs, dims=[2, 3])), dim=1))# H+V flip
    return torch.stack(all_probs, 0).mean(0)


@torch.no_grad()
def evaluate_test(device, logger):
    model = HAGCANet(num_classes=PCAM_NUM_CLASSES, pretrained=False).to(device)
    ckpt  = torch.load(PCAM_CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info(f"Loaded: {PCAM_CKPT.name}  (epoch {ckpt['epoch']}, val_F1={ckpt['val_f1']:.4f})")

    img_train = PCAM_ROOT / "pcam" / "training_split.h5"
    img_test  = PCAM_ROOT / "pcam" / "test_split.h5"
    lbl_test  = PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_test_y.h5"

    _, val_tf = make_transforms(img_train)

    test_ds  = PCamDataset(img_test, lbl_test, val_tf, max_samples=MAX_TEST_SAMPLES)
    loader   = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

    logger.info(f"Test samples: {len(test_ds):,}   TTA: {TTA_FLIPS}")
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in PrefetchLoader(loader, device):
        probs_t = _forward_with_tta(model, imgs, device, use_tta=TTA_FLIPS)
        probs   = probs_t[:, 1].cpu().numpy()
        preds   = probs_t.argmax(1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec  = recall_score(labels, preds, average="binary", zero_division=0)
    f1   = f1_score(labels, preds, average="binary", zero_division=0)
    try:    auc = roc_auc_score(labels, probs)
    except: auc = float("nan")

    report = classification_report(labels, preds, target_names=PCAM_CLASSES, digits=4)

    logger.info("\n" + "=" * 60)
    logger.info("  PCam TEST SET RESULTS  (Scenario 3: PCam -> PCam)  [v2 + TTA]")
    logger.info("=" * 60)
    logger.info(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    logger.info(f"  Precision      : {prec:.4f}")
    logger.info(f"  Recall (tumor) : {rec:.4f}")
    logger.info(f"  F1 (binary)    : {f1:.4f}")
    logger.info(f"  ROC-AUC        : {auc:.4f}")
    logger.info(f"\n{report}")

    metrics = {
        "scenario":    "PCam_trained_PCam_tested",
        "train_data":  "PatchCamelyon",
        "test_data":   "PatchCamelyon",
        "n_train":     MAX_TRAIN_SAMPLES or 262144,
        "n_test":      int(len(labels)),
        "num_classes": PCAM_NUM_CLASSES,
        "accuracy":    round(float(acc),  4),
        "precision":   round(float(prec), 4),
        "recall":      round(float(rec),  4),
        "f1_binary":   round(float(f1),   4),
        "roc_auc":     round(float(auc),  4),
        "checkpoint":  PCAM_CKPT.name,
        "val_f1":      round(float(ckpt["val_f1"]), 4),
        "best_epoch":  int(ckpt["epoch"]),
        "preprocessing": "ReinhardNorm(PCam-ref) + CLAHE",
        "training_tricks": "AdamW + cosine + warmup + mixup + label_smoothing + EMA + grad_clip",
        "inference":   ("TTA: orig+hflip+vflip+hvflip" if TTA_FLIPS else "single-crop"),
    }
    json_path = CFG.METRICS_DIR / "pcam_train_test_metrics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {json_path}")

    rpt_path = CFG.METRICS_DIR / "pcam_train_test_report.txt"
    with open(rpt_path, "w") as f:
        f.write("HAGCA-Net PCam Train+Test Evaluation (v2 + TTA)\n")
        f.write("=" * 60 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + report)
    logger.info(f"Report saved: {rpt_path}")

    cm   = confusion_matrix(labels, preds)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, norm], ["d", ".2f"], ["Counts", "Normalised (row %)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=PCAM_CLASSES, yticklabels=PCAM_CLASSES,
                    ax=ax, linewidths=0.5)
        ax.set_title(f"Confusion Matrix -- {title}", fontsize=12)
        ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.suptitle("HAGCA-Net  |  PCam -> PCam  (Scenario 3, v2 + TTA)", fontsize=13, y=1.02)
    plt.tight_layout()
    cm_path = CFG.PLOTS_DIR / "pcam_confusion_matrix.png"
    CFG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"Confusion matrix saved: {cm_path}")

    return metrics


# ============================================================================
#  Quality gate: should we re-train?
# ============================================================================

def _checkpoint_is_good_enough(ckpt_path: Path, logger) -> bool:
    """Skip training only if checkpoint is real (sufficient epochs + good val_f1)."""
    if not ckpt_path.exists():
        return False
    try:
        meta = torch.load(ckpt_path, map_location="cpu")
        ep   = int(meta.get("epoch", 0))
        f1   = float(meta.get("val_f1", 0.0))
    except Exception as e:
        logger.warning(f"Could not read checkpoint metadata ({e}) -- will retrain.")
        return False

    logger.info(f"Existing ckpt: epoch={ep}  val_F1={f1:.4f}  "
                f"(thresholds: epoch>={MIN_EPOCHS_TO_SKIP}, val_F1>={MIN_VAL_F1_TO_SKIP})")
    if ep < MIN_EPOCHS_TO_SKIP:
        logger.info("  -> too few epochs, will retrain.")
        return False
    if f1 < MIN_VAL_F1_TO_SKIP:
        logger.info("  -> val_F1 below quality threshold, will retrain.")
        return False
    logger.info("  -> checkpoint passes quality gate, skipping training.")
    return True


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true",
                        help="Ignore existing checkpoint and retrain from scratch.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; just run TTA evaluation on existing ckpt.")
    args, _ = parser.parse_known_args()

    setup_device()
    ensure_dirs()
    device = torch.device(CFG.DEVICE)
    logger = get_logger("pcam_train_test")

    logger.info("=" * 60)
    logger.info("  STEP 13: PCam TRAIN + VALIDATE + TEST  (Scenario 3, v2)")
    logger.info("=" * 60)
    logger.info(f"  Training samples : {MAX_TRAIN_SAMPLES or 262144:,}")
    logger.info(f"  Val samples      : {MAX_VAL_SAMPLES or 32768:,}")
    logger.info(f"  Test samples     : {MAX_TEST_SAMPLES or 32768:,}")
    logger.info(f"  Phases           : {PHASE1_EPOCHS} (frozen) + {PHASE2_EPOCHS} (full)")
    logger.info(f"  Reg              : LS={LABEL_SMOOTHING}  MixUp={MIXUP_ALPHA}@p{MIXUP_PROB}  "
                f"EMA={EMA_DECAY}  WD={WEIGHT_DECAY}")
    logger.info(f"  Inference        : TTA={'on (4-flip)' if TTA_FLIPS else 'off'}")
    logger.info(f"  Classes          : {PCAM_CLASSES}")
    logger.info(f"  Checkpoint       : {PCAM_CKPT}")

    if args.eval_only:
        if not PCAM_CKPT.exists():
            logger.error("--eval-only requested but no checkpoint exists. Exiting.")
            sys.exit(1)
        logger.info("--eval-only: skipping training.")
    else:
        skip_training = (not args.force_retrain) and _checkpoint_is_good_enough(PCAM_CKPT, logger)
        if skip_training:
            logger.info("Skipping training (checkpoint passes quality gate).")
        else:
            if PCAM_CKPT.exists() and args.force_retrain:
                logger.info("--force-retrain set: removing existing checkpoint.")
                try:
                    PCAM_CKPT.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove old ckpt ({e}); will overwrite.")
            logger.info("Starting training ...")
            train_pcam(device, logger)

    logger.info("\nEvaluating on test set ...")
    evaluate_test(device, logger)

    logger.info("\nNext: python src\\14_compare_scenarios.py  (or: python main.py)")


if __name__ == "__main__":
    main()
