"""
07_train.py
===========
Full HAGCA-Net training pipeline.

Strategy:
  - GroupKFold (5 folds) on the training split (9 936 images, group_id col)
  - Each fold: train on 4 sub-folds, validate on 1 sub-fold
  - Best checkpoint = fold with highest val F1-macro
  - Final held-out test evaluation is done in 08_evaluate.py

Outputs:
  checkpoints/hagcanet_fold{k}_best.pth  — per-fold best weights
  checkpoints/hagcanet_best.pth          — overall best weights copy
  results/metrics/training_log.csv       — epoch-level metrics
  logs/training.log                      — text log

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\07_train.py
"""

import os, sys, csv, time, shutil, importlib.util, threading, queue
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score

# ── bring project root onto path ─────────────────────────────────────────────
SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import CFG, setup_device, ensure_dirs, get_logger


# ── dynamic importer for files with numeric prefixes ─────────────────────────
def _load(alias: str, fname: str):
    spec   = importlib.util.spec_from_file_location(alias, SRC / fname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_aug   = _load("aug",   "04_augmentation.py")
_model = _load("model", "06_model_hagcanet.py")

LungDataset          = _aug.LungDataset
get_train_transforms = _aug.get_train_transforms
get_val_transforms   = _aug.get_val_transforms
HAGCANet             = _model.HAGCANet


# ════════════════════════════════════════════════════════════════════════════
#  Prefetch Loader  —  background thread keeps GPU fed on Windows
#
#  Problem: with num_workers=0 the pipeline is strictly sequential:
#    CPU loads batch N  →  GPU trains batch N  →  CPU loads batch N+1  →  ...
#  GPU idles while CPU is reading images. With multiprocessing workers the
#  next batch would be prepared in parallel, but Windows spawn is too slow.
#
#  Solution: a daemon Python thread (no spawn cost) reads ahead and pushes
#  tensors into a small queue.  The main thread pulls from the queue and
#  trains.  CPU work and GPU work now overlap in real time.
# ════════════════════════════════════════════════════════════════════════════

class PrefetchLoader:
    """Wraps any DataLoader and prefetches the next batch on a background thread."""

    def __init__(self, loader: DataLoader, device: torch.device, queue_size: int = 3):
        self.loader     = loader
        self.device     = device
        self.queue_size = queue_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        q         = queue.Queue(maxsize=self.queue_size)
        sentinel  = object()   # unique stop signal

        def _worker():
            try:
                for imgs, labels in self.loader:
                    # Move to GPU on the background thread using pinned memory
                    imgs   = imgs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    q.put((imgs, labels))
            finally:
                q.put(sentinel)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item


# ════════════════════════════════════════════════════════════════════════════
#  Early Stopping
# ════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None

    def __call__(self, val_f1: float) -> bool:
        if self.best is None or val_f1 > self.best + self.min_delta:
            self.best    = val_f1
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ════════════════════════════════════════════════════════════════════════════
#  One epoch — train / validate
# ════════════════════════════════════════════════════════════════════════════

def _run_epoch(model, loader, device, criterion,
               optimizer=None, scaler=None) -> dict:
    """
    If optimizer is provided → training mode.
    Otherwise              → eval mode (no gradients).
    Wraps loader with PrefetchLoader so CPU data prep and GPU compute overlap.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()

    # PrefetchLoader moves batches to GPU on a background thread.
    # By the time the main thread requests a batch, it's already on the GPU.
    prefetch = PrefetchLoader(loader, device)

    with ctx:
        for imgs, labels in prefetch:
            # imgs and labels are already on `device` — no .to() needed here
            amp_ctx = torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda" and CFG.AMP)
            )
            with amp_ctx:
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "acc":  float(accuracy_score(all_labels, all_preds)),
        "f1":   float(f1_score(all_labels, all_preds,
                               average="macro", zero_division=0)),
    }


# ════════════════════════════════════════════════════════════════════════════
#  DataLoader factory for a fold
# ════════════════════════════════════════════════════════════════════════════

def _make_loader(df: pd.DataFrame, transform, shuffle: bool,
                 drop_last: bool = False, batch_size: int = None):
    ds = LungDataset(df.reset_index(drop=True), transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size or CFG.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,       # tensors go into pinned memory → faster GPU transfer
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader   # caller wraps with PrefetchLoader per device


# ════════════════════════════════════════════════════════════════════════════
#  Train one fold
# ════════════════════════════════════════════════════════════════════════════

    # ── Two-phase training constants ─────────────────────────────────────────
    # Phase 1 (epochs 1..UNFREEZE_EPOCH): backbones FROZEN → only 3M params
    #   train → tiny memory, stable novel-module initialisation.
    # Phase 2 (epoch UNFREEZE_EPOCH+1..end): everything unfrozen → full fine-tune
    #   at half the LR.
UNFREEZE_EPOCH = 10    # switch phase after this many epochs
PHASE1_BS      = 32    # can use larger batch when backbones are frozen
PHASE2_BS      = 16    # must reduce when full model trains


def train_fold(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
               device: torch.device, logger, csv_writer) -> tuple:
    logger.info(f"\n{'='*60}")
    logger.info(f"  FOLD {fold + 1} / {CFG.KFOLD_SPLITS}  |  "
                f"train={len(train_df)}  val={len(val_df)}")
    logger.info(f"{'='*60}")
    logger.info(f"  Phase 1 (ep 1-{UNFREEZE_EPOCH}): backbones frozen, BS={PHASE1_BS}")
    logger.info(f"  Phase 2 (ep {UNFREEZE_EPOCH+1}-{CFG.NUM_EPOCHS}): full fine-tune, BS={PHASE2_BS}")

    # ── Model ────────────────────────────────────────────────────────────
    model = HAGCANet(
        num_classes=CFG.NUM_CLASSES,
        pretrained=CFG.PRETRAINED,
        dropout=CFG.DROPOUT_RATE,
    ).to(device)

    # ── Phase 1: freeze backbones ─────────────────────────────────────────
    model.freeze_backbones()
    logger.info(f"  Trainable params (phase 1): "
                f"{model.trainable_param_count()/1e6:.2f} M")

    # ── Weighted cross-entropy ────────────────────────────────────────────
    counts    = train_df["label_idx"].value_counts().sort_index().values
    weights   = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    weights  /= weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    def _make_optimizer(lr):
        return optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=CFG.WEIGHT_DECAY,
        )

    optimizer = _make_optimizer(CFG.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=CFG.LR_FACTOR,
        patience=CFG.LR_PATIENCE,
    )
    scaler     = torch.amp.GradScaler("cuda",
                   enabled=(device.type == "cuda" and CFG.AMP))
    early_stop = EarlyStopping(patience=CFG.EARLY_STOP_PATIENCE)

    # Phase-aware DataLoaders (created with correct batch size per phase)
    def _loaders(batch_size):
        tl = _make_loader(train_df, get_train_transforms(),
                          shuffle=True, drop_last=True,
                          batch_size=batch_size)
        vl = _make_loader(val_df, get_val_transforms(),
                          shuffle=False, batch_size=batch_size)
        return tl, vl

    train_loader, val_loader = _loaders(PHASE1_BS)
    current_phase = 1

    best_f1   = 0.0
    ckpt_path = CFG.CHECKPOINTS_DIR / f"hagcanet_fold{fold+1}_best.pth"

    for epoch in range(1, CFG.NUM_EPOCHS + 1):

        # ── Phase switch ─────────────────────────────────────────────────
        if epoch == UNFREEZE_EPOCH + 1 and current_phase == 1:
            model.unfreeze_backbones()
            # Rebuild optimiser so it covers newly unfrozen params
            optimizer = _make_optimizer(CFG.LEARNING_RATE / 2)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max",
                factor=CFG.LR_FACTOR,
                patience=CFG.LR_PATIENCE,
            )
            train_loader, val_loader = _loaders(PHASE2_BS)
            current_phase = 2
            logger.info(f"\n  ── Phase 2 start (epoch {epoch}): "
                        f"all {model.trainable_param_count()/1e6:.1f}M params unfrozen, "
                        f"LR={CFG.LEARNING_RATE/2:.1e}, BS={PHASE2_BS} ──\n")

        t0      = time.time()
        train_m = _run_epoch(model, train_loader, device, criterion,
                             optimizer=optimizer, scaler=scaler)
        val_m   = _run_epoch(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        scheduler.step(val_m["f1"])

        # ── Checkpoint ───────────────────────────────────────────────────
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save({
                "fold":       fold + 1,
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_f1":     best_f1,
                "val_acc":    val_m["acc"],
            }, ckpt_path)

        # ── Logging ──────────────────────────────────────────────────────
        logger.info(
            f"  F{fold+1} Ep{epoch:3d}/{CFG.NUM_EPOCHS} | "
            f"Tr loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} f1={train_m['f1']:.3f} | "
            f"Va loss={val_m['loss']:.4f}  acc={val_m['acc']:.3f}  f1={val_m['f1']:.3f} | "
            f"LR={lr_now:.1e} best={best_f1:.3f} [{elapsed:.0f}s]"
        )
        csv_writer.writerow({
            "fold": fold + 1, "epoch": epoch,
            "train_loss": round(train_m["loss"], 6),
            "train_acc":  round(train_m["acc"],  6),
            "train_f1":   round(train_m["f1"],   6),
            "val_loss":   round(val_m["loss"],   6),
            "val_acc":    round(val_m["acc"],     6),
            "val_f1":     round(val_m["f1"],      6),
            "lr":         round(lr_now, 8),
        })

        if early_stop(val_m["f1"]):
            logger.info(f"  ↳ Early stopping triggered at epoch {epoch}")
            break

    logger.info(f"  Fold {fold+1} → best val F1 = {best_f1:.4f}  saved: {ckpt_path.name}")
    return best_f1, ckpt_path


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    setup_device()
    ensure_dirs()
    device = torch.device(CFG.DEVICE)
    logger = get_logger("train")

    logger.info("=" * 60)
    logger.info("  STEP 7: HAGCA-Net Training (GroupKFold)")
    logger.info("=" * 60)
    logger.info(f"  Device={device}  AMP={CFG.AMP}  "
                f"BS={CFG.BATCH_SIZE}  Epochs={CFG.NUM_EPOCHS}  K={CFG.KFOLD_SPLITS}")
    logger.info(f"  LR={CFG.LEARNING_RATE}  WD={CFG.WEIGHT_DECAY}  "
                f"Dropout={CFG.DROPOUT_RATE}  ES_patience={CFG.EARLY_STOP_PATIENCE}")

    # ── Load training data ────────────────────────────────────────────────
    train_csv = CFG.SPLITS_DIR / "train_processed.csv"
    if not train_csv.exists():
        logger.error(f"Missing: {train_csv}  → run 03_preprocessing.py first")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    groups   = train_df["group_id"].values
    logger.info(f"  Training images  : {len(train_df)}")
    logger.info(f"  Unique group IDs : {len(np.unique(groups))}")
    logger.info(f"  Class counts     : "
                f"{dict(train_df['label'].value_counts().to_dict())}")

    # ── Open CSV log ──────────────────────────────────────────────────────
    log_csv = CFG.METRICS_DIR / "training_log.csv"
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    fields  = ["fold", "epoch",
               "train_loss", "train_acc", "train_f1",
               "val_loss",   "val_acc",   "val_f1", "lr"]
    log_fh     = open(log_csv, "w", newline="")
    csv_writer = csv.DictWriter(log_fh, fieldnames=fields)
    csv_writer.writeheader()

    # ── Build fold splits ─────────────────────────────────────────────────
    fold_results = []

    if CFG.KFOLD_SPLITS == 1:
        # Single-fold: use the pre-made train / val CSVs directly
        val_csv = CFG.SPLITS_DIR / "val_processed.csv"
        val_df  = pd.read_csv(val_csv)
        logger.info(f"  Single-fold mode: using pre-split val ({len(val_df)} images)")
        splits = [(train_df, val_df)]
    else:
        # GroupKFold cross-validation
        gkf    = GroupKFold(n_splits=CFG.KFOLD_SPLITS)
        splits = [
            (train_df.iloc[tr], train_df.iloc[va])
            for tr, va in gkf.split(train_df, train_df["label_idx"], groups)
        ]

    for fold, (fold_train_df, fold_val_df) in enumerate(splits):
        best_f1, ckpt = train_fold(
            fold=fold,
            train_df=fold_train_df,
            val_df=fold_val_df,
            device=device,
            logger=logger,
            csv_writer=csv_writer,
        )
        fold_results.append((best_f1, ckpt))

    log_fh.close()

    # ── Copy overall best checkpoint ──────────────────────────────────────
    best_f1_overall, best_ckpt = max(fold_results, key=lambda x: x[0])
    overall_best = CFG.BEST_MODEL_PATH
    shutil.copy(best_ckpt, overall_best)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    for i, (f1, _) in enumerate(fold_results):
        mark = " ← BEST" if f1 == best_f1_overall else ""
        logger.info(f"  Fold {i+1}: val F1 = {f1:.4f}{mark}")
    mean_f1 = sum(f for f, _ in fold_results) / len(fold_results)
    logger.info(f"  Mean val F1 across folds: {mean_f1:.4f}")
    logger.info(f"\n  Best checkpoint : {overall_best}")
    logger.info(f"  Training CSV    : {log_csv}")
    logger.info(f"\n  Next: python src\08_evaluate.py  (or: python main.py)")


if __name__ == "__main__":
    main()

