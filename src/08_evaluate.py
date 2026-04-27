"""
08_evaluate.py
==============
Evaluates the best HAGCA-Net checkpoint on the held-out test set.

Metrics reported:
  - Accuracy, Precision, Recall, F1-macro (per-class + overall)
  - ROC-AUC (one-vs-rest, macro)
  - Confusion matrix (saved as PNG)
  - Per-class classification report

Outputs:
  results/metrics/test_report.txt      -- full classification report
  results/metrics/test_metrics.json    -- JSON with all scalar metrics
  results/plots/confusion_matrix.png   -- confusion matrix heatmap

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\08_evaluate.py
"""

import sys, json, importlib.util
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import CFG, setup_device, ensure_dirs, get_logger

def _load(alias, fname):
    spec = importlib.util.spec_from_file_location(alias, SRC / fname)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_aug   = _load("aug",   "04_augmentation.py")
_model = _load("model", "06_model_hagcanet.py")

LungDataset        = _aug.LungDataset
get_val_transforms = _aug.get_val_transforms
HAGCANet           = _model.HAGCANet


# ════════════════════════════════════════════════════════════════════════════
#  Inference
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, loader, device):
    """Returns arrays: all_preds, all_labels, all_probs (N, C)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and CFG.AMP)):
            logits = model(imgs)

        probs  = softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


# ════════════════════════════════════════════════════════════════════════════
#  Confusion matrix plot
# ════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, title in zip(
        axes,
        [cm, norm],
        ["d", ".2f"],
        ["Counts", "Normalised (row %)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix — {title}", fontsize=13, pad=10)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

    plt.suptitle("HAGCA-Net  |  Test Set Evaluation", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved confusion matrix: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    setup_device()
    ensure_dirs()
    device = torch.device(CFG.DEVICE)
    logger = get_logger("evaluate")

    logger.info("=" * 60)
    logger.info("  STEP 8: EVALUATION on held-out test set")
    logger.info("=" * 60)

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = CFG.BEST_MODEL_PATH
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Run 07_train.py first.")
        sys.exit(1)

    model = HAGCANet(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    logger.info(f"Loaded checkpoint: {ckpt_path.name}")
    logger.info(f"  Saved at epoch {ckpt.get('epoch','?')}  "
                f"val_F1={ckpt.get('val_f1', '?'):.4f}")

    # ── Test DataLoader ───────────────────────────────────────────────────
    test_csv = CFG.SPLITS_DIR / "test_processed.csv"
    if not test_csv.exists():
        logger.error(f"Missing: {test_csv}  -> run 03_preprocessing.py")
        sys.exit(1)

    test_df = pd.read_csv(test_csv)
    logger.info(f"Test images: {len(test_df)}")

    test_ds     = LungDataset(test_df, transform=get_val_transforms())
    test_loader = DataLoader(
        test_ds, batch_size=CFG.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    # ── Inference ─────────────────────────────────────────────────────────
    logger.info("Running inference on test set ...")
    preds, labels, probs = run_inference(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────
    classes = CFG.LUNG_CLASSES

    acc     = accuracy_score(labels, preds)
    prec    = precision_score(labels, preds, average="macro", zero_division=0)
    rec     = recall_score(labels, preds, average="macro", zero_division=0)
    f1      = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per  = f1_score(labels, preds, average=None, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    report = classification_report(labels, preds, target_names=classes, digits=4)

    # ── Print results ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    logger.info(f"  Precision : {prec:.4f}")
    logger.info(f"  Recall    : {rec:.4f}")
    logger.info(f"  F1-macro  : {f1:.4f}")
    logger.info(f"  ROC-AUC   : {auc:.4f}")
    for i, cls in enumerate(classes):
        logger.info(f"    {cls:<12} F1={f1_per[i]:.4f}")
    logger.info("\n" + report)

    # ── Save metrics JSON ─────────────────────────────────────────────────
    metrics = {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1_macro":  round(float(f1),   4),
        "roc_auc":   round(float(auc),  4),
        "f1_per_class": {
            cls: round(float(f1_per[i]), 4)
            for i, cls in enumerate(classes)
        },
        "n_test":    int(len(labels)),
        "checkpoint": ckpt_path.name,
    }
    json_path = CFG.METRICS_DIR / "test_metrics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {json_path}")

    # ── Save classification report ─────────────────────────────────────────
    rpt_path = CFG.METRICS_DIR / "test_report.txt"
    with open(rpt_path, "w") as f:
        f.write("HAGCA-Net Test Set Evaluation\n")
        f.write("=" * 60 + "\n")
        for k, v in metrics.items():
            if k != "f1_per_class":
                f.write(f"{k}: {v}\n")
        f.write("\n" + report)
    logger.info(f"Report saved: {rpt_path}")

    # ── Confusion matrix plot ──────────────────────────────────────────────
    cm_path = CFG.PLOTS_DIR / "confusion_matrix.png"
    CFG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(labels, preds, classes, cm_path)

    logger.info("\nNext: python src\09_gradcam.py  (or: python main.py)")


if __name__ == "__main__":
    main()

