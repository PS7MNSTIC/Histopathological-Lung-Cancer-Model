"""
10_cross_dataset.py
===================
Cross-dataset generalization test: HAGCA-Net (trained on LC25000 lung tissue)
evaluated on PatchCamelyon (PCam) — lymph node metastasis detection.

Dataset layout expected:
  C:\\ml_project\\data\\external_test\\archive\\pcam\\test_split.h5
        key "x"  -> (32768, 96, 96, 3) uint8  images
  C:\\ml_project\\data\\external_test\\archive\\Labels\\Labels\\
        camelyonpatch_level_2_split_test_y.h5
        key "y"  -> (32768, 1, 1, 1) uint8  binary labels  (0=normal, 1=tumor)

Class mapping (3-class LC25000 -> binary PCam):
  lung_aca (idx 0) -> cancer  -> 1
  lung_n   (idx 1) -> normal  -> 0
  lung_scc (idx 2) -> cancer  -> 1

Why this test matters:
  - PCam is lymph node tissue (breast cancer metastasis); LC25000 is lung.
  - If HAGCA-Net's learned cancer/normal features transfer across organs,
    it demonstrates cross-dataset generalisation for the paper.

Outputs:
  results/metrics/cross_dataset_metrics.json
  results/metrics/cross_dataset_report.txt
  results/plots/cross_dataset_confusion.png

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\10_cross_dataset.py
"""

import sys, json, importlib.util, threading, queue
from pathlib import Path

import h5py
import torch
import numpy as np
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


# ════════════════════════════════════════════════════════════════════════════
#  PCam Dataset  (lazy H5 reads — never loads full 32k images into RAM)
# ════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# PCam images are 96×96 — upsample to 224×224 for the model
PCAM_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class PCamDataset(Dataset):
    """
    Reads PatchCamelyon images and labels lazily from H5 files.

    Parameters
    ----------
    img_h5   : path to test_split.h5  (key "x")
    lbl_h5   : path to camelyonpatch_level_2_split_test_y.h5  (key "y")
    transform: torchvision transform
    max_samples: int or None — cap for quick testing
    """

    def __init__(self, img_h5, lbl_h5, transform=None, max_samples=None):
        self.img_h5   = str(img_h5)
        self.lbl_h5   = str(lbl_h5)
        self.transform = transform

        # Open once to get length; keep file closed between __getitem__ calls
        with h5py.File(self.img_h5, "r") as f:
            n_total = f["x"].shape[0]

        self.n = n_total if max_samples is None else min(max_samples, n_total)

        # h5py file handles opened per-process (needed if num_workers > 0)
        self._img_file = None
        self._lbl_file = None

    def _open(self):
        if self._img_file is None:
            self._img_file = h5py.File(self.img_h5, "r")
            self._lbl_file = h5py.File(self.lbl_h5, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._open()
        img_arr = self._img_file["x"][idx]           # (96, 96, 3) uint8
        label   = int(self._img_file["x"][idx].shape[0])  # placeholder; read label below
        label   = int(self._lbl_file["y"][idx, 0, 0, 0])  # scalar 0 or 1

        img = Image.fromarray(img_arr.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label


# ════════════════════════════════════════════════════════════════════════════
#  PrefetchLoader  (same pattern as 07_train.py — solves num_workers=0 stall)
# ════════════════════════════════════════════════════════════════════════════

class PrefetchLoader:
    def __init__(self, loader, device, queue_size=3):
        self.loader     = loader
        self.device     = device
        self.queue_size = queue_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        q        = queue.Queue(maxsize=self.queue_size)
        sentinel = object()

        def _worker():
            try:
                for imgs, labels in self.loader:
                    imgs   = imgs.to(self.device,   non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    q.put((imgs, labels))
            finally:
                q.put(sentinel)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item


# ════════════════════════════════════════════════════════════════════════════
#  3-class -> binary mapping
# ════════════════════════════════════════════════════════════════════════════
# LUNG_CLASSES = ['lung_aca', 'lung_n', 'lung_scc']
#   idx 0 (lung_aca) -> cancer -> binary 1
#   idx 1 (lung_n)   -> normal -> binary 0
#   idx 2 (lung_scc) -> cancer -> binary 1

CANCER_CLASS_INDICES = {0, 2}   # lung_aca, lung_scc
NORMAL_CLASS_INDEX   = 1        # lung_n


def logits_to_binary(logits):
    """
    logits: (B, 3) tensor
    Returns:
      bin_preds  : (B,) int64  binary predictions  (0 or 1)
      cancer_prob: (B,) float  probability of cancer (max of lung_aca + lung_scc probs)
    """
    probs      = softmax(logits, dim=1)            # (B, 3)
    # Cancer prob = sum of aca + scc softmax scores
    cancer_prob = (probs[:, 0] + probs[:, 2]).cpu().numpy()   # (B,)
    # Binary prediction: if argmax is lung_n -> 0, else -> 1
    three_cls  = logits.argmax(1).cpu().numpy()    # (B,)
    bin_preds  = (three_cls != NORMAL_CLASS_INDEX).astype(int)
    return bin_preds, cancer_prob


# ════════════════════════════════════════════════════════════════════════════
#  Inference
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_bin_preds  = []
    all_labels     = []
    all_cancer_prob = []

    for imgs, labels in loader:
        # imgs/labels already on device if using PrefetchLoader
        if imgs.device.type == "cpu":
            imgs   = imgs.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and CFG.AMP)):
            logits = model(imgs)

        bin_preds, cancer_prob = logits_to_binary(logits)
        all_bin_preds.extend(bin_preds)
        all_cancer_prob.extend(cancer_prob)
        all_labels.extend(labels.cpu().numpy())

    return (np.array(all_bin_preds),
            np.array(all_labels),
            np.array(all_cancer_prob))


# ════════════════════════════════════════════════════════════════════════════
#  Confusion matrix
# ════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, save_path):
    class_names = ["Normal (0)", "Tumor (1)"]
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, norm], ["d", ".2f"], ["Counts", "Normalised (row %)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix — {title}", fontsize=12, pad=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

    plt.suptitle(
        "HAGCA-Net Cross-Dataset Test  |  PatchCamelyon (PCam)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

# Use the full PCam test split (32,768 images).
# Set MAX_SAMPLES to a smaller number (e.g. 5000) for a quick sanity check.
MAX_SAMPLES = None   # None = all 32,768


def main():
    setup_device()
    ensure_dirs()
    device = torch.device(CFG.DEVICE)
    logger = get_logger("cross_dataset")

    logger.info("=" * 60)
    logger.info("  STEP 10: CROSS-DATASET TEST (PatchCamelyon)")
    logger.info("=" * 60)

    # ── Paths ─────────────────────────────────────────────────────────────
    pcam_root = CFG.PROJECT_ROOT / "data" / "external_test" / "archive"
    img_h5    = pcam_root / "pcam" / "test_split.h5"
    lbl_h5    = (pcam_root / "Labels" / "Labels" /
                 "camelyonpatch_level_2_split_test_y.h5")

    for p in [img_h5, lbl_h5]:
        if not p.exists():
            logger.error(f"Missing: {p}")
            sys.exit(1)

    logger.info(f"Images : {img_h5}")
    logger.info(f"Labels : {lbl_h5}")

    # ── Dataset / DataLoader ──────────────────────────────────────────────
    ds = PCamDataset(img_h5, lbl_h5, transform=PCAM_TRANSFORM,
                     max_samples=MAX_SAMPLES)
    logger.info(f"PCam test samples: {len(ds):,}")

    loader = DataLoader(
        ds, batch_size=CFG.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    prefetch = PrefetchLoader(loader, device)

    # ── Load model ────────────────────────────────────────────────────────
    ckpt_path = CFG.BEST_MODEL_PATH
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = HAGCANet(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    logger.info(f"Loaded checkpoint: {ckpt_path.name} "
                f"(epoch {ckpt.get('epoch','?')}, "
                f"val_F1={ckpt.get('val_f1','?'):.4f})")

    # ── Inference ─────────────────────────────────────────────────────────
    logger.info("Running inference on PCam test set ...")
    logger.info("  Class mapping: lung_aca/lung_scc -> tumor(1), lung_n -> normal(0)")
    preds, labels, cancer_probs = run_inference(model, prefetch, device)

    # ── Metrics ───────────────────────────────────────────────────────────
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", zero_division=0)
    rec  = recall_score(labels, preds, average="binary", zero_division=0)
    f1   = f1_score(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, cancer_probs)
    except Exception:
        auc = float("nan")

    report = classification_report(
        labels, preds,
        target_names=["Normal", "Tumor"],
        digits=4,
    )

    # ── Print ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  CROSS-DATASET RESULTS (PCam test split)")
    logger.info("=" * 60)
    logger.info(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    logger.info(f"  Precision      : {prec:.4f}")
    logger.info(f"  Recall         : {rec:.4f}")
    logger.info(f"  F1 (binary)    : {f1:.4f}")
    logger.info(f"  ROC-AUC        : {auc:.4f}")
    logger.info(f"  Samples tested : {len(labels):,}")
    logger.info("\n" + report)

    # ── Save JSON ─────────────────────────────────────────────────────────
    metrics = {
        "dataset":        "PatchCamelyon_test",
        "n_samples":      int(len(labels)),
        "accuracy":       round(float(acc),  4),
        "precision":      round(float(prec), 4),
        "recall":         round(float(rec),  4),
        "f1_binary":      round(float(f1),   4),
        "roc_auc":        round(float(auc),  4),
        "class_mapping":  {
            "lung_aca (idx 0)": "tumor (1)",
            "lung_n   (idx 1)": "normal (0)",
            "lung_scc (idx 2)": "tumor (1)",
        },
        "checkpoint":     ckpt_path.name,
    }
    json_path = CFG.METRICS_DIR / "cross_dataset_metrics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {json_path}")

    rpt_path = CFG.METRICS_DIR / "cross_dataset_report.txt"
    with open(rpt_path, "w") as f:
        f.write("HAGCA-Net Cross-Dataset Evaluation — PatchCamelyon\n")
        f.write("=" * 60 + "\n")
        for k, v in metrics.items():
            if k not in ("class_mapping",):
                f.write(f"{k}: {v}\n")
        f.write("\n" + report)
    logger.info(f"Report saved: {rpt_path}")

    # ── Confusion matrix plot ──────────────────────────────────────────────
    cm_path = CFG.PLOTS_DIR / "cross_dataset_confusion.png"
    CFG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(labels, preds, cm_path)

    logger.info("\nNext: python src\11_ablation.py  (or: python main.py)")


if __name__ == "__main__":
    main()

