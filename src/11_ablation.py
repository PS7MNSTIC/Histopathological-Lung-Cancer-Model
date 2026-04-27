"""
11_ablation.py
==============
Architectural ablation study for HAGCA-Net.

Four model variants are evaluated on the held-out test set (2,272 images)
using the SAME trained checkpoint, with selected modules selectively bypassed.
This is a standard post-hoc ablation approach used when training each variant
from scratch is computationally prohibitive.

Ablation variants
-----------------
V1: CNN-Only        — EfficientNet-B3 features -> linear head (no Swin, no Graph, no Attention)
V2: +Transformer    — CNN + Swin Transformer fusion -> linear head (no Graph, no Attention)
V3: +Graph          — CNN + Swin + Graph Learning -> linear head (no Context Attention)
V4: Full HAGCA-Net  — complete model (= 08_evaluate.py result)

Why post-hoc ablation?
  Training 4 separate models would take 40+ GPU-hours. Post-hoc ablation
  isolates the contribution of each module using the learned weights,
  giving a valid lower-bound estimate of each component's impact.

Outputs:
  results/metrics/ablation_results.json
  results/plots/ablation_barplot.png

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\11_ablation.py
"""

import sys, json, importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
#  Ablated model wrappers
#  Each wrapper loads the same trained checkpoint, then overrides forward()
#  to skip selected modules, replacing their output with a zero tensor of
#  the correct shape.  The final classifier is always the original one.
# ════════════════════════════════════════════════════════════════════════════

class AblatedHAGCANet(nn.Module):
    """
    Wraps a fully-loaded HAGCANet and selectively disables modules.

    Strategy: always pipe through the TRAINED fusion + classifier so that
    the final linear weights are valid for all variants. Disabled branches
    are replaced with zero tensors of the correct shape, letting the fusion
    gate learn to ignore them (the gate was trained with all branches active,
    so this gives a conservative lower-bound estimate of each module's value).

    Variant definitions
    -------------------
    V1 CNN-Only      : CNN branch → zeros for trans/graph → fusion → attn → cls
    V2 +Transformer  : CNN + Swin → zeros for graph       → fusion → attn → cls
    V3 +Graph        : CNN + Swin + Graph → fusion → classifier  (skip context_attn)
    V4 Full HAGCA-Net: standard HAGCANet forward (all modules active)

    Parameters
    ----------
    base_model    : trained HAGCANet (weights already loaded)
    use_cnn       : bool — run EfficientNet-B3 branch (if False: zero tensor)
    use_transformer: bool — run Swin-Base branch      (if False: zero tensor)
    use_graph     : bool — run Graph Learning module  (if False: zero tensor)
    use_attention : bool — run Context Attention      (if False: f_fused → cls)
    """

    # Fixed dims from CFG / architecture — no need for model attributes
    FEAT_DIM  = 512   # CNN and Swin branch output dim  (CFG.CNN_OUT_DIM / TRANS_OUT_DIM)
    GRAPH_DIM = 256   # GraphLearningModule output dim  (CFG.GNN_HIDDEN_DIM)
    FEAT_CH   = 384   # EfficientNet-B3 last feature-map channels

    def __init__(self, base_model, use_cnn=True, use_transformer=True,
                 use_graph=True, use_attention=True):
        super().__init__()
        self.m               = base_model
        self.use_cnn         = use_cnn
        self.use_transformer = use_transformer
        self.use_graph       = use_graph
        self.use_attention   = use_attention

    @torch.no_grad()
    def forward(self, x):
        m   = self.m
        B   = x.size(0)
        dev = x.device

        # ── CNN branch ────────────────────────────────────────────────────
        if self.use_cnn:
            f_cnn, feat_map = m.cnn_branch(x)                 # (B,512),(B,384,7,7)
        else:
            f_cnn    = torch.zeros(B, self.FEAT_DIM, device=dev)
            feat_map = torch.zeros(B, self.FEAT_CH, 7, 7, device=dev)

        # ── Graph Learning ────────────────────────────────────────────────
        if self.use_graph:
            f_graph = m.graph_module(feat_map)                 # (B,256)
        else:
            f_graph = torch.zeros(B, self.GRAPH_DIM, device=dev)

        # ── Transformer branch ────────────────────────────────────────────
        if self.use_transformer:
            f_trans = m.trans_branch(x)                        # (B,512)
        else:
            f_trans = torch.zeros(B, self.FEAT_DIM, device=dev)

        # ── Adaptive Fusion (always runs — uses trained weights) ──────────
        f_fused = m.fusion(f_cnn, f_trans, f_graph)           # (B,512)

        # ── Context Attention (optional) ──────────────────────────────────
        if self.use_attention:
            f_ctx = m.context_attn(f_fused)                   # (B,512)
        else:
            f_ctx = f_fused                                    # bypass attention

        # ── Classifier (always runs — trained weights) ────────────────────
        return m.classifier(f_ctx)                             # (B,3)


# ════════════════════════════════════════════════════════════════════════════
#  Inference helper
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, loader, device):
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
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(preds, labels, probs):
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return {"accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1),  4),
            "roc_auc":  round(float(auc), 4)}


# ════════════════════════════════════════════════════════════════════════════
#  Ablation bar-plot
# ════════════════════════════════════════════════════════════════════════════

def plot_ablation(results, save_path):
    variants = [r["variant"] for r in results]
    metrics  = ["accuracy", "f1_macro", "roc_auc"]
    labels_m = ["Accuracy", "F1-macro", "ROC-AUC"]
    colors   = ["#4C72B0", "#DD8452", "#55A868"]

    x    = np.arange(len(variants))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (metric, label, color) in enumerate(zip(metrics, labels_m, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x + w)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("HAGCA-Net — Ablation Study (test set)", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.axhline(y=0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="_nolegend_")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Ablation bar chart saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

VARIANTS = [
    # (name,               use_cnn, use_transformer, use_graph, use_attention)
    # V1: only EfficientNet-B3 CNN; trans/graph zeroed; attention bypassed
    ("V1: CNN-Only",        True,   False,           False,     False),
    # V2: CNN + Swin fusion; graph zeroed; attention bypassed
    ("V2: +Transformer",    True,   True,            False,     False),
    # V3: all three branches fused; attention bypassed (f_fused -> classifier)
    ("V3: +Graph",          True,   True,            True,      False),
    # V4: full HAGCA-Net — should reproduce 08_evaluate.py result
    ("V4: Full HAGCA-Net",  True,   True,            True,      True),
]


def main():
    setup_device()
    ensure_dirs()
    device = torch.device(CFG.DEVICE)
    logger = get_logger("ablation")

    logger.info("=" * 60)
    logger.info("  STEP 11: ABLATION STUDY")
    logger.info("=" * 60)

    # ── Load test DataLoader ──────────────────────────────────────────────
    test_csv = CFG.SPLITS_DIR / "test_processed.csv"
    if not test_csv.exists():
        logger.error(f"Missing: {test_csv}")
        sys.exit(1)

    test_df  = pd.read_csv(test_csv)
    test_ds  = LungDataset(test_df, transform=get_val_transforms())
    test_loader = DataLoader(
        test_ds, batch_size=CFG.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    logger.info(f"Test images: {len(test_df)}")

    # ── Load base checkpoint once ─────────────────────────────────────────
    ckpt_path = CFG.BEST_MODEL_PATH
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # ── Evaluate each variant ─────────────────────────────────────────────
    all_results = []

    for (name, use_cnn, use_tfm, use_graph, use_attn) in VARIANTS:
        logger.info(f"\n{'─'*50}")
        logger.info(f"  Variant: {name}")
        logger.info(f"    CNN={use_cnn}  Transformer={use_tfm}  "
                    f"Graph={use_graph}  Attention={use_attn}")

        # Fresh base model, load weights
        base = HAGCANet(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)
        base.load_state_dict(ckpt["state_dict"])
        base.eval()

        # Wrap with ablation logic
        ablated = AblatedHAGCANet(
            base_model=base,
            use_cnn=use_cnn, use_transformer=use_tfm,
            use_graph=use_graph, use_attention=use_attn,
        ).to(device)
        ablated.eval()

        preds, labels, probs = run_inference(ablated, test_loader, device)
        m = compute_metrics(preds, labels, probs)

        logger.info(f"    Accuracy : {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)")
        logger.info(f"    F1-macro : {m['f1_macro']:.4f}")
        logger.info(f"    ROC-AUC  : {m['roc_auc']:.4f}")

        all_results.append({
            "variant":    name,
            "use_cnn":    use_cnn,
            "use_transformer": use_tfm,
            "use_graph":  use_graph,
            "use_attention": use_attn,
            **m,
        })

        # Free memory between variants
        del ablated, base
        torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  ABLATION SUMMARY")
    logger.info("=" * 60)
    header = f"  {'Variant':<22}  {'Accuracy':>10}  {'F1-macro':>10}  {'ROC-AUC':>10}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for r in all_results:
        logger.info(
            f"  {r['variant']:<22}  "
            f"{r['accuracy']:>10.4f}  "
            f"{r['f1_macro']:>10.4f}  "
            f"{r['roc_auc']:>10.4f}"
        )

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = CFG.METRICS_DIR / "ablation_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved: {json_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────
    plot_path = CFG.PLOTS_DIR / "ablation_barplot.png"
    CFG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_ablation(all_results, plot_path)

    logger.info("\nAblation study complete.")
    logger.info("Next: python src\12_result_analysis.py  (or: python main.py)


if __name__ == "__main__":
    main()

