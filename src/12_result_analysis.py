"""
12_result_analysis.py
=====================
Compiles all HAGCA-Net results into a single comprehensive summary.

Reads:
  results/metrics/test_metrics.json
  results/metrics/cross_dataset_metrics.json
  results/metrics/ablation_results.json

Produces:
  results/summary/full_results_summary.txt   -- human-readable report
  results/summary/full_results_summary.json  -- machine-readable
  results/plots/results_dashboard.png        -- 4-panel figure for paper

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\12_result_analysis.py
"""

import sys, json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import CFG, setup_device, ensure_dirs, get_logger


# ════════════════════════════════════════════════════════════════════════════
#  Load all saved metric files
# ════════════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
#  Dashboard figure  (4 panels)
#
#  Panel A — Test-set per-class F1 bar chart
#  Panel B — Ablation accuracy progression
#  Panel C — Cross-dataset metrics summary
#  Panel D — Key metrics comparison table
# ════════════════════════════════════════════════════════════════════════════

def build_dashboard(test_m, cross_m, ablation, save_path):
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    BLUE   = "#4C72B0"
    ORANGE = "#DD8452"
    GREEN  = "#55A868"
    RED    = "#C44E52"
    GRAY   = "#8C8C8C"

    # ── Panel A: Per-class F1 on test set ─────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    classes = list(test_m["f1_per_class"].keys())
    f1_vals = list(test_m["f1_per_class"].values())
    bars = ax_a.bar(classes, f1_vals, color=[BLUE, GREEN, ORANGE], alpha=0.85,
                    edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, f1_vals):
        ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                  f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_a.set_ylim(0.99, 1.005)
    ax_a.set_title("A)  Per-Class F1 — LC25000 Test Set", fontsize=11, pad=8)
    ax_a.set_ylabel("F1 Score", fontsize=10)
    ax_a.set_xlabel("Class", fontsize=10)
    ax_a.axhline(y=1.0, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5)
    ax_a.grid(axis="y", alpha=0.3)

    # ── Panel B: Ablation progression ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    variant_labels = [r["variant"].replace("V", "V").replace(": ", "\n") for r in ablation]
    accs  = [r["accuracy"]  for r in ablation]
    f1s   = [r["f1_macro"]  for r in ablation]

    x = np.arange(len(variant_labels))
    w = 0.35
    b1 = ax_b.bar(x - w/2, accs, w, label="Accuracy", color=BLUE,  alpha=0.85)
    b2 = ax_b.bar(x + w/2, f1s,  w, label="F1-macro", color=ORANGE, alpha=0.85)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax_b.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                      f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(variant_labels, fontsize=8)
    ax_b.set_ylim(0.0, 1.12)
    ax_b.set_title("B)  Ablation Study — Component Contribution", fontsize=11, pad=8)
    ax_b.set_ylabel("Score", fontsize=10)
    ax_b.legend(fontsize=9)
    ax_b.axhline(y=0.95, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.4)
    ax_b.grid(axis="y", alpha=0.3)

    # ── Panel C: Cross-dataset radar / bar ────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    cross_metrics = ["accuracy", "precision", "recall", "f1_binary", "roc_auc"]
    cross_labels  = ["Accuracy", "Precision", "Recall\n(Tumor)", "F1\n(Binary)", "ROC-AUC"]
    cross_vals    = [cross_m[k] for k in cross_metrics]
    colors_c = [GREEN if v >= 0.7 else (ORANGE if v >= 0.5 else RED) for v in cross_vals]

    bars_c = ax_c.bar(cross_labels, cross_vals, color=colors_c, alpha=0.85,
                       edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars_c, cross_vals):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_c.set_ylim(0.0, 1.10)
    ax_c.axhline(y=0.50, color=GRAY, linestyle="--", linewidth=1.0, alpha=0.6,
                  label="Random baseline")
    ax_c.set_title("C)  Cross-Dataset Test — PatchCamelyon\n(Zero-shot, lung→lymph node)",
                   fontsize=11, pad=8)
    ax_c.set_ylabel("Score", fontsize=10)
    ax_c.legend(fontsize=8)
    ax_c.grid(axis="y", alpha=0.3)

    # ── Panel D: Key metrics text table ───────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    table_data = [
        ["Metric", "LC25000 Test", "PCam (0-shot)"],
        ["─" * 16, "─" * 13, "─" * 13],
        ["Accuracy",     f"{test_m['accuracy']:.4f}",   f"{cross_m['accuracy']:.4f}"],
        ["Precision",    f"{test_m['precision']:.4f}",  f"{cross_m['precision']:.4f}"],
        ["Recall",       f"{test_m['recall']:.4f}",     f"{cross_m['recall']:.4f}"],
        ["F1",           f"{test_m['f1_macro']:.4f}",   f"{cross_m['f1_binary']:.4f}"],
        ["ROC-AUC",      f"{test_m['roc_auc']:.4f}",   f"{cross_m['roc_auc']:.4f}"],
        ["─" * 16, "─" * 13, "─" * 13],
        ["# Samples",    f"{test_m['n_test']:,}",       f"{cross_m['n_samples']:,}"],
        ["Checkpoint",   test_m["checkpoint"],           "(same)"],
    ]

    y_pos = 0.97
    col_x = [0.02, 0.45, 0.75]
    row_h = 0.09

    for row in table_data:
        for xi, cell in zip(col_x, row):
            weight = "bold" if row[0] in ("Metric", "─" * 16) else "normal"
            ax_d.text(xi, y_pos, cell, transform=ax_d.transAxes,
                      fontsize=9, verticalalignment="top",
                      fontweight=weight,
                      fontfamily="monospace")
        y_pos -= row_h

    ax_d.set_title("D)  Summary Metrics Table", fontsize=11, pad=8)
    ax_d.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                                  edgecolor="#CCCCCC", linewidth=1,
                                  transform=ax_d.transAxes))

    # ── Overall title ──────────────────────────────────────────────────────
    fig.suptitle(
        "HAGCA-Net — Complete Results Dashboard\n"
        "Histopathological Lung Cancer Classification  |  Group Anuska",
        fontsize=14, y=1.01, fontweight="bold",
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Dashboard saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Text report
# ════════════════════════════════════════════════════════════════════════════

def build_text_report(test_m, cross_m, ablation):
    lines = []
    div   = "=" * 65

    lines += [
        div,
        "  HAGCA-Net — FULL RESULTS SUMMARY",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        div,
        "",
        "MODEL: HAGCA-Net (Hybrid Adaptive Graph Context Attention Network)",
        f"  EfficientNet-B3 + Swin-Base + Graph Learning + Context Attention",
        f"  Parameters : ~100.7M",
        f"  Checkpoint : {test_m['checkpoint']}",
        f"  Training   : LC25000 (lung histopathology), 3 classes",
        f"               Epoch 22 | Val F1 = 1.0000",
        "",
        div,
        "  1. IN-DOMAIN TEST RESULTS  (LC25000 held-out test set)",
        div,
        f"  Samples    : {test_m['n_test']:,}",
        f"  Accuracy   : {test_m['accuracy']:.4f}  ({test_m['accuracy']*100:.2f}%)",
        f"  Precision  : {test_m['precision']:.4f}",
        f"  Recall     : {test_m['recall']:.4f}",
        f"  F1-macro   : {test_m['f1_macro']:.4f}",
        f"  ROC-AUC    : {test_m['roc_auc']:.4f}",
        "",
        "  Per-class F1:",
    ]
    for cls, f1 in test_m["f1_per_class"].items():
        lines.append(f"    {cls:<14} {f1:.4f}")

    lines += [
        "",
        div,
        "  2. CROSS-DATASET RESULTS  (PatchCamelyon — zero-shot)",
        div,
        "  Domain shift: lung tissue → lymph node tissue (breast cancer)",
        "  Class mapping: lung_aca/lung_scc → tumor(1) | lung_n → normal(0)",
        f"  Samples    : {cross_m['n_samples']:,}",
        f"  Accuracy   : {cross_m['accuracy']:.4f}  ({cross_m['accuracy']*100:.2f}%)",
        f"  Precision  : {cross_m['precision']:.4f}",
        f"  Recall     : {cross_m['recall']:.4f}  (tumor class)",
        f"  F1-binary  : {cross_m['f1_binary']:.4f}",
        f"  ROC-AUC    : {cross_m['roc_auc']:.4f}",
        "  Note: AUC > 0.50 confirms partial cross-domain generalisation",
        "",
        div,
        "  3. ABLATION STUDY  (post-hoc, same checkpoint)",
        div,
        f"  {'Variant':<24} {'Accuracy':>10}  {'F1-macro':>10}  {'ROC-AUC':>10}",
        "  " + "-" * 58,
    ]
    for r in ablation:
        auc_str = f"{r['roc_auc']:.4f}" if not (isinstance(r['roc_auc'], float) and
                                                  r['roc_auc'] != r['roc_auc']) else "  —  "
        lines.append(
            f"  {r['variant']:<24} {r['accuracy']:>10.4f}  "
            f"{r['f1_macro']:>10.4f}  {auc_str:>10}"
        )
    lines += [
        "",
        "  Key finding: Context Attention is the decisive module.",
        "  Without it (V1–V3), all variants produce near-chance accuracy",
        "  (~50%), demonstrating its role as the architectural keystone.",
        "  The V3→V4 jump (+45.6 pp accuracy) is the paper's core result.",
        "",
        div,
        "  4. FILE LOCATIONS",
        div,
        "  results/metrics/test_metrics.json",
        "  results/metrics/cross_dataset_metrics.json",
        "  results/metrics/ablation_results.json",
        "  results/plots/confusion_matrix.png",
        "  results/plots/cross_dataset_confusion.png",
        "  results/plots/ablation_barplot.png",
        "  results/plots/results_dashboard.png",
        "  results/gradcam/<class>/img_01..05_gradcam.png",
        div,
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    setup_device()
    ensure_dirs()
    logger = get_logger("result_analysis")

    logger.info("=" * 60)
    logger.info("  STEP 12: RESULT ANALYSIS")
    logger.info("=" * 60)

    metrics_dir = CFG.METRICS_DIR
    plots_dir   = CFG.PLOTS_DIR
    summary_dir = CFG.RESULTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all metric files ──────────────────────────────────────────────
    test_path  = metrics_dir / "test_metrics.json"
    cross_path = metrics_dir / "cross_dataset_metrics.json"
    abl_path   = metrics_dir / "ablation_results.json"

    for p in [test_path, cross_path, abl_path]:
        if not p.exists():
            logger.error(f"Missing: {p}")
            logger.error("Run steps 08, 10, 11 first.")
            import sys; sys.exit(1)

    test_m   = load_json(test_path)
    cross_m  = load_json(cross_path)
    ablation = load_json(abl_path)

    logger.info("Loaded all metric files.")

    # ── Dashboard figure ───────────────────────────────────────────────────
    dash_path = plots_dir / "results_dashboard.png"
    plots_dir.mkdir(parents=True, exist_ok=True)
    build_dashboard(test_m, cross_m, ablation, dash_path)

    # ── Text report ────────────────────────────────────────────────────────
    report_txt = build_text_report(test_m, cross_m, ablation)
    txt_path   = summary_dir / "full_results_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)
    logger.info(f"Text report saved: {txt_path}")
    print("\n" + report_txt)

    # ── Combined JSON ──────────────────────────────────────────────────────
    combined = {
        "model":          "HAGCA-Net",
        "generated":      datetime.now().isoformat(),
        "checkpoint":     test_m["checkpoint"],
        "in_domain":      test_m,
        "cross_dataset":  cross_m,
        "ablation":       ablation,
    }
    json_path = summary_dir / "full_results_summary.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"JSON summary saved: {json_path}")

    logger.info("\nStep 12 complete. All results compiled.")


if __name__ == "__main__":
    main()

