"""
14_compare_scenarios.py
=======================
Three-way scenario comparison for HAGCA-Net.

Scenario 1 — LC25000 -> LC25000
  Model trained and tested on lung histopathology (3 classes).
  Source: results/metrics/test_metrics.json

Scenario 2 — LC25000 -> PCam  (zero-shot)
  Model trained on LC25000, evaluated on PatchCamelyon without fine-tuning.
  Source: results/metrics/cross_dataset_metrics.json

Scenario 3 — PCam -> PCam
  Model trained, validated, and tested entirely on PatchCamelyon.
  Source: results/metrics/pcam_train_test_metrics.json

Outputs:
  results/summary/scenario_comparison.txt
  results/summary/scenario_comparison.json
  results/plots/scenario_comparison.png

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\14_compare_scenarios.py
"""

import sys, json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import CFG, setup_device, ensure_dirs, get_logger


# ════════════════════════════════════════════════════════════════════════════
#  Load metrics
# ════════════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_scenario_rows(s1, s2, s3):
    """
    Normalise metrics across the three scenarios into a unified list of dicts.
    F1 is always the task-appropriate variant:
      S1  -> f1_macro  (3-class)
      S2  -> f1_binary (binary, cross-domain)
      S3  -> f1_binary (binary, in-domain)
    """
    rows = [
        {
            "label":       "S1: LC25000 → LC25000",
            "train_data":  "LC25000 (lung)",
            "test_data":   "LC25000 (lung)",
            "n_test":      s1["n_test"],
            "task":        "3-class (aca / n / scc)",
            "accuracy":    s1["accuracy"],
            "f1":          s1["f1_macro"],
            "roc_auc":     s1["roc_auc"],
            "precision":   s1["precision"],
            "recall":      s1["recall"],
            "f1_label":    "F1-macro",
            "domain_shift": False,
        },
        {
            "label":       "S2: LC25000 → PCam (zero-shot)",
            "train_data":  "LC25000 (lung)",
            "test_data":   "PatchCamelyon (lymph node)",
            "n_test":      s2["n_samples"],
            "task":        "Binary (normal / tumor)",
            "accuracy":    s2["accuracy"],
            "f1":          s2["f1_binary"],
            "roc_auc":     s2["roc_auc"],
            "precision":   s2["precision"],
            "recall":      s2["recall"],
            "f1_label":    "F1-binary",
            "domain_shift": True,
        },
        {
            "label":       "S3: PCam → PCam",
            "train_data":  "PatchCamelyon (lymph node)",
            "test_data":   "PatchCamelyon (lymph node)",
            "n_test":      s3["n_test"],
            "task":        "Binary (normal / tumor)",
            "accuracy":    s3["accuracy"],
            "f1":          s3["f1_binary"],
            "roc_auc":     s3["roc_auc"],
            "precision":   s3["precision"],
            "recall":      s3["recall"],
            "f1_label":    "F1-binary",
            "domain_shift": False,
        },
    ]
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  Comparison plot  (grouped bar chart + summary table)
# ════════════════════════════════════════════════════════════════════════════

COLORS = {
    "S1: LC25000 → LC25000":              "#2E75B6",
    "S2: LC25000 → PCam (zero-shot)":     "#ED7D31",
    "S3: PCam → PCam":                    "#70AD47",
}

def plot_comparison(rows, save_path):
    metrics     = ["accuracy", "f1", "roc_auc", "precision", "recall"]
    metric_lbls = ["Accuracy", "F1\n(task)", "ROC-AUC", "Precision", "Recall"]
    labels      = [r["label"] for r in rows]
    colors      = [COLORS[r["label"]] for r in rows]

    x = np.arange(len(metrics))
    n = len(rows)
    w = 0.22
    offsets = np.linspace(-(n - 1) / 2 * w, (n - 1) / 2 * w, n)

    fig, (ax_bar, ax_tbl) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [2.2, 1]}
    )

    # ── Bar chart ─────────────────────────────────────────────────────────
    for i, (row, color, offset) in enumerate(zip(rows, colors, offsets)):
        vals = [row[m] for m in metrics]
        bars = ax_bar.bar(x + offset, vals, w, color=color, alpha=0.88,
                          label=row["label"], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#333333", fontweight="bold",
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_lbls, fontsize=11)
    ax_bar.set_ylim(0.0, 1.13)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("HAGCA-Net — Three-Scenario Comparison", fontsize=13, pad=10)
    ax_bar.legend(fontsize=9, loc="upper right")
    ax_bar.axhline(y=0.50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
                   label="_nolegend_")
    ax_bar.axhline(y=1.00, color="gray", linestyle=":",  linewidth=0.8, alpha=0.4,
                   label="_nolegend_")
    ax_bar.grid(axis="y", alpha=0.25)

    # Annotate domain-shift scenario
    ax_bar.annotate(
        "Zero-shot\ndomain shift", xy=(x[2] + offsets[1], rows[1]["roc_auc"] + 0.04),
        xytext=(x[2] + offsets[1] + 0.6, rows[1]["roc_auc"] + 0.15),
        fontsize=8, color=COLORS["S2: LC25000 → PCam (zero-shot)"],
        arrowprops=dict(arrowstyle="->", color=COLORS["S2: LC25000 → PCam (zero-shot)"],
                        lw=1.2),
    )

    # ── Summary table ──────────────────────────────────────────────────────
    ax_tbl.axis("off")
    col_labels = ["", "S1", "S2\n(zero-shot)", "S3"]
    row_data   = [
        ["Train data",  "LC25000", "LC25000",  "PCam"],
        ["Test data",   "LC25000", "PCam",     "PCam"],
        ["Task",        "3-class", "Binary",   "Binary"],
        ["Accuracy",
            f"{rows[0]['accuracy']:.4f}",
            f"{rows[1]['accuracy']:.4f}",
            f"{rows[2]['accuracy']:.4f}"],
        ["F1 (task)",
            f"{rows[0]['f1']:.4f}",
            f"{rows[1]['f1']:.4f}",
            f"{rows[2]['f1']:.4f}"],
        ["ROC-AUC",
            f"{rows[0]['roc_auc']:.4f}",
            f"{rows[1]['roc_auc']:.4f}",
            f"{rows[2]['roc_auc']:.4f}"],
        ["Precision",
            f"{rows[0]['precision']:.4f}",
            f"{rows[1]['precision']:.4f}",
            f"{rows[2]['precision']:.4f}"],
        ["Recall",
            f"{rows[0]['recall']:.4f}",
            f"{rows[1]['recall']:.4f}",
            f"{rows[2]['recall']:.4f}"],
        ["# Test imgs",
            f"{rows[0]['n_test']:,}",
            f"{rows[1]['n_test']:,}",
            f"{rows[2]['n_test']:,}"],
    ]

    tbl = ax_tbl.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.05, 1, 0.95],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2E75B6")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Style metric rows
    scenario_colors_light = ["#D6E4F0", "#FCE4D1", "#E2EFDA"]
    for i, row_d in enumerate(row_data):
        tbl[i + 1, 0].set_text_props(fontweight="bold")
        tbl[i + 1, 0].set_facecolor("#F2F2F2")
        if i >= 3:   # metric rows
            for j in range(1, 4):
                vals = [float(row_data[i][jj]) for jj in range(1, 4)
                        if row_data[i][jj].replace('.','').isdigit()]
                if len(vals) == 3:
                    best_j = int(np.argmax(vals)) + 1
                    if j == best_j:
                        tbl[i + 1, j].set_facecolor("#C6EFCE")  # green = best
                        tbl[i + 1, j].set_text_props(fontweight="bold")

    ax_tbl.set_title("Summary Table", fontsize=11, pad=8)

    plt.suptitle(
        "HAGCA-Net  |  Scenario Comparison\n"
        "S1: In-domain lung | S2: Zero-shot cross-domain | S3: In-domain PCam",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Comparison figure saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Text report
# ════════════════════════════════════════════════════════════════════════════

def build_text_report(rows):
    div = "=" * 65
    lines = [
        div,
        "  HAGCA-Net — THREE-SCENARIO COMPARISON",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        div, "",
        "  SCENARIOS",
        "  ---------",
        "  S1: Train=LC25000 (lung),  Test=LC25000 (lung)   [3-class]",
        "  S2: Train=LC25000 (lung),  Test=PCam    (lymph)  [binary, zero-shot]",
        "  S3: Train=PCam    (lymph), Test=PCam    (lymph)  [binary]",
        "", div,
        f"  {'Metric':<18} {'S1 (LC->LC)':>14} {'S2 (LC->PCam)':>14} {'S3 (PCam->PCam)':>16}",
        "  " + "-" * 63,
    ]
    metric_map = [
        ("Accuracy",  "accuracy"),
        ("F1 (task)", "f1"),
        ("ROC-AUC",   "roc_auc"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
    ]
    for label, key in metric_map:
        vals = [r[key] for r in rows]
        best = max(vals)
        formatted = []
        for v in vals:
            s = f"{v:.4f}"
            formatted.append(s + " *" if v == best else s + "  ")
        lines.append(
            f"  {label:<18} {formatted[0]:>16} {formatted[1]:>15} {formatted[2]:>16}"
        )

    lines += [
        "",
        "  * = best value in row",
        "",
        div,
        "  KEY FINDINGS",
        div,
        "",
        "  1. S1 (in-domain lung): HAGCA-Net achieves near-perfect",
        f"     classification (Accuracy {rows[0]['accuracy']*100:.2f}%, AUC {rows[0]['roc_auc']:.4f}).",
        "",
        "  2. S2 (zero-shot domain shift): Without any fine-tuning on",
        "     lymph node images, the model still achieves above-chance AUC",
        f"     ({rows[1]['roc_auc']:.4f}) and high tumor recall ({rows[1]['recall']:.4f}).",
        "     However precision is limited due to severe domain shift.",
        "",
        "  3. S3 (in-domain PCam): Fine-tuning on PCam data substantially",
        f"     improves precision (from {rows[1]['precision']:.4f} to {rows[2]['precision']:.4f})",
        f"     and overall accuracy (from {rows[1]['accuracy']*100:.2f}% to {rows[2]['accuracy']*100:.2f}%),",
        "     confirming that domain-specific training is critical for",
        "     high-precision deployment in a new tissue type.",
        "",
        div,
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    setup_device()
    ensure_dirs()
    logger = get_logger("compare_scenarios")

    logger.info("=" * 60)
    logger.info("  STEP 14: THREE-SCENARIO COMPARISON")
    logger.info("=" * 60)

    m_dir = CFG.METRICS_DIR
    paths = {
        "s1": m_dir / "test_metrics.json",
        "s2": m_dir / "cross_dataset_metrics.json",
        "s3": m_dir / "pcam_train_test_metrics.json",
    }
    for name, p in paths.items():
        if not p.exists():
            logger.error(f"Missing {name} metrics: {p}")
            logger.error("Ensure steps 08, 10, and 13 have been run first.")
            sys.exit(1)

    s1 = load_json(paths["s1"])
    s2 = load_json(paths["s2"])
    s3 = load_json(paths["s3"])
    logger.info("All three scenario metric files loaded.")

    rows = build_scenario_rows(s1, s2, s3)

    # ── Plot ──────────────────────────────────────────────────────────────
    summary_dir = CFG.RESULTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_path = CFG.PLOTS_DIR / "scenario_comparison.png"
    CFG.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(rows, plot_path)

    # ── Text report ────────────────────────────────────────────────────────
    report = build_text_report(rows)
    txt_path = summary_dir / "scenario_comparison.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Text report saved: {txt_path}")
    print("\n" + report)

    # ── JSON ──────────────────────────────────────────────────────────────
    json_path = summary_dir / "scenario_comparison.json"
    with open(json_path, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "scenarios": rows,
        }, f, indent=2)
    logger.info(f"JSON saved: {json_path}")

    logger.info("\nPipeline complete! All 14 steps done.")
    logger.info("See results/summary/scenario_comparison.txt for the full comparison.")


if __name__ == "__main__":
    main()
