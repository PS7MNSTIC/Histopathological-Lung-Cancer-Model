"""
main.py — HAGCA-Net Full Pipeline Runner
=========================================
Runs all 12 pipeline steps in sequence.

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python main.py                  # run full pipeline, skip already-done steps
    python main.py --force          # re-run every step regardless
    python main.py --from 7         # resume from step 7
    python main.py --only 8         # run only step 8
    python main.py --list           # print pipeline steps and exit

Steps
-----
  01  Data Cleaning          (removes duplicates / corrupted images)
  02  Data Splitting         (group-based 70/15/15 split)
  03  Preprocessing          (Reinhard stain norm + CLAHE)
  04  DataLoader Smoke Test  (verifies augmentation pipeline on GPU)
  05  GAN Augmentation       (DCGAN — synthetic lung_scc images)
  06  Model Smoke Test       (builds HAGCA-Net, verifies forward pass)
  07  Training               (two-phase, AMP, GroupKFold)
  08  Evaluation             (accuracy, F1, ROC-AUC, confusion matrix)
  09  Grad-CAM               (explainability heatmaps, 5 per class)
  10  Cross-Dataset Test     (zero-shot PatchCamelyon evaluation)
  11  Ablation Study         (4 architectural variants)
  12  Result Analysis        (summary report + dashboard figure)
  13  PCam Train + Test      (train/val/test entirely on PatchCamelyon, ~45 min)
  14  Scenario Comparison    (3-way: LC->LC / LC->PCam zero-shot / PCam->PCam)
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import timedelta

# ── Project root = folder containing this file ────────────────────────────
PROJECT = Path(__file__).resolve().parent
PYTHON  = sys.executable          # uses whichever python ran main.py

# ════════════════════════════════════════════════════════════════════════════
#  Pipeline step definitions
#  done_check: relative path (from PROJECT) whose existence means the step
#              was already completed. None = always run (smoke tests).
# ════════════════════════════════════════════════════════════════════════════

PIPELINE = [
    {
        "num":        1,
        "name":       "Data Cleaning",
        "script":     "src/01_data_cleaning.py",
        "done_check": "data/splits/manifest.csv",
        "desc":       "Remove duplicates and corrupted images from LC25000",
    },
    {
        "num":        2,
        "name":       "Data Splitting",
        "script":     "src/02_data_splitting.py",
        "done_check": "data/splits/train.csv",
        "desc":       "Group-based 70/15/15 split (zero data leakage)",
    },
    {
        "num":        3,
        "name":       "Preprocessing",
        "script":     "src/03_preprocessing.py",
        "done_check": "data/splits/train_processed.csv",
        "desc":       "Reinhard stain normalisation + CLAHE (16 workers)",
    },
    {
        "num":        4,
        "name":       "DataLoader Smoke Test",
        "script":     "src/04_augmentation.py",
        "done_check": None,          # always run — quick GPU smoke test
        "desc":       "Verify augmentation pipeline and DataLoaders on GPU",
    },
    {
        "num":        5,
        "name":       "GAN Augmentation",
        "script":     "src/05_gan_augment.py",
        "done_check": "data/gan_synthetic/lung_scc",
        "desc":       "DCGAN: train 100 epochs, save 500 synthetic lung_scc images",
    },
    {
        "num":        6,
        "name":       "Model Smoke Test",
        "script":     "src/06_model_hagcanet.py",
        "done_check": None,          # always run — quick GPU smoke test
        "desc":       "Build HAGCA-Net (~100M params), verify forward pass on GPU",
    },
    {
        "num":        7,
        "name":       "Training",
        "script":     "src/07_train.py",
        "done_check": "checkpoints/hagcanet_best.pth",
        "desc":       "Two-phase training (frozen→full), AMP, early stopping",
    },
    {
        "num":        8,
        "name":       "Evaluation",
        "script":     "src/08_evaluate.py",
        "done_check": "results/metrics/test_metrics.json",
        "desc":       "Accuracy, F1-macro, ROC-AUC + confusion matrix on test set",
    },
    {
        "num":        9,
        "name":       "Grad-CAM",
        "script":     "src/09_gradcam.py",
        "done_check": "results/gradcam",
        "desc":       "Explainability heatmaps — 5 images per class (15 total)",
    },
    {
        "num":        10,
        "name":       "Cross-Dataset Test",
        "script":     "src/10_cross_dataset.py",
        "done_check": "results/metrics/cross_dataset_metrics.json",
        "desc":       "Zero-shot evaluation on PatchCamelyon (32,768 images)",
    },
    {
        "num":        11,
        "name":       "Ablation Study",
        "script":     "src/11_ablation.py",
        "done_check": "results/metrics/ablation_results.json",
        "desc":       "4-variant post-hoc architectural ablation",
    },
    {
        "num":        12,
        "name":       "Result Analysis",
        "script":     "src/12_result_analysis.py",
        "done_check": "results/summary/full_results_summary.txt",
        "desc":       "Compile all metrics into summary report + dashboard figure",
    },
    {
        "num":        13,
        "name":       "PCam Train + Test",
        "script":     "src/13_pcam_train_test.py",
        "done_check": "results/metrics/pcam_train_test_metrics.json",
        "desc":       "Train, validate & test HAGCA-Net entirely on PatchCamelyon (~45 min)",
    },
    {
        "num":        14,
        "name":       "Scenario Comparison",
        "script":     "src/14_compare_scenarios.py",
        "done_check": "results/summary/scenario_comparison.txt",
        "desc":       "3-way comparison: LC25000->LC25000 / LC25000->PCam / PCam->PCam",
    },
]


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

W  = 65   # banner width

def banner(text, char="="):
    print(char * W)
    print(f"  {text}")
    print(char * W)

def step_header(step):
    num  = step["num"]
    name = step["name"]
    desc = step["desc"]
    print()
    print("─" * W)
    print(f"  STEP {num:02d} / 12 — {name}")
    print(f"  {desc}")
    print("─" * W)

def is_done(step):
    """Return True if the step's sentinel file/folder already exists."""
    chk = step["done_check"]
    if chk is None:
        return False
    path = PROJECT / chk
    if path.is_file():
        return True
    if path.is_dir() and any(path.iterdir()):
        return True
    return False

def fmt_elapsed(seconds):
    return str(timedelta(seconds=int(seconds)))

def run_step(step):
    """Run a single pipeline step. Returns True on success, False on failure."""
    script = PROJECT / step["script"]
    if not script.exists():
        print(f"  [ERROR] Script not found: {script}")
        return False

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(PROJECT),
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] Step {step['num']:02d} completed in {fmt_elapsed(elapsed)}")
        return True
    else:
        print(f"\n  [FAILED] Step {step['num']:02d} exited with code {result.returncode}"
              f"  (elapsed: {fmt_elapsed(elapsed)})")
        return False


# ════════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="HAGCA-Net full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--force",    action="store_true",
                   help="Re-run every step even if already done")
    p.add_argument("--from",     dest="from_step", type=int, default=1,
                   metavar="N",  help="Start from step N (default: 1)")
    p.add_argument("--only",     dest="only_step", type=int, default=None,
                   metavar="N",  help="Run only step N")
    p.add_argument("--list",     action="store_true",
                   help="Print pipeline steps and exit")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    banner("HAGCA-Net  |  Full Pipeline Runner")
    print(f"  Project : {PROJECT}")
    print(f"  Python  : {PYTHON}")
    print()

    # ── --list ────────────────────────────────────────────────────────────
    if args.list:
        print(f"  {'#':>2}  {'Name':<24}  {'Script':<30}  {'Status'}")
        print("  " + "─" * 70)
        for s in PIPELINE:
            done = "DONE" if is_done(s) else "pending"
            skip = "(always)" if s["done_check"] is None else ""
            print(f"  {s['num']:>2}  {s['name']:<24}  {s['script']:<30}  {done} {skip}")
        return

    # ── Filter steps to run ───────────────────────────────────────────────
    if args.only_step is not None:
        steps = [s for s in PIPELINE if s["num"] == args.only_step]
        if not steps:
            print(f"  [ERROR] No step with number {args.only_step}")
            sys.exit(1)
    else:
        steps = [s for s in PIPELINE if s["num"] >= args.from_step]

    if not steps:
        print("  Nothing to run.")
        return

    # ── Status preview ────────────────────────────────────────────────────
    print(f"  Steps to run: {[s['num'] for s in steps]}")
    print()
    for s in steps:
        if not args.force and is_done(s):
            status = "SKIP (already done)"
        elif s["done_check"] is None:
            status = "WILL RUN (smoke test)"
        else:
            status = "WILL RUN"
        print(f"    Step {s['num']:02d} — {s['name']:<24}  {status}")

    print()
    input("  Press ENTER to start, Ctrl-C to abort ... ")
    print()

    # ── Run steps ─────────────────────────────────────────────────────────
    pipeline_start = time.time()
    skipped = 0
    passed  = 0
    failed  = 0

    for step in steps:
        step_header(step)

        # Skip if already done (unless --force)
        if not args.force and is_done(step):
            print(f"  [SKIP] Output already exists: {step['done_check']}")
            print(f"         Use --force to re-run this step.")
            skipped += 1
            continue

        success = run_step(step)

        if success:
            passed += 1
        else:
            failed += 1
            print()
            ans = input("  Step failed. Continue anyway? [y/N]: ").strip().lower()
            if ans != "y":
                print("  Aborting pipeline.")
                break

    # ── Final summary ─────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    print()
    banner("PIPELINE COMPLETE")
    print(f"  Total time : {fmt_elapsed(total_elapsed)}")
    print(f"  Passed     : {passed}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {failed}")
    print()

    if failed == 0:
        print("  All steps completed successfully.")
        print("  Results are in: C:\\ml_project\\results\\")
        print()
        print("        See results/summary/full_results_summary.txt for all metrics.")
    else:
        print(f"  {failed} step(s) failed. Check the error output above.")
        sys.exit(1)

    print("=" * W)


if __name__ == "__main__":
    main()
