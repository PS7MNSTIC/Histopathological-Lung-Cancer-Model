"""
02_data_splitting.py - Step 3 of the HAGCA-Net pipeline.

CRITICAL: Uses GROUP-BASED splitting, NOT random splitting.
Since LC25000 has no real patient/slide IDs, we create synthetic
groups by assigning sequential group_ids across images of the same class.

What this script does:
  1. Reads the manifest.csv produced by Step 2.
  2. Filters to valid images only.
  3. Assigns a group_id to every image (NUM_GROUPS groups per class).
  4. Performs GroupShuffleSplit: 70% train / 15% val / 15% test.
  5. Saves three CSVs: train.csv, val.csv, test.csv in data/splits/.
  6. Verifies no group leaks across splits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG, get_logger, ensure_dirs

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


logger = get_logger("data_splitting")


# ─────────────────────────────────────────────────────────────
# Group assignment
# ─────────────────────────────────────────────────────────────

def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a synthetic group_id to each image within each class.
    Images in the same group will ALWAYS stay in the same split.
    Groups are assigned by cycling 0..NUM_GROUPS-1 across sorted filenames.
    """
    df = df.copy()
    df["group_id"] = -1
    for cls in CFG.LUNG_CLASSES:
        mask = df["label"] == cls
        class_df = df[mask].reset_index()
        n = len(class_df)
        group_ids = [i % CFG.NUM_GROUPS for i in range(n)]
        df.loc[mask, "group_id"] = group_ids
        logger.info(f"  {cls}: {n} images -> {CFG.NUM_GROUPS} groups (~{n//CFG.NUM_GROUPS} per group)")
    return df


# ─────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────

def group_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits into train / val / test using GroupShuffleSplit.
    No group appears in more than one split (prevents data leakage).
    """
    X = df.index.values
    y = df["label_idx"].values
    g = df["group_id"].values

    # First split: train vs (val + test)
    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG.VAL_RATIO + CFG.TEST_RATIO,
        random_state=CFG.RANDOM_SEED
    )
    train_idx, valtest_idx = next(gss1.split(X, y, g))

    df_train  = df.iloc[train_idx].copy()
    df_valtest = df.iloc[valtest_idx].copy()

    # Second split: val vs test (from valtest pool)
    X2 = df_valtest.index.values
    y2 = df_valtest["label_idx"].values
    g2 = df_valtest["group_id"].values

    val_fraction = CFG.VAL_RATIO / (CFG.VAL_RATIO + CFG.TEST_RATIO)

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=1 - val_fraction,
        random_state=CFG.RANDOM_SEED
    )
    val_idx, test_idx = next(gss2.split(X2, y2, g2))

    df_val  = df_valtest.iloc[val_idx].copy()
    df_test = df_valtest.iloc[test_idx].copy()

    # Tag each split
    df_train["split"] = "train"
    df_val["split"]   = "val"
    df_test["split"]  = "test"

    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────
# Verification — no group leaks
# ─────────────────────────────────────────────────────────────

def verify_no_leakage(df_train, df_val, df_test):
    """Asserts that no group_id appears in more than one split."""
    train_groups = set(df_train["group_id"].unique())
    val_groups   = set(df_val["group_id"].unique())
    test_groups  = set(df_test["group_id"].unique())

    tv_leak = train_groups & val_groups
    tt_leak = train_groups & test_groups
    vt_leak = val_groups   & test_groups

    if tv_leak or tt_leak or vt_leak:
        logger.error(f"GROUP LEAKAGE DETECTED!")
        logger.error(f"  Train-Val overlap  : {tv_leak}")
        logger.error(f"  Train-Test overlap : {tt_leak}")
        logger.error(f"  Val-Test overlap   : {vt_leak}")
        raise RuntimeError("Group leakage found — split is invalid.")
    else:
        logger.info("  No group leakage detected.")


def report_split(df_train, df_val, df_test):
    """Log the class distribution for each split."""
    total = len(df_train) + len(df_val) + len(df_test)
    logger.info("=" * 55)
    logger.info("SPLIT REPORT")
    logger.info("=" * 55)
    logger.info(f"  Train  : {len(df_train):>5} ({100*len(df_train)/total:.1f}%)")
    logger.info(f"  Val    : {len(df_val):>5} ({100*len(df_val)/total:.1f}%)")
    logger.info(f"  Test   : {len(df_test):>5} ({100*len(df_test)/total:.1f}%)")
    logger.info("-" * 55)
    for split_name, sdf in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        logger.info(f"\n  {split_name} class distribution:")
        for cls in CFG.LUNG_CLASSES:
            n = (sdf["label"] == cls).sum()
            logger.info(f"    {cls:<15}: {n}")
    logger.info("=" * 55)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    logger.info("=" * 55)
    logger.info("STEP 3: DATA SPLITTING (Group-based)")
    logger.info("=" * 55)

    manifest_path = CFG.SPLITS_DIR / "manifest.csv"
    if not manifest_path.exists():
        logger.error(f"manifest.csv not found at {manifest_path}")
        logger.error("Run 01_data_cleaning.py first.")
        return

    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest: {len(df)} total records")

    # Filter to valid lung images only
    df = df[df["is_valid"] == True].reset_index(drop=True)
    df = df[df["label"].isin(CFG.LUNG_CLASSES)].reset_index(drop=True)
    logger.info(f"Valid lung images: {len(df)}")

    # Assign groups
    logger.info(f"Assigning {CFG.NUM_GROUPS} synthetic groups per class...")
    df = assign_groups(df)

    # Split
    logger.info("Performing GroupShuffleSplit (70/15/15)...")
    df_train, df_val, df_test = group_split(df)

    # Verify
    logger.info("Verifying no group leakage...")
    verify_no_leakage(df_train, df_val, df_test)

    # Report
    report_split(df_train, df_val, df_test)

    # Save
    df_train.to_csv(CFG.SPLITS_DIR / "train.csv", index=False)
    df_val.to_csv(CFG.SPLITS_DIR / "val.csv",   index=False)
    df_test.to_csv(CFG.SPLITS_DIR / "test.csv",  index=False)

    # Also save the full manifest with group_ids and splits
    full = pd.concat([df_train, df_val, df_test])
    full.to_csv(CFG.SPLITS_DIR / "full_split.csv", index=False)

    logger.info(f"Saved: train.csv, val.csv, test.csv, full_split.csv -> {CFG.SPLITS_DIR}")
    logger.info("[OK] Step 02 complete. Next: python src\03_preprocessing.py  (or: python main.py)")


if __name__ == "__main__":
    main()


