"""
01_data_cleaning.py - Step 2 of the HAGCA-Net pipeline.

What this script does:
  1. Scans all lung class images in both train and test folders.
  2. Detects corrupted / unreadable images.
  3. Detects exact duplicate images using MD5 hashing.
  4. Verifies labels match the expected class folders.
  5. Outputs a clean manifest CSV: data/splits/manifest.csv
     Columns: filepath, label, label_idx, split, group_id, is_valid
"""

import sys
import hashlib
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG, get_logger, ensure_dirs

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


logger = get_logger("data_cleaning")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def compute_md5(filepath: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 hash of a file for duplicate detection."""
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def is_valid_image(filepath: Path) -> bool:
    """Returns True if the image can be loaded and has valid shape."""
    try:
        img = cv2.imread(str(filepath))
        if img is None:
            return False
        if img.ndim != 3 or img.shape[2] != 3:
            return False
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# Main scan
# ─────────────────────────────────────────────────────────────

def scan_directory(base_dir: Path, split_name: str) -> list[dict]:
    """
    Walk base_dir/<class_folder>/*.jpg and collect metadata for each image.
    Only processes LUNG_CLASSES; skips colon classes.
    """
    records = []
    for cls in CFG.LUNG_CLASSES:
        cls_dir = base_dir / cls
        if not cls_dir.exists():
            logger.warning(f"Class folder missing: {cls_dir}")
            continue
        image_files = sorted([
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ])
        logger.info(f"  [{split_name}/{cls}] Found {len(image_files)} files")
        for fp in tqdm(image_files, desc=f"{split_name}/{cls}", leave=False):
            records.append({
                "filepath":  str(fp),
                "label":     cls,
                "label_idx": CFG.CLASS_TO_IDX[cls],
                "split":     split_name,
                "group_id":  -1,        # assigned in Step 3
                "md5":       "",        # computed below
                "is_valid":  False,     # computed below
            })
    return records


def clean(records: list[dict]) -> pd.DataFrame:
    """
    1. Check each image for readability.
    2. Compute MD5 and flag duplicates.
    Returns a DataFrame with all info including is_valid flag.
    """
    logger.info("Checking image validity and computing hashes...")
    hash_map = defaultdict(list)   # md5 -> list of filepaths

    for rec in tqdm(records, desc="Validating"):
        fp = Path(rec["filepath"])
        rec["is_valid"] = is_valid_image(fp)
        rec["md5"]      = compute_md5(fp) if rec["is_valid"] else ""
        if rec["md5"]:
            hash_map[rec["md5"]].append(rec["filepath"])

    # Flag duplicates: keep only first occurrence
    duplicate_paths = set()
    for md5, paths in hash_map.items():
        if len(paths) > 1:
            for dup in paths[1:]:   # keep paths[0], mark the rest
                duplicate_paths.add(dup)
                logger.warning(f"Duplicate found: {dup}  (original: {paths[0]})")

    for rec in records:
        if rec["filepath"] in duplicate_paths:
            rec["is_valid"] = False

    df = pd.DataFrame(records)
    return df


def report(df: pd.DataFrame):
    """Print and log a summary of the cleaning results."""
    total     = len(df)
    valid     = df["is_valid"].sum()
    invalid   = total - valid
    corrupted = df[~df["is_valid"] & (df["md5"] == "")]
    dups      = invalid - len(corrupted)

    logger.info("=" * 55)
    logger.info("DATA CLEANING REPORT")
    logger.info("=" * 55)
    logger.info(f"  Total images scanned : {total}")
    logger.info(f"  Valid images         : {valid}")
    logger.info(f"  Corrupted / unreadable: {len(corrupted)}")
    logger.info(f"  Duplicates removed   : {dups}")
    logger.info("-" * 55)
    for cls in CFG.LUNG_CLASSES:
        subset = df[(df["label"] == cls) & df["is_valid"]]
        logger.info(f"  {cls:<15} : {len(subset)} valid images")
    logger.info("=" * 55)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    logger.info("=" * 55)
    logger.info("STEP 2: DATA CLEANING")
    logger.info("=" * 55)

    # Scan both existing splits
    logger.info(f"Scanning train dir: {CFG.LC25000_TRAIN_DIR}")
    train_records = scan_directory(CFG.LC25000_TRAIN_DIR, "train")

    logger.info(f"Scanning test dir : {CFG.LC25000_TEST_DIR}")
    test_records  = scan_directory(CFG.LC25000_TEST_DIR,  "test")

    all_records = train_records + test_records
    logger.info(f"Total records before cleaning: {len(all_records)}")

    # Clean
    df = clean(all_records)

    # Report
    report(df)

    # Save manifest
    manifest_path = CFG.SPLITS_DIR / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    logger.info(f"Manifest saved to: {manifest_path}")
    logger.info("[OK] Step 01 complete. Next: python src\02_data_splitting.py  (or: python main.py)")

    return df


if __name__ == "__main__":
    main()


