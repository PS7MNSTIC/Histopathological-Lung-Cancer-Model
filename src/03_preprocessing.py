"""
03_preprocessing.py - Step 4 of the HAGCA-Net pipeline.

FAST VERSION: Parallel processing using all 20 CPU cores.

What changed from slow version:
  - Replaced Macenko staintools (slow SVD, single-threaded) with
    Reinhard normalization (pure NumPy, ~50x faster per image).
  - Processes images in parallel using multiprocessing.Pool (16 workers).
  - Expected time: ~2-4 minutes for 14k images (vs 7+ hours before).

What this script does:
  1. Computes Reinhard stats from a sample of training images (fast).
  2. Spawns 16 worker processes.
  3. Each worker: Reinhard stain norm -> CLAHE -> save to data/processed/.
  4. Saves updated CSVs with processed_path column.
"""

import sys
import os
import multiprocessing as mp
from pathlib import Path
from functools import partial

# Must be at top level for Windows multiprocessing (spawn method)
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Reinhard Stain Normalization (pure NumPy, parallel-safe)
# ─────────────────────────────────────────────────────────────

def compute_reinhard_stats(img_rgb: np.ndarray) -> tuple:
    """Returns (mean_L, std_L, mean_a, std_a, mean_b, std_b) in LAB space."""
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = cv2.split(img_lab)
    return (L.mean(), L.std(), a.mean(), a.std(), b.mean(), b.std())


def reinhard_normalize(img_rgb: np.ndarray, ref_stats: tuple) -> np.ndarray:
    """
    Normalizes img_rgb so its LAB statistics match the reference.
    ref_stats = (mean_L, std_L, mean_a, std_a, mean_b, std_b)
    """
    ref_mL, ref_sL, ref_ma, ref_sa, ref_mb, ref_sb = ref_stats

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = cv2.split(img_lab)

    def normalize_channel(ch, ref_mean, ref_std):
        src_std = ch.std()
        if src_std < 1e-6:          # uniform channel — skip
            return ch
        ch = (ch - ch.mean()) / src_std * ref_std + ref_mean
        return np.clip(ch, 0, 255)

    L = normalize_channel(L, ref_mL, ref_sL)
    a = normalize_channel(a, ref_ma, ref_sa)
    b = normalize_channel(b, ref_mb, ref_sb)

    img_norm = cv2.merge([L, a, b]).astype(np.uint8)
    return cv2.cvtColor(img_norm, cv2.COLOR_LAB2RGB)


def apply_clahe(img_rgb: np.ndarray, clip_limit: float, tile_size: tuple) -> np.ndarray:
    """CLAHE on the L channel of LAB — preserves colour, enhances contrast."""
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2RGB)


# ─────────────────────────────────────────────────────────────
# Worker function (must be top-level for pickling on Windows)
# ─────────────────────────────────────────────────────────────

def process_one(args):
    """
    Worker: loads one image, applies Reinhard + CLAHE, saves output.
    Returns (src_path, out_path, success).
    """
    src_path, out_path, ref_stats, clip_limit, tile_size = args
    try:
        img_bgr = cv2.imread(src_path)
        if img_bgr is None:
            return src_path, src_path, False

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Stain normalization
        img_norm = reinhard_normalize(img_rgb, ref_stats)

        # CLAHE
        img_final = apply_clahe(img_norm, clip_limit, tile_size)

        # Save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img_out = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img_out, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return src_path, out_path, True

    except Exception as e:
        return src_path, src_path, False


# ─────────────────────────────────────────────────────────────
# Reference stats computation
# ─────────────────────────────────────────────────────────────

def compute_global_ref_stats(train_csv: Path, n_sample: int = 200) -> tuple:
    """
    Computes average Reinhard stats across a random sample of training images.
    Using many images gives a more stable reference than a single image.
    """
    df = pd.read_csv(train_csv)
    sample = df.sample(n=min(n_sample, len(df)), random_state=42)

    all_stats = []
    print(f"Computing reference stats from {len(sample)} sample images...")
    for _, row in sample.iterrows():
        img = cv2.imread(row["filepath"])
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_stats.append(compute_reinhard_stats(img_rgb))

    arr = np.array(all_stats)   # shape (N, 6)
    ref = tuple(arr.mean(axis=0).tolist())
    print(f"Reference stats (mean_L, std_L, mean_a, std_a, mean_b, std_b):")
    print(f"  {[f'{v:.2f}' for v in ref]}")
    return ref


# ─────────────────────────────────────────────────────────────
# Process a full split in parallel
# ─────────────────────────────────────────────────────────────

def process_split_parallel(
    csv_path: Path,
    split_name: str,
    processed_dir: Path,
    ref_stats: tuple,
    clip_limit: float,
    tile_size: tuple,
    num_workers: int
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"\nProcessing [{split_name}]: {len(df)} images with {num_workers} workers...")

    # Build argument list for each image
    args_list = []
    for _, row in df.iterrows():
        src = row["filepath"]
        cls = row["label"]
        fname = os.path.basename(src)
        out  = str(processed_dir / split_name / cls / fname)
        args_list.append((src, out, ref_stats, clip_limit, tile_size))

    # Run in parallel
    results = []
    with mp.Pool(processes=num_workers) as pool:
        for r in tqdm(
            pool.imap_unordered(process_one, args_list, chunksize=20),
            total=len(args_list),
            desc=split_name
        ):
            results.append(r)

    # Map results back
    src_to_out = {r[0]: (r[1], r[2]) for r in results}
    processed_paths, n_ok, n_fail = [], 0, 0
    for _, row in df.iterrows():
        out_path, success = src_to_out.get(row["filepath"], (row["filepath"], False))
        processed_paths.append(out_path)
        if success:
            n_ok += 1
        else:
            n_fail += 1

    df["processed_path"] = processed_paths
    print(f"  Done: {n_ok} processed, {n_fail} failed (kept original path)")
    return df


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    # Import config here (inside main) to avoid issues with multiprocessing spawn
    sys.path.insert(0, str(Path(__file__).parent))
    from config import CFG, get_logger, ensure_dirs
    ensure_dirs()
    logger = get_logger("preprocessing")

    logger.info("=" * 55)
    logger.info("STEP 4: PREPROCESSING (Reinhard Norm + CLAHE) — FAST")
    logger.info(f"Workers: {CFG.NUM_WORKERS}  |  Method: {CFG.STAIN_NORM_METHOD}")
    logger.info("=" * 55)

    train_csv = CFG.SPLITS_DIR / "train.csv"
    val_csv   = CFG.SPLITS_DIR / "val.csv"
    test_csv  = CFG.SPLITS_DIR / "test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            logger.error(f"Missing: {p}. Run 02_data_splitting.py first.")
            return

    # Compute reference stats from training images
    ref_stats = compute_global_ref_stats(train_csv, n_sample=200)

    # Process splits in parallel
    df_train = process_split_parallel(
        train_csv, "train", CFG.PROCESSED_DIR, ref_stats,
        CFG.CLAHE_CLIP_LIMIT, CFG.CLAHE_TILE_SIZE, CFG.NUM_WORKERS
    )
    df_val = process_split_parallel(
        val_csv, "val", CFG.PROCESSED_DIR, ref_stats,
        CFG.CLAHE_CLIP_LIMIT, CFG.CLAHE_TILE_SIZE, CFG.NUM_WORKERS
    )
    df_test = process_split_parallel(
        test_csv, "test", CFG.PROCESSED_DIR, ref_stats,
        CFG.CLAHE_CLIP_LIMIT, CFG.CLAHE_TILE_SIZE, CFG.NUM_WORKERS
    )

    # Save updated CSVs
    df_train.to_csv(CFG.SPLITS_DIR / "train_processed.csv", index=False)
    df_val.to_csv(CFG.SPLITS_DIR / "val_processed.csv",     index=False)
    df_test.to_csv(CFG.SPLITS_DIR / "test_processed.csv",   index=False)

    logger.info("\n" + "=" * 55)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 55)
    logger.info(f"  Train : {len(df_train)} images -> {CFG.PROCESSED_DIR / 'train'}")
    logger.info(f"  Val   : {len(df_val)}   images -> {CFG.PROCESSED_DIR / 'val'}")
    logger.info(f"  Test  : {len(df_test)}  images -> {CFG.PROCESSED_DIR / 'test'}")
    logger.info("  Updated CSVs: train/val/test_processed.csv")
    logger.info("[OK] Step 03 complete. Next: python src\04_augmentation.py  (or: python main.py)")


if __name__ == "__main__":
    # Windows multiprocessing requires this guard
    mp.freeze_support()
    main()


