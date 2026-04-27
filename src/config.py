"""
config.py - Central configuration for the HAGCA-Net lung cancer project.
All paths, hyperparameters, and constants defined here.
Import in every script: from config import CFG
"""

from pathlib import Path


class CFG:
    # -------------------------------------------------
    # PROJECT ROOT
    # -------------------------------------------------
    PROJECT_ROOT = Path("C:/ml_project")

    # -------------------------------------------------
    # DATA PATHS
    # -------------------------------------------------
    DATA_DIR          = PROJECT_ROOT / "data"
    LC25000_TRAIN_DIR = DATA_DIR / "lc25000" / "train"
    LC25000_TEST_DIR  = DATA_DIR / "lc25000" / "test"
    PCAM_DIR          = DATA_DIR / "external_test" / "archive"
    PCAM_H5_TRAIN     = PCAM_DIR / "pcam" / "training_split.h5"
    PCAM_H5_TEST      = PCAM_DIR / "pcam" / "test_split.h5"
    PCAM_LABEL_TRAIN  = PCAM_DIR / "Labels" / "Labels" / "camelyonpatch_level_2_split_train_y.h5"
    PCAM_LABEL_TEST   = PCAM_DIR / "Labels" / "Labels" / "camelyonpatch_level_2_split_test_y.h5"
    SPLITS_DIR        = DATA_DIR / "splits"
    PROCESSED_DIR     = DATA_DIR / "processed"

    # -------------------------------------------------
    # OUTPUT PATHS
    # -------------------------------------------------
    CHECKPOINTS_DIR   = PROJECT_ROOT / "checkpoints"
    RESULTS_DIR       = PROJECT_ROOT / "results"
    PLOTS_DIR         = RESULTS_DIR / "plots"
    METRICS_DIR       = RESULTS_DIR / "metrics"
    GRADCAM_DIR       = RESULTS_DIR / "gradcam"
    ABLATION_DIR      = RESULTS_DIR / "ablation"
    LOGS_DIR          = PROJECT_ROOT / "logs"

    # -------------------------------------------------
    # DATASET CLASSES (LC25000 - lung only)
    # -------------------------------------------------
    LUNG_CLASSES = ["lung_aca", "lung_n", "lung_scc"]
    ALL_CLASSES  = ["lung_aca", "lung_n", "lung_scc", "colon_aca", "colon_n"]
    CLASS_TO_IDX = {cls: i for i, cls in enumerate(LUNG_CLASSES)}
    IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}
    NUM_CLASSES  = len(LUNG_CLASSES)   # 3

    # -------------------------------------------------
    # DATA SPLITTING
    # -------------------------------------------------
    NUM_GROUPS    = 50
    TRAIN_RATIO   = 0.70
    VAL_RATIO     = 0.15
    TEST_RATIO    = 0.15
    KFOLD_SPLITS  = 1
    RANDOM_SEED   = 42

    # -------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------
    IMAGE_SIZE        = 224
    MEAN              = [0.485, 0.456, 0.406]
    STD               = [0.229, 0.224, 0.225]
    CLAHE_CLIP_LIMIT  = 2.0
    CLAHE_TILE_SIZE   = (8, 8)
    STAIN_NORM_METHOD = "reinhard"   # fast parallel-safe normalization
    NUM_WORKERS       = 16           # multiprocessing workers for preprocessing

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------
    BATCH_SIZE          = 16         # 64 OOMs with Swin-Base; 16 + grad-checkpoint fits 8.5GB
    NUM_EPOCHS          = 30
    LEARNING_RATE       = 1e-4
    WEIGHT_DECAY        = 1e-4
    LR_PATIENCE         = 5
    LR_FACTOR           = 0.5
    EARLY_STOP_PATIENCE = 7
    DATALOADER_WORKERS  = 0          # 0 = main-process loading (Windows spawn is too slow for >0)
    DROPOUT_RATE        = 0.3
    PIN_MEMORY          = True       # faster GPU transfer
    AMP                 = True       # Automatic Mixed Precision (FP16) for RTX cards

    # -------------------------------------------------
    # MODEL ARCHITECTURE
    # -------------------------------------------------
    CNN_BACKBONE         = "efficientnet_b3"
    TRANSFORMER_BACKBONE = "swin_base_patch4_window7_224"
    CNN_OUT_DIM          = 512
    TRANS_OUT_DIM        = 512
    FUSION_DIM           = 512
    GNN_HIDDEN_DIM       = 256
    ATTENTION_DIM        = 256
    PRETRAINED           = True

    # -------------------------------------------------
    # GAN AUGMENTATION
    # -------------------------------------------------
    GAN_LATENT_DIM   = 100
    GAN_EPOCHS       = 100
    GAN_LR           = 2e-4
    GAN_BATCH_SIZE   = 64
    GAN_TARGET_CLASS = "lung_scc"

    # -------------------------------------------------
    # DEVICE (set at runtime by setup_device())
    # -------------------------------------------------
    DEVICE          = "cpu"
    LOG_FILE        = LOGS_DIR / "training.log"
    BEST_MODEL_PATH = CHECKPOINTS_DIR / "hagcanet_best.pth"
    LAST_MODEL_PATH = CHECKPOINTS_DIR / "hagcanet_last.pth"


def setup_device():
    """
    Detects GPU and sets CFG.DEVICE.
    Also enables cuDNN benchmark mode for faster convolutions.
    Call this at the top of every training/eval script.
    """
    import torch
    if torch.cuda.is_available():
        CFG.DEVICE = "cuda"
        torch.backends.cudnn.benchmark = True   # auto-tune kernels for fixed input sizes
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[CFG] GPU detected : {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"[CFG] AMP enabled  : {CFG.AMP}")
        print(f"[CFG] Batch size   : {CFG.BATCH_SIZE}")
    else:
        CFG.DEVICE = "cpu"
        CFG.AMP    = False   # AMP only works on CUDA
        print("[CFG] No GPU found — running on CPU.")
    print(f"[CFG] Device       : {CFG.DEVICE}")
    return CFG.DEVICE


def ensure_dirs():
    """Creates all output directories if they do not exist."""
    dirs = [
        CFG.SPLITS_DIR, CFG.PROCESSED_DIR,
        CFG.CHECKPOINTS_DIR, CFG.PLOTS_DIR,
        CFG.METRICS_DIR, CFG.GRADCAM_DIR,
        CFG.ABLATION_DIR, CFG.LOGS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name="hagcanet"):
    """Returns a logger writing to both console and log file."""
    import logging
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.stream.reconfigure(errors="replace") if hasattr(ch.stream, "reconfigure") else None
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(str(CFG.LOG_FILE), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


if __name__ == "__main__":
    ensure_dirs()
    logger = get_logger()
    setup_device()
    logger.info(f"Lung classes  : {CFG.LUNG_CLASSES}")
    logger.info(f"Image size    : {CFG.IMAGE_SIZE}")
    logger.info(f"Batch size    : {CFG.BATCH_SIZE}")
    logger.info(f"AMP           : {CFG.AMP}")
    logger.info(f"CNN backbone  : {CFG.CNN_BACKBONE}")
    logger.info(f"Trans backbone: {CFG.TRANSFORMER_BACKBONE}")


