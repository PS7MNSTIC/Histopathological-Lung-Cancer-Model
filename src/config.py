import torch
import os

class Config:
    # Paths
    BASE_DIR = os.getcwd()
    TRAIN_DIR = os.path.join(BASE_DIR, "data/lc25000")
    EXT_TEST_DIR = os.path.join(BASE_DIR, "data/external_test")

    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    # Data
    IMG_SIZE = 224
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    NUM_CLASSES = 5

    # Training
    EPOCHS = 30
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Early stopping
    PATIENCE = 5

    # Mixed precision
    USE_AMP = torch.cuda.is_available()