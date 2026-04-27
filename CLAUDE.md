# Histopathological Lung Cancer ML Project - Session Log

> This file is maintained by Claude across sessions. Read at the start of every new session.

---

## Environment & Paths

| Item | Value |
|------|-------|
| Conda env | lung_cancer |
| Python | C:\Users\bmsah\anaconda3\envs\lung_cancer\python.exe |
| Project root | C:\ml_project\ |
| Source code | C:\ml_project\src\ |
| Data | C:\ml_project\data\ |
| Checkpoints | C:\ml_project\checkpoints\ |
| Results | C:\ml_project\results\ |
| Logs | C:\ml_project\logs\ |

**Run any script from C:\ml_project:**
  conda activate lung_cancer
  cd C:\ml_project
  python src\<script_name>.py

---

## Dataset Facts (verified)

| Dataset | Details |
|---------|---------|
| LC25000 train | 3 lung classes x ~4500 images |
| LC25000 test | 3 lung classes x ~500 images |
| PatchCamelyon | HDF5 format in data\external_test\archive\pcam\ |

Lung classes: lung_aca, lung_n, lung_scc (colon excluded)

---

## Key Packages in lung_cancer env

torch 2.11.0+cu128, torchvision, timm 1.0.26, albumentations, opencv,
grad-cam 1.5.5, staintools + spams-bin, h5py, scikit-learn, pandas, numpy,
matplotlib, seaborn

NOTE: torch.cuda.is_available() = False. Training runs on CPU.
torch-geometric NOT installed. Graph module uses pure PyTorch.

---

## Pipeline Status

| Step | File | Status | Notes |
|------|------|--------|-------|
| 1 | Dataset Collection | DONE | Data already downloaded |
| 2 | 01_data_cleaning.py | DONE + TESTED | 805 dups removed, 0 corrupted |
| 3 | 02_data_splitting.py | DONE + TESTED | 9936/1987/2272 train/val/test |
| 4 | 03_preprocessing.py | WRITTEN | Stain norm + CLAHE |
| 5 | 04_augmentation.py | PENDING | |
| 5b | 05_gan_augment.py | PENDING | |
| 6 | 06_model_hagcanet.py | PENDING | |
| 7 | 07_train.py | PENDING | |
| 8 | 08_evaluate.py | PENDING | |
| 9 | 09_gradcam.py | PENDING | |
| 10 | 10_cross_dataset.py | PENDING | |
| 11 | 11_ablation.py | PENDING | |

---

## Cleaning Results (Step 2 - verified run)

- Total scanned: 15,000 (lung classes only)
- Valid: 14,195
- Corrupted: 0
- Duplicates removed: 805 (LC25000 known issue)
- Per class: lung_aca=4727, lung_n=4744, lung_scc=4724

## Split Results (Step 3 - verified run)

- Train: 9,936 (70%) — lung_aca:3308, lung_n:3321, lung_scc:3307
- Val:   1,987 (14%) — lung_aca:662,  lung_n:664,  lung_scc:661
- Test:  2,272 (16%) — lung_aca:757,  lung_n:759,  lung_scc:756
- Group leakage: NONE detected
- Files: data\splits\train.csv, val.csv, test.csv, full_split.csv, manifest.csv

---

## Preprocessing Design (Step 4)

- Macenko stain normalization via staintools (reference = first train image)
- CLAHE applied in LAB color space (L-channel only)
- Outputs to: data\processed\<split>\<class>\<filename>
- Fallback: if stain norm fails on an image, CLAHE-only is applied
- Produces: train_processed.csv, val_processed.csv, test_processed.csv

---

## Project Folder Structure

`
C:\ml_project\
    src\
        config.py               DONE
        01_data_cleaning.py     DONE + TESTED
        02_data_splitting.py    DONE + TESTED
        03_preprocessing.py     WRITTEN (run next)
    data\
        lc25000\train\ test\    raw data (DO NOT MODIFY)
        external_test\          PatchCamelyon HDF5
        splits\                 manifest + train/val/test CSVs
        processed\              preprocessed images go here
    checkpoints\
    results\plots\ metrics\ gradcam\ ablation\
    logs\
    run_pipeline.bat
    run_cleaning.bat
`

---

## Session Log

### Session 1 - 2026-04-25

**Work Done:**
- Read and analyzed NewWork(Group Anuska).docx
- Explored data: LC25000 + PatchCamelyon already downloaded and organized
- Checked conda env lung_cancer: all major packages confirmed
- Installed spams-bin (needed by staintools for stain normalization)
- Created full project folder structure in C:\ml_project\
- Wrote config.py, 01_data_cleaning.py, 02_data_splitting.py, 03_preprocessing.py
- Created run_pipeline.bat and run_cleaning.bat
- User ran Steps 2 and 3 successfully - outputs verified

**Key Findings:**
- LC25000 has 805 duplicate images (known dataset issue) - all removed
- No corrupted images
- Split is clean with zero group leakage
- CUDA not available despite +cu128 build (GPU driver mismatch likely)

### What To Do Next Session

1. User runs: python src\03_preprocessing.py  (takes 15-30 min for 14k images)
2. Write 04_augmentation.py (Dataset class + torchvision transforms)
3. Write 05_gan_augment.py (DCGAN for synthetic lung_scc samples)
4. Write 06_model_hagcanet.py (the main HAGCA-Net model)
