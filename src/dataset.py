import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from src.preprocessing import apply_clahe, stain_normalization
from albumentations import Compose, HorizontalFlip, Rotate, RandomScale, Resize

CLASS_MAP = {
    "lung_aca": 0,
    "lung_n": 1,
    "lung_scc": 2,
    "colon_aca": 3,
    "colon_n": 4
}
def get_transforms():
    return Compose([
        Resize(224, 224),   # 🔴 THIS IS REQUIRED
        HorizontalFlip(p=0.5),
        Rotate(limit=25, p=0.5),
        # RandomScale(scale_limit=0.1, p=0.5)
    ])
    
def get_val_transforms():
    return Compose([
        Resize(224, 224)
    ])

def load_lc25000(root_dir):
    train_paths, train_labels, groups = [], [], []
    test_paths, test_labels = [], []

    # -------- TRAIN FOLDER --------
    train_dir = os.path.join(root_dir, "train")

    for class_name in os.listdir(train_dir):
        if class_name not in CLASS_MAP:
            continue

        class_path = os.path.join(train_dir, class_name)
        label = CLASS_MAP[class_name]

        images = glob.glob(os.path.join(class_path, "*.jpeg"))

        for img_path in images:
            train_paths.append(img_path)
            train_labels.append(label)

            group_id = os.path.basename(img_path).split("_")[0]
            groups.append(group_id)

    # -------- TEST FOLDER --------
    test_dir = os.path.join(root_dir, "test")

    for class_name in os.listdir(test_dir):
        if class_name not in CLASS_MAP:
            continue

        class_path = os.path.join(test_dir, class_name)
        label = CLASS_MAP[class_name]

        images = glob.glob(os.path.join(class_path, "*.jpeg"))

        for img_path in images:
            test_paths.append(img_path)
            test_labels.append(label)

    return train_paths, train_labels, groups, test_paths, test_labels


class LungDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])

        if img is None:
            raise ValueError(f"Corrupted image: {self.image_paths[idx]}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = stain_normalization(img)
        img = apply_clahe(img)

        if self.transform:
            img = self.transform(image=img)['image']

        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(self.labels[idx]).long()

        return img, label


def get_group_split(image_paths, labels, groups):
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(gss.split(image_paths, labels, groups))

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(
        np.array(image_paths)[temp_idx],
        np.array(labels)[temp_idx],
        np.array(groups)[temp_idx]
    ))

    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    return train_idx, val_idx, test_idx

